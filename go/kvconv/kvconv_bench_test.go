// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for kvconv.go — the root↔metal KV snapshot conversions on
// the restore path. Moved from the root kv_snapshot_restore_bench_test.go
// in the orphan sweep: the conversions live here, so their benches do too.

package kvconv

// Restore-path doubling benchmarks (AX-11).
//
// ToMetalKVSnapshot is the pure-Go conversion that WarmPromptCacheFromKV
// runs before handing the snapshot to the Metal restorer. It is the State
// continuity multi-turn restore path. Two source encodings reach it:
//
//   - Native-bytes (KeyBytes/ValueBytes set, EncodingNative): the K/V
//     tensors pass through to metal.KVSnapshot BY REFERENCE — the metal
//     restorer then pins them zero-copy via fromPinnedRawBytes. No copy of
//     the cache bytes. This is the wired zero-copy path.
//
//   - Heads-float32 (head.Key/head.Value set): ToMetalKVSnapshot copies
//     every head's float32 K/V into a fresh slab (copy #1), and the metal
//     restorer copies AGAIN into an MLX array via FromValues (copy #2).
//     That second hold is the "doubling" — the whole cache materialised
//     twice during a single restore.
//
// These benches measure copy #1 (the pure-Go materialisation) directly so
// the doubling shows as ~full-cache-bytes B/op on the heads path and
// near-zero on the native path. No Metal device required.

import (
	"testing"

	"dappco.re/go/mlx/kv"
	"dappco.re/go/mlx/pkg/metal"
)

const (
	// Gemma-4-class warm-restore prefix: 26 cache layers, 4 KV heads,
	// 256 head-dim. tokensPerHead tensors are seqLen*headDim float32.
	benchRestoreLayers  = 26
	benchRestoreHeads   = 4
	benchRestoreSeqLen  = 2048
	benchRestoreHeadDim = 256
	benchRestorePerHead = benchRestoreSeqLen * benchRestoreHeadDim
	benchRestoreFloats  = benchRestoreLayers * benchRestoreHeads * benchRestorePerHead * 2 // K+V
	benchRestoreCacheB  = benchRestoreFloats * 4                                           // float32 cache bytes
)

// newHeadsRestoreSnapshot builds a heads-float32 encoded snapshot — the
// path ToMetalKVSnapshot materialises into a fresh slab.
func newHeadsRestoreSnapshot() *kv.Snapshot {
	tokens := make([]int32, benchRestoreSeqLen)
	for i := range tokens {
		tokens[i] = int32(i + 1)
	}
	layers := make([]kv.LayerSnapshot, benchRestoreLayers)
	for l := range layers {
		heads := make([]kv.HeadSnapshot, benchRestoreHeads)
		for h := range heads {
			key := make([]float32, benchRestorePerHead)
			value := make([]float32, benchRestorePerHead)
			for i := range key {
				key[i] = float32(l*benchRestoreHeads + h + i)
				value[i] = float32(l*benchRestoreHeads + h - i)
			}
			heads[h] = kv.HeadSnapshot{
				Key:        key,
				KeyDType:   "float32",
				Value:      value,
				ValueDType: "float32",
			}
		}
		layers[l] = kv.LayerSnapshot{
			Layer:      l,
			CacheIndex: l,
			Heads:      heads,
		}
	}
	return &kv.Snapshot{
		Version:      kv.SnapshotVersion,
		Architecture: "bench",
		Tokens:       tokens,
		TokenOffset:  benchRestoreSeqLen,
		NumLayers:    benchRestoreLayers,
		NumHeads:     benchRestoreHeads,
		SeqLen:       benchRestoreSeqLen,
		HeadDim:      benchRestoreHeadDim,
		Layers:       layers,
	}
}

// newNativeRestoreSnapshot builds a native-bytes encoded snapshot — the
// wired zero-copy path that ToMetalKVSnapshot passes through by reference.
func newNativeRestoreSnapshot() *kv.Snapshot {
	tokens := make([]int32, benchRestoreSeqLen)
	for i := range tokens {
		tokens[i] = int32(i + 1)
	}
	const layerFloats = benchRestoreHeads * benchRestorePerHead
	layers := make([]kv.LayerSnapshot, benchRestoreLayers)
	for l := range layers {
		keyBytes := make([]byte, layerFloats*4)
		valueBytes := make([]byte, layerFloats*4)
		layers[l] = kv.LayerSnapshot{
			Layer:      l,
			CacheIndex: l,
			KeyDType:   "float32",
			KeyBytes:   keyBytes,
			KeyShape:   []int32{1, benchRestoreHeads, benchRestoreSeqLen, benchRestoreHeadDim},
			ValueDType: "float32",
			ValueBytes: valueBytes,
			ValueShape: []int32{1, benchRestoreHeads, benchRestoreSeqLen, benchRestoreHeadDim},
		}
	}
	return &kv.Snapshot{
		Version:      kv.SnapshotVersion,
		Architecture: "bench",
		Tokens:       tokens,
		TokenOffset:  benchRestoreSeqLen,
		NumLayers:    benchRestoreLayers,
		NumHeads:     benchRestoreHeads,
		SeqLen:       benchRestoreSeqLen,
		HeadDim:      benchRestoreHeadDim,
		Layers:       layers,
	}
}

// newDualRestoreSnapshot builds the realistic v4-decode shape: layer-level
// native KeyBytes/ValueBytes AND decoded per-head float32 Key/Value both
// populated. This is what a default-options snapshot load produces and what
// WakeAgentMemory's snapshot-restore fallback feeds ToMetalKVSnapshot. The
// restorer pins the layer bytes zero-copy and ignores the heads — so the
// per-head float32 copy is pure doubling. This is the bench the fix targets.
func newDualRestoreSnapshot() *kv.Snapshot {
	s := newNativeRestoreSnapshot()
	for l := range s.Layers {
		heads := make([]kv.HeadSnapshot, benchRestoreHeads)
		for h := range heads {
			key := make([]float32, benchRestorePerHead)
			value := make([]float32, benchRestorePerHead)
			for i := range key {
				key[i] = float32(l*benchRestoreHeads + h + i)
				value[i] = float32(l*benchRestoreHeads + h - i)
			}
			heads[h] = kv.HeadSnapshot{
				Key:        key,
				KeyDType:   "float32",
				Value:      value,
				ValueDType: "float32",
			}
		}
		s.Layers[l].Heads = heads
	}
	return s
}

// newMetalHeadsCaptureSnapshot builds a heads-float32 metal snapshot — the
// capture-direction mirror of newHeadsRestoreSnapshot. ToRootKVSnapshot
// clones every head's Key/Value (and KeyBytes/ValueBytes, left empty here)
// into the root arenas.
func newMetalHeadsCaptureSnapshot() *metal.KVSnapshot {
	tokens := make([]int32, benchRestoreSeqLen)
	for i := range tokens {
		tokens[i] = int32(i + 1)
	}
	layers := make([]metal.KVLayerSnapshot, benchRestoreLayers)
	for l := range layers {
		heads := make([]metal.KVHeadSnapshot, benchRestoreHeads)
		for h := range heads {
			key := make([]float32, benchRestorePerHead)
			value := make([]float32, benchRestorePerHead)
			for i := range key {
				key[i] = float32(l*benchRestoreHeads + h + i)
				value[i] = float32(l*benchRestoreHeads + h - i)
			}
			heads[h] = metal.KVHeadSnapshot{
				Key:        key,
				KeyDType:   metal.DTypeFloat32,
				Value:      value,
				ValueDType: metal.DTypeFloat32,
			}
		}
		layers[l] = metal.KVLayerSnapshot{
			Layer:      l,
			CacheIndex: l,
			Heads:      heads,
		}
	}
	return &metal.KVSnapshot{
		Version:      kv.SnapshotVersion,
		Architecture: "bench",
		Tokens:       tokens,
		TokenOffset:  benchRestoreSeqLen,
		NumLayers:    benchRestoreLayers,
		NumHeads:     benchRestoreHeads,
		SeqLen:       benchRestoreSeqLen,
		HeadDim:      benchRestoreHeadDim,
		Layers:       layers,
	}
}

// newMetalTurboQuantCaptureSnapshot builds a metal snapshot whose layers
// carry a valid TurboQuant reference payload — the rootTurboQuantPayloads
// marshal path. Heads are left empty so the bench isolates the payload
// marshal/clone budget rather than the tensor copy.
func newMetalTurboQuantCaptureSnapshot() *metal.KVSnapshot {
	layers := make([]metal.KVLayerSnapshot, benchRestoreLayers)
	for l := range layers {
		layers[l] = metal.KVLayerSnapshot{
			Layer:              l,
			CacheIndex:         l,
			CacheMode:          metal.KVCacheModeTurboQuant,
			TurboQuantPayloads: []metal.TurboQuantKVReferencePagePayload{validTurboQuantPayload()},
		}
	}
	return &metal.KVSnapshot{
		Version:      kv.SnapshotVersion,
		Architecture: "bench",
		NumLayers:    benchRestoreLayers,
		Layers:       layers,
	}
}

var benchMetalSnapshotSink int

// BenchmarkToMetalKVSnapshot_DualNativePlusHeads measures the production v4
// shape. Before the fix ToMetalKVSnapshot copied the dead per-head float32
// into a fresh slab (~full-cache B/op) on top of the zero-copy layer-byte
// passthrough — the doubling. After the fix the heads pass through by
// reference and B/op collapses to the native-passthrough baseline.
func BenchmarkToMetalKVSnapshot_DualNativePlusHeads(b *testing.B) {
	snapshot := newDualRestoreSnapshot()
	b.ReportAllocs()
	b.SetBytes(int64(benchRestoreCacheB))
	for b.Loop() {
		out := ToMetalKVSnapshot(snapshot)
		benchMetalSnapshotSink = len(out.Layers)
	}
}

// BenchmarkToMetalKVSnapshot_HeadsFloat32 measures copy #1 on the heads
// path — ToMetalKVSnapshot materialising the full cache into a fresh slab.
// B/op should track benchRestoreCacheB (~107 MiB for the Gemma-4 fixture).
func BenchmarkToMetalKVSnapshot_HeadsFloat32(b *testing.B) {
	snapshot := newHeadsRestoreSnapshot()
	b.ReportAllocs()
	b.SetBytes(int64(benchRestoreCacheB))
	for b.Loop() {
		out := ToMetalKVSnapshot(snapshot)
		benchMetalSnapshotSink = len(out.Layers)
	}
}

// BenchmarkToMetalKVSnapshot_NativeBytes measures the wired zero-copy path
// — KeyBytes/ValueBytes pass through by reference, so B/op should be the
// small per-layer struct overhead only (no cache-byte copy).
func BenchmarkToMetalKVSnapshot_NativeBytes(b *testing.B) {
	snapshot := newNativeRestoreSnapshot()
	b.ReportAllocs()
	b.SetBytes(int64(benchRestoreCacheB))
	for b.Loop() {
		out := ToMetalKVSnapshot(snapshot)
		benchMetalSnapshotSink = len(out.Layers)
	}
}

var benchRootSnapshotSink int

// BenchmarkToRootKVSnapshot_HeadsFloat32 measures the capture direction
// (metal -> root) on the heads-float32 path — the heavier conversion that
// clones FOUR head families (Key + Value float32 into one slab, KeyBytes +
// ValueBytes into a byte slab) against ToMetalKVSnapshot's two. The restore
// direction is benched above; this closes the capture-side gap so both
// directions of kvconv carry a bench, and provides the instrument for the
// per-snapshot arena alloc count on the capture path.
func BenchmarkToRootKVSnapshot_HeadsFloat32(b *testing.B) {
	snapshot := newMetalHeadsCaptureSnapshot()
	b.ReportAllocs()
	b.SetBytes(int64(benchRestoreCacheB))
	for b.Loop() {
		out := ToRootKVSnapshot(snapshot)
		benchRootSnapshotSink = len(out.Layers)
	}
}

// BenchmarkRootTurboQuantPayloads measures the metal -> root TurboQuant
// reference-payload marshal path (rootTurboQuantPayloads via
// ToRootKVSnapshot). Each payload is JSON-marshalled once; the marshal
// already returns a freshly-owned buffer, so this is the instrument for the
// redundant-clone removal on that path. A small fixture keeps the JSON cost
// dominated by the per-payload allocation budget, not by huge tensor copies.
func BenchmarkRootTurboQuantPayloads(b *testing.B) {
	snapshot := newMetalTurboQuantCaptureSnapshot()
	b.ReportAllocs()
	for b.Loop() {
		out := ToRootKVSnapshot(snapshot)
		benchRootSnapshotSink = len(out.Layers)
	}
}
