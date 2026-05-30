// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"encoding/binary"
	"math"
	"testing"

	"dappco.re/go/mlx/kv"
)

// f32Bytes encodes float32 values as little-endian bytes — the on-disk K/V
// slab layout that fromPinnedRawBytes pins zero-copy.
func f32Bytes(values []float32) []byte {
	out := make([]byte, len(values)*4)
	for i, v := range values {
		binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(v))
	}
	return out
}

// TestToMetalKVSnapshot_DualNativePlusHeads_Good asserts the zero-copy
// passthrough fix preserves a byte-identical restore surface. For a v4 dual-
// populated snapshot (native layer KeyBytes/ValueBytes + decoded per-head
// float32) the metal snapshot must carry:
//   - layer KeyBytes/ValueBytes by reference (the restorer pins these), and
//   - the same per-head float32 values (now passed by reference, not copied).
//
// The restored cache is identical because the restorer reads only the layer
// bytes, and those are unchanged by the fix.
func TestToMetalKVSnapshot_DualNativePlusHeads_Good(t *testing.T) {
	src := &kv.Snapshot{
		Version:      kv.SnapshotVersion,
		Architecture: "gemma4_text",
		Tokens:       []int32{1, 2},
		NumLayers:    1,
		NumHeads:     1,
		SeqLen:       2,
		HeadDim:      2,
		Layers: []kv.LayerSnapshot{{
			Layer:      0,
			CacheIndex: 0,
			KeyDType:   "float32",
			KeyBytes:   f32Bytes([]float32{1, 2, 3, 4}),
			KeyShape:   []int32{1, 1, 2, 2},
			ValueDType: "float32",
			ValueBytes: f32Bytes([]float32{5, 6, 7, 8}),
			ValueShape: []int32{1, 1, 2, 2},
			Heads: []kv.HeadSnapshot{{
				Key:        []float32{1, 2, 3, 4},
				KeyDType:   "float32",
				Value:      []float32{5, 6, 7, 8},
				ValueDType: "float32",
			}},
		}},
	}

	out := toMetalKVSnapshot(src)
	if len(out.Layers) != 1 || len(out.Layers[0].Heads) != 1 {
		t.Fatalf("toMetalKVSnapshot() shape = %d layers / %d heads", len(out.Layers), len(out.Layers[0].Heads))
	}
	layer := out.Layers[0]

	// Layer native bytes must be byte-identical (passed by reference). This
	// is what the restorer pins zero-copy, so byte-equality here is the
	// State-continuity correctness assertion.
	if !bytesEqual(layer.KeyBytes, src.Layers[0].KeyBytes) {
		t.Fatalf("layer KeyBytes diverged: %v vs %v", layer.KeyBytes, src.Layers[0].KeyBytes)
	}
	if !bytesEqual(layer.ValueBytes, src.Layers[0].ValueBytes) {
		t.Fatalf("layer ValueBytes diverged: %v vs %v", layer.ValueBytes, src.Layers[0].ValueBytes)
	}

	// Per-head float32 must carry the same values (now by reference).
	head := layer.Heads[0]
	if !float32sEqual(head.Key, src.Layers[0].Heads[0].Key) {
		t.Fatalf("head Key diverged: %v vs %v", head.Key, src.Layers[0].Heads[0].Key)
	}
	if !float32sEqual(head.Value, src.Layers[0].Heads[0].Value) {
		t.Fatalf("head Value diverged: %v vs %v", head.Value, src.Layers[0].Heads[0].Value)
	}
	// Head dtype derives from head.KeyBytes (absent on a decoded-heads
	// layer), so it resolves to the zero DType — unchanged by the fix and
	// irrelevant for native layers, where the restorer reads layer bytes.
	if head.KeyDType != 0 || head.ValueDType != 0 {
		t.Fatalf("head dtype = %v/%v, want zero (no head bytes)", head.KeyDType, head.ValueDType)
	}

	// The head Key must alias the source (passed by reference, not copied)
	// — confirming the doubling is gone. Mutating the metal-side slice is
	// observable in the source; this aliasing is SAFE because the restorer
	// never reads heads on a native layer, and the source outlives the call.
	head.Key[0] = 42
	if src.Layers[0].Heads[0].Key[0] != 42 {
		t.Fatal("native-layer head Key was copied, not passed by reference — doubling not eliminated")
	}
}

// TestToMetalKVSnapshot_HeadsOnly_Good asserts the heads-only path (no layer
// native bytes — e.g. a v3 snapshot) still deep-copies per-head float32 into
// an independent slab, so a later mutation of the source does NOT corrupt the
// metal snapshot. This is the load-bearing defensive copy on the only path
// where heads ARE the cache data; the fix must leave it intact.
func TestToMetalKVSnapshot_HeadsOnly_Good(t *testing.T) {
	src := &kv.Snapshot{
		Version:      kv.SnapshotVersion,
		Architecture: "qwen3",
		Tokens:       []int32{1, 2},
		NumLayers:    1,
		NumHeads:     1,
		SeqLen:       2,
		HeadDim:      2,
		Layers: []kv.LayerSnapshot{{
			Layer:      0,
			CacheIndex: 0,
			Heads: []kv.HeadSnapshot{{
				Key:        []float32{1, 2, 3, 4},
				KeyDType:   "float32",
				Value:      []float32{5, 6, 7, 8},
				ValueDType: "float32",
			}},
		}},
	}

	out := toMetalKVSnapshot(src)
	head := out.Layers[0].Heads[0]
	if !float32sEqual(head.Key, []float32{1, 2, 3, 4}) {
		t.Fatalf("head Key = %v, want [1 2 3 4]", head.Key)
	}

	// Mutate the source; the heads-only path must have copied, so the metal
	// snapshot is unaffected.
	src.Layers[0].Heads[0].Key[0] = 99
	if head.Key[0] != 1 {
		t.Fatal("heads-only path aliased source key data — defensive copy lost")
	}
}

func bytesEqual(a, b []byte) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func float32sEqual(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
