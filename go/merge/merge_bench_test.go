// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the model-merge pure-math + plan-construction core.
// Per AX-11 — Packs is a slow IO-bound action overall, but the inner
// kernels (linearMerge, slerpMerge, writeFloat32Values, mergeTensorValues,
// normalizedWeights, buildMergedHeader, sameUint64Slice, clampFloat64)
// run per-tensor per-chunk and are the surface a budget pass can
// actually optimise. The chunked write paths (writeLinearChunks,
// writeSLERPChunks) are exercised at small sizes to surface the
// per-chunk overhead without making the bench IO-dominated.
//
// Run:    go test -bench='BenchmarkMerge|BenchmarkLinearMerge|BenchmarkSLERPMerge|BenchmarkMergeTensorValues|BenchmarkNormalizedWeights|BenchmarkBuildMergedHeader|BenchmarkSameUint64Slice|BenchmarkClampFloat64|BenchmarkWriteFloat32Values|BenchmarkWriteLinearChunks|BenchmarkWriteSLERPChunks' -benchmem -run='^$' ./go/merge

package merge

import (
	"context"
	"testing"

	core "dappco.re/go"
	mp "dappco.re/go/inference/modelpack"
	"dappco.re/go/inference/safetensors"
)

// Sinks defeat compiler DCE.
var (
	benchMergeF32    []float32
	benchMergeF64    []float64
	benchMergeErr    error
	benchMergeBool   bool
	benchMergeFloat  float64
	benchMergeBytes  []byte
	benchMergeHeader map[string]safetensors.HeaderEntry
	benchMergeResult *Result
)

// benchTensorValues builds two synthetic source slices the merge math
// can chew on. Same shape, same length — every value finite.
func benchTensorValues(n int) [][]float32 {
	left := make([]float32, n)
	right := make([]float32, n)
	for i := range left {
		left[i] = float32(i%256) * 0.0125
		right[i] = float32((i+1)%256) * 0.0125
	}
	return [][]float32{left, right}
}

// --- linearMerge — per-tensor inner loop ---

func benchSafetensorsIndex(tensorCount int) safetensors.Index {
	names := make([]string, 0, tensorCount)
	tensors := make(map[string]safetensors.TensorRef, tensorCount)
	var offset int64
	for i := range tensorCount {
		name := "blk." + core.Itoa(i/4) + ".w." + core.Itoa(i%4)
		shape := []uint64{4096, 4096}
		elements := 4096 * 4096
		byteLen := int64(elements * 4)
		tensors[name] = safetensors.TensorRef{
			Name:      name,
			DType:     "F32",
			Shape:     shape,
			Elements:  elements,
			DataStart: offset,
			ByteLen:   byteLen,
		}
		offset += byteLen
		names = append(names, name)
	}
	return safetensors.Index{Names: names, Tensors: tensors}
}

func BenchmarkSameUint64Slice_Match4D(b *testing.B) {
	a := []uint64{4, 28, 2048, 64}
	c := []uint64{4, 28, 2048, 64}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchMergeBool = sameUint64Slice(a, c)
	}
}

func BenchmarkSameUint64Slice_DifferentLength(b *testing.B) {
	a := []uint64{4, 28, 2048, 64}
	c := []uint64{4, 28, 2048}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchMergeBool = sameUint64Slice(a, c)
	}
}

func BenchmarkSameUint64Slice_LastDimMismatch(b *testing.B) {
	a := []uint64{4, 28, 2048, 64}
	c := []uint64{4, 28, 2048, 128}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchMergeBool = sameUint64Slice(a, c)
	}
}

// --- clampFloat64 — fires per SLERP angle clamp ---

func BenchmarkClampFloat64_InRange(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchMergeFloat = clampFloat64(0.5, -1, 1)
	}
}

func BenchmarkClampFloat64_ClampHigh(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchMergeFloat = clampFloat64(2.0, -1, 1)
	}
}

// --- writeFloat32Values — fires per chunk write; the encode loop ---

type discardFile struct {
	written int
}

// noopWriter is a minimal *core.OSFile substitute path: the merge
// writers expect *core.OSFile so we run a small temp file to keep the
// signature satisfied without touching disk for huge slices.

// benchWriteSafetensorsF32 lays down a small safetensors file in temp so
// the chunk readers have something to seek over — a single-tensor adapter
// onto the shared-codec-backed writeTestSafetensorsF32 (helpers_test.go).
func benchWriteSafetensorsF32(b *testing.B, path string, name string, shape []int, values []float32) {
	b.Helper()
	writeTestSafetensorsF32(b, path, []safetensorTestTensor{{Name: name, Shape: shape, Data: values}})
}

func BenchmarkValidateTensorIndexes_AllMatch(b *testing.B) {
	left := benchSafetensorsIndex(200)
	right := benchSafetensorsIndex(200)
	indexes := []safetensors.Index{left, right}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchMergeErr = validateTensorIndexes(indexes, false)
	}
}

// --- Packs end-to-end — the per-tensor write loop in
// writeMergedSafetensors. A real pack is many tensors in a few shard
// files; the loop opens the source shards once per tensor via
// OpenReaders, so the file-open allocs scale with tensor count even
// though only a handful of distinct shards exist. This bench writes a
// multi-tensor single-shard pack per source so that per-tensor reopen
// cost is the visible alloc surface (mirrors the fileCache win already
// in ComparePacks). ---

// benchWritePackTensors lays down one safetensors shard holding many
// small tensors and returns a pack pointed at it. Mirrors
// benchCompareScratchPack but lives on the merge side and keeps the
// per-tensor element count small so the bench stays write-loop-bound,
// not IO-bound.
func benchWritePackTensors(b *testing.B, names []string, perTensorElements int) mp.ModelPack {
	b.Helper()
	dir := b.TempDir()
	cfg := `{"model_type":"qwen3","vocab_size":151936,"hidden_size":2048,"num_hidden_layers":28,"max_position_embeddings":40960}`
	if result := core.WriteFile(core.PathJoin(dir, "config.json"), []byte(cfg), 0o644); !result.OK {
		b.Fatalf("write config: %v", result.Value)
	}
	// Real tokenizer.json files for qwen3-class checkpoints are multiple MB
	// (the BPE merge table dominates). The metadata copy reads this whole
	// file per merge, so a realistic size is needed to surface the
	// copyModelPackLocalFile slurp cost at -alloc_space (a few-byte stub
	// hides it). Build a valid ~4 MiB JSON merge list.
	tok := benchTokenizerJSON(4 << 20)
	if result := core.WriteFile(core.PathJoin(dir, "tokenizer.json"), []byte(tok), 0o644); !result.OK {
		b.Fatalf("write tokenizer: %v", result.Value)
	}
	values := make([]float32, perTensorElements)
	for i := range values {
		values[i] = float32(i%128) * 0.01
	}
	tensors := make([]safetensorTestTensor, len(names))
	for i, name := range names {
		tensors[i] = safetensorTestTensor{Name: name, Shape: []int{perTensorElements}, Data: values}
	}
	tensorPath := core.PathJoin(dir, "model.safetensors")
	writeTestSafetensorsF32(b, tensorPath, tensors)
	return mp.ModelPack{
		Root:          dir,
		Path:          dir,
		Format:        mp.ModelPackFormatSafetensors,
		WeightFiles:   []string{tensorPath},
		TokenizerPath: core.PathJoin(dir, "tokenizer.json"),
		Architecture:  "qwen3",
	}
}

// benchTokenizerJSON builds a deterministic, valid tokenizer.json of
// approximately approxBytes bytes — a BPE merge table padded out to a
// realistic multi-MB size. Deterministic so the base and fine-tuned
// packs share an identical tokenizer (validatePackCompatibility hashes
// both and requires they match).
func benchTokenizerJSON(approxBytes int) string {
	const prefix = `{"model":{"type":"BPE","vocab":{"a":0},"merges":[`
	const suffix = `]}}`
	const row = `"abcd efgh",` // ~12 bytes per merge entry (trailing comma)
	bodyBytes := approxBytes - len(prefix) - len(suffix)
	if bodyBytes < len(row) {
		bodyBytes = len(row)
	}
	rows := core.Repeat(row, bodyBytes/len(row))
	// Drop the trailing comma so the JSON array is well-formed.
	rows = rows[:len(rows)-1]
	return prefix + rows + suffix
}

func benchMergePackNames(tensorCount int) []string {
	names := make([]string, 0, tensorCount)
	for i := range tensorCount {
		names = append(names, "model.layers."+core.Itoa(i/4)+".w."+core.Itoa(i%4)+".weight")
	}
	return names
}

func BenchmarkPacksLinear_32Tensors(b *testing.B) {
	names := benchMergePackNames(32)
	left := benchWritePackTensors(b, names, 256)
	right := benchWritePackTensors(b, names, 256)
	outDir := b.TempDir()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out := core.PathJoin(outDir, "merged-"+core.Itoa(i))
		benchMergeResult, benchMergeErr = Packs(context.Background(), Options{
			OutputPath: out,
			Method:     MethodLinear,
			Sources: []Source{
				{Pack: left, Weight: 0.5},
				{Pack: right, Weight: 0.5},
			},
		})
	}
}

func BenchmarkPacksSLERP_32Tensors(b *testing.B) {
	names := benchMergePackNames(32)
	left := benchWritePackTensors(b, names, 256)
	right := benchWritePackTensors(b, names, 256)
	outDir := b.TempDir()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out := core.PathJoin(outDir, "merged-"+core.Itoa(i))
		benchMergeResult, benchMergeErr = Packs(context.Background(), Options{
			OutputPath: out,
			Method:     MethodSLERP,
			T:          0.5,
			Sources: []Source{
				{Pack: left},
				{Pack: right},
			},
		})
	}
}
