// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the merge/compare base-vs-fine-tuned weight delta
// surface. Per AX-11 — ComparePacks is invoked per "what changed in
// this fine-tune?" inspection (CLI/UI driven, but the inner work is
// IO + math heavy). The per-tensor compareTensorRefs walks every
// element across two readers and accumulates RMS / cosine — this is
// the surface a Codex optimisation pass would target if the eval
// surface gets called often. The aux helpers (compareCosine,
// cloneCompareLabels, cloneUint64s, recordTensorDelta, appendTensorDelta,
// validateComparePack) fire per call and are cheap enough that
// regressions show up only under N tensor reports.
//
// Run:    go test -bench='BenchmarkCompare|BenchmarkCompareTensorRefs|BenchmarkComparePacks|BenchmarkCompareCosine|BenchmarkCloneCompareLabels|BenchmarkCloneUint64s|BenchmarkRecordTensorDelta|BenchmarkAppendTensorDelta|BenchmarkValidateComparePack' -benchmem -run='^$' ./go/merge

package merge

import (
	"context"
	"math"
	"testing"

	core "dappco.re/go"
	mp "dappco.re/go/mlx/pack"
	"dappco.re/go/mlx/safetensors"
)

// Sinks defeat compiler DCE.
var (
	benchCompareResult *CompareResult
	benchCompareErr    error
	benchCompareFloat  float64
	benchCompareLabels map[string]string
	benchCompareDims   []uint64
	benchCompareDelta  TensorDelta
)

// benchCompareScratchPack writes a small dense safetensors pack to a
// temp dir and returns a pack pointed at it. Mirrors
// writeDenseSafetensorsPack in helpers_test.go but takes *testing.B.
func benchCompareScratchPack(b *testing.B, modelType string, tensorNames []string, shape []int, perTensorElements int) mp.ModelPack {
	b.Helper()
	dir := b.TempDir()
	// config.json + tokenizer.json — minimal pack metadata.
	cfg := core.Sprintf(`{"model_type":%q,"vocab_size":151936,"hidden_size":2048,"num_hidden_layers":28,"max_position_embeddings":40960}`, modelType)
	if result := core.WriteFile(core.PathJoin(dir, "config.json"), []byte(cfg), 0o644); !result.OK {
		b.Fatalf("write config: %v", result.Value)
	}
	tok := `{"model":{"type":"BPE","vocab":{"a":0},"merges":[]}}`
	if result := core.WriteFile(core.PathJoin(dir, "tokenizer.json"), []byte(tok), 0o644); !result.OK {
		b.Fatalf("write tokenizer: %v", result.Value)
	}

	// Each tensor — fill with deterministic finite values; vary by
	// index so cosine doesn't degenerate to 0/1.
	tensorPath := core.PathJoin(dir, "model.safetensors")
	values := make([]float32, perTensorElements)
	for i := range values {
		values[i] = float32(i%128) * 0.01
	}
	// Stage all tensors into a synthetic safetensors file in one go.
	type entry struct {
		DType       string `json:"dtype"`
		Shape       []int  `json:"shape"`
		DataOffsets []int  `json:"data_offsets"`
	}
	header := map[string]entry{}
	var body []byte
	for _, name := range tensorNames {
		start := len(body)
		buf := make([]byte, perTensorElements*4)
		for i, v := range values {
			bits := uint32FromFloat32Bits(v)
			buf[i*4+0] = byte(bits)
			buf[i*4+1] = byte(bits >> 8)
			buf[i*4+2] = byte(bits >> 16)
			buf[i*4+3] = byte(bits >> 24)
		}
		body = append(body, buf...)
		header[name] = entry{DType: "F32", Shape: shape, DataOffsets: []int{start, len(body)}}
	}
	encoded := core.JSONMarshal(header)
	if !encoded.OK {
		b.Fatalf("marshal header: %v", encoded.Value)
	}
	headerBytes := encoded.Value.([]byte)
	out := make([]byte, 8+len(headerBytes)+len(body))
	hl := uint64(len(headerBytes))
	for i := range 8 {
		out[i] = byte(hl >> (8 * i))
	}
	copy(out[8:], headerBytes)
	copy(out[8+len(headerBytes):], body)
	if result := core.WriteFile(tensorPath, out, 0o644); !result.OK {
		b.Fatalf("write safetensors: %v", result.Value)
	}

	return mp.ModelPack{
		Root:          dir,
		Path:          dir,
		Format:        mp.ModelPackFormatSafetensors,
		WeightFiles:   []string{tensorPath},
		TokenizerPath: core.PathJoin(dir, "tokenizer.json"),
		Architecture:  modelType,
	}
}

// uint32FromFloat32Bits exposes math.Float32bits under a bench-local
// name so the staging path stays grep-friendly.
func uint32FromFloat32Bits(f float32) uint32 {
	return math.Float32bits(f)
}

// --- compareTensorRefs — per-tensor inner math + IO ---

func BenchmarkCompareTensorRefs_4096Elements(b *testing.B) {
	name := "model.layers.0.self_attn.q_proj.weight"
	left := benchCompareScratchPack(b, "qwen3", []string{name}, []int{4096}, 4096)
	right := benchCompareScratchPack(b, "qwen3", []string{name}, []int{4096}, 4096)
	leftIdx, err := safetensors.IndexFiles(left.WeightFiles)
	if err != nil {
		b.Fatal(err)
	}
	rightIdx, err := safetensors.IndexFiles(right.WeightFiles)
	if err != nil {
		b.Fatal(err)
	}
	ref := leftIdx.Tensors[name]
	tunedRef := rightIdx.Tensors[name]
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cache := newFileCache()
		benchCompareDelta, benchCompareErr = compareTensorRefs(context.Background(), cache, ref, tunedRef, modelMergeTensorChunkElements)
		cache.close()
	}
}

func BenchmarkCompareTensorRefs_98304Elements(b *testing.B) {
	name := "model.layers.0.mlp.gate_proj.weight"
	left := benchCompareScratchPack(b, "qwen3", []string{name}, []int{98304}, 98304)
	right := benchCompareScratchPack(b, "qwen3", []string{name}, []int{98304}, 98304)
	leftIdx, err := safetensors.IndexFiles(left.WeightFiles)
	if err != nil {
		b.Fatal(err)
	}
	rightIdx, err := safetensors.IndexFiles(right.WeightFiles)
	if err != nil {
		b.Fatal(err)
	}
	ref := leftIdx.Tensors[name]
	tunedRef := rightIdx.Tensors[name]
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cache := newFileCache()
		benchCompareDelta, benchCompareErr = compareTensorRefs(context.Background(), cache, ref, tunedRef, modelMergeTensorChunkElements)
		cache.close()
	}
}

// Shape mismatch path — early-return without reading bytes.
func BenchmarkCompareTensorRefs_ShapeMismatch(b *testing.B) {
	name := "model.norm.weight"
	left := benchCompareScratchPack(b, "qwen3", []string{name}, []int{1024}, 1024)
	right := benchCompareScratchPack(b, "qwen3", []string{name}, []int{2048}, 2048)
	leftIdx, err := safetensors.IndexFiles(left.WeightFiles)
	if err != nil {
		b.Fatal(err)
	}
	rightIdx, err := safetensors.IndexFiles(right.WeightFiles)
	if err != nil {
		b.Fatal(err)
	}
	ref := leftIdx.Tensors[name]
	tunedRef := rightIdx.Tensors[name]
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cache := newFileCache()
		benchCompareDelta, benchCompareErr = compareTensorRefs(context.Background(), cache, ref, tunedRef, modelMergeTensorChunkElements)
		cache.close()
	}
}

// --- ComparePacks — end-to-end across a small multi-tensor pack ---

func BenchmarkComparePacks_8Tensors_1024Elements(b *testing.B) {
	names := []string{
		"model.layers.0.self_attn.q_proj.weight",
		"model.layers.0.self_attn.k_proj.weight",
		"model.layers.0.self_attn.v_proj.weight",
		"model.layers.0.self_attn.o_proj.weight",
		"model.layers.0.mlp.gate_proj.weight",
		"model.layers.0.mlp.up_proj.weight",
		"model.layers.0.mlp.down_proj.weight",
		"model.norm.weight",
	}
	base := benchCompareScratchPack(b, "qwen3", names, []int{1024}, 1024)
	tuned := benchCompareScratchPack(b, "qwen3", names, []int{1024}, 1024)
	opts := CompareOptions{
		Base:             base,
		FineTuned:        tuned,
		IncludeUnchanged: true,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchCompareResult, benchCompareErr = ComparePacks(context.Background(), opts)
	}
}

// --- compareCosine — per-tensor inline post-chunk arithmetic ---

func BenchmarkCompareCosine_NonZero(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchCompareFloat = compareCosine(1.5, 2.0, 3.0)
	}
}

func BenchmarkCompareCosine_BothZero(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchCompareFloat = compareCosine(0, 0, 0)
	}
}

func BenchmarkCompareCosine_OneZero(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchCompareFloat = compareCosine(1.5, 0, 3.0)
	}
}

// --- cloneCompareLabels / cloneUint64s — small hot helpers ---

func BenchmarkCloneCompareLabels_Empty(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchCompareLabels = cloneCompareLabels(nil)
	}
}

func BenchmarkCloneCompareLabels_FourEntries(b *testing.B) {
	in := map[string]string{
		"experiment": "delta-1",
		"runner":     "cladius",
		"base":       "qwen3-7b",
		"adapter":    "lora-a",
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchCompareLabels = cloneCompareLabels(in)
	}
}

func BenchmarkCloneUint64s_Shape4D(b *testing.B) {
	in := []uint64{4, 28, 2048, 64}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchCompareDims = cloneUint64s(in)
	}
}

func BenchmarkCloneUint64s_Empty(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchCompareDims = cloneUint64s(nil)
	}
}

// --- recordTensorDelta / appendTensorDelta — accumulator helpers; run
// per tensor inside ComparePacks ---

func BenchmarkRecordTensorDelta_Changed(b *testing.B) {
	result := &CompareResult{}
	acc := compareAccumulator{}
	opts := CompareOptions{IncludeUnchanged: true}
	delta := TensorDelta{
		Name:         "model.layers.0.self_attn.q_proj.weight",
		Status:       CompareStatusChanged,
		Elements:     98304,
		MeanAbsDelta: 0.01,
		RMSDelta:     0.02,
		MaxAbsDelta:  0.05,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		recordTensorDelta(result, &acc, opts, delta)
	}
}

func BenchmarkAppendTensorDelta_UnchangedIncluded(b *testing.B) {
	result := &CompareResult{}
	opts := CompareOptions{IncludeUnchanged: true, MaxTensorReports: 0}
	delta := TensorDelta{
		Name:     "model.norm.weight",
		Status:   CompareStatusUnchanged,
		Elements: 1024,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		appendTensorDelta(result, opts, delta)
	}
}

func BenchmarkAppendTensorDelta_UnchangedSkipped(b *testing.B) {
	result := &CompareResult{}
	opts := CompareOptions{IncludeUnchanged: false}
	delta := TensorDelta{
		Name:     "model.norm.weight",
		Status:   CompareStatusUnchanged,
		Elements: 1024,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		appendTensorDelta(result, opts, delta)
	}
}

// --- validateComparePack — gate on every ComparePacks call ---

func BenchmarkValidateComparePack_Valid(b *testing.B) {
	pack := mp.ModelPack{
		Root:          "/tmp/bench-pack",
		Path:          "/tmp/bench-pack",
		Format:        mp.ModelPackFormatSafetensors,
		WeightFiles:   []string{"/tmp/bench-pack/model.safetensors"},
		TokenizerPath: "/tmp/bench-pack/tokenizer.json",
		Architecture:  "qwen3",
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchCompareErr = validateComparePack("base", pack)
	}
}

func BenchmarkValidateComparePack_MissingRoot(b *testing.B) {
	pack := mp.ModelPack{Format: mp.ModelPackFormatSafetensors, WeightFiles: []string{"x"}}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchCompareErr = validateComparePack("base", pack)
	}
}
