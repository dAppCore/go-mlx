// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the HuggingFace fit-planning + architecture-name
// classifier surface.
// Per AX-11 — PlanFits is the local-cache walker every "what models do
// I have / can I run" call hits. The architecture classifier fires per
// candidate model (search results return 10s, lists return 100s).
// InferJANG runs on every JANG/JANGTQ pack discovered.
//
// Run:    go test -bench=Benchmark -benchmem -run='^$' ./go/hf

package hf

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/memory"
)

// Sinks defeat compiler DCE.
var (
	hfSinkString string
	hfSinkInt    int
	hfSinkBool   bool
	hfSinkFit    *FitReport
	hfSinkErr    error
	hfSinkU64    uint64
)

// --- ModelConfig.architecture / contextLength / quantization helpers ---

func BenchmarkHF_ModelConfig_Architecture_Qwen3(b *testing.B) {
	config := ModelConfig{
		ModelType:     "qwen3",
		Architectures: []string{"Qwen3ForCausalLM"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		hfSinkString = config.architecture()
	}
}

func BenchmarkHF_ModelConfig_Architecture_NestedText(b *testing.B) {
	config := ModelConfig{
		ModelType: "qwen3_5",
		TextConfig: &ModelConfig{
			ModelType:     "qwen3_next",
			Architectures: []string{"Qwen3NextForCausalLM"},
		},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		hfSinkString = config.architecture()
	}
}

func BenchmarkHF_ModelConfig_ContextLength(b *testing.B) {
	config := ModelConfig{
		ContextLength:         0,
		MaxPositionEmbeddings: 40960,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		hfSinkInt = config.contextLength()
	}
}

func BenchmarkHF_ModelConfig_Quantization(b *testing.B) {
	config := ModelConfig{
		QuantizationConfig: &QuantizationConfig{Bits: 4, GroupSize: 64, Type: "affine"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bits, group := config.quantization()
		hfSinkInt = bits + group
	}
}

// --- weightFormatAndBytes / inferQuantBits ---

func BenchmarkHF_WeightFormatAndBytes_Safetensors(b *testing.B) {
	files := []ModelFile{
		{Name: "model-00001-of-00003.safetensors", Size: 1 << 30},
		{Name: "model-00002-of-00003.safetensors", Size: 1 << 30},
		{Name: "model-00003-of-00003.safetensors", Size: 1 << 30},
		{Name: "tokenizer.json", Size: 4 << 20},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		format, bytes := weightFormatAndBytes(files)
		hfSinkString = format
		hfSinkU64 = bytes
	}
}

func BenchmarkHF_WeightFormatAndBytes_Mixed(b *testing.B) {
	files := []ModelFile{
		{Name: "model.safetensors", Size: 1 << 30},
		{Name: "model.gguf", Size: 1 << 30},
		{Name: "pytorch_model.bin", Size: 1 << 30},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		format, bytes := weightFormatAndBytes(files)
		hfSinkString = format
		hfSinkU64 = bytes
	}
}

func BenchmarkHF_InferQuantBits_Q4(b *testing.B) {
	files := []ModelFile{{Name: "model-q4_k_m.gguf"}}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		hfSinkInt = inferQuantBits(files)
	}
}

func BenchmarkHF_InferQuantBits_BF16(b *testing.B) {
	files := []ModelFile{{Name: "model-bf16.safetensors"}}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		hfSinkInt = inferQuantBits(files)
	}
}

// --- estimateModelKVBytes — fires per fit-plan model ---

func BenchmarkHF_EstimateModelKVBytes_Qwen3(b *testing.B) {
	config := ModelConfig{
		HiddenSize:        2048,
		NumHiddenLayers:   28,
		NumAttentionHeads: 16,
		NumKeyValueHeads:  8,
		HeadDim:           128,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		hfSinkU64 = estimateModelKVBytes(config, 40960, 1, 2)
	}
}

// --- InferJANG — runs against tag + filename needles for JANG packs ---

func BenchmarkHF_InferJANG_JANGTQ(b *testing.B) {
	meta := ModelMetadata{
		ID:   "dealignai/MiniMax-M2.7-JANGTQ-CRACK",
		Tags: []string{"mlx", "jang", "jangtq", "minimax_m2"},
		Files: []ModelFile{
			{Name: "model-00001-of-00061.safetensors"},
			{Name: "jangtq_runtime.safetensors"},
		},
		Config: ModelConfig{
			QuantizationConfig: &QuantizationConfig{GroupSize: 64},
		},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		info := InferJANG(meta)
		if info != nil {
			hfSinkString = info.Profile
		}
	}
}

func BenchmarkHF_InferJANG_Miss(b *testing.B) {
	meta := ModelMetadata{
		ID:    "Qwen/Qwen3-0.6B",
		Tags:  []string{"mlx", "text-generation"},
		Files: []ModelFile{{Name: "model.safetensors"}},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		info := InferJANG(meta)
		hfSinkBool = info != nil
	}
}

// --- PlanFits — end-to-end against a fake source (no network) ---

type benchFitSource struct {
	meta ModelMetadata
}

func (s *benchFitSource) SearchModels(_ context.Context, _ string, _ int) ([]ModelMetadata, error) {
	return []ModelMetadata{s.meta}, nil
}

func (s *benchFitSource) ModelMetadata(_ context.Context, _ string) (ModelMetadata, error) {
	return s.meta, nil
}

func BenchmarkHF_PlanFits_SingleRemote(b *testing.B) {
	source := &benchFitSource{
		meta: ModelMetadata{
			ID: "Qwen/Qwen3-0.6B",
			Config: ModelConfig{
				ModelType:             "qwen3",
				HiddenSize:            1024,
				NumHiddenLayers:       28,
				NumAttentionHeads:     16,
				NumKeyValueHeads:      8,
				MaxPositionEmbeddings: 40960,
				Quantization:          &QuantizationConfig{Bits: 4, GroupSize: 64},
			},
			Files: []ModelFile{
				{Name: "model.safetensors", Size: 420 * 1024 * 1024},
				{Name: "tokenizer.json", Size: 4 * 1024 * 1024},
			},
		},
	}
	cfg := FitConfig{
		Query:      "qwen 0.6b",
		MaxResults: 5,
		Device: memory.DeviceInfo{
			Architecture:                 "apple-m3-ultra",
			MemorySize:                   96 * memory.GiB,
			MaxRecommendedWorkingSetSize: 86 * memory.GiB,
		},
		Source: source,
	}
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		hfSinkFit, hfSinkErr = PlanFits(ctx, cfg)
	}
}

func BenchmarkHF_PlanFits_LocalCache(b *testing.B) {
	cacheRoot := core.JoinPath(b.TempDir(), "models--mlx-community--gemma-4-e2b-it-4bit")
	dir := core.JoinPath(cacheRoot, "snapshots", "abc123")
	if result := core.MkdirAll(dir, 0o755); !result.OK {
		b.Fatalf("mkdir %s: %v", dir, result.Value)
	}
	if r := core.WriteFile(core.JoinPath(dir, "config.json"), []byte(`{
		"model_type": "gemma4_text",
		"hidden_size": 2048,
		"num_hidden_layers": 26,
		"num_attention_heads": 8,
		"num_key_value_heads": 4,
		"max_position_embeddings": 131072,
		"quantization_config": {"bits": 4, "group_size": 64}
	}`), 0o644); !r.OK {
		b.Fatalf("write config: %v", r.Value)
	}
	if r := core.WriteFile(core.JoinPath(dir, "model-00001-of-00001.safetensors"), []byte("stub"), 0o644); !r.OK {
		b.Fatalf("write weights: %v", r.Value)
	}
	cfg := FitConfig{
		LocalPaths: []string{cacheRoot},
		Device: memory.DeviceInfo{
			Architecture:                 "apple-m1-pro",
			MemorySize:                   16 * memory.GiB,
			MaxRecommendedWorkingSetSize: 13 * memory.GiB,
		},
	}
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		hfSinkFit, hfSinkErr = PlanFits(ctx, cfg)
	}
}
