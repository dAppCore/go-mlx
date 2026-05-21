// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for helpers.go — pure-functional helpers used across the
// mlx root package. Per AX-11 — firstNonEmpty / firstPositive fire per
// model load (config resolution); modelInfoToMemory / modelInfoToBundle
// fire per session create + per eval/bench report (one event per call,
// hundreds per process); indexString backs the openai.go and hf_fit
// surfaces; cloneStringMap and renderTokensText sit in the dataset
// stream + state-chapter assembly path. Per AX-11 — anything that
// fires per request/per sample wants its alloc shape pinned.
//
// Run:    go test -bench='BenchmarkHelpers' -benchmem -run='^$' ./go

package mlx

import (
	"testing"

	"dappco.re/go/mlx/bundle"
	"dappco.re/go/mlx/memory"
)

// Sinks defeat compiler DCE.
var (
	helpersBenchSinkString   string
	helpersBenchSinkInt      int
	helpersBenchSinkMemory   memory.ModelInfo
	helpersBenchSinkBundle   bundle.ModelInfo
	helpersBenchSinkSampler  bundle.Sampler
	helpersBenchSinkMap      map[string]string
	helpersBenchSinkText     string
	helpersBenchSinkIndexInt int
)

// --- firstNonEmpty ---

// First arg is empty/whitespace; second wins. Mirrors the "primary then
// fallback" pattern dataset_stream / model_pack callers use.
func BenchmarkHelpers_FirstNonEmpty_FallsThrough(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		helpersBenchSinkString = firstNonEmpty("", "  ", "fallback-name")
	}
}

func BenchmarkHelpers_FirstNonEmpty_FirstWins(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		helpersBenchSinkString = firstNonEmpty("primary", "fallback", "fallback")
	}
}

// --- firstPositive ---

func BenchmarkHelpers_FirstPositive_FirstWins(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		helpersBenchSinkInt = firstPositive(2048, 1024, 256)
	}
}

func BenchmarkHelpers_FirstPositive_FallsThrough(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		helpersBenchSinkInt = firstPositive(0, -1, 0, 256)
	}
}

// --- modelInfoToMemory ---
// Typical-shape ModelInfo, no Adapter (the agent / memory / fast-eval
// path) — matches the qwen3-class fixture in the existing memory_plan
// tests.

func benchHelpersModelInfo() ModelInfo {
	return ModelInfo{
		Architecture:  "qwen3",
		VocabSize:     151936,
		NumLayers:     28,
		HiddenSize:    2048,
		QuantBits:     4,
		QuantGroup:    64,
		ContextLength: 40960,
	}
}

func BenchmarkHelpers_ModelInfoToMemory(b *testing.B) {
	info := benchHelpersModelInfo()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		helpersBenchSinkMemory = modelInfoToMemory(info)
	}
}

// --- modelInfoToBundle ---

func BenchmarkHelpers_ModelInfoToBundle(b *testing.B) {
	info := benchHelpersModelInfo()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		helpersBenchSinkBundle = modelInfoToBundle(info)
	}
}

// --- sampleFromGenerateConfig ---
// Mirrors the fast_eval_runner code path — config copied per generation
// call. StopTokens slice copy is the dominant alloc.

func BenchmarkHelpers_SampleFromGenerateConfig_NoStops(b *testing.B) {
	cfg := GenerateConfig{MaxTokens: 256, Temperature: 0.7, TopK: 40, TopP: 0.9}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		helpersBenchSinkSampler = sampleFromGenerateConfig(cfg)
	}
}

func BenchmarkHelpers_SampleFromGenerateConfig_WithStops(b *testing.B) {
	cfg := GenerateConfig{
		MaxTokens:   256,
		Temperature: 0.7,
		TopK:        40,
		TopP:        0.9,
		MinP:        0.05,
		StopTokens:  []int32{1, 2, 3, 4, 5, 6, 7, 8},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		helpersBenchSinkSampler = sampleFromGenerateConfig(cfg)
	}
}

// --- renderTokensText ---
// Lower-bound (32 tokens) is the small-prompt fast-eval shape; typical
// (256 tokens) is one generated response in a fast-eval call.

func benchHelpersTokens(n int) []Token {
	out := make([]Token, n)
	for i := range out {
		out[i] = Token{ID: int32(i), Text: "tok"}
	}
	return out
}

func BenchmarkHelpers_RenderTokensText_32(b *testing.B) {
	tokens := benchHelpersTokens(32)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		helpersBenchSinkText = renderTokensText(tokens)
	}
}

func BenchmarkHelpers_RenderTokensText_256(b *testing.B) {
	tokens := benchHelpersTokens(256)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		helpersBenchSinkText = renderTokensText(tokens)
	}
}

// --- cloneStringMap ---

func BenchmarkHelpers_CloneStringMap_Empty(b *testing.B) {
	var meta map[string]string
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		helpersBenchSinkMap = cloneStringMap(meta)
	}
}

func BenchmarkHelpers_CloneStringMap_Typical(b *testing.B) {
	meta := map[string]string{
		"architecture": "qwen3",
		"quant":        "q4_0",
		"source":       "fast-eval",
		"adapter":      "lora",
		"run_id":       "0x1234abcd",
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		helpersBenchSinkMap = cloneStringMap(meta)
	}
}

// --- indexString ---
// Substring search — kicks in for openai.go / hf_fit substring matches.
// Worst case is when the needle exists deep in the haystack.

func BenchmarkHelpers_IndexString_EarlyHit(b *testing.B) {
	haystack := "model.layers.0.self_attn.q_proj.weight"
	needle := "self_attn"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		helpersBenchSinkIndexInt = indexString(haystack, needle)
	}
}

func BenchmarkHelpers_IndexString_LateHit(b *testing.B) {
	haystack := "model.layers.27.self_attn.q_proj.weight"
	needle := "weight"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		helpersBenchSinkIndexInt = indexString(haystack, needle)
	}
}

func BenchmarkHelpers_IndexString_Miss(b *testing.B) {
	haystack := "model.layers.12.self_attn.q_proj.weight"
	needle := "expert.gate"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		helpersBenchSinkIndexInt = indexString(haystack, needle)
	}
}

func BenchmarkHelpers_IndexString_EmptyNeedle(b *testing.B) {
	haystack := "model.layers.12.self_attn.q_proj.weight"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		helpersBenchSinkIndexInt = indexString(haystack, "")
	}
}
