// SPDX-Licence-Identifier: EUPL-1.2

// Real-model benchmarks for the root-package model entry points that the
// existing live suite leaves un-benched: Model.InspectAttention,
// Model.CaptureKVChunks, Model.GenerateChunks / GenerateChunkTokens, and the
// inference-contract adapter surface metaladapter.Decode /
// metaladapter.ApplyChatTemplate.
//
// They share the single cached gemma-4 e2b 4bit pack via requireLiveE2BModel
// (load once, RAM-budget-honouring). The two adapter-surface benches wrap the
// already-loaded *metal.Model in a fresh metaladapter so Decode /
// ApplyChatTemplate are exercised WITHOUT a second resident model — the adapter
// is a thin contract shim over the same engine, not a re-load.
//
// Each bench fixes a deterministic input so the result is byte-identical run to
// run, which is both the per-op stability anchor and the verification anchor for
// any production change: Decode over a fixed token slice, ApplyChatTemplate over
// a fixed message list (both pure → output equality asserted), and the
// prefill-only InspectAttention / CaptureKVChunks / chunked-generate paths over
// a fixed short prompt (greedy, temperature 0).
//
// Profile a single bench at full allocation resolution (the default
// byte-weighted MemProfileRate under-resolves the small per-call conversion
// allocs and would report a real site as a deceptive "0 samples"):
//
//	MLX_METALLIB_PATH=$PWD/../dist/lib/mlx.metallib \
//	go test -tags metal_runtime -run '^$' -bench BenchmarkInspectAttention_LiveE2B \
//	    -benchmem -benchtime=5x -memprofile inspect.prof ./go
//	go tool pprof -alloc_objects -list 'InspectAttention' inspect.prof

package mlx

import (
	"context"
	"runtime"
	"runtime/debug"
	"testing"

	"dappco.re/go/inference"
	"dappco.re/go/mlx/pkg/metal"
)

// entrypointLiveBenchMemLimit caps the heap so a runaway load can never march
// the host toward the 143 GB OOM earlier full-model benches hit. The e2b 4bit
// pack sits far under this; the limit is a guard rail, not a working figure.
const entrypointLiveBenchMemLimit = 60 << 30

// benchEntrypointPrompt is a fixed short completion-style prompt for the
// prefill-only paths (InspectAttention / CaptureKVChunks). Content is
// irrelevant to allocation counting; a stable prompt keeps any token-count or
// shape-dependent allocation deterministic across runs.
const benchEntrypointPrompt = "The cat sat on the mat."

// liveBenchProfileFull raises MemProfileRate to 1 (exact-count every
// allocation) for the timed region and restores it on the returned cleanup, so
// the process-wide rate is not left at 1 for any later bench in the binary. The
// one-time model load stays coarsely sampled — these benches attribute only the
// per-call entry-point path, not the tokenizer-JSON / safetensors load allocs.
func liveBenchProfileFull() func() {
	prev := runtime.MemProfileRate
	runtime.MemProfileRate = 1
	return func() { runtime.MemProfileRate = prev }
}

// liveBenchAdapter wraps the shared already-loaded engine in a metaladapter so
// the inference-contract surface (Decode / ApplyChatTemplate) is benched
// without a second resident model. requireLiveE2BModel has already gated Metal +
// cached weights + the reshape flake, so the cast is safe on any machine that
// got a live model back.
func liveBenchAdapter(b *testing.B) *metaladapter {
	b.Helper()
	model := requireLiveE2BModel(b)
	native, ok := model.model.(*metal.Model)
	if !ok {
		b.Skipf("live e2b engine is %T, want *metal.Model for the adapter surface", model.model)
	}
	return &metaladapter{model: native}
}

// BenchmarkInspectAttention_LiveE2B measures heap allocations on the root
// Model.InspectAttention prefill-only path against a real gemma-4 e2b model. The
// engine runs one prefill pass and returns K/Q tensors; the root layer adds the
// toRootAttentionSnapshot conversion (one struct, slices shared by reference).
func BenchmarkInspectAttention_LiveE2B(b *testing.B) {
	debug.SetMemoryLimit(entrypointLiveBenchMemLimit)
	model := requireLiveE2BModel(b)
	defer liveBenchProfileFull()()

	// One untimed pass establishes the snapshot shape and proves the path
	// returns a populated result before timing.
	warm, err := model.InspectAttention(benchEntrypointPrompt)
	if err != nil {
		b.Fatalf("InspectAttention warm-up: %v", err)
	}
	if warm == nil || warm.NumLayers == 0 {
		b.Skip("live e2b InspectAttention returned an empty snapshot — cannot measure")
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		snap, err := model.InspectAttention(benchEntrypointPrompt)
		if err != nil {
			b.Fatalf("InspectAttention: %v", err)
		}
		if snap == nil {
			b.Fatal("InspectAttention returned nil snapshot")
		}
	}
}

// BenchmarkCaptureKVChunks_LiveE2B measures heap allocations on the root
// Model.CaptureKVChunks prefill-only path against a real gemma-4 e2b model. The
// engine prefills the chunked prompt and returns the K/V cache; the root layer
// adds kvconv.ToRootKVSnapshot (arena-slab conversion of the per-layer cache).
func BenchmarkCaptureKVChunks_LiveE2B(b *testing.B) {
	debug.SetMemoryLimit(entrypointLiveBenchMemLimit)
	model := requireLiveE2BModel(b)
	defer liveBenchProfileFull()()

	ctx := context.Background()
	// Fixed two-chunk prompt: the chunked path concatenates these into the
	// same logical prompt, so the captured snapshot is deterministic.
	chunks := func() func(func(string) bool) {
		return func(yield func(string) bool) {
			if !yield("The cat sat ") {
				return
			}
			yield("on the mat.")
		}
	}

	warm, err := model.CaptureKVChunks(ctx, chunks())
	if err != nil {
		b.Fatalf("CaptureKVChunks warm-up: %v", err)
	}
	if warm == nil || len(warm.Layers) == 0 {
		b.Skip("live e2b CaptureKVChunks returned an empty snapshot — cannot measure")
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		snap, err := model.CaptureKVChunks(ctx, chunks())
		if err != nil {
			b.Fatalf("CaptureKVChunks: %v", err)
		}
		if snap == nil {
			b.Fatal("CaptureKVChunks returned nil snapshot")
		}
	}
}

// generateChunksLiveBenchSink defeats dead-code elimination of the result.
var generateChunksLiveBenchSink string

// BenchmarkGenerateChunks_LiveE2B measures per-token heap allocations on the
// root Model.GenerateChunks path against a real gemma-4 e2b model. It rides the
// same generateChunkTokensWithConfig core GenerateChunkTokens streams, plus the
// buffered Builder, so this one bench covers both the chunked buffered and the
// chunked iterator entry points (the iterator is GenerateChunks without the
// Builder). Greedy (temperature 0) makes the emitted token IDs identical run to
// run — the per-token denominator and the verification anchor.
func BenchmarkGenerateChunks_LiveE2B(b *testing.B) {
	debug.SetMemoryLimit(entrypointLiveBenchMemLimit)
	model := requireLiveE2BModel(b)
	defer liveBenchProfileFull()()

	ctx := context.Background()
	chunks := func() func(func(string) bool) {
		return func(yield func(string) bool) {
			if !yield("Once upon ") {
				return
			}
			yield("a time,")
		}
	}

	// Untimed deterministic pass counts the greedy token output via the
	// iterator entry point (GenerateChunkTokens), establishing the per-token
	// denominator (temp 0 ⇒ identical every call).
	var tokenCount int
	for range model.GenerateChunkTokens(ctx, chunks(),
		WithMaxTokens(benchGenerateMaxTokens), WithTemperature(0)) {
		tokenCount++
	}
	if tokenCount == 0 {
		b.Skip("live e2b model emitted no tokens from chunks — cannot measure per-token allocs")
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, err := model.GenerateChunks(ctx, chunks(),
			WithMaxTokens(benchGenerateMaxTokens), WithTemperature(0))
		if err != nil {
			b.Fatalf("GenerateChunks: %v", err)
		}
		generateChunksLiveBenchSink = out
	}
	b.StopTimer()

	// allocs/op is per full generation; tokens/op records the deterministic
	// denominator so a reader divides allocs/op by tokens/op for the per-token
	// figure from the bench line alone.
	b.ReportMetric(float64(tokenCount), "tokens/op")
	_ = generateChunksLiveBenchSink
}

// decodeLiveBenchSink defeats dead-code elimination of the decoded string.
var decodeLiveBenchSink string

// BenchmarkDecode_LiveE2B measures heap allocations on the inference-contract
// metaladapter.Decode path against a real gemma-4 e2b tokenizer. Decode is pure
// (token IDs → string), so the bench asserts byte-identical output across calls
// — the change-detection anchor — and the allocation count is the per-Decode
// cost the contract surface pays over a fixed-length token slice.
func BenchmarkDecode_LiveE2B(b *testing.B) {
	debug.SetMemoryLimit(entrypointLiveBenchMemLimit)
	adapter := liveBenchAdapter(b)
	defer liveBenchProfileFull()()

	// Encode a fixed string once to obtain a real, stable token slice; Decode
	// of it must round-trip to a stable string. Encoding is outside the timed
	// region — only Decode is measured.
	tokenIDs := adapter.Encode("The quick brown fox jumps over the lazy dog.")
	if len(tokenIDs) == 0 {
		b.Skip("live e2b tokenizer produced no token IDs — cannot measure Decode")
	}
	want := adapter.Decode(tokenIDs)
	if want == "" {
		b.Skip("live e2b Decode produced an empty string — cannot measure")
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		got := adapter.Decode(tokenIDs)
		if got != want {
			b.Fatalf("Decode output drifted: got %q want %q", got, want)
		}
		decodeLiveBenchSink = got
	}
	b.StopTimer()
	_ = decodeLiveBenchSink
}

// applyChatTemplateLiveBenchSink defeats dead-code elimination of the rendered
// template string.
var applyChatTemplateLiveBenchSink string

// BenchmarkApplyChatTemplate_LiveE2B measures heap allocations on the
// inference-contract metaladapter.ApplyChatTemplate path against a real gemma-4
// e2b model. ApplyChatTemplate reads the engine Info() for architecture/heads
// then renders via chat.Format — pure given the model — so the bench asserts
// byte-identical output across calls (the change anchor) and reports the
// per-render allocation cost over a fixed message list.
func BenchmarkApplyChatTemplate_LiveE2B(b *testing.B) {
	debug.SetMemoryLimit(entrypointLiveBenchMemLimit)
	adapter := liveBenchAdapter(b)
	defer liveBenchProfileFull()()

	messages := []inference.Message{
		{Role: "system", Content: "You are a helpful assistant."},
		{Role: "user", Content: "What is the capital of France?"},
	}
	want, err := adapter.ApplyChatTemplate(messages)
	if err != nil {
		b.Fatalf("ApplyChatTemplate warm-up: %v", err)
	}
	if want == "" {
		b.Skip("live e2b ApplyChatTemplate produced an empty render — cannot measure")
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		got, err := adapter.ApplyChatTemplate(messages)
		if err != nil {
			b.Fatalf("ApplyChatTemplate: %v", err)
		}
		if got != want {
			b.Fatalf("ApplyChatTemplate output drifted: got %q want %q", got, want)
		}
		applyChatTemplateLiveBenchSink = got
	}
	b.StopTimer()
	_ = applyChatTemplateLiveBenchSink
}
