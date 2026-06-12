// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for backend.go dispatch helpers. Per AX-11 — these fire on
// toMetalProbeSink. Per AX-11 — both fire on every Generate / Chat /
// Classify / BatchGenerate call, so the per-call allocation budget for
// the inference hot path runs through here.
//
// Run:    go test -bench='BenchmarkBackend_ToMetal' -benchmem -run='^$' ./go

package mlx

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/parser"
	state "dappco.re/go/inference/state"
	"dappco.re/go/mlx/adapter"
	"dappco.re/go/mlx/internal/metaltest"
	"dappco.re/go/mlx/kv"
	"dappco.re/go/mlx/kvconv"
	"dappco.re/go/mlx/lora"
	"dappco.re/go/mlx/pkg/metal"
	"dappco.re/go/mlx/probe"
)

// Sinks defeat compiler DCE.
var (
	backendBenchSinkHint         parser.Hint
	backendBenchSinkProbeEvent   probe.Event
	backendBenchSinkRootMetrics  Metrics
	backendBenchSinkRootToken    Token
	backendBenchSinkRootAdapter  lora.AdapterInfo
	backendBenchSinkChatMessages []metal.ChatMessage
	backendBenchSinkBlockSource  metal.KVSnapshotBlockSource
)

// --- hintForParser cache (Wave6-W1A) ---
// Per-Generate parser.Hint dispatch — pre-cached at LoadModel + on LoRA
// mutation; the cached read is the hot-path replacement for the prior
// per-call m.model.Info() fan-out (which itself cloned the native
// AdapterInfo.TargetKeys slice).

func BenchmarkBackend_HintForParser_Cached(b *testing.B) {
	model := &Model{
		model: &fakeNativeModel{
			info: metal.ModelInfo{
				Architecture: "qwen3",
				Adapter:      metal.AdapterInfo{Name: "probe-lora"},
			},
		},
		adapterInfo: lora.AdapterInfo{Name: "probe-lora"},
	}
	// Warm the cache so we measure the steady-state read, not the
	// one-time lazy build.
	model.refreshParserHint()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		backendBenchSinkHint = model.hintForParser()
	}
}

func BenchmarkBackend_HintForParser_Build(b *testing.B) {
	model := &Model{
		model: &fakeNativeModel{
			info: metal.ModelInfo{
				Architecture: "qwen3",
				Adapter:      metal.AdapterInfo{Name: "probe-lora"},
			},
		},
		adapterInfo: lora.AdapterInfo{Name: "probe-lora"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		backendBenchSinkHint = model.buildParserHint()
	}
}

// --- kvconv.MetalKVSnapshotBlockSource ---
// Retained-State prompt restore builds this source once per warm wake before
// native code streams block payloads. Keep source construction allocation-free
// so the restore path stays proportional to block payloads, not manifest size.

func BenchmarkBackend_MetalKVSnapshotBlockSource_Construct96Blocks(b *testing.B) {
	store := state.NewInMemoryStore(nil)
	bundle := benchmarkBackendStateBlockBundle(96, 512)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		source, err := kvconv.MetalKVSnapshotBlockSource(context.Background(), store, bundle, bundle.TokenCount)
		if err != nil {
			b.Fatal(err)
		}
		backendBenchSinkBlockSource = source
	}
}

func benchmarkBackendStateBlockBundle(blockCount, tokensPerBlock int) *kv.StateBlockBundle {
	blocks := make([]kv.StateBlockRef, blockCount)
	for i := range blocks {
		blocks[i] = kv.StateBlockRef{
			Index:      i,
			TokenStart: i * tokensPerBlock,
			TokenCount: tokensPerBlock,
		}
	}
	return &kv.StateBlockBundle{
		Version:    kv.StateBlockVersion,
		Kind:       kv.StateBlockBundleKind,
		TokenCount: blockCount * tokensPerBlock,
		BlockSize:  tokensPerBlock,
		Blocks:     blocks,
	}
}

// --- toRootToken (W10-AN) ---
// Per-token shuffler used by toRootClassifyResults / toRootBatchResults /
// every *Stream entry. Tiny but fires once per emitted token.

func BenchmarkBackend_ToRootToken(b *testing.B) {
	token := metal.Token{ID: 42, Text: "hello"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		backendBenchSinkRootToken = toRootToken(token)
	}
}

// --- toRootAdapterInfo (W10-AN) ---
// Called from toRootMetrics on every Metrics() read AND from
// adapterFromNativeInfo on every Info() read. Clones TargetKeys slice.

func BenchmarkBackend_ToRootAdapterInfo_Empty(b *testing.B) {
	info := metal.AdapterInfo{}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		backendBenchSinkRootAdapter = toRootAdapterInfo(info)
	}
}

func BenchmarkBackend_ToRootAdapterInfo_Typical(b *testing.B) {
	info := metal.AdapterInfo{
		Name:       "probe-lora",
		Path:       "/models/lora.safetensors",
		Hash:       "sha256:abc",
		Rank:       16,
		Alpha:      32.0,
		Scale:      2.0,
		TargetKeys: []string{"q_proj", "k_proj", "v_proj", "o_proj"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		backendBenchSinkRootAdapter = toRootAdapterInfo(info)
	}
}

// --- toRootMetrics (W10-AN) ---
// Per-Metrics() call: field-by-field shuffler. Fires on every read of
// Model.Metrics() — typically once per Generate but call sites vary.

func BenchmarkBackend_ToRootMetrics_Simple(b *testing.B) {
	metrics := metal.Metrics{
		PromptTokens:        128,
		GeneratedTokens:     64,
		PrefillTokensPerSec: 1000.0,
		DecodeTokensPerSec:  100.0,
		Adapter:             metal.AdapterInfo{Name: "probe-lora"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		backendBenchSinkRootMetrics = toRootMetrics(metrics)
	}
}

func BenchmarkBackend_ToRootMetrics_LoRA(b *testing.B) {
	metrics := metal.Metrics{
		PromptTokens:        128,
		GeneratedTokens:     64,
		PrefillTokensPerSec: 1000.0,
		DecodeTokensPerSec:  100.0,
		Adapter: metal.AdapterInfo{
			Name:       "probe-lora",
			Path:       "/models/lora.safetensors",
			TargetKeys: []string{"q_proj", "k_proj", "v_proj", "o_proj"},
		},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		backendBenchSinkRootMetrics = toRootMetrics(metrics)
	}
}

func BenchmarkBackend_ToRootMetrics_CacheProfile(b *testing.B) {
	metrics := metal.Metrics{
		PromptTokens:        30000,
		GeneratedTokens:     1024,
		PrefillTokensPerSec: 1800.0,
		DecodeTokensPerSec:  94.0,
		CacheProfile: &metal.CacheProfile{
			Architecture:       "gemma4_text",
			TotalCaches:        6,
			LocalCaches:        5,
			GlobalCaches:       1,
			SharedLayers:       2,
			LocalWindowTokens:  512,
			MaxLocalTokens:     512,
			MaxLocalCapacity:   512,
			MaxGlobalTokens:    48712,
			MaxGlobalCapacity:  71040,
			MaxProcessedTokens: 48712,
			FixedCaches:        6,
		},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		backendBenchSinkRootMetrics = toRootMetrics(metrics)
	}
}

// --- chatMessagesAsMetal (W10-AN) ---
// Per-Chat call shuffler from []inference.Message to []metal.ChatMessage.
// W10-AN replaced a make + per-message copy with a layout-guarded
// unsafe.Slice reinterpret — the bench surfaces the cost going from
// O(N) struct copy + 1 alloc to 0 / 0.

func BenchmarkBackend_ChatMessagesAsMetal_Short(b *testing.B) {
	messages := []inference.Message{
		{Role: "system", Content: "You are helpful."},
		{Role: "user", Content: "What is the capital of France?"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		backendBenchSinkChatMessages = chatMessagesAsMetal(messages)
	}
}

func BenchmarkBackend_ChatMessagesAsMetal_Long(b *testing.B) {
	messages := make([]inference.Message, 20)
	for i := range messages {
		messages[i] = inference.Message{Role: "user", Content: "turn"}
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		backendBenchSinkChatMessages = chatMessagesAsMetal(messages)
	}
}

// --- merged from backend_growth_bench_test.go (orphan sweep: benches backend.go context growth) ---
// BenchmarkBackend_ContextGrowth is the serve-path twin of
// BenchmarkGenerate_ContextGrowth (pkg/metal). The raw decode loop
// (model.Generate) is leak-free; this drives the SAME growth sweep through the
// inference-layer path the serve actually uses — NewMLXBackend → adapter.Generate
// → the inference.TextModel — to localise the serve's per-token memory leak. A
// climbing resid_mb here (where the raw loop stayed flat) puts the leak in the
// inference/adapter wrapper, not the engine core.
//
//	go test -tags 'metal_runtime model_eval' -run '^$' \
//	  -bench BenchmarkBackend_ContextGrowth -benchtime=1x dappco.re/go/mlx/
func BenchmarkBackend_ContextGrowth(b *testing.B) {
	if !metaltest.RunModelEvalTests {
		b.Skip("model-eval benchmark; build with -tags model_eval and cache mlx-community/gemma-4-e2b-it-4bit")
	}
	dir := metaltest.HFModelPath(b, "mlx-community/gemma-4-e2b-it-4bit")
	backend, err := NewMLXBackend(dir)
	if err != nil {
		b.Fatalf("NewMLXBackend: %v", err)
	}

	const prompt = "Write a long, detailed story about a lighthouse keeper and the deep ocean."
	for _, length := range []int{512, 1024, 2048} {
		b.Run(core.Sprintf("tokens_%d", length), func(b *testing.B) {
			before := GetActiveMemory()
			for b.Loop() {
				if _, err := backend.Generate(context.Background(), prompt, adapter.GenOpts{MaxTokens: length}); err != nil {
					b.Fatalf("Generate: %v", err)
				}
			}
			b.ReportMetric(float64(GetActiveMemory()-before)/(1<<20), "resid_mb")
		})
	}
}
