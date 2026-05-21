// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the CPU-only side of fast_eval.go + fast_eval_runner.go.
// Per AX-11 — these are the pure converters that sit on the bench harness
// boundary (mlx-side <-> bench-side <-> decode-side). They fire on every
// run of the fast-eval harness: once per generation pass (Metrics +
// GenerateOptions), once per report aggregation (Info + Adapter), and
// once per decode-optimisation result (decodeResultToBench across the
// token slice). When fast-eval is run as part of an autotune loop the
// per-call cost compounds.
//
// Model-bound functions (RunFastEvalBench, NewModelFastEvalRunner's
// callbacks, the bench* state-store helpers) require a loaded *Model
// and are intentionally OUT of scope.
//
// Run:    go test -bench='BenchmarkFastEval' -benchmem -run='^$' ./go

package mlx

import (
	"testing"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference/bench"
	"dappco.re/go/inference/decode"
	"dappco.re/go/mlx/lora"
	"dappco.re/go/mlx/probe"
)

// Sinks defeat compiler DCE. Distinct from other bench files in this package.
var (
	fastEvalBenchGenConfig   GenerateConfig
	fastEvalBenchBenchMetric bench.GenerationMetrics
	fastEvalBenchBenchInfo   bench.Info
	fastEvalBenchModelInfo   ModelInfo
	fastEvalBenchBenchAdapt  bench.AdapterInfo
	fastEvalBenchLoraAdapt   lora.AdapterInfo
	fastEvalBenchModelOpts   []GenerateOption
	fastEvalBenchDecodeRes   bench.DecodeOptimisationResult
	fastEvalBenchFloat       float64
	fastEvalBenchErr         error
)

// fastEvalBenchMlxMetrics builds a populated Metrics fixture mirroring
// the shape an mlx Model returns after a single inference call.
func fastEvalBenchMlxMetrics() Metrics {
	return Metrics{
		PromptTokens:               2048,
		GeneratedTokens:            128,
		FirstTokenDuration:         12 * time.Millisecond,
		PrefillDuration:            45 * time.Millisecond,
		DecodeDuration:             950 * time.Millisecond,
		TotalDuration:              1010 * time.Millisecond,
		PrefillTokensPerSec:        14222.2,
		DecodeTokensPerSec:         134.7,
		PeakMemoryBytes:            8 << 30,
		ActiveMemoryBytes:          4 << 30,
		CacheMemoryBytes:           1 << 30,
		PromptCacheHits:            1,
		PromptCacheMisses:          0,
		PromptCacheHitTokens:       1024,
		PromptCacheMissTokens:      0,
		PromptCacheRestoreDuration: 4 * time.Millisecond,
	}
}

// fastEvalBenchMlxInfo builds a populated ModelInfo for the bench-side
// Info converters (qwen3-class adapter attached).
func fastEvalBenchMlxInfo() ModelInfo {
	return ModelInfo{
		Architecture:  "qwen3",
		VocabSize:     151936,
		NumLayers:     28,
		HiddenSize:    2048,
		QuantBits:     4,
		QuantGroup:    64,
		ContextLength: 131072,
		Adapter: lora.AdapterInfo{
			Name:       "qwen3-coder-lora",
			Path:       "/models/adapters/qwen3-coder",
			Hash:       "sha256:" + core.SHA256Hex([]byte("qwen3-coder-lora")),
			Rank:       16,
			Alpha:      32,
			Scale:      0.5,
			TargetKeys: []string{"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"},
		},
	}
}

// fastEvalBenchBenchInfo mirrors fastEvalBenchMlxInfo on the bench side
// — used as the converter input for benchInfoToModel.
func fastEvalBenchBenchInfoFixture() bench.Info {
	return bench.Info{
		Architecture:  "qwen3",
		VocabSize:     151936,
		NumLayers:     28,
		HiddenSize:    2048,
		QuantBits:     4,
		QuantGroup:    64,
		ContextLength: 131072,
		Adapter: bench.AdapterInfo{
			Name:       "qwen3-coder-lora",
			Path:       "/models/adapters/qwen3-coder",
			Hash:       "sha256:" + core.SHA256Hex([]byte("qwen3-coder-lora")),
			Rank:       16,
			Alpha:      32,
			Scale:      0.5,
			TargetKeys: []string{"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"},
		},
	}
}

// fastEvalBenchOpts builds a populated bench.GenerateOptions fixture.
func fastEvalBenchOpts(withProbe bool) bench.GenerateOptions {
	opts := bench.GenerateOptions{
		MaxTokens:     256,
		Temperature:   0.7,
		TopK:          40,
		TopP:          0.95,
		MinP:          0.05,
		StopTokens:    []int32{1, 2, 3},
		RepeatPenalty: 1.1,
	}
	if withProbe {
		opts.ProbeSink = probe.NewRecorder()
	}
	return opts
}

// fastEvalBenchDecodeResult builds a representative decode.Result for
// decodeResultToBench. A 32-token speculative-decode trace is the typical
// shape — the converter loops over Tokens to build the bench-side ID slice.
func fastEvalBenchDecodeResult(tokenCount int) decode.Result {
	tokens := make([]decode.Token, tokenCount)
	for i := range tokens {
		tokens[i] = decode.Token{ID: int32(i + 1), Text: "tok"}
	}
	return decode.Result{
		Mode:   decode.ModeSpeculative,
		Prompt: "The quick brown fox",
		Text:   "Jumps over the lazy dog",
		Tokens: tokens,
		Metrics: decode.Metrics{
			TargetTokens:   tokenCount,
			DraftTokens:    tokenCount,
			AcceptedTokens: tokenCount - 2,
			RejectedTokens: 2,
			EmittedTokens:  tokenCount,
			AcceptanceRate: float64(tokenCount-2) / float64(tokenCount),
			TargetCalls:    1,
			DraftCalls:     1,
			Duration:       500 * time.Millisecond,
			TargetDuration: 300 * time.Millisecond,
			DraftDuration:  200 * time.Millisecond,
		},
	}
}

// --- toBenchGenerateOptions — fast_eval.go boundary helper ---

func BenchmarkFastEval_ToBenchGenerateOptions_NoProbe(b *testing.B) {
	opts := fastEvalBenchOpts(false)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fastEvalBenchGenConfig = toBenchGenerateOptions(opts)
	}
}

func BenchmarkFastEval_ToBenchGenerateOptions_WithProbe(b *testing.B) {
	opts := fastEvalBenchOpts(true)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fastEvalBenchGenConfig = toBenchGenerateOptions(opts)
	}
}

// --- fromMlxMetrics — runs once per generation pass ---

func BenchmarkFastEval_FromMlxMetrics(b *testing.B) {
	metrics := fastEvalBenchMlxMetrics()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fastEvalBenchBenchMetric = fromMlxMetrics(metrics)
	}
}

// --- modelInfoToBench / benchInfoToModel ---

func BenchmarkFastEval_ModelInfoToBench(b *testing.B) {
	info := fastEvalBenchMlxInfo()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fastEvalBenchBenchInfo = modelInfoToBench(info)
	}
}

func BenchmarkFastEval_BenchInfoToModel(b *testing.B) {
	info := fastEvalBenchBenchInfoFixture()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fastEvalBenchModelInfo = benchInfoToModel(info)
	}
}

// --- loraToBenchAdapter / benchAdapterToLora ---

func BenchmarkFastEval_LoraToBenchAdapter(b *testing.B) {
	info := fastEvalBenchMlxInfo().Adapter
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fastEvalBenchBenchAdapt = loraToBenchAdapter(info)
	}
}

func BenchmarkFastEval_BenchAdapterToLora(b *testing.B) {
	info := fastEvalBenchBenchInfoFixture().Adapter
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fastEvalBenchLoraAdapt = benchAdapterToLora(info)
	}
}

// --- toModelGenerateOptions (fast_eval_runner.go) ---

func BenchmarkFastEval_ToModelGenerateOptions_Minimal(b *testing.B) {
	opts := bench.GenerateOptions{MaxTokens: 64, Temperature: 0.0}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fastEvalBenchModelOpts = toModelGenerateOptions(opts)
	}
}

func BenchmarkFastEval_ToModelGenerateOptions_FullKnobs(b *testing.B) {
	opts := fastEvalBenchOpts(false)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fastEvalBenchModelOpts = toModelGenerateOptions(opts)
	}
}

// --- decodeResultToBench — token-loop converter on the speculative path ---

func BenchmarkFastEval_DecodeResultToBench_32Tokens(b *testing.B) {
	result := fastEvalBenchDecodeResult(32)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fastEvalBenchDecodeRes = decodeResultToBench(result)
	}
}

func BenchmarkFastEval_DecodeResultToBench_256Tokens(b *testing.B) {
	result := fastEvalBenchDecodeResult(256)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fastEvalBenchDecodeRes = decodeResultToBench(result)
	}
}

// --- decodeTokensPerSecond — hit per decode-optimisation aggregation ---

func BenchmarkFastEval_DecodeTokensPerSecond_Positive(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fastEvalBenchFloat = decodeTokensPerSecond(256, 500*time.Millisecond)
	}
}

func BenchmarkFastEval_DecodeTokensPerSecond_ZeroDuration(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fastEvalBenchFloat = decodeTokensPerSecond(256, 0)
	}
}

// --- fastEvalResultError — pure result-to-error unwrapping ---

func BenchmarkFastEval_FastEvalResultError_OK(b *testing.B) {
	result := core.Result{OK: true, Value: []byte("payload")}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fastEvalBenchErr = fastEvalResultError(result)
	}
}

func BenchmarkFastEval_FastEvalResultError_FailedErr(b *testing.B) {
	result := core.Result{OK: false, Value: core.NewError("fast-eval bench failure")}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fastEvalBenchErr = fastEvalResultError(result)
	}
}
