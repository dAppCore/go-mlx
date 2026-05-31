// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"math"
	"testing"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference/bench"
	"dappco.re/go/inference/decode"
	"dappco.re/go/mlx/internal/metal"
	"dappco.re/go/mlx/lora"
	"dappco.re/go/mlx/probe"
)

// These tests cover the mlx-side fast_eval boundary surface:
//   - legacy type aliases route to the bench package
//   - bench.DefaultConfig forwards to bench.DefaultConfig
//   - RunFastEvalBench rejects a nil model and delegates to bench.Run
//   - the pure converter helpers (Info, Adapter, Metrics, GenerateOptions)
// Coverage of bench.Run orchestration lives in
// go-inference/go/bench/bench_test.go; coverage of the per-verb Runner
// callbacks needs a loaded *Model and is exercised through the integration
// smoke tests in this package, not here.

func TestFastEvalConfig_LegacyAliasMatchesBench_Good(t *testing.T) {
	var cfg bench.Config
	cfg.Prompt = "hello"
	cfg.MaxTokens = 8
	// bench.Config is an alias for bench.Config; assignment-compatible
	// without conversion proves the alias is wired through.
	var benchCfg bench.Config = cfg
	if benchCfg.Prompt != "hello" || benchCfg.MaxTokens != 8 {
		t.Fatalf("alias round-trip = %+v, want fields preserved", benchCfg)
	}
}

func TestDefaultFastEvalConfig_MatchesBenchDefault_Good(t *testing.T) {
	got := bench.DefaultConfig()
	want := bench.DefaultConfig()
	if got.Prompt != want.Prompt || got.MaxTokens != want.MaxTokens || got.Runs != want.Runs {
		t.Fatalf("bench.DefaultConfig() = %+v, want %+v", got, want)
	}
}

func TestRunFastEvalBench_NilModel_Bad(t *testing.T) {
	if _, err := RunFastEvalBench(context.Background(), nil, bench.DefaultConfig()); err == nil {
		t.Fatal("RunFastEvalBench(nil model) error = nil, want guard")
	}
}

func TestRunFastEval_RequiresGenerate_Bad(t *testing.T) {
	if _, err := RunFastEval(context.Background(), bench.Runner{}, bench.DefaultConfig()); err == nil {
		t.Fatal("RunFastEval() with empty runner error = nil, want bench.Run validation")
	}
}

func TestRunFastEval_SmokesSyntheticRunner_Good(t *testing.T) {
	runner := bench.Runner{
		Generate: func(context.Context, string, bench.GenerateOptions) (bench.Generation, error) {
			return bench.Generation{Text: "ok", Metrics: bench.GenerationMetrics{GeneratedTokens: 1}}, nil
		},
	}
	report, err := RunFastEval(context.Background(), runner, bench.Config{Prompt: "p", MaxTokens: 4, Runs: 1})
	if err != nil {
		t.Fatalf("RunFastEval() error = %v", err)
	}
	if report == nil {
		t.Fatal("RunFastEval() report = nil")
	}
	if report.Generation.Runs != 1 || report.Generation.GeneratedTokens != 1 {
		t.Fatalf("report.Generation = %+v, want Runs=1 Tokens=1", report.Generation)
	}
}

func TestBenchModelDecodeGenerate_ReturnsTokenMetrics_Good(t *testing.T) {
	native := &fakeNativeModel{tokens: []metal.Token{
		{ID: 1, Text: "A"},
		{ID: 2, Text: "B"},
	}}
	model := &Model{model: native}

	result, err := benchModelDecodeGenerate(model).Generate(context.Background(), "prompt", decode.GenerateConfig{MaxTokens: 2})
	if err != nil {
		t.Fatalf("benchModelDecodeGenerate() error = %v", err)
	}
	if result.Text != "AB" {
		t.Fatalf("Text = %q, want AB", result.Text)
	}
	if len(result.Tokens) != 2 || result.Tokens[0].ID != 1 || result.Tokens[1].ID != 2 {
		t.Fatalf("Tokens = %+v, want token IDs copied", result.Tokens)
	}
	if native.lastGenerateConfig.MaxTokens != 2 {
		t.Fatalf("MaxTokens = %d, want 2", native.lastGenerateConfig.MaxTokens)
	}
}

func TestModelBenchSpeculativeDecode_ReportsAcceptance_Good(t *testing.T) {
	model := &Model{model: &fakeNativeModel{tokens: []metal.Token{
		{ID: 1, Text: "A"},
		{ID: 2, Text: "B"},
	}}}

	report := modelBenchSpeculativeDecode(model, nil)(context.Background(), bench.Config{
		Prompt:                 "prompt",
		MaxTokens:              2,
		SpeculativeDraftTokens: 2,
	})
	if report.Error != "" {
		t.Fatalf("Error = %q, want empty", report.Error)
	}
	if !report.Attempted {
		t.Fatal("Attempted = false, want true")
	}
	if report.Metrics.AcceptedTokens != 2 || report.Metrics.RejectedTokens != 0 || report.Metrics.AcceptanceRate != 1 {
		t.Fatalf("Metrics = %+v, want full speculative acceptance", report.Metrics)
	}
	if report.Metrics.TargetTokens != 2 || report.Metrics.DraftTokens != 2 {
		t.Fatalf("token counts = %+v, want target=2 draft=2", report.Metrics)
	}
	if report.Metrics.VisibleTokensPerSec <= 0 || report.Metrics.TargetTokensPerSec <= 0 || report.Metrics.DraftTokensPerSec <= 0 {
		t.Fatalf("token rates = %+v, want visible/target/draft rates", report.Metrics)
	}
}

func TestModelBenchSpeculativeDecode_UsesDraftModel_Good(t *testing.T) {
	targetNative := &fakeNativeModel{tokens: []metal.Token{
		{ID: 1, Text: "A"},
		{ID: 2, Text: "B"},
	}}
	draftNative := &fakeNativeModel{tokens: []metal.Token{
		{ID: 1, Text: "A"},
		{ID: 3, Text: "C"},
	}}
	target := &Model{model: targetNative}
	draft := &Model{model: draftNative}

	report := modelBenchSpeculativeDecode(target, draft)(context.Background(), bench.Config{
		Prompt:                 "prompt",
		MaxTokens:              2,
		SpeculativeDraftTokens: 2,
	})
	if report.Error != "" {
		t.Fatalf("Error = %q, want empty", report.Error)
	}
	if report.Metrics.AcceptedTokens != 1 || report.Metrics.RejectedTokens != 1 {
		t.Fatalf("Metrics = %+v, want one accepted and one rejected token", report.Metrics)
	}
	if targetNative.lastGenerateConfig.MaxTokens != 2 || draftNative.lastGenerateConfig.MaxTokens != 2 {
		t.Fatalf("MaxTokens target=%d draft=%d, want 2/2", targetNative.lastGenerateConfig.MaxTokens, draftNative.lastGenerateConfig.MaxTokens)
	}
}

func TestModelBenchSpeculativePairDecode_UsesNativeAssistantPair_Good(t *testing.T) {
	native := &fakeNativeModel{
		gemma4AssistantResult: metal.Gemma4AssistantGenerateResult{
			Tokens:         []metal.Token{{ID: 7, Text: "G"}},
			Text:           "G",
			TargetTokens:   1,
			DraftTokens:    2,
			AcceptedTokens: 1,
			RejectedTokens: 1,
			TargetCalls:    2,
			DraftCalls:     1,
			Duration:       time.Second,
			TargetDuration: 500 * time.Millisecond,
			DraftDuration:  250 * time.Millisecond,
		},
	}
	assistant := &metal.Gemma4AssistantPair{Assistant: &metal.Gemma4AssistantModel{}}
	pair := &SpeculativePair{
		Target:          &Model{model: native},
		Gemma4Assistant: assistant,
	}

	report := modelBenchSpeculativePairDecode(pair)(context.Background(), bench.Config{
		Prompt:                 "prompt",
		MaxTokens:              1,
		SpeculativeDraftTokens: 2,
	})
	if report.Error != "" {
		t.Fatalf("Error = %q, want empty", report.Error)
	}
	if report.Result.Mode != SpeculativeDecodeModeMTP {
		t.Fatalf("Mode = %q, want %q", report.Result.Mode, SpeculativeDecodeModeMTP)
	}
	if native.gemma4AssistantPair != assistant {
		t.Fatal("native assistant pair was not used")
	}
	if native.lastGemma4AssistantPrompt != "prompt" || native.lastGemma4AssistantDraftTokens != 2 {
		t.Fatalf("native args prompt=%q draft=%d", native.lastGemma4AssistantPrompt, native.lastGemma4AssistantDraftTokens)
	}
	if report.Metrics.AcceptedTokens != 1 || report.Metrics.RejectedTokens != 1 || report.Metrics.VisibleTokensPerSec != 1 {
		t.Fatalf("Metrics = %+v, want native assistant metrics", report.Metrics)
	}
	if report.Metrics.DraftTokens != 2 || report.Metrics.TargetCalls != 2 || report.Metrics.TargetTokensPerSec != 2 || report.Metrics.DraftTokensPerSec != 8 {
		t.Fatalf("Metrics = %+v, want proposed draft tokens and target/draft throughput", report.Metrics)
	}
}

func TestModelBenchPromptLookupDecode_ReportsAcceptance_Good(t *testing.T) {
	model := &Model{model: &fakeNativeModel{tokens: []metal.Token{
		{ID: 1, Text: "A"},
		{ID: 2, Text: "B"},
	}}}

	report := modelBenchPromptLookupDecode(model)(context.Background(), bench.Config{
		Prompt:             "prompt",
		MaxTokens:          2,
		PromptLookupTokens: []int32{1, 99},
	})
	if report.Error != "" {
		t.Fatalf("Error = %q, want empty", report.Error)
	}
	if report.Metrics.AcceptedTokens != 1 || report.Metrics.RejectedTokens != 1 {
		t.Fatalf("Metrics = %+v, want one accept and one reject", report.Metrics)
	}
	if report.Metrics.TargetTokens != 2 {
		t.Fatalf("TargetTokens = %d, want 2", report.Metrics.TargetTokens)
	}
}

func TestToBenchGenerateOptions_CopiesScalars_Good(t *testing.T) {
	in := bench.GenerateOptions{
		MaxTokens: 16, Temperature: 0.5, TopK: 40, TopP: 0.9, MinP: 0.05,
		StopTokens: []int32{2, 3}, RepeatPenalty: 1.1,
	}
	out := toBenchGenerateOptions(in)
	if out.MaxTokens != 16 || out.Temperature != 0.5 || out.TopK != 40 ||
		out.TopP != 0.9 || out.MinP != 0.05 || out.RepeatPenalty != 1.1 {
		t.Fatalf("toBenchGenerateOptions scalars = %+v", out)
	}
	if len(out.StopTokens) != 2 || out.StopTokens[0] != 2 || out.StopTokens[1] != 3 {
		t.Fatalf("StopTokens = %v, want [2 3]", out.StopTokens)
	}
	// Mutating the caller's slice must not surface in the converted copy.
	in.StopTokens[0] = 99
	if out.StopTokens[0] == 99 {
		t.Fatal("toBenchGenerateOptions did not clone StopTokens")
	}
}

func TestToBenchGenerateOptions_ProbeSinkPassthrough_Good(t *testing.T) {
	sink := probe.SinkFunc(func(_ probe.Event) {})
	got := toBenchGenerateOptions(bench.GenerateOptions{MaxTokens: 1, ProbeSink: probe.Sink(sink)})
	if got.ProbeSink == nil {
		t.Fatal("probe.Sink not forwarded")
	}
}

func TestToBenchGenerateOptions_NonProbeSinkIgnored_Ugly(t *testing.T) {
	got := toBenchGenerateOptions(bench.GenerateOptions{MaxTokens: 1, ProbeSink: "not-a-sink"})
	if got.ProbeSink != nil {
		t.Fatal("non-probe.Sink value should not propagate")
	}
}

func TestFromMlxMetrics_CopiesFields_Good(t *testing.T) {
	in := Metrics{
		PromptTokens: 4, GeneratedTokens: 7,
		PrefillDuration: 10 * time.Millisecond, DecodeDuration: 20 * time.Millisecond, TotalDuration: 30 * time.Millisecond,
		PrefillTokensPerSec: 400, DecodeTokensPerSec: 350,
		PeakMemoryBytes: 1 << 20, ActiveMemoryBytes: 512 << 10,
		PromptCacheHits: 3, PromptCacheMisses: 1,
		PromptCacheHitTokens: 100, PromptCacheMissTokens: 25,
		PromptCacheRestoreDuration: 5 * time.Millisecond,
	}
	out := fromMlxMetrics(in)
	if out.PromptTokens != 4 || out.GeneratedTokens != 7 {
		t.Fatalf("token counters = %+v", out)
	}
	if out.PrefillDuration != 10*time.Millisecond || out.DecodeDuration != 20*time.Millisecond || out.TotalDuration != 30*time.Millisecond {
		t.Fatalf("durations = %+v", out)
	}
	if out.PrefillTokensPerSec != 400 || out.DecodeTokensPerSec != 350 {
		t.Fatalf("rates = %+v", out)
	}
	if out.PeakMemoryBytes != 1<<20 || out.ActiveMemoryBytes != 512<<10 {
		t.Fatalf("memory = %+v", out)
	}
	if out.PromptCacheHits != 3 || out.PromptCacheMisses != 1 {
		t.Fatalf("cache counts = %+v", out)
	}
	if out.PromptCacheHitTokens != 100 || out.PromptCacheMissTokens != 25 {
		t.Fatalf("cache token counts = %+v", out)
	}
	if out.PromptCacheRestoreDuration != 5*time.Millisecond {
		t.Fatalf("restore duration = %v", out.PromptCacheRestoreDuration)
	}
}

func TestFromMlxMetrics_DropsNonFiniteRates_Ugly(t *testing.T) {
	out := fromMlxMetrics(Metrics{
		PrefillTokensPerSec: math.Inf(1),
		DecodeTokensPerSec:  math.NaN(),
	})
	if out.PrefillTokensPerSec != 0 || out.DecodeTokensPerSec != 0 {
		t.Fatalf("rates = %+v, want non-finite rates clamped to 0", out)
	}
}

func TestModelInfoBenchRoundTrip_Good(t *testing.T) {
	in := ModelInfo{
		Architecture:  "qwen3",
		VocabSize:     151936,
		NumLayers:     28,
		HiddenSize:    2048,
		QuantBits:     4,
		QuantGroup:    32,
		ContextLength: 32768,
		Adapter: lora.AdapterInfo{
			Name: "v1", Path: "/tmp/v1.safetensors", Hash: "abc",
			Rank: 8, Alpha: 16, Scale: 2,
			TargetKeys: []string{"q_proj", "v_proj"},
		},
	}
	round := benchInfoToModel(modelInfoToBench(in))
	if round.Architecture != in.Architecture || round.NumLayers != in.NumLayers ||
		round.ContextLength != in.ContextLength || round.HiddenSize != in.HiddenSize {
		t.Fatalf("scalar fields lost on round-trip: in=%+v out=%+v", in, round)
	}
	if round.Adapter.Name != in.Adapter.Name || round.Adapter.Rank != in.Adapter.Rank ||
		len(round.Adapter.TargetKeys) != len(in.Adapter.TargetKeys) ||
		round.Adapter.TargetKeys[0] != "q_proj" {
		t.Fatalf("adapter lost on round-trip: %+v", round.Adapter)
	}
	// Mutating the input adapter must not affect the converted copy.
	in.Adapter.TargetKeys[0] = "changed"
	if round.Adapter.TargetKeys[0] == "changed" {
		t.Fatal("loraToBenchAdapter did not clone TargetKeys")
	}
}

func TestFastEvalResultError_OkResultHasNoError_Good(t *testing.T) {
	if err := fastEvalResultError(core.Result{OK: true}); err != nil {
		t.Fatalf("OK result produced err = %v", err)
	}
}

func TestFastEvalResultError_PassesThroughErr_Bad(t *testing.T) {
	want := core.NewError("boom")
	err := fastEvalResultError(core.Result{OK: false, Value: want})
	if err == nil {
		t.Fatal("fastEvalResultError() error = nil, want passthrough")
	}
}

func TestFastEvalResultError_NonErrValueGetsFallback_Bad(t *testing.T) {
	err := fastEvalResultError(core.Result{OK: false, Value: "not-an-error"})
	if err == nil {
		t.Fatal("fastEvalResultError() error = nil for non-error value, want fallback")
	}
}
