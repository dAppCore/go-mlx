// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package mlx

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/mlx/internal/metaltest"
)

// requireLiveNativeTextModel loads the cached e2b checkpoint through the no-cgo
// native contract stack (LoadNativeTextModel) — a SEPARATE load path from the
// shared cgo metal *Model, so it carries its own load/close/GC. Skips (never
// fails) when Metal is off, the weights are uncached, or the load hits the
// reshape flake.
func requireLiveNativeTextModel(t *testing.T) (*nativeTextModel, func()) {
	t.Helper()
	if !metaltest.RunMetalTests {
		t.Skip("build with -tags metal_runtime to drive the live native text model")
	}
	if !MetalAvailable() {
		t.Skip("Metal runtime unavailable")
	}
	dir := metaltest.HFModelPath(t, liveE2BModelRepo)

	ClearCache()
	GC()

	var (
		tm  inference.TextModel
		err error
	)
	for attempt := 0; attempt < 3; attempt++ {
		tm, err = LoadNativeTextModel(dir, WithContextLength(2048))
		if err == nil {
			break
		}
		if !core.Contains(err.Error(), reshapeFlakeNeedle) {
			t.Fatalf("LoadNativeTextModel(%s): %v", dir, err)
		}
		ClearCache()
	}
	if err != nil {
		t.Skipf("native text model load did not survive the reshape flake: %v", err)
	}
	native, ok := tm.(*nativeTextModel)
	if !ok {
		_ = tm.Close()
		t.Fatalf("LoadNativeTextModel returned %T, want *nativeTextModel", tm)
	}
	return native, func() {
		_ = native.Close()
		ClearCache()
		GC()
	}
}

// TestNativeTextModel_GenerateChatStreamLiveModel drives the no-cgo native
// token-loop surface — Generate, Chat, and the shared stream body (greedy and
// the temperature-sampled arm), plus the early-break yield path and Metrics off
// a real run. These three methods are 0% without driving the contract forward.
func TestNativeTextModel_GenerateChatStreamLiveModel(t *testing.T) {
	native, done := requireLiveNativeTextModel(t)
	defer done()
	ctx := context.Background()

	// Generate (greedy: Temperature unset => model.Generate lane).
	gen := core.NewBuilder()
	for token := range native.Generate(ctx, "The capital of France is") {
		gen.WriteString(token.Text)
	}
	if err := resultError(native.Err()); err != nil {
		t.Fatalf("Generate Err: %v", err)
	}
	t.Logf("native Generate -> %q", gen.String())

	// Metrics off the greedy run — the contract records prompt/gen counters.
	metrics := native.Metrics()
	if metrics.GeneratedTokens == 0 {
		t.Errorf("native Metrics show no generated tokens: %+v", metrics)
	}

	// Chat (greedy) through the gemma turn template.
	reply := core.NewBuilder()
	for token := range native.Chat(ctx,
		[]inference.Message{{Role: "user", Content: "Say the word ready."}}) {
		reply.WriteString(token.Text)
	}
	if err := resultError(native.Err()); err != nil {
		t.Fatalf("Chat Err: %v", err)
	}
	t.Logf("native Chat -> %q", reply.String())

	// Temperature-sampled stream arm (model.GenerateSampled lane).
	sampled := core.NewBuilder()
	for token := range native.Generate(ctx, "A short story:", inference.WithTemperature(0.7), inference.WithMaxTokens(12)) {
		sampled.WriteString(token.Text)
	}
	if err := resultError(native.Err()); err != nil {
		t.Fatalf("sampled Generate Err: %v", err)
	}
	t.Logf("native sampled Generate -> %q", sampled.String())

	// Early-break arm of the stream yield loop.
	seen := 0
	for range native.Generate(ctx, "Numbers: one two three four five", inference.WithMaxTokens(16)) {
		seen++
		if seen == 1 {
			break
		}
	}
	if seen != 1 {
		t.Fatalf("early-break native stream saw %d tokens, want 1", seen)
	}
}

// TestNativeTextModel_EvaluateLiveModel drives the no-cgo native eval surface
// through the same E2B live model as the native generate smoke.
func TestNativeTextModel_EvaluateLiveModel(t *testing.T) {
	native, done := requireLiveNativeTextModel(t)
	defer done()

	report, err := native.Evaluate(context.Background(), newLiveDatasetStream(), inference.EvalConfig{
		MaxSamples: 1,
		BatchSize:  1,
		MaxSeqLen:  64,
	})
	if err != nil {
		t.Fatalf("Evaluate: %v", err)
	}
	if report == nil {
		t.Fatal("Evaluate returned nil report")
	}
	if report.Metrics.Tokens == 0 {
		t.Errorf("Evaluate report has no token coverage: %+v", report.Metrics)
	}
	t.Logf("native Evaluate: samples=%d tokens=%d loss=%.4f ppl=%.4f",
		report.Metrics.Samples, report.Metrics.Tokens, report.Metrics.Loss, report.Metrics.Perplexity)
}
