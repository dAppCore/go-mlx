// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package mlx

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/mlx/internal/metaltest"
)

func TestNativeSpeculativeTextModel_LiveE2BAssistantPair(t *testing.T) {
	if !metaltest.RunMetalTests {
		t.Skip("build with -tags metal_runtime to drive the live native speculative pair")
	}
	if !MetalAvailable() {
		t.Skip("Metal runtime unavailable")
	}
	targetDir := metaltest.HFModelPath(t, "mlx-community/gemma-4-e2b-it-4bit")
	assistantDir := metaltest.HFModelPath(t, "mlx-community/gemma-4-E2B-it-assistant-bf16")

	ClearCache()
	GC()

	tm, err := LoadNativeSpeculativePairAsTextModelBlock(targetDir, assistantDir, 2, WithContextLength(512))
	if err != nil {
		if core.Contains(err.Error(), reshapeFlakeNeedle) {
			t.Skipf("native speculative pair load did not survive the reshape flake: %v", err)
		}
		t.Fatalf("LoadNativeSpeculativePairAsTextModelBlock: %v", err)
	}
	defer func() {
		_ = tm.Close()
		ClearCache()
		GC()
	}()
	spec, ok := tm.(*nativeSpeculativeTextModel)
	if !ok {
		t.Fatalf("LoadNativeSpeculativePairAsTextModelBlock returned %T, want *nativeSpeculativeTextModel", tm)
	}
	if spec.nativeAssistant == nil {
		t.Fatal("native speculative text model did not attach the native Gemma 4 assistant pair")
	}

	var out []inference.Token
	for token := range spec.Generate(context.Background(), "The capital of France is", inference.WithMaxTokens(2), inference.WithTemperature(0)) {
		out = append(out, token)
	}
	if err := resultError(spec.Err()); err != nil {
		t.Fatalf("Generate Err: %v", err)
	}
	if len(out) == 0 {
		t.Fatal("native speculative Generate produced no tokens")
	}
	metrics := spec.MTPMetrics()
	if metrics == nil || metrics.ProposedTokens == 0 || metrics.TargetVerifyCalls == 0 {
		t.Fatalf("native speculative MTPMetrics = %+v, want assistant counters", metrics)
	}
}
