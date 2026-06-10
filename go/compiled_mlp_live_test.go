// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package mlx

import (
	"context"
	"testing"
	"time"

	core "dappco.re/go"
	"dappco.re/go/mlx/internal/metaltest"
	"dappco.re/go/mlx/pkg/metal"
)

// TestCompiledMLPDecode_LiveModel proves the compiled decode MLP on a real
// model: byte-exact greedy output against the uncompiled gemm path (the same
// math, op by op), with decode rates logged against the default fused path.
//
//	go test -tags model_eval -run TestCompiledMLPDecode_LiveModel -count=1 dappco.re/go/mlx
func TestCompiledMLPDecode_LiveModel(t *testing.T) {
	if !metaltest.RunModelEvalTests {
		t.Skip("model-eval test; build with -tags model_eval and cache mlx-community/gemma-4-e2b-it-4bit")
	}
	dir := metaltest.HFModelPath(t, "mlx-community/gemma-4-e2b-it-4bit")
	m, err := LoadModel(dir)
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	defer m.Close()

	const prompt = "Write a long, detailed story about a clockmaker who repairs time itself."
	ctx := context.Background()

	gen := func(label string) (string, float64) {
		t.Helper()
		sess, err := m.NewSession()
		if err != nil {
			t.Fatalf("%s: NewSession: %v", label, err)
		}
		defer sess.Close()
		if err := sess.Prefill(prompt); err != nil {
			t.Fatalf("%s: Prefill: %v", label, err)
		}
		text := core.NewBuilder()
		tokens := 0
		start := time.Now()
		for tok := range sess.GenerateStream(ctx, WithMaxTokens(200), WithTemperature(0)) {
			text.WriteString(tok.Text)
			tokens++
		}
		rate := float64(tokens) / time.Since(start).Seconds()
		if err := sess.Err(); err != nil {
			t.Fatalf("%s: generate: %v", label, err)
		}
		t.Logf("%s: %.1f tok/s (%d tok)", label, rate, tokens)
		return text.String(), rate
	}

	// Default path (fused native matvec) — the perf baseline.
	_, defaultRate := gen("default (fused matvec)")

	// Uncompiled gemm path — the exactness baseline: the same ops the
	// compiled closure traces, run op by op.
	restoreFused := metal.SetRuntimeGate(metal.GateNativeMLPMatVec, false)
	gemmText, gemmRate := gen("uncompiled gemm")

	// Compiled closure — must match the gemm path byte for byte.
	restoreCompiled := metal.SetRuntimeGate(metal.GateCompiledMLPDecode, true)
	compiledText, compiledRate := gen("compiled MLP")
	restoreCompiled()
	restoreFused()

	if compiledText != gemmText {
		t.Errorf("compiled MLP diverged from the uncompiled gemm path:\n  gemm     %q\n  compiled %q", gemmText, compiledText)
	}
	t.Logf("rates: default %.1f · gemm %.1f · compiled %.1f tok/s", defaultRate, gemmRate, compiledRate)
}
