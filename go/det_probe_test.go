// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package mlx

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/internal/metaltest"
	"dappco.re/go/mlx/memory"
	"dappco.re/go/mlx/pkg/metal"
)

// Determinism probes for the bf16 activation stream. Greedy decode must be
// bit-deterministic run to run — the state system's byte-exact sleep/wake
// story depends on it. mlx-lm on the same snapshot is hash-identical across
// runs, so any fork here is ours. 256 tokens keeps the probe inside the
// sliding window (pre-cap only), excluding the post-cap unit from the
// suspect set; the known fork reproduces by ~token 20.
//
// Trace caveat: compiled-layer trace keys do not carry gate state, so each
// gate configuration must run in a FRESH process — invoke one test per
// `go test -run` call, never both in one binary run.

func decodeDeterminismProbe(t *testing.T, pairs int) {
	t.Helper()
	if !metaltest.RunModelEvalTests {
		t.Skip("model-eval test")
	}
	dir := metaltest.HFModelPath(t, "mlx-community/gemma-4-e2b-it-4bit")
	m, err := LoadModel(dir, WithKVCacheMode(memory.KVCacheModePaged), WithContextLength(4096))
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	defer m.Close()
	ctx := context.Background()
	run := func() string {
		sess, err := m.NewSession()
		if err != nil {
			t.Fatalf("NewSession: %v", err)
		}
		defer sess.Close()
		if err := sess.Prefill("Write a long, detailed story about a clockmaker who repairs time itself."); err != nil {
			t.Fatalf("Prefill: %v", err)
		}
		text := core.NewBuilder()
		for tok := range sess.GenerateStream(ctx, WithMaxTokens(256), WithTemperature(0)) {
			text.WriteString(tok.Text)
		}
		if err := sess.Err(); err != nil {
			t.Fatalf("generate: %v", err)
		}
		return text.String()
	}
	reference := run()
	for pair := 1; pair <= pairs; pair++ {
		got := run()
		if got != reference {
			i := 0
			for i < len(reference) && i < len(got) && reference[i] == got[i] {
				i++
			}
			t.Fatalf("non-deterministic at pair %d, first byte diff at %d:\n  a %q\n  b %q",
				pair, i, reference[max(0, i-40):min(len(reference), i+40)], got[max(0, i-40):min(len(got), i+40)])
		}
	}
	t.Logf("deterministic across %d repeat runs", pairs)
}

// TestDecodeDeterminism_LiveModel — everything on (the shipping config).
//
//	go test -tags model_eval -run 'TestDecodeDeterminism_LiveModel$' -count=1 dappco.re/go/mlx
func TestDecodeDeterminism_LiveModel(t *testing.T) {
	decodeDeterminismProbe(t, 4)
}

// TestDecodeDeterminism_GemmMLP_LiveModel — the custom fused MLP kernels off
// (gemm via MLX quantized_matmul, the ops mlx-lm itself runs). If this is
// deterministic while the default probe forks, the fused MLP kernels are the
// culprit. MUST run in its own process (trace keys do not carry gate state).
//
//	go test -tags model_eval -run TestDecodeDeterminism_GemmMLP_LiveModel -count=1 dappco.re/go/mlx
func TestDecodeDeterminism_GemmMLP_LiveModel(t *testing.T) {
	restore := metal.SetRuntimeGate(metal.GateNativeMLPMatVec, false)
	defer restore()
	decodeDeterminismProbe(t, 4)
}

// TestDecodeDeterminism_SerialCompiled_LiveModel — one-ahead pipeline off,
// compiled layers on. Splits loop structure from layer math.
//
//	go test -tags model_eval -run TestDecodeDeterminism_SerialCompiled_LiveModel -count=1 dappco.re/go/mlx
func TestDecodeDeterminism_SerialCompiled_LiveModel(t *testing.T) {
	restore := metal.SetRuntimeGate(metal.GatePipelinedDecode, false)
	defer restore()
	decodeDeterminismProbe(t, 4)
}

// TestDecodeDeterminism_Uncompiled_LiveModel — pipeline AND compiled layers
// off: the plain serial loop over the uncompiled paths.
//
//	go test -tags model_eval -run TestDecodeDeterminism_Uncompiled_LiveModel -count=1 dappco.re/go/mlx
func TestDecodeDeterminism_Uncompiled_LiveModel(t *testing.T) {
	restorePipe := metal.SetRuntimeGate(metal.GatePipelinedDecode, false)
	defer restorePipe()
	restoreCompiled := metal.SetRuntimeGate(metal.GateCompiledLayerDecode, false)
	defer restoreCompiled()
	decodeDeterminismProbe(t, 4)
}
