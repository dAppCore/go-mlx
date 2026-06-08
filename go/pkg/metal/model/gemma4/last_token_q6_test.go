// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"testing"

	"dappco.re/go/mlx/internal/metaltest"
	"dappco.re/go/mlx/pkg/metal"
)

// TestGemma4Decode_FusedQ6LastTokenMatchesGeneric_Good proves the fused
// quantized last-token greedy kernel (one compiled graph: RMSNorm +
// mx::quantized_matmul affine 6-bit + argmax) picks the SAME token as the
// generic RMSNorm + Output.Forward + Argmax path on the real e2b q6 output
// projection.
//
// The e2b output weights are bitstream-packed 6-bit (6 does not divide 32, so
// values straddle uint32 words — the layout the custom Q6Group64 matvec
// hand-unpacks). q4 and q8 divide 32 evenly, so the fused last-token output was
// only wired for them; this is the gate that mx's affine q6 matmul reads the
// same packing, making it correctness-safe to put q6 on the fused fast-path too.
func TestGemma4Decode_FusedQ6LastTokenMatchesGeneric_Good(t *testing.T) {
	if !metaltest.RunMetalTests {
		t.Skip("build with -tags metal_runtime to run the e2b q6 fused last-token parity check")
	}
	modelPath := metaltest.HFModelPath(t, "mlx-community/gemma-4-e2b-it-6bit")
	m, err := LoadGemma4(modelPath)
	if err != nil {
		t.Fatalf("LoadGemma4(%s): %v", modelPath, err)
	}
	defer m.CloseModel()

	if m.Output == nil || m.Output.Scales == nil || m.Output.Bits != 6 || m.Output.GroupSize != 64 {
		bits, gs, quant := 0, 0, false
		if m.Output != nil {
			bits, gs, quant = m.Output.Bits, m.Output.GroupSize, m.Output.Scales != nil
		}
		t.Fatalf("e2b output projection is not q6/group64: bits=%d groupSize=%d quantized=%v", bits, gs, quant)
	}

	caches := m.NewCache()
	defer metal.FreeCaches(caches)
	prefill := metal.FromValues([]int32{2, 1000, 2000, 3000, 4000}, 5)
	prefillInput := metal.Reshape(prefill, 1, 5)
	prefillLogits, hidden := m.ForwardLastTokenLogitsAndHidden(prefillInput, nil, caches)
	metal.Free(prefill, prefillInput, prefillLogits)
	defer metal.Free(hidden)
	if err := metal.Eval(hidden); err != nil {
		t.Fatalf("target prefill: %v", err)
	}

	fused, ok, err := metal.NativeLastTokenGreedyTokenWithArray(hidden, m.NormScaled, m.Output, m.Cfg.RMSNormEps, nil)
	if err != nil {
		t.Fatalf("fused q6 last-token: %v", err)
	}
	if !ok {
		t.Fatal("fused q6 last-token unavailable; want available after q6 wiring")
	}
	defer metal.Free(fused)

	normed := metal.RMSNorm(hidden, m.NormScaled, m.Cfg.RMSNormEps)
	logits := m.Output.Forward(normed)
	want := metal.Argmax(logits, -1, false)
	metal.Free(normed, logits)
	defer metal.Free(want)

	if err := metal.Eval(fused, want); err != nil {
		t.Fatalf("eval tokens: %v", err)
	}
	if got, wantID := fused.Int(), want.Int(); got != wantID {
		t.Fatalf("fused q6 token = %d, generic = %d — mx affine q6 disagrees with the model's bitstream packing", got, wantID)
	}
}
