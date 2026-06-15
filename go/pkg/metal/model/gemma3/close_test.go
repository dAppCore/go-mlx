// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma3

import (
	"testing"

	"dappco.re/go/mlx/internal/metaltest"
	"dappco.re/go/mlx/pkg/metal"
)

func requireMetalRuntime(t testing.TB) {
	t.Helper()
	if !metaltest.RunMetalTests {
		t.Skip("build with -tags metal_runtime to enable Metal runtime tests")
	}
	if !metal.MetalAvailable() {
		t.Skip("Metal runtime unavailable")
	}
}

func TestClose_CloseGemma_MinimalModel_Good(t *testing.T) {
	requireMetalRuntime(t)

	// Build a minimal GemmaModel with one layer to test cleanup.
	embedW := metal.FromValues([]float32{1, 2, 3, 4}, 2, 2)
	normW := metal.FromValues([]float32{1, 1}, 2)
	normScaled := metal.FromValues([]float32{2, 2}, 2)
	metal.Materialize(embedW, normW, normScaled)

	// Layer components
	inW := metal.FromValues([]float32{1, 1}, 2)
	qW := metal.FromValues([]float32{1, 0, 0, 1}, 2, 2)
	kW := metal.FromValues([]float32{1, 0, 0, 1}, 2, 2)
	vW := metal.FromValues([]float32{1, 0, 0, 1}, 2, 2)
	oW := metal.FromValues([]float32{1, 0, 0, 1}, 2, 2)
	qnW := metal.FromValues([]float32{1, 1}, 2)
	knW := metal.FromValues([]float32{1, 1}, 2)
	gateW := metal.FromValues([]float32{1, 0, 0, 1}, 2, 2)
	upW := metal.FromValues([]float32{1, 0, 0, 1}, 2, 2)
	downW := metal.FromValues([]float32{1, 0, 0, 1}, 2, 2)
	metal.Materialize(inW, qW, kW, vW, oW, qnW, knW, gateW, upW, downW)

	m := &GemmaModel{
		EmbedTokens: &metal.Embedding{Weight: embedW},
		Norm:        &metal.RMSNormModule{Weight: normW},
		NormScaled:  normScaled,
		Output:      nil, // Tied to embed — skip
		Layers: []*DecoderLayer{{
			InputNorm: &metal.RMSNormModule{Weight: inW},
			Attention: &Attention{
				QProj: metal.NewLinear(qW, nil),
				KProj: metal.NewLinear(kW, nil),
				VProj: metal.NewLinear(vW, nil),
				OProj: metal.NewLinear(oW, nil),
				QNorm: &metal.RMSNormModule{Weight: qnW},
				KNorm: &metal.RMSNormModule{Weight: knW},
			},
			MLP: &metal.MLP{
				GateProj: metal.NewLinear(gateW, nil),
				UpProj:   metal.NewLinear(upW, nil),
				DownProj: metal.NewLinear(downW, nil),
			},
		}},
	}

	closeGemma(m)

	// Verify key arrays freed
	if embedW.Valid() {
		t.Error("embed weight should be freed")
	}
	if normW.Valid() {
		t.Error("norm weight should be freed")
	}
	if qW.Valid() {
		t.Error("q_proj weight should be freed")
	}
	if gateW.Valid() {
		t.Error("gate_proj weight should be freed")
	}
}

// TestClose_CloseGemma_NilModel_Ugly guards Mantis #1829: a Metal library load
// failure aborts model construction before any field is populated, so the
// deferred cleanup must return cleanly rather than panic on a nil model.
func TestClose_CloseGemma_NilModel_Ugly(t *testing.T) {
	defer func() {
		if recovered := recover(); recovered != nil {
			t.Fatalf("closeGemma(nil) panicked: %v", recovered)
		}
	}()
	closeGemma(nil)
}

// TestClose_GemmaModel_CloseModel_UntiedOutput_Good drives the exported
// CloseModel entry point (the InternalModel close hook) on a synthetic model
// whose Output projection has its OWN weight — distinct from EmbedTokens — so
// the "free Output only when untied" branch in closeGemma actually frees it.
// The minimal-model test above uses tied (nil) output and so never reached it.
func TestClose_GemmaModel_CloseModel_UntiedOutput_Good(t *testing.T) {
	requireMetalRuntime(t)

	embedW := metal.FromValues([]float32{1, 2, 3, 4}, 2, 2)
	normW := metal.FromValues([]float32{1, 1}, 2)
	normScaled := metal.FromValues([]float32{2, 2}, 2)
	outW := metal.FromValues([]float32{5, 6, 7, 8}, 2, 2) // separate lm_head weight
	metal.Materialize(embedW, normW, normScaled, outW)

	m := &GemmaModel{
		EmbedTokens: &metal.Embedding{Weight: embedW},
		Norm:        &metal.RMSNormModule{Weight: normW},
		NormScaled:  normScaled,
		Output:      metal.NewLinear(outW, nil), // untied — own weight
	}

	m.CloseModel()

	if embedW.Valid() {
		t.Error("embed weight should be freed")
	}
	if outW.Valid() {
		t.Error("untied output weight should be freed")
	}
}

// TestClose_GemmaModel_CloseModel_NilModel_Ugly drives the exported entry
// point on a nil receiver — CloseModel must delegate to closeGemma's nil
// guard rather than panic.
func TestClose_GemmaModel_CloseModel_NilModel_Ugly(t *testing.T) {
	defer func() {
		if recovered := recover(); recovered != nil {
			t.Fatalf("(*GemmaModel)(nil).CloseModel() panicked: %v", recovered)
		}
	}()
	var m *GemmaModel
	m.CloseModel()
}
