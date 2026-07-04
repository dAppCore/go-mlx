// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package mixtral

import (
	"testing"

	"dappco.re/go/mlx/pkg/metal"
)

// --- closeMixtral (close.go) ---

// TestClose_CloseModel_Good releases a loaded model and asserts the layer slice
// is cleared. A second CloseModel and a nil-receiver CloseModel must not panic
// (idempotent teardown). This test owns its own model so it can close early
// rather than leaning on the shared t.Cleanup.
func TestClose_CloseModel_Good(t *testing.T) {
	requireMetalRuntime(t)
	dir := t.TempDir()
	writeMixtralModel(t, dir)
	model, err := LoadMixtral(dir)
	if err != nil {
		t.Fatalf("LoadMixtral: %v", err)
	}
	if len(model.Layers) == 0 {
		t.Fatal("loaded model has no layers")
	}
	model.CloseModel()
	if model.Layers != nil {
		t.Fatalf("after CloseModel Layers = %v, want nil", model.Layers)
	}
	model.CloseModel()                // idempotent: second close is a no-op
	(*MixtralModel)(nil).CloseModel() // nil receiver must not panic
}

// TestClose_CloseModel_MinimalModel_Good closes a hand-built model with a tied
// output (Output shares EmbedTokens.Weight) and asserts the freed weights become
// invalid. Closing a tied Output must NOT double-free the shared embed weight —
// the close.go guard skips FreeLinear when Output.Weight == EmbedTokens.Weight.
func TestClose_CloseModel_MinimalModel_Good(t *testing.T) {
	requireMetalRuntime(t)

	embedW := metal.FromValues([]float32{1, 2, 3, 4}, 2, 2)
	normW := metal.FromValues([]float32{1, 1}, 2)
	inW := metal.FromValues([]float32{1, 1}, 2)
	postW := metal.FromValues([]float32{1, 1}, 2)
	qW := metal.FromValues([]float32{1, 0, 0, 1}, 2, 2)
	kW := metal.FromValues([]float32{1, 0, 0, 1}, 2, 2)
	vW := metal.FromValues([]float32{1, 0, 0, 1}, 2, 2)
	oW := metal.FromValues([]float32{1, 0, 0, 1}, 2, 2)
	gateW := metal.FromValues([]float32{1, 0, 0, 1}, 2, 2)
	upW := metal.FromValues([]float32{1, 0, 0, 1}, 2, 2)
	downW := metal.FromValues([]float32{1, 0, 0, 1}, 2, 2)
	metal.Materialize(embedW, normW, inW, postW, qW, kW, vW, oW, gateW, upW, downW)

	embed := &metal.Embedding{Weight: embedW}
	m := &MixtralModel{
		EmbedTokens: embed,
		Norm:        &metal.RMSNormModule{Weight: normW},
		Output:      embed.AsLinear(), // tied: Output.Weight == embedW
		Layers: []*MixtralDecoderLayer{{
			Dense: &metal.DenseDecoderLayer{
				InputNorm:    &metal.RMSNormModule{Weight: inW},
				PostAttnNorm: &metal.RMSNormModule{Weight: postW},
				Attention: &metal.GQAAttention{
					QProj: metal.NewLinear(qW, nil),
					KProj: metal.NewLinear(kW, nil),
					VProj: metal.NewLinear(vW, nil),
					OProj: metal.NewLinear(oW, nil),
				},
				MLP: &metal.SiLUMLP{
					GateProj: metal.NewLinear(gateW, nil),
					UpProj:   metal.NewLinear(upW, nil),
					DownProj: metal.NewLinear(downW, nil),
				},
			},
		}},
	}

	closeMixtral(m)

	if embedW.Valid() {
		t.Error("embed weight should be freed")
	}
	if qW.Valid() {
		t.Error("q_proj weight should be freed")
	}
	if downW.Valid() {
		t.Error("down_proj weight should be freed")
	}
	if m.Layers != nil {
		t.Errorf("Layers = %v after close, want nil", m.Layers)
	}
}

// TestClose_CloseModel_NilLayer_Ugly drives the nil-layer continue in
// closeMixtral: a Layers slice carrying a nil entry (and a layer with a nil
// Dense) must be skipped without a panic, while the surrounding model weights
// are still freed.
func TestClose_CloseModel_NilLayer_Ugly(t *testing.T) {
	requireMetalRuntime(t)

	embedW := metal.FromValues([]float32{1, 2, 3, 4}, 2, 2)
	normW := metal.FromValues([]float32{1, 1}, 2)
	metal.Materialize(embedW, normW)

	m := &MixtralModel{
		EmbedTokens: &metal.Embedding{Weight: embedW},
		Norm:        &metal.RMSNormModule{Weight: normW},
		Layers: []*MixtralDecoderLayer{
			nil,          // nil layer → continue
			{Dense: nil}, // nil Dense → continue
		},
	}

	closeMixtral(m) // must not panic on the nil entries

	if embedW.Valid() {
		t.Error("embed weight should be freed even with nil layers present")
	}
	if m.Layers != nil {
		t.Errorf("Layers = %v after close, want nil", m.Layers)
	}
}

// TestClose_CloseModel_NilReceiver_Ugly asserts a nil-receiver close is a no-op
// rather than a nil dereference.
func TestClose_CloseModel_NilReceiver_Ugly(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("closeMixtral(nil) panicked: %v", r)
		}
	}()
	closeMixtral(nil)
	(*MixtralModel)(nil).CloseModel()
}
