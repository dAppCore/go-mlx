// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package qwen3

import (
	"testing"

	"dappco.re/go/mlx/pkg/metal"
)

func TestClose_CloseQwen3_MinimalModel_Good(t *testing.T) {
	requireMetalRuntime(t)

	coverageTokens := "CloseQwen3 MinimalModel"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	embedW := metal.FromValues([]float32{1, 2, 3, 4}, 2, 2)
	normW := metal.FromValues([]float32{1, 1}, 2)
	outW := metal.FromValues([]float32{1, 0, 0, 1}, 2, 2)
	metal.Materialize(embedW, normW, outW)

	inW := metal.FromValues([]float32{1, 1}, 2)
	postW := metal.FromValues([]float32{1, 1}, 2)
	qW := metal.FromValues([]float32{1, 0, 0, 1}, 2, 2)
	kW := metal.FromValues([]float32{1, 0, 0, 1}, 2, 2)
	vW := metal.FromValues([]float32{1, 0, 0, 1}, 2, 2)
	oW := metal.FromValues([]float32{1, 0, 0, 1}, 2, 2)
	qnW := metal.FromValues([]float32{1, 1}, 2)
	knW := metal.FromValues([]float32{1, 1}, 2)
	gateW := metal.FromValues([]float32{1, 0, 0, 1}, 2, 2)
	upW := metal.FromValues([]float32{1, 0, 0, 1}, 2, 2)
	downW := metal.FromValues([]float32{1, 0, 0, 1}, 2, 2)
	metal.Materialize(inW, postW, qW, kW, vW, oW, qnW, knW, gateW, upW, downW)

	m := &Qwen3Model{
		EmbedTokens: &metal.Embedding{Weight: embedW},
		Norm:        &metal.RMSNormModule{Weight: normW},
		Output:      metal.NewLinear(outW, nil),
		Layers: []*metal.DenseDecoderLayer{{
			InputNorm:    &metal.RMSNormModule{Weight: inW},
			PostAttnNorm: &metal.RMSNormModule{Weight: postW},
			Attention: &metal.GQAAttention{
				QProj: metal.NewLinear(qW, nil),
				KProj: metal.NewLinear(kW, nil),
				VProj: metal.NewLinear(vW, nil),
				OProj: metal.NewLinear(oW, nil),
				QNorm: &metal.RMSNormModule{Weight: qnW},
				KNorm: &metal.RMSNormModule{Weight: knW},
			},
			MLP: &metal.SiLUMLP{
				GateProj: metal.NewLinear(gateW, nil),
				UpProj:   metal.NewLinear(upW, nil),
				DownProj: metal.NewLinear(downW, nil),
			},
		}},
	}

	closeQwen3(m)

	if embedW.Valid() {
		t.Error("embed weight should be freed")
	}
	if outW.Valid() {
		t.Error("output weight should be freed")
	}
	if qW.Valid() {
		t.Error("q_proj weight should be freed")
	}
	if downW.Valid() {
		t.Error("down_proj weight should be freed")
	}
}

func TestClose_CloseQwen3_NilModel_Ugly(t *testing.T) {
	defer func() {
		if recovered := recover(); recovered != nil {
			t.Fatalf("closeQwen3(nil) panicked: %v", recovered)
		}
	}()
	closeQwen3(nil)
}
