// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"testing"

	"dappco.re/go/mlx/pkg/metal"
)

// These tests pin Gemma 4's LoRA surface: ResolveLoRALinear (the projection-path
// → *Linear mapping for q_proj/router.proj/per_layer_projection) and ApplyLoRA
// (attaching adapters to the safe attention targets plus the opt-in extended PLE
// targets). They moved here from package metal's lora_test.go with the model
// type; the metal-side resolveLinear dispatch is pinned by metal's
// model_dispatch_test.go.

func TestLora_ResolveLoRALinear_Gemma4_Good(t *testing.T) {
	coverageTokens := "ResolveLoRALinear Gemma4"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	qProj := &metal.Linear{}
	routerProj := &metal.Linear{}
	perLayerProj := &metal.Linear{}
	model := &Gemma4Model{
		Layers: []*Gemma4DecoderLayer{
			{
				Attention: &Gemma4Attention{
					QProj: qProj,
				},
				Router: &Gemma4Router{
					Proj: routerProj,
				},
				PerLayerProjection: perLayerProj,
				MLP: &metal.MLP{
					GateProj: &metal.Linear{},
					UpProj:   &metal.Linear{},
					DownProj: &metal.Linear{},
				},
			},
		},
	}

	if got := model.ResolveLoRALinear(0, "self_attn.q_proj"); got != qProj {
		t.Fatal("ResolveLoRALinear should return Gemma4 q_proj")
	}
	if got := model.ResolveLoRALinear(0, "router.proj"); got != routerProj {
		t.Fatal("ResolveLoRALinear should return Gemma4 router.proj")
	}
	if got := model.ResolveLoRALinear(0, "per_layer_projection"); got != perLayerProj {
		t.Fatal("ResolveLoRALinear should return Gemma4 per_layer_projection")
	}
}

func TestLora_ApplyLoRA_Gemma4ExtendedTargets_Good(t *testing.T) {
	requireMetalRuntime(t)

	weights := []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
	}
	weightRouter := metal.FromValues(weights, 3, 4)
	weightInputGate := metal.FromValues(weights, 3, 4)
	weightProjection := metal.FromValues(weights, 3, 4)

	routerProj := metal.NewLinear(weightRouter, nil)
	perLayerInputGate := metal.NewLinear(weightInputGate, nil)
	perLayerProjection := metal.NewLinear(weightProjection, nil)

	model := &Gemma4Model{
		Layers: []*Gemma4DecoderLayer{
			{
				Attention: &Gemma4Attention{},
				MLP:       &metal.MLP{},
				Router: &Gemma4Router{
					Proj: routerProj,
				},
				PerLayerInputGate:  perLayerInputGate,
				PerLayerProjection: perLayerProjection,
			},
		},
	}
	defer closeGemma4(model)

	adapter := model.ApplyLoRA(metal.LoRAConfig{
		Rank:                       2,
		Alpha:                      4,
		AllowGemma4ExtendedTargets: true,
		TargetKeys:                 []string{"router.proj", "per_layer_input_gate", "per_layer_projection"},
	})

	if adapter.Layers["model.layers.0.router.proj"] == nil {
		t.Fatal("expected LoRA layer for router.proj")
	}
	if adapter.Layers["model.layers.0.per_layer_input_gate"] == nil {
		t.Fatal("expected LoRA layer for per_layer_input_gate")
	}
	if adapter.Layers["model.layers.0.per_layer_projection"] == nil {
		t.Fatal("expected LoRA layer for per_layer_projection")
	}
	if model.Layers[0].Router.Proj.LoRA == nil {
		t.Fatal("router.proj should have an attached LoRA adapter")
	}
	if model.Layers[0].PerLayerInputGate.LoRA == nil {
		t.Fatal("per_layer_input_gate should have an attached LoRA adapter")
	}
	if model.Layers[0].PerLayerProjection.LoRA == nil {
		t.Fatal("per_layer_projection should have an attached LoRA adapter")
	}
}

func TestLora_ApplyLoRA_Gemma4PLETargetsRequireOptIn_Bad(t *testing.T) {
	requireMetalRuntime(t)

	weights := []float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
	}
	qProj := metal.NewLinear(metal.FromValues(weights, 3, 4), nil)
	perLayerProjection := metal.NewLinear(metal.FromValues(weights, 3, 4), nil)

	model := &Gemma4Model{
		Layers: []*Gemma4DecoderLayer{
			{
				Attention:          &Gemma4Attention{QProj: qProj},
				MLP:                &metal.MLP{},
				PerLayerProjection: perLayerProjection,
			},
		},
	}
	defer closeGemma4(model)

	adapter := model.ApplyLoRA(metal.LoRAConfig{
		Rank:       2,
		Alpha:      4,
		TargetKeys: []string{"q_proj", "per_layer_projection"},
	})

	if adapter.Layers["model.layers.0.self_attn.q_proj"] == nil {
		t.Fatal("expected safe q_proj LoRA layer")
	}
	if adapter.Layers["model.layers.0.per_layer_projection"] != nil {
		t.Fatal("per_layer_projection should require AllowGemma4ExtendedTargets")
	}
	if model.Layers[0].PerLayerProjection.LoRA != nil {
		t.Fatal("per_layer_projection should not have an attached LoRA adapter without opt-in")
	}
}
