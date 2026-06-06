// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"testing"

	"dappco.re/go/mlx/pkg/metal"
)

// These tests pin Gemma 4's LoRA surface: ResolveLoRALinear (the projection-path
// to *Linear mapping for attention/MLP/router/PLE targets) and ApplyLoRA
// (attaching adapters through the same names that adapter load/save metadata
// uses). The metal-side resolveLinear dispatch is pinned by metal's
// model_dispatch_test.go.

func TestLora_ResolveLoRALinear_Gemma4_Good(t *testing.T) {
	qProj := &metal.Linear{}
	kProj := &metal.Linear{}
	vProj := &metal.Linear{}
	oProj := &metal.Linear{}
	gateProj := &metal.Linear{}
	upProj := &metal.Linear{}
	downProj := &metal.Linear{}
	routerProj := &metal.Linear{}
	perLayerInputGate := &metal.Linear{}
	perLayerProj := &metal.Linear{}
	model := &Gemma4Model{
		Layers: []*Gemma4DecoderLayer{
			{
				Attention: &Gemma4Attention{
					QProj: qProj,
					KProj: kProj,
					VProj: vProj,
					OProj: oProj,
				},
				Router: &Gemma4Router{
					Proj: routerProj,
				},
				PerLayerInputGate:  perLayerInputGate,
				PerLayerProjection: perLayerProj,
				MLP: &metal.MLP{
					GateProj: gateProj,
					UpProj:   upProj,
					DownProj: downProj,
				},
			},
		},
	}

	tests := []struct {
		path string
		want *metal.Linear
	}{
		{"self_attn.q_proj", qProj},
		{"self_attn.k_proj", kProj},
		{"self_attn.v_proj", vProj},
		{"self_attn.o_proj", oProj},
		{"mlp.gate_proj", gateProj},
		{"mlp.up_proj", upProj},
		{"mlp.down_proj", downProj},
		{"router.proj", routerProj},
		{"per_layer_input_gate", perLayerInputGate},
		{"per_layer_projection", perLayerProj},
	}
	for _, tt := range tests {
		if got := model.ResolveLoRALinear(0, tt.path); got != tt.want {
			t.Fatalf("ResolveLoRALinear(%q) = %p, want %p", tt.path, got, tt.want)
		}
	}
}

func TestLora_ResolveLoRALinear_Gemma4NilSafe_Bad(t *testing.T) {
	model := &Gemma4Model{
		Layers: []*Gemma4DecoderLayer{
			nil,
			{},
			{Attention: &Gemma4Attention{}},
			{MLP: &metal.MLP{}},
		},
	}

	tests := []struct {
		name     string
		model    *Gemma4Model
		layerIdx int
		path     string
	}{
		{"nil_model", nil, 0, "self_attn.q_proj"},
		{"negative_layer", model, -1, "self_attn.q_proj"},
		{"past_end_layer", model, len(model.Layers), "self_attn.q_proj"},
		{"nil_layer", model, 0, "self_attn.q_proj"},
		{"missing_attention", model, 1, "self_attn.q_proj"},
		{"missing_projection", model, 2, "self_attn.q_proj"},
		{"missing_mlp", model, 1, "mlp.gate_proj"},
		{"missing_mlp_projection", model, 3, "mlp.gate_proj"},
		{"missing_router", model, 1, "router.proj"},
		{"unknown_path", model, 2, "self_attn.nope"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				if recovered := recover(); recovered != nil {
					t.Fatalf("ResolveLoRALinear panicked for %s: %v", tt.name, recovered)
				}
			}()
			if got := tt.model.ResolveLoRALinear(tt.layerIdx, tt.path); got != nil {
				t.Fatalf("ResolveLoRALinear(%d, %q) = %p, want nil", tt.layerIdx, tt.path, got)
			}
		})
	}
}

func TestLora_ApplyLoRA_Gemma4FullPathAttentionTargets_Good(t *testing.T) {
	requireMetalRuntime(t)

	qProj := newLoraTestLinear()
	kProj := newLoraTestLinear()
	vProj := newLoraTestLinear()
	oProj := newLoraTestLinear()

	model := &Gemma4Model{
		Layers: []*Gemma4DecoderLayer{
			{
				Attention: &Gemma4Attention{
					QProj: qProj,
					KProj: kProj,
					VProj: vProj,
					OProj: oProj,
				},
				MLP: &metal.MLP{},
			},
		},
	}
	defer closeGemma4(model)

	adapter := model.ApplyLoRA(metal.LoRAConfig{
		Rank:  2,
		Alpha: 4,
		TargetKeys: []string{
			"self_attn.q_proj",
			"self_attn.k_proj",
			"self_attn.v_proj",
			"self_attn.o_proj",
		},
	})

	expected := map[string]*metal.Linear{
		"model.layers.0.self_attn.q_proj": qProj,
		"model.layers.0.self_attn.k_proj": kProj,
		"model.layers.0.self_attn.v_proj": vProj,
		"model.layers.0.self_attn.o_proj": oProj,
	}
	for name, proj := range expected {
		lora := adapter.Layers[name]
		if lora == nil {
			t.Fatalf("expected LoRA layer for %s; got keys %v", name, adapter.SortedNames())
		}
		if proj.LoRA != lora {
			t.Fatalf("%s projection LoRA = %p, want adapter layer %p", name, proj.LoRA, lora)
		}
		if lora.Base != proj {
			t.Fatalf("%s LoRA base = %p, want projection %p", name, lora.Base, proj)
		}
	}
}

func TestLora_ApplyLoRA_Gemma4MLPTargetAliases_Good(t *testing.T) {
	requireMetalRuntime(t)

	gateProj := newLoraTestLinear()
	upProj := newLoraTestLinear()
	downProj := newLoraTestLinear()

	model := &Gemma4Model{
		Layers: []*Gemma4DecoderLayer{
			{
				Attention: &Gemma4Attention{},
				MLP: &metal.MLP{
					GateProj: gateProj,
					UpProj:   upProj,
					DownProj: downProj,
				},
			},
		},
	}
	defer closeGemma4(model)

	adapter := model.ApplyLoRA(metal.LoRAConfig{
		Rank:       2,
		Alpha:      4,
		TargetKeys: []string{"gate_proj", "up_proj", "down_proj"},
	})

	expected := map[string]*metal.Linear{
		"model.layers.0.mlp.gate_proj": gateProj,
		"model.layers.0.mlp.up_proj":   upProj,
		"model.layers.0.mlp.down_proj": downProj,
	}
	for name, proj := range expected {
		lora := adapter.Layers[name]
		if lora == nil {
			t.Fatalf("expected LoRA layer for %s; got keys %v", name, adapter.SortedNames())
		}
		if proj.LoRA != lora {
			t.Fatalf("%s projection LoRA = %p, want adapter layer %p", name, proj.LoRA, lora)
		}
		if lora.Base != proj {
			t.Fatalf("%s LoRA base = %p, want projection %p", name, lora.Base, proj)
		}
	}
}

func TestLora_ApplyLoRA_Gemma4ExtendedTargets_Good(t *testing.T) {
	requireMetalRuntime(t)

	routerProj := newLoraTestLinear()
	perLayerInputGate := newLoraTestLinear()
	perLayerProjection := newLoraTestLinear()

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
		Rank:                 2,
		Alpha:                4,
		AllowExtendedTargets: true,
		TargetKeys:           []string{"router.proj", "per_layer_input_gate", "per_layer_projection"},
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

func TestLora_ApplyLoRA_Gemma4ExtendedTargetsRequireOptIn_Bad(t *testing.T) {
	requireMetalRuntime(t)

	qProj := newLoraTestLinear()
	routerProj := newLoraTestLinear()
	perLayerInputGate := newLoraTestLinear()
	perLayerProjection := newLoraTestLinear()

	model := &Gemma4Model{
		Layers: []*Gemma4DecoderLayer{
			{
				Attention:          &Gemma4Attention{QProj: qProj},
				MLP:                &metal.MLP{},
				Router:             &Gemma4Router{Proj: routerProj},
				PerLayerInputGate:  perLayerInputGate,
				PerLayerProjection: perLayerProjection,
			},
		},
	}
	defer closeGemma4(model)

	adapter := model.ApplyLoRA(metal.LoRAConfig{
		Rank:       2,
		Alpha:      4,
		TargetKeys: []string{"q_proj", "router.proj", "per_layer_input_gate", "per_layer_projection"},
	})

	if adapter.Layers["model.layers.0.self_attn.q_proj"] == nil {
		t.Fatal("expected safe q_proj LoRA layer")
	}
	for _, target := range []struct {
		name   string
		linear *metal.Linear
	}{
		{"router.proj", model.Layers[0].Router.Proj},
		{"per_layer_input_gate", model.Layers[0].PerLayerInputGate},
		{"per_layer_projection", model.Layers[0].PerLayerProjection},
	} {
		if adapter.Layers["model.layers.0."+target.name] != nil {
			t.Fatalf("%s should require AllowExtendedTargets", target.name)
		}
		if target.linear.LoRA != nil {
			t.Fatalf("%s should not have an attached LoRA adapter without opt-in", target.name)
		}
	}
}

func newLoraTestLinear() *metal.Linear {
	return metal.NewLinear(metal.FromValues([]float32{
		1, 2, 3, 4,
		5, 6, 7, 8,
		9, 10, 11, 12,
	}, 3, 4), nil)
}
