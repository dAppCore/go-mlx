// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/safetensors"
)

// TestAssembleVision builds a synthetic 2-layer SigLIP tower + projector and pins that AssembleVision
// gathers every role, infers the layer count, and validates presence.
func TestAssembleVision(t *testing.T) {
	mk := func(rows, cols int) safetensors.Tensor {
		return safetensors.Tensor{Dtype: "BF16", Shape: []int{rows, cols}, Data: make([]byte, rows*cols*2)}
	}
	vec := func(n int) safetensors.Tensor {
		return safetensors.Tensor{Dtype: "BF16", Shape: []int{n}, Data: make([]byte, n*2)}
	}
	const H, layers = 64, 2
	w := map[string]safetensors.Tensor{"patch_embedding.weight": mk(H, 588)} // hidden 64, patchDim 588 → patch 14
	for i := 0; i < layers; i++ {
		p := core.Sprintf("encoder.layers.%d", i)
		for _, n := range []string{".input_layernorm", ".post_attention_layernorm", ".pre_feedforward_layernorm", ".post_feedforward_layernorm", ".self_attn.q_norm", ".self_attn.k_norm"} {
			w[p+n+".weight"] = vec(H)
		}
		for _, n := range []string{".self_attn.q_proj", ".self_attn.k_proj", ".self_attn.v_proj", ".self_attn.o_proj"} {
			w[p+n+".weight"] = mk(H, H)
		}
		w[p+".mlp.gate_proj.weight"] = mk(H*4, H)
		w[p+".mlp.up_proj.weight"] = mk(H*4, H)
		w[p+".mlp.down_proj.weight"] = mk(H, H*4)
	}
	w["multi_modal_projector.proj.weight"] = mk(H, H)

	tc := &Gemma4TextConfig{}
	tc.ModelType = "gemma4"
	tc.VisionConfig = &Gemma4VisionConfig{}
	tc.VisionConfig.NumAttentionHeads = 8

	v, err := AssembleVision(w, tc)
	if err != nil {
		t.Fatalf("AssembleVision: %v", err)
	}
	if v == nil {
		t.Fatal("expected a vision tower")
	}
	if len(v.Layers) != layers {
		t.Fatalf("layers = %d, want %d", len(v.Layers), layers)
	}
	if v.PatchEmbedding == nil {
		t.Fatal("patch embedding missing")
	}
	if v.Layers[0].Q.Weight == nil || v.Layers[0].QNorm == nil || v.Layers[0].Gate.Weight == nil {
		t.Fatal("layer 0 q/qnorm/gate missing")
	}
	if v.Projector.Projection.Weight == nil {
		t.Fatal("projector missing")
	}
}

// TestAssembleVisionTextOnly pins that a pack with no vision tower yields (nil, nil).
func TestAssembleVisionTextOnly(t *testing.T) {
	tc := &Gemma4TextConfig{}
	tc.ModelType = "gemma4"
	v, err := AssembleVision(map[string]safetensors.Tensor{
		"model.layers.0.self_attn.q_proj.weight": {Shape: []int{4, 4}},
	}, tc)
	if err != nil || v != nil {
		t.Fatalf("text-only pack should yield (nil,nil), got (%v, %v)", v, err)
	}
}
