// SPDX-Licence-Identifier: EUPL-1.2

package qwen3

import (
	"math"
	"testing"

	"dappco.re/go/mlx/pkg/model"
)

// TestQwen3Arch verifies the dense qwen3 config→Arch: a standard GQA transformer, scale 1/sqrt(head_dim),
// full rotary, all-global attention, no value-norm/softcap.
func TestQwen3Arch(t *testing.T) {
	const layers, headDim = 28, 128
	c := &Config{
		HiddenSize: 2048, NumHiddenLayers: layers, NumAttentionHeads: 16, NumKeyValueHeads: 8,
		HeadDim: headDim, IntermediateSize: 6144, VocabSize: 151936, RMSNormEps: 1e-6, RopeTheta: 1_000_000,
	}
	a, err := c.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	if want := float32(1.0 / math.Sqrt(headDim)); a.AttnScale != want {
		t.Errorf("AttnScale = %v, want %v", a.AttnScale, want)
	}
	if a.RotaryDim != headDim || a.RopeBase != 1_000_000 || a.SoftCap != 0 || a.ValueNorm {
		t.Errorf("qwen3 arch specifics wrong: rotary=%d rope=%v softcap=%v valuenorm=%v", a.RotaryDim, a.RopeBase, a.SoftCap, a.ValueNorm)
	}
	if len(a.Layer) != layers {
		t.Fatalf("layers = %d, want %d", len(a.Layer), layers)
	}
	for i := range a.Layer { // qwen3 dense: every layer global
		if a.Layer[i].Attention != model.GlobalAttention {
			t.Fatalf("layer %d not global (qwen3 dense has no sliding)", i)
		}
	}
	t.Logf("qwen3 Arch: %d global layers, scale=1/sqrt(%d), full rotary, QK-norm via weight names", layers, headDim)
}

// TestQwen3Registered confirms qwen3 is registered with the llama/mistral norm layout (MLP norm =
// post_attention_layernorm, no gemma post-attn norm), QK-norm names kept, and plain RMSNorm.
func TestQwen3Registered(t *testing.T) {
	spec, ok := model.LookupArch("qwen3")
	if !ok {
		t.Fatal("qwen3 not registered")
	}
	if spec.Weights.MLPNorm != ".post_attention_layernorm.weight" {
		t.Errorf("MLPNorm = %q, want llama-style post_attention_layernorm", spec.Weights.MLPNorm)
	}
	if spec.Weights.PostAttnNorm != "" {
		t.Errorf("PostAttnNorm = %q, want empty (qwen3 has no gemma post-attn norm)", spec.Weights.PostAttnNorm)
	}
	if spec.Weights.QNorm != ".self_attn.q_norm.weight" {
		t.Errorf("QNorm = %q, want QK-norm bound by name", spec.Weights.QNorm)
	}
	if spec.Weights.NormBiasOne {
		t.Error("NormBiasOne must be false for qwen3 (plain RMSNorm, not gemma)")
	}
	t.Log("qwen3 registered: dense qwen3 loads via the reactive loader (mistral-style norms + QK-norm, plain RMSNorm)")
}
