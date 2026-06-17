// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package native

import (
	"testing"

	core "dappco.re/go"
	g4 "dappco.re/go/mlx/pkg/model/gemma4"
	"dappco.re/go/mlx/pkg/safetensors"
)

// tinyGemma4Arch is a synthetic dense bf16 gemma4 (no SDPA is run, so the head dim is
// free) — small enough to build every named tensor by hand.
func tinyGemma4Arch(t *testing.T) g4.Arch {
	t.Helper()
	a, err := g4.Config{
		HiddenSize: 8, NumHiddenLayers: 2, IntermediateSize: 16,
		NumAttentionHeads: 2, NumKeyValueHeads: 1, HeadDim: 4, VocabSize: 10, RMSNormEps: 1e-6,
	}.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	return a
}

// gemma4Tensors builds the full named bf16 tensor set for arch, each tensor filled with a
// distinct byte (recorded in fills) so a wrong field assignment is detectable. withLMHead
// adds a separate lm_head.weight (untied); otherwise the model ties to the embedding.
func gemma4Tensors(arch g4.Arch, withLMHead bool) (map[string]safetensors.Tensor, map[string]byte) {
	ts := map[string]safetensors.Tensor{}
	fills := map[string]byte{}
	next := byte(1)
	mk := func(name string, elems int) {
		data := make([]byte, elems*bf16Size)
		for j := range data {
			data[j] = next
		}
		ts[name] = safetensors.Tensor{Dtype: "BF16", Shape: []int{elems}, Data: data}
		fills[name] = next
		next++
	}
	dModel, headDim, dFF, vocab := arch.Hidden, arch.HeadDim, arch.FF, arch.Vocab
	qDim, kvDim := arch.Heads*headDim, arch.KVHeads*headDim
	mk("model.embed_tokens.weight", vocab*dModel)
	mk("model.norm.weight", dModel)
	if withLMHead {
		mk("lm_head.weight", vocab*dModel)
	}
	for i := range arch.Layer {
		p := core.Sprintf("model.layers.%d", i)
		mk(p+".input_layernorm.weight", dModel)
		mk(p+".self_attn.q_proj.weight", qDim*dModel)
		mk(p+".self_attn.k_proj.weight", kvDim*dModel)
		mk(p+".self_attn.v_proj.weight", kvDim*dModel)
		mk(p+".self_attn.o_proj.weight", dModel*qDim)
		mk(p+".self_attn.q_norm.weight", headDim)
		mk(p+".self_attn.k_norm.weight", headDim)
		mk(p+".post_attention_layernorm.weight", dModel)
		mk(p+".pre_feedforward_layernorm.weight", dModel)
		mk(p+".post_feedforward_layernorm.weight", dModel)
		mk(p+".mlp.gate_proj.weight", dFF*dModel)
		mk(p+".mlp.up_proj.weight", dFF*dModel)
		mk(p+".mlp.down_proj.weight", dModel*dFF)
	}
	return ts, fills
}

func allEq(b []byte, v byte) bool {
	if len(b) == 0 {
		return false
	}
	for _, x := range b {
		if x != v {
			return false
		}
	}
	return true
}

// TestAssembleGemma4BF16 gates the gemma4 bf16 assembly: every native weight field gets
// the bytes of its correctly-named source tensor (distinct fills catch a misassignment),
// tied vs untied LM head is resolved, and malformed inputs are rejected.
func TestAssembleGemma4BF16(t *testing.T) {
	arch := tinyGemma4Arch(t)
	ts, fills := gemma4Tensors(arch, false) // tied

	g, err := AssembleGemma4BF16(ts, arch)
	if err != nil {
		t.Fatalf("AssembleGemma4BF16: %v", err)
	}
	if !g.Tied || !allEq(g.LMHead, fills["model.embed_tokens.weight"]) {
		t.Fatalf("expected LM head tied to the embedding")
	}
	check := func(name string, got []byte, src string) {
		if !allEq(got, fills[src]) {
			t.Fatalf("%s did not get %s's bytes (fill %d)", name, src, fills[src])
		}
	}
	check("Embed", g.Embed, "model.embed_tokens.weight")
	check("FinalNorm", g.FinalNorm, "model.norm.weight")
	for i := range arch.Layer {
		p := core.Sprintf("model.layers.%d", i)
		l := g.Layers[i]
		check("AttnNormW", l.AttnNormW, p+".input_layernorm.weight")
		check("WQ", l.WQ, p+".self_attn.q_proj.weight")
		check("WK", l.WK, p+".self_attn.k_proj.weight")
		check("WV", l.WV, p+".self_attn.v_proj.weight")
		check("WO", l.WO, p+".self_attn.o_proj.weight")
		check("MLPNormW(preFF)", l.MLPNormW, p+".pre_feedforward_layernorm.weight")
		check("WGate", l.WGate, p+".mlp.gate_proj.weight")
		check("WUp", l.WUp, p+".mlp.up_proj.weight")
		check("WDown", l.WDown, p+".mlp.down_proj.weight")
		check("QNormW", l.QNormW, p+".self_attn.q_norm.weight")
		check("KNormW", l.KNormW, p+".self_attn.k_norm.weight")
		check("PostAttnNormW", l.PostAttnNormW, p+".post_attention_layernorm.weight")
		check("PostFFNormW", l.PostFFNormW, p+".post_feedforward_layernorm.weight")
	}

	// untied: a separate lm_head.weight wins over the tied embedding.
	tsU, fillsU := gemma4Tensors(arch, true)
	gU, err := AssembleGemma4BF16(tsU, arch)
	if err != nil {
		t.Fatalf("AssembleGemma4BF16 untied: %v", err)
	}
	if gU.Tied || !allEq(gU.LMHead, fillsU["lm_head.weight"]) {
		t.Fatal("expected untied LM head from lm_head.weight")
	}

	// rejections: missing required, wrong dtype, wrong size, MoE arch.
	drop, _ := gemma4Tensors(arch, false)
	delete(drop, "model.layers.0.self_attn.q_proj.weight")
	if _, err := AssembleGemma4BF16(drop, arch); err == nil {
		t.Fatal("expected error on a missing required tensor")
	}
	badDtype, _ := gemma4Tensors(arch, false)
	bd := badDtype["model.layers.0.mlp.gate_proj.weight"]
	bd.Dtype = "F32"
	badDtype["model.layers.0.mlp.gate_proj.weight"] = bd
	if _, err := AssembleGemma4BF16(badDtype, arch); err == nil {
		t.Fatal("expected error on a non-BF16 tensor")
	}
	badSize, _ := gemma4Tensors(arch, false)
	bs := badSize["model.layers.0.self_attn.k_proj.weight"]
	bs.Data = bs.Data[:len(bs.Data)-2]
	badSize["model.layers.0.self_attn.k_proj.weight"] = bs
	if _, err := AssembleGemma4BF16(badSize, arch); err == nil {
		t.Fatal("expected error on a wrong-size tensor")
	}
	moeArch, _ := g4.Config{
		HiddenSize: 8, NumHiddenLayers: 2, IntermediateSize: 16, NumAttentionHeads: 2,
		NumKeyValueHeads: 1, HeadDim: 4, VocabSize: 10, RMSNormEps: 1e-6,
		EnableMoEBlock: true, NumExperts: 4, TopKExperts: 2, MoEIntermediateSize: 16,
	}.Arch()
	tsMoE, _ := gemma4Tensors(arch, false)
	if _, err := AssembleGemma4BF16(tsMoE, moeArch); err == nil {
		t.Fatal("expected error: MoE arch not supported yet")
	}

	t.Logf("assemble: %d layers × bf16 names → native structs (Q/K/V/O + 4 gemma4 norms + dense MLP + embed/norm/lm_head); tied+untied; malformed rejected", len(arch.Layer))
}
