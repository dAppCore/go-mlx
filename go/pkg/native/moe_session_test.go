// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package native

import (
	"math"
	"os"
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
	"dappco.re/go/mlx/pkg/model"
	g4 "dappco.re/go/mlx/pkg/model/gemma4"
	"dappco.re/go/mlx/pkg/safetensors"
)

// moeQuantTensors builds a synthetic MIXED-PRECISION MoE gemma4 checkpoint (gemma4 26B-A4B
// shape): attention + embedding + experts 4-bit, local MLP + router 8-bit. The experts are the
// batched SwitchGLU layout. quant.For drives the per-tensor width.
func moeQuantTensors(t *testing.T, arch g4.Arch, quant *g4.QuantConfig) map[string]safetensors.Tensor {
	t.Helper()
	ts := map[string]safetensors.Tensor{}
	salt := 1
	mkBF16 := func(name string, elems int) {
		f := make([]float32, elems)
		for i := range f {
			f[i] = float32((i*salt+7)%83-41) * 0.02
		}
		ts[name] = safetensors.Tensor{Dtype: "BF16", Shape: []int{elems}, Data: toBF16Bytes(f)}
		salt++
	}
	mkQuant := func(prefix string, outDim, inDim int) {
		_, bits := quant.For(prefix)
		p, s, b := quantizeProj(t, outDim, inDim, 64, bits, salt)
		salt++
		ts[prefix+".weight"] = safetensors.Tensor{Dtype: "U32", Shape: []int{outDim, inDim * bits / 32}, Data: p}
		ts[prefix+".scales"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{outDim, inDim / 64}, Data: s}
		ts[prefix+".biases"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{outDim, inDim / 64}, Data: b}
	}
	dModel, headDim, dFF, vocab := arch.Hidden, arch.HeadDim, arch.FF, arch.Vocab
	qDim, kvDim := arch.Heads*headDim, arch.KVHeads*headDim
	nE, eFF := arch.Experts, arch.ExpertFF
	mkQuant("model.embed_tokens", vocab, dModel)
	mkBF16("model.norm.weight", dModel)
	for i := range arch.Layer {
		p := core.Sprintf("model.layers.%d", i)
		mkBF16(p+".input_layernorm.weight", dModel)
		mkBF16(p+".post_attention_layernorm.weight", dModel)
		mkBF16(p+".self_attn.q_norm.weight", headDim)
		mkBF16(p+".self_attn.k_norm.weight", headDim)
		mkQuant(p+".self_attn.q_proj", qDim, dModel)
		mkQuant(p+".self_attn.k_proj", kvDim, dModel)
		mkQuant(p+".self_attn.v_proj", kvDim, dModel)
		mkQuant(p+".self_attn.o_proj", dModel, qDim)
		// MoE dual-branch: 5 norms, local MLP (8-bit), router (8-bit), batched experts (4-bit).
		mkBF16(p+".pre_feedforward_layernorm.weight", dModel)
		mkBF16(p+".pre_feedforward_layernorm_2.weight", dModel)
		mkBF16(p+".post_feedforward_layernorm_1.weight", dModel)
		mkBF16(p+".post_feedforward_layernorm_2.weight", dModel)
		mkBF16(p+".post_feedforward_layernorm.weight", dModel)
		mkQuant(p+".mlp.gate_proj", dFF, dModel)
		mkQuant(p+".mlp.up_proj", dFF, dModel)
		mkQuant(p+".mlp.down_proj", dModel, dFF)
		mkBF16(p+".router.scale", dModel)
		mkBF16(p+".router.per_expert_scale", nE)
		mkQuant(p+".router.proj", nE, dModel)
		mkQuant(p+".experts.switch_glu.gate_proj", nE*eFF, dModel)
		mkQuant(p+".experts.switch_glu.up_proj", nE*eFF, dModel)
		mkQuant(p+".experts.switch_glu.down_proj", nE*dModel, eFF)
	}
	return ts
}

// TestLoadGemma4QuantMoE gates the whole mixed-precision MoE path (gemma4 26B-A4B): a synthetic
// model (4-bit experts + attention, 8-bit local MLP + router) assembles into a session that
// generates; the first token equals the manual chain (embed → stepToken-with-MoEBlockQuant →
// lm_head → greedy); and a config.json carrying the per-tensor overrides dir-loads to the same.
func TestLoadGemma4QuantMoE(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, vocab = 64, 2, 1, 64, 32
	const dFF, expertDFF, numExperts, topK, numLayers = 128, 64, 4, 2, 2
	const maxLen, n = 16, 4
	// mixed precision: default 4-bit, local MLP + router 8-bit (the 26B-A4B QAT pattern).
	quant := &g4.QuantConfig{GroupSize: 64, Bits: 4, Overrides: map[string]g4.ModuleQuant{}}
	for i := 0; i < numLayers; i++ {
		for _, m := range []string{"mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "router.proj"} {
			quant.Overrides[core.Sprintf("model.layers.%d.%s", i, m)] = g4.ModuleQuant{GroupSize: 64, Bits: 8}
		}
	}
	cfg := g4.Config{
		HiddenSize: dModel, NumHiddenLayers: numLayers, IntermediateSize: dFF,
		NumAttentionHeads: nHeads, NumKeyValueHeads: nKV, HeadDim: headDim, VocabSize: vocab, RMSNormEps: 1e-6,
		EnableMoEBlock: true, NumExperts: numExperts, TopKExperts: topK, MoEIntermediateSize: expertDFF,
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	if !arch.HasMoE() {
		t.Fatal("arch should be MoE")
	}
	ts := moeQuantTensors(t, arch, quant)
	prompt := []int32{1, 5, 3}

	g, err := AssembleGemma4Quant(ts, arch, quant)
	if err != nil {
		t.Fatalf("AssembleGemma4Quant: %v", err)
	}
	if g.Layers[0].MoE == nil {
		t.Fatal("layer 0 should carry the quant MoE block")
	}
	if g.Layers[0].MoE.ExpertBits != 4 || g.Layers[0].MoE.LocalBits != 8 || g.Layers[0].MoE.RouterBits != 8 {
		t.Fatalf("per-component bits wrong: experts %d local %d router %d", g.Layers[0].MoE.ExpertBits, g.Layers[0].MoE.LocalBits, g.Layers[0].MoE.RouterBits)
	}
	sess, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchQuantSession: %v", err)
	}
	gen, err := sess.Generate(prompt, n, -1)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	for i, id := range gen {
		if id < 0 || int(id) >= vocab {
			t.Fatalf("token %d = %d out of range", i, id)
		}
	}

	// manual chain: embed → stepToken (MoEBlockQuant via moeQuant) → lm_head → greedy.
	attnScale := arch.AttnScale // the model-declared scale (gemma4 1.0), matching the session
	embedScale := float32(math.Sqrt(float64(dModel)))
	var manualFirst int32
	withAutoreleasePool(func() {
		lb, moeQ, _ := buildQuantArchLayerBufs(g.Layers, arch.Layer, dModel, nHeads, nKV, headDim, dFF, maxLen, arch.SlidingWindow, nil)
		st := newArchDecodeState(arch.Layer, lb, make([]*MoELayerWeights, numLayers), dModel, nHeads, nKV, headDim, dFF, arch.SlidingWindow, arch.RotaryDim, arch.RotaryDimLocal, arch.RopeBase, arch.RopeLocalBase, attnScale, arch.Eps, false)
		st.moeQuant = moeQ
		var hidden []byte
		for p, id := range prompt {
			embs, err := EmbedTokensQuant(g.Embed, g.EmbedScales, g.EmbedBiases, []int32{id}, vocab, dModel, 64, 4, embedScale)
			if err != nil {
				t.Fatalf("EmbedTokensQuant: %v", err)
			}
			if hidden, err = st.stepToken(embs[0], p); err != nil {
				t.Fatalf("stepToken: %v", err)
			}
		}
		logits, err := LMHeadQuant(hidden, g.FinalNorm, g.LMHead, g.LMHeadScales, g.LMHeadBiases, dModel, vocab, 64, 4, arch.Eps, arch.SoftCap)
		if err != nil {
			t.Fatalf("LMHeadQuant: %v", err)
		}
		if manualFirst, err = model.Greedy(logits, vocab); err != nil {
			t.Fatalf("Greedy: %v", err)
		}
	})
	if gen[0] != manualFirst {
		t.Fatalf("session first token %d != manual MoE chain %d", gen[0], manualFirst)
	}

	// dir-load: a config.json carrying the per-tensor overrides → LoadGemma4Quant4Dir ≡ in-memory.
	ovr := ""
	for i := 0; i < numLayers; i++ {
		for _, m := range []string{"mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "router.proj"} {
			ovr += core.Sprintf(`,"model.layers.%d.%s":{"group_size":64,"bits":8}`, i, m)
		}
	}
	configJSON := core.Sprintf(`{"hidden_size":%d,"num_hidden_layers":%d,"intermediate_size":%d,`+
		`"num_attention_heads":%d,"num_key_value_heads":%d,"head_dim":%d,"vocab_size":%d,"rms_norm_eps":1e-6,`+
		`"enable_moe_block":true,"num_experts":%d,"top_k_experts":%d,"moe_intermediate_size":%d,`+
		`"quantization":{"group_size":64,"bits":4%s}}`,
		dModel, numLayers, dFF, nHeads, nKV, headDim, vocab, numExperts, topK, expertDFF, ovr)
	dir := t.TempDir()
	if err := coreio.Local.Write(core.PathJoin(dir, "config.json"), configJSON); err != nil {
		t.Fatalf("write config: %v", err)
	}
	blob, err := safetensors.Encode(ts)
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	if err := coreio.Local.Write(core.PathJoin(dir, "model.safetensors"), string(blob)); err != nil {
		t.Fatalf("write weights: %v", err)
	}
	dirSess, err := LoadGemma4Quant4Dir(dir, maxLen)
	if err != nil {
		t.Fatalf("LoadGemma4Quant4Dir: %v", err)
	}
	genDir, err := dirSess.Generate(prompt, n, -1)
	if err != nil {
		t.Fatalf("dir Generate: %v", err)
	}
	if !idsEqual(genDir, gen) {
		t.Fatalf("dir-loaded MoE %v != in-memory %v", genDir, gen)
	}
	t.Logf("mixed-precision MoE end to end: 4-bit experts + 8-bit local/router assemble → session generates %v; first token ≡ manual chain; config.json overrides dir-load ≡ in-memory", gen)
}
