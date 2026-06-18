// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"testing"

	core "dappco.re/go"
	g4 "dappco.re/go/mlx/pkg/model/gemma4"
	"dappco.re/go/mlx/pkg/safetensors"
)

// quantTensors builds synthetic HF-named 4-bit gemma4 tensors — packed/scales/biases of the
// CORRECT byte sizes with distinct per-tensor fills. No real quantisation is needed: the
// consumers under test only map + size-check bytes, so an arbitrary byte pattern of the right
// length exercises every path.
func quantTensors(arch g4.Arch, gs, bits int) map[string]safetensors.Tensor {
	ts := map[string]safetensors.Tensor{}
	next := byte(1)
	fill := func(n int) []byte {
		d := make([]byte, n)
		for j := range d {
			d[j] = next
		}
		next++
		return d
	}
	mkNorm := func(name string, elems int) {
		ts[name] = safetensors.Tensor{Dtype: "BF16", Shape: []int{elems}, Data: fill(elems * bf16Size)}
	}
	mkQuant := func(prefix string, outDim, inDim int) {
		ts[prefix+".weight"] = safetensors.Tensor{Dtype: "U32", Shape: []int{outDim, inDim * bits / 32}, Data: fill(outDim * inDim * bits / 8)}
		ts[prefix+".scales"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{outDim, inDim / gs}, Data: fill(outDim * (inDim / gs) * bf16Size)}
		ts[prefix+".biases"] = safetensors.Tensor{Dtype: "BF16", Shape: []int{outDim, inDim / gs}, Data: fill(outDim * (inDim / gs) * bf16Size)}
	}
	dModel, headDim, dFF := arch.Hidden, arch.HeadDim, arch.FF
	qDim, kvDim := arch.Heads*headDim, arch.KVHeads*headDim
	for i := range arch.Layer {
		p := core.Sprintf("model.layers.%d", i)
		mkNorm(p+".input_layernorm.weight", dModel)
		mkNorm(p+".pre_feedforward_layernorm.weight", dModel)
		mkNorm(p+".self_attn.q_norm.weight", headDim)
		mkNorm(p+".self_attn.k_norm.weight", headDim)
		mkNorm(p+".post_attention_layernorm.weight", dModel)
		mkNorm(p+".post_feedforward_layernorm.weight", dModel)
		mkNorm(p+".layer_scalar", 1)
		mkQuant(p+".self_attn.q_proj", qDim, dModel)
		mkQuant(p+".self_attn.k_proj", kvDim, dModel)
		mkQuant(p+".self_attn.v_proj", kvDim, dModel)
		mkQuant(p+".self_attn.o_proj", dModel, qDim)
		mkQuant(p+".mlp.gate_proj", dFF, dModel)
		mkQuant(p+".mlp.up_proj", dFF, dModel)
		mkQuant(p+".mlp.down_proj", dModel, dFF)
	}
	return ts
}

func quantArch(t *testing.T, layers int) g4.Arch {
	t.Helper()
	// asymmetric dims (qDim 32, kvDim 16, dModel 64, dFF 128 — all distinct) so a wrong-dim
	// mapping can't hide behind a symmetric coincidence.
	arch, err := g4.Config{
		HiddenSize: 64, NumHiddenLayers: layers, IntermediateSize: 128,
		NumAttentionHeads: 2, NumKeyValueHeads: 1, HeadDim: 16, VocabSize: 32, RMSNormEps: 1e-6,
	}.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	return arch
}

// TestNormalizeGemma4Names gates the gemma4_unified name handling (used by the bf16 + Mistral
// assemblers): a real checkpoint wraps the text weights under language_model.model.* and ships
// vision/audio tower tensors; normalizeGemma4Names strips the prefix and drops the towers, so a
// consumer that looks for bare model.* names sees the text weights byte-for-byte. (The quant path
// no longer hand-assembles — pkg/model/gemma4.Assemble owns that, with its own tests; this gates
// the native normalize the remaining bf16/Mistral assemblers still depend on.)
func TestNormalizeGemma4Names(t *testing.T) {
	const gs, bits = 32, 4
	arch := quantArch(t, 2)
	bare := quantTensors(arch, gs, bits)

	// a real gemma4_unified shape: every text tensor under language_model.*, plus vision/audio
	// towers the text path must ignore.
	prefixed := map[string]safetensors.Tensor{
		"vision_embedder.patch_dense.weight":      {Dtype: "BF16", Shape: []int{4}, Data: make([]byte, 8)},
		"embed_audio.embedding_projection.weight": {Dtype: "BF16", Shape: []int{2}, Data: make([]byte, 4)},
	}
	for k, v := range bare {
		prefixed["language_model."+k] = v
	}

	norm := normalizeGemma4Names(prefixed)
	if len(norm) != len(bare) {
		t.Fatalf("normalised map has %d tensors, want %d (towers dropped, text kept)", len(norm), len(bare))
	}
	if _, ok := norm["vision_embedder.patch_dense.weight"]; ok {
		t.Fatal("vision tower tensor was not dropped")
	}
	if len(normalizeGemma4Names(bare)) != len(bare) {
		t.Fatal("bare names should pass through normalize unchanged")
	}
	for k, v := range bare {
		if !bytes.Equal(norm[k].Data, v.Data) {
			t.Fatalf("normalised %s does not match the bare tensor byte-for-byte", k)
		}
	}
	t.Logf("normalize: language_model.* text weights unprefix to bare names byte-for-byte; vision/audio towers dropped")
}
