// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"reflect"
	"testing"

	core "dappco.re/go"
	g4 "dappco.re/go/mlx/pkg/model/gemma4"
	"dappco.re/go/mlx/pkg/safetensors"
)

// quantTensors builds synthetic HF-named 4-bit gemma4 tensors — packed/scales/biases of the
// CORRECT byte sizes with distinct per-tensor fills. No real quantisation is needed: the layer
// assembler only maps + size-validates bytes, so an arbitrary byte pattern of the right length
// exercises every path.
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

// TestAssembleGemma4QuantLayers proves the quant layer assembler maps every gemma4-named
// tensor onto the right QuantizedLayerWeights field, byte-for-byte, with groupSize/bits set.
func TestAssembleGemma4QuantLayers(t *testing.T) {
	const gs, bits = 32, 4
	arch := quantArch(t, 2)
	ts := quantTensors(arch, gs, bits)

	layers, err := AssembleGemma4QuantLayers(ts, arch, gs, bits)
	if err != nil {
		t.Fatalf("assemble: %v", err)
	}
	if len(layers) != len(arch.Layer) {
		t.Fatalf("assembled %d layers, want %d", len(layers), len(arch.Layer))
	}
	checkQ := func(prefix string, qw QuantWeight) {
		t.Helper()
		if !bytes.Equal(qw.Packed, ts[prefix+".weight"].Data) {
			t.Fatalf("%s.weight not mapped", prefix)
		}
		if !bytes.Equal(qw.Scales, ts[prefix+".scales"].Data) {
			t.Fatalf("%s.scales not mapped", prefix)
		}
		if !bytes.Equal(qw.Biases, ts[prefix+".biases"].Data) {
			t.Fatalf("%s.biases not mapped", prefix)
		}
	}
	checkNorm := func(name string, got []byte) {
		t.Helper()
		if !bytes.Equal(got, ts[name].Data) {
			t.Fatalf("%s not mapped", name)
		}
	}
	for i := range layers {
		p := core.Sprintf("model.layers.%d", i)
		checkQ(p+".self_attn.q_proj", layers[i].Q)
		checkQ(p+".self_attn.k_proj", layers[i].K)
		checkQ(p+".self_attn.v_proj", layers[i].V)
		checkQ(p+".self_attn.o_proj", layers[i].O)
		checkQ(p+".mlp.gate_proj", layers[i].Gate)
		checkQ(p+".mlp.up_proj", layers[i].Up)
		checkQ(p+".mlp.down_proj", layers[i].Down)
		if layers[i].GroupSize != gs || layers[i].Bits != bits {
			t.Fatalf("layer %d groupSize/bits = %d/%d, want %d/%d", i, layers[i].GroupSize, layers[i].Bits, gs, bits)
		}
		checkNorm(p+".input_layernorm.weight", layers[i].AttnNormW)
		checkNorm(p+".pre_feedforward_layernorm.weight", layers[i].MLPNormW)
		checkNorm(p+".self_attn.q_norm.weight", layers[i].QNormW)
		checkNorm(p+".self_attn.k_norm.weight", layers[i].KNormW)
		checkNorm(p+".post_attention_layernorm.weight", layers[i].PostAttnNormW)
		checkNorm(p+".post_feedforward_layernorm.weight", layers[i].PostFFNormW)
		checkNorm(p+".layer_scalar", layers[i].LayerScalarW)
	}
	t.Logf("quant layer assembly: %d layers, 7 projections × (packed/scales/biases) + 6 norms mapped by gemma4 name, byte-for-byte", len(layers))
}

// TestAssembleGemma4QuantLayersErrors checks the rejections: a missing projection component, a
// wrong packed byte span, non-bf16 scales, and a bad/indivisible groupSize.
func TestAssembleGemma4QuantLayersErrors(t *testing.T) {
	const gs, bits = 32, 4
	arch := quantArch(t, 1)

	missing := quantTensors(arch, gs, bits)
	delete(missing, "model.layers.0.self_attn.q_proj.scales")
	if _, err := AssembleGemma4QuantLayers(missing, arch, gs, bits); err == nil {
		t.Fatal("missing .scales: expected an error")
	}

	shortPacked := quantTensors(arch, gs, bits)
	w := shortPacked["model.layers.0.mlp.gate_proj.weight"]
	w.Data = w.Data[:len(w.Data)-2]
	shortPacked["model.layers.0.mlp.gate_proj.weight"] = w
	if _, err := AssembleGemma4QuantLayers(shortPacked, arch, gs, bits); err == nil {
		t.Fatal("short packed weight: expected an error")
	}

	badScaleDtype := quantTensors(arch, gs, bits)
	s := badScaleDtype["model.layers.0.self_attn.k_proj.scales"]
	s.Dtype = "F32"
	badScaleDtype["model.layers.0.self_attn.k_proj.scales"] = s
	if _, err := AssembleGemma4QuantLayers(badScaleDtype, arch, gs, bits); err == nil {
		t.Fatal("non-bf16 scales: expected an error")
	}

	ok := quantTensors(arch, gs, bits)
	if _, err := AssembleGemma4QuantLayers(ok, arch, 0, bits); err == nil {
		t.Fatal("zero groupSize: expected an error")
	}
	if _, err := AssembleGemma4QuantLayers(ok, arch, 48, bits); err == nil {
		t.Fatal("groupSize not dividing inDim: expected an error")
	}
	t.Logf("quant assembly rejections: missing component, short packed, non-bf16 scales, zero/indivisible groupSize all error")
}

// TestAssembleGemma4UnifiedPrefix gates the gemma4_unified name handling: a real checkpoint
// wraps the text weights under language_model.model.* and ships vision/audio tower tensors;
// normalizeGemma4Names must strip the prefix and drop the towers, so the assembler — which
// looks for bare model.* names — produces the SAME layers it does from bare names.
func TestAssembleGemma4UnifiedPrefix(t *testing.T) {
	const gs, bits = 32, 4
	arch := quantArch(t, 2)
	bare := quantTensors(arch, gs, bits)

	// a real gemma4_unified shape: every text tensor under language_model.*, plus vision/audio
	// towers the text assembler must ignore.
	prefixed := map[string]safetensors.Tensor{
		"vision_embedder.patch_dense.weight":      {Dtype: "BF16", Shape: []int{4}, Data: make([]byte, 8)},
		"embed_audio.embedding_projection.weight": {Dtype: "BF16", Shape: []int{2}, Data: make([]byte, 4)},
	}
	for k, v := range bare {
		prefixed["language_model."+k] = v
	}

	// normalize: drops the 2 towers, unprefixes the text tensors; bare names pass through.
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

	// the assembler consumes the real-checkpoint names identically to bare names.
	bareLayers, err := AssembleGemma4QuantLayers(bare, arch, gs, bits)
	if err != nil {
		t.Fatalf("bare assemble: %v", err)
	}
	prefixedLayers, err := AssembleGemma4QuantLayers(prefixed, arch, gs, bits)
	if err != nil {
		t.Fatalf("prefixed assemble (real-checkpoint names): %v", err)
	}
	if !reflect.DeepEqual(bareLayers, prefixedLayers) {
		t.Fatal("language_model.model.* + vision/audio junk did not normalise to the bare assembly")
	}
	t.Logf("unified prefix: language_model.model.* text weights assemble identically to bare names; vision/audio towers dropped")
}
