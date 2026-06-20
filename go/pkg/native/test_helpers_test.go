// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"os"
	"testing"

	core "dappco.re/go"
	mlxmetal "dappco.re/go/mlx/pkg/metal"
	g4 "dappco.re/go/mlx/pkg/model/gemma4"
	"dappco.re/go/mlx/pkg/model/mistral"
	"dappco.re/go/mlx/pkg/safetensors"
)

func requireNativeRuntime(t testing.TB) {
	t.Helper()
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if err := ensureInit(); err != nil {
		t.Fatal(err)
	}
}

func syntheticFloat32(n, salt int) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = float32((i*salt+7)%101-50) * 0.03125
	}
	return v
}

func quantWeightFixture(tb testing.TB, outDim, inDim, groupSize, bits, salt int) QuantWeight {
	tb.Helper()
	arr := mlxmetal.FromRawBytes(toBF16Bytes(syntheticFloat32(outDim*inDim, salt)), []int{outDim, inDim}, mlxmetal.DTypeBFloat16)
	wq, scales, biases, err := mlxmetal.Quantize(arr, groupSize, bits, "affine")
	if err != nil {
		mlxmetal.Free(arr)
		tb.Fatalf("Quantize: %v", err)
	}
	mlxmetal.Materialize(wq, scales, biases)
	out := QuantWeight{
		Packed:    append([]byte(nil), wq.RawBytes()...),
		Scales:    append([]byte(nil), scales.RawBytes()...),
		Biases:    append([]byte(nil), biases.RawBytes()...),
		GroupSize: groupSize,
		Bits:      bits,
	}
	mlxmetal.Free(arr, wq, scales, biases)
	return out
}

func decodeInputsFixture(tokens, dModel int) [][]byte {
	inputs := make([][]byte, tokens)
	for i := range inputs {
		inputs[i] = toBF16Bytes(syntheticFloat32(dModel, i+3))
	}
	return inputs
}

func decodeLayerFixture(dModel, nHeads, nKVHeads, headDim, dFF, salt int) DecodeLayerWeights {
	qDim, kvDim := nHeads*headDim, nKVHeads*headDim
	return DecodeLayerWeights{
		AttnNormW: toBF16Bytes(syntheticFloat32(dModel, salt+1)),
		WQ:        toBF16Bytes(syntheticFloat32(qDim*dModel, salt+3)),
		WK:        toBF16Bytes(syntheticFloat32(kvDim*dModel, salt+5)),
		WV:        toBF16Bytes(syntheticFloat32(kvDim*dModel, salt+7)),
		WO:        toBF16Bytes(syntheticFloat32(dModel*qDim, salt+11)),
		MLPNormW:  toBF16Bytes(syntheticFloat32(dModel, salt+13)),
		WGate:     toBF16Bytes(syntheticFloat32(dFF*dModel, salt+17)),
		WUp:       toBF16Bytes(syntheticFloat32(dFF*dModel, salt+19)),
		WDown:     toBF16Bytes(syntheticFloat32(dModel*dFF, salt+23)),
	}
}

func quantizedLayerFixture(tb testing.TB, dModel, nHeads, nKVHeads, headDim, dFF, groupSize, bits, salt int) QuantizedLayerWeights {
	tb.Helper()
	qDim, kvDim := nHeads*headDim, nKVHeads*headDim
	return QuantizedLayerWeights{
		AttnNormW: toBF16Bytes(syntheticFloat32(dModel, salt+1)),
		MLPNormW:  toBF16Bytes(syntheticFloat32(dModel, salt+13)),
		Q:         quantWeightFixture(tb, qDim, dModel, groupSize, bits, salt+3),
		K:         quantWeightFixture(tb, kvDim, dModel, groupSize, bits, salt+5),
		V:         quantWeightFixture(tb, kvDim, dModel, groupSize, bits, salt+7),
		O:         quantWeightFixture(tb, dModel, qDim, groupSize, bits, salt+11),
		Gate:      quantWeightFixture(tb, dFF, dModel, groupSize, bits, salt+17),
		Up:        quantWeightFixture(tb, dFF, dModel, groupSize, bits, salt+19),
		Down:      quantWeightFixture(tb, dModel, dFF, groupSize, bits, salt+23),
		GroupSize: groupSize,
		Bits:      bits,
	}
}

func archFixture(tb testing.TB, dModel, nHeads, nKVHeads, headDim, dFF, vocab, nLayers int) g4.Arch {
	tb.Helper()
	cfg := g4.Config{
		HiddenSize: dModel, NumHiddenLayers: nLayers, IntermediateSize: dFF,
		NumAttentionHeads: nHeads, NumKeyValueHeads: nKVHeads, HeadDim: headDim,
		VocabSize: vocab, RMSNormEps: 1e-5, RopeTheta: 10000,
	}
	arch, err := cfg.Arch()
	if err != nil {
		tb.Fatalf("Config.Arch: %v", err)
	}
	return arch
}

func gemma4BF16Fixture(tb testing.TB, dModel, nHeads, nKVHeads, headDim, dFF, vocab, nLayers int) (*Gemma4BF16, g4.Arch) {
	tb.Helper()
	arch := archFixture(tb, dModel, nHeads, nKVHeads, headDim, dFF, vocab, nLayers)
	layers := make([]DecodeLayerWeights, len(arch.Layer))
	for i := range layers {
		layers[i] = decodeLayerFixture(dModel, nHeads, nKVHeads, headDim, dFF, (i+1)*100)
	}
	g := &Gemma4BF16{
		Layers:    layers,
		Embed:     toBF16Bytes(syntheticFloat32(vocab*dModel, 11)),
		FinalNorm: toBF16Bytes(syntheticFloat32(dModel, 7)),
	}
	g.LMHead, g.Tied = g.Embed, true
	return g, arch
}

func gemma4TensorFixture(arch g4.Arch, withLMHead bool) map[string]safetensors.Tensor {
	tensors := map[string]safetensors.Tensor{}
	salt := 1
	mk := func(name string, elems int) {
		tensors[name] = safetensors.Tensor{
			Dtype: "BF16",
			Shape: []int{elems},
			Data:  toBF16Bytes(syntheticFloat32(elems, salt)),
		}
		salt++
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
	return tensors
}

func mistralConfigFixture(tb testing.TB, dModel, nHeads, nKVHeads, headDim, dFF, vocab, nLayers int) (mistral.Config, g4.Arch) {
	tb.Helper()
	cfg := mistral.Config{
		HiddenSize: dModel, NumHiddenLayers: nLayers, IntermediateSize: dFF,
		NumAttentionHeads: nHeads, NumKeyValueHeads: nKVHeads, HeadDim: headDim,
		VocabSize: vocab, RMSNormEps: 1e-6,
	}
	arch, err := cfg.Arch()
	if err != nil {
		tb.Fatalf("mistral Config.Arch: %v", err)
	}
	return cfg, arch
}

func mistralTensorFixture(tb testing.TB, dModel, nHeads, nKVHeads, headDim, dFF, vocab, nLayers int) map[string]safetensors.Tensor {
	tb.Helper()
	tensors := map[string]safetensors.Tensor{}
	salt := 1
	mk := func(name string, elems int) {
		tensors[name] = safetensors.Tensor{
			Dtype: "BF16",
			Shape: []int{elems},
			Data:  toBF16Bytes(syntheticFloat32(elems, salt)),
		}
		salt++
	}
	qDim, kvDim := nHeads*headDim, nKVHeads*headDim
	mk("language_model.model.embed_tokens.weight", vocab*dModel)
	mk("language_model.model.norm.weight", dModel)
	for i := 0; i < nLayers; i++ {
		p := core.Sprintf("language_model.model.layers.%d", i)
		mk(p+".input_layernorm.weight", dModel)
		mk(p+".post_attention_layernorm.weight", dModel)
		mk(p+".self_attn.q_proj.weight", qDim*dModel)
		mk(p+".self_attn.k_proj.weight", kvDim*dModel)
		mk(p+".self_attn.v_proj.weight", kvDim*dModel)
		mk(p+".self_attn.o_proj.weight", dModel*qDim)
		mk(p+".mlp.gate_proj.weight", dFF*dModel)
		mk(p+".mlp.up_proj.weight", dFF*dModel)
		mk(p+".mlp.down_proj.weight", dModel*dFF)
	}
	mk("vision_tower.transformer.layers.0.attention.q_proj.weight", dModel*dModel)
	mk("multi_modal_projector.linear_1.weight", dModel*dModel)
	return tensors
}

func moeLayerWeightsFixture(numExperts, topK, dModel, dFF, expertDFF, salt int) MoELayerWeights {
	scale := make([]float32, numExperts)
	for i := range scale {
		scale[i] = 0.5 + float32(i)*0.125
	}
	return MoELayerWeights{
		NumExperts: numExperts, TopK: topK, ExpertDFF: expertDFF,
		PreFFNormW: toBF16Bytes(syntheticFloat32(dModel, salt+1)), PreFFNorm2W: toBF16Bytes(syntheticFloat32(dModel, salt+2)),
		PostFFNorm1W: toBF16Bytes(syntheticFloat32(dModel, salt+3)), PostFFNorm2W: toBF16Bytes(syntheticFloat32(dModel, salt+4)),
		PostFFNormW:       toBF16Bytes(syntheticFloat32(dModel, salt+5)),
		WGate:             toBF16Bytes(syntheticFloat32(dFF*dModel, salt+6)),
		WUp:               toBF16Bytes(syntheticFloat32(dFF*dModel, salt+7)),
		WDown:             toBF16Bytes(syntheticFloat32(dModel*dFF, salt+8)),
		RouterNormWScaled: toBF16Bytes(syntheticFloat32(dModel, salt+9)),
		RouterW:           toBF16Bytes(syntheticFloat32(numExperts*dModel, salt+10)),
		PerExpertScale:    toBF16Bytes(scale),
		ExpGateW:          toBF16Bytes(syntheticFloat32(numExperts*expertDFF*dModel, salt+11)),
		ExpUpW:            toBF16Bytes(syntheticFloat32(numExperts*expertDFF*dModel, salt+12)),
		ExpDownW:          toBF16Bytes(syntheticFloat32(numExperts*dModel*expertDFF, salt+13)),
	}
}

func toBF16Bytes(f []float32) []byte {
	b := make([]byte, len(f)*bf16Size)
	for i, v := range f {
		h := f32ToBF16(v)
		b[i*bf16Size] = byte(h)
		b[i*bf16Size+1] = byte(h >> 8)
	}
	return b
}

func bf16Floats(b []byte) []float32 {
	out := make([]float32, len(b)/bf16Size)
	for i := range out {
		out[i] = bf16ToF32(b[i*bf16Size], b[i*bf16Size+1])
	}
	return out
}

func assertFloat32Near(t *testing.T, name string, got, want []float32, tol float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s length mismatch: got %d, want %d", name, len(got), len(want))
	}
	for i := range want {
		if d := float32(math.Abs(float64(got[i] - want[i]))); d > tol {
			t.Fatalf("%s[%d] = %v, want %v (diff %v > %v)", name, i, got[i], want[i], d, tol)
		}
	}
}

func eqBytes(t *testing.T, what string, got, want []byte) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: len %d != %d", what, len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("%s: differ at byte %d: %#x vs %#x", what, i, got[i], want[i])
		}
	}
}

func cosineBF16(a, b []byte) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, na, nb float64
	for i := 0; i+1 < len(a); i += bf16Size {
		av := float64(bf16ToF32(a[i], a[i+1]))
		bv := float64(bf16ToF32(b[i], b[i+1]))
		dot += av * bv
		na += av * av
		nb += bv * bv
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}
