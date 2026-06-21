// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package native

import (
	"crypto/sha256"
	"os"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/model"
)

// TestDecodeForwardArchICBQuant gates the stacked fast path — 4-bit qmv weights AND the
// ICB encode-bypass replay, arch-driven. It must equal DecodeForwardArchQuant (the quant
// re-encode arch path) byte-for-byte across every arch axis: all-owner/global, KV-share,
// sliding-window, and KV-share + sliding combined. The all-owner case is also tied to the
// non-arch DecodeForwardICBQuant. MoE is rejected.
func TestDecodeForwardArchICBQuant(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, gs, bits = 512, 8, 4, 64, 1024, 64, 4
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	const maxLen = 8

	mkInputs := func(n int) [][]byte {
		in := make([][]byte, n)
		for i := range in {
			f := make([]float32, dModel)
			for j := range f {
				f[j] = float32((j*(i+3)+5)%97-48) * 0.02
			}
			in[i] = toBF16Bytes(f)
		}
		return in
	}
	buildLayers := func(n int) []QuantizedLayerWeights {
		ls := make([]QuantizedLayerWeights, n)
		for li := range ls {
			ls[li] = buildQuantLayer(t, dModel, nHeads, nKV, headDim, dFF, gs, bits, (li+1)*100)
		}
		return ls
	}

	// check: DecodeForwardArchICBQuant ≡ DecodeForwardArchQuant byte-for-byte.
	check := func(name string, qlayers []QuantizedLayerWeights, specs []model.LayerSpec, T, slidingWindow int) {
		inputs := mkInputs(T)
		got, err := DecodeForwardArchICBQuant(inputs, qlayers, specs, dModel, nHeads, nKV, headDim, maxLen, dFF, slidingWindow, base, scale, eps, false)
		if err != nil {
			t.Fatalf("%s: DecodeForwardArchICBQuant: %v", name, err)
		}
		want, err := DecodeForwardArchQuant(inputs, qlayers, specs, dModel, nHeads, nKV, headDim, maxLen, dFF, slidingWindow, base, scale, eps, false)
		if err != nil {
			t.Fatalf("%s: DecodeForwardArchQuant: %v", name, err)
		}
		for tok := 0; tok < T; tok++ {
			eqBytes(t, core.Sprintf("%s tok%d", name, tok), got[tok], want[tok])
		}
	}

	// (a) all-owner/global — also tie to the non-arch quant ICB (DecodeForwardICBQuant).
	full3 := []string{"full_attention", "full_attention", "full_attention"}
	ql3 := buildLayers(3)
	check("all-owner/global", ql3, model.DeriveLayers(full3, 0), 4, 0)
	{
		inputs := mkInputs(4)
		gotArch, err := DecodeForwardArchICBQuant(inputs, ql3, model.DeriveLayers(full3, 0), dModel, nHeads, nKV, headDim, maxLen, dFF, 0, base, scale, eps, false)
		if err != nil {
			t.Fatalf("arch-icb-quant: %v", err)
		}
		gotPlain, err := DecodeForwardICBQuant(inputs, ql3, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps)
		if err != nil {
			t.Fatalf("DecodeForwardICBQuant: %v", err)
		}
		for tok := 0; tok < 4; tok++ {
			eqBytes(t, core.Sprintf("all-owner vs DecodeForwardICBQuant tok%d", tok), gotArch[tok], gotPlain[tok])
		}
	}

	// (b) KV-share.
	check("kv-share", buildLayers(2), model.DeriveLayers([]string{"full_attention", "full_attention"}, 1), 4, 0)

	// (c) sliding-window W=3 over 6 tokens.
	slide3 := []string{"sliding_attention", "sliding_attention", "sliding_attention"}
	check("sliding-W3", buildLayers(3), model.DeriveLayers(slide3, 0), 6, 3)

	// (d) KV-share + sliding combined.
	mixed := []string{"sliding_attention", "full_attention", "sliding_attention", "full_attention"}
	check("kv-share+sliding", buildLayers(4), model.DeriveLayers(mixed, 2), 6, 3)

	// (e) MoE rejected.
	moeSpecs := model.DeriveLayers([]string{"full_attention", "full_attention"}, 0)
	moeSpecs[1].MoE = true
	if _, err := DecodeForwardArchICBQuant(mkInputs(3), buildLayers(2), moeSpecs, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, base, scale, eps, false); err == nil {
		t.Fatal("expected DecodeForwardArchICBQuant to reject a MoE layer, got nil error")
	}

	t.Logf("stacked quant+ICB arch: replay ≡ DecodeForwardArchQuant byte-for-byte across all-owner/global, KV-share, sliding(W=3), KV-share+sliding; all-owner ≡ DecodeForwardICBQuant; MoE rejected — both levers on the arch path")
}

func TestDecodeForwardArchICBQuantPLE(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, gs, bits = 128, 2, 1, 64, 256, 32, 4
	const vocab, vocabPLI, pliDim = 19, 23, 32
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	const maxLen = 6
	tokenIDs := []int32{1, 5, 3, 7}
	specs := model.DeriveLayers([]string{"full_attention", "full_attention"}, 0)

	embed, embedScales, embedBiases := quantizeProj(t, vocab, dModel, gs, bits, 31)
	inputs, err := EmbedTokensQuant(embed, embedScales, embedBiases, tokenIDs, vocab, dModel, gs, bits, 1)
	if err != nil {
		t.Fatalf("EmbedTokensQuant: %v", err)
	}

	qLayers := make([]QuantizedLayerWeights, len(specs))
	for li := range qLayers {
		qLayers[li] = buildQuantLayer(t, dModel, nHeads, nKV, headDim, dFF, gs, bits, (li+1)*100)
		qLayers[li].PerLayerGate = quantWeightFixture(t, pliDim, dModel, gs, bits, li*10+41)
		qLayers[li].PerLayerProjection = quantWeightFixture(t, dModel, pliDim, gs, bits, li*10+43)
		qLayers[li].PostPerLayerInputNormW = toBF16Bytes(syntheticFloat32(dModel, li*10+47))
		qLayers[li].LayerScalarW = toBF16Bytes([]float32{0.75 + float32(li)*0.125})
	}

	plDim := len(specs) * pliDim
	embedPL, embedPLScales, embedPLBiases := quantizeProj(t, vocabPLI, plDim, gs, bits, 53)
	projPL, projPLScales, projPLBiases := quantizeProj(t, plDim, dModel, gs, bits, 59)
	ple := ArchPLEQuant{
		TokenIDs:      tokenIDs,
		EmbedPerLayer: embedPL, EmbedPerLayerScales: embedPLScales, EmbedPerLayerBiases: embedPLBiases,
		PerLayerModelProjW: projPL, PerLayerModelProjScales: projPLScales, PerLayerModelProjBiases: projPLBiases,
		PerLayerProjNormW: toBF16Bytes(syntheticFloat32(pliDim, 61)),
		VocabPLI:          vocabPLI, PliDim: pliDim,
		GroupSize: gs, Bits: bits, ProjGroupSize: gs, ProjBits: bits,
	}

	got, err := DecodeForwardArchICBQuant(inputs, qLayers, specs, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, base, scale, eps, false, ple)
	if err != nil {
		t.Fatalf("DecodeForwardArchICBQuant PLE: %v", err)
	}
	want, err := DecodeForwardArchQuant(inputs, qLayers, specs, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, base, scale, eps, false, ple)
	if err != nil {
		t.Fatalf("DecodeForwardArchQuant PLE: %v", err)
	}
	h := sha256.New()
	for tok := range tokenIDs {
		eqBytes(t, core.Sprintf("quant PLE ICB tok%d", tok), got[tok], want[tok])
		_, _ = h.Write(got[tok])
	}
	gotHash := core.Sprintf("%x", h.Sum(nil))
	const wantHash = "f83918ed05160ddf4533f1253c6498889d9ba3736e66539902f98176edc637cb"
	if gotHash != wantHash {
		t.Fatalf("quant PLE ICB hash = %s, want %s", gotHash, wantHash)
	}
	t.Logf("quant PLE arch ICB: replay ≡ DecodeForwardArchQuant byte-for-byte with token-id PerLayerInputs, PLE gate, post norm, and layer scalar; sha256=%s", gotHash)
}
