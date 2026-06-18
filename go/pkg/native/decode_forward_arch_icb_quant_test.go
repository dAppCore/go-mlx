// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package native

import (
	"os"
	"testing"

	core "dappco.re/go"
	g4 "dappco.re/go/mlx/pkg/model/gemma4"
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
	check := func(name string, qlayers []QuantizedLayerWeights, specs []g4.LayerSpec, T, slidingWindow int) {
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
	check("all-owner/global", ql3, g4.DeriveLayers(full3, 0), 4, 0)
	{
		inputs := mkInputs(4)
		gotArch, err := DecodeForwardArchICBQuant(inputs, ql3, g4.DeriveLayers(full3, 0), dModel, nHeads, nKV, headDim, maxLen, dFF, 0, base, scale, eps, false)
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
	check("kv-share", buildLayers(2), g4.DeriveLayers([]string{"full_attention", "full_attention"}, 1), 4, 0)

	// (c) sliding-window W=3 over 6 tokens.
	slide3 := []string{"sliding_attention", "sliding_attention", "sliding_attention"}
	check("sliding-W3", buildLayers(3), g4.DeriveLayers(slide3, 0), 6, 3)

	// (d) KV-share + sliding combined.
	mixed := []string{"sliding_attention", "full_attention", "sliding_attention", "full_attention"}
	check("kv-share+sliding", buildLayers(4), g4.DeriveLayers(mixed, 2), 6, 3)

	// (e) MoE rejected.
	moeSpecs := g4.DeriveLayers([]string{"full_attention", "full_attention"}, 0)
	moeSpecs[1].MoE = true
	if _, err := DecodeForwardArchICBQuant(mkInputs(3), buildLayers(2), moeSpecs, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, base, scale, eps, false); err == nil {
		t.Fatal("expected DecodeForwardArchICBQuant to reject a MoE layer, got nil error")
	}

	t.Logf("stacked quant+ICB arch: replay ≡ DecodeForwardArchQuant byte-for-byte across all-owner/global, KV-share, sliding(W=3), KV-share+sliding; all-owner ≡ DecodeForwardICBQuant; MoE rejected — both levers on the arch path")
}
