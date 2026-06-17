// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package native

import (
	"os"
	"testing"

	core "dappco.re/go"
	g4 "dappco.re/go/mlx/pkg/model/gemma4"
)

// TestDecodeForwardArchICB gates the arch-driven cache-grow ICB (the encode-bypass
// replay) against the proven re-encode arch forward DecodeForwardArch — byte-for-byte
// across every arch axis: all-owner/global, KV-share, sliding-window, and KV-share +
// sliding combined. Same weights + inputs + arch → the ICB replay must equal the
// re-encode path exactly. MoE layers are rejected.
func TestDecodeForwardArchICB(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF = 512, 8, 4, 64, 1024
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
	buildLayers := func(n int) []DecodeLayerWeights {
		ls := make([]DecodeLayerWeights, n)
		for li := range ls {
			ls[li] = forwardLayer(dModel, nHeads, nKV, headDim, dFF, (li+1)*100)
		}
		return ls
	}

	// check: DecodeForwardArchICB ≡ DecodeForwardArch byte-for-byte on the given arch.
	check := func(name string, layers []DecodeLayerWeights, specs []g4.LayerSpec, T, slidingWindow int) {
		inputs := mkInputs(T)
		got, err := DecodeForwardArchICB(inputs, layers, specs, dModel, nHeads, nKV, headDim, maxLen, dFF, slidingWindow, base, scale, eps)
		if err != nil {
			t.Fatalf("%s: DecodeForwardArchICB: %v", name, err)
		}
		want, err := DecodeForwardArch(inputs, layers, specs, dModel, nHeads, nKV, headDim, maxLen, dFF, slidingWindow, base, scale, eps)
		if err != nil {
			t.Fatalf("%s: DecodeForwardArch: %v", name, err)
		}
		for tok := 0; tok < T; tok++ {
			eqBytes(t, core.Sprintf("%s tok%d", name, tok), got[tok], want[tok])
		}
	}

	// (a) all-owner, all-global.
	full3 := []string{"full_attention", "full_attention", "full_attention"}
	check("all-owner/global", buildLayers(3), g4.DeriveLayers(full3, 0), 4, 0)

	// (b) KV-share: layer 1 shares layer 0's cache.
	check("kv-share", buildLayers(2), g4.DeriveLayers([]string{"full_attention", "full_attention"}, 1), 4, 0)

	// (c) sliding-window: all sliding, W=3 over 6 tokens (toks 3..5 clip).
	slide3 := []string{"sliding_attention", "sliding_attention", "sliding_attention"}
	check("sliding-W3", buildLayers(3), g4.DeriveLayers(slide3, 0), 6, 3)

	// (d) KV-share + sliding combined: 4 layers, mixed types, 2 shared → the last
	// sliding/full layers share the matching owner's cache, sliding layers windowed.
	mixed := []string{"sliding_attention", "full_attention", "sliding_attention", "full_attention"}
	check("kv-share+sliding", buildLayers(4), g4.DeriveLayers(mixed, 2), 6, 3)

	// (e) MoE is rejected on the ICB path.
	moeLayers := buildLayers(2)
	moeSpecs := g4.DeriveLayers([]string{"full_attention", "full_attention"}, 0)
	moeSpecs[1].MoE = true
	if _, err := DecodeForwardArchICB(mkInputs(3), moeLayers, moeSpecs, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, base, scale, eps); err == nil {
		t.Fatal("expected DecodeForwardArchICB to reject a MoE layer, got nil error")
	}

	t.Logf("arch ICB: replay ≡ DecodeForwardArch byte-for-byte across all-owner/global, KV-share, sliding(W=3), and KV-share+sliding; MoE rejected")
}

// TestDecodeForwardArchICBNorms gates the gemma4 norms on the ICB path: with all four
// gemma4 norms set (QK-norm + post-attn + post-FF), the cache-grow ICB replay equals the
// now-norm-complete re-encode arch forward byte-for-byte — across a mixed sliding +
// KV-share arch, for both bf16 and 4-bit — and differs from the same arch with the norms
// dropped (the recorded norm ops are genuinely live).
func TestDecodeForwardArchICBNorms(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, gs, bits = 512, 8, 4, 64, 1024, 64, 4
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	const maxLen, T, W = 8, 6, 3
	mixed := []string{"sliding_attention", "full_attention", "sliding_attention", "full_attention"}
	specs := g4.DeriveLayers(mixed, 2)
	nL := len(specs)

	inputs := make([][]byte, T)
	for i := range inputs {
		f := make([]float32, dModel)
		for j := range f {
			f[j] = float32((j*(i+3)+5)%97-48) * 0.02
		}
		inputs[i] = toBF16Bytes(f)
	}
	dnorm := func(salt int) []byte {
		f := make([]float32, dModel)
		for j := range f {
			f[j] = float32((j*salt+3)%29-14) * 0.03
		}
		return toBF16Bytes(f)
	}
	hnorm := func(salt int) []byte {
		f := make([]float32, headDim)
		for j := range f {
			f[j] = float32((j*salt+5)%23-11) * 0.04
		}
		return toBF16Bytes(f)
	}

	// bf16: ICB ≡ re-encode, with the four norms.
	layers := make([]DecodeLayerWeights, nL)
	for li := range layers {
		layers[li] = forwardLayer(dModel, nHeads, nKV, headDim, dFF, (li+1)*100)
		layers[li].QNormW, layers[li].KNormW = hnorm(li*4+1), hnorm(li*4+2)
		layers[li].PostAttnNormW, layers[li].PostFFNormW = dnorm(li*4+3), dnorm(li*4+4)
	}
	gotICB, err := DecodeForwardArchICB(inputs, layers, specs, dModel, nHeads, nKV, headDim, maxLen, dFF, W, base, scale, eps)
	if err != nil {
		t.Fatalf("DecodeForwardArchICB norms: %v", err)
	}
	want, err := DecodeForwardArch(inputs, layers, specs, dModel, nHeads, nKV, headDim, maxLen, dFF, W, base, scale, eps)
	if err != nil {
		t.Fatalf("DecodeForwardArch norms: %v", err)
	}
	for tok := 0; tok < T; tok++ {
		eqBytes(t, core.Sprintf("bf16 ICB-norms vs re-encode tok%d", tok), gotICB[tok], want[tok])
	}

	// non-vacuous: dropping the norms changes the ICB output.
	bare := make([]DecodeLayerWeights, nL)
	copy(bare, layers)
	for li := range bare {
		bare[li].QNormW, bare[li].KNormW, bare[li].PostAttnNormW, bare[li].PostFFNormW = nil, nil, nil, nil
	}
	gotBare, err := DecodeForwardArchICB(inputs, bare, specs, dModel, nHeads, nKV, headDim, maxLen, dFF, W, base, scale, eps)
	if err != nil {
		t.Fatalf("DecodeForwardArchICB bare: %v", err)
	}
	if !lastTokenDiffers(gotICB, gotBare) {
		t.Fatal("ICB norms made no difference — the recorded norm ops were not live")
	}

	// 4-bit: ICB ≡ re-encode, with the four norms.
	ql := make([]QuantizedLayerWeights, nL)
	for li := range ql {
		ql[li] = buildQuantLayer(t, dModel, nHeads, nKV, headDim, dFF, gs, bits, (li+1)*100)
		ql[li].QNormW, ql[li].KNormW = hnorm(li*4+1), hnorm(li*4+2)
		ql[li].PostAttnNormW, ql[li].PostFFNormW = dnorm(li*4+3), dnorm(li*4+4)
	}
	gotQICB, err := DecodeForwardArchICBQuant(inputs, ql, specs, dModel, nHeads, nKV, headDim, maxLen, dFF, W, base, scale, eps)
	if err != nil {
		t.Fatalf("DecodeForwardArchICBQuant norms: %v", err)
	}
	wantQ, err := DecodeForwardArchQuant(inputs, ql, specs, dModel, nHeads, nKV, headDim, maxLen, dFF, W, base, scale, eps)
	if err != nil {
		t.Fatalf("DecodeForwardArchQuant norms: %v", err)
	}
	for tok := 0; tok < T; tok++ {
		eqBytes(t, core.Sprintf("quant ICB-norms vs re-encode tok%d", tok), gotQICB[tok], wantQ[tok])
	}

	t.Logf("arch ICB norms: replay ≡ norm-complete re-encode byte-for-byte (bf16 + 4-bit) across sliding+KV-share with QK-norm + post-attn + post-FF, and differs from without — the ICB fast path is now gemma4-norm-complete")
}
