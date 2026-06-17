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
