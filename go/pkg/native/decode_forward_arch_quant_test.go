// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package native

import (
	"os"
	"testing"

	core "dappco.re/go"
	g4 "dappco.re/go/mlx/pkg/model/gemma4"
)

// lastTokenDiffers reports whether two forwards' final-token outputs differ.
func lastTokenDiffers(a, b [][]byte) bool {
	la, lb := a[len(a)-1], b[len(b)-1]
	if len(la) != len(lb) {
		return true
	}
	for i := range la {
		if la[i] != lb[i] {
			return true
		}
	}
	return false
}

// TestDecodeForwardArchQuant gates the 4-bit arch-driven forward. (a) an all-owner,
// all-global, dense quant arch is byte-for-byte the proven DecodeForwardQuant (the arch
// executor + qmv projector ≡ the standalone quant forward when the arch routes nothing)
// — the correctness anchor. (b) a KV-share quant arch differs from the all-owner one
// (sharing genuinely reroutes layer 1's attention on the quant path). (c) a sliding
// quant arch (W=3) differs from full attention over 6 tokens (the window clips on the
// quant path). (d) a MoE layer is rejected (quantised experts are a later slice).
func TestDecodeForwardArchQuant(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, gs, bits = 512, 8, 4, 64, 1024, 64, 4
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)

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

	// (a) all-owner all-global ≡ DecodeForwardQuant byte-for-byte.
	const nL, T, maxLen = 3, 4, 8
	ql := make([]QuantizedLayerWeights, nL)
	types := make([]string, nL)
	for l := range ql {
		ql[l] = buildQuantLayer(t, dModel, nHeads, nKV, headDim, dFF, gs, bits, (l+1)*100)
		types[l] = "full_attention"
	}
	inputs := mkInputs(T)
	specsOwn := g4.DeriveLayers(types, 0)
	gotArch, err := DecodeForwardArchQuant(inputs, ql, specsOwn, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, base, scale, eps, false)
	if err != nil {
		t.Fatalf("DecodeForwardArchQuant all-owner: %v", err)
	}
	ref, err := DecodeForwardQuant(inputs, ql, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps)
	if err != nil {
		t.Fatalf("DecodeForwardQuant: %v", err)
	}
	for tok := 0; tok < T; tok++ {
		eqBytes(t, core.Sprintf("quant all-owner vs DecodeForwardQuant tok%d", tok), gotArch[tok], ref[tok])
	}

	// (b) KV-share reroutes attention: 2 layers, layer 1 shares layer 0's cache vs both
	// own. Different layer weights → the shared and owned results must differ.
	ql2 := []QuantizedLayerWeights{
		buildQuantLayer(t, dModel, nHeads, nKV, headDim, dFF, gs, bits, 100),
		buildQuantLayer(t, dModel, nHeads, nKV, headDim, dFF, gs, bits, 200),
	}
	in2 := mkInputs(T)
	specsShare := g4.DeriveLayers([]string{"full_attention", "full_attention"}, 1)
	specsBothOwn := g4.DeriveLayers([]string{"full_attention", "full_attention"}, 0)
	gotShare, err := DecodeForwardArchQuant(in2, ql2, specsShare, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, base, scale, eps, false)
	if err != nil {
		t.Fatalf("DecodeForwardArchQuant share: %v", err)
	}
	gotBothOwn, err := DecodeForwardArchQuant(in2, ql2, specsBothOwn, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, base, scale, eps, false)
	if err != nil {
		t.Fatalf("DecodeForwardArchQuant both-own: %v", err)
	}
	if !lastTokenDiffers(gotShare, gotBothOwn) {
		t.Fatal("quant KV-share produced the same output as all-owner — sharing did not reroute attention")
	}

	// (c) sliding clips on the quant path: all-sliding W=3 over 6 tokens vs full (W=0).
	const W, T2, maxLen2 = 3, 6, 8
	slideTypes := make([]string, nL)
	for i := range slideTypes {
		slideTypes[i] = "sliding_attention"
	}
	specsSlide := g4.DeriveLayers(slideTypes, 0)
	in3 := mkInputs(T2)
	gotSlide, err := DecodeForwardArchQuant(in3, ql, specsSlide, dModel, nHeads, nKV, headDim, maxLen2, dFF, W, base, scale, eps, false)
	if err != nil {
		t.Fatalf("DecodeForwardArchQuant sliding: %v", err)
	}
	gotFull, err := DecodeForwardArchQuant(in3, ql, specsSlide, dModel, nHeads, nKV, headDim, maxLen2, dFF, 0, base, scale, eps, false)
	if err != nil {
		t.Fatalf("DecodeForwardArchQuant sliding-full: %v", err)
	}
	if !lastTokenDiffers(gotSlide, gotFull) {
		t.Fatal("quant sliding (W=3) matched full attention over 6 tokens — the window did not clip")
	}

	// (d) MoE layers are rejected on the quant path.
	moeSpecs := g4.DeriveLayers(types, 0)
	moeSpecs[1].MoE = true
	if _, err := DecodeForwardArchQuant(inputs, ql, moeSpecs, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, base, scale, eps, false); err == nil {
		t.Fatal("expected DecodeForwardArchQuant to reject a MoE layer, got nil error")
	}

	t.Logf("quant arch: all-owner ≡ DecodeForwardQuant byte-for-byte; KV-share reroutes; sliding (W=%d, %d toks) clips; MoE rejected — 4-bit on the arch path", W, T2)
}
