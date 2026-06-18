// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package native

import (
	"os"
	"testing"

	core "dappco.re/go"
	g4 "dappco.re/go/mlx/pkg/model/gemma4"
)

// archShareRef is the arch-aware oracle for DecodeForwardArch, composed from the
// parity-proven standalone ops: owner layers project+append+attend their own
// seq-major cache; sharer layers project only Q and attend the OWNER's cache (read
// head-major for the proven SDPA). Mirrors DecodeForwardArch op-for-op.
func archShareRef(t *testing.T, layers []DecodeLayerWeights, specs []g4.LayerSpec, inputs [][]byte, dModel, nHeads, nKV, headDim, dFF, maxLen, slidingWindow int, base, scale, eps float32) [][]byte {
	t.Helper()
	qDim, kvDim := nHeads*headDim, nKV*headDim
	rowBytes := kvDim * bf16Size
	nLayers, T := len(layers), len(inputs)
	must := func(b []byte, err error) []byte {
		if err != nil {
			t.Fatalf("archShareRef op: %v", err)
		}
		return b
	}
	kC := make([][]byte, nLayers)
	vC := make([][]byte, nLayers)
	for li := range specs {
		if specs[li].OwnsCache() {
			kC[li] = make([]byte, maxLen*rowBytes)
			vC[li] = make([]byte, maxLen*rowBytes)
		}
	}
	out := make([][]byte, T)
	for tok := 0; tok < T; tok++ {
		x := inputs[tok]
		for li := 0; li < nLayers; li++ {
			w := layers[li]
			normed := must(RMSNormBF16(x, w.AttnNormW, 1, dModel, eps))
			qr := must(RoPEBF16(must(MatVecBF16(w.WQ, normed, qDim, dModel)), 1, nHeads, headDim, base, scale, tok, false))
			var aK, aV []byte
			if specs[li].OwnsCache() {
				knew := must(RoPEBF16(must(MatVecBF16(w.WK, normed, kvDim, dModel)), 1, nKV, headDim, base, scale, tok, false))
				vnew := must(MatVecBF16(w.WV, normed, kvDim, dModel))
				copy(kC[li][tok*rowBytes:(tok+1)*rowBytes], knew)
				copy(vC[li][tok*rowBytes:(tok+1)*rowBytes], vnew)
				aK, aV = kC[li], vC[li]
			} else {
				own := specs[li].KVShareFrom
				aK, aV = kC[own], vC[own] // owner wrote row tok earlier this token
			}
			slideW := 0
			if specs[li].Attention == g4.SlidingAttention {
				slideW = slidingWindow
			}
			start, n := slideWindow(tok, slideW)
			off := start * rowBytes
			attn := must(SDPA(qr, seqToHeadMajor(aK[off:], nKV, headDim, n), seqToHeadMajor(aV[off:], nKV, headDim, n), 1, nHeads, nKV, headDim, n, scale))
			h := must(AddBF16(x, must(MatVecBF16(w.WO, attn, dModel, qDim))))
			if w.MoE != nil {
				x = moeBlockRef(t, h, *w.MoE, dModel, dFF, eps) // dual-branch MoE FFN
			} else {
				x = must(MLPBlockBF16(h, w.MLPNormW, w.WGate, w.WUp, w.WDown, dModel, dFF, eps))
			}
		}
		out[tok] = x
	}
	return out
}

// TestDecodeForwardArch gates the executor's first slice — the arch-driven forward
// honouring KV-cache-sharing. (a) an all-owner arch is byte-for-byte the proven
// DecodeForward (the arch consumes the spec but routes nothing → identical), and
// equals the composed reference. (b) a 2-layer arch where layer 1 SHARES layer 0's
// cache equals the reference where layer 1 attends layer 0's KV — proving the
// sharer skips its own K/V and reads the owner's, the cache-topology made live.
func TestDecodeForwardArch(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF = 512, 8, 4, 64, 1024
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	const T, maxLen = 4, 8
	inputs := make([][]byte, T)
	for i := range inputs {
		f := make([]float32, dModel)
		for j := range f {
			f[j] = float32((j*(i+3)+5)%97-48) * 0.02
		}
		inputs[i] = toBF16Bytes(f)
	}

	// (a) all-owner ≡ DecodeForward AND ≡ the reference
	const nL = 3
	layers := make([]DecodeLayerWeights, nL)
	ownTypes := make([]string, nL)
	for li := range layers {
		layers[li] = forwardLayer(dModel, nHeads, nKV, headDim, dFF, (li+1)*100)
		ownTypes[li] = "full_attention"
	}
	specsOwn := g4.DeriveLayers(ownTypes, 0)
	ref0, err := DecodeForward(inputs, layers, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps)
	if err != nil {
		t.Fatalf("DecodeForward: %v", err)
	}
	gotOwn, err := DecodeForwardArch(inputs, layers, specsOwn, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, base, scale, eps, false)
	if err != nil {
		t.Fatalf("DecodeForwardArch all-owner: %v", err)
	}
	refOwn := archShareRef(t, layers, specsOwn, inputs, dModel, nHeads, nKV, headDim, dFF, maxLen, 0, base, scale, eps)
	for tok := 0; tok < T; tok++ {
		eqBytes(t, core.Sprintf("all-owner vs DecodeForward tok%d", tok), gotOwn[tok], ref0[tok])
		eqBytes(t, core.Sprintf("all-owner vs ref tok%d", tok), gotOwn[tok], refOwn[tok])
	}

	// (b) KV-share: 2 layers, layer 1 shares layer 0's cache
	layers2 := []DecodeLayerWeights{
		forwardLayer(dModel, nHeads, nKV, headDim, dFF, 100),
		forwardLayer(dModel, nHeads, nKV, headDim, dFF, 200),
	}
	specsShare := g4.DeriveLayers([]string{"full_attention", "full_attention"}, 1)
	if specsShare[1].OwnsCache() || specsShare[1].KVShareFrom != 0 {
		t.Fatalf("expected layer 1 to share layer 0: %+v", specsShare[1])
	}
	gotShare, err := DecodeForwardArch(inputs, layers2, specsShare, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, base, scale, eps, false)
	if err != nil {
		t.Fatalf("DecodeForwardArch share: %v", err)
	}
	refShare := archShareRef(t, layers2, specsShare, inputs, dModel, nHeads, nKV, headDim, dFF, maxLen, 0, base, scale, eps)
	for tok := 0; tok < T; tok++ {
		eqBytes(t, core.Sprintf("KV-share vs ref tok%d", tok), gotShare[tok], refShare[tok])
	}

	// (c) sliding-window: W=3 with T2=6 tokens (so toks 3..5 clip to the last 3),
	// a sliding arch all-owner. Gated vs the windowed reference — proving sliding
	// layers attend only the last W cache rows. Also assert it DIFFERS from the
	// global forward on the same weights (the window genuinely clips, not vacuous).
	const W, T2, maxLen2 = 3, 6, 8
	in2 := make([][]byte, T2)
	for i := range in2 {
		f := make([]float32, dModel)
		for j := range f {
			f[j] = float32((j*(i+2)+3)%89-44) * 0.02
		}
		in2[i] = toBF16Bytes(f)
	}
	slideTypes := make([]string, nL)
	for li := range slideTypes {
		slideTypes[li] = "sliding_attention"
	}
	specsSlide := g4.DeriveLayers(slideTypes, 0) // all sliding, all own
	gotSlide, err := DecodeForwardArch(in2, layers, specsSlide, dModel, nHeads, nKV, headDim, maxLen2, dFF, W, base, scale, eps, false)
	if err != nil {
		t.Fatalf("DecodeForwardArch sliding: %v", err)
	}
	refSlide := archShareRef(t, layers, specsSlide, in2, dModel, nHeads, nKV, headDim, dFF, maxLen2, W, base, scale, eps)
	for tok := 0; tok < T2; tok++ {
		eqBytes(t, core.Sprintf("sliding vs windowed ref tok%d", tok), gotSlide[tok], refSlide[tok])
	}
	// the window must actually clip: full-attention on the same weights differs at a
	// token past the window (tok 5 sees all 6 vs only the last 3).
	gotFull := archShareRef(t, layers, g4.DeriveLayers(slideTypes, 0), in2, dModel, nHeads, nKV, headDim, dFF, maxLen2, 0, base, scale, eps)
	same := true
	for i := range gotSlide[T2-1] {
		if gotSlide[T2-1][i] != gotFull[T2-1][i] {
			same = false
			break
		}
	}
	if same {
		t.Fatal("sliding (W=3) produced the same last-token output as full attention over 6 tokens — window did not clip")
	}
	t.Logf("executor: DecodeForwardArch honours the arch — all-owner ≡ DecodeForward; KV-share ≡ ref; sliding-window (W=%d, %d tokens) ≡ windowed ref and clips vs full attention", W, T2)
}

// TestDecodeForwardArchMoE gates the MoE wiring into the executor: a multi-layer arch
// where one layer is MoE (spec.MoE + layer.MoE weights) decodes byte-for-byte the
// arch reference (which routes that layer through moeBlockRef instead of the dense
// MLP). A non-vacuous check confirms the MoE layer genuinely changes the output: the
// same arch with that layer forced dense differs at the final token.
func TestDecodeForwardArchMoE(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	// headDim 64: the metallib ships sdpa_vector specializations for {64,96,128,256},
	// not 32 (real gemma4 E2B uses 256) — match the proven attention dims here.
	const dModel, nHeads, nKV, headDim, dFF = 512, 8, 4, 64, 1024
	const numExperts, topK, expertDFF = 8, 2, 768
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	const T, maxLen, nL, moeIdx = 3, 8, 3, 1

	inputs := make([][]byte, T)
	for i := range inputs {
		f := make([]float32, dModel)
		for j := range f {
			f[j] = float32((j*(i+3)+5)%97-48) * 0.02
		}
		inputs[i] = toBF16Bytes(f)
	}
	layers := make([]DecodeLayerWeights, nL)
	types := make([]string, nL)
	for li := range layers {
		layers[li] = forwardLayer(dModel, nHeads, nKV, headDim, dFF, (li+1)*100)
		types[li] = "full_attention"
	}
	specs := g4.DeriveLayers(types, 0)
	specs[moeIdx].MoE = true
	layers[moeIdx].MoE = buildMoEWeights(numExperts, topK, dModel, dFF, expertDFF, 200)

	got, err := DecodeForwardArch(inputs, layers, specs, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, base, scale, eps, false)
	if err != nil {
		t.Fatalf("DecodeForwardArch MoE: %v", err)
	}
	ref := archShareRef(t, layers, specs, inputs, dModel, nHeads, nKV, headDim, dFF, maxLen, 0, base, scale, eps)
	for tok := 0; tok < T; tok++ {
		eqBytes(t, core.Sprintf("MoE-layer arch vs ref tok%d", tok), got[tok], ref[tok])
	}

	// non-vacuous: forcing that one layer dense changes the output (the MoE FFN is
	// genuinely live, not a no-op that happens to match the dense path).
	denseLayers := make([]DecodeLayerWeights, nL)
	copy(denseLayers, layers)
	denseLayers[moeIdx].MoE = nil
	denseSpecs := g4.DeriveLayers(types, 0) // all MoE=false
	gotDense, err := DecodeForwardArch(inputs, denseLayers, denseSpecs, dModel, nHeads, nKV, headDim, maxLen, dFF, 0, base, scale, eps, false)
	if err != nil {
		t.Fatalf("DecodeForwardArch dense: %v", err)
	}
	same := true
	for i := range got[T-1] {
		if got[T-1][i] != gotDense[T-1][i] {
			same = false
			break
		}
	}
	if same {
		t.Fatal("the MoE layer produced the same final output as forcing it dense — the MoE FFN did not engage")
	}
	t.Logf("executor MoE wiring: layer %d MoE decodes ≡ arch ref over %d tokens and differs from the all-dense arch", moeIdx, T)
}
