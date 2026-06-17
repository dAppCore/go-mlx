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
func archShareRef(t *testing.T, layers []DecodeLayerWeights, specs []g4.LayerSpec, inputs [][]byte, dModel, nHeads, nKV, headDim, dFF, maxLen int, base, scale, eps float32) [][]byte {
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
			L := tok + 1
			attn := must(SDPA(qr, seqToHeadMajor(aK, nKV, headDim, L), seqToHeadMajor(aV, nKV, headDim, L), 1, nHeads, nKV, headDim, L, scale))
			h := must(AddBF16(x, must(MatVecBF16(w.WO, attn, dModel, qDim))))
			x = must(MLPBlockBF16(h, w.MLPNormW, w.WGate, w.WUp, w.WDown, dModel, dFF, eps))
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
	gotOwn, err := DecodeForwardArch(inputs, layers, specsOwn, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps)
	if err != nil {
		t.Fatalf("DecodeForwardArch all-owner: %v", err)
	}
	refOwn := archShareRef(t, layers, specsOwn, inputs, dModel, nHeads, nKV, headDim, dFF, maxLen, base, scale, eps)
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
	gotShare, err := DecodeForwardArch(inputs, layers2, specsShare, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps)
	if err != nil {
		t.Fatalf("DecodeForwardArch share: %v", err)
	}
	refShare := archShareRef(t, layers2, specsShare, inputs, dModel, nHeads, nKV, headDim, dFF, maxLen, base, scale, eps)
	for tok := 0; tok < T; tok++ {
		eqBytes(t, core.Sprintf("KV-share vs ref tok%d", tok), gotShare[tok], refShare[tok])
	}
	t.Logf("executor: DecodeForwardArch honours the arch — all-owner ≡ DecodeForward; layer-1-shares-layer-0 ≡ composed reference (sharer attends owner's KV)")
}
