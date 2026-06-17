// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package native

import (
	"os"
	"testing"
	"time"
)

// forwardLayer builds one layer's synthetic weights (salt varies them per layer).
func forwardLayer(dModel, nHeads, nKV, headDim, dFF, salt int) DecodeLayerWeights {
	qDim, kvDim := nHeads*headDim, nKV*headDim
	mk := func(n, s int) []byte {
		f := make([]float32, n)
		for i := range f {
			f[i] = float32((i*s+7)%101-50) * 0.02
		}
		return toBF16Bytes(f)
	}
	return DecodeLayerWeights{
		AttnNormW: mk(dModel, salt+13), WQ: mk(qDim*dModel, salt+53),
		WK: mk(kvDim*dModel, salt+71), WV: mk(kvDim*dModel, salt+83), WO: mk(dModel*qDim, salt+17),
		MLPNormW: mk(dModel, salt+19), WGate: mk(dFF*dModel, salt+61),
		WUp: mk(dFF*dModel, salt+29), WDown: mk(dModel*dFF, salt+47),
	}
}

// TestDecodeForward gates the multi-layer, multi-token forward against the
// parity-proven single step: DecodeForward must equal stepping DecodeStepKV
// token-by-token, layer-by-layer (each layer's own growing cache). This anchors
// the loop wiring — the residual stream flowing layer→layer, the per-layer cache
// growth across tokens, the per-token position — to the proven real step.
func TestDecodeForward(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF = 512, 8, 4, 64, 1024
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	const nLayers, T, maxLen = 3, 4, 8
	kvDim := nKV * headDim

	layers := make([]DecodeLayerWeights, nLayers)
	for l := range layers {
		layers[l] = forwardLayer(dModel, nHeads, nKV, headDim, dFF, (l+1)*100)
	}
	inputs := make([][]byte, T)
	for i := range inputs {
		f := make([]float32, dModel)
		for j := range f {
			f[j] = float32((j*(i+3)+5)%97-48) * 0.02
		}
		inputs[i] = toBF16Bytes(f)
	}

	// reference: step DecodeStepKV through the loop, each layer its own Go cache
	kC := make([][]byte, nLayers)
	vC := make([][]byte, nLayers)
	for l := range kC {
		kC[l] = make([]byte, maxLen*kvDim*bf16Size)
		vC[l] = make([]byte, maxLen*kvDim*bf16Size)
	}
	ref := make([][]byte, T)
	for tok := 0; tok < T; tok++ {
		x := inputs[tok]
		for l := 0; l < nLayers; l++ {
			w := layers[l]
			var err error
			x, err = DecodeStepKV(x, w.AttnNormW, w.WQ, w.WK, w.WV, w.WO, kC[l], vC[l], w.MLPNormW, w.WGate, w.WUp, w.WDown, dModel, nHeads, nKV, headDim, maxLen, dFF, tok, base, scale, eps)
			if err != nil {
				t.Fatalf("DecodeStepKV ref t=%d l=%d: %v", tok, l, err)
			}
		}
		ref[tok] = x
	}

	got, err := DecodeForward(inputs, layers, dModel, nHeads, nKV, headDim, maxLen, dFF, base, scale, eps)
	if err != nil {
		t.Fatalf("DecodeForward: %v", err)
	}
	if len(got) != T {
		t.Fatalf("DecodeForward returned %d outputs, want %d", len(got), T)
	}
	for tok := 0; tok < T; tok++ {
		eqBytes(t, "DecodeForward token", got[tok], ref[tok])
	}
	t.Logf("DecodeForward(%d layers × %d tokens, GQA %d/%d, growing cache): byte-identical to stepped DecodeStepKV", nLayers, T, nHeads, nKV)
}

// TestDecodeForwardHostCost measures the real forward's per-token wall as the KV
// cache grows. The per-token host encode is a fixed op count regardless of window
// length (N layers × the same ops), so at these synthetic dims — where GPU work is
// tiny — the per-token cost stays ~flat as the cache fills, the structural reason
// the encode-bypass (single-submit per-token ICB) pays off: constant host work per
// token, flat memory pressure, no per-token sawtooth. Shared weights keep it
// AX-11-light; this is host-cost at synthetic dims, NOT real-model tok/s.
func TestDecodeForwardHostCost(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF = 512, 8, 4, 64, 1024
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	const nLayers = 24

	w := forwardLayer(dModel, nHeads, nKV, headDim, dFF, 100)
	layers := make([]DecodeLayerWeights, nLayers)
	for l := range layers {
		layers[l] = w // shared weights: host encode cost is bind-count, not which buffer
	}
	mkInputs := func(T int) [][]byte {
		in := make([][]byte, T)
		for i := range in {
			f := make([]float32, dModel)
			for j := range f {
				f[j] = float32((j*(i+3)+5)%97-48) * 0.02
			}
			in[i] = toBF16Bytes(f)
		}
		return in
	}
	// warm
	if _, err := DecodeForward(mkInputs(4), layers, dModel, nHeads, nKV, headDim, 4, dFF, base, scale, eps); err != nil {
		t.Fatalf("warm: %v", err)
	}
	for _, T := range []int{8, 16, 32} {
		inputs := mkInputs(T)
		t0 := time.Now()
		if _, err := DecodeForward(inputs, layers, dModel, nHeads, nKV, headDim, T, dFF, base, scale, eps); err != nil {
			t.Fatalf("DecodeForward T=%d: %v", T, err)
		}
		d := time.Since(t0)
		t.Logf("%2d-layer forward, %2d tokens (cache 1..%d): %.2f ms total, %6.1f µs/token",
			nLayers, T, T, float64(d.Microseconds())/1000, float64(d.Microseconds())/float64(T))
	}
}
