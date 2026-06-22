// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/model"
)

// TestStepTokensBatchedDense asserts the session-level MTP batched verify (K tokens through the whole
// resident layer stack in one pass) is BYTE-IDENTICAL to stepping the same K tokens one at a time with
// stepToken over the same growing cache. This is the bar that lets MTPDecode swap its sequential
// stepGreedy verify for one batched pass without changing the emitted token stream.
func TestStepTokensBatchedDense(t *testing.T) {
	requireNativeRuntime(t)
	const dModel, nHeads, nKV, headDim, dFF = 512, 8, 4, 64, 1024
	const base, scale, eps = float32(10000), float32(0.125), float32(1e-5)
	const nL, maxLen, prefix, K = 3, 32, 5, 4

	layers := make([]DecodeLayerWeights, nL)
	types := make([]string, nL)
	for li := range layers {
		layers[li] = forwardLayer(dModel, nHeads, nKV, headDim, dFF, (li+1)*100)
		types[li] = "full_attention"
	}
	specs := model.DeriveLayers(types, 0)

	emb := func(seed int) []byte {
		f := make([]float32, dModel)
		for j := range f {
			f[j] = float32((j*(seed+3)+5)%97-48) * 0.02
		}
		return toBF16Bytes(f)
	}
	embs := make([][]byte, prefix+K)
	for i := range embs {
		embs[i] = emb(i + 1)
	}

	build := func() *archDecodeState {
		lb, moe, err := buildBF16ArchLayerBufs(layers, specs, dModel, nHeads, nKV, headDim, dFF, maxLen, 0, nil)
		if err != nil {
			t.Fatalf("buildBF16ArchLayerBufs: %v", err)
		}
		st := newArchDecodeState(specs, lb, moe, dModel, nHeads, nKV, headDim, dFF, 0, headDim, headDim, base, base, scale, eps, false)
		return &st
	}

	// sequential reference: prefill the prefix, then step K tokens one at a time.
	var seqOut [][]byte
	withAutoreleasePool(func() {
		st := build()
		for i := 0; i < prefix; i++ {
			if _, err := st.stepToken(embs[i], i); err != nil {
				t.Fatalf("prefill stepToken %d: %v", i, err)
			}
		}
		for i := 0; i < K; i++ {
			h, err := st.stepToken(embs[prefix+i], prefix+i)
			if err != nil {
				t.Fatalf("seq stepToken %d: %v", prefix+i, err)
			}
			seqOut = append(seqOut, append([]byte(nil), h...))
		}
	})

	// batched: fresh state, same prefix, then ONE stepTokensBatchedDense over the K tokens.
	var batOut [][]byte
	var ok bool
	withAutoreleasePool(func() {
		st := build()
		for i := 0; i < prefix; i++ {
			if _, err := st.stepToken(embs[i], i); err != nil {
				t.Fatalf("batched prefill stepToken %d: %v", i, err)
			}
		}
		var err error
		batOut, ok, err = st.stepTokensBatchedDense(embs[prefix:prefix+K], prefix)
		if err != nil {
			t.Fatalf("stepTokensBatchedDense: %v", err)
		}
	})
	if !ok {
		t.Fatal("stepTokensBatchedDense reported !ok for a dense full-attention arch")
	}
	if len(batOut) != K {
		t.Fatalf("batched returned %d rows, want %d", len(batOut), K)
	}
	for i := 0; i < K; i++ {
		eqBytes(t, core.Sprintf("batched session row %d vs stepToken", i), batOut[i], seqOut[i])
	}
}
