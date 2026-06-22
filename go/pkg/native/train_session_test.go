// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"

	"dappco.re/go/mlx/pkg/model"
)

// TestForwardCaptureHiddens verifies the activation-saving forward on a real (synthetic) dense
// ArchSession: it returns one residual-stream tensor per layer, and the final layer's last-token hidden
// is BYTE-IDENTICAL to the session's ordinary forward (so saving activations doesn't perturb the
// engine's result — the captured hiddens are the real layer outputs the backward will use).
func TestForwardCaptureHiddens(t *testing.T) {
	requireNativeRuntime(t)
	const dModel, nHeads, nKV, headDim, dFF = 512, 8, 4, 64, 1024
	const vocab, nL, maxLen = 64, 3, 64
	layers := make([]DecodeLayerWeights, nL)
	types := make([]string, nL)
	for li := range layers {
		layers[li] = forwardLayer(dModel, nHeads, nKV, headDim, dFF, (li+1)*100)
		types[li] = "full_attention"
	}
	specs := model.DeriveLayers(types, 0)
	embed := toBF16Bytes(syntheticFloat32(vocab*dModel, 21))
	g := &BF16Model{Layers: layers, Embed: embed, FinalNorm: toBF16Bytes(syntheticFloat32(dModel, 22)), LMHead: embed, Tied: true}
	arch := model.Arch{
		Hidden: dModel, Heads: nHeads, KVHeads: nKV, HeadDim: headDim, FF: dFF, Vocab: vocab,
		GlobalHeadDim: headDim, GlobalKVHeads: nKV,
		Eps: 1e-5, AttnScale: 0.125, RopeBase: 10000, RopeScale: 1, RopeLocalBase: 10000,
		RotaryDim: headDim, RotaryDimLocal: headDim, Layer: specs,
	}
	mk := func() *ArchSession {
		s, err := NewArchSession(g, arch, maxLen)
		if err != nil {
			t.Fatalf("NewArchSession: %v", err)
		}
		return s
	}
	ids := []int32{1, 2, 3, 4, 5}
	T, rowBytes := len(ids), dModel*bf16Size

	embeds, perLayer, err := mk().ForwardCaptureHiddens(ids)
	if err != nil {
		t.Fatalf("ForwardCaptureHiddens: %v", err)
	}
	if len(embeds) != T {
		t.Fatalf("got %d embeddings, want %d", len(embeds), T)
	}
	if len(perLayer) != nL {
		t.Fatalf("got %d per-layer tensors, want %d", len(perLayer), nL)
	}
	for l := range perLayer {
		if len(perLayer[l]) != T*rowBytes {
			t.Fatalf("perLayer[%d] is %d bytes, want %d", l, len(perLayer[l]), T*rowBytes)
		}
	}

	// the final layer's last-token hidden must equal the ordinary forward's last hidden (capture is faithful).
	ref := mk()
	var lastHidden []byte
	for _, id := range ids {
		h, e := ref.stepID(id)
		if e != nil {
			t.Fatalf("ref stepID: %v", e)
		}
		lastHidden = h
	}
	gotLast := perLayer[nL-1][(T-1)*rowBytes:]
	eqBytes(t, "captured final-layer last-token hidden vs ordinary forward", gotLast, lastHidden)
	t.Logf("activation-saving forward faithful: %d layers × %d tokens captured, final hidden byte-identical to the plain forward", nL, T)
}
