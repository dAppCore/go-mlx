// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"testing"

	"dappco.re/go/mlx/pkg/model/mistral"
)

// TestMistralLoadedModelMatchesAssemble proves the registry path is byte-identical to the hand-coded
// native assembler it replaces: pkg/model/mistral.Assemble → neutral model.LoadedModel → the generic
// loadedToBF16 must produce exactly the BF16Model the old AssembleMistralBF16 builds directly. This is
// what lets step 4 delete AssembleMistralBF16 (and mistral_*.go) and route Mistral through the registry
// with no behaviour change.
func TestMistralLoadedModelMatchesAssemble(t *testing.T) {
	const dModel, nHeads, nKV, headDim, dFF, vocab, nLayers = 64, 1, 1, 64, 128, 32, 2
	_, arch := mistralConfigFixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, nLayers)
	tensors := mistralTensorFixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, nLayers)

	want, err := AssembleMistralBF16(tensors, arch) // the hand-coded native path (to be deleted)
	if err != nil {
		t.Fatalf("AssembleMistralBF16: %v", err)
	}
	lm, err := mistral.Assemble(tensors, arch) // the registry path: tensor names → neutral LoadedModel
	if err != nil {
		t.Fatalf("mistral.Assemble: %v", err)
	}
	got := loadedToBF16(lm) // the generic backend conversion

	if got.Tied != want.Tied || !bytes.Equal(got.Embed, want.Embed) ||
		!bytes.Equal(got.FinalNorm, want.FinalNorm) || !bytes.Equal(got.LMHead, want.LMHead) {
		t.Fatal("model-level weights (Embed/FinalNorm/LMHead/Tied) differ between the two paths")
	}
	if len(got.Layers) != len(want.Layers) {
		t.Fatalf("layer count: got %d, want %d", len(got.Layers), len(want.Layers))
	}
	for i := range want.Layers {
		w, g := &want.Layers[i], &got.Layers[i]
		same := bytes.Equal(g.AttnNormW, w.AttnNormW) && bytes.Equal(g.MLPNormW, w.MLPNormW) &&
			bytes.Equal(g.WQ, w.WQ) && bytes.Equal(g.WK, w.WK) && bytes.Equal(g.WV, w.WV) && bytes.Equal(g.WO, w.WO) &&
			bytes.Equal(g.WGate, w.WGate) && bytes.Equal(g.WUp, w.WUp) && bytes.Equal(g.WDown, w.WDown)
		if !same {
			t.Fatalf("layer %d weights differ between AssembleMistralBF16 and mistral.Assemble→loadedToBF16", i)
		}
		if len(g.QNormW) != 0 || len(g.KNormW) != 0 || len(g.PostAttnNormW) != 0 || len(g.PostFFNormW) != 0 {
			t.Fatalf("layer %d: the mistral registry path populated a gemma4-only norm extra", i)
		}
	}
}
