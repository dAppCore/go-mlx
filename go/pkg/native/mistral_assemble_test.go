// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func TestAssembleMistralBF16MapsDenseTiedWeights(t *testing.T) {
	const dModel, nHeads, nKV, headDim, dFF, vocab, nLayers = 64, 1, 1, 64, 128, 32, 2
	_, arch := mistralConfigFixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, nLayers)
	tensors := mistralTensorFixture(t, dModel, nHeads, nKV, headDim, dFF, vocab, nLayers)

	g, err := AssembleMistralBF16(tensors, arch)
	if err != nil {
		t.Fatalf("AssembleMistralBF16: %v", err)
	}
	if !g.Tied || len(g.LMHead) != len(g.Embed) {
		t.Fatal("Mistral assembly should tie LMHead to Embed when lm_head.weight is absent")
	}
	if len(g.Layers) != nLayers {
		t.Fatalf("layers = %d, want %d", len(g.Layers), nLayers)
	}
	l := g.Layers[0]
	if len(l.AttnNormW) != dModel*bf16Size || len(l.MLPNormW) != dModel*bf16Size {
		t.Fatal("Mistral layer norms were not mapped to AttnNormW and MLPNormW")
	}
	if len(l.QNormW) != 0 || len(l.KNormW) != 0 || len(l.PostAttnNormW) != 0 || len(l.PostFFNormW) != 0 {
		t.Fatal("Mistral assembly should not populate Gemma4-only norm extras")
	}

	delete(tensors, "language_model.model.layers.0.self_attn.q_proj.weight")
	if _, err := AssembleMistralBF16(tensors, arch); err == nil {
		t.Fatal("missing required q_proj should fail")
	}
}
