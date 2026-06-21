// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import (
	"bytes"
	"testing"

	"dappco.re/go/mlx/pkg/model"
)

// TestAssembleMatchesEngine is the GUARD for the model.Assemble extraction (engine 2/6): the engine's
// generic assembler, driven by StandardWeightNames, must produce a LoadedModel byte-identical to
// gemma4's own Assemble for the same tensors. The metal-vs-neutral PARSE parity test does NOT cover
// assembly (it stays green whatever Assemble does) — this is the discriminating check. Synthetic
// tensors, no model load (AX-11). Dense bf16 here; the PLE/MoE/quant name sets are guarded by the
// native gate once model.Load wires the real path onto model.Assemble (engine 3).
func TestAssembleMatchesEngine(t *testing.T) {
	arch, err := Config{
		HiddenSize: 64, NumHiddenLayers: 2, IntermediateSize: 128,
		NumAttentionHeads: 2, NumKeyValueHeads: 1, HeadDim: 16, VocabSize: 32, RMSNormEps: 1e-6,
	}.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	ts := minimalGemma4Tensors(arch)

	want, err := Assemble(ts, arch) // gemma4's own assembler
	if err != nil {
		t.Fatalf("gemma4.Assemble: %v", err)
	}
	got, err := model.Assemble(ts, arch, model.StandardWeightNames()) // the engine's generic assembler
	if err != nil {
		t.Fatalf("model.Assemble: %v", err)
	}

	eqLin := func(name string, a, b *model.Linear) {
		t.Helper()
		if (a == nil) != (b == nil) {
			t.Fatalf("%s: nil mismatch (gemma4 nil=%v, engine nil=%v)", name, a == nil, b == nil)
		}
		if a == nil {
			return
		}
		if !bytes.Equal(a.Weight, b.Weight) || !bytes.Equal(a.Scales, b.Scales) || !bytes.Equal(a.Biases, b.Biases) {
			t.Fatalf("%s: Linear bytes differ — a weight-name mismatch in StandardWeightNames", name)
		}
		if a.OutDim != b.OutDim {
			t.Fatalf("%s: OutDim differ (%d vs %d)", name, a.OutDim, b.OutDim)
		}
	}
	eqB := func(name string, a, b []byte) {
		t.Helper()
		if !bytes.Equal(a, b) {
			t.Fatalf("%s: norm bytes differ — a weight-name mismatch", name)
		}
	}

	eqLin("Embed", want.Embed, got.Embed)
	eqLin("LMHead", want.LMHead, got.LMHead)
	eqB("FinalNorm", want.FinalNorm, got.FinalNorm)
	eqLin("EmbedPerLayer", want.EmbedPerLayer, got.EmbedPerLayer)
	eqLin("PerLayerModelProj", want.PerLayerModelProj, got.PerLayerModelProj)
	eqB("PerLayerProjNorm", want.PerLayerProjNorm, got.PerLayerProjNorm)

	if len(want.Layers) != len(got.Layers) {
		t.Fatalf("layer count: gemma4 %d, engine %d", len(want.Layers), len(got.Layers))
	}
	for i := range want.Layers {
		w, g := &want.Layers[i], &got.Layers[i]
		eqB("AttnNorm", w.AttnNorm, g.AttnNorm)
		eqB("PostAttnNorm", w.PostAttnNorm, g.PostAttnNorm)
		eqB("QNorm", w.QNorm, g.QNorm)
		eqB("KNorm", w.KNorm, g.KNorm)
		eqB("LayerScalar", w.LayerScalar, g.LayerScalar)
		eqLin("Q", w.Q, g.Q)
		eqLin("K", w.K, g.K)
		eqLin("V", w.V, g.V)
		eqLin("O", w.O, g.O)
		eqB("MLPNorm", w.MLPNorm, g.MLPNorm)
		eqLin("Gate", w.Gate, g.Gate)
		eqLin("Up", w.Up, g.Up)
		eqLin("Down", w.Down, g.Down)
		eqB("PostFFNorm", w.PostFFNorm, g.PostFFNorm)
		eqLin("PerLayerGate", w.PerLayerGate, g.PerLayerGate)
		eqLin("PerLayerProjection", w.PerLayerProjection, g.PerLayerProjection)
		eqB("PostPerLayerInputNorm", w.PostPerLayerInputNorm, g.PostPerLayerInputNorm)
		if (w.MoE == nil) != (g.MoE == nil) {
			t.Fatalf("layer %d: MoE nil mismatch", i)
		}
	}
}
