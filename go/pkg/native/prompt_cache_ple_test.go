// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"

	"dappco.re/go/mlx/pkg/model"
	g4 "dappco.re/go/mlx/pkg/model/gemma4"
)

func newPromptCachePLEFixture(t testing.TB) *ArchSession {
	t.Helper()
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 32
	const numLayers, pliDim, gs, bits = 2, 64, 64, 4
	const maxLen = 16
	cfg := g4.Config{
		HiddenSize: dModel, NumHiddenLayers: numLayers, IntermediateSize: dFF,
		NumAttentionHeads: nHeads, NumKeyValueHeads: nKV, HeadDim: headDim, VocabSize: vocab, RMSNormEps: 1e-6,
		HiddenSizePerLayerInput: pliDim, VocabSizePerLayerInput: vocab,
		Quantization: &model.QuantConfig{GroupSize: gs, Bits: bits},
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	ts := quantGemma4Tensors(t, arch, gs, bits)
	addPLETensors(t, ts, arch, gs, bits)
	lm, err := model.Assemble(ts, arch, model.StandardWeightNames())
	if err != nil {
		t.Fatalf("model.Assemble: %v", err)
	}
	g, err := loadedToQuant(lm, gs, bits)
	if err != nil {
		t.Fatalf("loadedToQuant: %v", err)
	}
	if !g.HasPLE() {
		t.Fatal("fixture model should have PLE")
	}
	sess, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchQuantSession: %v", err)
	}
	return sess
}

func TestWarmPromptCachePLESequentialAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)
	sess := newPromptCachePLEFixture(t)
	prefix := []int32{1, 5, 3, 7}
	if err := sess.WarmPromptCache(prefix); err != nil {
		t.Fatalf("WarmPromptCache warmup: %v", err)
	}

	var warmErr error
	allocs := testing.AllocsPerRun(3, func() {
		sess.pos = 0
		sess.cachedIDs = sess.cachedIDs[:0]
		warmErr = sess.WarmPromptCache(prefix)
	})
	if warmErr != nil {
		t.Fatalf("WarmPromptCache: %v", warmErr)
	}
	if allocs > 5000 {
		t.Fatalf("PLE WarmPromptCache allocations = %.0f, want <= 5000", allocs)
	}
}
