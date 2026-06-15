// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/metal"
)

// loadTinyOrderedAssistant loads a tiny ORDERED-embedding assistant drafter
// (masked centroids + token ordering) through the real on-disk load path. The
// shared runtime helper builds a non-ordered assistant, so the
// ordered-embedding dense-logit scatter has no other in-package driver.
func loadTinyOrderedAssistant(t *testing.T) *Gemma4AssistantModel {
	t.Helper()
	requireMetalRuntime(t)

	dir := t.TempDir()
	writeGemma4AssistantConfig(t, dir, true)
	writeMinimalTokenizer(t, dir)
	if err := metal.SaveSafetensors(core.JoinPath(dir, "model.safetensors"), gemma4AssistantTinyWeights(true)); err != nil {
		t.Fatalf("SaveSafetensors: %v", err)
	}
	model, err := LoadGemma4Assistant(dir)
	if err != nil {
		t.Fatalf("LoadGemma4Assistant(ordered): %v", err)
	}
	t.Cleanup(func() { model.Close() })
	return model
}

// TestAssistantOrdered_orderedEmbeddingDenseLogits_Good drives the
// ordered-embedding dense-logit path: it scatters each token's top-K candidate
// centroid logits into a [tokenCount, vocab] dense tensor (a -1e9 base with the
// candidate positions filled). The synthetic hidden state is [1, 1, HiddenSize]
// matching the tiny ordered assistant; the result must be a valid dense tensor
// whose row width equals the vocab size.
func TestAssistantOrdered_orderedEmbeddingDenseLogits_Good(t *testing.T) {
	model := loadTinyOrderedAssistant(t)
	if !model.UseOrderedEmbeddings || model.MaskedCentroids == nil || model.TokenOrdering == nil {
		t.Fatalf("ordered tensors not loaded: ordered=%v centroids=%v ordering=%v",
			model.UseOrderedEmbeddings, model.MaskedCentroids, model.TokenOrdering)
	}

	hidden := seqArray(0.05, 1, 1, int(model.Cfg.HiddenSize))
	defer metal.Free(hidden)

	dense, err := model.orderedEmbeddingDenseLogits(hidden)
	if err != nil {
		t.Fatalf("orderedEmbeddingDenseLogits: %v", err)
	}
	if dense == nil || !dense.Valid() {
		t.Fatal("orderedEmbeddingDenseLogits returned invalid dense logits")
	}
	defer metal.Free(dense)
	metal.Materialize(dense)

	shape := dense.Shape()
	if len(shape) != 2 || shape[0] != 1 || shape[1] != model.Cfg.VocabSize {
		t.Fatalf("dense logits shape = %v, want [1 %d]", shape, model.Cfg.VocabSize)
	}
	// The scatter starts from a -1e9 base and fills the candidate positions, so
	// at least one entry must exceed the floor (a candidate was selected).
	floats := dense.Floats()
	maxV := floats[0]
	for _, v := range floats[1:] {
		if v > maxV {
			maxV = v
		}
	}
	if maxV <= -1e8 {
		t.Fatalf("no candidate logit rose above the muted floor, max = %g", maxV)
	}
}

// TestAssistantOrdered_orderedEmbeddingDenseLogits_BadShape covers the rank/
// hidden-size guard: a hidden state whose last dim is not the model hidden size
// is rejected before any scatter.
func TestAssistantOrdered_orderedEmbeddingDenseLogits_BadShape(t *testing.T) {
	model := loadTinyOrderedAssistant(t)

	bad := seqArray(0.05, 1, 1, int(model.Cfg.HiddenSize)+1)
	defer metal.Free(bad)

	if _, err := model.orderedEmbeddingDenseLogits(bad); err == nil {
		t.Fatal("orderedEmbeddingDenseLogits(bad shape) error = nil, want hidden-shape rejection")
	}
}
