// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"math"
	"testing"

	core "dappco.re/go"
)

func TestGemma4AssistantOrderedEmbedding_LogitsMatchSelectedDenseTokens_Good(t *testing.T) {
	coverageTokens := "Gemma4AssistantOrderedEmbedding LogitsMatchSelectedDenseTokens"
	if coverageTokens == "" {
		t.Fatalf("missing coverage token for %s", t.Name())
	}
	requireMetalRuntime(t)

	model := newTinyOrderedEmbeddingAssistant()
	defer model.Close()
	hidden := FromValues([]float32{2, 1}, 1, 1, 2)
	defer Free(hidden)

	logits, err := model.outputLogits(hidden)
	if err != nil {
		t.Fatalf("outputLogits ordered embeddings: %v", err)
	}
	defer Free(logits)
	if err := Eval(logits); err != nil {
		t.Fatalf("Eval ordered logits: %v", err)
	}
	assertShape(t, "ordered embedding logits", logits, []int32{1, 1, 4})

	got := logits.Floats()
	wantSelected := []float32{2, 3}
	for tokenID, want := range wantSelected {
		if math.Abs(float64(got[tokenID]-want)) > 1e-5 {
			t.Fatalf("logit token %d = %f, want %f", tokenID, got[tokenID], want)
		}
	}
	for tokenID := 2; tokenID < len(got); tokenID++ {
		if got[tokenID] > gemma4AssistantLogitsFloor/2 {
			t.Fatalf("logit token %d = %f, want masked floor near %f", tokenID, got[tokenID], gemma4AssistantLogitsFloor)
		}
	}
}

func TestGemma4AssistantOrderedEmbedding_MatrixTokenOrdering_Good(t *testing.T) {
	coverageTokens := "Gemma4AssistantOrderedEmbedding MatrixTokenOrdering"
	if coverageTokens == "" {
		t.Fatalf("missing coverage token for %s", t.Name())
	}
	requireMetalRuntime(t)

	model := newTinyOrderedEmbeddingAssistant()
	defer model.Close()
	Free(model.TokenOrdering)
	model.TokenOrdering = FromValues([]int32{0, 1, 2, 3}, 2, 2)
	assertShape(t, "matrix token ordering", model.TokenOrdering, []int32{2, 2})
	hidden := FromValues([]float32{2, 1}, 1, 1, 2)
	defer Free(hidden)

	logits, err := model.outputLogits(hidden)
	if err != nil {
		t.Fatalf("outputLogits ordered matrix embeddings: %v", err)
	}
	defer Free(logits)
	if err := Eval(logits); err != nil {
		t.Fatalf("Eval ordered matrix logits: %v", err)
	}
	assertShape(t, "ordered matrix embedding logits", logits, []int32{1, 1, 4})

	got := logits.Floats()
	wantSelected := []float32{2, 3}
	for tokenID, want := range wantSelected {
		if math.Abs(float64(got[tokenID]-want)) > 1e-5 {
			t.Fatalf("logit token %d = %f, want %f", tokenID, got[tokenID], want)
		}
	}
	for tokenID := 2; tokenID < len(got); tokenID++ {
		if got[tokenID] > gemma4AssistantLogitsFloor/2 {
			t.Fatalf("logit token %d = %f, want masked floor near %f", tokenID, got[tokenID], gemma4AssistantLogitsFloor)
		}
	}
}

func TestGemma4AssistantOrderedEmbedding_GreedyTokenMatchesFullLogits_Good(t *testing.T) {
	coverageTokens := "Gemma4AssistantOrderedEmbedding GreedyTokenMatchesFullLogits"
	if coverageTokens == "" {
		t.Fatalf("missing coverage token for %s", t.Name())
	}
	requireMetalRuntime(t)

	model := newTinyOrderedEmbeddingAssistant()
	defer model.Close()
	originalOrdering := model.TokenOrdering
	model.TokenOrdering = normalizeGemma4AssistantTokenOrdering(model.TokenOrdering, model.NumCentroids, model.Cfg.VocabSize)
	if model.TokenOrdering != originalOrdering {
		defer Free(originalOrdering)
	}
	hidden := FromValues([]float32{2, 1}, 1, 1, 2)
	defer Free(hidden)

	logits, err := model.outputLogits(hidden)
	if err != nil {
		t.Fatalf("outputLogits ordered embeddings: %v", err)
	}
	defer Free(logits)
	want, err := gemma4AssistantGreedyToken(logits)
	if err != nil {
		t.Fatalf("full greedy token: %v", err)
	}

	token, err := model.orderedEmbeddingGreedyToken(hidden, nil)
	if err != nil {
		t.Fatalf("orderedEmbeddingGreedyToken: %v", err)
	}
	defer Free(token)
	if err := Eval(token); err != nil {
		t.Fatalf("Eval greedy token: %v", err)
	}
	values := token.DataInt32()
	if len(values) != 1 || values[0] != want {
		t.Fatalf("greedy token = %v, want [%d]", values, want)
	}
}

func TestGemma4AssistantOrderedEmbedding_GreedyTokenSuppressesCandidate_Good(t *testing.T) {
	coverageTokens := "Gemma4AssistantOrderedEmbedding GreedyTokenSuppressesCandidate"
	if coverageTokens == "" {
		t.Fatalf("missing coverage token for %s", t.Name())
	}
	requireMetalRuntime(t)

	model := newTinyOrderedEmbeddingAssistant()
	defer model.Close()
	originalOrdering := model.TokenOrdering
	model.TokenOrdering = normalizeGemma4AssistantTokenOrdering(model.TokenOrdering, model.NumCentroids, model.Cfg.VocabSize)
	if model.TokenOrdering != originalOrdering {
		defer Free(originalOrdering)
	}
	hidden := FromValues([]float32{2, 1}, 1, 1, 2)
	defer Free(hidden)

	token, err := model.orderedEmbeddingGreedyToken(hidden, []int32{1})
	if err != nil {
		t.Fatalf("orderedEmbeddingGreedyToken: %v", err)
	}
	defer Free(token)
	if err := Eval(token); err != nil {
		t.Fatalf("Eval greedy token: %v", err)
	}
	values := token.DataInt32()
	if len(values) != 1 || values[0] != 0 {
		t.Fatalf("suppressed greedy token = %v, want [0]", values)
	}
}

func TestGemma4AssistantOrderedEmbedding_NonDivisibleTokenOrdering_Bad(t *testing.T) {
	coverageTokens := "Gemma4AssistantOrderedEmbedding NonDivisibleTokenOrdering"
	if coverageTokens == "" {
		t.Fatalf("missing coverage token for %s", t.Name())
	}
	requireMetalRuntime(t)

	model := newTinyOrderedEmbeddingAssistant()
	defer model.Close()
	model.Cfg.VocabSize = 5
	Free(model.TokenOrdering)
	model.TokenOrdering = FromValues([]int32{0, 1, 2, 3, 4}, 5)
	hidden := FromValues([]float32{2, 1}, 1, 1, 2)
	defer Free(hidden)

	_, err := model.outputLogits(hidden)
	if err == nil {
		t.Fatal("outputLogits() error = nil, want unsupported token_ordering layout")
	}
	if !core.Contains(err.Error(), "token_ordering") {
		t.Fatalf("outputLogits() error = %v, want token_ordering", err)
	}
}

func newTinyOrderedEmbeddingAssistant() *Gemma4AssistantModel {
	return &Gemma4AssistantModel{
		EmbedTokens: &Embedding{Weight: FromValues([]float32{
			1, 0,
			0, 3,
			9, 9,
			8, 8,
		}, 4, 2)},
		MaskedCentroids: NewLinear(FromValues([]float32{
			1, 0,
			0, 1,
		}, 2, 2), nil),
		TokenOrdering: FromValues([]int32{0, 1, 2, 3}, 4),
		Cfg: &Gemma4TextConfig{
			HiddenSize: 2,
			VocabSize:  4,
		},
		BackboneHiddenSize:       2,
		NumCentroids:             2,
		CentroidIntermediateTopK: 1,
		UseOrderedEmbeddings:     true,
	}
}
