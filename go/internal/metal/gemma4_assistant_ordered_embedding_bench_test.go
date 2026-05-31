// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

func BenchmarkGemma4AssistantOrderedEmbedding_FlatTokenOrdering(b *testing.B) {
	benchmarkGemma4AssistantOrderedEmbedding(b, false)
}

func BenchmarkGemma4AssistantOrderedEmbedding_MatrixTokenOrdering(b *testing.B) {
	benchmarkGemma4AssistantOrderedEmbedding(b, true)
}

func BenchmarkGemma4AssistantOrderedEmbedding_LoadNormalisedTokenOrdering(b *testing.B) {
	benchmarkGemma4AssistantOrderedEmbeddingLoadNormalised(b)
}

func benchmarkGemma4AssistantOrderedEmbedding(b *testing.B, matrixOrdering bool) {
	requireMetalRuntime(b)

	model := newTinyOrderedEmbeddingAssistant()
	defer model.Close()
	if matrixOrdering {
		Free(model.TokenOrdering)
		model.TokenOrdering = FromValues([]int32{0, 1, 2, 3}, 2, 2)
	}
	hidden := FromValues([]float32{2, 1}, 1, 1, 2)
	defer Free(hidden)

	warm, err := model.outputLogits(hidden)
	if err != nil {
		b.Fatalf("warmup outputLogits: %v", err)
	}
	if err := Eval(warm); err != nil {
		Free(warm)
		b.Fatalf("warmup Eval: %v", err)
	}
	Free(warm)

	b.ReportAllocs()
	for b.Loop() {
		logits, err := model.outputLogits(hidden)
		if err != nil {
			b.Fatalf("outputLogits: %v", err)
		}
		if err := Eval(logits); err != nil {
			Free(logits)
			b.Fatalf("Eval: %v", err)
		}
		Free(logits)
	}
}

func benchmarkGemma4AssistantOrderedEmbeddingLoadNormalised(b *testing.B) {
	requireMetalRuntime(b)

	model := newTinyOrderedEmbeddingAssistant()
	defer model.Close()
	originalOrdering := model.TokenOrdering
	model.TokenOrdering = normalizeGemma4AssistantTokenOrdering(model.TokenOrdering, model.NumCentroids, model.Cfg.VocabSize)
	if model.TokenOrdering != originalOrdering {
		defer Free(originalOrdering)
	}
	hidden := FromValues([]float32{2, 1}, 1, 1, 2)
	defer Free(hidden)

	warm, err := model.outputLogits(hidden)
	if err != nil {
		b.Fatalf("warmup outputLogits: %v", err)
	}
	if err := Eval(warm); err != nil {
		Free(warm)
		b.Fatalf("warmup Eval: %v", err)
	}
	Free(warm)

	b.ReportAllocs()
	for b.Loop() {
		logits, err := model.outputLogits(hidden)
		if err != nil {
			b.Fatalf("outputLogits: %v", err)
		}
		if err := Eval(logits); err != nil {
			Free(logits)
			b.Fatalf("Eval: %v", err)
		}
		Free(logits)
	}
}
