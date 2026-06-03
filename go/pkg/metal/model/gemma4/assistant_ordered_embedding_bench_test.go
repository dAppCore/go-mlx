// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"testing"

	"dappco.re/go/mlx/pkg/metal"
)

func BenchmarkGemma4AssistantOrderedEmbedding_FlatTokenOrdering(b *testing.B) {
	benchmarkGemma4AssistantOrderedEmbedding(b, false)
}

func BenchmarkGemma4AssistantOrderedEmbedding_MatrixTokenOrdering(b *testing.B) {
	benchmarkGemma4AssistantOrderedEmbedding(b, true)
}

func BenchmarkGemma4AssistantOrderedEmbedding_LoadNormalisedTokenOrdering(b *testing.B) {
	benchmarkGemma4AssistantOrderedEmbeddingLoadNormalised(b)
}

func BenchmarkGemma4AssistantOrderedEmbedding_GreedyToken(b *testing.B) {
	benchmarkGemma4AssistantOrderedEmbeddingGreedyToken(b, nil)
}

func BenchmarkGemma4AssistantOrderedEmbedding_GreedyTokenSuppressed(b *testing.B) {
	benchmarkGemma4AssistantOrderedEmbeddingGreedyToken(b, []int32{1})
}

func benchmarkGemma4AssistantOrderedEmbedding(b *testing.B, matrixOrdering bool) {
	requireMetalRuntime(b)

	model := newTinyOrderedEmbeddingAssistant()
	defer model.Close()
	if matrixOrdering {
		metal.Free(model.TokenOrdering)
		model.TokenOrdering = metal.FromValues([]int32{0, 1, 2, 3}, 2, 2)
	}
	hidden := metal.FromValues([]float32{2, 1}, 1, 1, 2)
	defer metal.Free(hidden)

	warm, err := model.outputLogits(hidden)
	if err != nil {
		b.Fatalf("warmup outputLogits: %v", err)
	}
	if err := metal.Eval(warm); err != nil {
		metal.Free(warm)
		b.Fatalf("warmup Eval: %v", err)
	}
	metal.Free(warm)

	b.ReportAllocs()
	for b.Loop() {
		logits, err := model.outputLogits(hidden)
		if err != nil {
			b.Fatalf("outputLogits: %v", err)
		}
		if err := metal.Eval(logits); err != nil {
			metal.Free(logits)
			b.Fatalf("Eval: %v", err)
		}
		metal.Free(logits)
	}
}

func benchmarkGemma4AssistantOrderedEmbeddingGreedyToken(b *testing.B, suppressTokens []int32) {
	requireMetalRuntime(b)

	model := newTinyOrderedEmbeddingAssistant()
	defer model.Close()
	originalOrdering := model.TokenOrdering
	model.TokenOrdering = normalizeGemma4AssistantTokenOrdering(model.TokenOrdering, model.NumCentroids, model.Cfg.VocabSize)
	if model.TokenOrdering != originalOrdering {
		defer metal.Free(originalOrdering)
	}
	hidden := metal.FromValues([]float32{2, 1}, 1, 1, 2)
	defer metal.Free(hidden)

	warm, err := model.orderedEmbeddingGreedyToken(hidden, suppressTokens)
	if err != nil {
		b.Fatalf("warmup orderedEmbeddingGreedyToken: %v", err)
	}
	if err := metal.Eval(warm); err != nil {
		metal.Free(warm)
		b.Fatalf("warmup Eval: %v", err)
	}
	metal.Free(warm)

	b.ReportAllocs()
	for b.Loop() {
		token, err := model.orderedEmbeddingGreedyToken(hidden, suppressTokens)
		if err != nil {
			b.Fatalf("orderedEmbeddingGreedyToken: %v", err)
		}
		if err := metal.Eval(token); err != nil {
			metal.Free(token)
			b.Fatalf("Eval: %v", err)
		}
		metal.Free(token)
	}
}

func benchmarkGemma4AssistantOrderedEmbeddingLoadNormalised(b *testing.B) {
	requireMetalRuntime(b)

	model := newTinyOrderedEmbeddingAssistant()
	defer model.Close()
	originalOrdering := model.TokenOrdering
	model.TokenOrdering = normalizeGemma4AssistantTokenOrdering(model.TokenOrdering, model.NumCentroids, model.Cfg.VocabSize)
	if model.TokenOrdering != originalOrdering {
		defer metal.Free(originalOrdering)
	}
	hidden := metal.FromValues([]float32{2, 1}, 1, 1, 2)
	defer metal.Free(hidden)

	warm, err := model.outputLogits(hidden)
	if err != nil {
		b.Fatalf("warmup outputLogits: %v", err)
	}
	if err := metal.Eval(warm); err != nil {
		metal.Free(warm)
		b.Fatalf("warmup Eval: %v", err)
	}
	metal.Free(warm)

	b.ReportAllocs()
	for b.Loop() {
		logits, err := model.outputLogits(hidden)
		if err != nil {
			b.Fatalf("outputLogits: %v", err)
		}
		if err := metal.Eval(logits); err != nil {
			metal.Free(logits)
			b.Fatalf("Eval: %v", err)
		}
		metal.Free(logits)
	}
}
