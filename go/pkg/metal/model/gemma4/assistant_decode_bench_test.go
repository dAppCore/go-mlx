// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"testing"

	"dappco.re/go/mlx/pkg/metal"
)

func BenchmarkGemma4AssistantDecode_OrderedEmbeddingGreedyToken(b *testing.B) {
	requireMetalRuntime(b)

	pair := loadTinyGemma4AssistantPair(b, true)
	defer pair.Close()
	hidden := seqArray(0.07, 1, 1, int(pair.Assistant.Cfg.HiddenSize))
	defer metal.Free(hidden)
	if _, err := pair.Assistant.orderedEmbeddingGreedyToken(hidden, nil); err != nil {
		b.Fatalf("warm orderedEmbeddingGreedyToken: %v", err)
	}

	b.ReportAllocs()
	for b.Loop() {
		token, err := pair.Assistant.orderedEmbeddingGreedyToken(hidden, nil)
		if err != nil {
			b.Fatalf("orderedEmbeddingGreedyToken: %v", err)
		}
		if err := metal.Eval(token); err != nil {
			metal.Free(token)
			b.Fatalf("eval ordered token: %v", err)
		}
		metal.Free(token)
	}
}
