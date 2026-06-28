// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"

	"dappco.re/go/mlx/pkg/model"
)

var sessionStateBlockBytesSink int

func BenchmarkSessionStateSerializeCachedPrefix(b *testing.B) {
	requireNativeRuntime(b)
	s := newSessionStateFixture(b)
	if _, err := s.GenerateCached([]int32{1, 2, 3, 4, 5}, 6, -1); err != nil {
		b.Fatalf("GenerateCached warmup: %v", err)
	}
	blob, err := s.SerializeState()
	if err != nil {
		b.Fatalf("SerializeState warmup: %v", err)
	}
	b.SetBytes(int64(len(blob)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := s.SerializeState(); err != nil {
			b.Fatalf("SerializeState: %v", err)
		}
	}
}

func BenchmarkSessionStateRangeBlocksCachedPrefix(b *testing.B) {
	requireNativeRuntime(b)
	s := newSessionStateFixture(b)
	if _, err := s.GenerateCached([]int32{1, 2, 3, 4, 5}, 6, -1); err != nil {
		b.Fatalf("GenerateCached warmup: %v", err)
	}
	blob, err := s.SerializeState()
	if err != nil {
		b.Fatalf("SerializeState warmup: %v", err)
	}
	b.SetBytes(int64(len(blob)))
	b.ReportAllocs()
	if err := s.RangeStateBlocks(2, func(SessionStateBlock) (bool, error) {
		return true, nil
	}); err != nil {
		b.Fatalf("RangeStateBlocks warmup: %v", err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		total := 0
		if err := s.RangeStateBlocks(2, func(block SessionStateBlock) (bool, error) {
			for _, layer := range block.Layers {
				total += len(layer.KeyBytes) + len(layer.ValueBytes)
			}
			return true, nil
		}); err != nil {
			b.Fatalf("RangeStateBlocks: %v", err)
		}
		sessionStateBlockBytesSink = total
	}
}

func BenchmarkSessionStateRangeBlocksTrustedPrefix(b *testing.B) {
	requireNativeRuntime(b)
	s := newSessionStateFixture(b)
	if _, err := s.GenerateCached([]int32{1, 2, 3, 4, 5}, 6, -1); err != nil {
		b.Fatalf("GenerateCached warmup: %v", err)
	}
	blob, err := s.SerializeState()
	if err != nil {
		b.Fatalf("SerializeState warmup: %v", err)
	}
	b.SetBytes(int64(len(blob)))
	b.ReportAllocs()
	if err := s.RangeStateBlocksFrom(4, 2, func(SessionStateBlock) (bool, error) {
		return true, nil
	}); err != nil {
		b.Fatalf("RangeStateBlocksFrom warmup: %v", err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		total := 0
		if err := s.RangeStateBlocksFrom(4, 2, func(block SessionStateBlock) (bool, error) {
			for _, layer := range block.Layers {
				total += len(layer.KeyBytes) + len(layer.ValueBytes)
			}
			return true, nil
		}); err != nil {
			b.Fatalf("RangeStateBlocksFrom: %v", err)
		}
		sessionStateBlockBytesSink = total
	}
}

func BenchmarkSessionStateRestorePromptCacheEntry(b *testing.B) {
	requireNativeRuntime(b)
	saved := newSessionStateFixture(b)
	prompt := []int32{1, 2, 3, 4, 5}
	if err := saved.WarmPromptCache(prompt); err != nil {
		b.Fatalf("WarmPromptCache: %v", err)
	}
	blob, err := saved.SerializeState()
	if err != nil {
		b.Fatalf("SerializeState: %v", err)
	}
	restored := newSessionStateFixture(b)
	if err := restored.RestoreState(blob); err != nil {
		b.Fatalf("RestoreState warmup: %v", err)
	}
	if hit := restored.CachedPrefixLen(prompt); hit != len(prompt) {
		b.Fatalf("restored prompt-cache hit = %d, want %d", hit, len(prompt))
	}
	b.SetBytes(int64(len(blob)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := restored.RestoreState(blob); err != nil {
			b.Fatalf("RestoreState: %v", err)
		}
	}
}

func BenchmarkSessionStateRestoreBlocksPromptCacheEntry(b *testing.B) {
	requireNativeRuntime(b)
	saved := newSessionStateFixture(b)
	prompt := []int32{1, 2, 3, 4, 5}
	if err := saved.WarmPromptCache(prompt); err != nil {
		b.Fatalf("WarmPromptCache: %v", err)
	}
	blob, err := saved.SerializeState()
	if err != nil {
		b.Fatalf("SerializeState: %v", err)
	}
	source, err := saved.StateBlockSource(2)
	if err != nil {
		b.Fatalf("StateBlockSource: %v", err)
	}
	restored := newSessionStateFixture(b)
	if err := restored.RestoreStateBlocks(source); err != nil {
		b.Fatalf("RestoreStateBlocks warmup: %v", err)
	}
	if hit := restored.CachedPrefixLen(prompt); hit != len(prompt) {
		b.Fatalf("restored prompt-cache hit = %d, want %d", hit, len(prompt))
	}
	b.SetBytes(int64(len(blob)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := restored.RestoreStateBlocks(source); err != nil {
			b.Fatalf("RestoreStateBlocks: %v", err)
		}
	}
}

func BenchmarkSessionStateRestoreBlocksTrustedPrefix(b *testing.B) {
	requireNativeRuntime(b)
	prefix := []int32{1, 2, 3, 4}
	suffix := []int32{5, 6, 7}
	prompt := append(append([]int32(nil), prefix...), suffix...)

	saved := newSessionStateFixture(b)
	if err := saved.PrefillTokens(prompt); err != nil {
		b.Fatalf("PrefillTokens full prompt: %v", err)
	}
	source, err := saved.StateBlockSourceFrom(len(prefix), 2)
	if err != nil {
		b.Fatalf("StateBlockSourceFrom: %v", err)
	}
	blockBytes := 0
	for i := 0; i < source.BlockCount; i++ {
		block, err := source.Load(i)
		if err != nil {
			b.Fatalf("source.Load(%d): %v", i, err)
		}
		for _, layer := range block.Layers {
			blockBytes += len(layer.KeyBytes) + len(layer.ValueBytes)
		}
	}
	restored := newSessionStateFixture(b)
	if err := restored.PrefillTokens(prefix); err != nil {
		b.Fatalf("PrefillTokens prefix: %v", err)
	}
	if err := restored.RestoreStateBlocks(source); err != nil {
		b.Fatalf("RestoreStateBlocks warmup: %v", err)
	}
	b.SetBytes(int64(blockBytes))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := restored.RestoreStateBlocks(source); err != nil {
			b.Fatalf("RestoreStateBlocks trusted prefix: %v", err)
		}
	}
}

func BenchmarkSessionStateRestoreBlocksGenerateFromBoundaryLogits(b *testing.B) {
	requireNativeRuntime(b)
	prompt := []int32{1, 2, 3, 4, 5}

	saved := newSessionStateFixture(b)
	if err := saved.PrefillTokens(prompt); err != nil {
		b.Fatalf("PrefillTokens: %v", err)
	}
	logits, err := saved.BoundaryLogits()
	if err != nil {
		b.Fatalf("BoundaryLogits: %v", err)
	}
	source, err := saved.StateBlockSource(2)
	if err != nil {
		b.Fatalf("StateBlockSource: %v", err)
	}
	source.RetainedLogits = nil
	blockBytes := 0
	for i := 0; i < source.BlockCount; i++ {
		block, err := source.Load(i)
		if err != nil {
			b.Fatalf("source.Load(%d): %v", i, err)
		}
		for _, layer := range block.Layers {
			blockBytes += len(layer.KeyBytes) + len(layer.ValueBytes)
		}
	}
	restored := newSessionStateFixture(b)
	if err := restored.RestoreStateBlocks(source); err != nil {
		b.Fatalf("RestoreStateBlocks warmup: %v", err)
	}
	restored.resetRetainedHidden()
	if _, err := restored.GenerateFromCacheLogitsEach(logits, 1, -1, nil); err != nil {
		b.Fatalf("GenerateFromCacheLogitsEach warmup: %v", err)
	}
	b.SetBytes(int64(blockBytes))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := restored.RestoreStateBlocks(source); err != nil {
			b.Fatalf("RestoreStateBlocks: %v", err)
		}
		restored.resetRetainedHidden()
		if _, err := restored.GenerateFromCacheLogitsEach(logits, 1, -1, nil); err != nil {
			b.Fatalf("GenerateFromCacheLogitsEach: %v", err)
		}
	}
}

func BenchmarkSessionStateRestoreBlocksGenerateSampledFromRetainedLogits(b *testing.B) {
	requireNativeRuntime(b)
	prompt := []int32{1, 2, 3, 4, 5}
	params := model.SampleParams{Temperature: 0.8, TopK: 4, TopP: 0.9}
	stopTokens := []int32{63}

	saved := newSessionStateFixture(b)
	if err := saved.PrefillTokens(prompt); err != nil {
		b.Fatalf("PrefillTokens: %v", err)
	}
	source, err := saved.StateBlockSource(2)
	if err != nil {
		b.Fatalf("StateBlockSource: %v", err)
	}
	if len(source.RetainedLogits) == 0 {
		b.Fatal("StateBlockSource did not carry retained boundary logits")
	}
	blockBytes := 0
	for i := 0; i < source.BlockCount; i++ {
		block, err := source.Load(i)
		if err != nil {
			b.Fatalf("source.Load(%d): %v", i, err)
		}
		for _, layer := range block.Layers {
			blockBytes += len(layer.KeyBytes) + len(layer.ValueBytes)
		}
	}
	restored := newSessionStateFixture(b)
	if err := restored.RestoreStateBlocks(source); err != nil {
		b.Fatalf("RestoreStateBlocks warmup: %v", err)
	}
	if _, err := restored.GenerateSampledFromCacheEach(1, stopTokens, model.NewSampler(1), params, nil, nil); err != nil {
		b.Fatalf("GenerateSampledFromCacheEach warmup: %v", err)
	}
	b.SetBytes(int64(blockBytes))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := restored.RestoreStateBlocks(source); err != nil {
			b.Fatalf("RestoreStateBlocks: %v", err)
		}
		if _, err := restored.GenerateSampledFromCacheEach(1, stopTokens, model.NewSampler(uint64(i+1)), params, nil, nil); err != nil {
			b.Fatalf("GenerateSampledFromCacheEach: %v", err)
		}
	}
}

func BenchmarkSessionStateRestoreBlocksGenerateSampledFromBoundaryLogits(b *testing.B) {
	requireNativeRuntime(b)
	prompt := []int32{1, 2, 3, 4, 5}
	params := model.SampleParams{Temperature: 0.8, TopK: 4, TopP: 0.9}
	stopTokens := []int32{63}

	saved := newSessionStateFixture(b)
	if err := saved.PrefillTokens(prompt); err != nil {
		b.Fatalf("PrefillTokens: %v", err)
	}
	logits, err := saved.BoundaryLogits()
	if err != nil {
		b.Fatalf("BoundaryLogits: %v", err)
	}
	source, err := saved.StateBlockSource(2)
	if err != nil {
		b.Fatalf("StateBlockSource: %v", err)
	}
	blockBytes := 0
	for i := 0; i < source.BlockCount; i++ {
		block, err := source.Load(i)
		if err != nil {
			b.Fatalf("source.Load(%d): %v", i, err)
		}
		for _, layer := range block.Layers {
			blockBytes += len(layer.KeyBytes) + len(layer.ValueBytes)
		}
	}
	restored := newSessionStateFixture(b)
	if err := restored.RestoreStateBlocks(source); err != nil {
		b.Fatalf("RestoreStateBlocks warmup: %v", err)
	}
	restored.resetRetainedHidden()
	if _, err := restored.GenerateSampledFromCacheLogitsEach(logits, 1, stopTokens, model.NewSampler(1), params, nil, nil); err != nil {
		b.Fatalf("GenerateSampledFromCacheLogitsEach warmup: %v", err)
	}
	b.SetBytes(int64(blockBytes))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := restored.RestoreStateBlocks(source); err != nil {
			b.Fatalf("RestoreStateBlocks: %v", err)
		}
		restored.resetRetainedHidden()
		if _, err := restored.GenerateSampledFromCacheLogitsEach(logits, 1, stopTokens, model.NewSampler(uint64(i+1)), params, nil, nil); err != nil {
			b.Fatalf("GenerateSampledFromCacheLogitsEach: %v", err)
		}
	}
}

func BenchmarkSessionStateRestoreBlocksGenerateSampledFromRetainedHidden(b *testing.B) {
	requireNativeRuntime(b)
	prompt := []int32{1, 2, 3, 4, 5}
	params := model.SampleParams{Temperature: 0.8, TopK: 4, TopP: 0.9}
	stopTokens := []int32{63}

	saved := newSessionStateFixture(b)
	if err := saved.PrefillTokens(prompt); err != nil {
		b.Fatalf("PrefillTokens: %v", err)
	}
	source, err := saved.StateBlockSource(2)
	if err != nil {
		b.Fatalf("StateBlockSource: %v", err)
	}
	blockBytes := 0
	for i := 0; i < source.BlockCount; i++ {
		block, err := source.Load(i)
		if err != nil {
			b.Fatalf("source.Load(%d): %v", i, err)
		}
		for _, layer := range block.Layers {
			blockBytes += len(layer.KeyBytes) + len(layer.ValueBytes)
		}
	}
	restored := newSessionStateFixture(b)
	if err := restored.RestoreStateBlocks(source); err != nil {
		b.Fatalf("RestoreStateBlocks warmup: %v", err)
	}
	if _, err := restored.GenerateSampledFromCacheEach(1, stopTokens, model.NewSampler(1), params, nil, nil); err != nil {
		b.Fatalf("GenerateSampledFromCacheEach warmup: %v", err)
	}
	b.SetBytes(int64(blockBytes))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := restored.RestoreStateBlocks(source); err != nil {
			b.Fatalf("RestoreStateBlocks: %v", err)
		}
		if _, err := restored.GenerateSampledFromCacheEach(1, stopTokens, model.NewSampler(uint64(i+1)), params, nil, nil); err != nil {
			b.Fatalf("GenerateSampledFromCacheEach: %v", err)
		}
	}
}

func BenchmarkSessionStateRestoreBlocksGenerateSampledFromRetainedHiddenTopPOnly(b *testing.B) {
	requireNativeRuntime(b)
	prompt := []int32{1, 2, 3, 4, 5}
	params := model.SampleParams{Temperature: 1, TopP: 0.72}
	stopTokens := []int32{63}

	saved := newSessionStateFixture(b)
	if err := saved.PrefillTokens(prompt); err != nil {
		b.Fatalf("PrefillTokens: %v", err)
	}
	source, err := saved.StateBlockSource(2)
	if err != nil {
		b.Fatalf("StateBlockSource: %v", err)
	}
	source.RetainedLogits = nil
	blockBytes := 0
	for i := 0; i < source.BlockCount; i++ {
		block, err := source.Load(i)
		if err != nil {
			b.Fatalf("source.Load(%d): %v", i, err)
		}
		for _, layer := range block.Layers {
			blockBytes += len(layer.KeyBytes) + len(layer.ValueBytes)
		}
	}
	restored := newSessionStateFixture(b)
	if err := restored.RestoreStateBlocks(source); err != nil {
		b.Fatalf("RestoreStateBlocks warmup: %v", err)
	}
	if _, err := restored.GenerateSampledFromCacheEach(1, stopTokens, model.NewSampler(1), params, nil, nil); err != nil {
		b.Fatalf("GenerateSampledFromCacheEach warmup: %v", err)
	}
	b.SetBytes(int64(blockBytes))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := restored.RestoreStateBlocks(source); err != nil {
			b.Fatalf("RestoreStateBlocks: %v", err)
		}
		if _, err := restored.GenerateSampledFromCacheEach(1, stopTokens, model.NewSampler(uint64(i+1)), params, nil, nil); err != nil {
			b.Fatalf("GenerateSampledFromCacheEach: %v", err)
		}
	}
}

func BenchmarkSessionStateRestoreBlocksGenerateSampledFromRetainedLogitsTopPOnlyLargeVocab(b *testing.B) {
	requireNativeRuntime(b)
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 128
	const maxLen = 24
	g, arch := gemma4BF16Fixture(b, dModel, nHeads, nKV, headDim, dFF, vocab, 2)
	prompt := []int32{1, 2, 3, 4, 5}
	params := model.SampleParams{Temperature: 1, TopP: 0.72}
	stopTokens := []int32{int32(vocab - 1)}

	saved, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		b.Fatalf("NewArchSession saved: %v", err)
	}
	if err := saved.PrefillTokens(prompt); err != nil {
		b.Fatalf("PrefillTokens: %v", err)
	}
	source, err := saved.StateBlockSource(2)
	if err != nil {
		b.Fatalf("StateBlockSource: %v", err)
	}
	if len(source.RetainedLogits) == 0 {
		b.Fatal("StateBlockSource did not retain boundary logits")
	}
	source.RetainedHidden = nil
	blockBytes := 0
	for i := 0; i < source.BlockCount; i++ {
		block, err := source.Load(i)
		if err != nil {
			b.Fatalf("source.Load(%d): %v", i, err)
		}
		for _, layer := range block.Layers {
			blockBytes += len(layer.KeyBytes) + len(layer.ValueBytes)
		}
	}
	restored, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		b.Fatalf("NewArchSession restored: %v", err)
	}
	if err := restored.RestoreStateBlocks(source); err != nil {
		b.Fatalf("RestoreStateBlocks warmup: %v", err)
	}
	if _, err := restored.GenerateSampledFromCacheEach(1, stopTokens, model.NewSampler(1), params, nil, nil); err != nil {
		b.Fatalf("GenerateSampledFromCacheEach warmup: %v", err)
	}
	b.SetBytes(int64(blockBytes))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := restored.RestoreStateBlocks(source); err != nil {
			b.Fatalf("RestoreStateBlocks: %v", err)
		}
		if _, err := restored.GenerateSampledFromCacheEach(1, stopTokens, model.NewSampler(uint64(i+1)), params, nil, nil); err != nil {
			b.Fatalf("GenerateSampledFromCacheEach: %v", err)
		}
	}
}
