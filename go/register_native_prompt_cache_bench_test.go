// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package mlx

import (
	"context"
	"iter"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	pkgtokenizer "dappco.re/go/mlx/pkg/tokenizer"
	"dappco.re/go/mlx/spine"
)

func benchmarkNativeTextPromptCacheModel(b *testing.B) *nativeTextModel {
	b.Helper()
	tok, err := pkgtokenizer.LoadTokenizer(writeRootTokenizerForBenchmark(b))
	if err != nil {
		b.Fatalf("LoadTokenizer: %v", err)
	}
	return &nativeTextModel{
		tm:     &promptCacheTextTokenModel{session: &promptCacheTextSession{}},
		tok:    tok,
		maxLen: 4096,
	}
}

func writeRootTokenizerForBenchmark(b *testing.B) string {
	b.Helper()
	dir := b.TempDir()
	path := core.PathJoin(dir, "tokenizer.json")
	if result := core.WriteFile(path, []byte(rootTokenizerJSON), 0o644); !result.OK {
		b.Fatalf("write tokenizer: %v", result.Value)
	}
	return path
}

func nativePromptCacheBenchChunks(n int) iter.Seq[string] {
	return func(yield func(string) bool) {
		for i := 0; i < n; i++ {
			if !yield("hello") {
				return
			}
		}
	}
}

func BenchmarkNativeTextModelWarmPromptCacheChunks(b *testing.B) {
	model := benchmarkNativeTextPromptCacheModel(b)
	ctx := context.Background()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if err := model.WarmPromptCacheChunks(ctx, nativePromptCacheBenchChunks(128)); err != nil {
			b.Fatalf("WarmPromptCacheChunks: %v", err)
		}
	}
}

func BenchmarkNativeTextModelWarmPromptCacheFallbackJoin(b *testing.B) {
	model := benchmarkNativeTextPromptCacheModel(b)
	ctx := context.Background()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		prompt := spine.PromptChunksToString(nativePromptCacheBenchChunks(128))
		if err := model.WarmPromptCache(ctx, prompt); err != nil {
			b.Fatalf("WarmPromptCache: %v", err)
		}
	}
}

func BenchmarkNativeTextModelGenerateChunks(b *testing.B) {
	model := benchmarkNativeTextPromptCacheModel(b)
	ctx := context.Background()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		n := 0
		for range model.GenerateChunks(ctx, nativePromptCacheBenchChunks(128), inference.WithMaxTokens(2)) {
			n++
		}
		if err := resultError(model.Err()); err != nil {
			b.Fatalf("GenerateChunks: %v", err)
		}
		if n == 0 {
			b.Fatal("GenerateChunks produced no tokens")
		}
	}
}

func BenchmarkNativeTextModelGenerateFallbackJoin(b *testing.B) {
	model := benchmarkNativeTextPromptCacheModel(b)
	ctx := context.Background()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		prompt := spine.PromptChunksToString(nativePromptCacheBenchChunks(128))
		n := 0
		for range model.Generate(ctx, prompt, inference.WithMaxTokens(2)) {
			n++
		}
		if err := resultError(model.Err()); err != nil {
			b.Fatalf("Generate: %v", err)
		}
		if n == 0 {
			b.Fatal("Generate produced no tokens")
		}
	}
}
