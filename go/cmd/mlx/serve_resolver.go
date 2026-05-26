// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"sync"

	"dappco.re/go/inference"
	openaicompat "dappco.re/go/inference/openai"
	mlx "dappco.re/go/mlx"
	"dappco.re/go/mlx/memory"
)

// candidateToMLXLoadOpts converts a tuned profile's TuningCandidate
// into the full mlx.LoadOption set so every field the auto-tune loop
// found (CacheMode, BatchSize, PromptCache, memory limits, etc.)
// actually reaches the model load. The inference.LoadOption boundary
// only carried ContextLength + ParallelSlots + AdapterPath; the other
// 9 fields were silently dropped because inference.LoadOption can't
// express them. Bridged via mlx.LoadModelAsTextModel.
//
//	opts := candidateToMLXLoadOpts(report.Profile.Candidate)
//	resolver := mlxResolverFunc(modelPath, opts)
func candidateToMLXLoadOpts(c inference.TuningCandidate) []mlx.LoadOption {
	opts := []mlx.LoadOption{}
	if c.ContextLength > 0 {
		opts = append(opts, mlx.WithContextLength(c.ContextLength))
	}
	if c.ParallelSlots > 0 {
		opts = append(opts, mlx.WithParallelSlots(c.ParallelSlots))
	}
	opts = append(opts, mlx.WithPromptCache(c.PromptCache))
	if c.PromptCacheMinTokens > 0 {
		opts = append(opts, mlx.WithPromptCacheMinTokens(c.PromptCacheMinTokens))
	}
	if c.CachePolicy != "" {
		opts = append(opts, mlx.WithCachePolicy(memory.KVCachePolicy(c.CachePolicy)))
	}
	if c.CacheMode != "" {
		opts = append(opts, mlx.WithKVCacheMode(memory.KVCacheMode(c.CacheMode)))
	}
	if c.BatchSize > 0 {
		opts = append(opts, mlx.WithBatchSize(c.BatchSize))
	}
	if c.PrefillChunkSize > 0 {
		opts = append(opts, mlx.WithPrefillChunkSize(c.PrefillChunkSize))
	}
	if c.ExpectedQuantization > 0 {
		opts = append(opts, mlx.WithExpectedQuantization(c.ExpectedQuantization))
	}
	if c.MemoryLimitBytes > 0 || c.CacheLimitBytes > 0 || c.WiredLimitBytes > 0 {
		opts = append(opts, mlx.WithAllocatorLimits(c.MemoryLimitBytes, c.CacheLimitBytes, c.WiredLimitBytes))
	}
	if c.Adapter.Path != "" {
		opts = append(opts, mlx.WithAdapterPath(c.Adapter.Path))
	}
	return opts
}

// mlxResolverFunc returns an openaicompat.Resolver that lazily loads
// modelPath via mlx.LoadModelAsTextModel — the rich-options path that
// preserves all 13 metal.LoadConfig fields the tuned profile carries.
// First ResolveModel call triggers the actual load; subsequent calls
// return the cached model + any load error.
//
//	resolver := mlxResolverFunc(modelPath, candidateToMLXLoadOpts(candidate))
//	openaiMux := openai.NewMuxWithAdmin(resolver, adminCfg)
func mlxResolverFunc(modelPath string, opts []mlx.LoadOption) openaicompat.Resolver {
	var (
		once    sync.Once
		model   inference.TextModel
		loadErr error
	)
	return openaicompat.ResolverFunc(func(_ context.Context, _ string) (inference.TextModel, error) {
		once.Do(func() {
			model, loadErr = mlx.LoadModelAsTextModel(modelPath, opts...)
		})
		return model, loadErr
	})
}
