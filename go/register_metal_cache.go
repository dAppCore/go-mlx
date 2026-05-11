// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && !nomlx

package mlx

import (
	"context"

	"dappco.re/go/inference"
)

func (adapter *metaladapter) CacheStats(ctx context.Context) (inference.CacheStats, error) {
	return adapter.blockCacheService().CacheStats(ctx)
}

func (adapter *metaladapter) CacheEntries(ctx context.Context, labels map[string]string) ([]inference.CacheBlockRef, error) {
	return adapter.blockCacheService().CacheEntries(ctx, labels)
}

func (adapter *metaladapter) WarmCache(ctx context.Context, req inference.CacheWarmRequest) (inference.CacheWarmResult, error) {
	return adapter.blockCacheService().WarmCache(ctx, req)
}

func (adapter *metaladapter) ClearCache(ctx context.Context, labels map[string]string) (inference.CacheStats, error) {
	return adapter.blockCacheService().ClearCache(ctx, labels)
}

func (adapter *metaladapter) blockCacheService() *BlockCacheService {
	if adapter == nil {
		return NewBlockCacheService(BlockCacheConfig{})
	}
	adapter.cacheMu.Lock()
	defer adapter.cacheMu.Unlock()
	if adapter.cacheService == nil {
		info := adapter.Info()
		adapter.cacheService = NewBlockCacheService(BlockCacheConfig{
			BlockSize:     DefaultCacheBlockSize,
			ModelHash:     inferenceModelInfoHash(info),
			AdapterHash:   adapter.ActiveAdapter().Hash,
			TokenizerHash: adapterTokenizerHash(adapter),
			Tokenize: func(prompt string) ([]int32, error) {
				root := adapter.rootModel()
				if root == nil || root.Tokenizer() == nil {
					return nil, nil
				}
				return root.Tokenizer().Encode(prompt)
			},
			WarmPrompt: func(ctx context.Context, prompt string) error {
				if adapter == nil || adapter.model == nil {
					return nil
				}
				return adapter.model.WarmPromptCache(ctx, prompt)
			},
			ClearRuntime: func() {
				if adapter != nil && adapter.model != nil {
					adapter.model.ClearPromptCache()
				}
				ClearCache()
			},
			DiskPath: DefaultBlockCacheDiskPath(),
		})
	}
	return adapter.cacheService
}

func inferenceModelInfoHash(info inference.ModelInfo) string {
	return coreHashModelParts(info.Architecture, info.VocabSize, info.NumLayers, info.HiddenSize, info.QuantBits, info.QuantGroup)
}

func adapterTokenizerHash(adapter *metaladapter) string {
	if adapter == nil || adapter.model == nil {
		return ""
	}
	root := adapter.rootModel()
	if root == nil || root.Tokenizer() == nil {
		return ""
	}
	info := modelInfoFromInference(adapter.Info())
	tok := root.Tokenizer()
	return coreHashModelParts(info.Architecture, info.VocabSize, tok.BOS(), tok.EOS())
}
