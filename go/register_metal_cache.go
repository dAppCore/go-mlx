// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"dappco.re/go/mlx/blockcache"

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

func (adapter *metaladapter) blockCacheService() *blockcache.Service {
	if adapter == nil {
		return blockcache.New(blockcache.Config{})
	}
	adapter.cacheMu.Lock()
	defer adapter.cacheMu.Unlock()
	if adapter.cacheService == nil {
		info := adapter.Info()
		adapter.cacheService = blockcache.New(blockcache.Config{
			BlockSize:     blockcache.DefaultBlockSize,
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
			DiskPath: blockcache.DefaultDiskPath(),
		})
	}
	return adapter.cacheService
}

func inferenceModelInfoHash(info inference.ModelInfo) string {
	return blockcache.HashModelParts(info.Architecture, info.VocabSize, info.NumLayers, info.HiddenSize, info.QuantBits, info.QuantGroup)
}

func adapterTokenizerHash(adapter *metaladapter) string {
	if adapter == nil || adapter.model == nil {
		return ""
	}
	root := adapter.rootModel()
	if root == nil || root.Tokenizer() == nil {
		return ""
	}
	info := adapter.Info()
	tok := root.Tokenizer()
	return blockcache.HashModelParts(info.Architecture, info.VocabSize, tok.BOS(), tok.EOS())
}
