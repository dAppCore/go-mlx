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
		// Pre-build the tokenizer wrapper once so the Tokenize closure does
		// not allocate a fresh *Model + *Tokenizer per call, nor pay the
		// rootModel() cgo crossings (Adapter() + Info()) on every tokenize.
		// adapter.model may still be nil here for zero-value test fixtures;
		// in that case tokenizer.tok stays nil and the closure short-circuits.
		var tokenizer *Tokenizer
		if adapter.model != nil {
			tokenizer = &Tokenizer{tok: adapter.model.Tokenizer()}
		}
		adapter.cacheService = blockcache.New(blockcache.Config{
			BlockSize:     blockcache.DefaultBlockSize,
			ModelHash:     inferenceModelInfoHash(info),
			AdapterHash:   adapter.ActiveAdapter().Hash,
			TokenizerHash: adapterTokenizerHashFromInfo(adapter, info),
			Tokenize: func(prompt string) ([]int32, error) {
				if tokenizer == nil || tokenizer.tok == nil {
					return nil, nil
				}
				return tokenizer.Encode(prompt)
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
	return adapterTokenizerHashFromInfo(adapter, adapter.Info())
}

// adapterTokenizerHashFromInfo is the inner form that lets callers pass an
// already-resolved inference.ModelInfo, avoiding a second adapter.Info() cgo
// crossing when the caller has just made the call themselves.
func adapterTokenizerHashFromInfo(adapter *metaladapter, info inference.ModelInfo) string {
	if adapter == nil || adapter.model == nil {
		return ""
	}
	root := adapter.rootModel()
	if root == nil {
		return ""
	}
	tok := root.Tokenizer()
	if tok == nil {
		return ""
	}
	return blockcache.HashModelParts(info.Architecture, info.VocabSize, tok.BOS(), tok.EOS())
}
