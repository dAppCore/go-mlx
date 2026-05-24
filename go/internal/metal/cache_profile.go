// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

// CacheProfile reports how the live K/V caches are shaped after a generation
// turn. It is intentionally small and allocation-light so production retained
// runs can record whether Gemma 4 local layers are bounded at the sliding
// window while global owner layers carry long-context state.
type CacheProfile struct {
	Architecture       string
	TotalCaches        int
	LocalCaches        int
	GlobalCaches       int
	SharedLayers       int
	LocalWindowTokens  int
	MaxLocalTokens     int
	MaxLocalCapacity   int
	MaxGlobalTokens    int
	MaxGlobalCapacity  int
	MaxCacheTokens     int
	MaxCacheCapacity   int
	MaxProcessedTokens int
	FullCaches         int
	RotatingCaches     int
	FixedCaches        int
	PagedCaches        int
	QuantizedCaches    int
	UnknownCaches      int
	UnboundedCaches    int
	LocalWindowLeaked  bool
}

func modelCacheProfile(model InternalModel, caches []Cache) *CacheProfile {
	if len(caches) == 0 {
		return nil
	}
	profile := &CacheProfile{TotalCaches: len(caches)}
	if model != nil {
		profile.Architecture = model.ModelType()
	}
	for _, cache := range caches {
		profile.recordCache(cache)
	}
	gemma4, ok := model.(*Gemma4Model)
	if !ok || gemma4 == nil || gemma4.Cfg == nil {
		return profile
	}
	gemma4.ensureCacheLayout()
	profile.LocalWindowTokens = int(gemma4.Cfg.SlidingWindow)
	for layerIdx, cacheIdx := range gemma4.CacheIndexByLayer {
		if cacheIdx < 0 {
			profile.SharedLayers++
			continue
		}
		if int(cacheIdx) >= len(caches) || layerIdx >= len(gemma4.Layers) {
			continue
		}
		cache := caches[cacheIdx]
		tokens := cacheLen(cache)
		capacity, bounded := cacheCapacity(cache)
		if gemma4.Layers[layerIdx].LayerType == "full_attention" {
			profile.GlobalCaches++
			profile.MaxGlobalTokens = max(profile.MaxGlobalTokens, tokens)
			profile.MaxGlobalCapacity = max(profile.MaxGlobalCapacity, capacity)
			continue
		}
		profile.LocalCaches++
		profile.MaxLocalTokens = max(profile.MaxLocalTokens, tokens)
		profile.MaxLocalCapacity = max(profile.MaxLocalCapacity, capacity)
		if profile.LocalWindowTokens > 0 && (tokens > profile.LocalWindowTokens || capacity > profile.LocalWindowTokens || !bounded) {
			profile.LocalWindowLeaked = true
		}
	}
	return profile
}

func (p *CacheProfile) recordCache(cache Cache) {
	if p == nil || cache == nil {
		return
	}
	tokens := cacheLen(cache)
	capacity, bounded := cacheCapacity(cache)
	p.MaxCacheTokens = max(p.MaxCacheTokens, tokens)
	p.MaxCacheCapacity = max(p.MaxCacheCapacity, capacity)
	p.MaxProcessedTokens = max(p.MaxProcessedTokens, cache.Offset())
	if !bounded {
		p.UnboundedCaches++
	}
	switch cache.(type) {
	case *KVCache:
		p.FullCaches++
	case *RotatingKVCache:
		p.RotatingCaches++
	case *FixedKVCache:
		p.FixedCaches++
	case *PagedKVCache:
		p.PagedCaches++
	case *QuantizedKVCache:
		p.QuantizedCaches++
	default:
		p.UnknownCaches++
	}
}

func cacheLen(cache Cache) int {
	if cache == nil {
		return 0
	}
	return cache.Len()
}

func cacheCapacity(cache Cache) (capacity int, bounded bool) {
	switch c := cache.(type) {
	case *RotatingKVCache:
		return c.maxSize, c.maxSize > 0
	case *FixedKVCache:
		return c.maxSize, c.maxSize > 0
	case *PagedKVCache:
		return c.maxSize, c.maxSize > 0
	case *QuantizedKVCache:
		return c.maxSize, c.maxSize > 0
	default:
		return 0, false
	}
}
