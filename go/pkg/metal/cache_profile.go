// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

// CacheProfile reports how the live K/V caches are shaped after a generation
// turn. It is intentionally small and allocation-light so production retained
// runs can record whether local/cacheless layers stay bounded or absent while
// global owner layers carry long-context state.
type CacheProfile struct {
	Architecture       string
	TotalCaches        int
	LocalCaches        int
	GlobalCaches       int
	SharedLayers       int
	CachelessLayers    int
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
	switch concrete := model.(type) {
	case cacheTopologyRecorder:
		concrete.recordCacheTopology(profile, caches)
	case qwen36HybridCachePlanner:
		profile.recordQwen36HybridTopology(concrete, caches)
	}
	return profile
}

func (p *CacheProfile) recordQwen36HybridTopology(model qwen36HybridCachePlanner, caches []Cache) {
	if p == nil || model == nil {
		return
	}
	plan, ok := model.qwen36HybridCachePlan()
	if !ok {
		return
	}
	p.CachelessLayers += plan.LinearLayers
	for _, layer := range plan.Layers {
		if !layer.RequiresKV {
			continue
		}
		if layer.CacheIndex < 0 || layer.CacheIndex >= len(caches) {
			continue
		}
		cache := caches[layer.CacheIndex]
		tokens := cacheLen(cache)
		capacity, _ := cacheCapacity(cache)
		p.GlobalCaches++
		p.MaxGlobalTokens = max(p.MaxGlobalTokens, tokens)
		p.MaxGlobalCapacity = max(p.MaxGlobalCapacity, capacity)
	}
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
