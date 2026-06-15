// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import core "dappco.re/go"

func detachEvalState(logits *Array, caches []Cache) {
	Detach(logits)
	DetachCaches(caches)
}

func DetachCaches(caches []Cache) {
	for _, cache := range caches {
		if cache != nil {
			cache.Detach()
		}
	}
}

// newCaches creates per-layer KV caches. If contextLen is set, all unbounded
// caches are replaced with RotatingKVCache to cap memory usage.
func (m *Model) newCaches() []Cache {
	return m.newCachesWithRequestFixedSize(0)
}

func (m *Model) newGenerationCaches(promptTokens int, cfg GenerateConfig) []Cache {
	budget := generationTokenBudget(cfg.MaxTokens, m.Info().ContextLength, promptTokens)
	return m.newCachesWithRequestFixedSize(m.generationFixedSlidingCacheSize(promptTokens, budget))
}

func (m *Model) newCachesWithRequestFixedSize(requestFixedSize int) []Cache {
	caches := m.model.NewCache()
	mode := KVCacheMode(m.cacheMode)
	// The fixed-cache regime: a model that declares the fixed-sliding cache
	// (EngineFeatures, e.g. hybrid gemma4) gets sized FixedKVCaches — the
	// compiled+pipelined decode shape — with zero flags in the default mode,
	// or under the explicit -kv-cache paged + -context pair. The serve and
	// the CLI must not need a magic flag to reach the fast lane (#72).
	if mode == KVCacheModeDefault || mode == KVCacheModePaged {
		if replaced, ok := m.fixedSlidingReplacement(caches, requestFixedSize); ok {
			return replaced
		}
	}
	if mode == KVCacheModeQ8 || mode == KVCacheModeKQ8VQ4 || mode == KVCacheModePaged || mode == KVCacheModeTurboQuant {
		maxSize := 0
		if m.cachePolicy != "full" && m.contextLen > 0 {
			maxSize = m.contextLen
		}
		storageDType, hasStorageDType := parseKVCacheStorageDType(m.kvCacheStorageDType)
		for i := range caches {
			layerMaxSize := replacementCacheMaxSize(caches[i], maxSize)
			switch mode {
			case KVCacheModeQ8:
				caches[i] = NewQuantizedKVCache(layerMaxSize, 8, 8)
			case KVCacheModeKQ8VQ4:
				caches[i] = NewQuantizedKVCache(layerMaxSize, 8, 4)
			case KVCacheModePaged:
				if hasStorageDType {
					caches[i] = NewPagedKVCacheWithDTypeAndPrealloc(layerMaxSize, m.pagedKVPageSize, storageDType, m.pagedKVPrealloc)
				} else {
					caches[i] = NewPagedKVCacheWithPrealloc(layerMaxSize, m.pagedKVPageSize, m.pagedKVPrealloc)
				}
			case KVCacheModeTurboQuant:
				cache := NewTurboQuantKVCache(layerMaxSize, 0)
				cache.SetLayerIdentity(i, i, i, "unknown")
				caches[i] = cache
			}
		}
		return caches
	}
	return m.applyContextCachePolicy(caches)
}

// DefaultFixedCacheBound is the zero-flag context bound for the fixed-cache
// regime: ample for agent multi-turn work (the ten-chapter book demo peaks
// under 10K tokens) while keeping the lazily-allocated fixed buffers modest,
// and free in decode speed — the rate is flat in the bound (e2b: 181 tok/s
// at 8K, 24K and 64K alike). -context overrides it in either direction.
const DefaultFixedCacheBound = 24576

// defaultFixedCacheBound resolves the zero-flag bound: the model's declared
// context clamped to DefaultFixedCacheBound — a 128K-context model must not
// allocate 128K-token fixed buffers on the first request.
func (m *Model) defaultFixedCacheBound() int {
	ctx := m.Info().ContextLength
	if ctx <= 0 {
		return DefaultFixedCacheBound
	}
	return min(ctx, DefaultFixedCacheBound)
}

// fixedSlidingReplacement swaps the model's template caches for sized
// FixedKVCaches when the fixed-cache regime applies: the model declares the
// fixed-sliding cache, the cache policy permits bounding, and a bound
// resolves (-context, or the zero-flag default in the default mode). Sliding
// layers clamp to their window (the bound gate); global layers carry the
// request size when known, else the bound.
func (m *Model) fixedSlidingReplacement(caches []Cache, requestFixedSize int) ([]Cache, bool) {
	if !fixedSlidingCacheEnabled() || !modelUsesFixedSlidingCache(m.model) {
		return nil, false
	}
	if m.cachePolicy == "full" {
		return nil, false
	}
	bound := m.contextLen
	if bound <= 0 {
		// Explicit paged mode without -context keeps its paged semantics;
		// only the default mode derives the zero-flag bound from the model.
		if KVCacheMode(m.cacheMode) == KVCacheModePaged {
			return nil, false
		}
		bound = m.defaultFixedCacheBound()
	}
	if bound <= 0 {
		return nil, false
	}
	fixedSize := fixedSlidingCacheSize(bound, requestFixedSize, m.fixedSlidingCacheSize)
	storageDType, hasStorageDType := parseKVCacheStorageDType(m.kvCacheStorageDType)
	for i := range caches {
		layerSize := fixedSize
		if layerMaxSize := replacementCacheMaxSize(caches[i], bound); fixedSlidingCacheBoundEnabled() && layerMaxSize > 0 {
			layerSize = min(layerSize, layerMaxSize)
		}
		if hasStorageDType {
			caches[i] = NewFixedKVCacheWithDType(layerSize, storageDType)
		} else {
			caches[i] = NewFixedKVCache(layerSize)
		}
	}
	return caches, true
}

func parseKVCacheStorageDType(value string) (DType, bool) {
	value = core.Lower(core.Trim(value))
	switch value {
	case "", "native", "default":
		return DTypeFloat32, false
	case "fp16", "float16", "f16":
		return DTypeFloat16, true
	case "bf16", "bfloat16":
		return DTypeBFloat16, true
	default:
		return DTypeFloat32, false
	}
}

// generationTokenBudget resolves how many tokens a request may generate. A
// caller-set MaxTokens (>0) is honoured verbatim — the caller's word, even past
// the context window (sliding-window models rotate). MaxTokens <= 0 means
// "generate to the model's context": the budget is the room left in the window
// (contextLength - promptLen), so the loop runs until EOS/stop or the context
// fills — never a hardcoded cap. Returns 0 when the prompt already fills the
// context or no context is known, so generation is bounded by truth, not a
// guessed default.
func generationTokenBudget(maxTokens, contextLength, promptLen int) int {
	if maxTokens > 0 {
		return maxTokens
	}
	if contextLength > promptLen {
		return contextLength - promptLen
	}
	return 0
}

func (m *Model) generationFixedSlidingCacheSize(promptTokens, maxTokens int) int {
	if m == nil || !fixedSlidingCacheEnabled() || promptTokens <= 0 || maxTokens <= 0 {
		return 0
	}
	if !m.fixedCacheRegimeActive() {
		return 0
	}
	size := promptTokens + maxTokens
	if size < promptTokens {
		return 0
	}
	return roundUpPositive(size, 32)
}

// fixedCacheRegimeActive reports whether generation caches run the sized
// fixed-cache shape: by model declaration in the default mode (zero-flag),
// or explicitly via -kv-cache paged with -context. Quantised and turbo cache
// modes keep their own storage strategies.
func (m *Model) fixedCacheRegimeActive() bool {
	if !modelUsesFixedSlidingCache(m.model) || m.cachePolicy == "full" {
		return false
	}
	switch KVCacheMode(m.cacheMode) {
	case KVCacheModeDefault:
		return true
	case KVCacheModePaged:
		return m.contextLen > 0
	default:
		return false
	}
}

// modelUsesFixedSlidingCache reports whether the loaded model declares the
// fixed-size sliding-window KV cache (FixedSlidingCacheModel) — the engine
// dispatches on the capability, not the model family.
func modelUsesFixedSlidingCache(model InternalModel) bool {
	cache, ok := model.(FixedSlidingCacheModel)
	return ok && cache.UsesFixedSlidingCache()
}

func fixedSlidingCacheSize(maxSize, requestSize, configuredSize int) int {
	if maxSize <= 0 {
		return maxSize
	}
	if configuredSize > 0 {
		return min(configuredSize, maxSize)
	}
	if requestSize > 0 {
		return min(requestSize, maxSize)
	}
	return maxSize
}

func roundUpPositive(value, multiple int) int {
	if value <= 0 || multiple <= 0 {
		return value
	}
	remainder := value % multiple
	if remainder == 0 {
		return value
	}
	return value + multiple - remainder
}

func replacementCacheMaxSize(cache Cache, maxSize int) int {
	if maxSize <= 0 {
		return maxSize
	}
	if rotating, ok := cache.(*RotatingKVCache); ok && rotating.maxSize > 0 {
		return min(maxSize, rotating.maxSize)
	}
	return maxSize
}

func (m *Model) newPromptSnapshotCaches() []Cache {
	switch KVCacheMode(m.cacheMode) {
	case KVCacheModeKQ8VQ4:
		return m.applyContextCachePolicy(m.model.NewCache())
	default:
		return m.newCaches()
	}
}

func (m *Model) applyContextCachePolicy(caches []Cache) []Cache {
	if m.cachePolicy == "full" {
		return caches
	}
	if m.contextLen <= 0 {
		return caches
	}
	for i, c := range caches {
		switch cache := c.(type) {
		// Replace unbounded caches with rotating caches to honour the requested
		// context cap.
		case *KVCache:
			caches[i] = NewRotatingKVCache(m.contextLen)
		// Sliding-window caches are already bounded, but still need shrinking
		// when the caller requests a smaller context than the model default.
		case *RotatingKVCache:
			if cache.maxSize > m.contextLen {
				caches[i] = NewRotatingKVCache(m.contextLen)
			}
		default:
			continue
		}
	}
	return caches
}
