// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

// Cache-scheme factory registry — cache mode → builder, mirroring
// model_registry.go. A scheme (default, fixed, paged, q8, k-q8-v-q4, turboquant,
// compaction, mla-latent, the recurrent holder, …) self-registers from its own
// package init(); the load/generation path looks it up by the mode the config
// declares, with no central switch.
type CacheLoader func(p CacheParams) Cache

var cacheLoaders = map[string]CacheLoader{}

// RegisterCacheLoader registers fn for a cache mode; a later call overrides.
// Call from init() so a scheme self-registers. No-op on empty mode or nil fn.
func RegisterCacheLoader(mode string, fn CacheLoader) {
	if mode == "" || fn == nil {
		return
	}
	cacheLoaders[mode] = fn
}

// lookupCacheLoader returns the loader for mode, or nil if none is registered.
func lookupCacheLoader(mode string) CacheLoader { return cacheLoaders[mode] }
