// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import scheme "dappco.re/go/mlx/pkg/scheme"

// turboQuantCacheScheme attaches the metal CacheCompute surface to the
// "turboquant" catalogue entry — the 3.5-bit cold-page compressed K/V format.
// It wraps the engine's existing TurboQuantKVCache compute (turboquant_kv_cache.go)
// rather than reimplementing it; NewCache hands back a fresh page-compressed
// store sized from the universal CacheParams. Registering this overwrites the
// metadata-only scheme.cacheInfo{"turboquant"} seeded in pkg/scheme/builtin.go,
// the same Mode-overwrites-metadata move kvCacheScheme makes for the modes the
// engine ships.
//
//	cc, _ := metal.CacheComputeFor("turboquant") // resolves this scheme
//	c := cc.NewCache(metal.CacheParams{MaxSize: 4096, PageSize: 2048})
type turboQuantCacheScheme struct{}

func (turboQuantCacheScheme) Mode() string             { return string(KVCacheModeTurboQuant) }
func (turboQuantCacheScheme) Serves() scheme.StateKind { return scheme.StateKVCache }

// NewCache builds the page-compressed TurboQuant store. PageSize 0 falls back to
// the cache's own default page size; MaxSize 0 leaves the cache unbounded (no
// sliding window), matching the growing-cache convention of CacheParams.
func (turboQuantCacheScheme) NewCache(p CacheParams) Cache {
	return NewTurboQuantKVCache(p.MaxSize, p.PageSize)
}

func init() { registerCachePreservingWidth(turboQuantCacheScheme{}) }

// compile-time proof turboQuantCacheScheme is a full metal.CacheCompute.
var _ CacheCompute = turboQuantCacheScheme{}
