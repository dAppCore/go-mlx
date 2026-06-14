// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import scheme "dappco.re/go/mlx/pkg/scheme"

// cache_factory.go is the single construction front-door for per-layer KV
// caches. The cache *registry* (scheme.RegisterCache / CacheComputeFor) maps a
// scheme mode to a builder; this factory resolves the mode a given mixer needs
// and builds it — so the composed runner and the loaders never hand-pick a
// concrete cache type. It is the cache counterpart to MixerComputeFor and the
// quant-loader registry: the same component-factory shape (#39).
//
//	c := metal.NewCacheForMixer(mlaMixer, metal.CacheParams{MaxSize: ctxLen})
//
// A mixer's cache shape is its declared CacheMode() when it names one (MLA →
// "mla-latent"); otherwise the default scheme for its StateKind — the growing KV
// cache for StateKVCache, the recurrent holder for StateRecurrent.

// cacheModeDefault is the growing KV-cache scheme every softmax / KV mixer gets
// unless it names a more specific one.
const cacheModeDefault = "default"

// cacheModeRecurrent is the recurrent-state holder scheme the SSM / linear-
// attention mixers (Mamba2, RWKV7, GLA, RetNet, DeltaNet, GSA) pair with.
const cacheModeRecurrent = "recurrent"

// CacheModer is the optional surface a mixer implements to name the SPECIFIC
// cache scheme it needs — MLA returns "mla-latent" for its compressed-latent
// store. A mixer without a bespoke cache shape does not implement it and gets
// the default scheme for its StateKind, so only the special cases carry the
// method.
type CacheModer interface {
	CacheMode() string
}

// CacheModeForMixer is the scheme mode the factory builds for m: its declared
// CacheMode() when it implements CacheModer and names a non-empty mode, else the
// default scheme for its State().
func CacheModeForMixer(m scheme.Mixer) string {
	if cm, ok := m.(CacheModer); ok {
		if mode := cm.CacheMode(); mode != "" {
			return mode
		}
	}
	if m.State() == scheme.StateRecurrent {
		return cacheModeRecurrent
	}
	return cacheModeDefault
}

// NewCacheForMode builds the cache registered for a scheme mode. ok is false for
// an unregistered (or metadata-only, non-compute) mode; the caller decides
// whether that is fatal or a fall-back point.
func NewCacheForMode(mode string, p CacheParams) (Cache, bool) {
	cc, ok := CacheComputeFor(mode)
	if !ok {
		return nil, false
	}
	return cc.NewCache(p), true
}

// NewCacheForMixer resolves the cache a mixer needs (CacheModeForMixer) and
// builds it. An unresolvable mode falls back to the default growing KV cache so
// a model still loads rather than nil-derefs; a mixer that genuinely needs its
// bespoke scheme registers that scheme from init(), so this never silently
// downgrades a correctly-wired mixer.
func NewCacheForMixer(m scheme.Mixer, p CacheParams) Cache {
	if c, ok := NewCacheForMode(CacheModeForMixer(m), p); ok {
		return c
	}
	return NewKVCache()
}
