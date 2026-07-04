// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import scheme "dappco.re/go/mlx/pkg/scheme"

// CacheParams are the universal knobs a KV-cache scheme builds its store from; a
// scheme reads only the ones it needs (the default growing cache ignores all).
type CacheParams struct {
	MaxSize  int // rotating / fixed / paged / quantized cap; 0 = the default growing cache
	PageSize int // paged schemes only
}

// CacheCompute is the driver-side contract a KV-cache scheme fulfils on the
// metal Engine: build the K/V store for its mode. The engine resolves it with
// scheme.CacheFor(mode); TurboQuant and Attention-Matching compaction (task #7)
// register their own here. Recurrent mixers use a separate state holder — not
// this KV-shaped contract — so Serves() reports StateKVCache for every scheme.
type CacheCompute interface {
	scheme.CacheScheme // Mode() + Serves()
	NewCache(p CacheParams) Cache
}

// CacheComputeFor resolves a registered KV-cache scheme and asserts it carries
// the metal compute surface (metadata-only entries report (nil,false)).
func CacheComputeFor(mode string) (CacheCompute, bool) {
	c, ok := scheme.CacheFor(mode)
	if !ok {
		return nil, false
	}
	cc, ok := c.(CacheCompute)
	return cc, ok
}

// kvCacheScheme adapts the engine's existing KV-cache constructors into the
// CacheCompute contract — the catalogue's entry-one for the modes the engine
// already implements. The mode→cache switch in the loader reroutes through
// CacheComputeFor in the cache-integration step; for now these prove the
// contract and give task #7 (TurboQuant, compaction) the pattern to mirror.
type kvCacheScheme struct {
	mode string
	make func(CacheParams) Cache
}

func (s kvCacheScheme) Mode() string                 { return s.mode }
func (s kvCacheScheme) Serves() scheme.StateKind     { return scheme.StateKVCache }
func (s kvCacheScheme) NewCache(p CacheParams) Cache { return s.make(p) }

func init() {
	for _, s := range []kvCacheScheme{
		{"default", func(CacheParams) Cache { return NewKVCache() }},
		{"fixed", func(p CacheParams) Cache { return NewFixedKVCache(p.MaxSize) }},
		{"paged", func(p CacheParams) Cache { return NewPagedKVCache(p.MaxSize, p.PageSize) }},
		{"q8", func(p CacheParams) Cache { return NewQuantizedKVCache(p.MaxSize, 8, 8) }},
		{"k-q8-v-q4", func(p CacheParams) Cache { return NewQuantizedKVCache(p.MaxSize, 8, 4) }},
	} {
		registerCachePreservingWidth(s)
	}
}

// widthPreservingCacheScheme carries a driver compute value together with the
// width the mode's PRIOR registration declared — RegisterCache overwrites by
// mode, and a compute value without scheme.CacheWidth would strip the
// per-element byte ratio the memory planner sizes from (the turboquant
// width-stripping bug class, caught upstream in inference/kv).
type widthPreservingCacheScheme struct {
	CacheCompute
	width scheme.CacheWidth
}

func (w widthPreservingCacheScheme) KVBytesPerElement() (uint64, uint64, bool) {
	return w.width.KVBytesPerElement()
}

// registerCachePreservingWidth registers a driver cache scheme, forwarding
// the prior registration's scheme.CacheWidth when the driver value lacks its
// own. Modes with no prior width (compaction, mla-latent) register as-is.
func registerCachePreservingWidth(c CacheCompute) {
	if _, hasWidth := c.(scheme.CacheWidth); hasWidth {
		scheme.RegisterCache(c)
		return
	}
	if prior, ok := scheme.CacheFor(c.Mode()); ok {
		if width, ok := prior.(scheme.CacheWidth); ok {
			scheme.RegisterCache(widthPreservingCacheScheme{CacheCompute: c, width: width})
			return
		}
	}
	scheme.RegisterCache(c)
}

// compile-time proof kvCacheScheme is a full metal.CacheCompute.
var _ CacheCompute = kvCacheScheme{}
