// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

// cache_latent.go is the dedicated KV cache for Multi-head Latent Attention. MLA
// caches ONE compressed latent per token (c_kv = x·W_DKV, shape [B, L, latentDim])
// rather than a full per-head K/V pair, which is the whole point of the family —
// a fraction of the KV footprint. The standard KVCache.Update is a two-tensor
// (k, v), rank-4 [B,H,L,D] API: representing a single rank-3 latent through it
// would mean storing it twice, defeating that footprint win. latentKVCache stores
// the latent ONCE and concatenates it across decode chunks along the sequence
// axis — the single-tensor store the mla-latent scheme builds through the KV
// factory (NewCacheForMixer → CacheComputeFor("mla-latent")).
//
// MLA's Forward hands the chunk's latent as Update's k (v is unused) and reads the
// full concatenated latent back; the up-projection to per-head K/V happens in the
// mixer, not here. The cache OWNS the stored latent across the session — Update
// returns the live handle and the caller must not free it (the same ownership the
// recurrent holder uses), so a long decode keeps one growing latent, not a leak.
type latentKVCache struct {
	latent *Array // accumulated [B, totalL, latentDim]; nil before the first Update
	offset int
}

// NewLatentKVCache creates an empty single-tensor latent store — one per MLA
// layer, built by the mla-latent scheme.
//
//	c := metal.NewLatentKVCache()
func NewLatentKVCache() Cache { return &latentKVCache{} }

// Update stores this chunk's latent (k), concatenated onto the prior latent along
// the sequence axis (axis 1 of [B, L, latentDim]), and returns the full latent. v
// is ignored — MLA carries one latent, not a K/V pair. Both returns are the same
// latent handle so a caller that up-projects "K" and "V" reconstructs both from
// it. The cache adopts k (first chunk) or frees the superseded handles (later
// chunks); the caller hands ownership over and must not reuse k after the call.
func (c *latentKVCache) Update(k, _ *Array, seqLen int) (*Array, *Array) {
	if c.latent == nil {
		c.latent = k
	} else {
		full := Concatenate2(c.latent, k, 1)
		Free(c.latent, k)
		c.latent = full
	}
	c.offset += seqLen
	return c.latent, c.latent
}

func (c *latentKVCache) Offset() int { return c.offset }
func (c *latentKVCache) Len() int    { return c.offset }

// State returns the stored latent so the engine's prompt-cache and session
// snapshot persist an MLA layer alongside the KV / recurrent layers.
func (c *latentKVCache) State() []*Array {
	if c.latent == nil {
		return nil
	}
	return []*Array{c.latent}
}

// Reset clears the latent for a new generation session.
func (c *latentKVCache) Reset() {
	if c.latent != nil {
		Free(c.latent)
		c.latent = nil
	}
	c.offset = 0
}

// Detach replaces the stored latent with a graph-parentless copy after Eval so
// the prior graph's Metal memory can be freed — the same lifecycle hook the KV
// and recurrent caches honour.
func (c *latentKVCache) Detach() {
	if c.latent != nil {
		Detach(c.latent)
	}
}

// compile-time proof the latent store satisfies the Cache contract.
var _ Cache = (*latentKVCache)(nil)
