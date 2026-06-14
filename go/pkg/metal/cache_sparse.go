// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import scheme "dappco.re/go/mlx/pkg/scheme"

// cache_sparse.go registers the ONE decode-time KV-cache scheme the sparse /
// latent attention family genuinely needs: mla-latent. It Serves StateKVCache —
// a growing per-token cache, not a recurrent holder — so scheme.Compatible
// pairs it with the MLA mixer and the engine's prompt-cache / session-reset /
// graph-detach lifecycle treats it like any KV cache.
//
//	cc, _ := metal.CacheComputeFor("mla-latent")
//	c := cc.NewCache(metal.CacheParams{MaxSize: 4096})
//
// NSA and MoBA deliberately get NO new cache scheme: they store full K/V (their
// selection / block branches attend over the original tokens), which is exactly
// what the engine's existing growing / rotating KV cache already holds, and
// KVCache.Update is dimension-agnostic. Registering "nsa" / "moba" modes that
// merely alias NewKVCache would pollute the catalogue with redundant entries —
// their per-step decode work (block-mean, sliding tail, block scoring) lives in
// each mixer's Forward against the standard cache, not in a bespoke scheme.
//
// MLA is the one family with a genuinely different STORED shape — the compressed
// KV latent c_kv [B,1,L,latentDim] (one rank-r vector per token) rather than
// full per-head K/V. That smaller footprint is the whole point of MLA, and it is
// what justifies a distinct mode: the engine allocates the latent, and a future
// quant / compaction scheme attaches to "mla-latent" to shrink it further.
// Mechanically the store is the dedicated single-tensor latentKVCache
// (cache_latent.go): it persists and concatenates the one latent across decode,
// where the standard two-tensor KVCache would store it twice. The compression
// itself happens in the MLA mixer's projections; this scheme only persists it.
//
// Registering this overwrites the metadata-only catalogue seed the engine adds
// for the mode (the standard Mode-overwrites-metadata move kvCacheScheme makes
// for the modes the engine already ships), so the loader's CacheComputeFor
// resolves a real, compute-bearing scheme once a hybrid pack selects it.

// KVCacheModeMLALatent names MLA's compressed-latent store. It names the stored
// shape, not a new storage engine — the backing store is the engine's existing
// growing (or rotating) KV cache, sized to the latent width.
const KVCacheModeMLALatent KVCacheMode = "mla-latent"

// mlaLatentCacheScheme stores MLA's compressed KV latent. NewCache builds the
// dedicated single-tensor latentKVCache (cache_latent.go) — one rank-r latent
// per token, concatenated across decode and stored ONCE, rather than the
// two-tensor KVCache which would store the latent twice and lose the footprint
// win that is MLA's whole reason to exist.
type mlaLatentCacheScheme struct{}

func (mlaLatentCacheScheme) Mode() string             { return string(KVCacheModeMLALatent) }
func (mlaLatentCacheScheme) Serves() scheme.StateKind { return scheme.StateKVCache }
func (mlaLatentCacheScheme) NewCache(CacheParams) Cache {
	return NewLatentKVCache()
}

func init() { scheme.RegisterCache(mlaLatentCacheScheme{}) }

// compile-time proof the scheme is a full metal.CacheCompute.
var _ CacheCompute = mlaLatentCacheScheme{}
