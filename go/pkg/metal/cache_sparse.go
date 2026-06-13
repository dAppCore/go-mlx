// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import scheme "dappco.re/go/mlx/pkg/scheme"

// cache_sparse.go registers the decode-time KV-cache schemes for the sparse /
// latent attention mixers (MLA, NSA, MoBA). All three Serve StateKVCache — they
// keep a growing per-token cache, not a recurrent state holder — so scheme
// .Compatible pairs them with their mixers and the engine's prompt-cache /
// session-reset / graph-detach lifecycle treats them like any KV cache.
//
//	cc, _ := metal.CacheComputeFor("mla-latent")
//	c := cc.NewCache(metal.CacheParams{MaxSize: 4096})
//
// The schemes are thin by design: a KVCache.Update is already dimension-agnostic
// (it grows axis-2 for any [B,H,L,D]), so the per-family decode work — MLA's
// up-projection of the cached latent, NSA's block-mean + sliding tail, MoBA's
// per-step block scoring — lives in each mixer's Forward, NOT here. What differs
// per scheme is WHAT is stored and HOW it is bounded:
//
//   - mla-latent: stores the COMPRESSED KV latent c_kv [B,1,L,latentDim] (one
//     rank-r vector per token), not full per-head K/V — the whole point of MLA's
//     small KV footprint. Mechanically a growing cache over the latent width; a
//     future quant/compaction scheme attaches to this mode to shrink the latent
//     further.
//   - nsa: stores full K/V growing (the selection branch attends over original
//     tokens, so they must be retained); the compressed blocks are derived by
//     the mixer. A bounded build caps the retained window.
//   - moba: stores full K/V growing (attention runs over the selected blocks'
//     original tokens); block scores are derived by the mixer.
//
// Registering these overwrites the metadata-only catalogue seeds the engine adds
// for new modes (the standard Mode-overwrites-metadata move kvCacheScheme makes
// for the modes the engine already ships), so the loader's CacheComputeFor
// resolves a real, compute-bearing scheme once a hybrid pack selects one.

// Sparse-family KVCacheMode tokens. They name the stored shape / bounding
// strategy, not a new storage engine — the underlying store is the engine's
// existing growing (or rotating) KV cache.
const (
	KVCacheModeMLALatent KVCacheMode = "mla-latent"
	KVCacheModeNSA       KVCacheMode = "nsa"
	KVCacheModeMoBA      KVCacheMode = "moba"
)

// mlaLatentCacheScheme stores MLA's compressed KV latent. NewCache builds a
// growing cache (MaxSize 0) or a rotating one (MaxSize > 0) over whatever last
// dimension the mixer hands it — the latent width, not full K/V. The latent
// compression happens in the MLA mixer's projections; this scheme only persists
// and streams the result, so the cache stays storage-agnostic to the width.
type mlaLatentCacheScheme struct{}

func (mlaLatentCacheScheme) Mode() string             { return string(KVCacheModeMLALatent) }
func (mlaLatentCacheScheme) Serves() scheme.StateKind { return scheme.StateKVCache }
func (mlaLatentCacheScheme) NewCache(p CacheParams) Cache {
	if p.MaxSize > 0 {
		return NewRotatingKVCache(p.MaxSize)
	}
	return NewKVCache()
}

// nsaCacheScheme stores NSA's full K/V. The selection branch attends over the
// original tokens of the chosen blocks, so the tokens must be retained; the
// per-block summaries the compression branch uses are derived in the mixer from
// this cache, not stored separately. MaxSize > 0 bounds the retained window
// (the sliding branch only needs the recent tail; the block branches a longer
// reach — a bounded build trades recall for footprint).
type nsaCacheScheme struct{}

func (nsaCacheScheme) Mode() string             { return string(KVCacheModeNSA) }
func (nsaCacheScheme) Serves() scheme.StateKind { return scheme.StateKVCache }
func (nsaCacheScheme) NewCache(p CacheParams) Cache {
	if p.MaxSize > 0 {
		return NewRotatingKVCache(p.MaxSize)
	}
	return NewKVCache()
}

// mobaCacheScheme stores MoBA's full K/V. Attention runs over the selected
// blocks' original tokens, so the cache retains them; block gate scores are
// derived per step in the mixer from this cache.
type mobaCacheScheme struct{}

func (mobaCacheScheme) Mode() string             { return string(KVCacheModeMoBA) }
func (mobaCacheScheme) Serves() scheme.StateKind { return scheme.StateKVCache }
func (mobaCacheScheme) NewCache(p CacheParams) Cache {
	if p.MaxSize > 0 {
		return NewRotatingKVCache(p.MaxSize)
	}
	return NewKVCache()
}

func init() {
	scheme.RegisterCache(mlaLatentCacheScheme{})
	scheme.RegisterCache(nsaCacheScheme{})
	scheme.RegisterCache(mobaCacheScheme{})
}

// compile-time proofs each sparse scheme is a full metal.CacheCompute.
var (
	_ CacheCompute = mlaLatentCacheScheme{}
	_ CacheCompute = nsaCacheScheme{}
	_ CacheCompute = mobaCacheScheme{}
)
