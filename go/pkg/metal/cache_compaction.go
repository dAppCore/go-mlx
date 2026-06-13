// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	core "dappco.re/go"
	scheme "dappco.re/go/mlx/pkg/scheme"
)

// KVCacheModeCompaction names the Attention-Matching compaction scheme. It is a
// research mode (opt-in, never the loader default) like turboquant.
const KVCacheModeCompaction KVCacheMode = "compaction"

// defaultCompactionBudget is the compacted token count a CompactionKVCache keeps
// when CacheParams.MaxSize leaves it unset — chosen so a compacted store sits
// around the ~10% target the Attention-Matching paper reports for a few-thousand
// token context.
const defaultCompactionBudget = 256

// CompactionKVCache is the Attention-Matching compaction K/V cache — instead of
// evicting whole tokens (the rotating/fixed sliding window) it keeps a smaller
// set of keys and values, selected in latent space, that reproduces the cache's
// attention behaviour. This is the AM-HighestAttnKeys variant of
// github.com/adamzweiger/compaction: score every key by the maximum softmax
// attention any query places on it, keep the top-`budget` keys, and take the
// values at those same positions (the paper's `c2_method='direct'`,
// `beta_method='zero'` configuration — the configuration that composes purely
// from MLX ops with no linear-algebra solve).
//
//	c := metal.NewCompactionKVCache(256, 0) // keep ≤256 compacted tokens
//	c.Update(k, v, seqLen)                  // appends, then compacts past budget
//	state := c.State()                      // [B,H,≤256,D] — reduced, round-trips
//
// The cache grows uncompacted while Len ≤ budget (so short contexts pay nothing
// and round-trip exactly). The first Update that would carry the length past the
// budget triggers a compaction down to `budget` tokens. Compaction is NOT a
// per-token hot path: it fires once per budget-crossing, so the score+select+
// gather graph runs on the accumulated window, not on every decoded token.
//
// Query proxy: the AM algorithm scores keys against a set of training queries.
// A standalone decode cache has no separate query stream, so the cached keys
// themselves stand in as the query proxy (self-attention over the cache). This
// is the self-contained fallback the upstream `query_generation` package
// replaces with model-derived queries when one is available; the selection
// mathematics is identical, only the query source differs.
type CompactionKVCache struct {
	keys, values *Array
	offset       int // total tokens ever processed (monotonic, like KVCache.Offset)
	length       int // tokens currently held (≤ budget after a compaction)
	budget       int // compacted token target; ≤0 means defaultCompactionBudget
	lastErr      error
}

// NewCompactionKVCache builds an Attention-Matching compaction cache that keeps
// at most `budget` compacted tokens. budget ≤ 0 falls back to
// defaultCompactionBudget. pageSize is accepted for CacheParams symmetry with
// the paged/turboquant schemes but the compaction store has no page structure,
// so it is ignored.
func NewCompactionKVCache(budget, pageSize int) *CompactionKVCache {
	_ = pageSize
	if budget <= 0 {
		budget = defaultCompactionBudget
	}
	return &CompactionKVCache{budget: budget}
}

// Update appends the incoming K/V to the held window, then compacts down to the
// budget when the window grows past it. Returns the full (post-compaction) K/V
// for the current attention pass, mirroring the Cache contract.
func (c *CompactionKVCache) Update(k, v *Array, seqLen int) (*Array, *Array) {
	if c == nil {
		return k, v
	}
	c.lastErr = nil
	if k == nil || v == nil || !k.Valid() || !v.Valid() {
		c.lastErr = core.NewError("mlx: compaction KV cache received invalid arrays")
		return k, v
	}
	if k.NumDims() < 4 || v.NumDims() < 4 {
		// Non rank-4 K/V — pass through unchanged, matching KVCache's guard.
		if c.keys == nil {
			c.keys, c.values = k.Clone(), v.Clone()
		}
		c.offset += seqLen
		return c.keys, c.values
	}

	appendedK, appendedV := c.appendWindow(k, v)
	c.offset += seqLen
	c.length += incomingTokenLen(k, seqLen)

	if c.budget > 0 && c.length > c.budget {
		if err := c.compactToBudget(); err != nil {
			c.lastErr = err
			// Leave the uncompacted window in place so attention still has the
			// full state; the caller surfaces Err().
			return c.viewState()
		}
	}
	_ = appendedK
	_ = appendedV
	return c.viewState()
}

// appendWindow concatenates the incoming K/V onto the held window along the
// sequence axis (axis 2), cloning on first write so the cache owns its storage.
func (c *CompactionKVCache) appendWindow(k, v *Array) (*Array, *Array) {
	if c.keys == nil || !c.keys.Valid() {
		c.keys, c.values = k.Clone(), v.Clone()
		return c.keys, c.values
	}
	oldK, oldV := c.keys, c.values
	c.keys = Concatenate2(oldK, k, 2)
	c.values = Concatenate2(oldV, v, 2)
	Free(oldK, oldV)
	return c.keys, c.values
}

// compactToBudget runs the AM-HighestAttnKeys selection on the held window and
// replaces it with the `budget` highest-attention keys and their values.
func (c *CompactionKVCache) compactToBudget() error {
	if c.keys == nil || c.values == nil || !c.keys.Valid() || !c.values.Valid() {
		return core.NewError("mlx: compaction KV cache has no state to compact")
	}
	if c.length <= c.budget {
		return nil
	}
	newK, newV, err := compactAttentionMatching(c.keys, c.values, c.budget)
	if err != nil {
		return err
	}
	Free(c.keys, c.values)
	c.keys, c.values = newK, newV
	c.length = c.budget
	return nil
}

// viewState returns sequence-bounded slice views over the held window so a
// caller can Free its handles without invalidating the cache's storage.
func (c *CompactionKVCache) viewState() (*Array, *Array) {
	if c.keys == nil || c.values == nil || !c.keys.Valid() || !c.values.Valid() {
		return nil, nil
	}
	B, H := int32(c.keys.Dim(0)), int32(c.keys.Dim(1))
	Dk, Dv := int32(c.keys.Dim(3)), int32(c.values.Dim(3))
	L := int32(c.keys.Dim(2))
	return Slice4(c.keys, 0, 0, 0, 0, B, H, L, Dk),
		Slice4(c.values, 0, 0, 0, 0, B, H, L, Dv)
}

func (c *CompactionKVCache) Offset() int {
	if c == nil {
		return 0
	}
	return c.offset
}

func (c *CompactionKVCache) Len() int {
	if c == nil {
		return 0
	}
	return c.length
}

func (c *CompactionKVCache) State() []*Array {
	if c == nil || c.keys == nil || !c.keys.Valid() {
		return nil
	}
	return []*Array{c.keys, c.values}
}

// AppendState appends valid state arrays into dst. See stateAppender.
func (c *CompactionKVCache) AppendState(dst []*Array) []*Array {
	if c == nil || c.keys == nil {
		return dst
	}
	if c.keys != nil && c.keys.Valid() {
		dst = append(dst, c.keys)
	}
	if c.values != nil && c.values.Valid() {
		dst = append(dst, c.values)
	}
	return dst
}

func (c *CompactionKVCache) Reset() {
	if c == nil {
		return
	}
	Free(c.keys, c.values)
	c.keys = nil
	c.values = nil
	c.offset = 0
	c.length = 0
	c.lastErr = nil
}

func (c *CompactionKVCache) Detach() {
	if c == nil || c.keys == nil {
		return
	}
	Detach(c.keys, c.values)
}

// Compact forces an Attention-Matching compaction of the held window down to the
// budget now, even when Len has not yet crossed it. Returns false when there is
// nothing to compact (empty, or already at/below budget). Callers use it to
// reclaim K/V memory at a turn boundary rather than waiting for the next Update
// to cross the budget.
func (c *CompactionKVCache) Compact() bool {
	if c == nil || c.keys == nil || c.length <= c.budget {
		return false
	}
	if err := c.compactToBudget(); err != nil {
		c.lastErr = err
		return false
	}
	return true
}

// Budget reports the compacted token target this cache compacts down to.
func (c *CompactionKVCache) Budget() int {
	if c == nil {
		return 0
	}
	return c.budget
}

// Err returns the last error recorded by Update / Compact, or nil.
func (c *CompactionKVCache) Err() error {
	if c == nil {
		return nil
	}
	return c.lastErr
}

// incomingTokenLen returns the sequence length the incoming K contributes,
// honouring an explicit seqLen but clamping to the tensor's own L dimension.
func incomingTokenLen(k *Array, seqLen int) int {
	l := k.Dim(2)
	if seqLen > 0 && seqLen < l {
		return seqLen
	}
	return l
}

// compactAttentionMatching builds the AM-HighestAttnKeys compacted (K,V) pair.
// Given a window K,V of shape [B,H,L,D] it returns [B,H,t,D] where t = budget:
//
//  1. scores = softmax(K·Kᵀ / √D)        — self-attention over the cache [B,H,L,L]
//  2. keyScore = max over the query axis  — each key's peak attention      [B,H,L]
//  3. select the top-t keys by keyScore   — descending Argsort, take tail   [B,H,t]
//  4. gather K and V rows at those indices (TakeAlongAxis over the L axis).
//
// This is the `c2_method='direct'`, `beta_method='zero'` configuration of the
// reference algorithm: beta is implicitly zero (no per-key bias) and the
// compacted values are taken straight from V at the selected positions, so the
// whole construction is MLX ops with no linear-algebra solve. The lsq/NNLS
// refinement (a least-squares fit of compacted values + a per-key beta bias)
// needs a batched linear solver MLX does not expose here — see the package
// SECURITY-NOTE in the scheme registration.
func compactAttentionMatching(k, v *Array, budget int) (*Array, *Array, error) {
	if budget <= 0 {
		return nil, nil, core.NewError("mlx: compaction budget must be positive")
	}
	if k.NumDims() != 4 || v.NumDims() != 4 {
		return nil, nil, core.NewError("mlx: compaction requires rank-4 K/V arrays")
	}
	B, H := k.Dim(0), k.Dim(1)
	L, Dk := k.Dim(2), k.Dim(3)
	Dv := v.Dim(3)
	if L <= budget {
		return k.Clone(), v.Clone(), nil
	}
	if v.Dim(0) != B || v.Dim(1) != H || v.Dim(2) != L {
		return nil, nil, core.NewError("mlx: compaction K/V batch/head/seq dims differ")
	}

	// 1. Self-attention scores over the cache: softmax(K·Kᵀ / √D).
	kT := Transpose4(k, 0, 1, 3, 2) // [B,H,D,L]
	rawScores := Matmul(k, kT)      // [B,H,L,L]
	Free(kT)
	scaled := MulScalar(rawScores, float32(1.0/sqrtFloat(float64(Dk))))
	Free(rawScores)
	attn := Softmax(scaled) // [B,H,L,L] — last axis (the key axis) normalised
	Free(scaled)

	// 2. Each key's peak attention across all queries (axis 2 = the query axis).
	keyScore := MaxAxis(attn, 2, false) // [B,H,L]
	Free(attn)

	// 3. Descending key order, keep the top-`budget` indices.
	order := Argsort(keyScore, -1) // ascending indices over L → [B,H,L]
	Free(keyScore)
	// The top-`budget` keys are the tail of an ascending sort.
	topIdx := sliceLastK(order, B, H, L, budget) // [B,H,budget]
	Free(order)

	// 4. Gather K and V rows at the selected positions along the L axis.
	idxK := broadcastIndexToDim(topIdx, B, H, budget, Dk) // [B,H,budget,Dk]
	gatheredK := TakeAlongAxis(k, idxK, 2)
	Free(idxK)
	idxV := broadcastIndexToDim(topIdx, B, H, budget, Dv) // [B,H,budget,Dv]
	gatheredV := TakeAlongAxis(v, idxV, 2)
	Free(idxV, topIdx)

	return gatheredK, gatheredV, nil
}

// sliceLastK returns the last k entries along the L axis of an ascending sort
// index tensor [B,H,L], i.e. the top-k highest-scoring keys — [B,H,k]. (The
// order within the kept set does not matter: attention is permutation-invariant
// over keys, so any ordering of the selected positions yields the same output.)
func sliceLastK(order *Array, _, _, L, k int) *Array {
	return SliceAxis(order, 2, int32(L-k), int32(L))
}

// broadcastIndexToDim expands a position-index tensor [B,H,k] to [B,H,k,D] so
// TakeAlongAxis gathers the full D-vector at each selected sequence position.
func broadcastIndexToDim(idx *Array, B, H, k, D int) *Array {
	expanded := ExpandDims(idx, 3) // [B,H,k,1]
	out := BroadcastTo(expanded, []int32{int32(B), int32(H), int32(k), int32(D)})
	Free(expanded)
	return out
}

// sqrtFloat is a tiny local sqrt — the metal package bans the math import on its
// hot paths, but compaction is a cold once-per-budget-crossing path. A Newton
// step keeps it dependency-free and is exact enough for the 1/√D attention
// scale (D is a small power of two in every shipping model).
func sqrtFloat(x float64) float64 {
	if x <= 0 {
		return 0
	}
	guess := x
	for range 32 {
		next := 0.5 * (guess + x/guess)
		if next == guess {
			break
		}
		guess = next
	}
	return guess
}

// compactionCacheScheme attaches the metal CacheCompute surface to the
// "compaction" catalogue entry. Registering it overwrites the metadata-only
// entry seeded for the mode and gives the engine the Attention-Matching store.
//
// SECURITY-NOTE: this ships the AM-HighestAttnKeys *direct* variant (top-attention
// key selection + direct value take, beta=0), which composes entirely from the
// existing MLX ops in ops.go. The paper's lsq/NNLS refinement — a batched
// least-squares fit of the compacted values plus a learned per-key beta bias —
// needs a batched linear solver (torch.linalg.lstsq / cholesky / pinv) that this
// driver does not expose. Adding it would require a new cgo MLX linalg op; that
// is flagged, not implemented, per the task's "flag a new kernel rather than
// write one" instruction.
type compactionCacheScheme struct{}

func (compactionCacheScheme) Mode() string             { return string(KVCacheModeCompaction) }
func (compactionCacheScheme) Serves() scheme.StateKind { return scheme.StateKVCache }

// NewCache builds the Attention-Matching compaction store. MaxSize carries the
// compacted token budget; 0 falls back to defaultCompactionBudget. PageSize is
// unused by this scheme.
func (compactionCacheScheme) NewCache(p CacheParams) Cache {
	return NewCompactionKVCache(p.MaxSize, p.PageSize)
}

func init() { scheme.RegisterCache(compactionCacheScheme{}) }

// compile-time proof compactionCacheScheme is a full metal.CacheCompute, and
// CompactionKVCache is a full metal.Cache.
var (
	_ CacheCompute = compactionCacheScheme{}
	_ Cache        = (*CompactionKVCache)(nil)
)
