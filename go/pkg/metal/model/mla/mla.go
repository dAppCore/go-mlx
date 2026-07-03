// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

// Package mla implements Multi-head Latent Attention (DeepSeek-V2) as an
// engine sequence mixer. MLA is softmax-family attention whose distinguishing
// move is a low-rank KV compression: instead of caching full per-head K/V, the
// layer caches a single compressed latent c_kv = x·W_DKV and reconstructs K/V
// from it with up-projections (k = c_kv·W_UK, v = c_kv·W_UV). The query takes
// the same compress→expand treatment (c_q = x·W_DQ, q = c_q·W_UQ). The cache
// therefore holds the latent (one rank-r vector per token) rather than H×D
// floats, which is the whole point — a fraction of the KV footprint with no
// change to the attention score itself.
//
//	out, kv := (&Mixer{...}).Forward(x, &metal.MixerCtx{Cache: c, B: B, L: L, Mask: mask})
//
// The reference is gemma4/softmax_mixer.go + attention.go (the softmax-family
// mixer this mirrors); the compression math follows fla-org's MLA layer ported
// to MLX metal.Array ops (no Triton). State is scheme.StateKVCache: the cache
// layer holds the compressed latent and the engine pairs it with a KV-cache
// scheme at load.
package mla

import (
	metal "dappco.re/go/mlx/pkg/metal"
	scheme "dappco.re/go/mlx/pkg/scheme"
)

// MixerKind is the config token a model declares to select Multi-head Latent
// Attention. The engine resolves the mixer with scheme.MixerFor(mla.MixerKind).
const MixerKind = "mla"

// Mixer is the MLA sequence mixer. The projection weights are the low-rank
// down/up factors that define the compression; HeadDim/NumHeads/Scale are the
// attention geometry. A model build fills these from its config + safetensors;
// the kernel (attendLatent) is what the unit test exercises directly with
// hand-built fixtures so the math is checked without loading weights.
type Mixer struct {
	WDKV     *metal.Linear // down-projection x → compressed KV latent [hidden → kvLatentDim]
	WUK      *metal.Linear // up-projection latent → keys [kvLatentDim → NumHeads*HeadDim]
	WUV      *metal.Linear // up-projection latent → values [kvLatentDim → NumHeads*HeadDim]
	WDQ      *metal.Linear // down-projection x → compressed query latent [hidden → qLatentDim]
	WUQ      *metal.Linear // up-projection query latent → queries [qLatentDim → NumHeads*HeadDim]
	OProj    *metal.Linear // output projection [NumHeads*HeadDim → hidden]
	NumHeads int32         // number of attention heads
	HeadDim  int32         // per-head dimension
	Scale    float32       // attention score scale (typically 1/sqrt(HeadDim))
}

// Kind reports the config token this mixer answers to (the scheme registry key).
func (m *Mixer) Kind() string { return MixerKind }

// State declares MLA keeps a growing per-token cache — but of the compressed
// KV latent, not full K/V. It is scheme.StateKVCache because the cache layer
// (quant / compaction) still operates on a growing per-token tensor; the
// compression only shrinks that tensor's last dimension.
func (m *Mixer) State() scheme.StateKind { return scheme.StateKVCache }

// CacheMode names MLA's compressed-latent KV store ("mla-latent") so the cache
// factory (metal.NewCacheForMixer) builds it the latent scheme rather than a
// full-K/V cache. This is the metal.CacheModer surface the factory checks; only
// mixers with a bespoke cache shape implement it.
func (m *Mixer) CacheMode() string { return string(metal.KVCacheModeMLALatent) }

// Forward mixes one chunk. It computes the compressed query and KV latents,
// persists+concatenates the KV latent through ctx.Cache, reconstructs per-head
// K/V from the FULL cached latent, and runs softmax attention — so a decode step
// attends over the whole history, not just the chunk it was handed.
//
// The query stays the current chunk (L tokens); only K/V span the cached
// history (totalL tokens). When the caller supplies no mask, Forward builds the
// causal mask itself (as gemma4's attention does) via MultiTokenCausalMask with
// offset histLen = totalL - L, so row i (the query at absolute position
// histLen+i) attends keys 0..histLen+i: a [L,L] causal mask for prefill
// (histLen 0), the [L,totalL] offset-causal mask for chunked prefill over prior
// history (histLen>0), and no mask for single-token decode (L == 1, the lone
// query legitimately sees every cached key). A caller that DOES supply ctx.Mask
// owns its shape ([.,.,L,totalL]) and MLA honours it verbatim.
func (m *Mixer) Forward(x *metal.Array, ctx *metal.MixerCtx) (*metal.Array, metal.SharedKV) {
	B, L := ctx.B, ctx.L

	// Compress then expand the query: x → c_q → q.
	cQ := m.WDQ.Forward(x)
	qFlat := m.WUQ.Forward(cQ)
	metal.Free(cQ)

	// Compress this chunk's KV latent once; both K and V reconstruct from it
	// (MLA's footprint win). Persist+concatenate it through the cache so decode
	// attends the full history: the latent cache adopts cKV and returns its own
	// growing handle ([B, totalL, latentDim]) — we hand ownership over and read
	// the returned latent WITHOUT freeing it; only the uncached one-shot
	// transient is ours to release. With no RoPE the latent is
	// position-independent, so up-projecting the concatenated latent is per-row
	// identical to up-projecting a one-shot latent — what the decode oracle pins.
	cKV := m.WDKV.Forward(x)
	totalL := L
	if ctx.Cache != nil {
		cKV, _ = ctx.Cache.Update(cKV, cKV, int(L))
		totalL = int32(cKV.Dim(1))
	}
	kFlat, vFlat := m.upProjectKV(cKV, B, totalL)
	if ctx.Cache == nil {
		metal.Free(cKV)
	}

	// [B,·,H*D] → [B,H,·,D]: the query keeps L, K/V span the cached totalL.
	q := splitHeads(qFlat, B, L, m.NumHeads, m.HeadDim)
	metal.Free(qFlat)
	k := splitHeads(kFlat, B, totalL, m.NumHeads, m.HeadDim)
	metal.Free(kFlat)
	v := splitHeads(vFlat, B, totalL, m.NumHeads, m.HeadDim)
	metal.Free(vFlat)

	// Causal mask: honour an explicit ctx.Mask, else build the offset-causal mask
	// for this chunk over the cached history. L==1 decode needs none — the lone
	// query (the last position) sees every key. histLen = totalL - L is the
	// prior-token count: 0 for prefill ([L,L] causal), >0 for chunked prefill
	// ([L,totalL] offset-causal).
	mask := ctx.Mask
	var builtMask *metal.Array
	if mask == nil && L > 1 {
		offsetArr := metal.FromValue(int(totalL - L))
		builtMask = metal.MultiTokenCausalMask(int(totalL), offsetArr, int(L))
		metal.Free(offsetArr)
		mask = builtMask
	}

	out := attendLatent(q, k, v, mask, m.Scale)
	metal.Free(q, k, v)
	if builtMask != nil {
		metal.Free(builtMask)
	}

	// [B,H,L,D] → [B,L,H*D] → output projection.
	merged := mergeHeads(out, B, L, m.NumHeads, m.HeadDim)
	metal.Free(out)
	result := m.OProj.Forward(merged)
	metal.Free(merged)

	return result, metal.SharedKV{}
}

// upProjectKV reconstructs the per-token K and V activations from the compressed
// KV latent. DeepSeek-V2 packs both into ONE up-projection (kv_b_proj) whose
// output is PER-HEAD interleaved, not block-concatenated: the reference does
// kv_b_proj(...).view(B, L, num_heads, qk_nope_head_dim + v_head_dim) then
// splits the LAST axis into k_nope and value. So head h's K and V are collocated
// — [h0_K, h0_V, h1_K, h1_V, …], NOT [all-K | all-V]. upProjectKV reproduces
// that: reshape the [B,L,heads*2*HeadDim] projection to [B,L,heads,2*HeadDim],
// slice the head axis's last dim into K (first HeadDim) and V (last HeadDim),
// and flatten each back to [B,L,heads*HeadDim] for splitHeads. (This mixer uses
// one HeadDim for both K and V; a model with distinct qk/v head dims carries
// distinct WUK/WUV and takes the else branch.) A block-layout split here would
// pair head h's K with head h+1's V — silently wrong, shape-identical.
func (m *Mixer) upProjectKV(cKV *metal.Array, B, L int32) (kFlat, vFlat *metal.Array) {
	if m.WUV == nil || m.WUV == m.WUK {
		kv := m.WUK.Forward(cKV) // [B,L,heads*2*HeadDim], per-head interleaved
		perHead := metal.Reshape(kv, B, L, m.NumHeads, 2*m.HeadDim)
		metal.Free(kv)
		// Slice the per-head K (first HeadDim) and V (last HeadDim) sub-vectors,
		// then flatten the head axis back so splitHeads sees [B,L,heads*HeadDim].
		// Slice4 passes the 8 bounds as scalars (mlx_slice_inline_4): the rank-4
		// single-axis split that metal.Slice's []int32{…} literals heap-alloc on
		// every Forward (prefill + each decode step) becomes 0-alloc here.
		kHeads := metal.Slice4(perHead, 0, 0, 0, 0, B, L, m.NumHeads, m.HeadDim)
		vHeads := metal.Slice4(perHead, 0, 0, 0, m.HeadDim, B, L, m.NumHeads, 2*m.HeadDim)
		metal.Free(perHead)
		kFlat = metal.Reshape(kHeads, B, L, m.NumHeads*m.HeadDim)
		vFlat = metal.Reshape(vHeads, B, L, m.NumHeads*m.HeadDim)
		metal.Free(kHeads, vHeads)
		return kFlat, vFlat
	}
	return m.WUK.Forward(cKV), m.WUV.Forward(cKV)
}

// attendLatent is the MLA attention kernel: standard softmax attention over the
// reconstructed per-head Q/K/V. MLA does not change the score — the compression
// happens upstream — so this is the same Q·Kᵀ → scale → (+mask) → softmax → ·V
// decomposition the metal SDPA paths use, written in metal.Array ops so the
// unit test can pin it on a fixed input. q/k/v are [B,H,L,D]; mask (additive,
// [.,.,L,L]) may be nil.
//
//	out := attendLatent(q, k, v, mask, 1.0/float32(math.Sqrt(float64(headDim))))
func attendLatent(q, k, v, mask *metal.Array, scale float32) *metal.Array {
	kT := metal.Transpose4(k, 0, 1, 3, 2) // [B,H,D,L]
	scores := metal.Matmul(q, kT)         // [B,H,L,L]
	metal.Free(kT)

	if scale != 1 {
		scaled := metal.MulScalar(scores, scale)
		metal.Free(scores)
		scores = scaled
	}
	if mask != nil {
		masked := metal.Add(scores, mask)
		metal.Free(scores)
		scores = masked
	}

	probs := metal.Softmax(scores) // softmax over last axis (keys)
	metal.Free(scores)
	out := metal.Matmul(probs, v) // [B,H,L,D]
	metal.Free(probs)
	return out
}

// splitHeads reshapes a packed [B,L,H*D] projection into the [B,H,L,D] attention
// layout (transpose of the head and sequence axes).
func splitHeads(x *metal.Array, B, L, heads, dim int32) *metal.Array {
	r := metal.Reshape(x, B, L, heads, dim) // [B,L,H,D]
	t := metal.Transpose4(r, 0, 2, 1, 3)    // [B,H,L,D]
	metal.Free(r)
	return t
}

// mergeHeads is the inverse of splitHeads: [B,H,L,D] → [B,L,H*D].
func mergeHeads(x *metal.Array, B, L, heads, dim int32) *metal.Array {
	t := metal.Transpose4(x, 0, 2, 1, 3) // [B,L,H,D]
	r := metal.Reshape(t, B, L, heads*dim)
	metal.Free(t)
	return r
}

// compile-time proof Mixer is a full metal.MixerCompute.
var _ metal.MixerCompute = (*Mixer)(nil)
