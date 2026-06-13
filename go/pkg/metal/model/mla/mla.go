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
	WDKV    *metal.Linear // down-projection x → compressed KV latent [hidden → kvLatentDim]
	WUK     *metal.Linear // up-projection latent → keys [kvLatentDim → NumHeads*HeadDim]
	WUV     *metal.Linear // up-projection latent → values [kvLatentDim → NumHeads*HeadDim]
	WDQ     *metal.Linear // down-projection x → compressed query latent [hidden → qLatentDim]
	WUQ     *metal.Linear // up-projection query latent → queries [qLatentDim → NumHeads*HeadDim]
	OProj   *metal.Linear // output projection [NumHeads*HeadDim → hidden]
	NumHeads int32        // number of attention heads
	HeadDim  int32        // per-head dimension
	Scale    float32      // attention score scale (typically 1/sqrt(HeadDim))
}

// Kind reports the config token this mixer answers to (the scheme registry key).
func (m *Mixer) Kind() string { return MixerKind }

// State declares MLA keeps a growing per-token cache — but of the compressed
// KV latent, not full K/V. It is scheme.StateKVCache because the cache layer
// (quant / compaction) still operates on a growing per-token tensor; the
// compression only shrinks that tensor's last dimension.
func (m *Mixer) State() scheme.StateKind { return scheme.StateKVCache }

// Forward mixes one chunk. It computes the compressed query and KV latents,
// reconstructs per-head K/V from the cached latent, and runs softmax attention.
//
// The recurrent-state caching of the latent across decode steps is the
// engine's KV-cache responsibility (ctx.Cache) and is wired when #1's decoder
// integration lands; until then Forward attends within the chunk it is handed.
// The compression + attention math below is final and is what the kernel test
// pins.
func (m *Mixer) Forward(x *metal.Array, ctx *metal.MixerCtx) (*metal.Array, metal.SharedKV) {
	B, L := ctx.B, ctx.L

	// Compress then expand the query: x → c_q → q.
	cQ := m.WDQ.Forward(x)
	qFlat := m.WUQ.Forward(cQ)
	metal.Free(cQ)

	// Compress the KV once; both K and V reconstruct from the same latent.
	// TODO(#1 decoder integration): persist cKV into ctx.Cache and concatenate
	// the cached latent before the up-projection so decode attends over history.
	// The recurrent-latent holder is the engine's KV-cache layer, not this
	// mixer — same boundary the gemma4 reference draws.
	cKV := m.WDKV.Forward(x)
	kFlat, vFlat := m.upProjectKV(cKV, B, L)
	metal.Free(cKV)

	// [B,L,H*D] → [B,H,L,D] for all three.
	q := splitHeads(qFlat, B, L, m.NumHeads, m.HeadDim)
	metal.Free(qFlat)
	k := splitHeads(kFlat, B, L, m.NumHeads, m.HeadDim)
	metal.Free(kFlat)
	v := splitHeads(vFlat, B, L, m.NumHeads, m.HeadDim)
	metal.Free(vFlat)

	out := attendLatent(q, k, v, ctx.Mask, m.Scale)
	metal.Free(q, k, v)

	// [B,H,L,D] → [B,L,H*D] → output projection.
	merged := mergeHeads(out, B, L, m.NumHeads, m.HeadDim)
	metal.Free(out)
	result := m.OProj.Forward(merged)
	metal.Free(merged)

	return result, metal.SharedKV{}
}

// upProjectKV reconstructs the per-token K and V activations from the compressed
// KV latent. DeepSeek-V2 packs both into ONE up-projection (kv_b_proj), whose
// output is the K+V concatenation of width 2*NumHeads*HeadDim — the builder sets
// WUV == WUK in that case, and this slices the projection into its K half (first
// NumHeads*HeadDim columns) and V half (the rest). When the model carries
// distinct WUK / WUV projections (each width NumHeads*HeadDim), it runs them
// separately. Both yield flat [B,L,NumHeads*HeadDim] K and V for splitHeads.
func (m *Mixer) upProjectKV(cKV *metal.Array, B, L int32) (kFlat, vFlat *metal.Array) {
	width := m.NumHeads * m.HeadDim
	if m.WUV == nil || m.WUV == m.WUK {
		// Shared kv_b_proj → [B,L,2*width]; slice the K and V halves.
		kv := m.WUK.Forward(cKV)
		kFlat = metal.Slice(kv, []int32{0, 0, 0}, []int32{B, L, width})
		vFlat = metal.Slice(kv, []int32{0, 0, width}, []int32{B, L, 2 * width})
		metal.Free(kv)
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
	scores := metal.Matmul(q, kT)          // [B,H,L,L]
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
