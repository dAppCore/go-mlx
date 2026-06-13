// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

// Package nsa implements Native Sparse Attention as an engine sequence mixer.
// NSA (Yuan et al. 2025) is softmax-family attention split into three branches
// that a learned per-token sigmoid gate blends:
//
//   - compression — mean-pool K/V into non-overlapping blocks, attend the query
//     over the coarse per-block summaries (cheap global context);
//   - selection   — score every block with the compression attention, keep the
//     top-n blocks per query, attend over the original tokens of those blocks
//     (fine detail where it matters);
//   - sliding     — plain local causal attention over the last `window` tokens
//     (recency the block grid would smear).
//
// The output is g_cmp·o_cmp + g_slc·o_slc + g_swa·o_swa.
//
//	out, kv := (&Mixer{...}).Forward(x, &metal.MixerCtx{Cache: c, B: B, L: L})
//
// The reference is fla-org's NativeSparseAttention layer ported to MLX
// metal.Array ops (no Triton). The three branch kernels and the top-n selection
// mask are composed from existing ops — see selectionMask for how the top-n
// keep set is expressed with TopK+Greater rather than a discrete gather (no new
// Metal kernel). State is scheme.StateKVCache.
package nsa

import (
	"math"

	metal "dappco.re/go/mlx/pkg/metal"
	scheme "dappco.re/go/mlx/pkg/scheme"
)

// MixerKind is the config token a model declares to select Native Sparse
// Attention. The engine resolves the mixer with scheme.MixerFor(nsa.MixerKind).
const MixerKind = "nsa"

// negInf is the additive-mask "forbidden" fill; softmax drives these to ~0.
var negInf = float32(math.Inf(-1))

// Mixer is the NSA sequence mixer. QKV/gate/output projections plus the sparse
// geometry (BlockSize, SelectBlocks top-n, Window). A model build fills the
// weights + geometry from its config; the branch kernels (compressKV,
// blockCausalMask, slidingMask, selectionMask, attend) are what the unit test
// pins directly with hand-built fixtures.
type Mixer struct {
	QProj    *metal.Linear // [hidden → NumHeads*HeadDim]
	KProj    *metal.Linear // [hidden → NumHeads*HeadDim]
	VProj    *metal.Linear // [hidden → NumHeads*HeadDim]
	GProj    *metal.Linear // gate logits [hidden → NumHeads*3] (cmp, slc, swa per head)
	OProj    *metal.Linear // [NumHeads*HeadDim → hidden]
	NumHeads int32         // attention heads
	HeadDim  int32         // per-head dimension
	BlockSize int32        // tokens per compression/selection block
	SelectBlocks int32     // top-n blocks the selection branch keeps per query
	Window   int32         // sliding-window span (tokens)
	Scale    float32       // attention score scale (1/sqrt(HeadDim))
}

// Kind reports the config token this mixer answers to.
func (m *Mixer) Kind() string { return MixerKind }

// State declares NSA keeps a growing per-layer K/V cache (the selection and
// sliding branches read original tokens; compression derives from them).
func (m *Mixer) State() scheme.StateKind { return scheme.StateKVCache }

// Forward mixes one chunk by running the three branches and blending them with
// the per-token sigmoid gate.
//
// Decode-time cache concatenation (attending over history) is the engine's
// KV-cache responsibility (ctx.Cache) and is wired when #1's decoder
// integration lands; until then Forward attends within the chunk it is handed.
// The branch math + gate blend below are final and are what the kernel tests
// pin.
func (m *Mixer) Forward(x *metal.Array, ctx *metal.MixerCtx) (*metal.Array, metal.SharedKV) {
	B, L := ctx.B, ctx.L

	qFlat := m.QProj.Forward(x)
	kFlat := m.KProj.Forward(x)
	vFlat := m.VProj.Forward(x)
	q := splitHeads(qFlat, B, L, m.NumHeads, m.HeadDim)
	k := splitHeads(kFlat, B, L, m.NumHeads, m.HeadDim)
	v := splitHeads(vFlat, B, L, m.NumHeads, m.HeadDim)
	metal.Free(qFlat, kFlat, vFlat)

	oCmp, oSlc := m.compressionAndSelection(q, k, v, L)
	oSwa := m.sliding(q, k, v, L)
	metal.Free(q, k, v)

	// Gate: sigmoid over the [B,L,H,3] projection; one scalar per (head,branch).
	gFlat := m.GProj.Forward(x)            // [B,L,H*3]
	gates := metal.Sigmoid(gFlat)          // gate ∈ (0,1)
	metal.Free(gFlat)
	out := m.blendBranches(oCmp, oSlc, oSwa, gates, B, L)
	metal.Free(oCmp, oSlc, oSwa, gates)

	merged := mergeHeads(out, B, L, m.NumHeads, m.HeadDim)
	metal.Free(out)
	result := m.OProj.Forward(merged)
	metal.Free(merged)
	return result, metal.SharedKV{}
}

// compressionAndSelection runs the compression branch (attend over mean-pooled
// blocks) and the selection branch (attend over the top-n blocks' original
// tokens), sharing the per-block scores between them. q/k/v are [B,H,L,D].
func (m *Mixer) compressionAndSelection(q, k, v *metal.Array, L int32) (oCmp, oSlc *metal.Array) {
	kCmp, vCmp := compressKV(k, v, m.BlockSize) // [B,H,nBlocks,D]
	nBlocks := int32(kCmp.Dim(2))

	// Compression branch: query attends over the per-block summaries, masked so
	// a query at token t may only see blocks whose first token index <= t.
	kCmpT := metal.Transpose4(kCmp, 0, 1, 3, 2) // [B,H,D,nBlocks]
	blockScores := metal.Matmul(q, kCmpT)        // [B,H,L,nBlocks]
	metal.Free(kCmpT)
	scaled := metal.MulScalar(blockScores, m.Scale)
	metal.Free(blockScores)
	blockScores = scaled

	cmpMask := blockCausalMask(L, nBlocks, m.BlockSize) // [1,1,L,nBlocks]
	maskedCmp := metal.Add(blockScores, cmpMask)
	cmpProbs := metal.Softmax(maskedCmp)
	metal.Free(maskedCmp)
	oCmp = metal.Matmul(cmpProbs, vCmp) // [B,H,L,D]
	metal.Free(cmpProbs, kCmp, vCmp)

	// Selection branch: rank blocks by the (masked) block scores, keep the top-n
	// per query, expand the kept-block set to a per-token additive mask, attend
	// over the ORIGINAL tokens of the kept blocks.
	selMaskBlocks := selectionMask(blockScores, cmpMask, m.SelectBlocks) // [B,H,L,nBlocks] additive
	metal.Free(blockScores, cmpMask)
	tokenMask := expandBlockMaskToTokens(selMaskBlocks, m.BlockSize, L) // [B,H,L,L]
	metal.Free(selMaskBlocks)
	oSlc = attend(q, k, v, tokenMask, m.Scale)
	metal.Free(tokenMask)
	return oCmp, oSlc
}

// sliding runs the sliding-window branch: local causal attention over the last
// Window tokens. q/k/v are [B,H,L,D].
func (m *Mixer) sliding(q, k, v *metal.Array, L int32) *metal.Array {
	mask := slidingMask(L, m.Window) // [1,1,L,L]
	out := attend(q, k, v, mask, m.Scale)
	metal.Free(mask)
	return out
}

// blendBranches forms g_cmp·o_cmp + g_slc·o_slc + g_swa·o_swa. gates is the
// [B,L,H*3] sigmoid projection; it is reshaped to broadcast one scalar per
// (head, branch) across the [B,H,L,D] branch outputs.
func (m *Mixer) blendBranches(oCmp, oSlc, oSwa, gates *metal.Array, B, L int32) *metal.Array {
	// [B,L,H,3] → per-branch [B,H,L,1] gate views that broadcast over D.
	g := metal.Reshape(gates, B, L, m.NumHeads, 3)
	gCmp := branchGate(g, B, L, m.NumHeads, 0)
	gSlc := branchGate(g, B, L, m.NumHeads, 1)
	gSwa := branchGate(g, B, L, m.NumHeads, 2)
	metal.Free(g)

	wCmp := metal.Mul(oCmp, gCmp)
	wSlc := metal.Mul(oSlc, gSlc)
	wSwa := metal.Mul(oSwa, gSwa)
	metal.Free(gCmp, gSlc, gSwa)

	sum1 := metal.Add(wCmp, wSlc)
	out := metal.Add(sum1, wSwa)
	metal.Free(wCmp, wSlc, wSwa, sum1)
	return out
}

// branchGate slices branch b's scalar gate out of the [B,L,H,3] projection and
// reshapes it to [B,H,L,1] so it broadcasts over the head dimension D.
func branchGate(g *metal.Array, B, L, heads, b int32) *metal.Array {
	slc := metal.SliceAxis(g, 3, b, b+1)     // [B,L,H,1]
	r := metal.Reshape(slc, B, L, heads, 1)  // [B,L,H,1]
	metal.Free(slc)
	t := metal.Transpose4(r, 0, 2, 1, 3)     // [B,H,L,1]
	metal.Free(r)
	return t
}

// compileTimeProof keeps the build honest.
var _ metal.MixerCompute = (*Mixer)(nil)
