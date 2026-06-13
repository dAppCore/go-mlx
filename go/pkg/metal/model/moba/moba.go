// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

// Package moba implements Mixture of Block Attention (Lu et al. 2025) as an
// engine sequence mixer. MoBA partitions the keys into fixed blocks and lets
// each query attend to only a few of them: it scores every block by dotting the
// query with the block's mean-pooled key (the block representative), keeps the
// top-k blocks, and ALWAYS keeps the block the query itself sits in (the
// self-block, causal-masked within itself). Blocks strictly in the future are
// forbidden. The kept set becomes an additive mask over the original tokens and
// a single softmax attention runs over it — so MoBA is exactly causal softmax
// attention with a learned, query-dependent block-sparsity mask.
//
//	out, kv := (&Mixer{...}).Forward(x, &metal.MixerCtx{Cache: c, B: B, L: L})
//
// Reference: MoonshotAI/MoBA moba_naive, ported to MLX metal.Array ops (no
// Triton). The block scoring + top-k mask (TopK+Greater, self-block forced in,
// tril causal) compose from existing ops — no new Metal kernel. State is
// scheme.StateKVCache.
package moba

import (
	"math"

	metal "dappco.re/go/mlx/pkg/metal"
	scheme "dappco.re/go/mlx/pkg/scheme"
)

// MixerKind is the config token a model declares to select Mixture of Block
// Attention. The engine resolves it with scheme.MixerFor(moba.MixerKind).
const MixerKind = "moba"

var negInf = float32(math.Inf(-1))

// Mixer is the MoBA sequence mixer. QKV/output projections plus the block
// geometry (BlockSize, TopK). A model build fills the weights + geometry from
// its config; the kernels (blockScores, blockSelectMask, expandToTokens,
// attend) are what the unit test pins with hand-built fixtures.
type Mixer struct {
	QProj    *metal.Linear // [hidden → NumHeads*HeadDim]
	KProj    *metal.Linear // [hidden → NumHeads*HeadDim]
	VProj    *metal.Linear // [hidden → NumHeads*HeadDim]
	OProj    *metal.Linear // [NumHeads*HeadDim → hidden]
	NumHeads int32         // attention heads
	HeadDim  int32         // per-head dimension
	BlockSize int32        // tokens per block
	TopK     int32         // number of blocks each query keeps (excluding the always-on self-block)
	Scale    float32       // attention score scale (1/sqrt(HeadDim))
}

// Kind reports the config token this mixer answers to.
func (m *Mixer) Kind() string { return MixerKind }

// State declares MoBA keeps a growing per-layer K/V cache — it attends over the
// original tokens of the selected blocks.
func (m *Mixer) State() scheme.StateKind { return scheme.StateKVCache }

// Forward mixes one chunk: project Q/K/V, score and select blocks per query,
// expand the selection to a token-level additive mask, and run one softmax
// attention over it.
//
// Decode-time cache concatenation (attending over history blocks) is the
// engine's KV-cache responsibility (ctx.Cache) and is wired when #1's decoder
// integration lands; until then Forward attends within the chunk it is handed.
// The block-selection + attention math below is final and is what the kernel
// tests pin.
func (m *Mixer) Forward(x *metal.Array, ctx *metal.MixerCtx) (*metal.Array, metal.SharedKV) {
	B, L := ctx.B, ctx.L

	qFlat := m.QProj.Forward(x)
	kFlat := m.KProj.Forward(x)
	vFlat := m.VProj.Forward(x)
	q := splitHeads(qFlat, B, L, m.NumHeads, m.HeadDim)
	k := splitHeads(kFlat, B, L, m.NumHeads, m.HeadDim)
	v := splitHeads(vFlat, B, L, m.NumHeads, m.HeadDim)
	metal.Free(qFlat, kFlat, vFlat)

	mask := m.buildMask(q, k, L)
	out := attend(q, k, v, mask, m.Scale)
	metal.Free(q, k, v, mask)

	merged := mergeHeads(out, B, L, m.NumHeads, m.HeadDim)
	metal.Free(out)
	result := m.OProj.Forward(merged)
	metal.Free(merged)
	return result, metal.SharedKV{}
}

// buildMask produces the [B,H,L,L] additive attention mask: top-k block
// selection (with the self-block always kept and future blocks forbidden)
// expanded to token granularity, then tril causal masking. q/k are [B,H,L,D].
//
// When the chunk is shorter than one block (L < BlockSize → nBlocks == 0, the
// common decode case until the first full block fills), there are no blocks to
// score or select: every token sits in the single partial self-block, so MoBA
// reduces to plain causal attention over all L tokens. buildMask returns that
// full causal mask directly rather than building a degenerate zero-block grid.
func (m *Mixer) buildMask(q, k *metal.Array, L int32) *metal.Array {
	if L < m.BlockSize {
		return causalMask(L, L) // every key j <= t allowed; no block structure yet
	}
	scores := blockScores(q, k, m.BlockSize, m.Scale) // [B,H,L,nBlocks]
	nBlocks := int32(scores.Dim(3))

	blockMask := blockSelectMask(scores, m.BlockSize, m.TopK, L) // [B,H,L,nBlocks] additive
	metal.Free(scores)
	tokenMask := expandToTokens(blockMask, m.BlockSize, L, nBlocks) // [B,H,L,L] additive
	metal.Free(blockMask)
	return tokenMask
}

// compileTimeProof keeps the build honest.
var _ metal.MixerCompute = (*Mixer)(nil)
