// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"math"

	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/metal"
)

// The Gemma 4 audio tower is a Universal Speech Model Conformer encoder
// (Mantis #1839): a strided conv subsampler (time and mel each ÷4), then
// NumHiddenLayers macaron blocks of [½FFN → chunked relative-position
// self-attention → GLU causal conv → ½FFN], then output_proj into the
// multimodal embedding width. Every forward below mirrors the HF
// transformers Gemma4Audio* reference module-for-module — the maths is
// ported, not guessed (wrong relative shift or mask = silent garbage audio).
//
// The encoder consumes log-mel input features [B, frames, melBins] and
// returns [B, frames/4 (ceil), OutputProjDims]. The existing
// Gemma4AudioProjector (embed_audio.embedding_projection) then maps those
// into the decoder embedding space.

// Gemma4AudioClippableLinear is a linear whose input and output are clamped
// to checkpoint-recorded bounds (use_clipped_linears). The bound tensors are
// scalars shipped in the checkpoint; absent bounds mean no clamping.
type Gemma4AudioClippableLinear struct {
	Linear    *metal.Linear
	InputMin  *metal.Array
	InputMax  *metal.Array
	OutputMin *metal.Array
	OutputMax *metal.Array
}

func (l *Gemma4AudioClippableLinear) Forward(x *metal.Array) *metal.Array {
	in := x
	if l.InputMin != nil && l.InputMax != nil {
		in = metal.Clip(x, l.InputMin, l.InputMax)
	}
	out := l.Linear.Forward(in)
	if in != x {
		metal.Free(in)
	}
	if l.OutputMin != nil && l.OutputMax != nil {
		clipped := metal.Clip(out, l.OutputMin, l.OutputMax)
		metal.Free(out)
		out = clipped
	}
	return out
}

// Gemma4AudioSubSampleConvLayer is one conv2d (3×3, stride 2×2, pad 1) +
// scale-only LayerNorm over channels + ReLU. MLX convs are NHWC, so the
// channel axis is already last — the reference's permute pair collapses away.
type Gemma4AudioSubSampleConvLayer struct {
	ConvWeight *metal.Array // [outC, 3, 3, inC] (transposed from torch at load)
	NormWeight *metal.Array // [outC]
	NormBias   *metal.Array // zeros (scale-only LayerNorm; fast kernel wants a bias)
	Eps        float32
}

func (l *Gemma4AudioSubSampleConvLayer) Forward(x *metal.Array) *metal.Array {
	conv := metal.Conv2d(x, l.ConvWeight, 2, 2, 1, 1, 1, 1, 1)
	normed := metal.LayerNorm(conv, l.NormWeight, l.NormBias, l.Eps)
	metal.Free(conv)
	zero := metal.FromValue(float32(0))
	out := metal.Maximum(normed, zero)
	metal.Free(normed, zero)
	return out
}

// Gemma4AudioSubSampleConvProjection stacks the two strided conv layers and
// projects the flattened (freq × channels) features to the encoder width.
type Gemma4AudioSubSampleConvProjection struct {
	Layer0    *Gemma4AudioSubSampleConvLayer
	Layer1    *Gemma4AudioSubSampleConvLayer
	InputProj *metal.Linear
}

// Forward consumes [B, frames, melBins] and returns [B, ceil(frames/4), hidden].
func (p *Gemma4AudioSubSampleConvProjection) Forward(features *metal.Array) *metal.Array {
	shape := features.Shape()
	x := metal.Reshape(features, shape[0], shape[1], shape[2], 1) // NHWC, C=1
	h0 := p.Layer0.Forward(x)
	metal.Free(x)
	h1 := p.Layer1.Forward(h0)
	metal.Free(h0)
	hs := h1.Shape() // [B, T', F', C']
	flat := metal.Reshape(h1, hs[0], hs[1], hs[2]*hs[3])
	metal.Free(h1)
	out := p.InputProj.Forward(flat)
	metal.Free(flat)
	return out
}

// Gemma4AudioAttention is chunked local self-attention with Transformer-XL
// relative position bias, per-dim query scaling and a tanh logit soft-cap.
type Gemma4AudioAttention struct {
	QProj *Gemma4AudioClippableLinear
	KProj *Gemma4AudioClippableLinear
	VProj *Gemma4AudioClippableLinear
	Post  *Gemma4AudioClippableLinear

	RelativeKProj *metal.Linear
	// QScalePerDim folds q_scale (head_dim^-0.5 / ln 2) into
	// softplus(per_dim_scale), precomputed at load: [1, 1, 1, headDim].
	QScalePerDim *metal.Array
	// PosEmbed is the host-built sinusoid table the relative bias projects:
	// [ContextLeft, hidden] for positions [ContextLeft-1 .. 0].
	PosEmbed *metal.Array

	NumHeads  int32
	HeadDim   int32
	ChunkSize int32
	// PastHorizon = ContextLeft-1, FutureHorizon = ContextRight.
	PastHorizon   int32
	FutureHorizon int32
	KScale        float32
	LogitCap      float32
	InvalidLogit  float32
}

func (a *Gemma4AudioAttention) contextSize() int32 {
	return a.ChunkSize + a.PastHorizon + a.FutureHorizon
}

// blockedMask builds the validity mask for the blocked context windows:
// query at global position q = blk*chunk + i may attend key at global
// kv = blk*chunk - past + j iff both are in-sequence and kv lies inside
// [q-past, q+future]. Host-built per length; tiny (nB×chunk×context bools).
func (a *Gemma4AudioAttention) blockedMask(seqLen, numBlocks int32) *metal.Array {
	chunk, ctx := a.ChunkSize, a.contextSize()
	vals := make([]float32, numBlocks*chunk*ctx)
	idx := 0
	for blk := int32(0); blk < numBlocks; blk++ {
		for i := int32(0); i < chunk; i++ {
			q := blk*chunk + i
			for j := int32(0); j < ctx; j++ {
				kv := blk*chunk - a.PastHorizon + j
				if q < seqLen && kv >= 0 && kv < seqLen && kv >= q-a.PastHorizon && kv <= q+a.FutureHorizon {
					vals[idx] = 1
				}
				idx++
			}
		}
	}
	return metal.FromValues(vals, 1, 1, int(numBlocks), int(chunk), int(ctx))
}

// extractBlockContext pads the sequence axis by [past, future+chunk-1] and
// unfolds overlapping context windows strided by chunk:
// [B, T, H, D] → [B, nB, context, H, D]. Zero-copy via AsStrided.
func (a *Gemma4AudioAttention) extractBlockContext(x *metal.Array, numBlocks int32) *metal.Array {
	padded := metal.PadAxis(x, 1, int(a.PastHorizon), int(a.FutureHorizon+a.ChunkSize-1))
	ps := padded.Shape() // [B, Tp, H, D]
	b, tp, h, d := ps[0], ps[1], ps[2], ps[3]
	rowH := int64(h * d)
	strides := []int64{int64(tp) * rowH, int64(a.ChunkSize) * rowH, rowH, int64(d), 1}
	out := metal.AsStrided(padded, []int32{b, numBlocks, a.contextSize(), h, d}, strides, 0)
	contig := metal.Contiguous(out)
	metal.Free(padded, out)
	return contig
}

// relShift aligns the relative-position logits with the context windows
// (Transformer-XL appendix B): pad the position axis to context+1, fold,
// truncate, refold. [B, H, nB, chunk, P] → [B, H, nB, chunk, context].
func (a *Gemma4AudioAttention) relShift(x *metal.Array) *metal.Array {
	s := x.Shape() // [B, H, nB, chunk, P]
	ctx := a.contextSize()
	padded := metal.PadAxis(x, 4, 0, int(ctx+1-s[4]))
	folded := metal.Reshape(padded, s[0], s[1], s[2], s[3]*(ctx+1))
	metal.Free(padded)
	sliced := metal.SliceAxis(folded, -1, 0, s[3]*ctx)
	metal.Free(folded)
	out := metal.Reshape(sliced, s[0], s[1], s[2], s[3], ctx)
	metal.Free(sliced)
	return out
}

// Forward runs chunked relative attention over [B, T, hidden].
func (a *Gemma4AudioAttention) Forward(x *metal.Array) *metal.Array {
	s := x.Shape()
	batch, seqLen := s[0], s[1]
	numBlocks := (seqLen + a.ChunkSize - 1) / a.ChunkSize
	srcDtype := x.Dtype()

	// Projections, computed in float32 like the reference (.float()).
	project := func(p *Gemma4AudioClippableLinear) *metal.Array {
		raw := p.Forward(x)
		f32 := metal.AsType(raw, metal.DTypeFloat32)
		metal.Free(raw)
		out := metal.Reshape(f32, batch, seqLen, a.NumHeads, a.HeadDim)
		metal.Free(f32)
		return out
	}
	q := project(a.QProj)
	k := project(a.KProj)
	v := project(a.VProj)

	// q *= q_scale * softplus(per_dim_scale); k *= k_scale.
	qs := metal.Mul(q, a.QScalePerDim)
	metal.Free(q)
	ks := metal.MulScalar(k, a.KScale)
	metal.Free(k)

	// Blocked queries [B, nB, chunk, H, D] (pad T to a chunk multiple).
	qp := metal.PadAxis(qs, 1, 0, int(numBlocks*a.ChunkSize-seqLen))
	metal.Free(qs)
	qb := metal.Reshape(qp, batch, numBlocks, a.ChunkSize, a.NumHeads, a.HeadDim)
	metal.Free(qp)

	// Overlapping key/value context windows [B, nB, ctx, H, D].
	kc := a.extractBlockContext(ks, numBlocks)
	metal.Free(ks)
	vc := a.extractBlockContext(v, numBlocks)
	metal.Free(v)

	// matrix_ac = q · kᵀ over each window: [B, H, nB, chunk, ctx].
	qh := metal.Transpose(qb, 0, 3, 1, 2, 4) // [B, H, nB, chunk, D]
	metal.Free(qb)
	kh := metal.Transpose(kc, 0, 3, 1, 4, 2) // [B, H, nB, D, ctx]
	metal.Free(kc)
	matrixAC := metal.Matmul(qh, kh)
	metal.Free(kh)

	// matrix_bd = q · rel_kᵀ over the sinusoid positions, then rel-shifted.
	relK := a.RelativeKProj.Forward(a.PosEmbed) // [P, H*D]
	relKf := metal.AsType(relK, metal.DTypeFloat32)
	metal.Free(relK)
	posCount := a.PosEmbed.Dim(0)
	relKh := metal.Reshape(relKf, int32(posCount), a.NumHeads, a.HeadDim)
	metal.Free(relKf)
	relKt := metal.Transpose(relKh, 1, 2, 0) // [H, D, P]
	metal.Free(relKh)
	relKb := metal.ExpandDims(relKt, 0) // [1, H, D, P]
	metal.Free(relKt)

	qFlat := metal.Reshape(qh, batch, a.NumHeads, numBlocks*a.ChunkSize, a.HeadDim)
	metal.Free(qh)
	bdFlat := metal.Matmul(qFlat, relKb) // [B, H, nB*chunk, P]
	metal.Free(qFlat, relKb)
	bd := metal.Reshape(bdFlat, batch, a.NumHeads, numBlocks, a.ChunkSize, int32(posCount))
	metal.Free(bdFlat)
	bdShifted := a.relShift(bd)
	metal.Free(bd)

	logits := metal.Add(matrixAC, bdShifted)
	metal.Free(matrixAC, bdShifted)

	// tanh soft-cap, then mask invalid positions to the config logit floor.
	scaled := metal.MulScalar(logits, 1/a.LogitCap)
	metal.Free(logits)
	capped := metal.Tanh(scaled)
	metal.Free(scaled)
	soft := metal.MulScalar(capped, a.LogitCap)
	metal.Free(capped)

	mask := a.blockedMask(seqLen, numBlocks)
	invalid := metal.FromValue(a.InvalidLogit)
	masked := metal.Where(mask, soft, invalid)
	metal.Free(mask, invalid, soft)

	weights := metal.Softmax(masked) // float32 softmax over the context axis
	metal.Free(masked)

	vh := metal.Transpose(vc, 0, 3, 1, 2, 4) // [B, H, nB, ctx, D]
	metal.Free(vc)
	ctxOut := metal.Matmul(weights, vh) // [B, H, nB, chunk, D]
	metal.Free(weights, vh)

	merged := metal.Transpose(ctxOut, 0, 2, 3, 1, 4) // [B, nB, chunk, H, D]
	metal.Free(ctxOut)
	flat := metal.Reshape(merged, batch, numBlocks*a.ChunkSize, a.NumHeads*a.HeadDim)
	metal.Free(merged)
	trimmed := metal.SliceAxis(flat, 1, 0, seqLen)
	metal.Free(flat)
	cast := metal.AsType(trimmed, srcDtype)
	metal.Free(trimmed)
	out := a.Post.Forward(cast)
	metal.Free(cast)
	return out
}

// Gemma4AudioFeedForward is one macaron half-FFN: pre-norm → ffw (h→4h) →
// act → ffw (4h→h) → post-norm, residually combined at ResidualWeight.
type Gemma4AudioFeedForward struct {
	FFW1     *Gemma4AudioClippableLinear
	FFW2     *Gemma4AudioClippableLinear
	PreNorm  *metal.Array
	PostNorm *metal.Array
	Eps      float32
	Residual float32
	Act      string
	ClipMin  *metal.Array
	ClipMax  *metal.Array
}

func gemma4AudioActivate(x *metal.Array, act string) *metal.Array {
	switch act {
	case "silu", "swish", "":
		return metal.SiLU(x)
	case "relu":
		zero := metal.FromValue(float32(0))
		out := metal.Maximum(x, zero)
		metal.Free(zero)
		return out
	case "gelu", "gelu_pytorch_tanh":
		return metal.GeluActivation(x)
	default:
		panic(core.E("gemma4.audio", core.Sprintf("unsupported audio hidden_act %q", act), nil))
	}
}

func gemma4AudioClamp(x *metal.Array, minArr, maxArr *metal.Array) *metal.Array {
	if minArr == nil || maxArr == nil {
		return x.Clone()
	}
	return metal.Clip(x, minArr, maxArr)
}

func (f *Gemma4AudioFeedForward) Forward(x *metal.Array) *metal.Array {
	clamped := gemma4AudioClamp(x, f.ClipMin, f.ClipMax)
	pre := metal.RMSNorm(clamped, f.PreNorm, f.Eps)
	metal.Free(clamped)
	up := f.FFW1.Forward(pre)
	metal.Free(pre)
	activated := gemma4AudioActivate(up, f.Act)
	metal.Free(up)
	down := f.FFW2.Forward(activated)
	metal.Free(activated)
	downClamped := gemma4AudioClamp(down, f.ClipMin, f.ClipMax)
	metal.Free(down)
	post := metal.RMSNorm(downClamped, f.PostNorm, f.Eps)
	metal.Free(downClamped)
	scaled := metal.MulScalar(post, f.Residual)
	metal.Free(post)
	out := metal.Add(scaled, x)
	metal.Free(scaled)
	return out
}

// Gemma4AudioLightConv is the Conformer GLU conv module: pre-norm →
// linear_start (h→2h) → GLU → causal depthwise conv1d → conv-norm → act →
// linear_end → residual.
type Gemma4AudioLightConv struct {
	LinearStart *Gemma4AudioClippableLinear
	LinearEnd   *Gemma4AudioClippableLinear
	// DepthwiseWeight is [channels, kernel, 1] (transposed from torch at load).
	DepthwiseWeight *metal.Array
	PreNorm         *metal.Array
	ConvNorm        *metal.Array
	Eps             float32
	KernelSize      int32
	Channels        int32
	Act             string
	ClipMin         *metal.Array
	ClipMax         *metal.Array
}

func (c *Gemma4AudioLightConv) Forward(x *metal.Array) *metal.Array {
	pre := metal.RMSNorm(x, c.PreNorm, c.Eps)
	start := c.LinearStart.Forward(pre)
	metal.Free(pre)

	// GLU: first half gated by sigmoid of the second half.
	gate := metal.SliceAxis(start, -1, 0, c.Channels)
	gateIn := metal.SliceAxis(start, -1, c.Channels, 2*c.Channels)
	metal.Free(start)
	sig := metal.Sigmoid(gateIn)
	metal.Free(gateIn)
	glu := metal.Mul(gate, sig)
	metal.Free(gate, sig)

	// Causal depthwise conv over time: left-pad kernel-1, NLC layout.
	padded := metal.PadAxis(glu, 1, int(c.KernelSize-1), 0)
	metal.Free(glu)
	conv := metal.Conv1d(padded, c.DepthwiseWeight, 1, 0, 1, int(c.Channels))
	metal.Free(padded)

	clamped := gemma4AudioClamp(conv, c.ClipMin, c.ClipMax)
	metal.Free(conv)
	normed := metal.RMSNorm(clamped, c.ConvNorm, c.Eps)
	metal.Free(clamped)
	activated := gemma4AudioActivate(normed, c.Act)
	metal.Free(normed)
	end := c.LinearEnd.Forward(activated)
	metal.Free(activated)
	out := metal.Add(end, x)
	metal.Free(end)
	return out
}

// Gemma4AudioLayer is one Conformer block:
// ff1 → norm_pre_attn → attn → norm_post_attn (+res) → lconv → ff2 → norm_out.
type Gemma4AudioLayer struct {
	FeedForward1 *Gemma4AudioFeedForward
	FeedForward2 *Gemma4AudioFeedForward
	SelfAttn     *Gemma4AudioAttention
	LConv        *Gemma4AudioLightConv
	NormPreAttn  *metal.Array
	NormPostAttn *metal.Array
	NormOut      *metal.Array
	Eps          float32
	ClipMin      *metal.Array
	ClipMax      *metal.Array
}

func (l *Gemma4AudioLayer) Forward(x *metal.Array) *metal.Array {
	h := l.FeedForward1.Forward(x)

	clamped := gemma4AudioClamp(h, l.ClipMin, l.ClipMax)
	pre := metal.RMSNorm(clamped, l.NormPreAttn, l.Eps)
	metal.Free(clamped)
	attn := l.SelfAttn.Forward(pre)
	metal.Free(pre)
	attnClamped := gemma4AudioClamp(attn, l.ClipMin, l.ClipMax)
	metal.Free(attn)
	post := metal.RMSNorm(attnClamped, l.NormPostAttn, l.Eps)
	metal.Free(attnClamped)
	res := metal.Add(post, h)
	metal.Free(post, h)

	conv := l.LConv.Forward(res)
	metal.Free(res)
	ff2 := l.FeedForward2.Forward(conv)
	metal.Free(conv)

	outClamped := gemma4AudioClamp(ff2, l.ClipMin, l.ClipMax)
	metal.Free(ff2)
	out := metal.RMSNorm(outClamped, l.NormOut, l.Eps)
	metal.Free(outClamped)
	return out
}

// Gemma4AudioEncoder is the full audio tower: subsampler → Conformer layers →
// output projection into the multimodal embedding width.
type Gemma4AudioEncoder struct {
	Subsample  *Gemma4AudioSubSampleConvProjection
	Layers     []*Gemma4AudioLayer
	OutputProj *metal.Linear
	// PosEmbed is the host-built sinusoid relative-position table shared by
	// every layer's attention ([ContextLeft, hidden]).
	PosEmbed *metal.Array
	// GCMin/GCMax are the ±gradient_clipping clamp scalars every module
	// borrows; the encoder owns them (freed once on close).
	GCMin *metal.Array
	GCMax *metal.Array
	Cfg   *Gemma4AudioConfig
}

// Forward encodes log-mel features [B, frames, melBins] to
// [B, ceil(frames/4), OutputProjDims].
func (e *Gemma4AudioEncoder) Forward(features *metal.Array) *metal.Array {
	h := e.Subsample.Forward(features)
	for _, layer := range e.Layers {
		next := layer.Forward(h)
		metal.Free(h)
		h = next
	}
	out := e.OutputProj.Forward(h)
	metal.Free(h)
	return out
}

// gemma4AudioPositionTable hosts the sinusoid relative-position embeddings
// the reference computes on the fly: positions [count-1 .. 0], concatenated
// [sin..., cos...] over hidden/2 timescales. Returns [count, hidden].
func gemma4AudioPositionTable(count, hidden int32) *metal.Array {
	half := int(hidden) / 2
	logIncrement := math.Log(10000.0) / float64(max(half-1, 1))
	vals := make([]float32, int(count)*int(hidden))
	for p := 0; p < int(count); p++ {
		position := float64(int(count) - 1 - p)
		row := p * int(hidden)
		for i := 0; i < half; i++ {
			scaled := position * math.Exp(float64(i)*-logIncrement)
			vals[row+i] = float32(math.Sin(scaled))
			vals[row+half+i] = float32(math.Cos(scaled))
		}
	}
	return metal.FromValues(vals, int(count), int(hidden))
}
