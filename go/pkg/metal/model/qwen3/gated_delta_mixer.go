// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package qwen3

import (
	core "dappco.re/go"
	metal "dappco.re/go/mlx/pkg/metal"
	deltanet "dappco.re/go/mlx/pkg/metal/model/deltanet"
	flakernel "dappco.re/go/mlx/pkg/metal/model/internal/flakernel"
	scheme "dappco.re/go/mlx/pkg/scheme"
)

// gated_delta_mixer.go is the Qwen3.5/3.6 GatedDeltaNet linear-attention layer as
// a pluggable engine mixer — the full forward composing the part-A gated-delta
// kernel and the part-B decay/β resolution with the shared conv + gated-norm, in
// the exact Qwen3-Next order:
//
//	in_proj_qkv → causal conv → SiLU → split(q|k|v) → repeat q,k (key→value heads)
//	→ l2norm(q) → gated-delta-rule(q·scale, l2norm(k), v, β, α) → RMSNorm(o)·SiLU(z)
//	→ out_proj
//
// The kernel L2-normalises k internally; q is L2-normalised here and scaled by
// 1/√Dk (use_qk_l2norm_in_kernel). β = sigmoid(in_proj_b); α = exp(−exp(A_log)·
// softplus(in_proj_a + dt_bias)) (resolveDecay/resolveBeta). The gate is applied
// AFTER the norm (RMSNorm(o)·SiLU(z) — Qwen3NextRMSNormGated), so flakernel's
// gated norm fits directly. Two recurrent slots thread decode: the conv-state ring
// and the [B,H,Dk,Dv] delta memory.

// GatedDeltaConfig is the per-layer GatedDeltaNet geometry. q/k carry KeyHeads
// heads of KeyHeadDim; v carries ValueHeads of ValueHeadDim. q,k are repeated
// ValueHeads/KeyHeads times to run the recurrence at ValueHeads heads (linear-
// attention GQA). On the 27B: 16 key / 48 value heads, dim 128, conv kernel 4.
type GatedDeltaConfig struct {
	Hidden       int32
	KeyHeads     int32
	KeyHeadDim   int32
	ValueHeads   int32
	ValueHeadDim int32
	ConvKernel   int32
	RMSNormEps   float32
}

func (c *GatedDeltaConfig) qDim() int32    { return c.KeyHeads * c.KeyHeadDim }
func (c *GatedDeltaConfig) vDim() int32    { return c.ValueHeads * c.ValueHeadDim }
func (c *GatedDeltaConfig) convDim() int32 { return 2*c.qDim() + c.vDim() } // q | k | v
func (c *GatedDeltaConfig) scale() float32 { return flakernel.DefaultScale(c.KeyHeadDim) }

// GatedDeltaWeights is the per-layer linear_attn weight set: the fused QKV
// projection + its causal conv, the decay (a) / write-strength (b) / gate (z)
// projections, the A_log + dt_bias decay parameters, the gated-norm weight and
// the output projection.
type GatedDeltaWeights struct {
	InProjQKV  *metal.Linear // [convDim, hidden]
	InProjA    *metal.Linear // [valueHeads, hidden] → decay input
	InProjB    *metal.Linear // [valueHeads, hidden] → write-strength input
	InProjZ    *metal.Linear // [vDim, hidden] → output gate
	ConvWeight *metal.Array  // [convDim, 1, convKernel]
	ConvBias   *metal.Array  // [convDim] or nil
	ALog       *metal.Array  // [valueHeads]
	DtBias     *metal.Array  // [valueHeads] or nil
	Norm       *metal.Array  // [valueHeadDim] gated-RMSNorm weight
	OutProj    *metal.Linear // [hidden, vDim]
	cfg        *GatedDeltaConfig
}

// GatedDeltaMixer is the compute-bearing GatedDeltaNet scheme: the per-layer
// weights + the MixerCompute surface. The loader builds one per linear-attention
// layer and registers the family kind once.
type GatedDeltaMixer struct {
	W *GatedDeltaWeights
}

// NewGatedDeltaMixer builds a GatedDeltaNet mixer from a loaded weight set + geometry.
func NewGatedDeltaMixer(w *GatedDeltaWeights, cfg *GatedDeltaConfig) *GatedDeltaMixer {
	if w != nil {
		w.cfg = cfg
	}
	return &GatedDeltaMixer{W: w}
}

// Kind is the config token the engine resolves with scheme.MixerFor("gated_deltanet").
func (m *GatedDeltaMixer) Kind() string { return "gated_deltanet" }

// State declares the mixer keeps a fixed-size recurrent state (conv ring + delta
// memory), so the engine pairs it with the recurrent cache scheme.
func (m *GatedDeltaMixer) State() scheme.StateKind { return scheme.StateRecurrent }

// the two recurrent slots: the causal-conv ring and the delta value-memory.
const (
	gdSlotConv  = 0
	gdSlotDelta = 1
	gdNumSlots  = 2
)

// Forward mixes one chunk of hidden states x [B,L,D] through the GatedDeltaNet
// layer, threading the [conv-state, delta-state] slots from ctx's recurrent
// holder. With no holder it runs the stateless prefill path (a direct unit test).
func (m *GatedDeltaMixer) Forward(x *metal.Array, ctx *metal.MixerCtx) (*metal.Array, metal.SharedKV) {
	if m.W == nil || m.W.cfg == nil {
		return nil, metal.SharedKV{}
	}
	cfg := m.W.cfg
	B, L := ctx.B, ctx.L

	var priorConv, priorDelta *metal.Array
	if rc, ok := ctx.Recurrent(); ok {
		if slots := rc.RecurrentState(); len(slots) == gdNumSlots {
			priorConv, priorDelta = slots[gdSlotConv], slots[gdSlotDelta]
		}
	}

	// in_proj_qkv → causal depthwise conv (conv-state ring) → SiLU → split q|k|v.
	qkv := m.W.InProjQKV.Forward(x) // [B,L,convDim]
	convOut, newConv := flakernel.CausalDepthwiseConv(qkv, priorConv, m.W.ConvWeight, m.W.ConvBias, cfg.convDim(), cfg.ConvKernel, B, L)
	metal.Free(qkv)
	activated := metal.SiLU(convOut)
	metal.Free(convOut)
	qDim, vDim := cfg.qDim(), cfg.vDim()
	qFlat := metal.SliceAxis(activated, 2, 0, qDim)
	kFlat := metal.SliceAxis(activated, 2, qDim, 2*qDim)
	vFlat := metal.SliceAxis(activated, 2, 2*qDim, 2*qDim+vDim)
	metal.Free(activated)

	// per-head layout; repeat q,k from key heads to value heads; l2-normalise q
	// (the kernel l2-normalises k itself, and scales q by 1/√Dk).
	q := groupedHeads(qFlat, B, L, cfg.KeyHeads, cfg.KeyHeadDim, cfg.ValueHeads) // [B,VH,L,Dk]
	k := groupedHeads(kFlat, B, L, cfg.KeyHeads, cfg.KeyHeadDim, cfg.ValueHeads)
	v := toHeads(vFlat, B, L, cfg.ValueHeads, cfg.ValueHeadDim) // [B,VH,L,Dv]
	metal.Free(qFlat, kFlat, vFlat)
	qn := flakernel.L2NormalizeLastAxis(q, 1e-6)
	metal.Free(q)

	// decay α + write-strength β [B,VH,L,1].
	aProj := m.W.InProjA.Forward(x)
	alpha := resolveDecay(aProj, m.W.ALog, m.W.DtBias, B, L, cfg.ValueHeads)
	metal.Free(aProj)
	bProj := m.W.InProjB.Forward(x)
	beta := resolveBeta(bProj, B, L, cfg.ValueHeads)
	metal.Free(bProj)

	// gated delta rule (k L2-normalised inside the kernel; q scaled by 1/√Dk).
	o, newDelta := deltanet.GatedDeltaRuleChunkSequential(qn, k, v, beta, alpha, priorDelta, cfg.scale(), 0)
	metal.Free(qn, k, v, alpha, beta)
	if o == nil {
		metal.Free(newConv, newDelta)
		return nil, metal.SharedKV{}
	}

	// merge heads [B,VH,L,Dv] → [B,L,VH,Dv]; gated RMSNorm with z (gate after norm);
	// flatten + out-proj.
	oHeads := metal.Transpose4(o, 0, 2, 1, 3) // [B,L,VH,Dv]
	metal.Free(o)
	z := m.W.InProjZ.Forward(x) // [B,L,vDim]
	zHeads := metal.Reshape(z, B, L, cfg.ValueHeads, cfg.ValueHeadDim)
	metal.Free(z)
	gated := flakernel.GatedRMSNorm(oHeads, zHeads, m.W.Norm, cfg.RMSNormEps) // RMSNorm(o)·SiLU(z)
	metal.Free(oHeads, zHeads)
	gatedFlat := metal.Reshape(gated, B, L, vDim)
	metal.Free(gated)
	out := m.W.OutProj.Forward(gatedFlat) // [B,L,hidden]
	metal.Free(gatedFlat)

	// write the advanced [conv-state, delta-state] slots back to the holder.
	if rc, ok := ctx.Recurrent(); ok {
		if err := metal.Eval(newConv, newDelta); err != nil {
			core.Error("qwen3 gated-delta: state eval failed", "error", err)
		}
		rc.SetRecurrentState([]*metal.Array{newConv, newDelta})
	} else {
		metal.Free(newConv, newDelta)
	}
	return out, metal.SharedKV{}
}

// toHeads views a per-token projection [B,L,H*D] as the per-head decode layout
// [B,H,L,D] via a strided view (no copy), mirroring deltanet/gemma4.
func toHeads(x *metal.Array, b, l, h, d int32) *metal.Array {
	return metal.AsStrided(x, []int32{b, h, l, d},
		[]int64{int64(l * h * d), int64(d), int64(h * d), 1}, 0)
}

// groupedHeads expands a key/query projection [B,L,KH*D] to the value-head decode
// layout [B,VH,L,D], repeating each key head (VH/KH) times — value head j reads
// key head j/(VH/KH), the consecutive repeat_kv expansion (matching Qwen's
// repeat_interleave on the head axis).
func groupedHeads(x *metal.Array, b, l, kh, d, vh int32) *metal.Array {
	rep := vh / kh
	grouped := metal.Reshape(x, b, l, kh, 1, d)                    // [B,L,KH,1,D]
	bcast := metal.BroadcastTo(grouped, []int32{b, l, kh, rep, d}) // [B,L,KH,rep,D]
	metal.Free(grouped)
	heads := metal.Reshape(bcast, b, l, vh, d) // [B,L,VH,D]
	metal.Free(bcast)
	out := metal.Transpose4(heads, 0, 2, 1, 3) // [B,VH,L,D]
	metal.Free(heads)
	return out
}

// compile-time proof *GatedDeltaMixer is a full metal.MixerCompute.
var _ metal.MixerCompute = (*GatedDeltaMixer)(nil)

func init() {
	// register the family identity + state contract so scheme.MixerFor
	// ("gated_deltanet") resolves; a loaded model overwrites this with a
	// compute-bearing *GatedDeltaMixer carrying its weights.
	scheme.RegisterMixer(gatedDeltaFamily{})
}

type gatedDeltaFamily struct{}

func (gatedDeltaFamily) Kind() string            { return "gated_deltanet" }
func (gatedDeltaFamily) State() scheme.StateKind { return scheme.StateRecurrent }
