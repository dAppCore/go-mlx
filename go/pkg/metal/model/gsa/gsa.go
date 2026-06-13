// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

// Package gsa implements Gated Slot Attention (Zhang et al. 2024) as an engine
// sequence mixer. GSA is a recurrent (linear-attention) mechanism: it keeps a
// fixed-size memory of `slots` rather than a growing K/V cache, so its state
// kind is scheme.StateRecurrent. Each step writes the incoming key/value into
// the slot memory under a per-slot gate and reads the output back through a
// two-pass softmax over slots:
//
//	Sk_i = Sk_{i-1} ⊙ exp(g_i) + s_i ⊗ k_i      (key-slot memory)
//	Sv_i = Sv_{i-1} ⊙ exp(g_i) + s_i ⊗ v_i      (value-slot memory)
//	a_i  = softmax_slots(q_i · Sk_i)             (slot attention weights)
//	o_i  = a_i · Sv_i                            (read the value memory)
//
// where g_i = logsigmoid(f_i)/normaliser is the decay and s_i = 1 - exp(g_i) is
// the slot-write weight. The reference is fla-org's GSA layer + the naive GSA
// recurrence ported to MLX metal.Array ops (no Triton).
//
//	out, _ := (&Mixer{...}).Forward(x, &metal.MixerCtx{Cache: c, B: B, L: L})
//
// The recurrent-state holder (the StateRecurrent cache) is not built yet — the
// other recurrent mixer tasks (#2/#3) share the same TODO. Forward therefore
// starts each chunk from a zero state and marks where the cached state will be
// read/written once the holder lands; the recurrence kernel itself (recurrence)
// is final and is what the unit test pins.
package gsa

import (
	metal "dappco.re/go/mlx/pkg/metal"
	scheme "dappco.re/go/mlx/pkg/scheme"
)

// MixerKind is the config token a model declares to select Gated Slot
// Attention. The engine resolves it with scheme.MixerFor(gsa.MixerKind).
const MixerKind = "gsa"

// Mixer is the GSA sequence mixer. Q/K/V/forget/output projections plus the
// slot geometry (NumHeads, HeadK, HeadV, Slots) and the gate normaliser. A
// model build fills the weights + geometry from its config; the recurrence
// kernel is what the unit test pins with hand-built fixtures.
type Mixer struct {
	QProj    *metal.Linear // [hidden → NumHeads*HeadK]
	KProj    *metal.Linear // [hidden → NumHeads*HeadK]
	VProj    *metal.Linear // [hidden → NumHeads*HeadV]
	FProj    *metal.Linear // forget-gate logits [hidden → NumHeads*Slots]
	GProj    *metal.Linear // output gate logits [hidden → NumHeads*HeadV]
	OProj    *metal.Linear // [NumHeads*HeadV → hidden]
	NumHeads int32         // attention heads
	HeadK    int32         // per-head key/query dimension
	HeadV    int32         // per-head value dimension
	Slots    int32         // number of memory slots
	GateNorm float32       // gate_logit_normalizer (logsigmoid divisor)
}

// Kind reports the config token this mixer answers to.
func (m *Mixer) Kind() string { return MixerKind }

// State declares GSA keeps a fixed-size recurrent slot memory — scheme
// .StateRecurrent — so the engine pairs it with a recurrent-state cache scheme
// (not a growing K/V cache) at load.
func (m *Mixer) State() scheme.StateKind { return scheme.StateRecurrent }

// Forward mixes one chunk through the gated slot recurrence.
//
// TODO(#1 / recurrent-holder): read the initial (Sk, Sv) slot memory from
// ctx.Cache (a scheme.StateRecurrent holder) and write the final state back so
// decode continues across chunks. The holder is not built yet — the other
// recurrent mixers (GLA/RetNet/DeltaNet, Mamba2/RWKV7) share this TODO. Until it
// lands Forward starts from a zero slot memory, which is correct for a fresh
// prefill of the whole sequence. The recurrence math below is final.
func (m *Mixer) Forward(x *metal.Array, ctx *metal.MixerCtx) (*metal.Array, metal.SharedKV) {
	B, L := ctx.B, ctx.L

	qFlat := m.QProj.Forward(x) // [B,L,H*HeadK]
	kFlat := m.KProj.Forward(x)
	vFlat := m.VProj.Forward(x) // [B,L,H*HeadV]
	fFlat := m.FProj.Forward(x) // [B,L,H*Slots]

	q := splitHeads(qFlat, B, L, m.NumHeads, m.HeadK)
	k := splitHeads(kFlat, B, L, m.NumHeads, m.HeadK)
	v := splitHeads(vFlat, B, L, m.NumHeads, m.HeadV)
	f := splitHeads(fFlat, B, L, m.NumHeads, m.Slots)
	metal.Free(qFlat, kFlat, vFlat, fFlat)

	// Gate decay g = logsigmoid(f)/norm and slot-write weight s = 1 - exp(g).
	g := gateDecay(f, m.GateNorm)
	metal.Free(f)
	s := slotWrite(g)

	// Zero initial slot memory (see Forward TODO): Sk [B,H,HeadK,Slots],
	// Sv [B,H,Slots,HeadV].
	sk0 := metal.Zeros([]int32{B, m.NumHeads, m.HeadK, m.Slots}, q.Dtype())
	sv0 := metal.Zeros([]int32{B, m.NumHeads, m.Slots, m.HeadV}, q.Dtype())

	out, skN, svN := recurrence(q, k, v, g, s, sk0, sv0)
	metal.Free(q, k, v, g, s, sk0, sv0, skN, svN)

	// Output gate: swish(GProj) ⊙ o, then merge heads + output projection.
	gateFlat := m.GProj.Forward(x)
	gate := splitHeads(gateFlat, B, L, m.NumHeads, m.HeadV)
	metal.Free(gateFlat)
	swish := metal.SiLU(gate)
	metal.Free(gate)
	gated := metal.Mul(out, swish)
	metal.Free(out, swish)

	merged := mergeHeads(gated, B, L, m.NumHeads, m.HeadV)
	metal.Free(gated)
	result := m.OProj.Forward(merged)
	metal.Free(merged)
	return result, metal.SharedKV{}
}

// compileTimeProof keeps the build honest.
var _ metal.MixerCompute = (*Mixer)(nil)
