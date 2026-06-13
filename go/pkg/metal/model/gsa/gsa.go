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

// Forward mixes one chunk through the gated slot recurrence, threading the slot
// memory (Sk, Sv) through the recurrent holder (ctx.Recurrent()): it reads the
// prior state, runs the recurrence, and writes the advanced state back so a
// subsequent decode chunk continues from it. The first forward (nil prior)
// starts from the zero state, which is the correct prefill of a fresh sequence.
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

	// Read the prior slot memory from the recurrent holder; the load-time
	// Compatible() gate guarantees a StateRecurrent mixer is only ever paired
	// with the holder, so a missing one is a fatal misconfiguration. priorSk /
	// priorSv are nil before the first forward (or after a session reset) — the
	// recurrence then starts from its zero state. Sk [B,H,HeadK,Slots], Sv
	// [B,H,Slots,HeadV].
	rc, ok := ctx.Recurrent()
	if !ok {
		panic("gsa: StateRecurrent mixer requires a recurrent cache holder")
	}
	sk0, sv0, owned := m.priorState(rc, B, q.Dtype())

	out, skN, svN := recurrence(q, k, v, g, s, sk0, sv0)
	metal.Free(q, k, v, g, s)
	if owned {
		// We allocated the zero seed (first forward); free it. When the holder
		// supplied the prior state we leave its handles alone — SetRecurrentState
		// below frees whatever skN/svN supersede.
		metal.Free(sk0, sv0)
	}
	// Hand the advanced state to the holder for the next chunk. The holder takes
	// ownership of skN/svN; we must not free or reuse them after this.
	rc.SetRecurrentState([]*metal.Array{skN, svN})

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

// priorState resolves the recurrence's starting slot memory from the holder.
// GSA stores two slots — Sk [B,H,HeadK,Slots] and Sv [B,H,Slots,HeadV]. When the
// holder carries a valid pair (a continuing decode) they are returned as-is and
// owned reports false (the holder owns them, SetRecurrentState frees what they
// supersede). When the holder is empty (first forward / post-reset) a zero seed
// is allocated and owned reports true so Forward frees it after the recurrence.
func (m *Mixer) priorState(rc metal.RecurrentCache, B int32, dtype metal.DType) (sk0, sv0 *metal.Array, owned bool) {
	if prior := rc.RecurrentState(); len(prior) == 2 && prior[0].Valid() && prior[1].Valid() {
		return prior[0], prior[1], false
	}
	sk0 = metal.Zeros([]int32{B, m.NumHeads, m.HeadK, m.Slots}, dtype)
	sv0 = metal.Zeros([]int32{B, m.NumHeads, m.Slots, m.HeadV}, dtype)
	return sk0, sv0, true
}

// compileTimeProof keeps the build honest.
var _ metal.MixerCompute = (*Mixer)(nil)
