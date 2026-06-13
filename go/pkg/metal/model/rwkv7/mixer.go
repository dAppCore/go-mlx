// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package rwkv7

import (
	core "dappco.re/go"
	metal "dappco.re/go/mlx/pkg/metal"
	scheme "dappco.re/go/mlx/pkg/scheme"
)

// mixer.go makes an RWKV-7 layer satisfy the engine's pluggable sequence-mixer
// contract (metal.MixerCompute + the pure-Go scheme.Mixer it embeds), mirroring
// gemma4/softmax_mixer.go and the mamba2.Mixer peer: a Weights set plus a single
// Forward that projects the hidden state, threads the recurrent state through
// the holder, runs the WKV7 recurrence, and projects the read-out back.
//
// RWKV-7 ("Goose") keeps a single [K,V] state matrix per head (one holder slot,
// unlike Mamba-2's two), so the block is simpler: no causal conv, just the
// r/w/k/v/a/b projections feeding the generalised-delta-rule recurrence in
// recurrence.go (sequential, decode) / chunk.go (parallel, prefill).

// Weights is the per-layer projection set an RWKV-7 block needs. RProj/KProj
// project the hidden state to per-head receptance/key [H*K]; VProj to value
// [H*V]; WProj to the raw decay [H*K] (turned into the log-decay w = -exp(·) ≤ 0
// the recurrence consumes); AProj/BProj to the two in-context learning-rate
// vectors [H*K] (the removal/write-back directions of the delta rule). OutProj
// projects the per-head read-out [H*V] back to the model dimension.
//
// The a/b coupling a real RWKV-7 checkpoint applies (b derived from a normalised
// key and the a-gate) is the loader's to wire when it builds AProj/BProj from
// the checkpoint tensors; the kernel takes the resolved a and b directly so the
// recurrence stays the faithful generalised delta rule.
type Weights struct {
	RProj   *metal.Linear // [D, H*K] receptance (the read-out query)
	WProj   *metal.Linear // [D, H*K] raw decay; log-decay w = -exp(WProj)
	KProj   *metal.Linear // [D, H*K] key
	VProj   *metal.Linear // [D, H*V] value
	AProj   *metal.Linear // [D, H*K] in-context learning-rate (removal direction)
	BProj   *metal.Linear // [D, H*K] in-context learning-rate (write-back direction)
	OutProj *metal.Linear // [H*V, D]
	cfg     *Config
}

// Config is the per-layer WKV7 geometry the mixer needs.
type Config struct {
	NumHeads int32 // H
	KeyDim   int32 // K — per-head key/receptance/decay dimension
	ValueDim int32 // V — per-head value dimension
}

// Mixer is the compute-bearing RWKV-7 scheme: the per-layer Weights plus the
// MixerCompute surface. The loader builds one per RWKV-7 layer from the
// checkpoint (see gemma4/mixer_ssm.go) and registers the family Kind once.
type Mixer struct {
	W *Weights
}

// NewMixer builds an RWKV-7 mixer from a loaded weight set + geometry.
func NewMixer(w *Weights, cfg *Config) *Mixer {
	if w != nil {
		w.cfg = cfg
	}
	return &Mixer{W: w}
}

// Kind is the config token an RWKV-7 build declares. The engine resolves this
// mixer with scheme.MixerFor("rwkv7").
func (m *Mixer) Kind() string { return "rwkv7" }

// State declares this mixer keeps a fixed-size recurrent [K,V] state (no growing
// K/V cache), so the engine pairs it with the recurrent cache scheme and rejects
// a KV-cache scheme for it at load (the mixer-owns-state contract).
func (m *Mixer) State() scheme.StateKind { return scheme.StateRecurrent }

// slotWKVState is RWKV-7's single holder slot: the [B,H,K,V] state matrix.
const slotWKVState = 0

// Forward runs one chunk of the RWKV-7 block over hidden states x [B,L,D],
// threading the single [B,H,K,V] state slot through ctx's holder:
//
//	project r/k/v/a/b + log-decay w → WKV7 recurrence(prior state) → out-proj
//
// It reads the prior state from ctx.Recurrent(), runs the parallel scan for a
// prefill chunk (L>1) or the sequential single step for decode (L=1), and writes
// the advanced state back. With no holder (a bare MixerCtx in a unit test) it
// runs the self-contained zero-state path.
func (m *Mixer) Forward(x *metal.Array, ctx *metal.MixerCtx) (*metal.Array, metal.SharedKV) {
	if m.W == nil || m.W.cfg == nil {
		return nil, metal.SharedKV{}
	}
	cfg := m.W.cfg
	B := ctx.B
	L := ctx.L

	rc, hasHolder := ctx.Recurrent()
	var prior *metal.Array
	if hasHolder {
		if slots := rc.RecurrentState(); len(slots) == 1 {
			prior = slots[slotWKVState]
		}
	}

	in := StepInput{
		R: m.projectK(m.W.RProj, x, B, L),
		W: m.logDecay(x, B, L),
		K: m.projectK(m.W.KProj, x, B, L),
		V: m.projectV(m.W.VProj, x, B, L),
		A: m.projectK(m.W.AProj, x, B, L),
		B: m.projectK(m.W.BProj, x, B, L),
	}

	var o, next *metal.Array
	if L == 1 {
		o, next = WKV7(in, prior)
	} else {
		o, next = WKV7Chunked(in, prior)
	}
	metal.Free(in.R, in.W, in.K, in.V, in.A, in.B)

	if o == nil {
		metal.Free(next)
		return nil, metal.SharedKV{}
	}

	// o [B,L,H,V] → [B,L,H*V]; out-proj back to [B,L,D].
	merged := metal.Reshape(o, B, L, cfg.NumHeads*cfg.ValueDim)
	metal.Free(o)
	out := m.W.OutProj.Forward(merged)
	metal.Free(merged)

	if hasHolder {
		if err := metal.Eval(next); err != nil {
			core.Error("rwkv7: state eval failed", "error", err)
		}
		rc.SetRecurrentState([]*metal.Array{next})
	} else {
		metal.Free(next)
	}
	return out, metal.SharedKV{}
}

// projectK applies a [D, H*K] projection and reshapes to the per-head [B,L,H,K]
// layout the recurrence consumes (r/w/k/a/b all share the K dimension).
func (m *Mixer) projectK(proj *metal.Linear, x *metal.Array, B, L int32) *metal.Array {
	p := proj.Forward(x) // [B,L,H*K]
	view := metal.Reshape(p, B, L, m.W.cfg.NumHeads, m.W.cfg.KeyDim)
	metal.Free(p)
	return view
}

// projectV applies the [D, H*V] value projection and reshapes to [B,L,H,V].
func (m *Mixer) projectV(proj *metal.Linear, x *metal.Array, B, L int32) *metal.Array {
	p := proj.Forward(x) // [B,L,H*V]
	view := metal.Reshape(p, B, L, m.W.cfg.NumHeads, m.W.cfg.ValueDim)
	metal.Free(p)
	return view
}

// logDecay forms the per-channel log-decay w [B,L,H,K] the recurrence scales the
// state by: RWKV-7 parametrises the decay as w = -exp(WProj), keeping it strictly
// ≤ 0 so exp(w) ∈ (0,1] is a stable forget gate.
func (m *Mixer) logDecay(x *metal.Array, B, L int32) *metal.Array {
	raw := m.W.WProj.Forward(x) // [B,L,H*K]
	e := metal.Exp(raw)
	metal.Free(raw)
	neg := metal.Negative(e)
	metal.Free(e)
	view := metal.Reshape(neg, B, L, m.W.cfg.NumHeads, m.W.cfg.KeyDim)
	metal.Free(neg)
	return view
}

// compile-time proof *Mixer is a full metal.MixerCompute.
var _ metal.MixerCompute = (*Mixer)(nil)
