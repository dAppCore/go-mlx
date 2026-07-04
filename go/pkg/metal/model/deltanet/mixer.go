// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package deltanet

import (
	metal "dappco.re/go/mlx/pkg/metal"
	flakernel "dappco.re/go/mlx/pkg/metal/model/internal/flakernel"
	scheme "dappco.re/go/mlx/pkg/scheme"
)

// mixer.go makes a DeltaNet layer satisfy the engine's pluggable sequence-mixer
// contract (metal.MixerCompute + the pure-Go scheme.Mixer it embeds), mirroring
// gemma4/softmax_mixer.go: declare a Kind + State, then a single Forward that
// maps the universal metal.MixerCtx onto the delta-rule kernel.

// Mixer is the compute-bearing DeltaNet scheme: the per-layer Weights plus the
// MixerCompute surface. The loader builds one per DeltaNet layer from the
// checkpoint and registers the family Kind once (RegisterMixer).
type Mixer struct {
	W *Weights
	// Beta is the resolved per-token write-strength provider. The β projection
	// (a linear + sigmoid) is arch-specific; the loader wires it and the kernel
	// consumes the resulting [B,H,L,1] tensor. Until the loader lands this is nil
	// and Forward surfaces a clear nil result.
	Beta BetaFn
}

// BetaFn produces the per-token write strength β [B,H,L,1] for a chunk of
// hidden states x [B,L,D]. It is the arch-specific half of DeltaNet (a linear
// projection + sigmoid); the kernel is β-agnostic.
type BetaFn func(x *metal.Array, b, l int32) *metal.Array

// Kind is the config token a DeltaNet build declares. The engine resolves this
// mixer with scheme.MixerFor("deltanet").
func (m *Mixer) Kind() string { return "deltanet" }

// State declares DeltaNet keeps a fixed-size recurrent state (the per-head value
// memory), so the engine pairs it with the recurrent cache scheme.
func (m *Mixer) State() scheme.StateKind { return scheme.StateRecurrent }

// Forward mixes one chunk of hidden states x [B,L,D]: project to per-head
// Q/K/V, compute the per-token write strength β, run the delta-rule kernel
// threading the per-layer recurrent state, then project the read-out to [B,L,D].
//
// State threading: the recurrent holder (#39) carries one slot — the value-
// memory matrix [B,H,Dk,Dv]. The prior slot is read from ctx.Recurrent() (nil
// before the first forward → the kernel starts from zero); the advanced state is
// handed to SetRecurrentState for the next chunk. The kernel already routes the
// single-token (L==1) decode case to its exact sequential recurrence, so no
// separate decode branch is needed here. Without a holder Forward degrades to
// the stateless prefill path, so a unit test can drive it with no cache.
//
//	out, _ := m.Forward(normed, &metal.MixerCtx{B: B, L: L, Cache: metal.NewRecurrentCache()})
func (m *Mixer) Forward(x *metal.Array, ctx *metal.MixerCtx) (*metal.Array, metal.SharedKV) {
	if m.Beta == nil {
		// β projection not wired (the loader supplies it). Surface a clean nil
		// rather than fabricate a write strength.
		return nil, metal.SharedKV{}
	}
	q, k, v := m.projectHeads(x, ctx.B, ctx.L)
	beta := m.Beta(x, ctx.B, ctx.L) // [B,H,L,1]

	prev := flakernel.PriorRecurrentSlot(ctx, 0) // slot 0, or nil on the first forward
	out, newState := DeltaRuleChunk(q, k, v, beta, prev, m.W.Scale, m.W.NormEps)
	metal.Free(q, k, v, beta)
	flakernel.StoreRecurrentState(ctx, newState) // hands ownership to the holder (or frees if none)

	if out == nil {
		return nil, metal.SharedKV{}
	}

	merged := mergeHeads(out, ctx.B, ctx.L, int32(m.W.NumHeads), int32(m.W.HeadDim))
	metal.Free(out)
	result := m.W.Output.Forward(merged)
	metal.Free(merged)
	return result, metal.SharedKV{}
}

// projectHeads applies the Q/K/V projections and splits the per-head layout
// [B,L,H*D] → [B,H,L,D] via a strided view, mirroring gemma4/attention.go.
func (m *Mixer) projectHeads(x *metal.Array, b, l int32) (*metal.Array, *metal.Array, *metal.Array) {
	h := int32(m.W.NumHeads)
	d := int32(m.W.HeadDim)
	split := func(proj *metal.Linear) *metal.Array {
		p := proj.Forward(x) // [B,L,H*D]
		view := metal.AsStrided(p, []int32{b, h, l, d},
			[]int64{int64(l * h * d), int64(d), int64(h * d), 1}, 0)
		metal.Free(p)
		return view
	}
	return split(m.W.QProj), split(m.W.KProj), split(m.W.VProj)
}

// mergeHeads folds the per-head output [B,H,L,Dv] back to [B,L,H*Dv] for the
// output projection — the inverse of projectHeads.
func mergeHeads(out *metal.Array, b, l, h, dv int32) *metal.Array {
	transposed := metal.Transpose4(out, 0, 2, 1, 3) // [B,L,H,Dv]
	merged := metal.Reshape(transposed, b, l, h*dv) // [B,L,H*Dv]
	metal.Free(transposed)
	return merged
}

// compile-time proof *Mixer is a full metal.MixerCompute.
var _ metal.MixerCompute = (*Mixer)(nil)

func init() {
	// Register the family identity + state contract so scheme.MixerFor("deltanet")
	// resolves. A loaded model overwrites this with a compute-bearing *Mixer
	// carrying its weights + β.
	scheme.RegisterMixer(&familyInfo{})
}

// familyInfo is the weightless catalogue entry for the DeltaNet family.
type familyInfo struct{}

func (familyInfo) Kind() string            { return "deltanet" }
func (familyInfo) State() scheme.StateKind { return scheme.StateRecurrent }
