// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package mamba2

import (
	core "dappco.re/go"
	metal "dappco.re/go/mlx/pkg/metal"
	flakernel "dappco.re/go/mlx/pkg/metal/model/internal/flakernel"
	scheme "dappco.re/go/mlx/pkg/scheme"
)

// mixer.go makes a Mamba-2 layer satisfy the engine's pluggable sequence-mixer
// contract (metal.MixerCompute + the pure-Go scheme.Mixer it embeds), mirroring
// gemma4/softmax_mixer.go and the gla.Mixer peer: a Weights set plus a single
// Forward that projects the hidden state, threads the recurrent state through
// the holder, runs the selective scan, and projects the read-out back.
//
// Forward owns the full Mamba-2 block: in-projection → gating split → causal
// depthwise conv (conv-state threaded for decode) → SSD selective scan (ssm-
// state threaded) → gated RMSNorm → out-projection. The scan math lives in
// scan.go (sequential, decode) and chunk.go (parallel, prefill); this file is
// the block assembly + the state-holder wiring the #39 contract froze.

// Weights is the per-layer projection set a Mamba-2 block needs.
//
// InProj fans the hidden state [B,L,D] out to [B,L, 2*dInner + 2*nGroups*N + H]:
// the gate z [dInner], the conv input xBC = [x_inner dInner | B nGroups*N |
// C nGroups*N], and the per-head step dt [H]. ConvWeight/ConvBias are the
// causal depthwise conv1d over xBC. A (via ALog), D, DtBias are the SSD scan
// parameters. Norm is the gated RMSNorm weight applied before OutProj, which
// projects [B,L,dInner] back to [B,L,D].
type Weights struct {
	InProj     *metal.Linear // [D, 2*dInner + 2*nGroups*N + H]
	ConvWeight *metal.Array  // [convDim, 1, convKernel] depthwise conv1d, convDim = dInner + 2*nGroups*N
	ConvBias   *metal.Array  // [convDim] or nil
	ALog       *metal.Array  // [H] log of the (positive) per-head decay magnitude; A = -exp(ALog)
	D          *metal.Array  // [H] per-head skip; nil = no skip
	DtBias     *metal.Array  // [H] added to dt before softplus; nil = none
	Norm       *metal.Array  // [dInner] gated-RMSNorm weight; nil = no gated norm
	OutProj    *metal.Linear // [dInner, D]
	cfg        *Config
}

// Config is the per-layer SSD geometry the mixer needs.
type Config struct {
	NumHeads   int32 // H
	HeadDim    int32 // P (dInner / H)
	StateDim   int32 // N (d_state)
	NumGroups  int32 // nGroups for B/C (1 ⇒ shared across heads)
	ConvKernel int32 // depthwise conv1d kernel width (e.g. 4)
	RMSNormEps float32
}

// dInner is H*P — the inner SSD width.
func (c *Config) dInner() int32 { return c.NumHeads * c.HeadDim }

// convDim is the width the causal conv1d operates over: x_inner + B + C.
func (c *Config) convDim() int32 { return c.dInner() + 2*c.NumGroups*c.StateDim }

// Mixer is the compute-bearing Mamba-2 scheme: the per-layer Weights plus the
// MixerCompute surface. The loader builds one per Mamba-2 layer from the
// checkpoint (see gemma4/mixer_ssm.go) and registers the family Kind once.
type Mixer struct {
	W *Weights
}

// NewMixer builds a Mamba-2 mixer from a loaded weight set + geometry.
func NewMixer(w *Weights, cfg *Config) *Mixer {
	if w != nil {
		w.cfg = cfg
	}
	return &Mixer{W: w}
}

// Kind is the config token a Mamba-2 build declares. The engine resolves this
// mixer with scheme.MixerFor("mamba2").
func (m *Mixer) Kind() string { return "mamba2" }

// State declares this mixer keeps a fixed-size recurrent SSM state (no growing
// K/V), so the engine pairs it with the recurrent cache scheme and rejects a
// KV-cache scheme for it at load (the mixer-owns-state contract).
func (m *Mixer) State() scheme.StateKind { return scheme.StateRecurrent }

// recurrentSlots is the layout of the two state slots Mamba-2 threads through
// the holder: slot 0 is the conv-state ring [B, convDim, convKernel-1], slot 1
// is the SSD state [B, H, P, N]. The mixer owns this; the holder treats the
// slots opaquely (see metal.RecurrentCache).
const (
	slotConvState = 0
	slotSSMState  = 1
	numSlots      = 2
)

// Forward runs one chunk of the Mamba-2 block over hidden states x [B,L,D],
// threading the recurrent state through ctx's holder:
//
//	in-proj → split (z | xBC | dt) → causal conv(xBC) → SiLU → split (x|B|C)
//	→ SSD scan(prior ssm-state) → gated RMSNorm(z) → out-proj
//
// It reads the prior [conv-state, ssm-state] slots from ctx.Recurrent(), runs
// the parallel scan for a prefill chunk (L>1) or the sequential single step for
// decode (L=1), and writes the advanced slots back. With no holder (a bare
// MixerCtx in a unit test) it runs the self-contained zero-state path.
func (m *Mixer) Forward(x *metal.Array, ctx *metal.MixerCtx) (*metal.Array, metal.SharedKV) {
	if m.W == nil || m.W.cfg == nil {
		return nil, metal.SharedKV{}
	}
	cfg := m.W.cfg
	B := ctx.B
	L := ctx.L

	rc, hasHolder := ctx.Recurrent()
	var priorConv, priorSSM *metal.Array
	if hasHolder {
		if slots := rc.RecurrentState(); len(slots) == numSlots {
			priorConv = slots[slotConvState]
			priorSSM = slots[slotSSMState]
		}
	}

	// in-proj → [B,L, 2*dInner + 2*nGroups*N + H]
	proj := m.W.InProj.Forward(x)
	dInner := cfg.dInner()
	convDim := cfg.convDim()

	// Split the projection along the last axis: z | xBC | dt.
	z := metal.SliceAxis(proj, 2, 0, dInner)                // [B,L,dInner] gate
	xBC := metal.SliceAxis(proj, 2, dInner, dInner+convDim) // [B,L,convDim] conv input
	dt := metal.SliceAxis(proj, 2, dInner+convDim, dInner+convDim+cfg.NumHeads)
	metal.Free(proj)

	// Causal depthwise conv over xBC, threading the conv-state ring for decode.
	convOut, newConv := m.causalConv(xBC, priorConv, B, L)
	metal.Free(xBC)
	activated := metal.SiLU(convOut)
	metal.Free(convOut)

	// Split conv output into x_inner | B | C.
	groupDim := cfg.NumGroups * cfg.StateDim
	xInner := metal.SliceAxis(activated, 2, 0, dInner)              // [B,L,dInner]
	bProj := metal.SliceAxis(activated, 2, dInner, dInner+groupDim) // [B,L,nGroups*N]
	cProj := metal.SliceAxis(activated, 2, dInner+groupDim, dInner+2*groupDim)
	metal.Free(activated)

	// Reshape to the per-head SSD layout the scan kernel consumes.
	xHeads := metal.Reshape(xInner, B, L, cfg.NumHeads, cfg.HeadDim) // [B,L,H,P]
	metal.Free(xInner)
	bHeads := m.groupToHeads(bProj, B, L, cfg) // [B,L,H,N]
	metal.Free(bProj)
	cHeads := m.groupToHeads(cProj, B, L, cfg) // [B,L,H,N]
	metal.Free(cProj)

	// dt → softplus(dt + dt_bias), per head [B,L,H].
	dtAct := m.activateDt(dt, B, L, cfg)
	metal.Free(dt)

	// A = -exp(ALog), per head [H].
	aNeg := flakernel.NegExp(m.W.ALog)

	in := ScanInput{X: xHeads, Dt: dtAct, A: aNeg, B: bHeads, C: cHeads, D: m.W.D}

	var y, newSSM *metal.Array
	if L == 1 {
		y, newSSM = SSDScan(in, priorSSM)
	} else {
		y, newSSM = SSDScanChunked(in, priorSSM, 0)
	}
	metal.Free(xHeads, dtAct, aNeg, bHeads, cHeads)

	if y == nil {
		metal.Free(z, newConv, newSSM)
		return nil, metal.SharedKV{}
	}

	// y [B,L,H,P] → [B,L,dInner]; gated RMSNorm with z; out-proj.
	yFlat := metal.Reshape(y, B, L, dInner)
	metal.Free(y)
	gated := m.gatedNorm(yFlat, z, cfg)
	metal.Free(yFlat, z)
	out := m.W.OutProj.Forward(gated)
	metal.Free(gated)

	// Write the advanced [conv-state, ssm-state] slots back to the holder.
	if hasHolder {
		if err := metal.Eval(newConv, newSSM); err != nil {
			core.Error("mamba2: state eval failed", "error", err)
		}
		rc.SetRecurrentState([]*metal.Array{newConv, newSSM})
	} else {
		metal.Free(newConv, newSSM)
	}
	return out, metal.SharedKV{}
}

// compile-time proof *Mixer is a full metal.MixerCompute.
var _ metal.MixerCompute = (*Mixer)(nil)
