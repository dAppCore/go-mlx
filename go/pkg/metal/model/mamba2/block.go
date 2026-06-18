// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package mamba2

import (
	metal "dappco.re/go/mlx/pkg/metal"
	flakernel "dappco.re/go/mlx/pkg/metal/model/internal/flakernel"
)

// block.go is the Mamba-2 block-assembly half of the mixer: the projection,
// causal-conv, dt-activation, group-expansion and gated-norm helpers Forward
// composes around the SSD scan kernel (scan.go / chunk.go). The causal-conv,
// softplus discretisation and gated RMSNorm are shared with the other gated
// families via flakernel; group-expansion stays Mamba-2-specific glue.

// causalConv applies the causal depthwise conv1d over the conv input
// xBC [B,L,convDim], threading the conv-state ring — the shared
// flakernel.CausalDepthwiseConv with this layer's conv weights/dims.
func (m *Mixer) causalConv(xBC, priorRing *metal.Array, B, L int32) (*metal.Array, *metal.Array) {
	cfg := m.W.cfg
	return flakernel.CausalDepthwiseConv(xBC, priorRing, m.W.ConvWeight, m.W.ConvBias, cfg.convDim(), cfg.ConvKernel, B, L)
}

// groupToHeads expands a grouped B/C projection [B,L,nGroups*N] to the per-head
// SSD layout [B,L,H,N] by repeating each group across the heads that share it
// (H/nGroups heads per group). nGroups==H is the per-head case (no repeat);
// nGroups==1 broadcasts one B/C to every head.
func (m *Mixer) groupToHeads(proj *metal.Array, B, L int32, cfg *Config) *metal.Array {
	N := cfg.StateDim
	G := cfg.NumGroups
	H := cfg.NumHeads
	grouped := metal.Reshape(proj, B, L, G, N) // [B,L,G,N]
	if G == H {
		return grouped
	}
	repeat := H / G
	// [B,L,G,1,N] → broadcast to [B,L,G,repeat,N] → [B,L,H,N].
	expanded := metal.Reshape(grouped, B, L, G, 1, N)
	metal.Free(grouped)
	bcast := metal.BroadcastTo(expanded, []int32{B, L, G, repeat, N})
	metal.Free(expanded)
	heads := metal.Reshape(bcast, B, L, H, N)
	metal.Free(bcast)
	return heads
}

// activateDt forms the per-head discretisation step dt [B,L,H] from the raw dt
// projection: add the optional dt-bias then softplus, the non-negative step the
// SSD discretisation needs.
func (m *Mixer) activateDt(dt *metal.Array, B, L int32, cfg *Config) *metal.Array {
	d := metal.Reshape(dt, B, L, cfg.NumHeads)
	if m.W.DtBias != nil && m.W.DtBias.Valid() {
		biasRow := metal.Reshape(m.W.DtBias, 1, 1, cfg.NumHeads)
		biased := metal.Add(d, biasRow)
		metal.Free(d, biasRow)
		d = biased
	}
	return flakernel.Softplus(d)
}

// gatedNorm applies the Mamba-2 gated RMSNorm RMSNorm(y)·SiLU(z) — the shared
// flakernel.GatedRMSNorm with this layer's norm weight.
func (m *Mixer) gatedNorm(y, z *metal.Array, cfg *Config) *metal.Array {
	return flakernel.GatedRMSNorm(y, z, m.W.Norm, cfg.RMSNormEps)
}
