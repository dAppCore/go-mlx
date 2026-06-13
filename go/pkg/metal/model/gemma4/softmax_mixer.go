// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	metal "dappco.re/go/mlx/pkg/metal"
	scheme "dappco.re/go/mlx/pkg/scheme"
)

// softmax_mixer.go makes Gemma-4's attention satisfy the engine's pluggable
// sequence-mixer contract (metal.MixerCompute + the pure-Go scheme.Mixer it
// embeds), so the decoder loop dispatches it through the scheme registry exactly
// like any flash-linear-attention mixer would be. This is the reference
// implementation the FLA mixer tasks (#29–#31) mirror: declare a Kind + State,
// then a single Forward that maps the universal metal.MixerCtx onto the
// mixer's own compute.

// gemma4MixerExtra is the per-forward arch context Gemma-4 hands its mixer via
// metal.MixerCtx.Extra: the model config and the per-forward runtime mask cache
// (both gemma4-specific, so they ride the escape hatch rather than polluting the
// universal contract). The universal MixerCtx carries everything else.
type gemma4MixerExtra struct {
	cfg          *Gemma4TextConfig
	runtimeMasks *gemma4RuntimeMaskCache
}

// Kind is the config token a softmax-hybrid Gemma-4 build declares — sliding
// window local attention alternating with periodic full attention, plus
// shared-KV. The engine resolves this mixer with scheme.MixerFor("softmax-hybrid").
func (a *Gemma4Attention) Kind() string { return "softmax-hybrid" }

// State declares this mixer keeps a growing per-layer K/V cache, so the cache
// layer (quant / compaction / TurboQuant) operates on it and a recurrent cache
// scheme is rejected for it at load (the mixer-owns-state contract).
func (a *Gemma4Attention) State() scheme.StateKind { return scheme.StateKVCache }

// Forward mixes one chunk by delegating to the native attention forward, mapping
// the universal MixerCtx onto its parameters and reading the gemma4-specific
// config + runtime mask cache from ctx.Extra. Identical compute to the direct
// call — the decoder loop reroutes through here once #1's integration lands.
func (a *Gemma4Attention) Forward(x *metal.Array, ctx *metal.MixerCtx) (*metal.Array, metal.SharedKV) {
	extra, _ := ctx.Extra.(gemma4MixerExtra)
	return a.forward(x, ctx.Cache, ctx.B, ctx.L, ctx.Mask, ctx.Prev, extra.cfg, ctx.Window, ctx.FixedMask, extra.runtimeMasks, ctx.Materialize)
}

// compile-time proof Gemma4Attention is a full metal.MixerCompute.
var _ metal.MixerCompute = (*Gemma4Attention)(nil)
