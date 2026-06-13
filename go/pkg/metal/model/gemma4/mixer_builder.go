// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/metal"
)

// MixerKindFor maps a layer's declared type to the sequence-mixer kind the
// engine resolves for it. Gemma-4's attention layer types (full / sliding) are
// the softmax-hybrid mixer; any other token is taken as a recurrent mixer kind
// (mamba2, gla, …) a family registers a builder for — so a hybrid pack declaring
// layer_types: ["full_attention","mamba2",…] resolves each layer's mixer with no
// loader branch per family.
func (c *Gemma4TextConfig) MixerKindFor(layerIdx int) string {
	if c == nil || layerIdx < 0 || layerIdx >= len(c.LayerTypes) {
		return "softmax-hybrid"
	}
	switch c.LayerTypes[layerIdx] {
	case "full_attention", "sliding_attention", "":
		return "softmax-hybrid"
	default:
		return c.LayerTypes[layerIdx]
	}
}

// MixerBuildCtx is what a recurrent-mixer builder needs to construct its mixer
// from a Gemma-4 checkpoint: the layer's weight map + naming prefix, the config,
// and the layer index. Softmax attention is built inline by the loader (the
// default); a non-softmax family registers a builder and the loader resolves it
// by the layer's MixerKind.
type MixerBuildCtx struct {
	Weights  map[string]*metal.Array // the checkpoint weights
	Prefix   string                  // e.g. "model.layers.3"
	Cfg      *Gemma4TextConfig
	LayerIdx int32
}

// MixerBuilder constructs a recurrent sequence mixer for one layer from its
// checkpoint weights, returning the MixerCompute the decoder dispatches. The
// builder lives in gemma4 (it knows the weight-naming convention) and imports
// the mixer's package for the math; the mixer package never imports gemma4, so
// the mixer math stays relocatable to go-inference for the other engines.
type MixerBuilder func(ctx MixerBuildCtx) (metal.MixerCompute, error)

// gemma4MixerBuilders is the per-kind builder registry — the same named,
// insertion-ordered, thread-safe collection the scheme registries use. A
// recurrent family registers its builder in its own gemma4 wiring file; the
// loader resolves it by kind, with no branch here.
var gemma4MixerBuilders = core.NewRegistry[MixerBuilder]()

// RegisterMixerBuilder registers (or overwrites) the builder for a mixer kind.
//
//	func init() { gemma4.RegisterMixerBuilder("mamba2", buildMamba2Layer) }
func RegisterMixerBuilder(kind string, build MixerBuilder) core.Result {
	return gemma4MixerBuilders.Set(kind, build)
}

// mixerBuilderFor resolves a registered builder by kind.
func mixerBuilderFor(kind string) (MixerBuilder, bool) {
	if r := gemma4MixerBuilders.Get(kind); r.OK {
		return r.Value.(MixerBuilder), true
	}
	return nil, false
}
