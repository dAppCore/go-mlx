// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

// Sequence-mixer factory registry — mixer kind → builder, mirroring
// model_registry.go. A mixer family (softmax attention, GLA, RetNet, DeltaNet,
// Mamba2, RWKV7, GSA, …) self-registers from its own package init(); the load
// path looks it up by the kind the model config declares for a layer, with no
// central switch and no edit to any model package. A mixer's math and state
// contract live with it — only the build-from-weights wiring registers here.
type MixerLoader func(ctx MixerBuildCtx) (MixerCompute, error)

// MixerBuildCtx is the neutral input a mixer builder gets for one layer: a
// quant-ready Linear resolver (the quant factory is already applied, so the
// mixer never sees a quant format), a raw-tensor accessor for the non-Linear
// weights (conv1d, A_log, D, dt_bias, norms), the neutral transformer dims, and
// the layer index. It carries no model-specific config type — a mixer is built
// the same way whatever model declares it.
type MixerBuildCtx struct {
	Linear   func(proj string) *Linear // quant-ready Linear by projection name
	Weight   func(name string) *Array  // raw tensor by name (non-Linear weights)
	Cfg      TransformerConfig
	LayerIdx int32
}

var mixerLoaders = map[string]MixerLoader{}

// RegisterMixerLoader registers fn for a mixer kind; a later call for the same
// kind overrides. Call from init() so a family self-registers and the resolver
// needs no central switch. No-op on empty kind or nil fn.
func RegisterMixerLoader(kind string, fn MixerLoader) {
	if kind == "" || fn == nil {
		return
	}
	mixerLoaders[kind] = fn
}

// lookupMixerLoader returns the loader for kind, or nil if none is registered.
func lookupMixerLoader(kind string) MixerLoader { return mixerLoaders[kind] }

// MixerLoaderFor resolves a registered mixer loader by kind for a consumer
// outside this package — the generic config-composed model resolves each layer's
// mixer through it. Reports (nil,false) for an unregistered kind so the consumer
// refuses the config cleanly.
func MixerLoaderFor(kind string) (MixerLoader, bool) {
	fn, ok := mixerLoaders[kind]
	return fn, ok
}
