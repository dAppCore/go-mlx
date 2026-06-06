// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

// EngineFeatures is a model-owned declaration of which engine kernels a model
// activates. It is the single source of truth for fast-path selection: a model
// declares what it needs, every load path applies the same declaration, and the
// GO_MLX_ENABLE_* env gates become diagnostic overrides on top rather than the
// only way to switch a path on.
//
// Today the fields are the accepted native-kernel set — each is numerically
// equivalent to its generic Go path but faster. Per-model algorithm axes (which
// KV cache, which attention) land as typed enum fields next, e.g.
//
//	Cache     CacheAlgo     // {Auto, Plain, Rotating, Fixed, Paged, Quantized}
//	Attention AttentionAlgo // {GQA, FixedOwner, WideSDPA, ...}
//
// Usage — short-lived probes may apply a declaration and restore it:
//
//	restore := metal.DefaultEngineFeatures().Apply()
//	defer restore()
type EngineFeatures struct {
	DirectGreedyToken       bool // native greedy token pick (skips host argmax)
	NativeMLPMatVec         bool // fused native MLP matvec
	NativeLinearMatVec      bool // fused native linear matvec
	NativeQ6BitstreamMatVec bool // native q6 bitstream matvec vs generic dense
	NativeAttentionOMatVec  bool // native attention output matvec
	GenerationStream        bool // streaming decode path
	AsyncDecodePrefetch     bool // async next-step weight prefetch during decode

	// Cache/attention algorithm axis — config-derived per model, NOT part of the
	// accepted always-on set. A sliding-window model declares the bounded
	// fixed-sliding KV cache from its config so the engine reacts to the model
	// (a 256K sliding model bounds its local layers instead of paging the full
	// context). Dense models leave these zero and page as before.
	FixedSlidingCache      bool // bounded fixed-size sliding-window KV cache vs unbounded paged
	FixedSlidingCacheBound bool // clamp the fixed-cache size to the per-layer cap
}

// Runtime-gate names — the env-string identity each feature carries across the
// metal runtime-gate boundary. Kept beside the struct so the field↔gate mapping
// is one obvious place; the loose name lists in mlx + cmd/mlx fold onto this in
// a later slice.
const (
	gateDirectGreedyToken       = "GO_MLX_ENABLE_DIRECT_GREEDY_TOKEN"
	gateNativeMLPMatVec         = "GO_MLX_ENABLE_NATIVE_MLP_MATVEC"
	gateNativeLinearMatVec      = "GO_MLX_ENABLE_NATIVE_LINEAR_MATVEC"
	gateNativeQ6BitstreamMatVec = "GO_MLX_ENABLE_NATIVE_Q6_BITSTREAM_MATVEC"
	gateNativeAttentionOMatVec  = "GO_MLX_ENABLE_NATIVE_GEMMA4_ATTENTION_O_MATVEC"
	gateGenerationStream        = "GO_MLX_ENABLE_GENERATION_STREAM"
	gateAsyncDecodePrefetch     = "GO_MLX_ENABLE_ASYNC_DECODE_PREFETCH"
	gateFixedSlidingCache       = "GO_MLX_ENABLE_FIXED_SLIDING_CACHE"
	gateFixedSlidingCacheBound  = "GO_MLX_ENABLE_FIXED_SLIDING_CACHE_BOUND"
)

// DefaultEngineFeatures is the accepted, numerically-validated fast-path set —
// the kernels proven safe to run by default. It is the typed replacement for
// the loose defaultGemma4FastRuntimeGates string list; model load applies the
// selected declaration so serve, benchmarks, and reloads inherit the same path.
func DefaultEngineFeatures() EngineFeatures {
	return EngineFeatures{
		DirectGreedyToken:       true,
		NativeMLPMatVec:         true,
		NativeLinearMatVec:      true,
		NativeQ6BitstreamMatVec: true,
		NativeAttentionOMatVec:  true,
		GenerationStream:        true,
		AsyncDecodePrefetch:     true,
	}
}

// GateValues returns the runtime-gate name→value map for the enabled features
// (value "1"). Disabled features are omitted, so the result mirrors exactly
// "what this model turns on" — a zero EngineFeatures yields an empty map.
func (f EngineFeatures) GateValues() map[string]string {
	out := map[string]string{}
	set := func(name string, on bool) {
		if on {
			out[name] = "1"
		}
	}
	set(gateDirectGreedyToken, f.DirectGreedyToken)
	set(gateNativeMLPMatVec, f.NativeMLPMatVec)
	set(gateNativeLinearMatVec, f.NativeLinearMatVec)
	set(gateNativeQ6BitstreamMatVec, f.NativeQ6BitstreamMatVec)
	set(gateNativeAttentionOMatVec, f.NativeAttentionOMatVec)
	set(gateGenerationStream, f.GenerationStream)
	set(gateAsyncDecodePrefetch, f.AsyncDecodePrefetch)
	set(gateFixedSlidingCache, f.FixedSlidingCache)
	set(gateFixedSlidingCacheBound, f.FixedSlidingCacheBound)
	return out
}

// GateNames returns the runtime-gate names of the enabled features in a stable
// field order (the declaration order of the struct). A fresh slice is returned
// each call, so callers may retain or mutate it freely. This is the ordered
// counterpart to GateValues — the form indexed accessors fold onto.
func (f EngineFeatures) GateNames() []string {
	names := make([]string, 0, 9)
	add := func(name string, on bool) {
		if on {
			names = append(names, name)
		}
	}
	add(gateDirectGreedyToken, f.DirectGreedyToken)
	add(gateNativeMLPMatVec, f.NativeMLPMatVec)
	add(gateNativeLinearMatVec, f.NativeLinearMatVec)
	add(gateNativeQ6BitstreamMatVec, f.NativeQ6BitstreamMatVec)
	add(gateNativeAttentionOMatVec, f.NativeAttentionOMatVec)
	add(gateGenerationStream, f.GenerationStream)
	add(gateAsyncDecodePrefetch, f.AsyncDecodePrefetch)
	add(gateFixedSlidingCache, f.FixedSlidingCache)
	add(gateFixedSlidingCacheBound, f.FixedSlidingCacheBound)
	return names
}

// Apply turns on the declared features via the runtime-gate machinery and
// returns a restore func that reverts every gate it set. This is the bridge
// that lets a model's declaration drive the existing gate-consuming code paths
// unchanged while call sites migrate from string gates to typed fields.
func (f EngineFeatures) Apply() func() {
	values := f.GateValues()
	restores := make([]func(), 0, len(values))
	for name, value := range values {
		restores = append(restores, SetRuntimeGate(name, value))
	}
	return func() {
		for i := len(restores) - 1; i >= 0; i-- {
			restores[i]()
		}
	}
}

// EngineFeaturesModel optionally declares which engine fast-path kernels a model
// activates. The loader (backend.LoadAndInit) applies the declaration so every
// run path inherits the model's chosen path; a model that does not implement it
// falls back to DefaultEngineFeatures. Dispatching on this capability — rather
// than a type-switch — keeps model packages outside metal (the go-mlx #45 SDK
// boundary pattern, alongside GreedyTokenModel / QueryHeadCounter).
type EngineFeaturesModel interface {
	EngineFeatures() EngineFeatures
}

// EngineFeaturesFor returns the engine features a loaded model declares, or the
// accepted default set when the model does not declare its own. The loader
// applies the result so a model runs exactly the kernels it asks for.
func EngineFeaturesFor(model any) EngineFeatures {
	if m, ok := model.(EngineFeaturesModel); ok {
		return m.EngineFeatures()
	}
	return DefaultEngineFeatures()
}
