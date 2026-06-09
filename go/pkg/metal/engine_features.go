// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

// EngineFeatures is a model-owned declaration of which engine kernels a model
// activates. It is the single source of truth for fast-path selection: a model
// declares what it needs and every load path applies the same declaration via
// the typed runtime Gate enum (SetRuntimeGate) — never an env var.
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
	// NativeFixedSlidingAttention fuses attention + the drop/append sliding-window
	// cache update into ONE kernel for past-cap local layers. Without it serve
	// falls back to the Go-graph RotatingKVCache rotation (Slice4+Concatenate2 =
	// a fresh O(window) buffer copy per token). AX-11: fused 50µs/token vs the
	// rotation's 77-154µs cache-copy ALONE (BenchmarkDecodeLoop_NativeFixedSliding
	// Attention_PastCap_Batched32 vs RotatingKVCache_Append_SingleToken_PastCap).
	// Correctness: TestDecode_nativeFixedSlidingSingleTokenAttention_Good (value-
	// verified). attention.go falls back to the Go graph on any error, so this is
	// safe-by-default.
	NativeFixedSlidingAttention bool
	GenerationStream            bool // streaming decode path
	AsyncDecodePrefetch     bool // async next-step weight prefetch during decode

	// Cache/attention algorithm axis — config-derived per model, NOT part of the
	// accepted always-on set. A sliding-window model declares the bounded
	// fixed-sliding KV cache from its config so the engine reacts to the model
	// (a 256K sliding model bounds its local layers instead of paging the full
	// context). Dense models leave these zero and page as before.
	FixedSlidingCache      bool // bounded fixed-size sliding-window KV cache vs unbounded paged
	FixedSlidingCacheBound bool // clamp the fixed-cache size to the per-layer cap
}

// DefaultEngineFeatures is the accepted, numerically-validated fast-path set —
// the kernels proven safe to run by default. It is the typed replacement for
// the loose defaultGemma4FastRuntimeGates string list; model load applies the
// selected declaration so serve, benchmarks, and reloads inherit the same path.
func DefaultEngineFeatures() EngineFeatures {
	return EngineFeatures{
		DirectGreedyToken:       true,
		NativeMLPMatVec:         true,
		// NativeLinearMatVec stays on, but Linear.Forward (nn.go) now routes
		// q4 to MLX's quantized_matmul instead of the custom matvec kernel:
		// AX-11 benchmarks prove gemm ~35% faster for single-token q4 decode
		// (FFN pair 49.7us matvec vs 32.2us gemm). The native kernel is retained
		// for q6 (bespoke bitstream packing the gemm path cannot read) and q8.
		NativeLinearMatVec:      true,
		NativeQ6BitstreamMatVec: true,
		NativeAttentionOMatVec:  true,
		// Proven faster than the Go-graph rotation fallback for past-cap local
		// layers (50µs vs 77-154µs/token) and value-correct; safe fallback in
		// gemma4 attention.go. Was a never-enabled legacy gate before this.
		NativeFixedSlidingAttention: true,
		GenerationStream:            true,
		AsyncDecodePrefetch:         true,
	}
}

// EnabledGates returns the runtime Gates this declaration turns on, in stable
// struct-field order. Disabled features are omitted, so a zero EngineFeatures
// yields an empty slice — the result mirrors exactly "what this model turns on".
// A fresh slice is returned each call; Apply folds onto it. This is the typed
// replacement for the env-shaped GateValues/GateNames string forms.
func (f EngineFeatures) EnabledGates() []Gate {
	gates := make([]Gate, 0, 10)
	add := func(gate Gate, on bool) {
		if on {
			gates = append(gates, gate)
		}
	}
	add(GateDirectGreedyToken, f.DirectGreedyToken)
	add(GateNativeMLPMatVec, f.NativeMLPMatVec)
	add(GateNativeLinearMatVec, f.NativeLinearMatVec)
	add(GateNativeQ6BitstreamMatVec, f.NativeQ6BitstreamMatVec)
	add(GateNativeAttentionOMatVec, f.NativeAttentionOMatVec)
	add(GateNativeFixedSlidingAttention, f.NativeFixedSlidingAttention)
	add(GateGenerationStream, f.GenerationStream)
	add(GateAsyncDecodePrefetch, f.AsyncDecodePrefetch)
	add(GateFixedSlidingCache, f.FixedSlidingCache)
	add(GateFixedSlidingCacheBound, f.FixedSlidingCacheBound)
	return gates
}

// Apply turns on the declared features via the typed runtime Gate enum and
// returns a restore func that reverts every gate it set, in reverse order. The
// loader (backend.LoadAndInit) applies a model's declaration so serve,
// benchmarks, and reloads all inherit the same path; tests use the restore for
// scoped overrides. Only enabled features are touched (additive), matching the
// declared "what this model turns on" set.
func (f EngineFeatures) Apply() func() {
	gates := f.EnabledGates()
	restores := make([]func(), 0, len(gates))
	for _, gate := range gates {
		restores = append(restores, SetRuntimeGate(gate, true))
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
