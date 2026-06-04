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
// Usage — a load path applies the model's declaration and reverts on teardown:
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
)

// DefaultEngineFeatures is the accepted, numerically-validated fast-path set —
// the kernels proven safe to run by default. It is the typed replacement for
// the loose defaultGemma4FastRuntimeGates string list; serve and the benchmark
// commands apply this so they exercise the same path instead of diverging.
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
	return out
}

// Apply turns on the declared features via the runtime-gate machinery and
// returns a restore func that reverts every gate it set. This is the bridge
// that lets a model's declaration drive the existing gate-consuming code paths
// unchanged; later slices read EngineFeatures directly at each site and retire
// the gate.
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
