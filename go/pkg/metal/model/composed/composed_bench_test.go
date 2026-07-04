// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package composed

import (
	"math"
	"testing"

	"dappco.re/go/mlx/internal/metaltest"
	metal "dappco.re/go/mlx/pkg/metal"
)

// composed_bench_test.go measures the composed model's own forward orchestration
// — ForwardMasked driving the per-layer composedLayer.forward block (norm →
// mixer → add, norm → MLP → add) — at a representative synthetic geometry. No
// model load (AX-11): the model is assembled from metal.FromValues weights via
// buildComposed, the same synthetic fixture the unit tests use, never a
// checkpoint off disk.
//
// This closes a measurement gap that is the composed package's own, not a
// mixer's. Unlike qwen3/llama (whose ForwardMasked delegates to the SHARED
// metal.DenseDecoderLayer, so the per-op work lives in pkg/metal and was benched
// there), composed owns its block topology: composedLayer.forward composes the
// SDK's neutral pieces directly and builds a per-layer metal.MixerCtx to drive
// the mixer through the MixerCompute interface. The interface dispatch is the
// point — the compiler cannot prove the callee does not retain the ctx, so the
// &metal.MixerCtx{...} literal escapes to heap once PER LAYER PER FORWARD. That
// per-op, per-layer allocation is invisible at the single-mixer level (gla/mla
// bench their mixer in isolation) and only the multi-layer model forward shows
// it — which is why these benchmarks drive the whole ComposedModel.Forward, not
// an isolated layer. A wide (8-layer) variant makes the per-layer term stark.
//
// Two shapes matter, and they exercise different code:
//
//   - The prefill path (L>1, fresh caches) runs every layer's input-norm, the
//     softmax mixer's q/k/v projections + causal [L,L] SDPA, the residual adds,
//     and the SwiGLU FFN — the first turn of every sequence.
//   - The single-token decode path (L==1, warm caches) is the steady-state
//     generation kernel and the highest-multiplier path: run once per emitted
//     token. The warm-cache prefill and its Eval are excluded from the timer
//     (StopTimer/StartTimer, the mla/qwen3-bench pattern), so each iteration
//     times exactly one decode step appending to populated KV caches.
//
// Both forms are force-materialised inside the timed region: ForwardMasked
// returns a lazy graph (the engine evals once per token after the trunk
// composes), so a lazy return would measure graph CONSTRUCTION, not compute (the
// MLA-bench lesson, inherited via qwen3/gla). The absolute µs therefore carries
// one eval sync per Forward that the real decode loop amortises — read the
// alloc counts and the cross-bench deltas, not the absolute number.
//
// Run: go test -tags metal_runtime -bench='Composed' -benchmem -run='^$' \
//   -benchtime=50x ./go/pkg/metal/model/composed/
// (needs MLX_METALLIB_PATH set to dist/lib/mlx.metallib)

const (
	benchPrefillLen = 16 // prefill chunk length (L>1)
	benchWideLayers = 8  // wide trunk — the per-layer MixerCtx term is N× here
)

func composedBenchGate(b *testing.B) {
	if !metaltest.RunMetalTests || !metal.MetalAvailable() {
		b.Skip("build with -tags metal_runtime to enable Metal runtime benchmarks")
	}
}

// composedBenchModel assembles an n-layer full-attention composed model from the
// shared synthetic fixture (ramp-filled weights, tied embedding). Values are
// irrelevant to allocation counts and op timing — only the shape and a valid
// graph matter.
func composedBenchModel(b *testing.B, n int32) *ComposedModel {
	b.Helper()
	cfg := &metal.DenseConfig{
		TransformerConfig: metal.TransformerConfig{
			ModelType:             "composed",
			HiddenSize:            tHidden,
			NumHiddenLayers:       n,
			IntermediateSize:      tInter,
			NumAttentionHeads:     tHeads,
			NumKeyValueHeads:      tKVHeads,
			HeadDim:               tHeadDim,
			VocabSize:             tVocab,
			RMSNormEps:            1e-5,
			MaxPositionEmbeddings: 256,
		},
		RopeTheta: 10000,
		Scale:     float32(1.0 / math.Sqrt(float64(tHeadDim))),
	}
	cfg.ModelType = "full_attention" // uniform: every layer is softmax attention
	m, err := buildComposed(cfg, fullAttentionWeights(n), nil)
	if err != nil {
		b.Fatalf("buildComposed(%d layers): %v", n, err)
	}
	return m
}

// composedBenchTokens builds an [1,L] int32 token sequence; ids stay in vocab.
func composedBenchTokens(l int32) *metal.Array {
	ids := make([]int32, l)
	for i := range ids {
		ids[i] = int32(i % tVocab)
	}
	return metal.FromValues(ids, 1, int(l))
}

// benchPrefill runs the prefill forward for an n-layer model: fresh caches, an
// L-token chunk, force-materialised in the timed region.
func benchPrefill(b *testing.B, n int32) {
	composedBenchGate(b)
	m := composedBenchModel(b, n)
	defer m.CloseModel()
	tokens := composedBenchTokens(benchPrefillLen)
	defer metal.Free(tokens)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		caches := m.NewCache()
		out := m.Forward(tokens, caches)
		metal.Materialize(out) // ForwardMasked returns a lazy graph; force the compute
		metal.Free(out)
		metal.FreeCaches(caches)
	}
}

// benchDecode runs one single-token decode step for an n-layer model over warm
// caches. The warm-cache prefill and its materialise are excluded from the timer
// (StopTimer/StartTimer), so each iteration times exactly one L=1 step appending
// to populated KV caches.
func benchDecode(b *testing.B, n int32) {
	composedBenchGate(b)
	m := composedBenchModel(b, n)
	defer m.CloseModel()
	warm := composedBenchTokens(benchPrefillLen)
	defer metal.Free(warm)
	step := composedBenchTokens(1)
	defer metal.Free(step)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		caches := m.NewCache()
		pre := m.Forward(warm, caches)
		metal.Materialize(pre) // materialise the warm caches before timing the step
		metal.Free(pre)
		b.StartTimer()

		out := m.Forward(step, caches)
		metal.Materialize(out) // force the decode step's actual compute
		metal.Free(out)

		b.StopTimer()
		metal.FreeCaches(caches)
	}
}

// BenchmarkComposed_Forward_Prefill is the first-turn prefill over the 2-layer
// fixture geometry — the minimal multi-layer trunk that already shows the
// per-layer MixerCtx term (2 layers → 2 ctx allocs before the hoist, 1 after).
func BenchmarkComposed_Forward_Prefill(b *testing.B) { benchPrefill(b, tLayers) }

// BenchmarkComposed_Forward_Decode is the steady-state single-token decode over
// the 2-layer fixture — the per-token generation kernel.
func BenchmarkComposed_Forward_Decode(b *testing.B) { benchDecode(b, tLayers) }

// BenchmarkComposed_Forward_Prefill_Wide drives an 8-layer trunk so the
// per-layer MixerCtx allocation term is 8× — the shape where the hoist-and-reuse
// win is unambiguous (8 ctx allocs → 1).
func BenchmarkComposed_Forward_Prefill_Wide(b *testing.B) { benchPrefill(b, benchWideLayers) }

// BenchmarkComposed_Forward_Decode_Wide is the 8-layer single-token decode — the
// highest per-forward layer count at the steady-state kernel.
func BenchmarkComposed_Forward_Decode_Wide(b *testing.B) { benchDecode(b, benchWideLayers) }
