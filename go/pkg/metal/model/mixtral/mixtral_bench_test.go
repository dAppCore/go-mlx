// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package mixtral

import (
	"testing"

	"dappco.re/go/mlx/internal/metaltest"
	metal "dappco.re/go/mlx/pkg/metal"
)

// mixtral_bench_test.go measures the Mixtral sparse-MoE decoder layer at a
// representative synthetic geometry — no model load (AX-11), the
// gla/retnet/gemma4 bench style. It closes the long-standing measurement gap:
// mixtral had no benchmarks at all, so the Go-side allocation behaviour of its
// own orchestration code (mixtralDecoderLayerForward → norm → GQA attention →
// MoE router top-k → batched SwiGLU experts → residual adds) was invisible.
//
// Unlike gla/retnet — which export a self-contained recurrent kernel
// (GatedChunk / RetentionChunk) that takes synthetic q/k/v and needs no model
// — Mixtral has no standalone kernel: its hot path is a decoder layer that
// composes shared pkg/metal modules (GQAAttention, MoERouter top-k,
// MoESwiGLUExperts). So this bench follows the gemma4 MoE precedent
// (experts_decode_bench_test.go): hand-build a synthetic MixtralDecoderLayer
// with valid-shaped random weights and bench the REAL layer-forward, rather
// than duplicate the pure dispatch kernel that pkg/metal already benches
// (moe_bench_test.go, router_topk_decode_bench_test.go).
//
// Three shapes matter, exercising different code:
//
//   - The MoE prefill path (L=256): mixtralDecoderLayerForward on a sparse
//     layer runs RMSNorm → GQA attention over the whole window → MoESwiGLUForward
//     (router projection + top-k select + batched gate/up/down expert SwiGLU +
//     weighted sum) → two residual adds. This is the bulk per-MoE-layer prefill
//     cost and the path the router/expert-dispatch alloc question lives on.
//   - The MoE decode path (L=1): the steady-state per-token generation kernel —
//     the same layer-forward with a single query position and a primed KV cache,
//     so the router runs over one token and the experts gather for one row. This
//     is the analogue of the rwkv7/mamba2/retnet sequential decode benches but
//     for an attention+MoE layer; any per-token Go-side validation/clone alloc
//     in the dispatch surfaces here.
//   - The dense decode path (L=1): the non-MoE Mixtral layer (SiLU MLP instead
//     of experts) — the contrast that isolates how much of the per-token Go-side
//     cost is the MoE dispatch vs the shared attention+norm spine. Mixtral packs
//     are mostly MoE, but the decoder_sparse_step config can interleave dense
//     layers, so this path is live and worth a baseline.
//
// All forms are force-Eval'd inside the timed region: a lazy return measures
// graph build, not compute (the MLA-bench lesson, inherited via the family).
//
// Run: go test -tags metal_runtime -bench='Mixtral' -benchmem -run='^$' \
//   -benchtime=200ms ./go/pkg/metal/model/mixtral/
// (needs MLX_METALLIB_PATH set to dist/lib/mlx.metallib)

const (
	mixtralBenchHidden  = 256
	mixtralBenchHeads   = 4
	mixtralBenchKVHeads = 4
	mixtralBenchHeadDim = 64
	mixtralBenchInter   = 512 // expert intermediate (SwiGLU) dim
	mixtralBenchExperts = 8
	mixtralBenchTopK    = 2
	mixtralBenchPrefill = 256 // representative prefill window
)

func mixtralBenchGate(b *testing.B) {
	if !metaltest.RunMetalTests || !metal.MetalAvailable() {
		b.Skip("build with -tags metal_runtime to enable Metal runtime benchmarks")
	}
}

func mixtralBenchCfg() *MixtralConfig {
	return &MixtralConfig{
		HiddenSize:        mixtralBenchHidden,
		NumHiddenLayers:   1,
		IntermediateSize:  mixtralBenchInter,
		NumAttentionHeads: mixtralBenchHeads,
		NumKeyValueHeads:  mixtralBenchKVHeads,
		NumLocalExperts:   mixtralBenchExperts,
		NumExpertsPerTok:  mixtralBenchTopK,
		HeadDim:           mixtralBenchHeadDim,
		VocabSize:         32,
		RMSNormEps:        1e-5,
		RopeTheta:         1000000,
		Scale:             1.0,
	}
}

// mixtralBenchLinear builds a [out,in] random Linear and tracks it for free.
func mixtralBenchLinear(track *[]*metal.Array, out, in int32) *metal.Linear {
	w := metal.RandomUniform(-0.05, 0.05, []int32{out, in}, metal.DTypeFloat32)
	*track = append(*track, w)
	return metal.NewLinear(w, nil)
}

func mixtralBenchNorm(track *[]*metal.Array, hidden int32) *metal.RMSNormModule {
	w := metal.RandomUniform(0.5, 1.5, []int32{hidden}, metal.DTypeFloat32)
	*track = append(*track, w)
	return &metal.RMSNormModule{Weight: w}
}

// mixtralBenchAttention builds a GQA attention block with valid q/k/v/o
// projections for the bench head geometry.
func mixtralBenchAttention(track *[]*metal.Array) *metal.GQAAttention {
	qDim := int32(mixtralBenchHeads * mixtralBenchHeadDim)
	kvDim := int32(mixtralBenchKVHeads * mixtralBenchHeadDim)
	return &metal.GQAAttention{
		QProj: mixtralBenchLinear(track, qDim, mixtralBenchHidden),
		KProj: mixtralBenchLinear(track, kvDim, mixtralBenchHidden),
		VProj: mixtralBenchLinear(track, kvDim, mixtralBenchHidden),
		OProj: mixtralBenchLinear(track, mixtralBenchHidden, qDim),
	}
}

// mixtralBenchMoELayer assembles a synthetic sparse MoE decoder layer: dense
// attention spine + a router (hidden→experts) + batched SwiGLU experts built
// from per-expert w1/w3/w2 Linears, exactly as LoadMixtral wires them. Returns
// a free func that releases every backing array.
func mixtralBenchMoELayer(b *testing.B) (*MixtralDecoderLayer, func()) {
	b.Helper()
	var track []*metal.Array

	routerW := metal.RandomUniform(-0.05, 0.05, []int32{mixtralBenchExperts, mixtralBenchHidden}, metal.DTypeFloat32)
	track = append(track, routerW)

	gate := make([]*metal.Linear, mixtralBenchExperts)
	up := make([]*metal.Linear, mixtralBenchExperts)
	down := make([]*metal.Linear, mixtralBenchExperts)
	for e := 0; e < mixtralBenchExperts; e++ {
		gate[e] = mixtralBenchLinear(&track, mixtralBenchInter, mixtralBenchHidden) // w1
		up[e] = mixtralBenchLinear(&track, mixtralBenchInter, mixtralBenchHidden)   // w3
		down[e] = mixtralBenchLinear(&track, mixtralBenchHidden, mixtralBenchInter) // w2
	}
	switchExperts, ok := metal.NewMoESwiGLUExpertsFromLinears(gate, up, down)
	if !ok {
		b.Fatal("NewMoESwiGLUExpertsFromLinears returned ok=false")
	}

	layer := &MixtralDecoderLayer{
		Dense: &metal.DenseDecoderLayer{
			InputNorm:    mixtralBenchNorm(&track, mixtralBenchHidden),
			PostAttnNorm: mixtralBenchNorm(&track, mixtralBenchHidden),
			Attention:    mixtralBenchAttention(&track),
		},
		MoE: &MixtralMoEBlock{
			Router:           &metal.MoERouter{Weight: routerW},
			SwitchExperts:    switchExperts,
			NumLocalExperts:  mixtralBenchExperts,
			NumExpertsPerTok: mixtralBenchTopK,
		},
	}
	metal.Materialize(track...)
	free := func() {
		metal.FreeMoESwiGLUExperts(switchExperts)
		metal.Free(track...)
	}
	return layer, free
}

// mixtralBenchDenseLayer assembles a synthetic dense (non-MoE) decoder layer:
// attention spine + SiLU MLP, the interleaved-dense path.
func mixtralBenchDenseLayer(b *testing.B) (*MixtralDecoderLayer, func()) {
	b.Helper()
	var track []*metal.Array
	layer := &MixtralDecoderLayer{
		Dense: &metal.DenseDecoderLayer{
			InputNorm:    mixtralBenchNorm(&track, mixtralBenchHidden),
			PostAttnNorm: mixtralBenchNorm(&track, mixtralBenchHidden),
			Attention:    mixtralBenchAttention(&track),
			MLP: &metal.SiLUMLP{
				GateProj: mixtralBenchLinear(&track, mixtralBenchInter, mixtralBenchHidden),
				UpProj:   mixtralBenchLinear(&track, mixtralBenchInter, mixtralBenchHidden),
				DownProj: mixtralBenchLinear(&track, mixtralBenchHidden, mixtralBenchInter),
			},
		},
	}
	metal.Materialize(track...)
	return layer, func() { metal.Free(track...) }
}

// mixtralBenchInput builds the [B,L,hidden] hidden-state input the decoder
// layer consumes (post-embedding).
func mixtralBenchInput(l int32) *metal.Array {
	return metal.RandomUniform(-1, 1, []int32{1, l, mixtralBenchHidden}, metal.DTypeFloat32)
}

// BenchmarkMixtral_MoEPrefill is the MoE-layer prefill: a full L=256 window
// through mixtralDecoderLayerForward — RMSNorm, GQA attention over the window,
// router top-k + batched SwiGLU experts, two residual adds. The bulk
// per-MoE-layer prefill cost; the path the router/expert-dispatch alloc
// question lives on.
func BenchmarkMixtral_MoEPrefill(b *testing.B) {
	mixtralBenchGate(b)
	cfg := mixtralBenchCfg()
	layer, free := mixtralBenchMoELayer(b)
	defer free()
	x := mixtralBenchInput(mixtralBenchPrefill)
	metal.Materialize(x)
	defer metal.Free(x)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cache := metal.NewKVCache()
		out := mixtralDecoderLayerForward(layer, x, cache, 1, mixtralBenchPrefill, nil, cfg)
		if out == nil {
			b.Fatal("mixtralDecoderLayerForward returned nil")
		}
		_ = metal.Eval(out)
		metal.Free(out)
		cache.Reset()
	}
}

// BenchmarkMixtral_MoEDecode is the steady-state single-token MoE decode (L=1)
// with a primed KV cache — the per-token generation kernel and the analogue of
// the sequential-family decode benches, but for an attention+MoE layer. The
// router runs over one token and the experts gather one row, so any per-token
// Go-side dispatch alloc (the per-(token×expert) clone trap) surfaces here.
//
// Only the L=1 decode is timed: the per-iteration KV-cache prime (a full
// L=256 forward + its Eval, plus the cache reset) runs in StopTimer/StartTimer
// stopped time so its RandomUniform/forward cost is excluded — the gemma4
// router-bench lesson (a prime inside the timed loop dominates the number, not
// the kernel under test). The prefill input tensor is hoisted out and reused
// read-only; the cache is rebuilt each iteration so every decode attends to the
// same fixed history (a reused cache would grow its offset unboundedly).
func BenchmarkMixtral_MoEDecode(b *testing.B) {
	mixtralBenchGate(b)
	cfg := mixtralBenchCfg()
	layer, free := mixtralBenchMoELayer(b)
	defer free()
	step := mixtralBenchInput(1)
	prefill := mixtralBenchInput(mixtralBenchPrefill)
	metal.Materialize(step, prefill)
	defer metal.Free(step, prefill)
	cache := metal.NewKVCache()
	defer cache.Reset()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		cache.Reset()
		warm := mixtralDecoderLayerForward(layer, prefill, cache, 1, mixtralBenchPrefill, nil, cfg)
		_ = metal.Eval(warm)
		metal.Free(warm)
		b.StartTimer()

		out := mixtralDecoderLayerForward(layer, step, cache, 1, 1, nil, cfg)
		if out == nil {
			b.Fatal("mixtralDecoderLayerForward returned nil")
		}
		_ = metal.Eval(out)

		b.StopTimer()
		metal.Free(out)
		b.StartTimer()
	}
}

// BenchmarkMixtral_DenseDecode is the single-token decode of an interleaved
// dense (non-MoE) layer — SiLU MLP instead of experts. The contrast that
// isolates how much per-token Go-side cost is the MoE dispatch versus the
// shared attention+norm spine common to both paths. Same StopTimer-gated prime
// as MoEDecode: only the L=1 decode is timed.
func BenchmarkMixtral_DenseDecode(b *testing.B) {
	mixtralBenchGate(b)
	cfg := mixtralBenchCfg()
	layer, free := mixtralBenchDenseLayer(b)
	defer free()
	step := mixtralBenchInput(1)
	prefill := mixtralBenchInput(mixtralBenchPrefill)
	metal.Materialize(step, prefill)
	defer metal.Free(step, prefill)
	cache := metal.NewKVCache()
	defer cache.Reset()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		cache.Reset()
		warm := mixtralDecoderLayerForward(layer, prefill, cache, 1, mixtralBenchPrefill, nil, cfg)
		_ = metal.Eval(warm)
		metal.Free(warm)
		b.StartTimer()

		out := mixtralDecoderLayerForward(layer, step, cache, 1, 1, nil, cfg)
		if out == nil {
			b.Fatal("mixtralDecoderLayerForward returned nil")
		}
		_ = metal.Eval(out)

		b.StopTimer()
		metal.Free(out)
		b.StartTimer()
	}
}
