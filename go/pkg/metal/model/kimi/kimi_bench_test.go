// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package kimi

import (
	"testing"

	"dappco.re/go/mlx/internal/metaltest"
	metal "dappco.re/go/mlx/pkg/metal"
)

// kimi_bench_test.go measures the Kimi sparse-MoE decoder hot path at a
// representative synthetic geometry — no model load (AX-11), the
// gptoss/qwen3-family sub-component bench style. It closes the measurement gap:
// kimi had no benchmarks, so the per-(token×layer) Go-side allocation behaviour
// of its decoder loop was invisible while the dense + expert metal kernels it
// composes were swept elsewhere.
//
// The kimi package itself is thin orchestration over package metal: it embeds,
// loops decoder layers, norms, and projects out. The allocations it OWNS (and
// therefore the only ones it could fix) live in kimiDecoderLayerForward and
// ForwardMasked, NOT in the shared metal MoE dispatch (MoESwiGLUForward /
// MoESwiGLUExperts.Forward / moeRouterTopK), the GQA attention, the KV cache, or
// the metal Eval floor. The benches below isolate the kimi-owned layer driver;
// the MoE-dispatch bench additionally times the shared metal expert path so the
// split between owned and metal-intrinsic cost is legible in one place.
//
// Kimi is a MIXED dense/MoE arch — kimiDecoderLayerForward branches on
// isMoELayer() (decoder_sparse_step) and, when the layer is sparse, on whether
// metal.MoESwiGLUForward links (ok). The synthetic layer here is built so
// isMoELayer() is true AND MoESwiGLUForward returns ok=true (Router weight set +
// NewMoESwiGLUExpertsFromLinears succeeds), so the bench drives the real sparse
// path, not the near-no-op Add(h, normed2) fall-through. The b.Fatal guards make
// a misconstructed fixture fail loudly rather than measure nothing.
//
// ForwardMasked already reads geometry via tokens.ShapeInto into a stack buffer
// (no len(.Shape()) rank trap), and the only kimi-owned per-(token×layer) heap
// candidate is kimiToQwen3Config's &metal.DenseConfig{} build — which feeds the
// attention sub-call directly and is never retained, returned, or closure-
// captured by metal's attention path, so escape analysis stack-allocates it. The
// chained-layers bench (Batched32) is the instrument that PROVES this: chaining
// 32 layers per timed op amplifies any per-layer Go-side heap escape above the
// single-Eval pool/sync noise floor, so a real DenseConfig heap escape would be
// resolvable in allocs/op ÷ 32. It measures flat at the metal floor.
//
// Four shapes matter, and they exercise different code:
//
//   - Prefill (BenchmarkKimi_MoELayerPrefill, L=256): one MoE decoder layer over
//     a full chunk — input norm → GQA attention → residual → post-attn norm →
//     router top-k → batched experts → residual. Exercises the dense transformer
//     block + the sparse expert dispatch at chunk width.
//   - Decode (BenchmarkKimi_MoELayerDecode, L=1): the steady-state per-token
//     decoder layer — the kernel run once per generated token per layer over a
//     warm KV cache. This is the path where any per-(token×layer) Go-side
//     allocation lands.
//   - Batched decode (BenchmarkKimi_MoEDecodeBatched32): 32 chained single-token
//     MoE layers per timed op — the deep-model decode shape and the escape proof
//     for the per-layer DenseConfig build.
//   - Router/expert dispatch (BenchmarkKimi_MoEDispatch, L=1): the shared
//     metal.MoESwiGLUForward in isolation, so the sparse expert cost is separable
//     from the surrounding attention + residual work. Its allocations belong to
//     package metal, not kimi — included so the owned/intrinsic split is
//     measurable in one run.
//
// All force-Eval inside the timed region: a lazy return measures graph build,
// not compute (the MLA-bench lesson, inherited via the sibling benches).
//
// Run: go test -tags metal_runtime -bench='Kimi' -benchmem -run='^$' \
//   -benchtime=200ms ./go/pkg/metal/model/kimi/
// (needs MLX_METALLIB_PATH set to dist/lib/mlx.metallib)

const (
	kimiBenchHidden   = 256 // model hidden size
	kimiBenchHeads    = 8   // attention heads
	kimiBenchKVHeads  = 2   // grouped-query KV heads
	kimiBenchHeadDim  = 32  // hidden/heads
	kimiBenchExperts  = 8   // routed experts
	kimiBenchTopK     = 2   // experts per token
	kimiBenchMoEInter = 256 // expert intermediate size
	kimiBenchPrefillL = 256 // representative prefill chunk width
	kimiBenchLayers   = 32  // chained layers per op in the batched decode bench
)

func kimiBenchGate(b *testing.B) {
	if !metaltest.RunMetalTests || !metal.MetalAvailable() {
		b.Skip("build with -tags metal_runtime to enable Metal runtime benchmarks")
	}
}

// kimiBenchConfig builds a KimiConfig matching the synthetic layer geometry.
// SparseStep is left 0 so isMoELayer() is governed purely by the layer's MoE
// block being present (the synthetic layer below always sets it).
func kimiBenchConfig() *KimiConfig {
	return &KimiConfig{
		ModelType:             "kimi",
		HiddenSize:            kimiBenchHidden,
		NumHiddenLayers:       1,
		NumAttentionHeads:     kimiBenchHeads,
		NumKeyValueHeads:      kimiBenchKVHeads,
		NRoutedExperts:        kimiBenchExperts,
		NumExpertsPerTok:      kimiBenchTopK,
		HeadDim:               kimiBenchHeadDim,
		VocabSize:             1024,
		RMSNormEps:            1e-5,
		RopeTheta:             1000000,
		MaxPositionEmbeddings: 4096,
		Scale:                 1.0,
	}
}

// kimiBenchLinear builds an unquantized [out,in] Linear with a cheap
// deterministic fill — values do not affect Go-side allocation counts.
func kimiBenchLinear(out, in int32) *metal.Linear {
	s := make([]float32, out*in)
	for i := range s {
		s[i] = 0.02 - 0.01*float32(i%7)
	}
	return metal.NewLinear(metal.FromValues(s, int(out), int(in)), nil)
}

// kimiBenchNorm builds an RMSNorm weight [hidden].
func kimiBenchNorm(hidden int32) *metal.RMSNormModule {
	s := make([]float32, hidden)
	for i := range s {
		s[i] = 1.0 + 0.01*float32(i%5)
	}
	return &metal.RMSNormModule{Weight: metal.FromValues(s, int(hidden))}
}

// kimiRouterFill builds the [experts,hidden] router projection weight.
func kimiRouterFill() []float32 {
	s := make([]float32, kimiBenchExperts*kimiBenchHidden)
	for i := range s {
		s[i] = -0.05 + 0.01*float32(i%11)
	}
	return s
}

// kimiBenchMoELayer assembles a synthetic Kimi MoE decoder layer: dense pre-norm
// attention block + a sparse expert block (router + batched switch-experts).
// Mirrors the LoadKimi assembly without touching disk. The router weight and the
// switch-experts are both set so isMoELayer() is true and metal.MoESwiGLUForward
// returns ok=true — the real sparse path runs, not the Add(h, normed2)
// fall-through.
func kimiBenchMoELayer() (layer *KimiDecoderLayer, free func()) {
	attn := &metal.GQAAttention{
		QProj: kimiBenchLinear(kimiBenchHeads*kimiBenchHeadDim, kimiBenchHidden),
		KProj: kimiBenchLinear(kimiBenchKVHeads*kimiBenchHeadDim, kimiBenchHidden),
		VProj: kimiBenchLinear(kimiBenchKVHeads*kimiBenchHeadDim, kimiBenchHidden),
		OProj: kimiBenchLinear(kimiBenchHidden, kimiBenchHeads*kimiBenchHeadDim),
	}
	experts := make([]*KimiExpert, kimiBenchExperts)
	gate := make([]*metal.Linear, kimiBenchExperts)
	up := make([]*metal.Linear, kimiBenchExperts)
	down := make([]*metal.Linear, kimiBenchExperts)
	for e := range experts {
		g := kimiBenchLinear(kimiBenchMoEInter, kimiBenchHidden)
		u := kimiBenchLinear(kimiBenchMoEInter, kimiBenchHidden)
		d := kimiBenchLinear(kimiBenchHidden, kimiBenchMoEInter)
		experts[e] = &KimiExpert{GateProj: g, UpProj: u, DownProj: d}
		gate[e], up[e], down[e] = g, u, d
	}
	switchExperts, ok := metal.NewMoESwiGLUExpertsFromLinears(gate, up, down)
	if !ok {
		panic("kimi bench: NewMoESwiGLUExpertsFromLinears returned !ok")
	}
	router := &metal.MoERouter{Weight: metal.FromValues(kimiRouterFill(), kimiBenchExperts, kimiBenchHidden)}
	layer = &KimiDecoderLayer{
		Dense: &metal.DenseDecoderLayer{
			InputNorm:    kimiBenchNorm(kimiBenchHidden),
			PostAttnNorm: kimiBenchNorm(kimiBenchHidden),
			Attention:    attn,
		},
		MoE: &KimiMoEBlock{
			Router:        router,
			Experts:       experts,
			SwitchExperts: switchExperts,
		},
	}
	if !layer.isMoELayer() {
		panic("kimi bench: synthetic layer is not an MoE layer")
	}
	free = func() {
		metal.FreeLinear(attn.QProj)
		metal.FreeLinear(attn.KProj)
		metal.FreeLinear(attn.VProj)
		metal.FreeLinear(attn.OProj)
		metal.FreeRMSNorm(layer.Dense.InputNorm)
		metal.FreeRMSNorm(layer.Dense.PostAttnNorm)
		metal.Free(router.Weight)
		metal.FreeMoESwiGLUExperts(switchExperts)
		for _, e := range experts {
			metal.FreeLinear(e.GateProj)
			metal.FreeLinear(e.UpProj)
			metal.FreeLinear(e.DownProj)
		}
	}
	return layer, free
}

// kimiBenchInput builds a hidden-state input [B=1, L, hidden].
func kimiBenchInput(l int32) *metal.Array {
	s := make([]float32, l*kimiBenchHidden)
	for i := range s {
		s[i] = 0.1 + 0.005*float32(i%23) - 0.003*float32(i%5)
	}
	return metal.FromValues(s, 1, int(l), int(kimiBenchHidden))
}

// BenchmarkKimi_MoELayerPrefill is the chunk-width MoE decoder layer: the full
// kimiDecoderLayerForward (norm → GQA attention → residual → post-attn norm →
// router top-k → batched experts → residual) over an L=256 chunk. The per-call
// kimiToQwen3Config build feeds the attention sub-call directly and does not
// outlive it, so escape analysis stack-allocates it — this bench confirms the
// kimi-owned layer driver adds no heap alloc beyond the metal dispatch floor.
func BenchmarkKimi_MoELayerPrefill(b *testing.B) {
	kimiBenchGate(b)
	cfg := kimiBenchConfig()
	layer, free := kimiBenchMoELayer()
	defer free()
	x := kimiBenchInput(kimiBenchPrefillL)
	defer metal.Free(x)
	metal.Materialize(x)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Fresh KV cache per iteration: attention mutates (appends to) the cache,
		// so reusing one across b.N would grow it unboundedly and drift the
		// measurement. The single NewKVCache alloc is constant setup.
		c := metal.NewKVCache()
		out := kimiDecoderLayerForward(layer, x, c, 1, kimiBenchPrefillL, nil, cfg)
		if out == nil {
			b.Fatal("kimiDecoderLayerForward returned nil")
		}
		_ = metal.Eval(out)
		metal.Free(out)
	}
}

// BenchmarkKimi_MoELayerDecode is the steady-state single-token MoE decoder layer
// (L=1) — the per-token generation kernel run once per layer per token. This is
// the path where any per-(token×layer) Go-side allocation would surface; the
// kimiToQwen3Config build runs inside the call here, per layer, per token, but is
// consumed in-frame by the attention path and so does not escape to the heap
// (verified by the batched bench below, which sees no per-layer alloc delta).
func BenchmarkKimi_MoELayerDecode(b *testing.B) {
	kimiBenchGate(b)
	cfg := kimiBenchConfig()
	layer, free := kimiBenchMoELayer()
	defer free()
	x := kimiBenchInput(1)
	defer metal.Free(x)
	metal.Materialize(x)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c := metal.NewKVCache()
		out := kimiDecoderLayerForward(layer, x, c, 1, 1, nil, cfg)
		if out == nil {
			b.Fatal("kimiDecoderLayerForward returned nil")
		}
		_ = metal.Eval(out)
		metal.Free(out)
	}
}

// BenchmarkKimi_MoEDecodeBatched32 chains 32 single-token MoE decoder layers per
// timed op — the deep-model decode shape. Per-op allocs/op divided by 32 is the
// real per-layer cost below the Eval sync floor; this is the escape proof for the
// per-layer kimiToQwen3Config &DenseConfig{} build. It measures flat at the metal
// expert/attention dispatch floor, with no measurable Go-side per-layer heap
// allocation from the kimi layer driver.
func BenchmarkKimi_MoEDecodeBatched32(b *testing.B) {
	kimiBenchGate(b)
	cfg := kimiBenchConfig()
	layers := make([]*KimiDecoderLayer, kimiBenchLayers)
	frees := make([]func(), kimiBenchLayers)
	for i := range layers {
		layers[i], frees[i] = kimiBenchMoELayer()
	}
	defer func() {
		for _, f := range frees {
			f()
		}
	}()
	x0 := kimiBenchInput(1)
	defer metal.Free(x0)
	metal.Materialize(x0)
	caches := make([]metal.Cache, kimiBenchLayers)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j := range caches {
			caches[j] = metal.NewKVCache()
		}
		outs := make([]*metal.Array, 0, kimiBenchLayers)
		x := x0
		for j, layer := range layers {
			out := kimiDecoderLayerForward(layer, x, caches[j], 1, 1, nil, cfg)
			outs = append(outs, out)
			x = out
		}
		if err := metal.Eval(outs...); err != nil {
			b.Fatalf("Eval: %v", err)
		}
		metal.Free(outs...)
	}
}

// BenchmarkKimi_MoEDispatch isolates the shared metal MoE dispatch
// (MoESwiGLUForward: router projection → top-k select → batched gate/up/down
// switch-experts → weighted combine) at single-token width, separating the sparse
// expert cost from the surrounding attention + residual work. The allocations
// here belong to package metal, not kimi — included so the owned/intrinsic split
// is measurable in one run.
func BenchmarkKimi_MoEDispatch(b *testing.B) {
	kimiBenchGate(b)
	cfg := kimiBenchConfig()
	layer, free := kimiBenchMoELayer()
	defer free()
	x := kimiBenchInput(1)
	defer metal.Free(x)
	metal.Materialize(x)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, ok := metal.MoESwiGLUForward(x, layer.MoE.Router, cfg.topK(), layer.MoE.SwitchExperts)
		if !ok {
			b.Fatal("MoESwiGLUForward returned !ok")
		}
		_ = metal.Eval(out)
		metal.Free(out)
	}
}
