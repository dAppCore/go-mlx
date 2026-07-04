// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gptoss

import (
	"testing"

	"dappco.re/go/mlx/internal/metaltest"
	metal "dappco.re/go/mlx/pkg/metal"
)

// gptoss_bench_test.go measures the GPT-OSS sparse-MoE decoder hot path at a
// representative synthetic geometry — no model load (AX-11), the
// gemma4/qwen3-family sub-component bench style. It closes the measurement gap:
// gptoss had no benchmarks, so the per-(token×layer) Go-side allocation
// behaviour of its decoder loop was invisible while the dense + expert metal
// kernels it composes were swept elsewhere.
//
// The gptoss package itself is thin orchestration over package metal: it embeds,
// loops decoder layers, norms, and projects out. The allocations it OWNS (and
// therefore the only ones it could fix) live in gptOssDecoderLayerForward and
// ForwardMasked, NOT in the shared metal MoE dispatch (MoESwiGLUForward /
// MoESwiGLUExperts.Forward / moeRouterTopK). The benches below isolate the
// gptoss-owned layer driver; the MoE-dispatch bench additionally times the
// shared metal expert path so the split between owned and metal-intrinsic cost
// is legible in one place. As measured (M3 Ultra), the layer driver adds no
// Go-side heap allocation beyond the metal dispatch + Eval floor — the per-call
// gptOssToQwen3Config build is consumed in-frame and stack-allocated, so the
// per-(token×layer) DenseConfig is not a heap escape and there is nothing to
// hoist; the .Shape() rank check and metal.Slice bounds-literal traps are both
// absent (ForwardMasked already reads geometry via ShapeInto into a stack buf).
//
// Three shapes matter, and they exercise different code:
//
//   - Prefill (BenchmarkGptOss_MoELayerPrefill, L=256): one MoE decoder layer
//     over a full chunk — input norm → GQA attention → residual → post-attn
//     norm → router top-k → batched experts → residual. Exercises the dense
//     transformer block + the sparse expert dispatch at chunk width.
//   - Decode (BenchmarkGptOss_MoELayerDecode, L=1): the steady-state per-token
//     decoder layer — the kernel run once per generated token per layer. This
//     is the path where any per-(token×layer) Go-side allocation lands; the
//     gptoss-owned alloc profile is cache-length-insensitive (no prev==nil
//     branch-skip), so a fresh single-token input each call is representative.
//   - Router/expert dispatch (BenchmarkGptOss_MoEDispatch, L=1): the shared
//     metal.MoESwiGLUForward (router projection → top-k select → batched
//     gate/up/down switch-experts → weighted combine) in isolation, so its
//     cost is separable from the surrounding attention + residual work.
//
// All three force-Eval inside the timed region: a lazy return measures graph
// build, not compute (the MLA-bench lesson, inherited via the sibling benches).
//
// Run: go test -tags metal_runtime -bench='GptOss' -benchmem -run='^$' \
//   -benchtime=200ms ./go/pkg/metal/model/gptoss/
// (needs MLX_METALLIB_PATH set to dist/lib/mlx.metallib)

const (
	gptossBenchHidden   = 256 // model hidden size
	gptossBenchHeads    = 8   // attention heads
	gptossBenchKVHeads  = 2   // grouped-query KV heads
	gptossBenchHeadDim  = 32  // hidden/heads
	gptossBenchExperts  = 8   // local experts
	gptossBenchTopK     = 2   // experts per token
	gptossBenchMoEInter = 256 // expert intermediate size
	gptossBenchPrefillL = 256 // representative prefill chunk width
)

func gptossBenchGate(b *testing.B) {
	if !metaltest.RunMetalTests || !metal.MetalAvailable() {
		b.Skip("build with -tags metal_runtime to enable Metal runtime benchmarks")
	}
}

// gptossBenchConfig builds a GptOssConfig matching the synthetic layer geometry.
func gptossBenchConfig() *GptOssConfig {
	return &GptOssConfig{
		ModelType:             "gpt_oss",
		HiddenSize:            gptossBenchHidden,
		NumHiddenLayers:       1,
		NumAttentionHeads:     gptossBenchHeads,
		NumKeyValueHeads:      gptossBenchKVHeads,
		NumLocalExperts:       gptossBenchExperts,
		NumExpertsPerTok:      gptossBenchTopK,
		HeadDim:               gptossBenchHeadDim,
		VocabSize:             1024,
		RMSNormEps:            1e-5,
		RopeTheta:             1000000,
		MaxPositionEmbeddings: 4096,
		Scale:                 1.0,
	}
}

// gptossBenchLinear builds an unquantized [out,in] Linear with a cheap
// deterministic fill — values do not affect Go-side allocation counts.
func gptossBenchLinear(out, in int32) *metal.Linear {
	s := make([]float32, out*in)
	for i := range s {
		s[i] = 0.02 - 0.01*float32(i%7)
	}
	return metal.NewLinear(metal.FromValues(s, int(out), int(in)), nil)
}

// gptossBenchNorm builds an RMSNorm weight [hidden].
func gptossBenchNorm(hidden int32) *metal.RMSNormModule {
	s := make([]float32, hidden)
	for i := range s {
		s[i] = 1.0 + 0.01*float32(i%5)
	}
	return &metal.RMSNormModule{Weight: metal.FromValues(s, int(hidden))}
}

// gptossBenchMoELayer assembles a synthetic GPT-OSS MoE decoder layer: dense
// pre-norm attention block + a sparse expert block (router + batched
// switch-experts). Mirrors the LoadGptOss assembly without touching disk.
func gptossBenchMoELayer() (layer *GptOssDecoderLayer, free func()) {
	attn := &metal.GQAAttention{
		QProj: gptossBenchLinear(gptossBenchHeads*gptossBenchHeadDim, gptossBenchHidden),
		KProj: gptossBenchLinear(gptossBenchKVHeads*gptossBenchHeadDim, gptossBenchHidden),
		VProj: gptossBenchLinear(gptossBenchKVHeads*gptossBenchHeadDim, gptossBenchHidden),
		OProj: gptossBenchLinear(gptossBenchHidden, gptossBenchHeads*gptossBenchHeadDim),
	}
	experts := make([]*GptOssExpert, gptossBenchExperts)
	gate := make([]*metal.Linear, gptossBenchExperts)
	up := make([]*metal.Linear, gptossBenchExperts)
	down := make([]*metal.Linear, gptossBenchExperts)
	for e := range experts {
		g := gptossBenchLinear(gptossBenchMoEInter, gptossBenchHidden)
		u := gptossBenchLinear(gptossBenchMoEInter, gptossBenchHidden)
		d := gptossBenchLinear(gptossBenchHidden, gptossBenchMoEInter)
		experts[e] = &GptOssExpert{GateProj: g, UpProj: u, DownProj: d}
		gate[e], up[e], down[e] = g, u, d
	}
	switchExperts, ok := metal.NewMoESwiGLUExpertsFromLinears(gate, up, down)
	if !ok {
		panic("gptoss bench: NewMoESwiGLUExpertsFromLinears returned !ok")
	}
	router := &metal.MoERouter{Weight: metal.FromValues(routerFill(), gptossBenchExperts, gptossBenchHidden)}
	layer = &GptOssDecoderLayer{
		Dense: &metal.DenseDecoderLayer{
			InputNorm:    gptossBenchNorm(gptossBenchHidden),
			PostAttnNorm: gptossBenchNorm(gptossBenchHidden),
			Attention:    attn,
		},
		MoE: &GptOssMoEBlock{
			Router:        router,
			Experts:       experts,
			SwitchExperts: switchExperts,
		},
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

// routerFill builds the [experts,hidden] router projection weight.
func routerFill() []float32 {
	s := make([]float32, gptossBenchExperts*gptossBenchHidden)
	for i := range s {
		s[i] = -0.05 + 0.01*float32(i%11)
	}
	return s
}

// gptossBenchInput builds a hidden-state input [B=1, L, hidden].
func gptossBenchInput(l int32) *metal.Array {
	s := make([]float32, l*gptossBenchHidden)
	for i := range s {
		s[i] = 0.1 + 0.005*float32(i%23) - 0.003*float32(i%5)
	}
	return metal.FromValues(s, 1, int(l), int(gptossBenchHidden))
}

// BenchmarkGptOss_MoELayerPrefill is the chunk-width MoE decoder layer: the full
// gptOssDecoderLayerForward (norm → GQA attention → residual → post-attn norm →
// router top-k → batched experts → residual) over an L=256 chunk. The per-call
// gptOssToQwen3Config build feeds the attention sub-call directly and does not
// outlive it, so escape analysis stack-allocates it — this bench confirms the
// gptoss-owned layer driver adds no heap alloc beyond the metal dispatch floor.
func BenchmarkGptOss_MoELayerPrefill(b *testing.B) {
	gptossBenchGate(b)
	cfg := gptossBenchConfig()
	layer, free := gptossBenchMoELayer()
	defer free()
	x := gptossBenchInput(gptossBenchPrefillL)
	defer metal.Free(x)
	metal.Materialize(x)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Fresh KV cache per iteration: attention mutates (appends to) the cache,
		// so reusing one across b.N would grow it unboundedly and drift the
		// measurement. The single NewKVCache alloc is constant setup.
		c := metal.NewKVCache()
		out := gptOssDecoderLayerForward(layer, x, c, 1, gptossBenchPrefillL, nil, cfg)
		if out == nil {
			b.Fatal("gptOssDecoderLayerForward returned nil")
		}
		_ = metal.Eval(out)
		metal.Free(out)
	}
}

// BenchmarkGptOss_MoELayerDecode is the steady-state single-token MoE decoder
// layer (L=1) — the per-token generation kernel run once per layer per token.
// This is the path where any per-(token×layer) Go-side allocation would surface;
// the gptOssToQwen3Config build runs inside the call here, per layer, per token,
// but is consumed in-frame by the attention path and so does not escape to the
// heap (verified empirically by the batched bench below, which sees no per-layer
// alloc delta).
func BenchmarkGptOss_MoELayerDecode(b *testing.B) {
	gptossBenchGate(b)
	cfg := gptossBenchConfig()
	layer, free := gptossBenchMoELayer()
	defer free()
	x := gptossBenchInput(1)
	defer metal.Free(x)
	metal.Materialize(x)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c := metal.NewKVCache()
		out := gptOssDecoderLayerForward(layer, x, c, 1, 1, nil, cfg)
		if out == nil {
			b.Fatal("gptOssDecoderLayerForward returned nil")
		}
		_ = metal.Eval(out)
		metal.Free(out)
	}
}

// gptossBenchLayers is the layer count chained per timed op in the batched
// decode bench. A real GPT-OSS model runs every generated token through every
// decoder layer; chaining N layers per op amplifies the per-layer Go-side cost
// above the metal Eval/pool sampling noise floor, so any per-layer Go-side heap
// alloc would be resolvable in allocs/op. (It measures flat at the metal floor —
// the per-call gptOssToQwen3Config build is stack-allocated, not a heap escape.)
const gptossBenchLayers = 32

// BenchmarkGptOss_MoEDecodeBatched32 chains 32 single-token MoE decoder layers
// per timed op — the deep-model decode shape. Per-op allocs/op divided by 32 is
// the real per-layer cost below the Eval sync floor; it is dominated entirely by
// the metal expert/attention dispatch, with no measurable Go-side per-layer heap
// allocation from the gptoss layer driver.
func BenchmarkGptOss_MoEDecodeBatched32(b *testing.B) {
	gptossBenchGate(b)
	cfg := gptossBenchConfig()
	layers := make([]*GptOssDecoderLayer, gptossBenchLayers)
	frees := make([]func(), gptossBenchLayers)
	for i := range layers {
		layers[i], frees[i] = gptossBenchMoELayer()
	}
	defer func() {
		for _, f := range frees {
			f()
		}
	}()
	x0 := gptossBenchInput(1)
	defer metal.Free(x0)
	metal.Materialize(x0)
	caches := make([]metal.Cache, gptossBenchLayers)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j := range caches {
			caches[j] = metal.NewKVCache()
		}
		outs := make([]*metal.Array, 0, gptossBenchLayers)
		x := x0
		for j, layer := range layers {
			out := gptOssDecoderLayerForward(layer, x, caches[j], 1, 1, nil, cfg)
			outs = append(outs, out)
			x = out
		}
		if err := metal.Eval(outs...); err != nil {
			b.Fatalf("Eval: %v", err)
		}
		metal.Free(outs...)
	}
}

// BenchmarkGptOss_MoEDispatch isolates the shared metal MoE dispatch
// (MoESwiGLUForward: router projection → top-k select → batched gate/up/down
// switch-experts → weighted combine) at single-token width, separating the
// sparse expert cost from the surrounding attention + residual work. The
// allocations here belong to package metal, not gptoss — included so the
// owned/intrinsic split is measurable in one run.
func BenchmarkGptOss_MoEDispatch(b *testing.B) {
	gptossBenchGate(b)
	cfg := gptossBenchConfig()
	layer, free := gptossBenchMoELayer()
	defer free()
	x := gptossBenchInput(1)
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
