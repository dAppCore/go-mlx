// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package qwen3

import (
	"testing"

	"dappco.re/go/mlx/internal/metaltest"
	metal "dappco.re/go/mlx/pkg/metal"
)

// qwen3_bench_test.go measures the Qwen 3 dense decoder trunk — one
// metal.DenseDecoderLayer (the pre-norm GQA-attention + SwiGLU-MLP block that
// Qwen3Model.ForwardMasked composes per layer) — at a representative synthetic
// geometry. No model load (AX-11): the fixture builds the layer's weights from
// metal.FromValues, exactly the mla/gla bench style. This closes the
// measurement gap — qwen3 had loader-error tests but no allocation bench, so
// the dense-trunk per-token allocation behaviour was invisible while its mixer
// siblings (mla's Forward, gla's GatedChunk) were swept.
//
// The benched block is the SHARED dense decoder layer that every dense family
// (Qwen 2/3, Llama, Mistral, Hermes, Granite, Phi, GLM) runs, driven here at a
// Qwen 3 geometry specifically: GQA with a 2× K/V repeat factor AND the Q/K RMS
// norms that Qwen 3 adds (Qwen 2 and the others leave them nil). So this bench
// exercises the q_norm/k_norm Forward calls that the plain-Llama path skips.
//
// Two shapes matter, and they exercise different code:
//
//   - The prefill path (L>1, empty cache) runs the q/k/v projections, the
//     rank-4 AsStrided reshapes, Q/K-norm + RoPE, the causal [L,L] SDPA, and
//     the SwiGLU FFN — the first turn of every sequence.
//   - The single-token decode path (L==1, warm cache) is the steady-state
//     generation kernel and the highest-multiplier path: it is run once per
//     emitted token. The warm-cache prefill and its Eval are excluded from the
//     timer (StopTimer/StartTimer, the mla-bench pattern), so each iteration
//     times exactly one decode step appending to a populated KV cache.
//
// Both forms are force-Eval'd inside the timed region: DenseDecoderLayer.Forward
// returns a lazy graph (the engine evals once per token after all layers
// compose), so a lazy return would measure graph CONSTRUCTION, not compute (the
// MLA-bench lesson, inherited via retnet/gla). The absolute µs therefore carries
// one eval sync per layer-Forward that the real decode loop amortises across all
// layers — read the cross-bench deltas and the alloc counts, not the absolute
// ns/op.
//
// Run: go test -tags metal_runtime -bench='Qwen3' -benchmem -run='^$' \
//   -benchtime=50x ./go/pkg/metal/model/qwen3/
// (needs MLX_METALLIB_PATH set to dist/lib/mlx.metallib)

const (
	qwen3BenchHidden  = 256
	qwen3BenchHeads   = 4
	qwen3BenchKVHeads = 2  // GQA: repeatFactor = heads/kvHeads = 2
	qwen3BenchHeadDim = 64 // heads*headDim = 256 attention width
	qwen3BenchInter   = 512
	qwen3BenchPrefill = 128
)

func qwen3BenchMetal(b *testing.B) {
	b.Helper()
	if !metaltest.RunMetalTests || !metal.MetalAvailable() {
		b.Skip("build with -tags metal_runtime to enable Metal runtime benchmarks")
	}
}

// qwen3BenchLin builds an [out,in] projection weight with a cheap deterministic
// fill — content is irrelevant to allocation counts or op timing, only the
// shape and valid floats matter.
func qwen3BenchLin(out, in int32, base float32) *metal.Linear {
	n := out * in
	w := make([]float32, n)
	for i := int32(0); i < n; i++ {
		w[i] = base + 0.011*float32(i%13) - 0.007*float32(i%5)
	}
	return metal.NewLinear(metal.FromValues(w, int(out), int(in)), nil)
}

// qwen3BenchNorm builds an RMSNorm weight vector of the given width.
func qwen3BenchNorm(width int32) *metal.RMSNormModule {
	w := make([]float32, width)
	for i := int32(0); i < width; i++ {
		w[i] = 1.0 + 0.01*float32(i%7)
	}
	return &metal.RMSNormModule{Weight: metal.FromValues(w, int(width))}
}

// qwen3BenchLayer builds a single Qwen 3 dense decoder layer at the bench
// geometry: pre/post norms, GQA attention (4 query heads, 2 K/V heads → 2×
// repeat) WITH the Qwen 3 Q/K RMS norms, and a SwiGLU MLP.
func qwen3BenchLayer() *metal.DenseDecoderLayer {
	var attnW int32 = qwen3BenchHeads * qwen3BenchHeadDim // 256
	var kvW int32 = qwen3BenchKVHeads * qwen3BenchHeadDim // 128
	return &metal.DenseDecoderLayer{
		InputNorm:    qwen3BenchNorm(qwen3BenchHidden),
		PostAttnNorm: qwen3BenchNorm(qwen3BenchHidden),
		Attention: &metal.GQAAttention{
			QProj: qwen3BenchLin(attnW, qwen3BenchHidden, 0.02),
			KProj: qwen3BenchLin(kvW, qwen3BenchHidden, 0.015),
			VProj: qwen3BenchLin(kvW, qwen3BenchHidden, 0.018),
			OProj: qwen3BenchLin(qwen3BenchHidden, attnW, 0.012),
			QNorm: qwen3BenchNorm(qwen3BenchHeadDim),
			KNorm: qwen3BenchNorm(qwen3BenchHeadDim),
		},
		MLP: &metal.SiLUMLP{
			GateProj: qwen3BenchLin(qwen3BenchInter, qwen3BenchHidden, 0.01),
			UpProj:   qwen3BenchLin(qwen3BenchInter, qwen3BenchHidden, 0.013),
			DownProj: qwen3BenchLin(qwen3BenchHidden, qwen3BenchInter, 0.009),
		},
	}
}

func freeQwen3BenchLayer(l *metal.DenseDecoderLayer) {
	metal.FreeRMSNorm(l.InputNorm)
	metal.FreeRMSNorm(l.PostAttnNorm)
	metal.FreeLinear(l.Attention.QProj)
	metal.FreeLinear(l.Attention.KProj)
	metal.FreeLinear(l.Attention.VProj)
	metal.FreeLinear(l.Attention.OProj)
	metal.FreeRMSNorm(l.Attention.QNorm)
	metal.FreeRMSNorm(l.Attention.KNorm)
	metal.FreeLinear(l.MLP.GateProj)
	metal.FreeLinear(l.MLP.UpProj)
	metal.FreeLinear(l.MLP.DownProj)
}

// qwen3BenchCfg returns the DenseConfig at the bench geometry, with Scale
// (1/sqrt(head_dim)) set as the loader would.
func qwen3BenchCfg() *metal.DenseConfig {
	cfg := &metal.DenseConfig{
		RopeTheta: 1000000.0,
		Scale:     0.125, // 1/sqrt(64)
	}
	cfg.HiddenSize = qwen3BenchHidden
	cfg.NumAttentionHeads = qwen3BenchHeads
	cfg.NumKeyValueHeads = qwen3BenchKVHeads
	cfg.HeadDim = qwen3BenchHeadDim
	cfg.IntermediateSize = qwen3BenchInter
	cfg.RMSNormEps = 1e-6
	return cfg
}

// qwen3BenchInput builds a [B,L,hidden] activation with a cheap deterministic
// fill — content is irrelevant to timing, only the shape and valid floats matter.
func qwen3BenchInput(bsz, l int32) *metal.Array {
	n := bsz * l * qwen3BenchHidden
	vals := make([]float32, n)
	for i := int32(0); i < n; i++ {
		vals[i] = 0.05 + 0.01*float32(i%17) - 0.02*float32(i%5)
	}
	return metal.FromValues(vals, int(bsz), int(l), qwen3BenchHidden)
}

// BenchmarkQwen3_DecoderLayer_Prefill measures a fresh prefill chunk (L tokens,
// empty cache): the q/k/v projections, the rank-4 AsStrided reshapes, Q/K-norm,
// RoPE, the causal [L,L] SDPA with the 2× GQA K/V repeat, and the SwiGLU FFN —
// the first turn of every sequence.
func BenchmarkQwen3_DecoderLayer_Prefill(b *testing.B) {
	qwen3BenchMetal(b)
	layer := qwen3BenchLayer()
	defer freeQwen3BenchLayer(layer)
	cfg := qwen3BenchCfg()
	x := qwen3BenchInput(1, qwen3BenchPrefill)
	defer metal.Free(x)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c := metal.NewKVCache()
		out := layer.Forward(x, c, 1, qwen3BenchPrefill, nil, cfg)
		_ = metal.Eval(out) // Forward returns a lazy graph; force the compute
		metal.Free(out)
		c.Reset()
	}
}

// BenchmarkQwen3_DecoderLayer_Decode measures one single-token decode step
// (L==1) appending to a warm KV cache — the steady-state generation kernel and
// the highest-multiplier path (run once per emitted token). The warm-cache
// prefill and its Eval are excluded from the timer (StopTimer/StartTimer), so
// each iteration times exactly one decode step over a populated cache.
//
// This is where any per-token Go-side allocation in the qwen3-package trunk
// would surface; the dense layer's compute itself composes from package-metal
// primitives that the AX-11 sweeps have already been through (Transpose4 in the
// attention output transpose, ShapeInto in the model Forward).
func BenchmarkQwen3_DecoderLayer_Decode(b *testing.B) {
	qwen3BenchMetal(b)
	layer := qwen3BenchLayer()
	defer freeQwen3BenchLayer(layer)
	cfg := qwen3BenchCfg()
	warm := qwen3BenchInput(1, qwen3BenchPrefill)
	defer metal.Free(warm)
	step := qwen3BenchInput(1, 1)
	defer metal.Free(step)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		c := metal.NewKVCache()
		pre := layer.Forward(warm, c, 1, qwen3BenchPrefill, nil, cfg)
		_ = metal.Eval(pre) // materialise the warm cache before timing the step
		metal.Free(pre)
		b.StartTimer()

		out := layer.Forward(step, c, 1, 1, nil, cfg)
		_ = metal.Eval(out) // force the decode step's actual compute
		metal.Free(out)

		b.StopTimer()
		c.Reset()
	}
}
