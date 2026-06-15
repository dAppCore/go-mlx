// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma3

import (
	"testing"

	"dappco.re/go/mlx/internal/metaltest"
	metal "dappco.re/go/mlx/pkg/metal"
)

// gemma3_bench_test.go measures the Gemma 3 text trunk (Forward → per-layer
// forward → Attention.forward + MLP.Forward) at a small synthetic geometry —
// no model load (AX-11), the gla/mla/retnet/mamba2 bench style. It closes the
// measurement gap: gemma3 had no benchmarks, so its per-token decode-path
// allocation behaviour was invisible while its mixer siblings were swept.
//
// gemma3 runs the dense GQA attention + GELU MLP path. The two shapes that
// matter exercise different code:
//
//   - The prefill path (L>1, fresh cache) runs the causal [L,L] attention with
//     the AsStrided virtual transposes for q/k/v, the Q/K RMSNorm, RoPE, the
//     full SDPA, and the Transpose4 + Reshape read-out — the first turn of a
//     sequence. This is where any L-shaped or per-layer Go-side shape alloc
//     compounds across the trunk.
//   - The single-token decode path (L==1 over a warm cache) is the steady-state
//     generation kernel: the per-token KVCache.Update Slice views, GQA RepeatKV,
//     and the L=1 SDPA. It is the highest-multiplier path (one call per emitted
//     token per layer), so a per-call Go-side alloc here is paid on every token.
//
// Both forms are force-Eval'd inside the timed region: gemma3's Forward returns
// a lazy graph (the engine evals once per token after all layers compose), so a
// lazy return would measure graph BUILD, not compute (the MLA-bench lesson,
// inherited via gla/retnet). The absolute µs therefore carries one eval sync per
// Forward that the real decode loop amortises across all layers — read the
// cross-run deltas and the allocs/op, not the absolute number.
//
// The decode benchmark warms the cache with a prefill OUTSIDE the timer (the mla
// decode pattern), so each timed iteration runs exactly one decode step over a
// fixed history. Benching decode from a cold cache would measure the
// unrepresentative first-token case.

const (
	gemmaBenchHidden    = 256
	gemmaBenchHeads     = 4
	gemmaBenchKVHeads   = 1   // GQA: repeatFactor = 4 (exercises RepeatKV)
	gemmaBenchHeadDim   = 64  // Gemma 3 uses 256; 64 keeps the synthetic kernel cheap, same code path
	gemmaBenchInter     = 512 // MLP intermediate
	gemmaBenchVocab     = 320 // small token table
	gemmaBenchLayers    = 2   // one sliding + one full — both RoPE thetas + both cache types
	gemmaBenchPrefill   = 64
	gemmaBenchSlidingWin = 32
)

func requireGemmaMetalRuntime(b *testing.B) {
	b.Helper()
	if !metaltest.RunMetalTests || !metal.MetalAvailable() {
		b.Skip("build with -tags metal_runtime to enable Metal runtime benchmarks")
	}
}

// gemmaBenchLinear builds a [out,in] dense Linear with a cheap deterministic
// fill — values do not affect allocation counts or op timing.
func gemmaBenchLinear(out, in int32, base float32) *metal.Linear {
	n := out * in
	w := make([]float32, n)
	for i := int32(0); i < n; i++ {
		w[i] = base + 0.011*float32(i%7) - 0.013*float32(i%5)
	}
	return metal.NewLinear(metal.FromValues(w, int(out), int(in)), nil)
}

// gemmaBenchNorm builds an RMSNorm weight vector [dim].
func gemmaBenchNorm(dim int32, base float32) *metal.RMSNormModule {
	w := make([]float32, dim)
	for i := int32(0); i < dim; i++ {
		w[i] = base + 0.005*float32(i%9)
	}
	return &metal.RMSNormModule{Weight: metal.FromValues(w, int(dim))}
}

// gemmaBenchModel builds a synthetic GemmaModel at the bench geometry and runs
// precomputeScaledWeights so the (1+weight) norm scales are materialised exactly
// as a loaded model has them. No safetensors, no tokenizer — the trunk only
// touches weights and config.
func gemmaBenchModel() *GemmaModel {
	cfg := &TextConfig{
		ModelType:            "gemma3",
		HiddenSize:           gemmaBenchHidden,
		NumHiddenLayers:      gemmaBenchLayers,
		IntermediateSize:     gemmaBenchInter,
		NumAttentionHeads:    gemmaBenchHeads,
		NumKeyValueHeads:     gemmaBenchKVHeads,
		HeadDim:              gemmaBenchHeadDim,
		VocabSize:            gemmaBenchVocab,
		RMSNormEps:           1e-6,
		RopeTheta:            1000000,
		RopeLocalBaseFreq:    10000,
		SlidingWindow:        gemmaBenchSlidingWin,
		SlidingWindowPattern: 2, // layer 0 sliding, layer 1 full
		Scale:                0.125,
		EmbeddingScale:       16.0,
	}

	qOut := cfg.NumAttentionHeads * cfg.HeadDim
	kvOut := cfg.NumKeyValueHeads * cfg.HeadDim

	embedW := make([]float32, cfg.VocabSize*cfg.HiddenSize)
	for i := range embedW {
		embedW[i] = 0.02 + 0.01*float32(i%11)
	}
	embed := &metal.Embedding{Weight: metal.FromValues(embedW, int(cfg.VocabSize), int(cfg.HiddenSize))}

	m := &GemmaModel{
		EmbedTokens: embed,
		Layers:      make([]*DecoderLayer, cfg.NumHiddenLayers),
		Norm:        gemmaBenchNorm(cfg.HiddenSize, 0.9),
		Output:      embed.AsLinear(), // tied
		Cfg:         cfg,
		modelType:   "gemma3",
	}

	for i := int32(0); i < cfg.NumHiddenLayers; i++ {
		m.Layers[i] = &DecoderLayer{
			InputNorm:    gemmaBenchNorm(cfg.HiddenSize, 0.91),
			PostAttnNorm: gemmaBenchNorm(cfg.HiddenSize, 0.92),
			PreFFNorm:    gemmaBenchNorm(cfg.HiddenSize, 0.93),
			PostFFNorm:   gemmaBenchNorm(cfg.HiddenSize, 0.94),
			Attention: &Attention{
				QProj: gemmaBenchLinear(qOut, cfg.HiddenSize, 0.03),
				KProj: gemmaBenchLinear(kvOut, cfg.HiddenSize, 0.02),
				VProj: gemmaBenchLinear(kvOut, cfg.HiddenSize, 0.015),
				OProj: gemmaBenchLinear(cfg.HiddenSize, qOut, 0.012),
				QNorm: gemmaBenchNorm(cfg.HeadDim, 0.95),
				KNorm: gemmaBenchNorm(cfg.HeadDim, 0.96),
			},
			MLP: &metal.MLP{
				GateProj: gemmaBenchLinear(cfg.IntermediateSize, cfg.HiddenSize, 0.014),
				UpProj:   gemmaBenchLinear(cfg.IntermediateSize, cfg.HiddenSize, 0.016),
				DownProj: gemmaBenchLinear(cfg.HiddenSize, cfg.IntermediateSize, 0.018),
			},
			LayerIdx:  i,
			IsSliding: isLayerSliding(i, cfg.SlidingWindowPattern),
		}
	}

	precomputeScaledWeights(m)
	return m
}

// gemmaBenchTokens builds a [B,L] int32 token-index array within the synthetic
// vocab. Content is irrelevant to timing — only the shape and valid indices.
func gemmaBenchTokens(bsz, l int32) *metal.Array {
	ids := make([]int32, bsz*l)
	for i := range ids {
		ids[i] = int32(i) % gemmaBenchVocab
	}
	return metal.FromValues(ids, int(bsz), int(l))
}

// BenchmarkGemma3_Forward_Prefill measures a fresh prefill chunk (L tokens,
// empty cache) over the full 2-layer trunk: embedding + scale, per-layer
// input-norm → GQA attention (AsStrided q/k/v transposes, Q/K-norm, RoPE, causal
// SDPA, Transpose4 read-out) → post-norms → GELU MLP, then the final norm + tied
// output projection. The first turn of every sequence.
func BenchmarkGemma3_Forward_Prefill(b *testing.B) {
	requireGemmaMetalRuntime(b)
	m := gemmaBenchModel()
	tokens := gemmaBenchTokens(1, gemmaBenchPrefill)
	defer metal.Free(tokens)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		caches := m.NewCache()
		out := m.Forward(tokens, caches)
		_ = metal.Eval(out) // Forward returns a lazy graph; force the compute
		metal.Free(out)
	}
}

// BenchmarkGemma3_Forward_Decode measures one single-token decode step over a
// warm cache. The warm-cache prefill and its Eval run OUTSIDE the timer, so each
// iteration times exactly one decode step (L==1) over the per-layer KV caches:
// the highest-multiplier path, run once per emitted token per layer. Exercises
// KVCache.Update's Slice views, RepeatKV (GQA repeatFactor=4), and the L=1 SDPA.
func BenchmarkGemma3_Forward_Decode(b *testing.B) {
	requireGemmaMetalRuntime(b)
	m := gemmaBenchModel()
	prefill := gemmaBenchTokens(1, gemmaBenchPrefill)
	defer metal.Free(prefill)
	step := gemmaBenchTokens(1, 1)
	defer metal.Free(step)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		b.StopTimer()
		caches := m.NewCache()
		pre := m.Forward(prefill, caches)
		_ = metal.Eval(pre) // materialise the warm cache before timing the step
		metal.Free(pre)
		b.StartTimer()

		out := m.Forward(step, caches)
		_ = metal.Eval(out) // force the decode step's actual compute
		metal.Free(out)
	}
}
