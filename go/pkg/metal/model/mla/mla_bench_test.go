// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package mla

import (
	"fmt"
	"testing"

	metal "dappco.re/go/mlx/pkg/metal"
)

// mla_bench_test.go measures the MLA mixer's Forward at representative synthetic
// shapes — no model load (AX-11). It mirrors the mamba2/rwkv7 bench style: build
// the fixture once, ResetTimer, run the kernel + Free in the loop.
//
// Methodology gotcha: MLA's Forward returns a LAZY graph (the engine evals once
// per token after all layers compose), so the benchmarks force metal.Eval inside
// the timed region. Without it they measure graph CONSTRUCTION (~19µs, 5 allocs),
// not compute (~360µs). The absolute µs therefore carries one eval sync per
// Forward that the real decode loop amortises across all layers — read the
// cross-bench deltas, not the absolute number.
//
// The decode benchmark sweeps history because MLA re-up-projects the FULL cached
// latent to per-head K/V every step (upProjectKV over totalL); an O(history)
// climb would flag the naive re-expansion (vs DeepSeek's query-absorption).
// Measured: ~flat from hist=128 to 512 — the re-projection is cheap next to fixed
// per-Forward overhead, so absorption is not worth chasing at these lengths.

const (
	benchHidden  = 512
	benchHeads   = 8
	benchHeadDim = 64 // benchHeads*benchHeadDim = 512 attention width
	benchKVLat   = 128
	benchQLat    = 128
	benchPrefill = 128
)

// mlaBenchMixer builds an MLA mixer at the bench geometry: rank-128 KV/query
// latents, 8 heads × 64. WUK fans the KV latent to per-head interleaved K|V
// (heads*2*HeadDim); WUQ to the query heads (heads*HeadDim).
func mlaBenchMixer() *Mixer {
	return &Mixer{
		WDKV:     mlaLin(benchKVLat, benchHidden, 0.02),
		WUK:      mlaLin(benchHeads*2*benchHeadDim, benchKVLat, 0.01),
		WDQ:      mlaLin(benchQLat, benchHidden, 0.03),
		WUQ:      mlaLin(benchHeads*benchHeadDim, benchQLat, 0.015),
		OProj:    mlaLin(benchHidden, benchHeads*benchHeadDim, 0.012),
		NumHeads: benchHeads,
		HeadDim:  benchHeadDim,
		Scale:    0.125, // 1/sqrt(64)
	}
}

func freeMLABench(m *Mixer) {
	metal.FreeLinear(m.WDKV)
	metal.FreeLinear(m.WUK)
	metal.FreeLinear(m.WDQ)
	metal.FreeLinear(m.WUQ)
	metal.FreeLinear(m.OProj)
}

// mlaBenchInput builds a [B,L,hidden] activation with a cheap deterministic fill
// — content is irrelevant to timing, only the shape and valid floats matter.
func mlaBenchInput(bsz, l int32) *metal.Array {
	n := bsz * l * benchHidden
	vals := make([]float32, n)
	for i := int32(0); i < n; i++ {
		vals[i] = 0.05 + 0.01*float32(i%17) - 0.02*float32(i%5)
	}
	return metal.FromValues(vals, int(bsz), int(l), benchHidden)
}

// BenchmarkMLA_Forward_Prefill measures a fresh prefill chunk (L tokens, empty
// cache): the WDKV/WDQ down-projections, the latent up-projection to per-head
// K/V, the internal causal-mask build, and the [L,L] attention — the first turn
// of every sequence.
func BenchmarkMLA_Forward_Prefill(b *testing.B) {
	requireMetalRuntime(b)
	m := mlaBenchMixer()
	defer freeMLABench(m)
	x := mlaBenchInput(1, benchPrefill)
	defer metal.Free(x)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c := metal.NewLatentKVCache()
		out, _ := m.Forward(x, &metal.MixerCtx{Cache: c, B: 1, L: benchPrefill})
		_ = metal.Eval(out) // Forward returns a lazy graph; force the compute
		metal.Free(out)
		c.Reset()
	}
}

// BenchmarkMLA_Forward_Decode measures one single-token decode step over a warm
// latent cache, swept across history length. The warm-cache rebuild and its Eval
// are excluded from the timer, so each iteration times exactly one decode step at
// a fixed history. Reads ~flat across hist 128→512 (measured): the per-step
// full-latent re-up-projection is dwarfed by fixed per-Forward overhead.
func BenchmarkMLA_Forward_Decode(b *testing.B) {
	requireMetalRuntime(b)
	for _, hist := range []int32{128, 512} {
		b.Run(fmt.Sprintf("hist=%d", hist), func(b *testing.B) {
			m := mlaBenchMixer()
			defer freeMLABench(m)
			histX := mlaBenchInput(1, hist)
			defer metal.Free(histX)
			step := mlaBenchInput(1, 1)
			defer metal.Free(step)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				b.StopTimer()
				c := metal.NewLatentKVCache()
				pre, _ := m.Forward(histX, &metal.MixerCtx{Cache: c, B: 1, L: hist})
				_ = metal.Eval(pre) // materialise the warm cache before timing the step
				metal.Free(pre)
				b.StartTimer()

				out, _ := m.Forward(step, &metal.MixerCtx{Cache: c, B: 1, L: 1})
				_ = metal.Eval(out) // force the decode step's actual compute
				metal.Free(out)

				b.StopTimer()
				c.Reset()
			}
		})
	}
}
