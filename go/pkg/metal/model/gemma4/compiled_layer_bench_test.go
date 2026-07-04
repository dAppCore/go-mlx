// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package gemma4

import (
	"testing"

	"dappco.re/go/mlx/pkg/metal"
)

// BenchmarkCompiledLayer_OwnerPreCapDecode benches one Gemma 4 decoder layer's
// whole-layer compiled decode step (compiled_layer.go's compiledDecodeForward)
// through the owner pre-capacity regime — the per-token-per-layer serving path.
//
// It uses the in-process tiny quantised model the coverage tests build (no model
// download, AX-11-clean: hidden 32, two layers), which is the only construction
// proven to reach the compiled closure rather than declining. The decode-hit
// counter guards that the closure actually served every iteration, so the bench
// measures the compiled path and not a silent fall-through.
//
// The figure of interest is the per-call heap allocation of the canonical
// dynamic-inputs-plus-weights slice the closure is Call'd with; isolate it with
//
//	go tool pprof -alloc_objects -list compiledDecodeForward <memprofile>
//
// since the graph eval (RMSNorm, projections, SDPA) dominates the aggregate
// -benchmem totals.
func BenchmarkCompiledLayer_OwnerPreCapDecode(b *testing.B) {
	b.Cleanup(metal.SetRuntimeGate(metal.GateCompiledLayerDecode, true))
	m := cov4clBuildQuantModel(b, cov4clQuantConfig{
		hidden: 32, intermediate: 64, headDim: 32, layers: 2, sliding: 4, pattern: 2,
	})
	layer := m.Layers[1] // full-attention owner
	if layer.LayerType != "full_attention" {
		b.Fatalf("layer 1 type = %s, want full_attention", layer.LayerType)
	}
	const maxSize = 64

	x := seqArray(0.01, 1, 1, 32)
	defer metal.Free(x)

	// A fresh cache primed to one token: each compiled step advances the offset,
	// so the pre-cap regime only holds while offset < maxSize. The loop re-primes
	// (timer-stopped) before the cache would overflow; that setup allocates only
	// inside cov4clPrimeFixed, so it never colours the FLAT alloc profile of
	// compiledDecodeForward itself.
	cache := cov4clPrimeFixed(b, maxSize, 32, 1)
	defer func() { cache.Reset() }() // resets the live cache after any re-prime reassigns it

	// Warm the trace + verify the compiled closure serves before timing.
	warm, _, ok := layer.compiledDecodeForward(x, cache, 1, 1, nil, nil, sharedKV{}, m.Cfg)
	if !ok {
		b.Fatal("warm-up declined the compiled path")
	}
	if err := metal.Eval(warm); err != nil {
		b.Fatalf("warm-up eval: %v", err)
	}
	metal.Free(warm)

	before := CompiledLayerDecodeHits()
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		if cache.Offset() >= maxSize-2 {
			b.StopTimer()
			cache.Reset()
			cache = cov4clPrimeFixed(b, maxSize, 32, 1)
			b.StartTimer()
		}
		out, _, ok := layer.compiledDecodeForward(x, cache, 1, 1, nil, nil, sharedKV{}, m.Cfg)
		if !ok {
			b.Fatal("compiled path declined mid-bench")
		}
		if err := metal.Eval(out); err != nil {
			b.Fatalf("eval: %v", err)
		}
		metal.Free(out)
	}
	b.StopTimer()
	if CompiledLayerDecodeHits() == before {
		b.Fatal("compiled closure never served during the bench")
	}
}
