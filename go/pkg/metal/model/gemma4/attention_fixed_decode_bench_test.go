// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package gemma4

import (
	"testing"

	"dappco.re/go/mlx/pkg/metal"
)

// BenchmarkAttention_FixedDecode_Owner benches the Go-graph Gemma 4 attention
// decode step in isolation (attention.go's Gemma4Attention.forward, reached
// through the mixer Forward) over the FixedKVCache owner single-token regime —
// the per-token per-layer attention work the compiled whole-layer closure
// bypasses when it serves, and the path every layer takes whenever the compiled
// closure declines.
//
// Only the attention block runs (no FFN/MLP), so the alloc profile is the
// attention path alone: the QKV/output projections, q/k-norm, RoPE, the native
// fixed single-token attention call, the offset Array, and the output
// transpose/reshape/o-proj. The MixerCtx is built inline in the loop body (not
// captured by a returning closure) so escape analysis stack-allocates it exactly
// as the decoder loop's own construction does (decoder_layer.go:34 reports
// "&metal.MixerCtx{...} does not escape"); a closure that returns the result
// would heap-promote the ctx and mis-report a per-token alloc the real path does
// not have.
//
// It drives a real loaded attention block (the in-process tiny quantised model
// the coverage tests build — no download, AX-11-clean: hidden 32, headDim 32)
// against a primed FixedKVCache.
//
// The figure of interest is the per-call Go-side heap allocation of the
// attention step; isolate it with
//
//	go tool pprof -alloc_objects -list 'attention.go' <memprofile>
//
// since the graph eval (projections, RMSNorm, SDPA) dominates the aggregate
// -benchmem totals, and any remaining allocs that delegate into pkg/metal
// (the fixed-cache retire list, the eval bridge) show there, not in gemma4.
func BenchmarkAttention_FixedDecode_Owner(b *testing.B) {
	m := cov4clBuildQuantModel(b, cov4clQuantConfig{
		hidden: 32, intermediate: 64, headDim: 32, layers: 2, sliding: 4, pattern: 2,
	})
	layer := m.Layers[1] // full-attention owner
	if layer.LayerType != "full_attention" {
		b.Fatalf("layer 1 type = %s, want full_attention", layer.LayerType)
	}
	attn := layer.Attention
	cfg := m.Cfg
	const maxSize = 64

	x := seqArray(0.01, 1, 1, 32)
	defer metal.Free(x)

	runtimeMasks := newGemma4RuntimeMaskCache()
	defer runtimeMasks.Free()

	// A fresh cache primed to one token: each decode step advances the offset, so
	// the owner pre-cap regime only holds while offset < maxSize. The loop
	// re-primes (timer-stopped) before the cache would overflow; that setup
	// allocates only inside cov4clPrimeFixed, so it never colours the FLAT alloc
	// profile of the attention forward itself.
	cache := cov4clPrimeFixed(b, maxSize, 32, 1)
	defer func() { cache.Reset() }()

	// Warm the trace before timing.
	{
		ctx := &metal.MixerCtx{Cache: cache, B: 1, L: 1, Extra: gemma4MixerExtra{cfg: cfg, runtimeMasks: runtimeMasks}}
		warm, warmKV := attn.Forward(x, ctx)
		if err := metal.Eval(warm); err != nil {
			b.Fatalf("warm-up eval: %v", err)
		}
		warmKV.Free()
		metal.Free(warm)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		if cache.Offset() >= maxSize-2 {
			b.StopTimer()
			cache.Reset()
			cache = cov4clPrimeFixed(b, maxSize, 32, 1)
			b.StartTimer()
		}
		ctx := &metal.MixerCtx{Cache: cache, B: 1, L: 1, Extra: gemma4MixerExtra{cfg: cfg, runtimeMasks: runtimeMasks}}
		out, kv := attn.Forward(x, ctx)
		if err := metal.Eval(out); err != nil {
			b.Fatalf("eval: %v", err)
		}
		kv.Free()
		metal.Free(out)
	}
	b.StopTimer()
}
