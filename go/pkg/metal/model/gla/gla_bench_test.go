// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gla

import (
	"testing"

	"dappco.re/go/mlx/internal/metaltest"
	metal "dappco.re/go/mlx/pkg/metal"
)

// gla_bench_test.go measures the GatedChunk gated-linear-attention kernel at a
// representative synthetic geometry — no model load (AX-11), the
// retnet/mamba2/rwkv7/deltanet bench style. It closes the long-standing
// measurement gap: gla had no benchmarks at all, so its hot-path allocation
// behaviour was invisible while its siblings (rwkv7's WKV7, mamba2's SSDScan,
// retnet's retention) were swept.
//
// Two shapes matter, and they exercise different code:
//
//   - The chunked prefill path (compute → computeWindow → advanceWindowState,
//     plus the cross-window read-out when a prior state is carried) runs once
//     per chunk. At L=256 compute splits into ⌈256/64⌉ = 4 sub-chunk windows
//     and threads state between them through the rank-4 metal.Slice4 window
//     cuts (already the scalar-pass form — the SliceAxis trap that gave the
//     sequential siblings their wins is absent here by construction).
//   - The single-token decode path (decodeStep, L==1) is the steady-state
//     generation kernel: one recurrent update, no cumsum, no window split, no
//     exp(±b) re-basing. It is the analogue of the rwkv7/mamba2/retnet
//     sequential decode that gave the big wins — but it composes from different
//     ops (an outer-product Matmul + a per-key-dim α Exp), so its allocation
//     profile is its own question. decodeStep also re-runs resolve() per token,
//     so this is the path where any per-call Go-side validation alloc lands.
//
// decodeStep is benched with a non-nil prior state: that is the only branch
// that runs the α=exp(g₀) decay Mul/Add, i.e. the real steady-state decode
// after the first token. Benching it with prev==nil would skip that work and
// measure the unrepresentative first-token case.
//
// Both forms are force-Eval'd inside the timed region: a lazy return measures
// graph build, not compute (the MLA-bench lesson, inherited via retnet).
//
// Run: go test -tags metal_runtime -bench='GLA' -benchmem -run='^$' \
//   -benchtime=200ms ./go/pkg/metal/model/gla/
// (needs MLX_METALLIB_PATH set to dist/lib/mlx.metallib)

const (
	glaBenchH = 4
	glaBenchD = 64
	glaBenchL = 256 // 4 × subChunk(64) windows — exercises the inter-window state thread
)

func glaBenchGate(b *testing.B) {
	if !metaltest.RunMetalTests || !metal.MetalAvailable() {
		b.Skip("build with -tags metal_runtime to enable Metal runtime benchmarks")
	}
}

// glaBenchQKV builds q/k/v [1,H,L,D] with cheap deterministic fills — values
// do not affect allocation counts or op timing.
func glaBenchQKV(h, l, d int32) (q, k, v *metal.Array, free func()) {
	fill := func(seed float32) []float32 {
		s := make([]float32, h*l*d)
		for i := range s {
			s[i] = seed + 0.01*float32(i%13) - 0.02*float32(i%5)
		}
		return s
	}
	q = metal.FromValues(fill(0.3), 1, int(h), int(l), int(d))
	k = metal.FromValues(fill(-0.2), 1, int(h), int(l), int(d))
	v = metal.FromValues(fill(0.15), 1, int(h), int(l), int(d))
	free = func() { metal.Free(q, k, v) }
	return q, k, v, free
}

// glaBenchGateArr builds the per-position per-key-dim LOG-forget gate g [1,H,L,Dk].
// GLA requires g ≤ 0 (α = exp(g) ∈ (0,1]); the fill stays in a mild negative
// band so exp(±b) over a sub-chunk window is well-bounded. Values are honest but
// irrelevant to allocation counts.
func glaBenchGateArr(h, l, d int32) *metal.Array {
	s := make([]float32, h*l*d)
	for i := range s {
		s[i] = -0.05 - 0.01*float32(i%7) // (-0.11, -0.05], strictly negative
	}
	return metal.FromValues(s, 1, int(h), int(l), int(d))
}

// glaBenchState builds a non-nil prior recurrent state [1,H,Dk,Dv].
func glaBenchState(h, d int32) *metal.Array {
	s := make([]float32, h*d*d)
	for i := range s {
		s[i] = 0.05 + 0.01*float32(i%7)
	}
	return metal.FromValues(s, 1, int(h), int(d), int(d))
}

// BenchmarkGLA_ChunkPrefill is the parallel-form prefill: gated causal Q̃ K̃ᵀ V
// over an L=256 chunk with no incoming state. Exercises compute → 4×
// computeWindow → advanceWindowState (the cumsum, exp(±b), and the rank-4
// Slice4 window cuts), with no cross-window read-out from a prior state.
func BenchmarkGLA_ChunkPrefill(b *testing.B) {
	glaBenchGate(b)
	q, k, v, free := glaBenchQKV(glaBenchH, glaBenchL, glaBenchD)
	defer free()
	g := glaBenchGateArr(glaBenchH, glaBenchL, glaBenchD)
	defer metal.Free(g)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, st := GatedChunk(q, k, v, g, nil, 0)
		if out == nil {
			b.Fatal("GatedChunk returned nil")
		}
		_ = metal.Eval(out, st)
		metal.Free(out, st)
	}
}

// BenchmarkGLA_ChunkCarry is the cross-window/cross-chunk path: an L=256 chunk
// fed a non-nil prior state, so computeWindow also runs the q̃·S_prev read-out
// and advanceWindowState's diag(exp(b_wl))·S_prev decay. The representative
// mid-sequence prefill chunk.
func BenchmarkGLA_ChunkCarry(b *testing.B) {
	glaBenchGate(b)
	q, k, v, free := glaBenchQKV(glaBenchH, glaBenchL, glaBenchD)
	defer free()
	g := glaBenchGateArr(glaBenchH, glaBenchL, glaBenchD)
	defer metal.Free(g)
	prev := glaBenchState(glaBenchH, glaBenchD)
	defer metal.Free(prev)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, st := GatedChunk(q, k, v, g, prev, 0)
		if out == nil {
			b.Fatal("GatedChunk returned nil")
		}
		_ = metal.Eval(out, st)
		metal.Free(out, st)
	}
}

// BenchmarkGLA_DecodeStep is the steady-state single-token decode (L==1) with a
// carried state — the per-token generation kernel and the analogue of the
// rwkv7/mamba2/retnet sequential decode benches. resolve() runs once per call
// here, so this is the path that surfaces any per-token Go-side validation
// allocation. The state is fed back in each iteration's input (a fresh prev
// each call is fine; decodeStep's allocation profile is unaffected by whether
// prev is reused).
func BenchmarkGLA_DecodeStep(b *testing.B) {
	glaBenchGate(b)
	q, k, v, free := glaBenchQKV(glaBenchH, 1, glaBenchD)
	defer free()
	g := glaBenchGateArr(glaBenchH, 1, glaBenchD)
	defer metal.Free(g)
	prev := glaBenchState(glaBenchH, glaBenchD)
	defer metal.Free(prev)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, st := GatedChunk(q, k, v, g, prev, 0)
		if out == nil {
			b.Fatal("GatedChunk returned nil")
		}
		_ = metal.Eval(out, st)
		metal.Free(out, st)
	}
}
