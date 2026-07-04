// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package deltanet

import (
	"testing"

	"dappco.re/go/mlx/internal/metaltest"
	metal "dappco.re/go/mlx/pkg/metal"
)

// deltanet_bench_test.go measures the DeltaNet delta-rule kernel at a
// representative synthetic geometry — no model load (AX-11), Codex's
// mamba2/rwkv7 bench style. Unlike mamba2/rwkv7 the question here is sharper:
// DeltaRuleChunk's win rests on a batched triangular metal.Solve per ≤64 window
// (chunked.go), and mlx linalg runs on the CPU stream — MLX inserts a GPU↔CPU
// cross-stream copy per Solve. At L=256 the chunked path does 4 such windows, so
// this pair answers whether chunked still beats the serial form once that
// cross-stream cost is paid. Measured: it does — chunked wins clearly at L=256,
// but by less than the matmul-only mixers (mamba2 ~10×, rwkv7 ~6×) because the
// per-window CPU-stream Solve adds a cross-stream copy — a tax, not a cliff, and
// the deliberate no-custom-kernel tradeoff. Both forms are force-Eval'd in the
// timed region (the MLA-bench lesson: a lazy return measures graph build, not
// compute).

const (
	deltaBenchH = 4
	deltaBenchD = 64
	deltaBenchL = 256 // 4 × chunkWidth(64) windows — exercises the inter-window state thread
)

func deltaBenchGate(b *testing.B) {
	if !metaltest.RunMetalTests || !metal.MetalAvailable() {
		b.Skip("build with -tags metal_runtime to enable Metal runtime benchmarks")
	}
}

// deltaBenchInputs builds q/k/v [1,H,L,D] and beta [1,H,L,1] with cheap
// deterministic fills — content is irrelevant to timing. beta is kept small and
// strictly positive so M = I + diag(β)·strict_lower(KKᵀ) stays well-conditioned
// for the Solve.
func deltaBenchInputs(h, l, d int32) (q, k, v, beta *metal.Array, free func()) {
	qkv := func() []float32 {
		s := make([]float32, h*l*d)
		for i := range s {
			s[i] = 0.03 + 0.01*float32(i%13) - 0.02*float32(i%5)
		}
		return s
	}
	bvals := make([]float32, h*l)
	for i := range bvals {
		bvals[i] = 0.1 + 0.05*float32(i%7) // (0.1, 0.4], positive
	}
	q = metal.FromValues(qkv(), 1, int(h), int(l), int(d))
	k = metal.FromValues(qkv(), 1, int(h), int(l), int(d))
	v = metal.FromValues(qkv(), 1, int(h), int(l), int(d))
	beta = metal.FromValues(bvals, 1, int(h), int(l), 1)
	free = func() { metal.Free(q, k, v, beta) }
	return q, k, v, beta, free
}

// BenchmarkDeltaRule_Sequential is the serial reference: L delta-rule steps,
// one timestep at a time. The slow baseline the chunked form must beat.
func BenchmarkDeltaRule_Sequential(b *testing.B) {
	deltaBenchGate(b)
	q, k, v, beta, free := deltaBenchInputs(deltaBenchH, deltaBenchL, deltaBenchD)
	defer free()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, st := DeltaRuleChunkSequential(q, k, v, beta, nil, 0.125, 1e-6)
		if out == nil {
			b.Fatal("DeltaRuleChunkSequential returned nil")
		}
		_ = metal.Eval(out, st)
		metal.Free(out, st)
	}
}

// BenchmarkDeltaRule_Chunked is the parallel WY form: ≤64-wide windows, a batched
// triangular metal.Solve per window (CPU-stream LAPACK → cross-stream copy),
// threading state between windows. Compared to _Sequential this shows whether the
// chunked win survives the per-window cross-stream cost at a multi-window length.
func BenchmarkDeltaRule_Chunked(b *testing.B) {
	deltaBenchGate(b)
	q, k, v, beta, free := deltaBenchInputs(deltaBenchH, deltaBenchL, deltaBenchD)
	defer free()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, st := DeltaRuleChunk(q, k, v, beta, nil, 0.125, 1e-6)
		if out == nil {
			b.Fatal("DeltaRuleChunk returned nil")
		}
		_ = metal.Eval(out, st)
		metal.Free(out, st)
	}
}
