// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gsa

import (
	"math"
	"testing"

	"dappco.re/go/mlx/internal/metaltest"
	metal "dappco.re/go/mlx/pkg/metal"
)

// gsa_bench_test.go measures the GSA gated-slot recurrence at a representative
// synthetic geometry — no model load (AX-11), the retnet/mamba2/rwkv7/deltanet
// bench style. It closes the measurement gap: gsa had no benchmarks, so the
// per-timestep allocation behaviour of its sequential recurrence (the Go loop
// in recurrence() issuing stepSlice + the slot-update ops) was invisible while
// its siblings (rwkv7's WKV7, mamba2's SSDScan, retnet's retention) were swept.
//
// recurrence() is benched directly — the kernel, like retnet benches
// RetentionChunk — so the loop's allocations are isolated from the Q/K/V/F/G/O
// projections that Forward() runs around it. Two shapes matter:
//
//   - The chunked prefill (L=256): the loop runs once per timestep, and each
//     step issues five stepSlice calls (q/k/v/g/s) plus the slot-memory update
//     (two Matmul outer-products, two decay Mul, two Add, a Reshape, a Softmax).
//     stepSlice already routes through metal.Slice4 (the scalar-pass form — no
//     []int32{…} bounds-slice literal), so the named SliceAxis-family alloc the
//     other siblings carried is not present here; the prefill bench exists to
//     prove that on the profile, not to assume it.
//   - The single-token decode (L==1): the steady-state generation kernel — one
//     pass through the loop body with a carried prior slot memory, the analogue
//     of the rwkv7/mamba2/retnet sequential decode. Benched with a non-nil
//     prior (sk0/sv0) so it measures the real mid-sequence update, not the
//     unrepresentative zero-state first step.
//
// Both forms force-Eval the returned out + final slots inside the timed region:
// a lazy return measures graph build, not compute (the MLA-bench lesson,
// inherited via retnet/deltanet).
//
// Run: go test -tags metal_runtime -bench='GSA' -benchmem -run='^$' \
//   -benchtime=200ms ./go/pkg/metal/model/gsa/
// (needs MLX_METALLIB_PATH set to dist/lib/mlx.metallib)

const (
	gsaBenchH     = 4   // attention heads
	gsaBenchHeadK = 64  // key/query dim
	gsaBenchHeadV = 64  // value dim
	gsaBenchSlots = 16  // memory slots
	gsaBenchL     = 256 // representative prefill chunk
)

func gsaBenchGate(b *testing.B) {
	if !metaltest.RunMetalTests || !metal.MetalAvailable() {
		b.Skip("build with -tags metal_runtime to enable Metal runtime benchmarks")
	}
}

// gsaBenchFill builds a cheap deterministic []float32 — content does not affect
// allocation counts or op timing, but a non-degenerate fill keeps the math (and
// the softmax) honest.
func gsaBenchFill(n int, seed float32) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = seed + 0.01*float32(i%13) - 0.02*float32(i%5)
	}
	return s
}

// gsaBenchInputs builds the recurrence inputs at [B=1,H,L,*] plus a carried
// initial slot memory (sk0 [1,H,HeadK,Slots], sv0 [1,H,Slots,HeadV]). g is a
// log-domain decay in (-inf,0]; s = 1-exp(g) is the slot-write weight — both
// derived so the values are representative rather than arbitrary, though only
// the shapes drive allocation.
func gsaBenchInputs(h, l, headK, headV, slots int32) (q, k, v, g, s, sk0, sv0 *metal.Array, free func()) {
	q = metal.FromValues(gsaBenchFill(int(h*l*headK), 0.3), 1, int(h), int(l), int(headK))
	k = metal.FromValues(gsaBenchFill(int(h*l*headK), -0.2), 1, int(h), int(l), int(headK))
	v = metal.FromValues(gsaBenchFill(int(h*l*headV), 0.15), 1, int(h), int(l), int(headV))

	// g_i = log(sigmoid(raw)) ∈ (-inf,0]; s_i = 1-exp(g_i) ∈ [0,1).
	gRaw := gsaBenchFill(int(h*l*slots), 0.1)
	gVals := make([]float32, len(gRaw))
	sVals := make([]float32, len(gRaw))
	for i, x := range gRaw {
		gi := float32(math.Log(1.0 / (1.0 + math.Exp(-float64(x)))))
		gVals[i] = gi
		sVals[i] = 1 - float32(math.Exp(float64(gi)))
	}
	g = metal.FromValues(gVals, 1, int(h), int(l), int(slots))
	s = metal.FromValues(sVals, 1, int(h), int(l), int(slots))

	sk0 = metal.FromValues(gsaBenchFill(int(h*headK*slots), 0.05), 1, int(h), int(headK), int(slots))
	sv0 = metal.FromValues(gsaBenchFill(int(h*slots*headV), 0.05), 1, int(h), int(slots), int(headV))

	free = func() { metal.Free(q, k, v, g, s, sk0, sv0) }
	return q, k, v, g, s, sk0, sv0, free
}

// BenchmarkGSA_RecurrencePrefill is the L=256 chunk: the sequential gated-slot
// loop over 256 timesteps with a carried initial slot memory. Exercises the
// per-step stepSlice (×5) + the slot-update ops the whole recurrence is built
// from — the representative prefill chunk.
func BenchmarkGSA_RecurrencePrefill(b *testing.B) {
	gsaBenchGate(b)
	q, k, v, g, s, sk0, sv0, free := gsaBenchInputs(gsaBenchH, gsaBenchL, gsaBenchHeadK, gsaBenchHeadV, gsaBenchSlots)
	defer free()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, skN, svN := recurrence(q, k, v, g, s, sk0, sv0)
		if out == nil {
			b.Fatal("recurrence returned nil")
		}
		_ = metal.Eval(out, skN, svN)
		metal.Free(out, skN, svN)
	}
}

// BenchmarkGSA_DecodeStep is the steady-state single-token decode (L==1) with a
// carried prior slot memory — the per-token generation kernel and the analogue
// of the rwkv7/mamba2/retnet sequential decode benches. One pass through the
// loop body: five stepSlice calls + the slot update + the softmax read.
func BenchmarkGSA_DecodeStep(b *testing.B) {
	gsaBenchGate(b)
	q, k, v, g, s, sk0, sv0, free := gsaBenchInputs(gsaBenchH, 1, gsaBenchHeadK, gsaBenchHeadV, gsaBenchSlots)
	defer free()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, skN, svN := recurrence(q, k, v, g, s, sk0, sv0)
		if out == nil {
			b.Fatal("recurrence returned nil")
		}
		_ = metal.Eval(out, skN, svN)
		metal.Free(out, skN, svN)
	}
}
