// SPDX-Licence-Identifier: EUPL-1.2
//
// Quant ORDERING probe. By bandwidth math a single-token decode matmul MUST
// order q4 > q6 > q8 in tok/s (fewer weight bytes = faster) once the shape is
// big enough to be bandwidth-bound. The whole-model matrix shows the opposite
// (e2b q8=100 > q6=81; 31b q6/q4 = 0.44 vs the ~0.69 byte-ratio), which is
// impossible physics — so one of the q6 kernel paths is burning more than its
// byte advantage. This bench isolates WHERE: same fixture generator, same
// chained-single-token harness (64 calls -> 1 Eval, no per-op sync floor),
// SetBytes reports ACHIEVED bandwidth per path. If q6 achieves materially
// lower GB/s than q8 at the same dim, the q6 kernel is the defect; if q6
// orders correctly here, the inversion lives outside the matvec/gemm kernels
// (layer routing, cache, output head).
package metal

import (
	"fmt"
	"testing"
)

func benchmarkQuantDecodeOrdering(b *testing.B, bits, dim int, useMatVec bool) {
	const N = 64
	fixture := quantizedLinearDenseMatVecFixture(b, dim, dim, 64, bits, 41)
	lin := fixture.linear
	defer FreeLinear(lin)
	x0 := RandomUniform(-1, 1, []int32{1, 1, int32(dim)}, DTypeFloat32)
	Materialize(x0, lin.Weight, lin.Scales, lin.Biases)
	defer Free(x0)

	restoreNative := SetRuntimeGate(GateNativeLinearMatVec, true)
	defer restoreNative()
	restoreQ6 := SetRuntimeGate(GateNativeQ6BitstreamMatVec, true)
	defer restoreQ6()

	step := func(x *Array) *Array {
		if useMatVec {
			out, ok, err := QuantizedDenseMatVec(x, lin)
			if !ok || err != nil {
				b.Fatalf("matvec q%d dim%d ok=%v err=%v", bits, dim, ok, err)
			}
			return out
		}
		return quantizedMatmulMode(x, lin.Weight, lin.Scales, lin.Biases, true, lin.GroupSize, lin.Bits, lin.QuantizationMode)
	}

	// JIT-compile the kernel outside the timed loop (the 3x-vs-100x trap).
	warm := step(x0)
	Materialize(warm)
	Free(warm)

	weightBytes := int64(dim) * int64(quantizedDenseMatVecPackedIn(dim, bits)) * 4
	sidecarBytes := int64(2*dim*(dim/64)) * 4
	b.SetBytes(N * (weightBytes + sidecarBytes))
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		outs := make([]*Array, 0, N)
		x := x0
		for range N {
			y := step(x)
			outs = append(outs, y)
			x = y
		}
		if err := Eval(outs...); err != nil {
			b.Fatalf("Eval: %v", err)
		}
		Free(outs...)
	}
}

func BenchmarkQuantDecodeOrdering(b *testing.B) {
	requireMetalRuntime(b)
	for _, dim := range []int{2048, 6144} {
		for _, bits := range []int{4, 6, 8} {
			for _, path := range []struct {
				name   string
				matvec bool
			}{
				{name: "MatVec", matvec: true},
				{name: "Gemm", matvec: false},
			} {
				b.Run(fmt.Sprintf("dim%d/q%d/%s", dim, bits, path.name), func(b *testing.B) {
					benchmarkQuantDecodeOrdering(b, bits, dim, path.matvec)
				})
			}
		}
	}
}
