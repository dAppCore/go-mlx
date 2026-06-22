// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

// TestMatMulBF16NT pins MatMulBF16NT (fused steel GEMM, weight streamed once) BYTE-IDENTICAL to
// MatRowsBF16 (per-row gemv) across MTP projection shapes — the byte-identity that lets the batched
// verify use the GEMM for speed without changing output.
func TestMatMulBF16NT(t *testing.T) {
	requireNativeRuntime(t)
	for _, c := range []struct{ M, inDim, outDim int }{
		{1, 512, 256},
		{4, 512, 256},
		{4, 2048, 2048},
		{8, 1152, 256},
		{5, 640, 384}, // unaligned M/N/K
	} {
		w := toBF16Bytes(syntheticFloat32(c.outDim*c.inDim, 3))
		in := toBF16Bytes(syntheticFloat32(c.M*c.inDim, 5))
		gemv, err := MatRowsBF16(w, in, c.M, c.outDim, c.inDim)
		if err != nil {
			t.Fatalf("MatRowsBF16 M%d: %v", c.M, err)
		}
		gemm, err := MatMulBF16NT(in, w, c.M, c.inDim, c.outDim)
		if err != nil {
			t.Fatalf("MatMulBF16NT M%d: %v", c.M, err)
		}
		eqBytes(t, "MatMulBF16NT vs MatRowsBF16", gemm, gemv)
	}
}

// BenchmarkMatMulBF16NTvsRows measures the projection speedup: K=8 rows through the fused GEMM (weight
// streamed once) vs the per-row gemv (weight re-read 8×), on a large projection. Same bytes (the test
// above), this isolates the weight-bandwidth win the MTP batched verify gets from the GEMM.
func BenchmarkMatMulBF16NTvsRows(b *testing.B) {
	requireNativeRuntime(b)
	const M, inDim, outDim = 8, 2048, 2048
	w := toBF16Bytes(syntheticFloat32(outDim*inDim, 3))
	in := toBF16Bytes(syntheticFloat32(M*inDim, 5))
	b.Run("gemv-rows", func(b *testing.B) {
		for n := 0; n < b.N; n++ {
			if _, err := MatRowsBF16(w, in, M, outDim, inDim); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("steel-gemm", func(b *testing.B) {
		for n := 0; n < b.N; n++ {
			if _, err := MatMulBF16NT(in, w, M, inDim, outDim); err != nil {
				b.Fatal(err)
			}
		}
	})
}
