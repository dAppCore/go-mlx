// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"testing"
	"unsafe"
)

func matMulBF16NTFixture(M, K, N int) ([]byte, []byte) {
	w := toBF16Bytes(syntheticFloat32(N*K, N+3))
	in := toBF16Bytes(syntheticFloat32(M*K, K+5))
	return in, w
}

func TestMatMulBF16NTAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const M, K, N = 4, 256, 128
	in, w := matMulBF16NTFixture(M, K, N)
	if _, err := MatMulBF16NT(in, w, M, K, N); err != nil {
		t.Fatalf("MatMulBF16NT warmup: %v", err)
	}

	allocs := testing.AllocsPerRun(5, func() {
		if _, err := MatMulBF16NT(in, w, M, K, N); err != nil {
			t.Fatalf("MatMulBF16NT: %v", err)
		}
	})
	if allocs > 10 {
		t.Fatalf("MatMulBF16NT allocations = %.0f, want <= 10", allocs)
	}
}

func TestMatMulBF16NTIntoUsesCallerBacking(t *testing.T) {
	requireNativeRuntime(t)

	const M, K, N = 4, 256, 128
	in, w := matMulBF16NTFixture(M, K, N)
	out := make([]byte, M*N*bf16Size)
	for i := range out {
		out[i] = 0xA5
	}

	got, err := MatMulBF16NTInto(out, in, w, M, K, N)
	if err != nil {
		t.Fatalf("MatMulBF16NTInto: %v", err)
	}
	if len(got) != len(out) {
		t.Fatalf("MatMulBF16NTInto len = %d, want %d", len(got), len(out))
	}
	if unsafe.Pointer(&got[0]) != unsafe.Pointer(&out[0]) {
		t.Fatal("MatMulBF16NTInto did not return caller-owned output backing")
	}
	want, err := MatMulBF16NT(in, w, M, K, N)
	if err != nil {
		t.Fatalf("MatMulBF16NT reference: %v", err)
	}
	if !bytes.Equal(got, want) {
		t.Fatal("MatMulBF16NTInto output differs from allocating wrapper")
	}
}

// TestMatMulBF16NT pins MatMulBF16NT (fused steel GEMM, weight streamed once) byte-identical to the
// old looped MatVecBF16 row reference across MTP projection shapes.
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
		gemv := matRowsBF16LoopedMatVecReference(t, w, in, c.M, c.outDim, c.inDim)
		gemm, err := MatMulBF16NT(in, w, c.M, c.inDim, c.outDim)
		if err != nil {
			t.Fatalf("MatMulBF16NT M%d: %v", c.M, err)
		}
		eqBytes(t, "MatMulBF16NT vs looped MatVecBF16", gemm, gemv)
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
			_ = matRowsBF16LoopedMatVecReference(b, w, in, M, outDim, inDim)
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

func BenchmarkMatMulBF16NT_4x128x256(b *testing.B) {
	requireNativeRuntime(b)

	const M, K, N = 4, 256, 128
	in, w := matMulBF16NTFixture(M, K, N)
	b.SetBytes(int64(len(in) + len(w)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := MatMulBF16NT(in, w, M, K, N); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMatMulBF16NTInto_4x128x256(b *testing.B) {
	requireNativeRuntime(b)

	const M, K, N = 4, 256, 128
	in, w := matMulBF16NTFixture(M, K, N)
	out := make([]byte, M*N*bf16Size)
	b.SetBytes(int64(len(in) + len(w)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := MatMulBF16NTInto(out, in, w, M, K, N); err != nil {
			b.Fatal(err)
		}
	}
}
