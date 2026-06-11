// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

// Quantized op bench coverage map (W7-E, Wave 7).
//
// IDEAS.md flags MoE 26B-A4B as dispatching 128 tiny kernels in the
// naive path; the fix is `mlx_gather` + block-sparse matmul. Bench
// the underlying primitives:
//
//   - QuantizedMatmul (Q4 group-64, Q8 group-64) — the foundation of
//     all routed-expert paths.
//   - GatherMM — the fused gather + matmul that replaces the per-
//     expert kernel sprawl.
//   - Dequantize — when quantised weights need to round-trip to FP for
//     interop (LoRA training, output projection check).
//
// Q4/Q8 packing: Q4 packs 8 values per int32 (group_size=64 means each
// group has 64 elements + 1 scale + 1 bias). Q8 packs 4 per int32.

import "testing"

// --- QuantizedMatmul: hidden × packed_weight ---

// Q4 / group_size=64: matmul [1, 2048] × [2048, 32000] (output proj).
// Weight packed as [32000, 2048/8 = 256] int32. scales/biases shape
// is [32000, 2048/64 = 32].
func BenchmarkQuantizedMatmul_Q4_G64_OutputProj_H2048_V32k(b *testing.B) {
	const H, V, GS, Bits = 2048, 32000, 64, 4
	const packFactor = 32 / Bits
	x := RandomUniform(-1, 1, []int32{1, H}, DTypeFloat32)
	w := RandomUniform(-2, 2, []int32{V, H / packFactor}, DTypeUint32)
	scales := RandomUniform(0.01, 0.1, []int32{V, H / GS}, DTypeFloat32)
	biases := RandomUniform(-0.5, 0.5, []int32{V, H / GS}, DTypeFloat32)
	defer Free(x, w, scales, biases)
	Materialize(x, w, scales, biases)

	b.SetBytes(int64(H * 4))
	b.ReportAllocs()
	for b.Loop() {
		y := QuantizedMatmul(x, w, scales, biases, true, GS, Bits)
		Materialize(y)
		Free(y)
	}
}

// Q8 / group_size=64: same shape.
func BenchmarkQuantizedMatmul_Q8_G64_OutputProj_H2048_V32k(b *testing.B) {
	const H, V, GS, Bits = 2048, 32000, 64, 8
	const packFactor = 32 / Bits
	x := RandomUniform(-1, 1, []int32{1, H}, DTypeFloat32)
	w := RandomUniform(-2, 2, []int32{V, H / packFactor}, DTypeUint32)
	scales := RandomUniform(0.01, 0.1, []int32{V, H / GS}, DTypeFloat32)
	biases := RandomUniform(-0.5, 0.5, []int32{V, H / GS}, DTypeFloat32)
	defer Free(x, w, scales, biases)
	Materialize(x, w, scales, biases)

	b.SetBytes(int64(H * 4))
	b.ReportAllocs()
	for b.Loop() {
		y := QuantizedMatmul(x, w, scales, biases, true, GS, Bits)
		Materialize(y)
		Free(y)
	}
}

// Q4 / group_size=64, mid-size projection (attention path).
func BenchmarkQuantizedMatmul_Q4_G64_AttnProj_H2048(b *testing.B) {
	const H, GS, Bits = 2048, 64, 4
	const packFactor = 32 / Bits
	x := RandomUniform(-1, 1, []int32{1, H}, DTypeFloat32)
	w := RandomUniform(-2, 2, []int32{H, H / packFactor}, DTypeUint32)
	scales := RandomUniform(0.01, 0.1, []int32{H, H / GS}, DTypeFloat32)
	biases := RandomUniform(-0.5, 0.5, []int32{H, H / GS}, DTypeFloat32)
	defer Free(x, w, scales, biases)
	Materialize(x, w, scales, biases)

	b.SetBytes(int64(H * 4))
	b.ReportAllocs()
	for b.Loop() {
		y := QuantizedMatmul(x, w, scales, biases, true, GS, Bits)
		Materialize(y)
		Free(y)
	}
}

// Q4 / group_size=128 — alternate group size.
func BenchmarkQuantizedMatmul_Q4_G128_AttnProj_H2048(b *testing.B) {
	const H, GS, Bits = 2048, 128, 4
	const packFactor = 32 / Bits
	x := RandomUniform(-1, 1, []int32{1, H}, DTypeFloat32)
	w := RandomUniform(-2, 2, []int32{H, H / packFactor}, DTypeUint32)
	scales := RandomUniform(0.01, 0.1, []int32{H, H / GS}, DTypeFloat32)
	biases := RandomUniform(-0.5, 0.5, []int32{H, H / GS}, DTypeFloat32)
	defer Free(x, w, scales, biases)
	Materialize(x, w, scales, biases)

	b.SetBytes(int64(H * 4))
	b.ReportAllocs()
	for b.Loop() {
		y := QuantizedMatmul(x, w, scales, biases, true, GS, Bits)
		Materialize(y)
		Free(y)
	}
}

// --- Dequantize (Q4 → FP32 weight reconstruction) ---

func BenchmarkDequantize_Q4_G64_H2048(b *testing.B) {
	const H, GS, Bits = 2048, 64, 4
	const packFactor = 32 / Bits
	w := RandomUniform(-2, 2, []int32{H, H / packFactor}, DTypeUint32)
	scales := RandomUniform(0.01, 0.1, []int32{H, H / GS}, DTypeFloat32)
	biases := RandomUniform(-0.5, 0.5, []int32{H, H / GS}, DTypeFloat32)
	defer Free(w, scales, biases)
	Materialize(w, scales, biases)

	b.SetBytes(int64(H * H * 4))
	b.ReportAllocs()
	for b.Loop() {
		y := Dequantize(w, scales, biases, GS, Bits)
		Materialize(y)
		Free(y)
	}
}

func BenchmarkDequantize_Q8_G64_H2048(b *testing.B) {
	const H, GS, Bits = 2048, 64, 8
	const packFactor = 32 / Bits
	w := RandomUniform(-2, 2, []int32{H, H / packFactor}, DTypeUint32)
	scales := RandomUniform(0.01, 0.1, []int32{H, H / GS}, DTypeFloat32)
	biases := RandomUniform(-0.5, 0.5, []int32{H, H / GS}, DTypeFloat32)
	defer Free(w, scales, biases)
	Materialize(w, scales, biases)

	b.SetBytes(int64(H * H * 4))
	b.ReportAllocs()
	for b.Loop() {
		y := Dequantize(w, scales, biases, GS, Bits)
		Materialize(y)
		Free(y)
	}
}

// --- GatherMM — fused gather + matmul (full FP path) ---

// The gather+matmul fused op that replaces per-expert dispatching.
// Inputs: [1, K, H] × [N, M, H] with indices [K] picking expert rows.
// Synthetic K=2 (top-2), M=hidden, N=8 experts.
func BenchmarkGatherMM_K2_Experts8_H2048(b *testing.B) {
	const H, N, K = 2048, 8, 2
	// Per Gemma 4 MoE expert layout: weights shape [N_experts, hidden, intermediate].
	a := RandomUniform(-1, 1, []int32{1, K, H}, DTypeFloat32)
	w := RandomUniform(-0.05, 0.05, []int32{N, H, H}, DTypeFloat32)
	// rhsIndices selects expert rows: shape [1, K].
	rhsIndices := FromValues([]int32{2, 5}, 1, K)
	defer Free(a, w, rhsIndices)
	Materialize(a, w, rhsIndices)

	b.ReportAllocs()
	for b.Loop() {
		y := GatherMM(a, w, nil, rhsIndices, false)
		Materialize(y)
		Free(y)
	}
}

// N-scaling probe (K=2 fixed, N varies): if GatherMM time scales with the
// TOTAL expert count N rather than the ACTIVE count K, the gather reads all
// experts' weights and discards N-K of them — wasted decode bandwidth. A
// flat-in-N result means it already reads only the K selected rows. This
// decides whether an M=1 "direct K-expert matvec" rewrite is worth it for
// the 26B/31B MoE decode path. Companion to BenchmarkGatherMM_K2_Experts8.
func BenchmarkGatherMM_K2_Experts32_H2048(b *testing.B) {
	const H, N, K = 2048, 32, 2
	a := RandomUniform(-1, 1, []int32{1, K, H}, DTypeFloat32)
	w := RandomUniform(-0.05, 0.05, []int32{N, H, H}, DTypeFloat32)
	rhsIndices := FromValues([]int32{2, 5}, 1, K)
	defer Free(a, w, rhsIndices)
	Materialize(a, w, rhsIndices)

	b.ReportAllocs()
	for b.Loop() {
		y := GatherMM(a, w, nil, rhsIndices, false)
		Materialize(y)
		Free(y)
	}
}

func BenchmarkGatherMM_K2_Experts128_H2048(b *testing.B) {
	const H, N, K = 2048, 128, 2
	a := RandomUniform(-1, 1, []int32{1, K, H}, DTypeFloat32)
	w := RandomUniform(-0.05, 0.05, []int32{N, H, H}, DTypeFloat32)
	rhsIndices := FromValues([]int32{2, 5}, 1, K)
	defer Free(a, w, rhsIndices)
	Materialize(a, w, rhsIndices)

	b.ReportAllocs()
	for b.Loop() {
		y := GatherMM(a, w, nil, rhsIndices, false)
		Materialize(y)
		Free(y)
	}
}

// Batched (N-chained-1-Eval) gather probe — the per-op variants above are
// per-Eval-floor-bound (~200us, high variance: N=8 read 480us@50x but
// 816us@10x), so they cannot resolve the kernel's N-scaling. Chaining 16
// gathers into ONE Eval amortises the sync floor to ~1/16 and exposes the
// real per-gather kernel time. Read ns/op as ~16x the per-gather cost; the
// shape that matters is the RATIO across N: rising with N ⇒ all-expert read
// (K-direct rewrite wins for 26B/31B decode), flat ⇒ already K-selective.
func benchGatherMMBatchedNScaling(b *testing.B, nExperts int) {
	requireMetalRuntime(b)
	const H, K, chain = 2048, 2, 16
	a := RandomUniform(-1, 1, []int32{1, K, H}, DTypeFloat32)
	w := RandomUniform(-0.05, 0.05, []int32{int32(nExperts), H, H}, DTypeFloat32)
	rhsIndices := FromValues([]int32{2, 5}, 1, K)
	defer Free(a, w, rhsIndices)
	Materialize(a, w, rhsIndices)
	// Warm the kernel cache so the first Eval's JIT does not skew iteration 0.
	warm := GatherMM(a, w, nil, rhsIndices, false)
	Materialize(warm)
	Free(warm)

	b.ReportAllocs()
	for b.Loop() {
		outs := make([]*Array, 0, chain)
		for range chain {
			outs = append(outs, GatherMM(a, w, nil, rhsIndices, false))
		}
		if err := Eval(outs...); err != nil {
			b.Fatalf("Eval: %v", err)
		}
		Free(outs...)
	}
}

func BenchmarkGatherMM_Batched_K2_Experts8_H2048(b *testing.B)  { benchGatherMMBatchedNScaling(b, 8) }
func BenchmarkGatherMM_Batched_K2_Experts32_H2048(b *testing.B) { benchGatherMMBatchedNScaling(b, 32) }
func BenchmarkGatherMM_Batched_K2_Experts128_H2048(b *testing.B) {
	benchGatherMMBatchedNScaling(b, 128)
}

// --- AsType (FP32 ↔ FP16/BF16 conversions) ---

// Native dispatch may convert tensors between dtypes for the fused
// kernel input requirements. Bench the cost of those conversions at
// realistic shapes.
func BenchmarkQuant_AsType_FP32toFP16_Hidden2048(b *testing.B) {
	x := RandomUniform(-1, 1, []int32{1, 2048}, DTypeFloat32)
	defer Free(x)
	Materialize(x)
	b.SetBytes(int64(2048 * 4))
	b.ReportAllocs()
	for b.Loop() {
		y := AsType(x, DTypeFloat16)
		Materialize(y)
		Free(y)
	}
}

func BenchmarkQuant_AsType_FP16toFP32_Hidden2048(b *testing.B) {
	x := RandomUniform(-1, 1, []int32{1, 2048}, DTypeFloat16)
	defer Free(x)
	Materialize(x)
	b.SetBytes(int64(2048 * 2))
	b.ReportAllocs()
	for b.Loop() {
		y := AsType(x, DTypeFloat32)
		Materialize(y)
		Free(y)
	}
}

func BenchmarkQuant_AsType_FP32toBF16_Hidden2048(b *testing.B) {
	x := RandomUniform(-1, 1, []int32{1, 2048}, DTypeFloat32)
	defer Free(x)
	Materialize(x)
	b.SetBytes(int64(2048 * 4))
	b.ReportAllocs()
	for b.Loop() {
		y := AsType(x, DTypeBFloat16)
		Materialize(y)
		Free(y)
	}
}
