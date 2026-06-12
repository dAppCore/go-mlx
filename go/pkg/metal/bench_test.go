// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"math"
	"runtime"
	"testing"
)

// --- Helpers ---

// randomMatrix creates a random float32 matrix of the given shape.
func randomMatrix(rows, cols int32) *Array {
	return RandomUniform(0, 1, []int32{rows, cols}, DTypeFloat32)
}

// randomVector creates a random float32 vector.
func randomVector(n int32) *Array {
	return RandomUniform(0, 1, []int32{n}, DTypeFloat32)
}

// random4D creates a random float32 4D tensor [B, H, L, D].
func random4D(b, h, l, d int32) *Array {
	return RandomUniform(0, 1, []int32{b, h, l, d}, DTypeFloat32)
}

// --- MatMul benchmarks (various sizes) ---

func BenchmarkMatMul_128x128(b *testing.B) {
	a := randomMatrix(128, 128)
	w := randomMatrix(128, 128)
	Materialize(a, w)
	for b.Loop() {
		c := Matmul(a, w)
		Materialize(c)
	}
}

func BenchmarkMatMul_512x512(b *testing.B) {
	a := randomMatrix(512, 512)
	w := randomMatrix(512, 512)
	Materialize(a, w)
	for b.Loop() {
		c := Matmul(a, w)
		Materialize(c)
	}
}

func BenchmarkMatMul_1024x1024(b *testing.B) {
	a := randomMatrix(1024, 1024)
	w := randomMatrix(1024, 1024)
	Materialize(a, w)
	for b.Loop() {
		c := Matmul(a, w)
		Materialize(c)
	}
}

func BenchmarkMatMul_2048x2048(b *testing.B) {
	a := randomMatrix(2048, 2048)
	w := randomMatrix(2048, 2048)
	Materialize(a, w)
	for b.Loop() {
		c := Matmul(a, w)
		Materialize(c)
	}
}

func BenchmarkMatMul_4096x4096(b *testing.B) {
	a := randomMatrix(4096, 4096)
	w := randomMatrix(4096, 4096)
	Materialize(a, w)
	for b.Loop() {
		c := Matmul(a, w)
		Materialize(c)
	}
}

// Token-shaped matmul: [1, D] x [D, V] — single-token forward through output projection.
func BenchmarkMatMul_1x2048_x_2048x32000(b *testing.B) {
	x := randomMatrix(1, 2048)
	w := randomMatrix(2048, 32000)
	Materialize(x, w)
	for b.Loop() {
		c := Matmul(x, w)
		Materialize(c)
	}
}

// --- Softmax benchmarks ---

func BenchmarkSoftmax_1x1024(b *testing.B) {
	x := randomMatrix(1, 1024)
	Materialize(x)
	for b.Loop() {
		y := Softmax(x)
		Materialize(y)
	}
}

func BenchmarkSoftmax_32x32000(b *testing.B) {
	x := randomMatrix(32, 32000)
	Materialize(x)
	for b.Loop() {
		y := Softmax(x)
		Materialize(y)
	}
}

func BenchmarkSoftmax_1x128000(b *testing.B) {
	x := randomMatrix(1, 128000)
	Materialize(x)
	for b.Loop() {
		y := Softmax(x)
		Materialize(y)
	}
}

// --- Element-wise arithmetic ---

func BenchmarkAdd_1M(b *testing.B) {
	a := RandomUniform(0, 1, []int32{1000000}, DTypeFloat32)
	c := RandomUniform(0, 1, []int32{1000000}, DTypeFloat32)
	Materialize(a, c)
	for b.Loop() {
		y := Add(a, c)
		Materialize(y)
	}
}

func BenchmarkMul_1M(b *testing.B) {
	a := RandomUniform(0, 1, []int32{1000000}, DTypeFloat32)
	c := RandomUniform(0, 1, []int32{1000000}, DTypeFloat32)
	Materialize(a, c)
	for b.Loop() {
		y := Mul(a, c)
		Materialize(y)
	}
}

func BenchmarkSiLU_1M(b *testing.B) {
	a := RandomUniform(-3, 3, []int32{1000000}, DTypeFloat32)
	Materialize(a)
	for b.Loop() {
		y := SiLU(a)
		Materialize(y)
	}
}

// --- Fused Metal kernels ---

func BenchmarkRMSNorm_1x2048(b *testing.B) {
	x := randomMatrix(1, 2048)
	w := randomVector(2048)
	Materialize(x, w)
	for b.Loop() {
		y := RMSNorm(x, w, 1e-5)
		Materialize(y)
	}
}

func BenchmarkRMSNorm_32x2048(b *testing.B) {
	x := randomMatrix(32, 2048)
	w := randomVector(2048)
	Materialize(x, w)
	for b.Loop() {
		y := RMSNorm(x, w, 1e-5)
		Materialize(y)
	}
}

func BenchmarkLayerNorm_32x2048(b *testing.B) {
	x := randomMatrix(32, 2048)
	w := randomVector(2048)
	bias := randomVector(2048)
	Materialize(x, w, bias)
	for b.Loop() {
		y := LayerNorm(x, w, bias, 1e-5)
		Materialize(y)
	}
}

func BenchmarkRoPE_1x1x32x128(b *testing.B) {
	// Single head, 32 positions, 128 dims — typical decode step shape.
	x := random4D(1, 1, 32, 128)
	Materialize(x)
	for b.Loop() {
		y := RoPE(x, 128, false, 10000.0, 1.0, 0)
		Materialize(y)
	}
}

func BenchmarkRoPE_1x32x512x128(b *testing.B) {
	// 32 heads, 512 positions — typical prefill shape.
	x := random4D(1, 32, 512, 128)
	Materialize(x)
	for b.Loop() {
		y := RoPE(x, 128, false, 10000.0, 1.0, 0)
		Materialize(y)
	}
}

// --- Scaled Dot-Product Attention ---

func BenchmarkSDPA_1head_seq32(b *testing.B) {
	scale := float32(1.0 / math.Sqrt(128.0))
	q := random4D(1, 1, 32, 128)
	k := random4D(1, 1, 32, 128)
	v := random4D(1, 1, 32, 128)
	Materialize(q, k, v)
	for b.Loop() {
		y := ScaledDotProductAttention(q, k, v, scale, true)
		Materialize(y)
	}
}

func BenchmarkSDPA_32head_seq128(b *testing.B) {
	scale := float32(1.0 / math.Sqrt(128.0))
	q := random4D(1, 32, 128, 128)
	k := random4D(1, 32, 128, 128)
	v := random4D(1, 32, 128, 128)
	Materialize(q, k, v)
	for b.Loop() {
		y := ScaledDotProductAttention(q, k, v, scale, true)
		Materialize(y)
	}
}

func BenchmarkSDPA_32head_seq512(b *testing.B) {
	scale := float32(1.0 / math.Sqrt(128.0))
	q := random4D(1, 32, 512, 128)
	k := random4D(1, 32, 512, 128)
	v := random4D(1, 32, 512, 128)
	Materialize(q, k, v)
	for b.Loop() {
		y := ScaledDotProductAttention(q, k, v, scale, true)
		Materialize(y)
	}
}

// --- Neural network layers ---

func BenchmarkLinear_1x2048_to_2048(b *testing.B) {
	w := randomMatrix(2048, 2048)
	Materialize(w)
	layer := NewLinear(w, nil)
	x := randomMatrix(1, 2048)
	Materialize(x)
	for b.Loop() {
		y := layer.Forward(x)
		Materialize(y)
	}
}

func BenchmarkLinear_32x2048_to_8192(b *testing.B) {
	w := randomMatrix(8192, 2048)
	Materialize(w)
	layer := NewLinear(w, nil)
	x := randomMatrix(32, 2048)
	Materialize(x)
	for b.Loop() {
		y := layer.Forward(x)
		Materialize(y)
	}
}

// N-batched FFN decode matmul — the dominant per-token cost is the feed-forward
// matmuls (they read the bulk of the weights). The single-call Linear benches
// above are sync-floored (~250us) and cannot see the real matmul cost. This
// chains N up(2048->8192)+down(8192->2048) pairs (genuine serial dependency, no
// MLX dedup) and evals ONCE: ns/op / N = real per-FFN-pair GPU time. Each pair
// reads 64MB+64MB fp32 = 128MB, so its bandwidth floor on an M3 (~819 GB/s) is
// ~156us. per-pair AT ~156us = matmul is bandwidth-bound (already optimal, the
// model is at its memory wall); per-pair WELL ABOVE = real overhead to cut.
func BenchmarkLinear_FFNDecode_Batched64(b *testing.B) {
	const N = 64
	up := NewLinear(randomMatrix(8192, 2048), nil)
	down := NewLinear(randomMatrix(2048, 8192), nil)
	Materialize(up.Weight, down.Weight)
	x0 := randomMatrix(1, 2048)
	Materialize(x0)
	b.ReportAllocs()
	for b.Loop() {
		outs := make([]*Array, 0, N*2)
		x := x0
		for range N {
			h := up.Forward(x)
			outs = append(outs, h)
			x = down.Forward(h)
			outs = append(outs, x)
		}
		if err := Eval(outs...); err != nil {
			b.Fatalf("Eval: %v", err)
		}
		Free(outs...)
	}
}

func benchMakeQ4Linear(outDim, inDim int) *Linear {
	packedWidth := inDim / 8
	groups := inDim / 64
	weightWords := make([]uint32, outDim*packedWidth)
	for i := range weightWords {
		weightWords[i] = uint32(i*1664525 + 1013904223)
	}
	scales := make([]float32, outDim*groups)
	biases := make([]float32, outDim*groups)
	for i := range scales {
		scales[i] = 0.005 * float32((i%17)+1)
		biases[i] = -0.03 + 0.002*float32(i%31)
	}
	return NewQuantizedLinear(
		FromValues(weightWords, outDim, packedWidth),
		FromValues(scales, outDim, groups),
		FromValues(biases, outDim, groups),
		nil, 64, 4,
	)
}

// The REAL serve-path FFN: q4 weights, the same up/down chain as the fp32 bench
// above. q4 reads 4x fewer bytes (~18MB/pair, BW floor ~22us), but adds dequant.
// per-pair / 22us tells whether the q4 decode matmul is bandwidth-bound (optimal)
// or dominated by dequant + small-read + dispatch overhead — i.e. whether e4b's
// ~25%-of-peak aggregate is a real optimisation target or the q4 memory wall.
func BenchmarkLinear_FFNDecodeQ4_Batched64(b *testing.B) {
	const N = 64
	up := benchMakeQ4Linear(8192, 2048)
	down := benchMakeQ4Linear(2048, 8192)
	Materialize(up.Weight, up.Scales, up.Biases, down.Weight, down.Scales, down.Biases)
	x0 := randomMatrix(1, 2048)
	Materialize(x0)
	b.ReportAllocs()
	for b.Loop() {
		outs := make([]*Array, 0, N*2)
		x := x0
		for range N {
			h := up.Forward(x)
			outs = append(outs, h)
			x = down.Forward(h)
			outs = append(outs, x)
		}
		if err := Eval(outs...); err != nil {
			b.Fatalf("Eval: %v", err)
		}
		Free(outs...)
	}
}

// bf16 FFN — the heaviest goal quant ("50 tok/s q8/bf16"). bf16 reads 4x the q4
// bytes (~64MB/pair, BW floor ~78us). Measured alongside q4 (18MB) and fp32
// (128MB) it confirms decode tok/s scales LINEARLY with bytes-per-weight — so
// every (model x quant) projects from a measured anchor, no guessing. If bf16
// per-pair tracks its 78us floor, the matmul is BW-bound at every precision and
// the quant tiers are pure byte-count arithmetic off the q4 numbers.
func BenchmarkLinear_FFNDecodeBF16_Batched64(b *testing.B) {
	const N = 64
	up := NewLinear(AsType(randomMatrix(8192, 2048), DTypeBFloat16), nil)
	down := NewLinear(AsType(randomMatrix(2048, 8192), DTypeBFloat16), nil)
	Materialize(up.Weight, down.Weight)
	x0 := AsType(randomMatrix(1, 2048), DTypeBFloat16)
	Materialize(x0)
	b.ReportAllocs()
	for b.Loop() {
		outs := make([]*Array, 0, N*2)
		x := x0
		for range N {
			h := up.Forward(x)
			outs = append(outs, h)
			x = down.Forward(h)
			outs = append(outs, x)
		}
		if err := Eval(outs...); err != nil {
			b.Fatalf("Eval: %v", err)
		}
		Free(outs...)
	}
}

// Head-to-head: the two q4 decode kernels Linear.Forward can pick (nn.go:120) —
// QuantizedDenseMatVec (the single-token matvec, gated by GateNativeLinearMatVec)
// vs quantizedMatmulMode (the general gemm). Same affine q4 [2048x2048] weight,
// chained single-token. If matvec is meaningfully faster AND serve is on the gemm
// path (gate off / not applied), enabling the matvec gate is a real decode win —
// pure Go routing, no kernel surgery. If equal, the kernel is at its floor.
func benchmarkQ4DecodePath(b *testing.B, useMatVec bool) {
	const N, dim = 64, 2048
	lin := benchMakeQ4Linear(dim, dim)
	Materialize(lin.Weight, lin.Scales, lin.Biases)
	x0 := RandomUniform(-1, 1, []int32{1, 1, dim}, DTypeFloat32)
	Materialize(x0)
	defer Free(x0)
	b.ReportAllocs()
	for b.Loop() {
		outs := make([]*Array, 0, N)
		x := x0
		for range N {
			var y *Array
			if useMatVec {
				out, ok, err := QuantizedDenseMatVec(x, lin)
				if !ok || err != nil {
					b.Fatalf("matvec ok=%v err=%v", ok, err)
				}
				y = out
			} else {
				y = quantizedMatmulMode(x, lin.Weight, lin.Scales, lin.Biases, true, lin.GroupSize, lin.Bits, lin.QuantizationMode)
			}
			outs = append(outs, y)
			x = y
		}
		if err := Eval(outs...); err != nil {
			b.Fatalf("Eval: %v", err)
		}
		Free(outs...)
	}
}

func BenchmarkLinear_Q4Decode_MatVec_Batched64(b *testing.B) { benchmarkQ4DecodePath(b, true) }
func BenchmarkLinear_Q4Decode_Gemm_Batched64(b *testing.B)   { benchmarkQ4DecodePath(b, false) }

func benchMakeQ8Linear(outDim, inDim int) *Linear {
	packedWidth := inDim / 4 // q8: 4 values per uint32
	groups := inDim / 64
	weightWords := make([]uint32, outDim*packedWidth)
	for i := range weightWords {
		weightWords[i] = uint32(i*1664525 + 1013904223)
	}
	scales := make([]float32, outDim*groups)
	biases := make([]float32, outDim*groups)
	for i := range scales {
		scales[i] = 0.005 * float32((i%17)+1)
		biases[i] = -0.03 + 0.002*float32(i%31)
	}
	return NewQuantizedLinear(
		FromValues(weightWords, outDim, packedWidth),
		FromValues(scales, outDim, groups),
		FromValues(biases, outDim, groups),
		nil, 64, 8,
	)
}

// q8 head-to-head: same question as q4. q8 is byte-aligned (no bitstream packing),
// so MLX gemm handles it natively — if it wins, the nn.go exclusion should cover
// q8 too (bits != 4 && bits != 8), extending the win to the q8 goal quant.
func benchmarkQ8DecodePath(b *testing.B, useMatVec bool) {
	const N, dim = 64, 2048
	lin := benchMakeQ8Linear(dim, dim)
	Materialize(lin.Weight, lin.Scales, lin.Biases)
	x0 := RandomUniform(-1, 1, []int32{1, 1, dim}, DTypeFloat32)
	Materialize(x0)
	defer Free(x0)
	b.ReportAllocs()
	for b.Loop() {
		outs := make([]*Array, 0, N)
		x := x0
		for range N {
			var y *Array
			if useMatVec {
				out, ok, err := QuantizedDenseMatVec(x, lin)
				if !ok || err != nil {
					b.Fatalf("matvec ok=%v err=%v", ok, err)
				}
				y = out
			} else {
				y = quantizedMatmulMode(x, lin.Weight, lin.Scales, lin.Biases, true, lin.GroupSize, lin.Bits, lin.QuantizationMode)
			}
			outs = append(outs, y)
			x = y
		}
		if err := Eval(outs...); err != nil {
			b.Fatalf("Eval: %v", err)
		}
		Free(outs...)
	}
}

func BenchmarkLinear_Q8Decode_MatVec_Batched64(b *testing.B) { benchmarkQ8DecodePath(b, true) }
func BenchmarkLinear_Q8Decode_Gemm_Batched64(b *testing.B)   { benchmarkQ8DecodePath(b, false) }

// End-to-end proof the q8 exclusion took: gate ON, q8 Forward must land on the q8
// gemm number (~13.7us), not the q8 matvec number (~17.5us).
// BenchmarkGemma4PLE_Decode measures the Gemma 4 MatFormer per-layer-input gate
// path that runs EVERY layer for e2b/e4b: gate proj (hidden->ple) + GeluGateMul +
// projection (ple->hidden) + RMSNorm + Add. Small matmuls (q4, now gemm) but ~5
// dispatches/layer. ns/op / N = per-layer PLE cost; x34 layers is its contribution.
func BenchmarkGemma4PLE_Decode_Batched32(b *testing.B) {
	const hidden, ple, N = 2048, 256, 32
	gate := benchMakeQ4Linear(ple, hidden) // hidden -> ple
	proj := benchMakeQ4Linear(hidden, ple) // ple -> hidden
	normW := RandomUniform(0.5, 1.5, []int32{hidden}, DTypeFloat32)
	pli := RandomUniform(-1, 1, []int32{1, 1, ple}, DTypeFloat32)
	Materialize(gate.Weight, gate.Scales, gate.Biases, proj.Weight, proj.Scales, proj.Biases, normW, pli)
	x0 := RandomUniform(-1, 1, []int32{1, 1, hidden}, DTypeFloat32)
	Materialize(x0)
	defer Free(x0, normW, pli)
	b.ReportAllocs()
	for b.Loop() {
		outs := make([]*Array, 0, N)
		x := x0
		for range N {
			g := gate.Forward(x)
			mult := GeluGateMul(g, pli)
			projected := proj.Forward(mult)
			pn := RMSNorm(projected, normW, 1e-6)
			out := Add(x, pn)
			Free(g, mult, projected, pn)
			outs = append(outs, out)
			x = out
		}
		if err := Eval(outs...); err != nil {
			b.Fatalf("Eval: %v", err)
		}
		Free(outs...)
	}
}

// BenchmarkGemma4RoPE_Decode measures a single RoPE application (mlx_fast_rope,
// one fused kernel) on a decode-shape Q/K. Applied to both Q and K every layer,
// so x2 x34 layers is the RoPE contribution to the dispatch budget.
func BenchmarkGemma4RoPE_Decode_Batched32(b *testing.B) {
	const heads, headDim, N = 8, 256, 32
	q0 := RandomUniform(-1, 1, []int32{1, heads, 1, headDim}, DTypeFloat32)
	Materialize(q0)
	defer Free(q0)
	b.ReportAllocs()
	for b.Loop() {
		outs := make([]*Array, 0, N)
		x := q0
		for range N {
			out := RoPE(x, headDim, false, 10000, 1.0, 0)
			outs = append(outs, out)
			x = out
		}
		if err := Eval(outs...); err != nil {
			b.Fatalf("Eval: %v", err)
		}
		Free(outs...)
	}
}

func BenchmarkLinear_Q8Forward_GateOn_Batched64(b *testing.B) {
	restore := (EngineFeatures{NativeLinearMatVec: true}).Apply()
	defer restore()
	const N, dim = 64, 2048
	lin := benchMakeQ8Linear(dim, dim)
	Materialize(lin.Weight, lin.Scales, lin.Biases)
	x0 := RandomUniform(-1, 1, []int32{1, 1, dim}, DTypeFloat32)
	Materialize(x0)
	defer Free(x0)
	b.ReportAllocs()
	for b.Loop() {
		outs := make([]*Array, 0, N)
		x := x0
		for range N {
			y := lin.Forward(x)
			outs = append(outs, y)
			x = y
		}
		if err := Eval(outs...); err != nil {
			b.Fatalf("Eval: %v", err)
		}
		Free(outs...)
	}
}

// End-to-end proof of the nn.go fix: with GateNativeLinearMatVec ON (as serve runs
// it), Linear.Forward on a q4 weight must now take the gemm path (bits!=4 exclusion),
// landing near the gemm number (~12us/call), NOT the matvec number (~17us). If this
// reads ~17us the fix didn't take.
func BenchmarkLinear_Q4Forward_GateOn_Batched64(b *testing.B) {
	restore := (EngineFeatures{NativeLinearMatVec: true}).Apply()
	defer restore()
	const N, dim = 64, 2048
	lin := benchMakeQ4Linear(dim, dim)
	Materialize(lin.Weight, lin.Scales, lin.Biases)
	x0 := RandomUniform(-1, 1, []int32{1, 1, dim}, DTypeFloat32)
	Materialize(x0)
	defer Free(x0)
	b.ReportAllocs()
	for b.Loop() {
		outs := make([]*Array, 0, N)
		x := x0
		for range N {
			y := lin.Forward(x)
			outs = append(outs, y)
			x = y
		}
		if err := Eval(outs...); err != nil {
			b.Fatalf("Eval: %v", err)
		}
		Free(outs...)
	}
}

func benchApplyQ4(b *testing.B, lin *Linear, x *Array, useMatVec bool) *Array {
	if useMatVec {
		out, ok, err := QuantizedDenseMatVec(x, lin)
		if !ok || err != nil {
			b.Fatalf("matvec ok=%v err=%v", ok, err)
		}
		return out
	}
	return quantizedMatmulMode(x, lin.Weight, lin.Scales, lin.Biases, true, lin.GroupSize, lin.Bits, lin.QuantizationMode)
}

// Same head-to-head on the REAL FFN shapes (2048->8192 up, 8192->2048 down) — the
// bulk of decode weight. matvec vs gemm, chained single-token pair.
func benchmarkQ4FFNPath(b *testing.B, useMatVec bool) {
	const N = 64
	up := benchMakeQ4Linear(8192, 2048)
	down := benchMakeQ4Linear(2048, 8192)
	Materialize(up.Weight, up.Scales, up.Biases, down.Weight, down.Scales, down.Biases)
	x0 := RandomUniform(-1, 1, []int32{1, 1, 2048}, DTypeFloat32)
	Materialize(x0)
	defer Free(x0)
	b.ReportAllocs()
	for b.Loop() {
		outs := make([]*Array, 0, N*2)
		x := x0
		for range N {
			h := benchApplyQ4(b, up, x, useMatVec)
			outs = append(outs, h)
			x = benchApplyQ4(b, down, h, useMatVec)
			outs = append(outs, x)
		}
		if err := Eval(outs...); err != nil {
			b.Fatalf("Eval: %v", err)
		}
		Free(outs...)
	}
}

func BenchmarkLinear_Q4FFN_MatVec_Batched64(b *testing.B) { benchmarkQ4FFNPath(b, true) }
func BenchmarkLinear_Q4FFN_Gemm_Batched64(b *testing.B)   { benchmarkQ4FFNPath(b, false) }

// NativeMLPMatVec decision: the fused MLP path (gate+up+GELU in one kernel, then
// down matvec — 2 dispatches) vs the gemm fallback (3 quantized_matmul + 1
// GeluGateMul — 4 dispatches). Fused has fewer dispatches but each matvec is the
// 35%-slower kernel; gemm has more dispatches but each is faster. Whichever wins
// here decides whether NativeMLPMatVec stays on. Same q4 MLP, chained single token.
func benchmarkQ4MLPPath(b *testing.B, fused bool) {
	const N = 32
	mlp := &MLP{
		GateProj: benchMakeQ4Linear(8192, 2048),
		UpProj:   benchMakeQ4Linear(8192, 2048),
		DownProj: benchMakeQ4Linear(2048, 8192),
	}
	Materialize(mlp.GateProj.Weight, mlp.GateProj.Scales, mlp.GateProj.Biases,
		mlp.UpProj.Weight, mlp.UpProj.Scales, mlp.UpProj.Biases,
		mlp.DownProj.Weight, mlp.DownProj.Scales, mlp.DownProj.Biases)
	x0 := RandomUniform(-1, 1, []int32{1, 1, 2048}, DTypeFloat32)
	Materialize(x0)
	defer Free(x0)
	gemmProj := func(x *Array, l *Linear) *Array {
		return quantizedMatmulMode(x, l.Weight, l.Scales, l.Biases, true, l.GroupSize, l.Bits, l.QuantizationMode)
	}
	b.ReportAllocs()
	for b.Loop() {
		outs := make([]*Array, 0, N)
		x := x0
		for range N {
			var out *Array
			if fused {
				activated, ok, err := quantizedDenseGELUSplitGateUpMatVec(x, mlp.GateProj, mlp.UpProj)
				if !ok || err != nil {
					b.Fatalf("fused gate/up ok=%v err=%v", ok, err)
				}
				o, ok2, err2 := QuantizedDenseMatVec(activated, mlp.DownProj)
				Free(activated)
				if !ok2 || err2 != nil {
					b.Fatalf("fused down ok=%v err=%v", ok2, err2)
				}
				out = o
			} else {
				gate := gemmProj(x, mlp.GateProj)
				up := gemmProj(x, mlp.UpProj)
				activated := GeluGateMul(gate, up)
				Free(gate, up)
				out = gemmProj(activated, mlp.DownProj)
				Free(activated)
			}
			outs = append(outs, out)
			x = out
		}
		if err := Eval(outs...); err != nil {
			b.Fatalf("Eval: %v", err)
		}
		Free(outs...)
	}
}

func BenchmarkLinear_Q4MLP_Fused_Batched32(b *testing.B) { benchmarkQ4MLPPath(b, true) }
func BenchmarkLinear_Q4MLP_Gemm_Batched32(b *testing.B)  { benchmarkQ4MLPPath(b, false) }

func BenchmarkEmbedding_32tokens_vocab32000_dim2048(b *testing.B) {
	w := randomMatrix(32000, 2048)
	Materialize(w)
	emb := &Embedding{Weight: w}
	indices := FromValues(make([]int32, 32), 32)
	// Fill with random valid indices
	for i := range 32 {
		indices = FromValues([]int32{int32(i % 32000)}, 1)
	}
	indices = RandomUniform(0, 31999, []int32{32}, DTypeFloat32)
	indices = AsType(indices, DTypeInt32)
	Materialize(indices)
	for b.Loop() {
		y := emb.Forward(indices)
		Materialize(y)
	}
}

// --- Reductions ---

func BenchmarkSum_1M(b *testing.B) {
	a := RandomUniform(0, 1, []int32{1000000}, DTypeFloat32)
	Materialize(a)
	for b.Loop() {
		y := Sum(a, 0, false)
		Materialize(y)
	}
}

func BenchmarkArgmax_1x32000(b *testing.B) {
	a := randomMatrix(1, 32000)
	Materialize(a)
	for b.Loop() {
		y := Argmax(a, -1, false)
		Materialize(y)
	}
}

// --- Sampling ---

func BenchmarkSampler_Greedy(b *testing.B) {
	logits := RandomUniform(-5, 5, []int32{1, 32000}, DTypeFloat32)
	Materialize(logits)
	s := newSampler(0, 0, 0, 0) // Greedy
	for b.Loop() {
		tok := s.Sample(logits)
		Materialize(tok)
	}
}

func BenchmarkSampler_TopK50_Temp1(b *testing.B) {
	logits := RandomUniform(-5, 5, []int32{1, 32000}, DTypeFloat32)
	Materialize(logits)
	s := newSampler(1.0, 0, 0, 50)
	for b.Loop() {
		tok := s.Sample(logits)
		Materialize(tok)
	}
}

func BenchmarkSampler_TopP09_Temp1(b *testing.B) {
	logits := RandomUniform(-5, 5, []int32{1, 32000}, DTypeFloat32)
	Materialize(logits)
	s := newSampler(1.0, 0.9, 0, 0)
	for b.Loop() {
		tok := s.Sample(logits)
		Materialize(tok)
	}
}

func BenchmarkSampler_Full_TopP09_MinP01_TopK50(b *testing.B) {
	logits := RandomUniform(-5, 5, []int32{1, 32000}, DTypeFloat32)
	Materialize(logits)
	s := newSampler(0.8, 0.9, 0.1, 50) // temp=0.8, topP=0.9, minP=0.1, topK=50
	for b.Loop() {
		tok := s.Sample(logits)
		Materialize(tok)
	}
}

func BenchmarkSampler_LegacyTopPThenTopK_Vocab262k(b *testing.B) {
	b.ReportAllocs()
	logits := RandomUniform(-5, 5, []int32{1, 262208}, DTypeFloat32)
	defer Free(logits)
	Materialize(logits)
	s := chain{steps: []Sampler{Temperature(1.0), TopP(0.95), TopKSampler(64)}}
	b.ResetTimer()
	for b.Loop() {
		tok := s.Sample(logits)
		if err := Eval(tok); err != nil {
			Free(tok)
			b.Fatalf("Eval(sample): %v", err)
		}
		Free(tok)
	}
}

func BenchmarkSampler_TopKThenTopP_Vocab262k(b *testing.B) {
	b.ReportAllocs()
	logits := RandomUniform(-5, 5, []int32{1, 262208}, DTypeFloat32)
	defer Free(logits)
	Materialize(logits)
	s := newSampler(1.0, 0.95, 0, 64)
	b.ResetTimer()
	for b.Loop() {
		tok := s.Sample(logits)
		if err := Eval(tok); err != nil {
			Free(tok)
			b.Fatalf("Eval(sample): %v", err)
		}
		Free(tok)
	}
}

func BenchmarkSampler_TopKThenTopPTokenReadNoEval_Vocab262k(b *testing.B) {
	b.ReportAllocs()
	logits := RandomUniform(-5, 5, []int32{1, 262208}, DTypeFloat32)
	defer Free(logits)
	Materialize(logits)
	s := newSampler(1.0, 0.95, 0, 64)
	b.ResetTimer()
	for b.Loop() {
		tok := s.Sample(logits)
		_ = tok.Int()
		Free(tok)
	}
}

func BenchmarkSampler_TopKThenTopPTokenReadNoEvalChecked_Vocab262k(b *testing.B) {
	b.ReportAllocs()
	logits := RandomUniform(-5, 5, []int32{1, 262208}, DTypeFloat32)
	defer Free(logits)
	Materialize(logits)
	s := newSampler(1.0, 0.95, 0, 64)
	b.ResetTimer()
	for b.Loop() {
		tok := s.Sample(logits)
		_ = tok.Int()
		if err := LastError(); err != nil {
			Free(tok)
			b.Fatalf("token read: %v", err)
		}
		Free(tok)
	}
}

func BenchmarkSampler_TopKThenTopPWithSuppression_Vocab262k(b *testing.B) {
	b.ReportAllocs()
	logits := RandomUniform(-5, 5, []int32{1, 262208}, DTypeFloat32)
	defer Free(logits)
	Materialize(logits)
	suppress := []int32{0, 2, 3, 4, 46, 47, 48, 49, 51, 52, 98, 100, 101, 105, 255999, 256000, 258880, 258881, 258882, 258883, 258884}
	s := NewSamplerWithSuppression(1.0, 0.95, 0, 64, suppress)
	defer CloseSampler(s)
	b.ResetTimer()
	for b.Loop() {
		tok := s.Sample(logits)
		if err := Eval(tok); err != nil {
			Free(tok)
			b.Fatalf("Eval(sample): %v", err)
		}
		Free(tok)
	}
}

func BenchmarkSampler_PrefetchLogitsThenSampleEval_WithSuppression_Vocab262k(b *testing.B) {
	b.ReportAllocs()
	base := RandomUniform(-5, 5, []int32{1, 262208}, DTypeFloat32)
	zero := Zeros([]int32{1, 262208}, DTypeFloat32)
	defer Free(base, zero)
	Materialize(base, zero)
	suppress := []int32{0, 2, 3, 4, 46, 47, 48, 49, 51, 52, 98, 100, 101, 105, 255999, 256000, 258880, 258881, 258882, 258883, 258884}
	s := NewSamplerWithSuppression(1.0, 0.95, 0, 64, suppress)
	defer CloseSampler(s)
	b.ResetTimer()
	for b.Loop() {
		logits := Add(base, zero)
		if err := EvalAsync(logits); err != nil {
			Free(logits)
			b.Fatalf("EvalAsync(logits): %v", err)
		}
		tok := s.Sample(logits)
		if err := Eval(tok); err != nil {
			Free(logits, tok)
			b.Fatalf("Eval(sample): %v", err)
		}
		_ = tok.Int()
		Detach(logits, tok)
		Free(logits, tok)
	}
}

func BenchmarkSampler_CombinedLogitsSampleEval_WithSuppression_Vocab262k(b *testing.B) {
	b.ReportAllocs()
	base := RandomUniform(-5, 5, []int32{1, 262208}, DTypeFloat32)
	zero := Zeros([]int32{1, 262208}, DTypeFloat32)
	defer Free(base, zero)
	Materialize(base, zero)
	suppress := []int32{0, 2, 3, 4, 46, 47, 48, 49, 51, 52, 98, 100, 101, 105, 255999, 256000, 258880, 258881, 258882, 258883, 258884}
	s := NewSamplerWithSuppression(1.0, 0.95, 0, 64, suppress)
	defer CloseSampler(s)
	b.ResetTimer()
	for b.Loop() {
		logits := Add(base, zero)
		tok := s.Sample(logits)
		if err := EvalAsync(logits, tok); err != nil {
			Free(logits, tok)
			b.Fatalf("EvalAsync(logits, sample): %v", err)
		}
		_ = tok.Int()
		Detach(logits, tok)
		Free(logits, tok)
	}
}

func BenchmarkSampler_PrefetchLogitsDirtyThenSampleEval_WithSuppression_Vocab262k(b *testing.B) {
	b.ReportAllocs()
	base := RandomUniform(-5, 5, []int32{1, 262208}, DTypeFloat32)
	zero := Zeros([]int32{1, 262208}, DTypeFloat32)
	defer Free(base, zero)
	Materialize(base, zero)
	cache := NewPagedKVCache(0, 256)
	defer cache.Reset()
	k, v := makeSingleTokenKVShape(1, 2, 16)
	defer Free(k, v)
	state := cache.UpdateBorrowedPages(k, v, 1)
	state.Free()
	if err := Eval(cache.AppendDirtyState(nil)...); err != nil {
		b.Fatalf("Eval dirty state: %v", err)
	}
	suppress := []int32{0, 2, 3, 4, 46, 47, 48, 49, 51, 52, 98, 100, 101, 105, 255999, 256000, 258880, 258881, 258882, 258883, 258884}
	s := NewSamplerWithSuppression(1.0, 0.95, 0, 64, suppress)
	defer CloseSampler(s)
	var stack [8]*Array
	b.ResetTimer()
	for b.Loop() {
		logits := Add(base, zero)
		eval := stack[:0]
		eval = append(eval, logits)
		eval = appendCacheDirtyState(eval, cache)
		if err := EvalAsync(eval...); err != nil {
			Free(logits)
			b.Fatalf("EvalAsync(logits, dirty): %v", err)
		}
		tok := s.Sample(logits)
		if err := Eval(tok); err != nil {
			Free(logits, tok)
			b.Fatalf("Eval(sample): %v", err)
		}
		_ = tok.Int()
		Detach(logits, tok)
		Free(logits, tok)
	}
}

func BenchmarkSampler_CombinedLogitsSampleDirtyEval_WithSuppression_Vocab262k(b *testing.B) {
	b.ReportAllocs()
	base := RandomUniform(-5, 5, []int32{1, 262208}, DTypeFloat32)
	zero := Zeros([]int32{1, 262208}, DTypeFloat32)
	defer Free(base, zero)
	Materialize(base, zero)
	cache := NewPagedKVCache(0, 256)
	defer cache.Reset()
	k, v := makeSingleTokenKVShape(1, 2, 16)
	defer Free(k, v)
	state := cache.UpdateBorrowedPages(k, v, 1)
	state.Free()
	if err := Eval(cache.AppendDirtyState(nil)...); err != nil {
		b.Fatalf("Eval dirty state: %v", err)
	}
	suppress := []int32{0, 2, 3, 4, 46, 47, 48, 49, 51, 52, 98, 100, 101, 105, 255999, 256000, 258880, 258881, 258882, 258883, 258884}
	s := NewSamplerWithSuppression(1.0, 0.95, 0, 64, suppress)
	defer CloseSampler(s)
	var stack [8]*Array
	b.ResetTimer()
	for b.Loop() {
		logits := Add(base, zero)
		tok := s.Sample(logits)
		eval := stack[:0]
		eval = append(eval, logits, tok)
		eval = appendCacheDirtyState(eval, cache)
		if err := EvalAsync(eval...); err != nil {
			Free(logits, tok)
			b.Fatalf("EvalAsync(logits, sample, dirty): %v", err)
		}
		_ = tok.Int()
		Detach(logits, tok)
		Free(logits, tok)
	}
}

func BenchmarkSampler_CompiledTopKThenTopP_Vocab262k(b *testing.B) {
	b.ReportAllocs()
	logits := RandomUniform(-5, 5, []int32{1, 262208}, DTypeFloat32)
	defer Free(logits)
	Materialize(logits)
	// Production shape: the PRNG key is the second compiled input, created
	// + freed per token — this bench carries the full per-draw key cost.
	keys := NewSamplerKeys(1)
	compiled := CompileShapeless(func(inputs []*Array) []*Array {
		return []*Array{sampleTopKTopPToken(inputs[0], 64, 0.95, inputs[1])}
	}, false)
	defer compiled.Free()
	b.ResetTimer()
	for b.Loop() {
		key := keys.Next()
		tok := compiled.Call(logits, key)[0]
		Free(key)
		if err := Eval(tok); err != nil {
			Free(tok)
			b.Fatalf("Eval(compiled sample): %v", err)
		}
		Free(tok)
	}
}

func BenchmarkSampler_CompiledTopKThenTopPCallOne_Vocab262k(b *testing.B) {
	b.ReportAllocs()
	logits := RandomUniform(-5, 5, []int32{1, 262208}, DTypeFloat32)
	defer Free(logits)
	Materialize(logits)
	// CallOne is single-input, so the key is a CAPTURED constant — the
	// keyless lower bound for the lean call path, NOT the production shape
	// (a captured key repeats the identical draw; production threads the
	// key as a second Call input).
	key := RandomKey(1)
	Materialize(key)
	defer Free(key)
	compiled := CompileShapeless(func(inputs []*Array) []*Array {
		return []*Array{sampleTopKTopPToken(inputs[0], 64, 0.95, key)}
	}, false)
	defer compiled.Free()
	b.ResetTimer()
	for b.Loop() {
		tok := compiled.CallOne(logits)
		if err := Eval(tok); err != nil {
			Free(tok)
			b.Fatalf("Eval(compiled sample): %v", err)
		}
		Free(tok)
	}
}

// BenchmarkSampler_MinP01_Temp1 isolates min-p path which uses Softmax + MaxAxis
// + MulScalar + Greater(scalar) + Where.  Targets W11-R inline-Greater opportunity.
func BenchmarkSampler_MinP01_Temp1(b *testing.B) {
	logits := RandomUniform(-5, 5, []int32{1, 32000}, DTypeFloat32)
	Materialize(logits)
	s := newSampler(1.0, 0, 0.1, 0)
	for b.Loop() {
		tok := s.Sample(logits)
		Materialize(tok)
	}
}

// BenchmarkSampler_Temperature_PerToken isolates pure Temperature.Sample —
// already routes through MulScalar (W11-F).  Useful as floor reference.
func BenchmarkSampler_Temperature_PerToken(b *testing.B) {
	logits := RandomUniform(-5, 5, []int32{1, 32000}, DTypeFloat32)
	Materialize(logits)
	s := Temperature(0.7)
	for b.Loop() {
		y := s.Sample(logits)
		Materialize(y)
	}
}

// BenchmarkSampler_SuppressedGreedy_Gemma exercises the suppressedGreedy
// fast-path used by the Gemma assistant when only suppression is configured.
// Triggers suppressTokenLogits scalar FromValue (-inf) on each call.
func BenchmarkSampler_SuppressedGreedy_Gemma(b *testing.B) {
	logits := RandomUniform(-5, 5, []int32{1, 32000}, DTypeFloat32)
	Materialize(logits)
	suppress := []int32{0, 2, 3, 4, 46, 47, 48, 49, 50, 51, 52, 98, 100, 101, 105}
	s := NewSamplerWithSuppression(0, 0, 0, 0, suppress)
	defer CloseSampler(s)
	for b.Loop() {
		tok := s.Sample(logits)
		Materialize(tok)
		Free(tok)
	}
}

// BenchmarkApplyRepeatPenalty_Hist64 exercises applyRepeatPenalty with a
// realistic 64-token history.  Targets W10-V scratch pool + W11-R FromValue
// crossings (zero / invPenalty / penaltyVal).
func BenchmarkApplyRepeatPenalty_Hist64(b *testing.B) {
	logits := RandomUniform(-5, 5, []int32{1, 32000}, DTypeFloat32)
	Materialize(logits)
	hist := make([]int32, 64)
	for i := range hist {
		hist[i] = int32(i * 17 % 32000)
	}
	for b.Loop() {
		y := applyRepeatPenalty(logits, hist, 1.1)
		Materialize(y)
	}
}

// BenchmarkHostUnsuppressedGreedyToken_Gemma exercises the Gemma-sized
// host-side fallback that allocates suppressed map every call.  Stress on
// W10-V map elimination.
func BenchmarkHostUnsuppressedGreedyToken_Gemma(b *testing.B) {
	values := make([]float32, 258885)
	values[0] = 100
	values[123] = 10
	logits := FromValues(values, 1, len(values))
	Materialize(logits)
	suppress := []int32{0, 2, 3, 4, 46, 47, 48, 49, 50, 51, 52, 98, 100, 101, 105, 255999, 256000, 258880, 258881, 258882, 258883, 258884}
	for b.Loop() {
		tok, err := hostUnsuppressedGreedyToken(logits, suppress)
		if err != nil {
			b.Fatal(err)
		}
		Materialize(tok)
		Free(tok)
	}
}

// BenchmarkInspectAttentionCache_Realistic exercises the host-side
// inspectAttentionCache fan-out used by attention probes. Cache shape
// [1, 32, 1024, 128] = 4M float32 = 16MB — the per-call copy that the
// W11-R zero-copy view pattern eliminates.
func BenchmarkInspectAttentionCache_Realistic(b *testing.B) {
	cache := NewKVCache()
	// [1, 32 heads, 1024 tokens, 128 head_dim] = 4_194_304 float32 = 16 MB
	const heads, seqLen, headDim = 32, 1024, 128
	size := 1 * heads * seqLen * headDim
	data := make([]float32, size)
	for i := range data {
		data[i] = float32(i) * 0.0001
	}
	k := FromValues(data, 1, heads, seqLen, headDim)
	v := FromValues(data, 1, heads, seqLen, headDim)
	outK, outV := cache.Update(k, v, seqLen)
	Materialize(outK, outV)
	Detach(outK)
	Detach(outV)
	for b.Loop() {
		snapshot, ok := inspectAttentionCache(cache, seqLen)
		if !ok {
			b.Fatal("inspectAttentionCache returned not-ok")
		}
		if snapshot.NumHeads != heads {
			b.Fatalf("snapshot.NumHeads = %d, want %d", snapshot.NumHeads, heads)
		}
	}
}

// BenchmarkSummarizeProbeLogitsCompact_Gemma exercises the topK fan-out
// used by ProbeLogits.  TopK = 8 by default, so the topValues.Floats()
// candidate copies only 32 bytes per call, but the per-op alloc count
// matters when probes fire per-decoded-token.
func BenchmarkSummarizeProbeLogitsCompact_Gemma(b *testing.B) {
	const vocab = 258885
	values := make([]float32, vocab)
	for i := range values {
		values[i] = float32(i%1000) * 0.001
	}
	row := FromValues(values, 1, vocab)
	Materialize(row)
	shape := []int32{1, vocab}
	for b.Loop() {
		summary, _, err := summarizeProbeLogitsCompact(row, shape, vocab, defaultProbeTopK)
		if err != nil {
			b.Fatal(err)
		}
		if len(summary.Top) != defaultProbeTopK {
			b.Fatalf("len(Top) = %d, want %d", len(summary.Top), defaultProbeTopK)
		}
	}
}

// BenchmarkInspectKVCacheRange_Realistic exercises the per-block KV
// snapshot fan-out used by KVSnapshot capture. Same 16MB cache slice
// drives the kSliced.Floats() + vSliced.Floats() pair on the !RawKVOnly path.
//
// PRODUCTION NOTE (#76): the continuity serve never pays this 98MB/op —
// the sleep lane defaults to kv.EncodingNative (RawKVOnly: no float32
// side copies, agent/wake_sleep.go) and the trusted-prefix capture
// bounds each turn to its new range (BlockStartToken). This bench
// measures the non-native full-capture path for lib callers.
func BenchmarkInspectKVCacheRange_Realistic(b *testing.B) {
	cache := NewKVCache()
	const heads, seqLen, headDim = 32, 1024, 128
	size := 1 * heads * seqLen * headDim
	data := make([]float32, size)
	for i := range data {
		data[i] = float32(i) * 0.0001
	}
	k := FromValues(data, 1, heads, seqLen, headDim)
	v := FromValues(data, 1, heads, seqLen, headDim)
	outK, outV := cache.Update(k, v, seqLen)
	Materialize(outK, outV)
	Detach(outK)
	Detach(outV)
	opts := KVSnapshotCaptureOptions{}
	for b.Loop() {
		snapshot, ok := inspectKVCacheRangeWithOptions(cache, 0, seqLen, opts)
		if !ok {
			b.Fatal("inspectKVCacheRangeWithOptions returned not-ok")
		}
		if snapshot.NumHeads != heads {
			b.Fatalf("snapshot.NumHeads = %d, want %d", snapshot.NumHeads, heads)
		}
	}
}

// BenchmarkMaterialiseFloat32View_Slow_NB sizes the legacy helper across the
// realistic tensor-size range — characterises the cgo Materialize crossing
// cost as a function of payload bytes.  Compare against the
// BenchmarkMaterialiseFloat32ViewFast_FastPath_NB series to read off the
// crossover threshold.
func benchMaterialiseSlow(b *testing.B, n int) {
	b.Helper()
	values := make([]float32, n)
	for i := range values {
		values[i] = float32(i)
	}
	arr := FromValues(values, 1, n)
	Materialize(arr)
	defer Free(arr)
	for b.Loop() {
		src, converted, err := materialiseFloat32View(arr)
		if err != nil {
			b.Fatal(err)
		}
		_ = src.Size()
		runtime.KeepAlive(src)
		Free(converted)
	}
}

func benchMaterialiseFast(b *testing.B, n int) {
	b.Helper()
	values := make([]float32, n)
	for i := range values {
		values[i] = float32(i)
	}
	arr := FromValues(values, 1, n)
	Materialize(arr)
	defer Free(arr)
	for b.Loop() {
		view, cleanup, err := materialiseFloat32ViewFast(arr)
		if err != nil {
			b.Fatal(err)
		}
		_ = len(view)
		cleanup()
	}
}

// benchFloats sizes the legacy *Array.Floats() copy at the same size points
// so the fast-path crossover threshold can be read off directly.
func benchFloats(b *testing.B, n int) {
	b.Helper()
	values := make([]float32, n)
	for i := range values {
		values[i] = float32(i)
	}
	arr := FromValues(values, 1, n)
	Materialize(arr)
	defer Free(arr)
	for b.Loop() {
		out := arr.Floats()
		_ = len(out)
	}
}

func BenchmarkMaterialiseFloat32View_Floats_128B(b *testing.B)  { benchFloats(b, 32) }
func BenchmarkMaterialiseFloat32View_Floats_1KB(b *testing.B)   { benchFloats(b, 256) }
func BenchmarkMaterialiseFloat32View_Floats_10KB(b *testing.B)  { benchFloats(b, 2560) }
func BenchmarkMaterialiseFloat32View_Floats_100KB(b *testing.B) { benchFloats(b, 25600) }
func BenchmarkMaterialiseFloat32View_Floats_1MB(b *testing.B)   { benchFloats(b, 262144) }

func BenchmarkMaterialiseFloat32View_Slow_128B(b *testing.B)  { benchMaterialiseSlow(b, 32) }
func BenchmarkMaterialiseFloat32View_Slow_1KB(b *testing.B)   { benchMaterialiseSlow(b, 256) }
func BenchmarkMaterialiseFloat32View_Slow_10KB(b *testing.B)  { benchMaterialiseSlow(b, 2560) }
func BenchmarkMaterialiseFloat32View_Slow_100KB(b *testing.B) { benchMaterialiseSlow(b, 25600) }
func BenchmarkMaterialiseFloat32View_Slow_1MB(b *testing.B)   { benchMaterialiseSlow(b, 262144) }
func BenchmarkMaterialiseFloat32ViewFast_128B(b *testing.B)   { benchMaterialiseFast(b, 32) }
func BenchmarkMaterialiseFloat32ViewFast_1KB(b *testing.B)    { benchMaterialiseFast(b, 256) }
func BenchmarkMaterialiseFloat32ViewFast_10KB(b *testing.B)   { benchMaterialiseFast(b, 2560) }
func BenchmarkMaterialiseFloat32ViewFast_100KB(b *testing.B)  { benchMaterialiseFast(b, 25600) }
func BenchmarkMaterialiseFloat32ViewFast_1MB(b *testing.B)    { benchMaterialiseFast(b, 262144) }
