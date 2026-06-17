// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

// Parity is the dual-path correctness contract: an op on the native (no-cgo)
// path is only trusted once its output matches the mlx-c path (pkg/metal)
// byte-for-byte on the same metallib. These tests are the gate every new
// native kernel passes before it can replace its cgo counterpart. No model is
// loaded — synthetic tensors only (AX-11).
package native_test

import (
	"fmt"
	"os"
	"testing"

	"dappco.re/go/mlx/pkg/metal"
	"dappco.re/go/mlx/pkg/model"
	"dappco.re/go/mlx/pkg/native"
)

// assertParity drives op over both paths and fails on the first byte-level
// difference: native (tmc/apple → Metal) vs mlx-c (cgo → MLX). Both dispatch
// the same v_<Op> kernel from the same mlx.metallib, so any difference is a
// real defect in the native dispatch (or a divergent mlx-c path), not rounding.
// name labels the op in failures; nativeOp is the no-cgo wrapper; metalOp is its
// pkg/metal counterpart.
func assertParity(t *testing.T, name string, in []float32, nativeOp func([]float32) ([]float32, error), metalOp func(*metal.Array) *metal.Array) {
	t.Helper()
	if os.Getenv(native.MetallibPathEnv) == "" {
		t.Skipf("%s not set — needs the compiled metallib", native.MetallibPathEnv)
	}

	got, err := nativeOp(in)
	if err != nil {
		t.Fatalf("native.%s: %v", name, err)
	}

	arr := metal.FromValues(in, len(in))
	res := metalOp(arr)
	metal.Materialize(res)
	want := res.Floats()
	metal.Free(arr, res)

	if len(got) != len(want) {
		t.Fatalf("%s length mismatch: native %d, mlx-c %d", name, len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("%s parity break at [%d]: native %v, mlx-c %v (in %v, %d elems)", name, i, got[i], want[i], in[i], len(want))
		}
	}
}

// mixedInputs returns a spread of signs and magnitudes for ops valid over the
// whole real line (Abs, Negative, Tanh, Sigmoid, Round). Same shape as the
// original Square fixture.
func mixedInputs() []float32 {
	in := make([]float32, 4096)
	for i := range in {
		in[i] = float32(i)*0.5 - 1000.0
	}
	return in
}

// assertBinaryParity drives a binary op over both paths and fails on the first
// byte-level difference: native vs mlx-c, same vv_<Op> kernel from the same
// metallib, so bit-identical is expected.
func assertBinaryParity(t *testing.T, name string, a, b []float32, nativeOp func([]float32, []float32) ([]float32, error), metalOp func(*metal.Array, *metal.Array) *metal.Array) {
	t.Helper()
	if os.Getenv(native.MetallibPathEnv) == "" {
		t.Skipf("%s not set — needs the compiled metallib", native.MetallibPathEnv)
	}
	got, err := nativeOp(a, b)
	if err != nil {
		t.Fatalf("native.%s: %v", name, err)
	}
	aa := metal.FromValues(a, len(a))
	bb := metal.FromValues(b, len(b))
	res := metalOp(aa, bb)
	metal.Materialize(res)
	want := res.Floats()
	metal.Free(aa, bb, res)
	if len(got) != len(want) {
		t.Fatalf("%s length mismatch: native %d, mlx-c %d", name, len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("%s parity break at [%d]: native %v, mlx-c %v", name, i, got[i], want[i])
		}
	}
}

// TestNativeParity_Add asserts bit-identical a+b (the residual add).
func TestNativeParity_Add(t *testing.T) {
	a, b := mixedInputs(), mixedInputs()
	for i := range b {
		b[i] = float32((i*53)%97-48) * 0.7 // differ from a
	}
	assertBinaryParity(t, "Add", a, b, native.Add, metal.Add)
}

// TestNativeParity_Mul asserts bit-identical a*b (the MLP gate·up).
func TestNativeParity_Mul(t *testing.T) {
	a, b := mixedInputs(), mixedInputs()
	for i := range b {
		b[i] = float32((i*29)%53-20) * 0.05
	}
	assertBinaryParity(t, "Mul", a, b, native.Mul, metal.Mul)
}

// TestNativeParity_GeluGateMul checks the native composed GELU gate against
// mlx-c's fused GELUGateMul. Native runs the gelu_approx ops separately while
// mlx fuses them with mlx_compile, so this is the first op where bit-identical
// isn't guaranteed — the test reports the max abs diff and holds it to a tight
// fp tolerance. This is the "compose, don't drive" class of native op.
func TestNativeParity_GeluGateMul(t *testing.T) {
	if os.Getenv(native.MetallibPathEnv) == "" {
		t.Skipf("%s not set — needs the compiled metallib", native.MetallibPathEnv)
	}

	const n = 4096
	gate := make([]float32, n)
	up := make([]float32, n)
	for i := range gate {
		gate[i] = float32((i*37)%160-80) * 0.1 // ~[-8, 8): realistic pre-activation
		up[i] = float32((i*53)%97-48) * 0.04   // ~[-2, 2)
	}

	got, err := native.GeluGateMul(gate, up)
	if err != nil {
		t.Fatalf("native.GeluGateMul: %v", err)
	}

	gA := metal.FromValues(gate, n)
	uA := metal.FromValues(up, n)
	res := metal.GELUGateMul(gA, uA) // mlx-c fused reference
	metal.Materialize(res)
	want := res.Floats()
	metal.Free(gA, uA, res)

	if len(got) != len(want) {
		t.Fatalf("GeluGateMul length mismatch: native %d, mlx-c %d", len(got), len(want))
	}
	var maxAbs, maxRel float32
	exact := 0
	for i := range want {
		d := got[i] - want[i]
		if d < 0 {
			d = -d
		}
		if d == 0 {
			exact++
		}
		if d > maxAbs {
			maxAbs = d
		}
		den := want[i]
		if den < 0 {
			den = -den
		}
		if den > 1e-3 {
			if r := d / den; r > maxRel {
				maxRel = r
			}
		}
	}
	t.Logf("GeluGateMul vs mlx-c fused: %d/%d exact, maxAbs=%g maxRel=%g", exact, len(want), maxAbs, maxRel)
	if maxAbs > 1e-3 {
		t.Fatalf("GeluGateMul diff too large: maxAbs=%g maxRel=%g (native-composed vs mlx-fused)", maxAbs, maxRel)
	}
}

// positiveInputs returns strictly-positive values for ops whose domain excludes
// zero/negatives (Sqrt, Rsqrt, Log).
func positiveInputs() []float32 {
	in := make([]float32, 4096)
	for i := range in {
		in[i] = float32(i)*0.5 + 0.5
	}
	return in
}

// TestNativeParity_Square drives the same square over both paths and asserts
// bit-identical output. Both dispatch the same v_Square kernel from the same
// mlx.metallib, so any difference is a real defect in the native dispatch.
func TestNativeParity_Square(t *testing.T) {
	assertParity(t, "Square", mixedInputs(), native.Square, metal.Square)
}

// TestNativeParity_Abs asserts bit-identical |in| across both paths.
func TestNativeParity_Abs(t *testing.T) {
	assertParity(t, "Abs", mixedInputs(), native.Abs, metal.Abs)
}

// TestNativeParity_Negative asserts bit-identical -in across both paths.
func TestNativeParity_Negative(t *testing.T) {
	assertParity(t, "Negative", mixedInputs(), native.Negative, metal.Negative)
}

// TestNativeParity_Exp asserts bit-identical exp(in) across both paths. The
// input is kept in [-8, 8) so the result stays finite and the test exercises
// real values rather than saturating to +Inf.
func TestNativeParity_Exp(t *testing.T) {
	in := make([]float32, 4096)
	for i := range in {
		in[i] = float32(i)*(16.0/4096.0) - 8.0
	}
	assertParity(t, "Exp", in, native.Exp, metal.Exp)
}

// TestNativeParity_Sigmoid asserts bit-identical 1/(1+exp(-in)) across both paths.
func TestNativeParity_Sigmoid(t *testing.T) {
	assertParity(t, "Sigmoid", mixedInputs(), native.Sigmoid, metal.Sigmoid)
}

// TestNativeParity_Tanh asserts bit-identical tanh(in) across both paths.
func TestNativeParity_Tanh(t *testing.T) {
	assertParity(t, "Tanh", mixedInputs(), native.Tanh, metal.Tanh)
}

// TestNativeParity_Sqrt asserts bit-identical sqrt(in) across both paths over a
// strictly-positive domain.
func TestNativeParity_Sqrt(t *testing.T) {
	assertParity(t, "Sqrt", positiveInputs(), native.Sqrt, metal.Sqrt)
}

// TestNativeParity_Rsqrt asserts bit-identical 1/sqrt(in) across both paths over
// a strictly-positive domain.
func TestNativeParity_Rsqrt(t *testing.T) {
	assertParity(t, "Rsqrt", positiveInputs(), native.Rsqrt, metal.Rsqrt)
}

// TestNativeParity_Log asserts bit-identical ln(in) across both paths over a
// strictly-positive domain.
func TestNativeParity_Log(t *testing.T) {
	assertParity(t, "Log", positiveInputs(), native.Log, metal.Log)
}

// TestNativeParity_Round asserts bit-identical round-to-nearest (ties to even)
// across both paths.
func TestNativeParity_Round(t *testing.T) {
	assertParity(t, "Round", mixedInputs(), native.Round, metal.Round)
}

// TestNativeParity_MatVec gates the native gemv against mlx-c: out = mat @ vec
// for a row-major (outDim x inDim) matrix, driven directly on the native path
// vs metal.Matmul of (outDim x inDim) @ (inDim x 1). Both resolve to the same
// gemv variant from the same metallib, so the result is expected bit-identical —
// this is the first load-bearing kernel (threadgroup-parallel, real param ABI)
// proving the dual path reaches more than elementwise ops. The shape (outDim
// 512, inDim 256) lands on the realistic bm4_bn1_sm1_sn32_tm4_tn4 variant.
func TestNativeParity_MatVec(t *testing.T) {
	if os.Getenv(native.MetallibPathEnv) == "" {
		t.Skipf("%s not set — needs the compiled metallib", native.MetallibPathEnv)
	}

	const outDim, inDim = 512, 256
	mat := make([]float32, outDim*inDim)
	for i := range mat {
		mat[i] = float32((i*37)%101-50) * 0.01 // spread, deterministic
	}
	vec := make([]float32, inDim)
	for i := range vec {
		vec[i] = float32((i*53)%97-48) * 0.01
	}

	got, err := native.MatVec(mat, vec, outDim, inDim)
	if err != nil {
		t.Fatalf("native.MatVec: %v", err)
	}

	matArr := metal.FromValues(mat, outDim, inDim)
	vecArr := metal.FromValues(vec, inDim, 1)
	res := metal.Matmul(matArr, vecArr)
	metal.Materialize(res)
	want := res.Floats() // (outDim x 1) flattened
	metal.Free(matArr, vecArr, res)

	if len(got) != len(want) {
		t.Fatalf("MatVec length mismatch: native %d, mlx-c %d", len(got), len(want))
	}

	// Cross-check against a CPU reference so a both-paths-return-zero false pass
	// can't slip through: the GPU result must be the real dot product (within
	// fp32 accumulation-order tolerance), and a real non-zero magnitude.
	var maxMag float64
	for n := 0; n < outDim; n++ {
		var ref float64
		for k := 0; k < inDim; k++ {
			ref += float64(mat[n*inDim+k]) * float64(vec[k])
		}
		if d := float64(got[n]) - ref; d > 1e-3 || d < -1e-3 {
			t.Fatalf("MatVec wrong at [%d]: gpu %v, cpu-ref %v", n, got[n], ref)
		}
		if m := ref; m > maxMag || -m > maxMag {
			maxMag = m
			if m < 0 {
				maxMag = -m
			}
		}
	}
	if maxMag < 1e-6 {
		t.Fatalf("MatVec reference is ~zero (maxMag %g) — test is not exercising the kernel", maxMag)
	}

	// Then the real gate: native and mlx-c must agree bit-for-bit (same kernel).
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("MatVec parity break at [%d]: native %v, mlx-c %v (%dx%d @ %d)", i, got[i], want[i], outDim, inDim, inDim)
		}
	}
}

// TestNativeParity_QMV gates the native 4-bit quantised matvec against mlx-c:
// out = x @ Wᵀ where W is affine-quantised. mlx-c packs the weights (Quantize);
// both paths then run on the same packed bytes — native.QMV driving affine_qmv
// directly vs metal.QuantizedMatmul. This is the kernel the 4-bit models we serve
// actually decode through, so it's the load-bearing parity. Shape (512x512,
// gs 64, b 4) lands on the fast variant (N%8==0, K%512==0).
func TestNativeParity_QMV(t *testing.T) {
	if os.Getenv(native.MetallibPathEnv) == "" {
		t.Skipf("%s not set — needs the compiled metallib", native.MetallibPathEnv)
	}

	const outDim, inDim, gs, bits = 512, 512, 64, 4
	w := make([]float32, outDim*inDim)
	for i := range w {
		w[i] = float32((i*37)%101-50) * 0.01
	}
	x := make([]float32, inDim)
	for i := range x {
		x[i] = float32((i*53)%97-48) * 0.01
	}

	wArr := metal.FromValues(w, outDim, inDim)
	wq, scales, biases, err := metal.Quantize(wArr, gs, bits, "affine")
	if err != nil {
		t.Fatalf("Quantize: %v", err)
	}
	metal.Materialize(wq)
	metal.Materialize(scales)
	metal.Materialize(biases)

	// mlx-c reference: transpose=true means out = x @ Wᵀ (the standard linear).
	xArr := metal.FromValues(x, 1, inDim)
	res := metal.QuantizedMatmul(xArr, wq, scales, biases, true, gs, bits)
	metal.Materialize(res)
	want := res.Floats()

	// native: the SAME packed bytes (RawBytes — the real buffer, not Bytes()
	// which is one byte per element), driven through affine_qmv.
	got, err := native.QMV(x, wq.RawBytes(), scales.RawBytes(), biases.RawBytes(), outDim, inDim, gs, bits)
	if err != nil {
		metal.Free(wArr, wq, scales, biases, xArr, res)
		t.Fatalf("native.QMV: %v", err)
	}
	metal.Free(wArr, wq, scales, biases, xArr, res)

	var mag float64
	for _, v := range want {
		if float64(v) > mag {
			mag = float64(v)
		} else if float64(-v) > mag {
			mag = float64(-v)
		}
	}
	if mag < 1e-6 {
		t.Fatalf("QMV reference is ~zero (mag %g) — not exercising the kernel", mag)
	}
	if len(got) != len(want) {
		t.Fatalf("QMV length mismatch: native %d, mlx-c %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("QMV parity break at [%d]: native %v, mlx-c %v (gs %d b %d)", i, got[i], want[i], gs, bits)
		}
	}
}

// TestNativeParity_QMVBF16 gates the bf16-activation 4-bit quantised matvec — the
// quantised decode projection — against mlx-c: a bf16 weight is affine-quantised
// (Quantize → bf16 scales/biases), then native.QMVBF16 driving affine_qmv_bfloat16_t
// on the packed bytes must equal metal.QuantizedMatmul (transpose=true) on bf16
// inputs, byte-for-byte. 512x512 / gs64 / b4 lands on the fast variant.
func TestNativeParity_QMVBF16(t *testing.T) {
	if os.Getenv(native.MetallibPathEnv) == "" {
		t.Skipf("%s not set — needs the compiled metallib", native.MetallibPathEnv)
	}

	const outDim, inDim, gs, bits = 512, 512, 64, 4
	w := make([]float32, outDim*inDim)
	for i := range w {
		w[i] = float32((i*37)%101-50) * 0.01
	}
	x := make([]float32, inDim)
	for i := range x {
		x[i] = float32((i*53)%97-48) * 0.01
	}
	wb := bf16Bytes(t, w, outDim, inDim)
	xb := bf16Bytes(t, x, inDim)

	// quantise the bf16 weight (scales/biases come out bf16, matching the dtype)
	wArr := metal.FromRawBytes(wb, []int{outDim, inDim}, metal.DTypeBFloat16)
	wq, scales, biases, err := metal.Quantize(wArr, gs, bits, "affine")
	if err != nil {
		metal.Free(wArr)
		t.Fatalf("Quantize: %v", err)
	}
	metal.Materialize(wq)
	metal.Materialize(scales)
	metal.Materialize(biases)

	// mlx-c reference: bf16 quantised matmul, transpose=true (out = x @ Wᵀ)
	xArr := metal.FromRawBytes(xb, []int{1, inDim}, metal.DTypeBFloat16)
	res := metal.QuantizedMatmul(xArr, wq, scales, biases, true, gs, bits)
	metal.Materialize(res)
	want := append([]byte(nil), res.RawBytes()...) // (outDim x 1) bf16 bytes; copy before Free

	// native: the SAME packed bytes through affine_qmv_bfloat16_t
	got, err := native.QMVBF16(xb, wq.RawBytes(), scales.RawBytes(), biases.RawBytes(), outDim, inDim, gs, bits)
	if err != nil {
		metal.Free(wArr, wq, scales, biases, xArr, res)
		t.Fatalf("native.QMVBF16: %v", err)
	}
	metal.Free(wArr, wq, scales, biases, xArr, res)

	assertBytesEqual(t, "QMVBF16", got, want, fmt.Sprintf("%dx%d gs%d b%d", outDim, inDim, gs, bits))
}

// TestModelBackendQuantNative gates the backend cross-section of the quant compute
// (pkg/model, part one): the native backend resolved THROUGH the registry —
// model.BackendQuant("native","affine") — must compute the affine quant matvec
// byte-for-byte identical to mlx-c. Proves pkg/native plugs into the model
// machinery the same way a quant format does, on the smallest real surface.
func TestModelBackendQuantNative(t *testing.T) {
	if os.Getenv(native.MetallibPathEnv) == "" {
		t.Skipf("%s not set — needs the compiled metallib", native.MetallibPathEnv)
	}
	const outDim, inDim, gs, bits = 512, 512, 64, 4
	w := make([]float32, outDim*inDim)
	for i := range w {
		w[i] = float32((i*37)%101-50) * 0.01
	}
	x := make([]float32, inDim)
	for i := range x {
		x[i] = float32((i*53)%97-48) * 0.01
	}
	wb := bf16Bytes(t, w, outDim, inDim)
	xb := bf16Bytes(t, x, inDim)

	wArr := metal.FromRawBytes(wb, []int{outDim, inDim}, metal.DTypeBFloat16)
	wq, scales, biases, err := metal.Quantize(wArr, gs, bits, "affine")
	if err != nil {
		metal.Free(wArr)
		t.Fatalf("Quantize: %v", err)
	}
	metal.Materialize(wq)
	metal.Materialize(scales)
	metal.Materialize(biases)
	xArr := metal.FromRawBytes(xb, []int{1, inDim}, metal.DTypeBFloat16)
	res := metal.QuantizedMatmul(xArr, wq, scales, biases, true, gs, bits)
	metal.Materialize(res)
	want := append([]byte(nil), res.RawBytes()...)

	// resolve the native backend's affine compute through the cross-section registry
	qc, ok := model.BackendQuant("native", "affine")
	if !ok {
		metal.Free(wArr, wq, scales, biases, xArr, res)
		t.Fatal("model.BackendQuant(\"native\",\"affine\") not registered")
	}
	got, err := qc.MatVec(xb, wq.RawBytes(), scales.RawBytes(), biases.RawBytes(), outDim, inDim, gs, bits)
	if err != nil {
		metal.Free(wArr, wq, scales, biases, xArr, res)
		t.Fatalf("QuantMatVec: %v", err)
	}
	metal.Free(wArr, wq, scales, biases, xArr, res)

	assertBytesEqual(t, "model.BackendQuant native/affine", got, want, fmt.Sprintf("%dx%d gs%d b%d", outDim, inDim, gs, bits))
	if qc.Kind() != "affine" {
		t.Fatalf("Kind() = %q, want affine", qc.Kind())
	}
	t.Logf("backend cross-section: model.BackendQuant(native/affine).MatVec ≡ mlx-c QuantizedMatmul — pkg/native plugs into the model contract")
}

// TestNativeParity_RMSNorm gates the native RMS norm against mlx-c over a small
// batch of rows: same rms kernel, same metallib, so bit-identical is expected.
// RMSNorm is the per-block normaliser in every decode step. axisSize 512 is well
// under the looped limit, so this exercises the single-row kernel.
func TestNativeParity_RMSNorm(t *testing.T) {
	if os.Getenv(native.MetallibPathEnv) == "" {
		t.Skipf("%s not set — needs the compiled metallib", native.MetallibPathEnv)
	}

	const rows, axis = 4, 512
	const eps = float32(1e-5)
	x := make([]float32, rows*axis)
	for i := range x {
		x[i] = float32((i*37)%101-50) * 0.02
	}
	weight := make([]float32, axis)
	for i := range weight {
		weight[i] = float32((i*29)%53-20)*0.03 + 1.0
	}

	got, err := native.RMSNorm(x, weight, rows, axis, eps)
	if err != nil {
		t.Fatalf("native.RMSNorm: %v", err)
	}

	xArr := metal.FromValues(x, rows, axis)
	wArr := metal.FromValues(weight, axis)
	res := metal.RMSNorm(xArr, wArr, eps)
	metal.Materialize(res)
	want := res.Floats()
	metal.Free(xArr, wArr, res)

	var mag float64
	for _, v := range want {
		if float64(v) > mag {
			mag = float64(v)
		} else if float64(-v) > mag {
			mag = float64(-v)
		}
	}
	if mag < 1e-6 {
		t.Fatalf("RMSNorm reference is ~zero (mag %g) — not exercising the kernel", mag)
	}
	if len(got) != len(want) {
		t.Fatalf("RMSNorm length mismatch: native %d, mlx-c %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("RMSNorm parity break at [%d]: native %v, mlx-c %v (%d rows x %d)", i, got[i], want[i], rows, axis)
		}
	}
}

// TestNativeParity_RoPE gates the native rotary embedding against mlx-c for the
// single-token decode case: x is (b, nHeads, 1, headDim), one position offset,
// full-rotary, non-traditional. Both hit MLX's rope_single kernel (specialised
// by the same function constants) so bit-identical is expected. This is the
// first func-const kernel on the native path.
func TestNativeParity_RoPE(t *testing.T) {
	if os.Getenv(native.MetallibPathEnv) == "" {
		t.Skipf("%s not set — needs the compiled metallib", native.MetallibPathEnv)
	}

	const b, nHeads, headDim = 1, 8, 64
	const base, scale = float32(10000), float32(1)
	const offset = 5
	x := make([]float32, b*nHeads*headDim)
	for i := range x {
		x[i] = float32((i*37)%101-50) * 0.02
	}

	got, err := native.RoPE(x, b, nHeads, headDim, base, scale, offset, false)
	if err != nil {
		t.Fatalf("native.RoPE: %v", err)
	}

	// mlx-c reference: shape (b, nHeads, 1, headDim), full rotary, non-traditional.
	xArr := metal.FromValues(x, b, nHeads, 1, headDim)
	res := metal.RoPE(xArr, headDim, false, base, scale, offset)
	metal.Materialize(res)
	want := res.Floats()
	metal.Free(xArr, res)

	var mag float64
	for _, v := range want {
		if float64(v) > mag {
			mag = float64(v)
		} else if float64(-v) > mag {
			mag = float64(-v)
		}
	}
	if mag < 1e-6 {
		t.Fatalf("RoPE reference is ~zero (mag %g) — not exercising the kernel", mag)
	}
	if len(got) != len(want) {
		t.Fatalf("RoPE length mismatch: native %d, mlx-c %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("RoPE parity break at [%d]: native %v, mlx-c %v (h%d d%d off%d)", i, got[i], want[i], nHeads, headDim, offset)
		}
	}
}

// TestNativeParity_SDPA gates the native decode attention against mlx-c. The
// kernel is bfloat16-only (the dtype attention actually runs in), so the test
// casts q/k/v to bf16 via mlx-c, runs both paths on the SAME bf16 bytes, and
// compares the bf16 output byte-for-byte. q is one query (b, nHeads, 1, headDim);
// k/v are a short KV cache (kvLen < 1024 keeps mlx on the single-pass kernel both
// sides). This is the last decode-step kernel; the native side is dtype-agnostic
// (it shuffles bytes through sdpa_vector).
func TestNativeParity_SDPA(t *testing.T) {
	if os.Getenv(native.MetallibPathEnv) == "" {
		t.Skipf("%s not set — needs the compiled metallib", native.MetallibPathEnv)
	}

	const b, nHeads, nKV, headDim, kvLen = 1, 8, 8, 64, 16
	const scale = float32(0.125) // 1/sqrt(64)

	mk := func(n, salt int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32((i*salt+7)%101-50) * 0.02
		}
		return s
	}
	qf := mk(b*nHeads*headDim, 37)
	kf := mk(b*nKV*kvLen*headDim, 53)
	vf := mk(b*nKV*kvLen*headDim, 29)

	// cast to bf16 (the only compiled sdpa_vector dtype); both paths share these.
	qA := metal.AsType(metal.FromValues(qf, b, nHeads, 1, headDim), metal.DTypeBFloat16)
	kA := metal.AsType(metal.FromValues(kf, b, nKV, kvLen, headDim), metal.DTypeBFloat16)
	vA := metal.AsType(metal.FromValues(vf, b, nKV, kvLen, headDim), metal.DTypeBFloat16)
	metal.Materialize(qA)
	metal.Materialize(kA)
	metal.Materialize(vA)

	res := metal.ScaledDotProductAttention(qA, kA, vA, scale, false)
	metal.Materialize(res)
	want := res.RawBytes()

	got, err := native.SDPA(qA.RawBytes(), kA.RawBytes(), vA.RawBytes(), b, nHeads, nKV, headDim, kvLen, scale)
	if err != nil {
		metal.Free(qA, kA, vA, res)
		t.Fatalf("native.SDPA: %v", err)
	}

	// snapshot the comparison bytes before freeing the mlx arrays.
	wantCopy := make([]byte, len(want))
	copy(wantCopy, want)
	metal.Free(qA, kA, vA, res)
	want = wantCopy

	nonZero := false
	for _, by := range want {
		if by != 0 {
			nonZero = true
			break
		}
	}
	if !nonZero {
		t.Fatal("SDPA reference is all-zero bytes — not exercising the kernel")
	}
	if len(got) != len(want) {
		t.Fatalf("SDPA length mismatch: native %d, mlx-c %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("SDPA parity break at byte [%d]: native %d, mlx-c %d (h%d kv%d d%d)", i, got[i], want[i], nHeads, kvLen, headDim)
		}
	}
}

// bf16Bytes casts a float32 fixture to bfloat16 via mlx-c and returns the raw
// bf16 bytes — the exact representation the native bf16 ops take. The caller
// supplies the shape so the cast array is materialised correctly; the bytes are
// snapshot-copied before the array is freed so the returned slice owns its
// storage. This is the shared front-half of every bf16 parity test (mirrors how
// TestNativeParity_SDPA prepares its bf16 inputs).
func bf16Bytes(t *testing.T, data []float32, shape ...int) []byte {
	t.Helper()
	arr := metal.AsType(metal.FromValues(data, shape...), metal.DTypeBFloat16)
	metal.Materialize(arr)
	raw := arr.RawBytes()
	out := make([]byte, len(raw))
	copy(out, raw)
	metal.Free(arr)
	return out
}

// assertBytesEqual fails on the first differing byte (or a length mismatch) and
// first checks the reference is not all-zero, so a both-paths-return-zero false
// pass can't slip through. It is the byte-level gate the bf16 ops share.
func assertBytesEqual(t *testing.T, name string, got, want []byte, ctx string) {
	t.Helper()
	nonZero := false
	for _, by := range want {
		if by != 0 {
			nonZero = true
			break
		}
	}
	if !nonZero {
		t.Fatalf("%s reference is all-zero bytes — not exercising the kernel", name)
	}
	if len(got) != len(want) {
		t.Fatalf("%s length mismatch: native %d, mlx-c %d", name, len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("%s parity break at byte [%d]: native %d, mlx-c %d (%s)", name, i, got[i], want[i], ctx)
		}
	}
}

// TestNativeParity_RMSNormBF16 gates the native bf16 RMS norm against mlx-c on
// the same bf16 arrays: both dispatch the rmsbfloat16 kernel from the same
// metallib, so bit-identical output is expected. RMSNorm is the per-block
// normaliser every decode step runs, and bf16 is the decode dtype. axisSize 512
// is well under the looped limit (single-row kernel).
func TestNativeParity_RMSNormBF16(t *testing.T) {
	if os.Getenv(native.MetallibPathEnv) == "" {
		t.Skipf("%s not set — needs the compiled metallib", native.MetallibPathEnv)
	}

	const rows, axis = 4, 512
	const eps = float32(1e-5)
	x := make([]float32, rows*axis)
	for i := range x {
		x[i] = float32((i*37)%101-50) * 0.02
	}
	weight := make([]float32, axis)
	for i := range weight {
		weight[i] = float32((i*29)%53-20)*0.03 + 1.0
	}

	// both paths share the SAME bf16 bytes (cast once via mlx-c).
	xb := bf16Bytes(t, x, rows, axis)
	wb := bf16Bytes(t, weight, axis)

	got, err := native.RMSNormBF16(xb, wb, rows, axis, eps)
	if err != nil {
		t.Fatalf("native.RMSNormBF16: %v", err)
	}

	xArr := metal.FromRawBytes(xb, []int{rows, axis}, metal.DTypeBFloat16)
	wArr := metal.FromRawBytes(wb, []int{axis}, metal.DTypeBFloat16)
	res := metal.RMSNorm(xArr, wArr, eps)
	metal.Materialize(res)
	want := res.RawBytes()
	wantCopy := make([]byte, len(want))
	copy(wantCopy, want)
	metal.Free(xArr, wArr, res)

	assertBytesEqual(t, "RMSNormBF16", got, wantCopy, fmt.Sprintf("%d rows x %d", rows, axis))
}

// TestNativeParity_MatVecBF16 gates the native bf16 gemv against mlx-c: out =
// mat @ vec for a row-major (outDim x inDim) matrix on the same bf16 bytes, vs
// metal.Matmul of (outDim x inDim) @ (inDim x 1). Both resolve to the same
// gemv_bfloat16 tile variant from the same metallib, so the result is expected
// bit-identical. Shape (outDim 512, inDim 256) lands on the realistic
// bm4_bn1_sm1_sn32_tm4_tn4 variant.
func TestNativeParity_MatVecBF16(t *testing.T) {
	if os.Getenv(native.MetallibPathEnv) == "" {
		t.Skipf("%s not set — needs the compiled metallib", native.MetallibPathEnv)
	}

	const outDim, inDim = 512, 256
	mat := make([]float32, outDim*inDim)
	for i := range mat {
		mat[i] = float32((i*37)%101-50) * 0.01
	}
	vec := make([]float32, inDim)
	for i := range vec {
		vec[i] = float32((i*53)%97-48) * 0.01
	}

	matb := bf16Bytes(t, mat, outDim, inDim)
	vecb := bf16Bytes(t, vec, inDim)

	got, err := native.MatVecBF16(matb, vecb, outDim, inDim)
	if err != nil {
		t.Fatalf("native.MatVecBF16: %v", err)
	}

	matArr := metal.FromRawBytes(matb, []int{outDim, inDim}, metal.DTypeBFloat16)
	vecArr := metal.FromRawBytes(vecb, []int{inDim, 1}, metal.DTypeBFloat16)
	res := metal.Matmul(matArr, vecArr)
	metal.Materialize(res)
	want := res.RawBytes() // (outDim x 1) bf16 bytes
	wantCopy := make([]byte, len(want))
	copy(wantCopy, want)
	metal.Free(matArr, vecArr, res)

	assertBytesEqual(t, "MatVecBF16", got, wantCopy, fmt.Sprintf("%dx%d @ %d", outDim, inDim, inDim))
}

// TestNativeParity_RoPEBF16 gates the native bf16 rotary embedding against mlx-c
// for the single-token decode case: x is (b, nHeads, 1, headDim) bf16 bytes, one
// position offset, full-rotary, non-traditional. Both hit rope_single_bfloat16
// (specialised by the same function constants) so bit-identical is expected.
func TestNativeParity_RoPEBF16(t *testing.T) {
	if os.Getenv(native.MetallibPathEnv) == "" {
		t.Skipf("%s not set — needs the compiled metallib", native.MetallibPathEnv)
	}

	const b, nHeads, headDim = 1, 8, 64
	const base, scale = float32(10000), float32(1)
	const offset = 5
	x := make([]float32, b*nHeads*headDim)
	for i := range x {
		x[i] = float32((i*37)%101-50) * 0.02
	}

	xb := bf16Bytes(t, x, b, nHeads, 1, headDim)

	got, err := native.RoPEBF16(xb, b, nHeads, headDim, base, scale, offset, false)
	if err != nil {
		t.Fatalf("native.RoPEBF16: %v", err)
	}

	xArr := metal.FromRawBytes(xb, []int{b, nHeads, 1, headDim}, metal.DTypeBFloat16)
	res := metal.RoPE(xArr, headDim, false, base, scale, offset)
	metal.Materialize(res)
	want := res.RawBytes()
	wantCopy := make([]byte, len(want))
	copy(wantCopy, want)
	metal.Free(xArr, res)

	assertBytesEqual(t, "RoPEBF16", got, wantCopy, fmt.Sprintf("h%d d%d off%d", nHeads, headDim, offset))
}

// TestNativeParity_AddBF16 gates the native bf16 residual add against mlx-c on
// the same bf16 bytes: both dispatch vv_Addbfloat16 from the same metallib, so
// bit-identical is expected. n=4096 elements.
func TestNativeParity_AddBF16(t *testing.T) {
	if os.Getenv(native.MetallibPathEnv) == "" {
		t.Skipf("%s not set — needs the compiled metallib", native.MetallibPathEnv)
	}

	const n = 4096
	a := make([]float32, n)
	b := make([]float32, n)
	for i := range a {
		a[i] = float32(i)*0.5 - 1000.0
		b[i] = float32((i*53)%97-48) * 0.7
	}

	ab := bf16Bytes(t, a, n)
	bb := bf16Bytes(t, b, n)

	got, err := native.AddBF16(ab, bb)
	if err != nil {
		t.Fatalf("native.AddBF16: %v", err)
	}

	aArr := metal.FromRawBytes(ab, []int{n}, metal.DTypeBFloat16)
	bArr := metal.FromRawBytes(bb, []int{n}, metal.DTypeBFloat16)
	res := metal.Add(aArr, bArr)
	metal.Materialize(res)
	want := res.RawBytes()
	wantCopy := make([]byte, len(want))
	copy(wantCopy, want)
	metal.Free(aArr, bArr, res)

	assertBytesEqual(t, "AddBF16", got, wantCopy, fmt.Sprintf("n%d", n))
}

// TestNativeChain_NormProject proves on-device chaining: NormProject runs
// rmsnorm→projection in one command buffer with the intermediate resident on the
// GPU, and must equal the same two native kernels run separately (each of which
// is already mlx-c parity-green). Identical output means keeping the intermediate
// on-device — no host round-trip — is correct. This is the decode-step foundation.
func TestNativeChain_NormProject(t *testing.T) {
	if os.Getenv(native.MetallibPathEnv) == "" {
		t.Skipf("%s not set — needs the compiled metallib", native.MetallibPathEnv)
	}

	const dIn, dOut = 512, 256
	const eps = float32(1e-5)
	x := make([]float32, dIn)
	for i := range x {
		x[i] = float32((i*37)%101-50) * 0.02
	}
	normW := make([]float32, dIn)
	for i := range normW {
		normW[i] = float32((i*29)%53-20)*0.03 + 1.0
	}
	projW := make([]float32, dOut*dIn)
	for i := range projW {
		projW[i] = float32((i*53)%97-48) * 0.01
	}

	// on-device chain: intermediate never leaves the GPU.
	got, err := native.NormProject(x, normW, projW, dIn, dOut, eps)
	if err != nil {
		t.Fatalf("native.NormProject: %v", err)
	}

	// reference: the same kernels run separately (intermediate round-trips host).
	normed, err := native.RMSNorm(x, normW, 1, dIn, eps)
	if err != nil {
		t.Fatalf("native.RMSNorm: %v", err)
	}
	want, err := native.MatVec(projW, normed, dOut, dIn)
	if err != nil {
		t.Fatalf("native.MatVec: %v", err)
	}

	if len(got) != len(want) {
		t.Fatalf("NormProject length mismatch: chained %d, sequential %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("NormProject chain break at [%d]: chained %v, sequential %v (dIn %d dOut %d)", i, got[i], want[i], dIn, dOut)
		}
	}
}

// TestNativeParity_MulBF16 asserts bit-identical a*b in bf16 (the MLP gate·up).
func TestNativeParity_MulBF16(t *testing.T) {
	if os.Getenv(native.MetallibPathEnv) == "" {
		t.Skipf("%s not set", native.MetallibPathEnv)
	}
	const n = 4096
	a := make([]float32, n)
	b := make([]float32, n)
	for i := range a {
		a[i] = float32((i*37)%101-50) * 0.05
		b[i] = float32((i*29)%53-20) * 0.05
	}
	ab, bb := bf16Bytes(t, a, n), bf16Bytes(t, b, n)
	got, err := native.MulBF16(ab, bb)
	if err != nil {
		t.Fatalf("MulBF16: %v", err)
	}
	aA := metal.FromRawBytes(ab, []int{n}, metal.DTypeBFloat16)
	bA := metal.FromRawBytes(bb, []int{n}, metal.DTypeBFloat16)
	res := metal.Mul(aA, bA)
	metal.Materialize(res)
	want := append([]byte(nil), res.RawBytes()...)
	metal.Free(aA, bA, res)
	assertBytesEqual(t, "MulBF16", got, want, "n 4096")
}

// TestNativeParity_TanhBF16 asserts bit-identical tanh in bf16.
func TestNativeParity_TanhBF16(t *testing.T) {
	if os.Getenv(native.MetallibPathEnv) == "" {
		t.Skipf("%s not set", native.MetallibPathEnv)
	}
	const n = 4096
	x := make([]float32, n)
	for i := range x {
		x[i] = float32((i*37)%160-80) * 0.05
	}
	xb := bf16Bytes(t, x, n)
	got, err := native.TanhBF16(xb)
	if err != nil {
		t.Fatalf("TanhBF16: %v", err)
	}
	xA := metal.FromRawBytes(xb, []int{n}, metal.DTypeBFloat16)
	res := metal.Tanh(xA)
	metal.Materialize(res)
	want := append([]byte(nil), res.RawBytes()...)
	metal.Free(xA, res)
	assertBytesEqual(t, "TanhBF16", got, want, "n 4096")
}

// TestNativeParity_GeluGateMulBF16 checks the bf16 composed GELU gate against
// mlx-c's fused bf16 GELUGateMul. In bf16 each composed step rounds to 8-bit
// mantissa, so this answers whether the separately-run composition matches mlx's
// mlx_compile-fused kernel — the test reports the per-element exact count and
// holds it to a near-exact threshold (a few bf16-ULP allowed for any fusion
// precision difference).
func TestNativeParity_GeluGateMulBF16(t *testing.T) {
	if os.Getenv(native.MetallibPathEnv) == "" {
		t.Skipf("%s not set", native.MetallibPathEnv)
	}
	const n = 4096
	gate := make([]float32, n)
	up := make([]float32, n)
	for i := range gate {
		gate[i] = float32((i*37)%160-80) * 0.1
		up[i] = float32((i*53)%97-48) * 0.04
	}
	gb, ub := bf16Bytes(t, gate, n), bf16Bytes(t, up, n)
	got, err := native.GeluGateMulBF16(gb, ub)
	if err != nil {
		t.Fatalf("GeluGateMulBF16: %v", err)
	}
	gA := metal.FromRawBytes(gb, []int{n}, metal.DTypeBFloat16)
	uA := metal.FromRawBytes(ub, []int{n}, metal.DTypeBFloat16)
	res := metal.GELUGateMul(gA, uA)
	metal.Materialize(res)
	want := append([]byte(nil), res.RawBytes()...)
	metal.Free(gA, uA, res)

	if len(got) != len(want) {
		t.Fatalf("GeluGateMulBF16 length mismatch: %d vs %d", len(got), len(want))
	}
	exact, off := 0, 0
	for i := 0; i+1 < len(want); i += 2 {
		if got[i] == want[i] && got[i+1] == want[i+1] {
			exact++
		} else {
			off++
		}
	}
	total := exact + off
	t.Logf("GeluGateMulBF16 composed vs mlx-c fused: %d/%d bf16 elems exact (%d off)", exact, total, off)
	if off*100 > total*2 { // allow up to 2% to differ by bf16 rounding
		t.Fatalf("GeluGateMulBF16 too many bf16 elems differ: %d/%d off", off, total)
	}
}

// TestNativeICB_NormProject proves ICB replay on a real dependent sequence: the
// rms→gemv ops recorded once into an indirect command buffer and replayed must
// equal the regular on-device NormProject byte-for-byte (same kernels, same data,
// only the submission path differs). This de-risks the encode-bypass machinery —
// scalar-params-as-buffers, SetBarrier between dependent ops, UseResource
// residency — before it scales to the full DecodeLayer.
func TestNativeICB_NormProject(t *testing.T) {
	if os.Getenv(native.MetallibPathEnv) == "" {
		t.Skipf("%s not set — needs the compiled metallib", native.MetallibPathEnv)
	}

	const dIn, dOut = 512, 256
	const eps = float32(1e-5)
	x := make([]float32, dIn)
	for i := range x {
		x[i] = float32((i*37)%101-50) * 0.05
	}
	normW := make([]float32, dIn)
	for i := range normW {
		normW[i] = float32((i*29)%53-20)*0.02 + 1.0
	}
	projW := make([]float32, dOut*dIn)
	for i := range projW {
		projW[i] = float32((i*53)%97-48) * 0.01
	}

	got, err := native.NormProjectICB(x, normW, projW, dIn, dOut, eps, 1)
	if err != nil {
		t.Fatalf("native.NormProjectICB: %v", err)
	}
	want, err := native.NormProject(x, normW, projW, dIn, dOut, eps)
	if err != nil {
		t.Fatalf("native.NormProject: %v", err)
	}

	if len(got) != len(want) {
		t.Fatalf("ICB length mismatch: %d vs %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("ICB NormProject break at [%d]: replay %v, encode %v", i, got[i], want[i])
		}
	}
}

// TestNativeICB_AttentionBlock proves ICB replay across a real func-const
// multi-op chain: the bf16 attention block (rms→gemv→rope→sdpa→gemv→add, with
// the func-const rope/sdpa pipelines built ICB-capable) recorded once and
// replayed must equal the regular AttentionBlock byte-for-byte. This is the
// encode-bypass lever on a real decode sub-block.
func TestNativeICB_AttentionBlock(t *testing.T) {
	if os.Getenv(native.MetallibPathEnv) == "" {
		t.Skipf("%s not set — needs the compiled metallib", native.MetallibPathEnv)
	}

	const dModel, nHeads, nKV, headDim, kvLen = 512, 8, 8, 64, 16
	const base, scale, offset, eps = float32(10000), float32(0.125), 5, float32(1e-5)
	qDim := nHeads * headDim
	mk := func(n, salt int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32((i*salt+7)%101-50) * 0.02
		}
		return s
	}
	x := bf16Bytes(t, mk(dModel, 37), dModel)
	normW := bf16Bytes(t, mk(dModel, 13), dModel)
	wQ := bf16Bytes(t, mk(qDim*dModel, 53), qDim, dModel)
	wO := bf16Bytes(t, mk(dModel*qDim, 17), dModel, qDim)
	kCache := bf16Bytes(t, mk(nKV*kvLen*headDim, 23), nKV, kvLen, headDim)
	vCache := bf16Bytes(t, mk(nKV*kvLen*headDim, 41), nKV, kvLen, headDim)

	got, err := native.AttentionBlockICB(x, normW, wQ, wO, kCache, vCache, dModel, nHeads, nKV, headDim, kvLen, base, scale, offset, eps, 1)
	if err != nil {
		t.Fatalf("native.AttentionBlockICB: %v", err)
	}
	want, err := native.AttentionBlock(x, normW, wQ, wO, kCache, vCache, dModel, nHeads, nKV, headDim, kvLen, base, scale, offset, eps)
	if err != nil {
		t.Fatalf("native.AttentionBlock: %v", err)
	}
	assertBytesEqual(t, "AttentionBlockICB", got, want, fmt.Sprintf("dModel %d nHeads %d kvLen %d", dModel, nHeads, kvLen))
}

// TestNativeICB_DecodeLayer proves ICB replay across the WHOLE 21-op transformer
// layer: DecodeLayerICB records the attention half (rms→gemv→rope→sdpa→gemv→add)
// and the MLP half (rms→gemv×2→9-op gelu→mul→gemv→add) once into an indirect
// command buffer and replays it; replays=1 must equal the regular DecodeLayer
// byte-for-byte. This extends the proven AttentionBlockICB (ops 0-5) to the full
// layer (ops 6-20) — a break past op 5 isolates an MLP-op binding bug. This is
// the full-layer encode-bypass lever on a real decode layer.
func TestNativeICB_DecodeLayer(t *testing.T) {
	if os.Getenv(native.MetallibPathEnv) == "" {
		t.Skipf("%s not set — needs the compiled metallib", native.MetallibPathEnv)
	}

	const dModel, nHeads, nKV, headDim, kvLen, dFF = 512, 8, 8, 64, 16, 1024
	const base, scale, offset, eps = float32(10000), float32(0.125), 5, float32(1e-5)
	qDim := nHeads * headDim

	mk := func(n, salt int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32((i*salt+7)%101-50) * 0.02
		}
		return s
	}
	x := bf16Bytes(t, mk(dModel, 37), dModel)
	attnNormW := bf16Bytes(t, mk(dModel, 13), dModel)
	wQ := bf16Bytes(t, mk(qDim*dModel, 53), qDim, dModel)
	wO := bf16Bytes(t, mk(dModel*qDim, 17), dModel, qDim)
	kCache := bf16Bytes(t, mk(nKV*kvLen*headDim, 23), nKV, kvLen, headDim)
	vCache := bf16Bytes(t, mk(nKV*kvLen*headDim, 41), nKV, kvLen, headDim)
	mlpNormW := bf16Bytes(t, mk(dModel, 19), dModel)
	wGate := bf16Bytes(t, mk(dFF*dModel, 61), dFF, dModel)
	wUp := bf16Bytes(t, mk(dFF*dModel, 29), dFF, dModel)
	wDown := bf16Bytes(t, mk(dModel*dFF, 47), dModel, dFF)

	got, err := native.DecodeLayerICB(x, attnNormW, wQ, wO, kCache, vCache, mlpNormW, wGate, wUp, wDown, dModel, nHeads, nKV, headDim, kvLen, dFF, base, scale, offset, eps, 1)
	if err != nil {
		t.Fatalf("native.DecodeLayerICB: %v", err)
	}
	want, err := native.DecodeLayer(x, attnNormW, wQ, wO, kCache, vCache, mlpNormW, wGate, wUp, wDown, dModel, nHeads, nKV, headDim, kvLen, dFF, base, scale, offset, eps)
	if err != nil {
		t.Fatalf("native.DecodeLayer: %v", err)
	}
	assertBytesEqual(t, "DecodeLayerICB", got, want, fmt.Sprintf("dModel %d dFF %d nHeads %d kvLen %d", dModel, dFF, nHeads, kvLen))
}

// TestNativeChain_DecodeLayer proves a full transformer decode layer runs
// on-device in bf16: DecodeLayer chains the attention block and the MLP block
// (~21 dispatches) in one command buffer, and must equal AttentionBlock followed
// by MLPBlockBF16 run separately. Identical output means the whole layer — both
// residuals, the h hand-off from attention to MLP, the in-line gelu — is correct
// on the no-cgo path. This is a complete transformer layer, natively.
func TestNativeChain_DecodeLayer(t *testing.T) {
	if os.Getenv(native.MetallibPathEnv) == "" {
		t.Skipf("%s not set — needs the compiled metallib", native.MetallibPathEnv)
	}

	const dModel, nHeads, nKV, headDim, kvLen, dFF = 512, 8, 8, 64, 16, 1024
	const base, scale, offset, eps = float32(10000), float32(0.125), 5, float32(1e-5)
	qDim := nHeads * headDim

	mk := func(n, salt int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32((i*salt+7)%101-50) * 0.02
		}
		return s
	}
	x := bf16Bytes(t, mk(dModel, 37), dModel)
	attnNormW := bf16Bytes(t, mk(dModel, 13), dModel)
	wQ := bf16Bytes(t, mk(qDim*dModel, 53), qDim, dModel)
	wO := bf16Bytes(t, mk(dModel*qDim, 17), dModel, qDim)
	kCache := bf16Bytes(t, mk(nKV*kvLen*headDim, 23), nKV, kvLen, headDim)
	vCache := bf16Bytes(t, mk(nKV*kvLen*headDim, 41), nKV, kvLen, headDim)
	mlpNormW := bf16Bytes(t, mk(dModel, 19), dModel)
	wGate := bf16Bytes(t, mk(dFF*dModel, 61), dFF, dModel)
	wUp := bf16Bytes(t, mk(dFF*dModel, 29), dFF, dModel)
	wDown := bf16Bytes(t, mk(dModel*dFF, 47), dModel, dFF)

	// on-device: the whole layer in one command buffer.
	got, err := native.DecodeLayer(x, attnNormW, wQ, wO, kCache, vCache, mlpNormW, wGate, wUp, wDown, dModel, nHeads, nKV, headDim, kvLen, dFF, base, scale, offset, eps)
	if err != nil {
		t.Fatalf("native.DecodeLayer: %v", err)
	}

	// reference: attention block then MLP block, run as separate on-device blocks.
	h, err := native.AttentionBlock(x, attnNormW, wQ, wO, kCache, vCache, dModel, nHeads, nKV, headDim, kvLen, base, scale, offset, eps)
	if err != nil {
		t.Fatalf("AttentionBlock: %v", err)
	}
	want, err := native.MLPBlockBF16(h, mlpNormW, wGate, wUp, wDown, dModel, dFF, eps)
	if err != nil {
		t.Fatalf("MLPBlockBF16: %v", err)
	}

	assertBytesEqual(t, "DecodeLayer", got, want, fmt.Sprintf("dModel %d dFF %d nHeads %d kvLen %d", dModel, dFF, nHeads, kvLen))
}

// TestNativeChain_AttentionBlock proves the attention half of a decode step runs
// on-device in bf16: AttentionBlock chains rms→Qproj→RoPE→SDPA→Oproj→residual in
// one command buffer (every intermediate resident), and must equal the same
// native bf16 ops run separately (each mlx-c parity-green). Identical output means
// the attention wiring — per-head RoPE, single-query SDPA over the cache, the
// residual — is correct on the no-cgo path. Attends over a given KV cache.
func TestNativeChain_AttentionBlock(t *testing.T) {
	if os.Getenv(native.MetallibPathEnv) == "" {
		t.Skipf("%s not set — needs the compiled metallib", native.MetallibPathEnv)
	}

	const dModel, nHeads, nKV, headDim, kvLen = 512, 8, 8, 64, 16
	const base, scale, offset, eps = float32(10000), float32(0.125), 5, float32(1e-5)
	qDim := nHeads * headDim

	mk := func(n, salt int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32((i*salt+7)%101-50) * 0.02
		}
		return s
	}
	x := bf16Bytes(t, mk(dModel, 37), dModel)
	normW := bf16Bytes(t, mk(dModel, 13), dModel)
	wQ := bf16Bytes(t, mk(qDim*dModel, 53), qDim, dModel)
	wO := bf16Bytes(t, mk(dModel*qDim, 17), dModel, qDim)
	kCache := bf16Bytes(t, mk(nKV*kvLen*headDim, 23), nKV, kvLen, headDim)
	vCache := bf16Bytes(t, mk(nKV*kvLen*headDim, 41), nKV, kvLen, headDim)

	// on-device: the whole attention block in one command buffer.
	got, err := native.AttentionBlock(x, normW, wQ, wO, kCache, vCache, dModel, nHeads, nKV, headDim, kvLen, base, scale, offset, eps)
	if err != nil {
		t.Fatalf("native.AttentionBlock: %v", err)
	}

	// reference: the same native bf16 ops run separately (intermediates round-trip).
	normed, err := native.RMSNormBF16(x, normW, 1, dModel, eps)
	if err != nil {
		t.Fatalf("RMSNormBF16: %v", err)
	}
	q, err := native.MatVecBF16(wQ, normed, qDim, dModel)
	if err != nil {
		t.Fatalf("MatVecBF16 q: %v", err)
	}
	qr, err := native.RoPEBF16(q, 1, nHeads, headDim, base, scale, offset, false)
	if err != nil {
		t.Fatalf("RoPEBF16: %v", err)
	}
	attn, err := native.SDPA(qr, kCache, vCache, 1, nHeads, nKV, headDim, kvLen, scale)
	if err != nil {
		t.Fatalf("SDPA: %v", err)
	}
	attnOut, err := native.MatVecBF16(wO, attn, dModel, qDim)
	if err != nil {
		t.Fatalf("MatVecBF16 out: %v", err)
	}
	want, err := native.AddBF16(x, attnOut)
	if err != nil {
		t.Fatalf("AddBF16: %v", err)
	}

	assertBytesEqual(t, "AttentionBlock", got, want, fmt.Sprintf("dModel %d nHeads %d kvLen %d", dModel, nHeads, kvLen))
}

// TestNativeChain_MLPBlock proves a full gemma feed-forward block runs on-device:
// MLPBlock chains ~16 dispatches (norm→gate/up→gelu→mul→down→residual) in one
// command buffer with every intermediate resident, and must equal the same
// native ops run separately (each mlx-c parity-green). Identical output means the
// whole block — including the in-line gelu composition and the residual — is
// correct on the no-cgo path. This is a real decode sub-block.
func TestNativeChain_MLPBlock(t *testing.T) {
	if os.Getenv(native.MetallibPathEnv) == "" {
		t.Skipf("%s not set — needs the compiled metallib", native.MetallibPathEnv)
	}

	const dModel, dFF = 256, 1024
	const eps = float32(1e-5)
	x := make([]float32, dModel)
	for i := range x {
		x[i] = float32((i*37)%101-50) * 0.05
	}
	normW := make([]float32, dModel)
	for i := range normW {
		normW[i] = float32((i*29)%53-20)*0.02 + 1.0
	}
	mkW := func(rows, cols, salt int) []float32 {
		w := make([]float32, rows*cols)
		for i := range w {
			w[i] = float32((i*salt+11)%97-48) * 0.01
		}
		return w
	}
	wGate := mkW(dFF, dModel, 53)
	wUp := mkW(dFF, dModel, 31)
	wDown := mkW(dModel, dFF, 17)

	// on-device: the whole block in one command buffer.
	got, err := native.MLPBlock(x, normW, wGate, wUp, wDown, dModel, dFF, eps)
	if err != nil {
		t.Fatalf("native.MLPBlock: %v", err)
	}

	// reference: the same native ops run separately (intermediates round-trip).
	normed, err := native.RMSNorm(x, normW, 1, dModel, eps)
	if err != nil {
		t.Fatalf("RMSNorm: %v", err)
	}
	gate, err := native.MatVec(wGate, normed, dFF, dModel)
	if err != nil {
		t.Fatalf("MatVec gate: %v", err)
	}
	up, err := native.MatVec(wUp, normed, dFF, dModel)
	if err != nil {
		t.Fatalf("MatVec up: %v", err)
	}
	gated, err := native.GeluGateMul(gate, up)
	if err != nil {
		t.Fatalf("GeluGateMul: %v", err)
	}
	down, err := native.MatVec(wDown, gated, dModel, dFF)
	if err != nil {
		t.Fatalf("MatVec down: %v", err)
	}
	want, err := native.Add(x, down)
	if err != nil {
		t.Fatalf("Add: %v", err)
	}

	if len(got) != len(want) {
		t.Fatalf("MLPBlock length mismatch: chained %d, sequential %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("MLPBlock chain break at [%d]: chained %v, sequential %v (dModel %d dFF %d)", i, got[i], want[i], dModel, dFF)
		}
	}
}

// TestNativeChain_MLPBlockBF16 is the bf16 sibling of TestNativeChain_MLPBlock:
// MLPBlockBF16 chains the same block (norm→gate/up→gelu→mul→down→residual) in
// one command buffer with every intermediate resident, in the decode dtype. It
// must equal the same native bf16 ops run separately, byte-for-byte — the
// in-line gelu uses the same bf16ConstBytes constants as GeluGateMulBF16, so the
// two paths run identical kernels with identical operands and the result is
// exact, not merely within tolerance.
func TestNativeChain_MLPBlockBF16(t *testing.T) {
	if os.Getenv(native.MetallibPathEnv) == "" {
		t.Skipf("%s not set — needs the compiled metallib", native.MetallibPathEnv)
	}

	const dModel, dFF = 256, 1024
	const eps = float32(1e-5)
	xf := make([]float32, dModel)
	for i := range xf {
		xf[i] = float32((i*37)%101-50) * 0.05
	}
	normWf := make([]float32, dModel)
	for i := range normWf {
		normWf[i] = float32((i*29)%53-20)*0.02 + 1.0
	}
	mkW := func(rows, cols, salt int) []float32 {
		w := make([]float32, rows*cols)
		for i := range w {
			w[i] = float32((i*salt+11)%97-48) * 0.01
		}
		return w
	}
	wGatef := mkW(dFF, dModel, 53)
	wUpf := mkW(dFF, dModel, 31)
	wDownf := mkW(dModel, dFF, 17)

	// bf16 fixtures — the dtype the on-device block runs in.
	x := bf16Bytes(t, xf, dModel)
	normW := bf16Bytes(t, normWf, dModel)
	wGate := bf16Bytes(t, wGatef, dFF, dModel)
	wUp := bf16Bytes(t, wUpf, dFF, dModel)
	wDown := bf16Bytes(t, wDownf, dModel, dFF)

	// on-device: the whole block in one command buffer.
	got, err := native.MLPBlockBF16(x, normW, wGate, wUp, wDown, dModel, dFF, eps)
	if err != nil {
		t.Fatalf("native.MLPBlockBF16: %v", err)
	}

	// reference: the same native bf16 ops run separately (intermediates round-trip).
	normed, err := native.RMSNormBF16(x, normW, 1, dModel, eps)
	if err != nil {
		t.Fatalf("RMSNormBF16: %v", err)
	}
	gate, err := native.MatVecBF16(wGate, normed, dFF, dModel)
	if err != nil {
		t.Fatalf("MatVecBF16 gate: %v", err)
	}
	up, err := native.MatVecBF16(wUp, normed, dFF, dModel)
	if err != nil {
		t.Fatalf("MatVecBF16 up: %v", err)
	}
	gated, err := native.GeluGateMulBF16(gate, up)
	if err != nil {
		t.Fatalf("GeluGateMulBF16: %v", err)
	}
	down, err := native.MatVecBF16(wDown, gated, dModel, dFF)
	if err != nil {
		t.Fatalf("MatVecBF16 down: %v", err)
	}
	want, err := native.AddBF16(x, down)
	if err != nil {
		t.Fatalf("AddBF16: %v", err)
	}

	assertBytesEqual(t, "MLPBlockBF16", got, want, fmt.Sprintf("dModel %d dFF %d", dModel, dFF))
}
