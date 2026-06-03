// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

// FFN / MLP bench coverage map (W7-E, Wave 7).
//
// Gemma 3 MLP is gate_proj + up_proj + GELU gate-mul + down_proj.
// Gemma 4 retains the same shape but adds the gemma4_ffn_residual
// fused path (gemma4_ffn_residual.go) for the local-expert residual
// merge.
//
// Coverage:
//   - GELUGateMul vs SiLUGateMul vs unfused gate+mul — the activation
//     differential.
//   - MLP forward at decode (1 × hidden → intermediate → hidden) vs
//     prefill (L × hidden) for typical Gemma-4 E2B (1024) and E4B
//     (3072) sizes.
//   - nativeGemma4FFNResidual at the eps=1e-6 path it's registered for
//     — confirms the fused kernel is reachable and benches its
//     bandwidth-bound cost.
//   - Tanh + Mul + AddScalar primitives that compose geluApprox — if
//     the native path isn't enabled, these are the fallback.
//
// Hidden / intermediate ratios:
//   E2B:  hidden=1024, intermediate=8192 (8× expand)
//   E4B:  hidden=3072, intermediate=24576 (8× expand)
//   Test sizes are scaled for benchability — full E4B would dominate
//   the run, so we use proportional shapes.

import "testing"

// --- GELUGateMul (Gemma activation) ---

func BenchmarkFFN_GELUGateMul_Decode_Intermediate8192(b *testing.B) {
	gate := RandomUniform(-3, 3, []int32{1, 8192}, DTypeFloat32)
	up := RandomUniform(-3, 3, []int32{1, 8192}, DTypeFloat32)
	defer Free(gate, up)
	Materialize(gate, up)
	b.SetBytes(int64(2 * 8192 * 4))
	b.ReportAllocs()
	for b.Loop() {
		y := GELUGateMul(gate, up)
		Materialize(y)
		Free(y)
	}
}

func BenchmarkFFN_GELUGateMul_Prefill_4096_Intermediate8192(b *testing.B) {
	gate := RandomUniform(-3, 3, []int32{4096, 8192}, DTypeFloat32)
	up := RandomUniform(-3, 3, []int32{4096, 8192}, DTypeFloat32)
	defer Free(gate, up)
	Materialize(gate, up)
	b.SetBytes(int64(2 * 4096 * 8192 * 4))
	b.ReportAllocs()
	for b.Loop() {
		y := GELUGateMul(gate, up)
		Materialize(y)
		Free(y)
	}
}

// SiLU variant — LLaMA-family activation. Bench it for comparison.
func BenchmarkFFN_SiLUGateMul_Decode_Intermediate8192(b *testing.B) {
	gate := RandomUniform(-3, 3, []int32{1, 8192}, DTypeFloat32)
	up := RandomUniform(-3, 3, []int32{1, 8192}, DTypeFloat32)
	defer Free(gate, up)
	Materialize(gate, up)
	b.SetBytes(int64(2 * 8192 * 4))
	b.ReportAllocs()
	for b.Loop() {
		y := SiLUGateMul(gate, up)
		Materialize(y)
		Free(y)
	}
}

func BenchmarkFFN_SiLUGateMul_Prefill_512_Intermediate8192(b *testing.B) {
	gate := RandomUniform(-3, 3, []int32{512, 8192}, DTypeFloat32)
	up := RandomUniform(-3, 3, []int32{512, 8192}, DTypeFloat32)
	defer Free(gate, up)
	Materialize(gate, up)
	b.SetBytes(int64(2 * 512 * 8192 * 4))
	b.ReportAllocs()
	for b.Loop() {
		y := SiLUGateMul(gate, up)
		Materialize(y)
		Free(y)
	}
}

// --- Activation primitives (SiLU/Tanh) ---

func BenchmarkFFN_SiLU_Decode_8192(b *testing.B) {
	x := RandomUniform(-3, 3, []int32{1, 8192}, DTypeFloat32)
	defer Free(x)
	Materialize(x)
	b.SetBytes(int64(8192 * 4))
	b.ReportAllocs()
	for b.Loop() {
		y := SiLU(x)
		Materialize(y)
		Free(y)
	}
}

func BenchmarkFFN_Tanh_Decode_8192(b *testing.B) {
	x := RandomUniform(-3, 3, []int32{1, 8192}, DTypeFloat32)
	defer Free(x)
	Materialize(x)
	b.SetBytes(int64(8192 * 4))
	b.ReportAllocs()
	for b.Loop() {
		y := Tanh(x)
		Materialize(y)
		Free(y)
	}
}

// --- MLP forward (composed projection pattern) ---

// Build the MLP triple by hand and bench (m *MLP).forward through the
// internal entry — this exercises the native-MLP-matvec fast path if
// the runtime gate enables it.
func BenchmarkFFN_MLPForward_Decode_H1024_I8192(b *testing.B) {
	const H, I = 1024, 8192
	gateW := RandomUniform(-0.1, 0.1, []int32{I, H}, DTypeFloat32)
	upW := RandomUniform(-0.1, 0.1, []int32{I, H}, DTypeFloat32)
	downW := RandomUniform(-0.1, 0.1, []int32{H, I}, DTypeFloat32)
	defer Free(gateW, upW, downW)
	Materialize(gateW, upW, downW)

	mlp := &MLP{
		GateProj: NewLinear(gateW, nil),
		UpProj:   NewLinear(upW, nil),
		DownProj: NewLinear(downW, nil),
	}

	x := RandomUniform(-1, 1, []int32{1, H}, DTypeFloat32)
	defer Free(x)
	Materialize(x)
	b.SetBytes(int64(H * 4))
	b.ReportAllocs()
	for b.Loop() {
		y := mlp.forward(x)
		Materialize(y)
		Free(y)
	}
}

func BenchmarkFFN_MLPForward_Decode_H2048_I8192(b *testing.B) {
	const H, I = 2048, 8192
	gateW := RandomUniform(-0.1, 0.1, []int32{I, H}, DTypeFloat32)
	upW := RandomUniform(-0.1, 0.1, []int32{I, H}, DTypeFloat32)
	downW := RandomUniform(-0.1, 0.1, []int32{H, I}, DTypeFloat32)
	defer Free(gateW, upW, downW)
	Materialize(gateW, upW, downW)

	mlp := &MLP{
		GateProj: NewLinear(gateW, nil),
		UpProj:   NewLinear(upW, nil),
		DownProj: NewLinear(downW, nil),
	}

	x := RandomUniform(-1, 1, []int32{1, H}, DTypeFloat32)
	defer Free(x)
	Materialize(x)
	b.SetBytes(int64(H * 4))
	b.ReportAllocs()
	for b.Loop() {
		y := mlp.forward(x)
		Materialize(y)
		Free(y)
	}
}

func BenchmarkFFN_MLPForward_Prefill_512_H1024_I8192(b *testing.B) {
	const H, I = 1024, 8192
	gateW := RandomUniform(-0.1, 0.1, []int32{I, H}, DTypeFloat32)
	upW := RandomUniform(-0.1, 0.1, []int32{I, H}, DTypeFloat32)
	downW := RandomUniform(-0.1, 0.1, []int32{H, I}, DTypeFloat32)
	defer Free(gateW, upW, downW)
	Materialize(gateW, upW, downW)

	mlp := &MLP{
		GateProj: NewLinear(gateW, nil),
		UpProj:   NewLinear(upW, nil),
		DownProj: NewLinear(downW, nil),
	}

	x := RandomUniform(-1, 1, []int32{512, H}, DTypeFloat32)
	defer Free(x)
	Materialize(x)
	b.SetBytes(int64(512 * H * 4))
	b.ReportAllocs()
	for b.Loop() {
		y := mlp.forward(x)
		Materialize(y)
		Free(y)
	}
}

// --- nativeGemma4FFNResidual — fused local+expert residual merge ---

// The fused kernel takes residual + local + expert + 3× norm tensors
// and produces a single output. It's the workload Gemma 4 MoE layers
// run during the final residual merge.
//
// The kernel has a hard eps requirement of 1e-6 (see
// validateNativeGemma4FFNResidual). The shape must be [1, 1, hidden].
//
// We use hidden=2048 — realistic Gemma 4 E4B-scale shape.
func BenchmarkFFN_NativeGemma4FFNResidual_Hidden2048(b *testing.B) {
	const hidden = 2048
	residual := RandomUniform(-1, 1, []int32{1, 1, hidden}, DTypeFloat32)
	local := RandomUniform(-1, 1, []int32{1, 1, hidden}, DTypeFloat32)
	expert := RandomUniform(-1, 1, []int32{1, 1, hidden}, DTypeFloat32)
	localNorm := RandomUniform(0, 1, []int32{hidden}, DTypeFloat32)
	expertNorm := RandomUniform(0, 1, []int32{hidden}, DTypeFloat32)
	combinedNorm := RandomUniform(0, 1, []int32{hidden}, DTypeFloat32)
	defer Free(residual, local, expert, localNorm, expertNorm, combinedNorm)
	Materialize(residual, local, expert, localNorm, expertNorm, combinedNorm)

	b.SetBytes(int64(6 * hidden * 4))
	b.ReportAllocs()
	for b.Loop() {
		out, ok, err := nativeGemma4FFNResidual(residual, local, expert, localNorm, expertNorm, combinedNorm, 1e-6)
		if err != nil {
			b.Fatalf("nativeGemma4FFNResidual: %v", err)
		}
		if !ok {
			// Native path not available — skip the rest of this run rather
			// than asserting. The runtime gate may be off.
			b.Skip("nativeGemma4FFNResidual not available in this runtime")
		}
		Materialize(out)
		Free(out)
	}
}

func BenchmarkFFN_NativeGemma4FFNResidual_Hidden1024(b *testing.B) {
	const hidden = 1024
	residual := RandomUniform(-1, 1, []int32{1, 1, hidden}, DTypeFloat32)
	local := RandomUniform(-1, 1, []int32{1, 1, hidden}, DTypeFloat32)
	expert := RandomUniform(-1, 1, []int32{1, 1, hidden}, DTypeFloat32)
	localNorm := RandomUniform(0, 1, []int32{hidden}, DTypeFloat32)
	expertNorm := RandomUniform(0, 1, []int32{hidden}, DTypeFloat32)
	combinedNorm := RandomUniform(0, 1, []int32{hidden}, DTypeFloat32)
	defer Free(residual, local, expert, localNorm, expertNorm, combinedNorm)
	Materialize(residual, local, expert, localNorm, expertNorm, combinedNorm)

	b.SetBytes(int64(6 * hidden * 4))
	b.ReportAllocs()
	for b.Loop() {
		out, ok, err := nativeGemma4FFNResidual(residual, local, expert, localNorm, expertNorm, combinedNorm, 1e-6)
		if err != nil {
			b.Fatalf("nativeGemma4FFNResidual: %v", err)
		}
		if !ok {
			b.Skip("nativeGemma4FFNResidual not available in this runtime")
		}
		Materialize(out)
		Free(out)
	}
}

// --- Add/Mul primitives at FFN scale ---

// Element-wise Add — the residual-add step at hidden scale.
func BenchmarkFFN_ResidualAdd_Decode_Hidden2048(b *testing.B) {
	a := RandomUniform(-1, 1, []int32{1, 2048}, DTypeFloat32)
	c := RandomUniform(-1, 1, []int32{1, 2048}, DTypeFloat32)
	defer Free(a, c)
	Materialize(a, c)
	b.SetBytes(int64(2 * 2048 * 4))
	b.ReportAllocs()
	for b.Loop() {
		y := Add(a, c)
		Materialize(y)
		Free(y)
	}
}

// Mul at intermediate scale (FFN gate-mul fallback if GELUGateMul is
// disabled).
func BenchmarkFFN_GateMulOnly_Decode_Intermediate8192(b *testing.B) {
	a := RandomUniform(-1, 1, []int32{1, 8192}, DTypeFloat32)
	c := RandomUniform(-1, 1, []int32{1, 8192}, DTypeFloat32)
	defer Free(a, c)
	Materialize(a, c)
	b.SetBytes(int64(2 * 8192 * 4))
	b.ReportAllocs()
	for b.Loop() {
		y := Mul(a, c)
		Materialize(y)
		Free(y)
	}
}
