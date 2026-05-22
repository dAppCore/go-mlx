// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

// RMSNorm bench coverage map (W7-E, Wave 7).
//
// Gemma 3 / Gemma 4 apply RMSNorm 4× per transformer block (not 2× as
// in standard LLaMA-style): pre-attention, post-attention, pre-FFN,
// post-FFN. With zero-centered weights, the kernel must apply
// (1 + weight) scaling — see precomputeGemma4ScaledWeights in
// gemma4.go which pre-bakes the (1+w) factor at model load to avoid
// per-call add cost.
//
// Coverage:
//   - Single RMSNorm at decode shape (1 token × hidden).
//   - Single RMSNorm at prefill shape (L × hidden).
//   - Per-block 4× pattern at decode + prefill — gives the per-layer
//     cost direct from the bench rather than back-calculated.
//   - RMSNormNoScale — the variant that skips the weight multiply
//     entirely (used in attention path where the norm weight is
//     pre-folded into the projection).
//   - Hidden-size sweep matching realistic configs: 1024 (Gemma 4
//     E2B), 2048 (mid-size), 3072 (Gemma 4 E4B).

import "testing"

// --- Decode shape (single token) ---

func BenchmarkRMSNorm_Decode_Hidden1024(b *testing.B) {
	x := RandomUniform(0, 1, []int32{1, 1024}, DTypeFloat32)
	w := RandomUniform(0, 1, []int32{1024}, DTypeFloat32)
	defer Free(x, w)
	Materialize(x, w)
	b.SetBytes(int64(1024 * 4))
	b.ReportAllocs()
	for b.Loop() {
		y := RMSNorm(x, w, 1e-6)
		Materialize(y)
		Free(y)
	}
}

func BenchmarkRMSNorm_Decode_Hidden2048(b *testing.B) {
	x := RandomUniform(0, 1, []int32{1, 2048}, DTypeFloat32)
	w := RandomUniform(0, 1, []int32{2048}, DTypeFloat32)
	defer Free(x, w)
	Materialize(x, w)
	b.SetBytes(int64(2048 * 4))
	b.ReportAllocs()
	for b.Loop() {
		y := RMSNorm(x, w, 1e-6)
		Materialize(y)
		Free(y)
	}
}

func BenchmarkRMSNorm_Decode_Hidden3072(b *testing.B) {
	x := RandomUniform(0, 1, []int32{1, 3072}, DTypeFloat32)
	w := RandomUniform(0, 1, []int32{3072}, DTypeFloat32)
	defer Free(x, w)
	Materialize(x, w)
	b.SetBytes(int64(3072 * 4))
	b.ReportAllocs()
	for b.Loop() {
		y := RMSNorm(x, w, 1e-6)
		Materialize(y)
		Free(y)
	}
}

// --- Prefill shape (sequence × hidden) ---

func BenchmarkRMSNorm_Prefill_512_Hidden2048(b *testing.B) {
	x := RandomUniform(0, 1, []int32{512, 2048}, DTypeFloat32)
	w := RandomUniform(0, 1, []int32{2048}, DTypeFloat32)
	defer Free(x, w)
	Materialize(x, w)
	b.SetBytes(int64(512 * 2048 * 4))
	b.ReportAllocs()
	for b.Loop() {
		y := RMSNorm(x, w, 1e-6)
		Materialize(y)
		Free(y)
	}
}

func BenchmarkRMSNorm_Prefill_4096_Hidden2048(b *testing.B) {
	x := RandomUniform(0, 1, []int32{4096, 2048}, DTypeFloat32)
	w := RandomUniform(0, 1, []int32{2048}, DTypeFloat32)
	defer Free(x, w)
	Materialize(x, w)
	b.SetBytes(int64(4096 * 2048 * 4))
	b.ReportAllocs()
	for b.Loop() {
		y := RMSNorm(x, w, 1e-6)
		Materialize(y)
		Free(y)
	}
}

// --- 4× per-block pattern at decode ---

// One block = pre-attn-norm + post-attn-norm + pre-ffn-norm +
// post-ffn-norm. Bench the full sequence to get the per-block cost
// directly (instead of × 4ing the single-norm number).
func BenchmarkRMSNorm_BlockPattern4x_Decode_Hidden2048(b *testing.B) {
	x := RandomUniform(0, 1, []int32{1, 2048}, DTypeFloat32)
	w1 := RandomUniform(0, 1, []int32{2048}, DTypeFloat32)
	w2 := RandomUniform(0, 1, []int32{2048}, DTypeFloat32)
	w3 := RandomUniform(0, 1, []int32{2048}, DTypeFloat32)
	w4 := RandomUniform(0, 1, []int32{2048}, DTypeFloat32)
	defer Free(x, w1, w2, w3, w4)
	Materialize(x, w1, w2, w3, w4)
	b.SetBytes(int64(4 * 2048 * 4))
	b.ReportAllocs()
	for b.Loop() {
		y1 := RMSNorm(x, w1, 1e-6)
		y2 := RMSNorm(y1, w2, 1e-6)
		y3 := RMSNorm(y2, w3, 1e-6)
		y4 := RMSNorm(y3, w4, 1e-6)
		Materialize(y4)
		Free(y1, y2, y3, y4)
	}
}

// 4× pattern at prefill — 4k context.
func BenchmarkRMSNorm_BlockPattern4x_Prefill_4096_Hidden2048(b *testing.B) {
	x := RandomUniform(0, 1, []int32{4096, 2048}, DTypeFloat32)
	w1 := RandomUniform(0, 1, []int32{2048}, DTypeFloat32)
	w2 := RandomUniform(0, 1, []int32{2048}, DTypeFloat32)
	w3 := RandomUniform(0, 1, []int32{2048}, DTypeFloat32)
	w4 := RandomUniform(0, 1, []int32{2048}, DTypeFloat32)
	defer Free(x, w1, w2, w3, w4)
	Materialize(x, w1, w2, w3, w4)
	b.SetBytes(int64(4 * 4096 * 2048 * 4))
	b.ReportAllocs()
	for b.Loop() {
		y1 := RMSNorm(x, w1, 1e-6)
		y2 := RMSNorm(y1, w2, 1e-6)
		y3 := RMSNorm(y2, w3, 1e-6)
		y4 := RMSNorm(y3, w4, 1e-6)
		Materialize(y4)
		Free(y1, y2, y3, y4)
	}
}

// --- RMSNormNoScale (weight-less norm) ---

// The QK-norm path in Gemma 4 attention uses pre-folded weights and
// calls RMSNormNoScale. The cost should be lower than full RMSNorm
// by the weight-multiply step.
func BenchmarkRMSNormNoScale_Decode_Hidden2048(b *testing.B) {
	x := RandomUniform(0, 1, []int32{1, 2048}, DTypeFloat32)
	defer Free(x)
	Materialize(x)
	b.SetBytes(int64(2048 * 4))
	b.ReportAllocs()
	for b.Loop() {
		y := RMSNormNoScale(x, 1e-6)
		Materialize(y)
		Free(y)
	}
}

func BenchmarkRMSNormNoScale_Prefill_4096_Hidden2048(b *testing.B) {
	x := RandomUniform(0, 1, []int32{4096, 2048}, DTypeFloat32)
	defer Free(x)
	Materialize(x)
	b.SetBytes(int64(4096 * 2048 * 4))
	b.ReportAllocs()
	for b.Loop() {
		y := RMSNormNoScale(x, 1e-6)
		Materialize(y)
		Free(y)
	}
}

// --- QK-norm shape (per-head norm) ---

// Gemma 4 attention applies QNorm/KNorm per-head over the D dimension.
// Shape: [B=1, H=8, L=1, D=128] — the per-head decode-step norm cost.
// (Note: RMSNorm operates on the last axis, so this reduces over D.)
func BenchmarkRMSNorm_QKNorm_Decode_8heads_D128(b *testing.B) {
	x := RandomUniform(0, 1, []int32{1, 8, 1, 128}, DTypeFloat32)
	w := RandomUniform(0, 1, []int32{128}, DTypeFloat32)
	defer Free(x, w)
	Materialize(x, w)
	b.SetBytes(int64(8 * 128 * 4))
	b.ReportAllocs()
	for b.Loop() {
		y := RMSNorm(x, w, 1e-6)
		Materialize(y)
		Free(y)
	}
}

// QK-norm at prefill shape [B=1, H=8, L=512, D=128].
func BenchmarkRMSNorm_QKNorm_Prefill_8heads_seq512_D128(b *testing.B) {
	x := RandomUniform(0, 1, []int32{1, 8, 512, 128}, DTypeFloat32)
	w := RandomUniform(0, 1, []int32{128}, DTypeFloat32)
	defer Free(x, w)
	Materialize(x, w)
	b.SetBytes(int64(8 * 512 * 128 * 4))
	b.ReportAllocs()
	for b.Loop() {
		y := RMSNorm(x, w, 1e-6)
		Materialize(y)
		Free(y)
	}
}

// --- Zero-centered weight scaling pattern ---

// Note: Gemma 4 zero-centered weights are pre-folded at model load
// (see precomputeGemma4ScaledWeights). This bench measures the
// hypothetical un-baked path: if the (1+w) compute were per-call,
// it'd cost an extra AddScalar before each RMSNorm. We bench the
// add+norm together for comparison against the baked single-call.
func BenchmarkRMSNorm_ZeroCenteredAddThenNorm_Decode(b *testing.B) {
	x := RandomUniform(0, 1, []int32{1, 2048}, DTypeFloat32)
	w := RandomUniform(0, 1, []int32{2048}, DTypeFloat32)
	defer Free(x, w)
	Materialize(x, w)
	b.SetBytes(int64(2048 * 4))
	b.ReportAllocs()
	for b.Loop() {
		w1 := AddScalar(w, 1.0)
		y := RMSNorm(x, w1, 1e-6)
		Materialize(y)
		Free(w1, y)
	}
}

// --- Eps variation (1e-5 vs 1e-6) ---

// Eps shouldn't affect cost, but bench it so a regression here flags
// kernel-variant divergence.
func BenchmarkRMSNorm_Eps_1e5(b *testing.B) {
	x := RandomUniform(0, 1, []int32{1, 2048}, DTypeFloat32)
	w := RandomUniform(0, 1, []int32{2048}, DTypeFloat32)
	defer Free(x, w)
	Materialize(x, w)
	b.ReportAllocs()
	for b.Loop() {
		y := RMSNorm(x, w, 1e-5)
		Materialize(y)
		Free(y)
	}
}

func BenchmarkRMSNorm_Eps_1e6(b *testing.B) {
	x := RandomUniform(0, 1, []int32{1, 2048}, DTypeFloat32)
	w := RandomUniform(0, 1, []int32{2048}, DTypeFloat32)
	defer Free(x, w)
	Materialize(x, w)
	b.ReportAllocs()
	for b.Loop() {
		y := RMSNorm(x, w, 1e-6)
		Materialize(y)
		Free(y)
	}
}
