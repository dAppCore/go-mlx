// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

// RoPE bench coverage map (W7-E, Wave 7).
//
// Gemma 3 / Gemma 4 use dual RoPE frequencies depending on attention
// layer type:
//
//   Local layers:  base = 10,000      scale = 1.0
//   Global layers: base = 1,000,000   scale = 8.0 (Gemma 3)
//   Gemma 4:       global path uses Proportional RoPE (p-RoPE) with an
//                  explicit frequency tensor — RoPEWithFreqs.
//
// These benches cover:
//   - Plain RoPE (no explicit freqs) at decode + prefill shapes.
//   - Local-base vs global-base scaling cost — same shape, different
//     base; the cost differential should be ~0 since base only affects
//     the kernel's frequency table generation, not the inner loop.
//   - RoPEWithFreqs for the p-RoPE path — passes the precomputed
//     freq table the Gemma 4 layer uses.
//   - RoPEWithOffsetArray for the per-token dynamic-offset path used
//     by FixedKVCache sliding-window decode (offset is an array, not
//     a scalar).

import "testing"

// Gemma 4 head dim is typically 256 (global) and 128 (local). The
// rotated_dim parameter to RoPE is often headDim or 0.5×headDim
// depending on the rope_section split.

// --- Plain RoPE — single-token decode shapes ---

// Decode: [B=1, H=1, L=32, D=128] — single-position decode in the
// micro-bench style. (Existing bench_test.go has this; we extend with
// gemma4-specific shapes below.)
func BenchmarkRoPE_Local_Decode_32heads_seq1_D128(b *testing.B) {
	// One token per head — typical decode where L=1 across H=32 heads.
	x := RandomUniform(0, 1, []int32{1, 32, 1, 128}, DTypeFloat32)
	Materialize(x)
	b.ReportAllocs()
	for b.Loop() {
		y := RoPE(x, 128, false, 10000.0, 1.0, 0)
		Materialize(y)
		Free(y)
	}
}

func BenchmarkRoPE_Global_Decode_32heads_seq1_D256(b *testing.B) {
	x := RandomUniform(0, 1, []int32{1, 32, 1, 256}, DTypeFloat32)
	Materialize(x)
	b.ReportAllocs()
	for b.Loop() {
		// Global base = 1M, scale = 8 (Gemma 3 / pre-pRoPE Gemma 4).
		y := RoPE(x, 256, false, 1000000.0, 8.0, 0)
		Materialize(y)
		Free(y)
	}
}

// Same shape, different base — confirm bench surfaces base-cost is ~0.
func BenchmarkRoPE_Decode_BaseLocal10k(b *testing.B) {
	x := RandomUniform(0, 1, []int32{1, 8, 1, 128}, DTypeFloat32)
	Materialize(x)
	b.ReportAllocs()
	for b.Loop() {
		y := RoPE(x, 128, false, 10000.0, 1.0, 0)
		Materialize(y)
		Free(y)
	}
}

func BenchmarkRoPE_Decode_BaseLocal10k_WithFreqs(b *testing.B) {
	x := RandomUniform(0, 1, []int32{1, 8, 1, 128}, DTypeFloat32)
	freqs := gemma4ProportionalFreqs(128, 128, 10000.0, 1.0)
	defer Free(x, freqs)
	Materialize(x, freqs)
	b.ReportAllocs()
	for b.Loop() {
		y := RoPEWithFreqs(x, 128, false, 0, 1.0, 0, freqs)
		Materialize(y)
		Free(y)
	}
}

func BenchmarkRoPE_Decode_BaseGlobal1M(b *testing.B) {
	x := RandomUniform(0, 1, []int32{1, 8, 1, 128}, DTypeFloat32)
	Materialize(x)
	b.ReportAllocs()
	for b.Loop() {
		y := RoPE(x, 128, false, 1000000.0, 8.0, 0)
		Materialize(y)
		Free(y)
	}
}

// --- Plain RoPE — prefill shapes ---

func BenchmarkRoPE_Local_Prefill_8heads_seq512_D128(b *testing.B) {
	x := RandomUniform(0, 1, []int32{1, 8, 512, 128}, DTypeFloat32)
	Materialize(x)
	b.ReportAllocs()
	for b.Loop() {
		y := RoPE(x, 128, false, 10000.0, 1.0, 0)
		Materialize(y)
		Free(y)
	}
}

func BenchmarkRoPE_Global_Prefill_4heads_seq4096_D256(b *testing.B) {
	x := RandomUniform(0, 1, []int32{1, 4, 4096, 256}, DTypeFloat32)
	Materialize(x)
	b.ReportAllocs()
	for b.Loop() {
		y := RoPE(x, 256, false, 1000000.0, 8.0, 0)
		Materialize(y)
		Free(y)
	}
}

// 16k long-context prefill — the curve point where IDEAS.md flagged
// the dual-RoPE quirk matters.
func BenchmarkRoPE_Global_Prefill_4heads_seq16384_D256(b *testing.B) {
	x := RandomUniform(0, 1, []int32{1, 4, 16384, 256}, DTypeFloat32)
	Materialize(x)
	b.ReportAllocs()
	for b.Loop() {
		y := RoPE(x, 256, false, 1000000.0, 8.0, 0)
		Materialize(y)
		Free(y)
	}
}

// --- Offset variation — decode at long context ---

// Offset is what RoPE reads to phase-shift the rotation per cached
// token. At offset=8k the kernel should consume the same time as
// offset=0 if the rotation table is precomputed; if not, this surfaces
// the linear scan cost.
func BenchmarkRoPE_Decode_OffsetZero(b *testing.B) {
	x := RandomUniform(0, 1, []int32{1, 8, 1, 128}, DTypeFloat32)
	Materialize(x)
	b.ReportAllocs()
	for b.Loop() {
		y := RoPE(x, 128, false, 10000.0, 1.0, 0)
		Materialize(y)
		Free(y)
	}
}

func BenchmarkRoPE_Decode_Offset4k(b *testing.B) {
	x := RandomUniform(0, 1, []int32{1, 8, 1, 128}, DTypeFloat32)
	Materialize(x)
	b.ReportAllocs()
	for b.Loop() {
		y := RoPE(x, 128, false, 10000.0, 1.0, 4096)
		Materialize(y)
		Free(y)
	}
}

func BenchmarkRoPE_Decode_Offset32k(b *testing.B) {
	x := RandomUniform(0, 1, []int32{1, 8, 1, 128}, DTypeFloat32)
	Materialize(x)
	b.ReportAllocs()
	for b.Loop() {
		y := RoPE(x, 128, false, 10000.0, 1.0, 32768)
		Materialize(y)
		Free(y)
	}
}

// --- Traditional rotation order ---

// The `traditional` flag changes the rotation layout (LLaMA-style
// pairs vs Gemma-style halves). Bench both at matched shape so the
// kernel-variant cost is visible.
func BenchmarkRoPE_TraditionalOrder_Decode(b *testing.B) {
	x := RandomUniform(0, 1, []int32{1, 8, 1, 128}, DTypeFloat32)
	Materialize(x)
	b.ReportAllocs()
	for b.Loop() {
		y := RoPE(x, 128, true, 10000.0, 1.0, 0)
		Materialize(y)
		Free(y)
	}
}

func BenchmarkRoPE_HalvesOrder_Decode(b *testing.B) {
	x := RandomUniform(0, 1, []int32{1, 8, 1, 128}, DTypeFloat32)
	Materialize(x)
	b.ReportAllocs()
	for b.Loop() {
		y := RoPE(x, 128, false, 10000.0, 1.0, 0)
		Materialize(y)
		Free(y)
	}
}

// --- RoPEWithFreqs — explicit frequency table (p-RoPE) ---

// Gemma 4 global p-RoPE precomputes a frequency tensor and passes it
// per call. Reuses the same table across decode iterations, so the
// table allocation isn't a per-call cost. We pre-build it once.
func BenchmarkRoPE_WithFreqs_Decode_D256(b *testing.B) {
	x := RandomUniform(0, 1, []int32{1, 4, 1, 256}, DTypeFloat32)
	freqs := gemma4ProportionalFreqs(256, 256, 1000000.0, 8.0)
	defer Free(x, freqs)
	Materialize(x, freqs)
	b.ReportAllocs()
	for b.Loop() {
		y := RoPEWithFreqs(x, 256, false, 1000000.0, 1.0, 0, freqs)
		Materialize(y)
		Free(y)
	}
}

func BenchmarkRoPE_WithFreqs_Prefill_4k_D256(b *testing.B) {
	x := RandomUniform(0, 1, []int32{1, 4, 4096, 256}, DTypeFloat32)
	freqs := gemma4ProportionalFreqs(256, 256, 1000000.0, 8.0)
	defer Free(x, freqs)
	Materialize(x, freqs)
	b.ReportAllocs()
	for b.Loop() {
		y := RoPEWithFreqs(x, 256, false, 1000000.0, 1.0, 0, freqs)
		Materialize(y)
		Free(y)
	}
}

// --- RoPEWithOffsetArray — dynamic-offset path ---

// FixedKVCache sliding-window decode passes the offset as an array so
// the kernel can dispatch all per-cache positions in one launch
// without a Go-side scalar marshal.
func BenchmarkRoPE_WithOffsetArray_Decode_D128(b *testing.B) {
	x := RandomUniform(0, 1, []int32{1, 8, 1, 128}, DTypeFloat32)
	offsetArr := FromValues([]int32{4096}, 1)
	defer Free(x, offsetArr)
	Materialize(x, offsetArr)
	b.ReportAllocs()
	for b.Loop() {
		y := RoPEWithOffsetArray(x, 128, false, 10000.0, 1.0, offsetArr, nil)
		Materialize(y)
		Free(y)
	}
}

// --- gemma4ProportionalFreqs — table construction cost ---

// Built once at model load; if a path is recomputing per call, we
// want to see it here.
func BenchmarkRoPE_BuildProportionalFreqs_D256(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		freqs := gemma4ProportionalFreqs(256, 256, 1000000.0, 8.0)
		Materialize(freqs)
		Free(freqs)
	}
}

func BenchmarkRoPE_BuildProportionalFreqs_D128(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		freqs := gemma4ProportionalFreqs(128, 128, 1000000.0, 8.0)
		Materialize(freqs)
		Free(freqs)
	}
}
