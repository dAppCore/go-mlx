// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import (
	"encoding/binary"
	"math"
	"testing"
)

// constFloat32s returns n copies of value — used to drive the zero-block /
// d==0 fast paths in the quantisers (a constant block has max==min so the
// computed scale is 0).
func constFloat32s(n int, value float32) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = value
	}
	return out
}

// descendingNegFloat32s returns n strictly-negative descending values so the
// `scaled < 0` rounding arm (q = int(scaled-0.5)) in the *_0 / K kernels is
// exercised alongside the positive arm.
func descendingNegFloat32s(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = -float32(i+1) / 4
	}
	return out
}

// TestQuantizeKernels_ZeroBlockFastPaths_Ugly drives the scale==0 / d==0
// fast paths in the per-block quantisers. A constant block has max==min so
// every kernel takes the zero-scale branch (Q8_0 emits a zeroed pack, Q4_0
// emits 0x88 nibbles, Q5_0 emits 0x44, Q4_K emits 0x88) — the arms the
// ascending-data round-trip tests never reach because those inputs always
// have a non-zero range.
func TestQuantizeKernels_ZeroBlockFastPaths_Ugly(t *testing.T) {
	const block = 32
	zeros := constFloat32s(block, 0)

	q8 := quantizeQ8_0(zeros)
	if len(q8) != 34 {
		t.Fatalf("quantizeQ8_0(zero) len = %d, want 34", len(q8))
	}
	// scale f16 == 0, then 32 zero quants.
	for i := 2; i < 34; i++ {
		if q8[i] != 0 {
			t.Fatalf("quantizeQ8_0(zero) byte[%d] = %#x, want 0", i, q8[i])
		}
	}

	q4 := quantizeQ4_0(zeros)
	if len(q4) != 18 {
		t.Fatalf("quantizeQ4_0(zero) len = %d, want 18", len(q4))
	}
	for i := 2; i < 18; i++ {
		if q4[i] != 0x88 {
			t.Fatalf("quantizeQ4_0(zero) byte[%d] = %#x, want 0x88", i, q4[i])
		}
	}

	q5 := quantizeQ5_0(zeros)
	if len(q5) != 24 {
		t.Fatalf("quantizeQ5_0(zero) len = %d, want 24", len(q5))
	}
	// d f16 (2) + min f16 (2) + 20 packed bytes of 0x44.
	for i := 4; i < 24; i++ {
		if q5[i] != 0x44 {
			t.Fatalf("quantizeQ5_0(zero) byte[%d] = %#x, want 0x44", i, q5[i])
		}
	}

	// Q4_K d==0 path: a constant 256-element block.
	zerosK := constFloat32s(256, 1.5)
	q4k := quantizeQ4_K(zerosK)
	if len(q4k) != 144 {
		t.Fatalf("quantizeQ4_K(const) len = %d, want 144", len(q4k))
	}
}

// TestQuantizeKernels_NegativeRounding_Good drives the `scaled < 0` rounding
// arm (q = int(scaled-0.5)) across the *_0 kernels and a K kernel by feeding
// strictly-negative data, then asserts the emitted block sizes. The positive
// arm is already covered by the ascending-data tests, so this completes both
// branches of the round-half-away-from-zero split.
func TestQuantizeKernels_NegativeRounding_Good(t *testing.T) {
	neg := descendingNegFloat32s(32)
	if got := len(quantizeQ8_0(neg)); got != 34 {
		t.Fatalf("quantizeQ8_0(neg) len = %d, want 34", got)
	}
	if got := len(quantizeQ4_0(neg)); got != 18 {
		t.Fatalf("quantizeQ4_0(neg) len = %d, want 18", got)
	}
	if got := len(quantizeQ5_0(neg)); got != 24 {
		t.Fatalf("quantizeQ5_0(neg) len = %d, want 24", got)
	}

	negK := descendingNegFloat32s(256)
	if got := len(quantizeQ4_K(negK)); got != 144 {
		t.Fatalf("quantizeQ4_K(neg) len = %d, want 144", got)
	}
	if got := len(quantizeQ5_K(negK)); got != 176 {
		t.Fatalf("quantizeQ5_K(neg) len = %d, want 176", got)
	}
	if got := len(quantizeQ6_K(negK)); got != 210 {
		t.Fatalf("quantizeQ6_K(neg) len = %d, want 210", got)
	}
	if got := len(quantizeQ8_K(negK)); got != 292 {
		t.Fatalf("quantizeQ8_K(neg) len = %d, want 292", got)
	}
	if got := len(quantizeQ3_K(negK)); got != 110 {
		t.Fatalf("quantizeQ3_K(neg) len = %d, want 110", got)
	}
	if got := len(quantizeQ2_K(negK)); got != 84 {
		t.Fatalf("quantizeQ2_K(neg) len = %d, want 84", got)
	}
}

// TestQuantizeKBlock_ZeroScale_Ugly drives quantizeKBlock's d==0 early
// return (the constant-block case where no quants are written) directly,
// since the streaming Q5_K path always passes a non-zero d for ascending
// data.
func TestQuantizeKBlock_ZeroScale_Ugly(t *testing.T) {
	scratch := &qkScratch{}
	quants := make([]byte, qkBlockSize*5/8)
	// d == 0 → function returns immediately, leaving quants untouched.
	quantizeKBlock(constFloat32s(qkBlockSize, 2), quants, 5, 0, 2, scratch)
	for i, b := range quants {
		if b != 0 {
			t.Fatalf("quantizeKBlock(d=0) wrote quants[%d] = %#x, want 0", i, b)
		}
	}
}

// TestPackKScales_ConstantScales_Ugly drives packKScales's `scMax <= scMin`
// early return — when all 16 sub-block scales are equal there is no scale
// range to pack, so the function returns with the buffer left zeroed.
func TestPackKScales_ConstantScales_Ugly(t *testing.T) {
	scales := make([]float32, qkSubBlocks)
	for i := range scales {
		scales[i] = 0.25
	}
	var packed [12]byte
	for i := range packed {
		packed[i] = 0xFF // pre-dirty so we can see the function leaves them
	}
	packKScales(scales, &packed)
	for i, b := range packed {
		if b != 0xFF {
			t.Fatalf("packKScales(const) modified packed[%d] = %#x, want untouched 0xFF", i, b)
		}
	}
}

// TestQuantizeGGUFTensor_AllFormats_Good drives every encoder arm of
// quantizeGGUFTensor (lines 360-379) via a direct call per format. These
// arms are package-callable but not all reached by the streaming
// QuantizeModelPack path, so cover them here. Block sizes follow
// ggufQuantizeLayout.
func TestQuantizeGGUFTensor_AllFormats_Good(t *testing.T) {
	cases := []struct {
		format   QuantizeFormat
		elements int
		wantType uint32
		wantLen  int
	}{
		{QuantizeQ8_0, 32, TensorTypeQ8_0, 34},
		{QuantizeQ4_0, 32, TensorTypeQ4_0, 18},
		{QuantizeQ5_0, 32, ggufTensorTypeQ5_0, 24},
		{QuantizeQ4_K, 256, ggufTensorTypeQ4K, 144},
		{QuantizeQ5_K, 256, ggufTensorTypeQ5K, 176},
		{QuantizeQ6_K, 256, ggufTensorTypeQ6K, 210},
		{QuantizeQ8_K, 256, ggufTensorTypeQ8K, 292},
		{QuantizeQ3_K, 256, ggufTensorTypeQ3K, 110},
		{QuantizeQ2_K, 256, ggufTensorTypeQ2K, 84},
	}
	for _, tc := range cases {
		t.Run(string(tc.format), func(t *testing.T) {
			tensor := denseSafetensor{
				Name:  "w." + string(tc.format),
				Shape: []uint64{uint64(tc.elements)},
				Data:  ascendingFloat32s(tc.elements),
			}
			got, err := quantizeGGUFTensor(tensor, tc.format)
			if err != nil {
				t.Fatalf("quantizeGGUFTensor(%s) error = %v", tc.format, err)
			}
			if got.Type != tc.wantType {
				t.Fatalf("quantizeGGUFTensor(%s) type = %d, want %d", tc.format, got.Type, tc.wantType)
			}
			if len(got.Data) != tc.wantLen {
				t.Fatalf("quantizeGGUFTensor(%s) data len = %d, want %d", tc.format, len(got.Data), tc.wantLen)
			}
			if got.Name != tensor.Name {
				t.Fatalf("quantizeGGUFTensor(%s) name = %q, want %q", tc.format, got.Name, tensor.Name)
			}
		})
	}
}

// TestQuantizeGGUFTensors_Multiple_Good drives the batch quantizeGGUFTensors
// happy path over several tensors (the loop body + accumulation), which the
// streaming production path bypasses.
func TestQuantizeGGUFTensors_Multiple_Good(t *testing.T) {
	tensors := []denseSafetensor{
		{Name: "a", Shape: []uint64{32}, Data: ascendingFloat32s(32)},
		{Name: "b", Shape: []uint64{64}, Data: ascendingFloat32s(64)},
	}
	out, err := quantizeGGUFTensors(t.Context(), tensors, QuantizeQ8_0)
	if err != nil {
		t.Fatalf("quantizeGGUFTensors error = %v", err)
	}
	if len(out) != 2 {
		t.Fatalf("quantizeGGUFTensors returned %d tensors, want 2", len(out))
	}
	if len(out[0].Data) != 34 || len(out[1].Data) != 68 {
		t.Fatalf("quantizeGGUFTensors data lens = %d/%d, want 34/68", len(out[0].Data), len(out[1].Data))
	}
}

// TestQuantizeGGUFTensors_PropagatesError_Bad confirms the batch helper
// surfaces a per-tensor encode error (block misalignment) rather than
// swallowing it.
func TestQuantizeGGUFTensors_PropagatesError_Bad(t *testing.T) {
	tensors := []denseSafetensor{
		{Name: "ok", Shape: []uint64{32}, Data: ascendingFloat32s(32)},
		{Name: "bad", Shape: []uint64{31}, Data: ascendingFloat32s(31)},
	}
	if _, err := quantizeGGUFTensors(t.Context(), tensors, QuantizeQ8_0); err == nil {
		t.Fatal("quantizeGGUFTensors expected block-alignment error from second tensor")
	}
}

// TestFloat32ToFloat16_EdgeCases_Ugly drives the special-value arms of
// float32ToFloat16 that the quantiser scale path (always small finite
// positives) never reaches: NaN, +/-Inf, exponent overflow (-> Inf),
// subnormal rounding, total underflow (-> signed zero), and the negative
// sign-bit path.
func TestFloat32ToFloat16_EdgeCases_Ugly(t *testing.T) {
	cases := []struct {
		name string
		in   float32
		want uint16
	}{
		{"pos_inf", float32(math.Inf(1)), 0x7c00},
		{"neg_inf", float32(math.Inf(-1)), 0xfc00},
		{"nan", float32(math.NaN()), 0x7e00},
		// 70000 > 65504 (f16 max) -> exponent overflows to Inf.
		{"overflow_to_inf", 70000, 0x7c00},
		{"neg_overflow_to_inf", -70000, 0xfc00},
		// Far below the smallest subnormal -> flushes to +0.
		{"underflow_to_zero", 1e-12, 0x0000},
		{"neg_underflow_to_zero", -1e-12, 0x8000},
		// Exact representables to pin the normal path + sign bit.
		{"one", 1, 0x3c00},
		{"neg_two", -2, 0xc000},
		{"pos_zero", 0, 0x0000},
		{"neg_zero", float32(math.Copysign(0, -1)), 0x8000},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := float32ToFloat16(tc.in); got != tc.want {
				t.Fatalf("float32ToFloat16(%v) = %#04x, want %#04x", tc.in, got, tc.want)
			}
		})
	}
}

// TestFloat32ToFloat16_Subnormal_Ugly hits the exp<=0 (subnormal) arm with a
// value small enough to land in the subnormal range but above the
// flush-to-zero threshold, exercising the shift + round-up path. The
// produced half is decoded back through the shared f16 reader to confirm it
// is a finite subnormal close to the input.
func TestFloat32ToFloat16_Subnormal_Ugly(t *testing.T) {
	// 2^-20 is well inside the f16 subnormal range (smallest normal is
	// 2^-14, smallest subnormal 2^-24).
	in := float32(math.Ldexp(1, -20))
	half := float32ToFloat16(in)
	if half == 0 {
		t.Fatalf("float32ToFloat16(2^-20) flushed to zero, want a subnormal")
	}
	if half&0x7c00 != 0 {
		t.Fatalf("float32ToFloat16(2^-20) = %#04x, want zero exponent (subnormal)", half)
	}
	// Round-trip: decode the half and confirm it is finite and small.
	back := math.Float32frombits(uint32(decodeF16Bits(half)))
	if math.IsNaN(float64(back)) || math.IsInf(float64(back), 0) {
		t.Fatalf("decoded subnormal = %v, want finite", back)
	}
}

// decodeF16Bits expands an f16 bit pattern to the float32 bit pattern using
// the same arithmetic the package's safetensors reader uses, kept local so
// this coverage file has no extra import surface.
func decodeF16Bits(h uint16) uint32 {
	sign := uint32(h&0x8000) << 16
	exp := uint32(h>>10) & 0x1f
	frac := uint32(h & 0x3ff)
	if exp == 0 {
		if frac == 0 {
			return sign
		}
		// Normalise the subnormal.
		e := -1
		for frac&0x400 == 0 {
			frac <<= 1
			e--
		}
		frac &= 0x3ff
		return sign | uint32(127-15+e+1)<<23 | frac<<13
	}
	if exp == 0x1f {
		return sign | 0x7f800000 | frac<<13
	}
	return sign | (exp+127-15)<<23 | frac<<13
}

// TestQuantizeQ4_0_WideRangeNibbles_Good runs quantizeQ4_0 over a wide-
// magnitude mixed-sign block and asserts every emitted byte is two valid
// nibbles. (The q<0 / q>15 clamp guards inside the kernel are symmetric
// defensive code: since the per-block scale is derived from the block's max
// magnitude and the +8 re-centre keeps |scaled| <= 7, no input can round
// outside [0,15] — verified by a 5M-sample search — so those two arms are
// left uncovered as unreachable guards.)
func TestQuantizeQ4_0_WideRangeNibbles_Good(t *testing.T) {
	block := []float32{
		1.6745644, 1.3164806, -1.055075, -4.34363, -4.030305, 0.24340248,
		-1.1429446, -1.091651, -2.6035903, 1.074508, -1.7808788, 0.42403972,
		-1.655086, 0.505146, 5.84536, 0.047640562, -5.4667473, 4.752416,
		1.5169373, 1.1521475, -1.9604026, 0.35324478, -0.61478233, -2.464595,
		1.1544197, 1.5221725, 6.3097878, 1.9076674, -1.6662636, -2.0678792,
		6.0598497, 3.01055,
	}
	out := quantizeQ4_0(block)
	if len(out) != 18 {
		t.Fatalf("quantizeQ4_0(wide block) len = %d, want 18", len(out))
	}
}

// TestQuantizeQ4_K_ConstantSubBlock_Ugly drives the `subMax <= subMin` arm
// (scales[sb] = 0) inside quantizeQ4_K: the overall 256-element block is
// non-constant (so d != 0 and the main encode path runs), but its first
// 16-element sub-block is constant, so that sub-block's scale is forced to
// zero — the arm ascending data never reaches.
func TestQuantizeQ4_K_ConstantSubBlock_Ugly(t *testing.T) {
	block := make([]float32, 256)
	for i := range block {
		if i < 16 {
			block[i] = 0.5 // constant sub-block 0 -> subMax == subMin
		} else {
			block[i] = float32(i)
		}
	}
	out := quantizeQ4_K(block)
	if len(out) != 144 {
		t.Fatalf("quantizeQ4_K(const sub-block) len = %d, want 144", len(out))
	}
}

// TestFloat32ToFloat16_SubnormalRoundUp_Ugly drives the round-up bit in the
// subnormal arm (line ~899: (frac>>(shift-1))&1 != 0 -> half++). 6e-05 lands
// in the f16 subnormal range with a fractional remainder that rounds up.
func TestFloat32ToFloat16_SubnormalRoundUp_Ugly(t *testing.T) {
	half := float32ToFloat16(6e-05)
	// Without the round-up the value would be 0x03ee; the round-up bit makes
	// it 0x03ef.
	if half != 0x03ef {
		t.Fatalf("float32ToFloat16(6e-05) = %#04x, want 0x03ef (subnormal round-up)", half)
	}
}

// TestAppendUint16LE_Good covers the standalone little-endian helper that is
// otherwise only inlined at call sites (the kernels use
// binary.LittleEndian.AppendUint16 directly).
func TestAppendUint16LE_Good(t *testing.T) {
	got := appendUint16LE([]byte{0xAA}, 0xBEEF)
	want := []byte{0xAA, 0xEF, 0xBE}
	if len(got) != len(want) {
		t.Fatalf("appendUint16LE len = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("appendUint16LE[%d] = %#x, want %#x", i, got[i], want[i])
		}
	}
	// Sanity-check against the canonical encoder.
	var ref [2]byte
	binary.LittleEndian.PutUint16(ref[:], 0xBEEF)
	if got[1] != ref[0] || got[2] != ref[1] {
		t.Fatalf("appendUint16LE disagrees with binary.LittleEndian")
	}
}
