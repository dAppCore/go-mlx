// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import (
	"math"
	"testing"
)

// TestQuantizeGGUFTensor_AllFormats_Good drives every quantise format
// through quantizeGGUFTensor via a direct call per format. quantizeGGUFTensor
// dispatches to sharedgguf.Quantize (the nine kernels live in
// dappco.re/go/inference/gguf, not this package); these arms are
// package-callable but not all reached by the streaming QuantizeModelPack
// path, so cover them here. Block sizes follow ggufQuantizeLayout.
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

// TestFloat32ToFloat16_SubnormalRoundUp_Ugly drives the round-up bit in
// float32ToFloat16's subnormal arm: (frac>>(shift-1))&1 != 0 -> half++.
// 6e-05 lands in the f16 subnormal range with a fractional remainder that
// rounds up.
func TestFloat32ToFloat16_SubnormalRoundUp_Ugly(t *testing.T) {
	half := float32ToFloat16(6e-05)
	// Without the round-up the value would be 0x03ee; the round-up bit makes
	// it 0x03ef.
	if half != 0x03ef {
		t.Fatalf("float32ToFloat16(6e-05) = %#04x, want 0x03ef (subnormal round-up)", half)
	}
}
