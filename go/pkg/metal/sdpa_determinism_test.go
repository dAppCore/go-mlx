// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"crypto/sha256"
	"math"
	"testing"
)

// sdpaDeterminismRun hashes one masked single-token SDPA output on fixed
// inputs — the decode attention shape (q [1,H,1,D] over a band of K/V with
// an additive mask), in the dtype under test.
func sdpaDeterminismRun(t *testing.T, dtype DType, heads, kvHeads, band, headDim int32) [32]byte {
	t.Helper()
	mk := func(shape []int, seed float32) *Array {
		n := 1
		for _, d := range shape {
			n *= d
		}
		values := make([]float32, n)
		for i := range values {
			values[i] = seed + float32(i%17)*0.21 - float32(i%5)*0.13
		}
		arr := FromValues(values, shape...)
		if dtype != DTypeFloat32 {
			cast := AsType(arr, dtype)
			Free(arr)
			return cast
		}
		return arr
	}
	q := mk([]int{1, int(heads), 1, int(headDim)}, 0.3)
	k := mk([]int{1, int(kvHeads), int(band), int(headDim)}, -0.2)
	v := mk([]int{1, int(kvHeads), int(band), int(headDim)}, 0.7)
	offset := FromValue(int(band) - 40) // mask the tail like a part-filled band
	mask := SingleTokenCausalMask(int(band), offset)
	out := ScaledDotProductAttentionWithMask(q, k, v, mask, 0.0883)
	outF32 := AsType(out, DTypeFloat32)
	if err := Eval(outF32); err != nil {
		t.Fatalf("Eval: %v", err)
	}
	floats := outF32.Floats()
	bytes := make([]byte, 0, len(floats)*4)
	for _, f := range floats {
		u := mathFloat32bits(f)
		bytes = append(bytes, byte(u), byte(u>>8), byte(u>>16), byte(u>>24))
	}
	Free(q, k, v, offset, mask, out, outF32)
	return sha256.Sum256(bytes)
}

func mathFloat32bits(f float32) uint32 {
	return math.Float32bits(f)
}

// quantizedMatmulDeterminismRun hashes one M=1 quantized matmul (the decode
// projection shape: [1,1,in] bf16 activation × q4g64 weights).
func quantizedMatmulDeterminismRun(t *testing.T, dtype DType, in, out int32) [32]byte {
	t.Helper()
	// Synthetic-but-valid q4g64 weights: determinism needs valid layout, not
	// meaningful values. uint32-packed nibbles [out, in/8], scales/biases
	// [out, in/64] in the activation dtype.
	packed := make([]uint32, int(out)*int(in)/8)
	for i := range packed {
		packed[i] = uint32(i*2654435761 + 12345)
	}
	wq := FromValues(packed, int(out), int(in)/8)
	groups := int(in) / 64
	scaleF := make([]float32, int(out)*groups)
	biasF := make([]float32, int(out)*groups)
	for i := range scaleF {
		scaleF[i] = 0.01 + float32(i%9)*0.002
		biasF[i] = -0.05 + float32(i%5)*0.01
	}
	scales := FromValues(scaleF, int(out), groups)
	biases := FromValues(biasF, int(out), groups)
	if dtype != DTypeFloat32 {
		castS := AsType(scales, dtype)
		Free(scales)
		scales = castS
		castB := AsType(biases, dtype)
		Free(biases)
		biases = castB
	}

	xF := make([]float32, in)
	for i := range xF {
		xF[i] = float32(i%13)*0.19 - 0.4
	}
	x := FromValues(xF, 1, 1, int(in))
	if dtype != DTypeFloat32 {
		cast := AsType(x, dtype)
		Free(x)
		x = cast
	}
	y := quantizedMatmulMode(x, wq, scales, biases, true, 64, 4, "affine")
	yF32 := AsType(y, DTypeFloat32)
	if err := Eval(yF32); err != nil {
		t.Fatalf("Eval: %v", err)
	}
	floats := yF32.Floats()
	bytes := make([]byte, 0, len(floats)*4)
	for _, f := range floats {
		u := math.Float32bits(f)
		bytes = append(bytes, byte(u), byte(u>>8), byte(u>>16), byte(u>>24))
	}
	Free(x, wq, scales, biases, y, yF32)
	return sha256.Sum256(bytes)
}

// TestQuantizedMatmulDeterminism hammers the M=1 q4g64 quantized matmul at
// the decode projection shape across activation dtypes.
func TestQuantizedMatmulDeterminism(t *testing.T) {
	for _, tc := range []struct {
		name  string
		dtype DType
	}{
		{"float32", DTypeFloat32},
		{"bfloat16", DTypeBFloat16},
	} {
		t.Run(tc.name, func(t *testing.T) {
			reference := quantizedMatmulDeterminismRun(t, tc.dtype, 2048, 2048)
			for i := 0; i < 200; i++ {
				if got := quantizedMatmulDeterminismRun(t, tc.dtype, 2048, 2048); got != reference {
					t.Fatalf("M=1 quantized matmul non-deterministic in %s at repeat %d", tc.name, i)
				}
			}
			t.Logf("%s: 200 repeats hash-identical", tc.name)
		})
	}
}

// TestSDPAMaskedDeterminism hammers the masked single-token SDPA at the
// decode shape across dtypes: any hash change across repeats is kernel-level
// non-determinism. e2b geometry (8 query heads, 1 KV head, 256-band, 256-dim)
// in bf16 is the branch the bf16 activation stream exercises and mlx-lm's
// decode (no array mask) does not.
func TestSDPAMaskedDeterminism(t *testing.T) {
	for _, tc := range []struct {
		name  string
		dtype DType
	}{
		{"float32", DTypeFloat32},
		{"bfloat16", DTypeBFloat16},
		{"float16", DTypeFloat16},
	} {
		t.Run(tc.name, func(t *testing.T) {
			reference := sdpaDeterminismRun(t, tc.dtype, 8, 1, 256, 256)
			for i := 0; i < 200; i++ {
				if got := sdpaDeterminismRun(t, tc.dtype, 8, 1, 256, 256); got != reference {
					t.Fatalf("masked SDPA non-deterministic in %s at repeat %d", tc.name, i)
				}
			}
			t.Logf("%s: 200 repeats hash-identical", tc.name)
		})
	}
}
