// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"testing"
)

// Small-M regime benches — the MTP verify block (L=2..5) cost decomposition.
// go test ./pkg/metal -run XX -bench 'BenchmarkQMMSmallM|BenchmarkMaskedSDPASmallL' -benchtime 50x

// BenchmarkQMMSmallM measures mlx quantized_matmul at decode-block row counts
// on a 31B-ish projection shape. If qmv amortises the weight stream (mlx
// serves up to its qmv batch limit), per-call cost stays ~flat across M.
func BenchmarkQMMSmallM(b *testing.B) {
	benchmarkQMMSmallM(b, 5120, 5120)
}

// BenchmarkQMMSmallMWide runs the real 31B MLP projection shape, where the
// weight stream (not dispatch) dominates — the toy square shape is
// dispatch-bound and blind to per-row compute scaling.
func BenchmarkQMMSmallMWide(b *testing.B) {
	benchmarkQMMSmallM(b, 5120, 27648)
}

func benchmarkQMMSmallM(b *testing.B, in, out int) {
	packed := make([]uint32, out*in/8)
	for i := range packed {
		packed[i] = uint32(i)*2654435761 + 7
	}
	wq := FromValues(packed, out, in/8)
	_ = in
	groups := in / 64
	scaleF := make([]float32, out*groups)
	for i := range scaleF {
		scaleF[i] = 0.01
	}
	scales := AsType2(FromValues(scaleF, out, groups), DTypeBFloat16)
	biases := AsType2(FromValues(scaleF, out, groups), DTypeBFloat16)
	defer Free(wq, scales, biases)

	for _, m := range []int{1, 2, 3, 5, 8} {
		b.Run(byteSizeLabel("M", m), func(b *testing.B) {
			xF := make([]float32, m*in)
			for i := range xF {
				xF[i] = float32(i%7) * 0.1
			}
			x := AsType2(FromValues(xF, 1, m, in), DTypeBFloat16)
			defer Free(x)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				y := QuantizedMatmul(x, wq, scales, biases, true, 64, 4)
				if err := Eval(y); err != nil {
					b.Fatalf("Eval: %v", err)
				}
				Free(y)
			}
		})
	}
}

// BenchmarkMaskedSDPASmallL measures the masked SDPA at verify query lengths
// over a decode-band read set at 31B-ish geometry (GQA 8:1 over 512-dim
// global-layer heads). qLen=1 takes mlx's sdpa_vector; qLen>1 falls to the
// full attention kernel — this measures that cliff.
func BenchmarkMaskedSDPASmallL(b *testing.B) {
	const heads, kvHeads, band, headDim = 16, 2, 512, 512
	mk := func(shape []int) *Array {
		n := 1
		for _, d := range shape {
			n *= d
		}
		values := make([]float32, n)
		for i := range values {
			values[i] = float32(i%17)*0.21 - float32(i%5)*0.13
		}
		return AsType2(FromValues(values, shape...), DTypeBFloat16)
	}
	k := mk([]int{1, kvHeads, band, headDim})
	v := mk([]int{1, kvHeads, band, headDim})
	defer Free(k, v)

	for _, qLen := range []int{1, 2, 3, 5} {
		b.Run(byteSizeLabel("L", qLen), func(b *testing.B) {
			q := mk([]int{1, heads, qLen, headDim})
			offset := FromValue(band - 40)
			mask := MultiTokenCausalMask(band, offset, qLen)
			maskCast := AsType(mask, DTypeBFloat16)
			defer Free(q, offset, mask, maskCast)
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				out := ScaledDotProductAttentionWithMask(q, k, v, maskCast, 0.0442)
				if err := Eval(out); err != nil {
					b.Fatalf("Eval: %v", err)
				}
				Free(out)
			}
		})
	}
}

// AsType2 is a free-the-input convenience for bench setup.
func AsType2(a *Array, dtype DType) *Array {
	out := AsType(a, dtype)
	Free(a)
	return out
}

func byteSizeLabel(prefix string, n int) string {
	return prefix + "=" + string(rune('0'+n))
}
