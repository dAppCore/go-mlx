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

// synthQ4Linear builds a synthetic-but-valid q4g64 Linear for determinism
// probes: packed nibbles [out, in/8], scales/biases [out, in/64] in dtype.
func synthQ4Linear(t *testing.T, dtype DType, in, out int, seed uint32) *Linear {
	t.Helper()
	packed := make([]uint32, out*in/8)
	for i := range packed {
		packed[i] = uint32(i)*2654435761 + seed
	}
	groups := in / 64
	scaleF := make([]float32, out*groups)
	biasF := make([]float32, out*groups)
	for i := range scaleF {
		scaleF[i] = 0.008 + float32((i+int(seed))%9)*0.002
		biasF[i] = -0.04 + float32((i+int(seed))%5)*0.01
	}
	scales := FromValues(scaleF, out, groups)
	biases := FromValues(biasF, out, groups)
	if dtype != DTypeFloat32 {
		castS := AsType(scales, dtype)
		Free(scales)
		scales = castS
		castB := AsType(biases, dtype)
		Free(biases)
		biases = castB
	}
	return &Linear{
		Weight:           FromValues(packed, out, in/8),
		Scales:           scales,
		Biases:           biases,
		QuantizationMode: "affine",
		GroupSize:        64,
		Bits:             4,
	}
}

// TestCompiledFusedMLPDeterminism reproduces the decode-fork isolation: the
// fused MLP custom kernels are deterministic UNCOMPILED but fork INSIDE an
// mlx_compile trace under the bf16 stream (grid result: serial-compiled forks,
// compiled-with-gemm-MLP and uncompiled are clean). Hammers the traced fused
// path on fixed inputs; any hash change across repeats is the bug in a tube.
func TestCompiledFusedMLPDeterminism(t *testing.T) {
	const hidden, inter = 2048, 8192
	for _, tc := range []struct {
		name  string
		dtype DType
	}{
		{"float32", DTypeFloat32},
		{"bfloat16", DTypeBFloat16},
	} {
		t.Run(tc.name, func(t *testing.T) {
			gate := synthQ4Linear(t, tc.dtype, hidden, inter, 11)
			up := synthQ4Linear(t, tc.dtype, hidden, inter, 23)
			down := synthQ4Linear(t, tc.dtype, inter, hidden, 37)
			defer func() {
				for _, l := range []*Linear{gate, up, down} {
					Free(l.Weight, l.Scales, l.Biases)
				}
			}()

			fn := CompileShapeless(func(in []*Array) []*Array {
				return []*Array{TracedGELUMLPForward(in[0], gate, up, down)}
			}, false)

			xF := make([]float32, hidden)
			for i := range xF {
				xF[i] = float32(i%13)*0.19 - 0.4
			}
			mkInput := func() *Array {
				x := FromValues(xF, 1, 1, hidden)
				if tc.dtype != DTypeFloat32 {
					cast := AsType(x, tc.dtype)
					Free(x)
					return cast
				}
				return x
			}

			runHash := func() [32]byte {
				x := mkInput()
				outs := fn.Call(x)
				if len(outs) != 1 || outs[0] == nil || !outs[0].Valid() {
					t.Fatalf("compiled fused MLP returned invalid output")
				}
				f32 := AsType(outs[0], DTypeFloat32)
				if err := Eval(f32); err != nil {
					t.Fatalf("Eval: %v", err)
				}
				floats := f32.Floats()
				bytes := make([]byte, 0, len(floats)*4)
				for _, f := range floats {
					u := math.Float32bits(f)
					bytes = append(bytes, byte(u), byte(u>>8), byte(u>>16), byte(u>>24))
				}
				Free(x, outs[0], f32)
				return sha256.Sum256(bytes)
			}

			reference := runHash()
			for i := 0; i < 300; i++ {
				if got := runHash(); got != reference {
					t.Fatalf("compiled fused MLP non-deterministic in %s at repeat %d", tc.name, i)
				}
			}
			t.Logf("%s: 300 repeats hash-identical", tc.name)
		})
	}
}

// TestQuantizedDenseMatVecBF16Input pins down kernel behaviour on a
// half-precision activation: the FusedDownOnly live config GPU-page-faulted
// when the down matvec received a bf16 GeluGateMul output. Probes the kernel
// standalone and inside a compile trace.
func TestQuantizedDenseMatVecBF16Input(t *testing.T) {
	const in, out = 8192, 2048
	linear := synthQ4Linear(t, DTypeBFloat16, in, out, 51)
	defer Free(linear.Weight, linear.Scales, linear.Biases)

	xF := make([]float32, in)
	for i := range xF {
		xF[i] = float32(i%11)*0.13 - 0.3
	}
	mkX := func(dtype DType) *Array {
		x := FromValues(xF, 1, 1, in)
		if dtype != DTypeFloat32 {
			cast := AsType(x, dtype)
			Free(x)
			return cast
		}
		return x
	}

	hash := func(arr *Array) [32]byte {
		t.Helper()
		f32 := AsType(arr, DTypeFloat32)
		if err := Eval(f32); err != nil {
			t.Fatalf("Eval: %v", err)
		}
		floats := f32.Floats()
		bytes := make([]byte, 0, len(floats)*4)
		for _, f := range floats {
			u := math.Float32bits(f)
			bytes = append(bytes, byte(u), byte(u>>8), byte(u>>16), byte(u>>24))
		}
		Free(f32)
		return sha256.Sum256(bytes)
	}

	t.Run("uncompiled", func(t *testing.T) {
		run := func() [32]byte {
			x := mkX(DTypeBFloat16)
			y, ok, err := QuantizedDenseMatVec(x, linear)
			if err != nil || !ok {
				t.Fatalf("QuantizedDenseMatVec: ok=%v err=%v", ok, err)
			}
			h := hash(y)
			Free(x, y)
			return h
		}
		reference := run()
		for i := 0; i < 300; i++ {
			if got := run(); got != reference {
				t.Fatalf("uncompiled bf16 down matvec non-deterministic at repeat %d", i)
			}
		}
		t.Logf("uncompiled bf16: 300 repeats hash-identical")
	})

	t.Run("compiled", func(t *testing.T) {
		fn := CompileShapeless(func(ins []*Array) []*Array {
			y, ok, err := QuantizedDenseMatVec(ins[0], linear)
			if err != nil || !ok {
				panic("QuantizedDenseMatVec declined in trace")
			}
			return []*Array{y}
		}, false)
		run := func() [32]byte {
			x := mkX(DTypeBFloat16)
			outs := fn.Call(x)
			if len(outs) != 1 || outs[0] == nil || !outs[0].Valid() {
				t.Fatalf("compiled call returned invalid output")
			}
			h := hash(outs[0])
			Free(x, outs[0])
			return h
		}
		reference := run()
		for i := 0; i < 300; i++ {
			if got := run(); got != reference {
				t.Fatalf("compiled bf16 down matvec non-deterministic at repeat %d", i)
			}
		}
		t.Logf("compiled bf16: 300 repeats hash-identical")
	})
}
