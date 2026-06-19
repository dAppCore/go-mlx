// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

// bfloat16 MLP pieces: the bf16 elementwise ops and the composed bf16 GELU gate
// that a bf16 feed-forward block needs. Like the other bf16 ops these take/return
// raw bf16 []byte. GELU is composed (no metallib kernel) — in bf16 the question
// is whether the separately-run, each-step-rounded composition matches mlx's
// mlx_compile-fused gelu; the parity test answers it with data.

// f32ToBF16 converts a float32 to bfloat16 bits with round-to-nearest-even —
// matching mlx's AsType(..., bfloat16) so a constant produced here equals the
// same constant mlx would broadcast.
func f32ToBF16(v float32) uint16 {
	bits := math.Float32bits(v)
	if bits&0x7fffffff > 0x7f800000 { // NaN: keep it quiet, non-zero mantissa
		return uint16(bits>>16) | 0x0040
	}
	rounding := (bits>>16)&1 + 0x7fff
	return uint16((bits + rounding) >> 16)
}

// bf16ConstBytes returns n copies of v as bf16 bytes — a broadcast scalar operand
// for the contiguous bf16 binary kernels.
func bf16ConstBytes(n int, v float32) []byte {
	h := f32ToBF16(v)
	out := make([]byte, n*bf16Size)
	for i := 0; i < n; i++ {
		out[i*2] = byte(h)
		out[i*2+1] = byte(h >> 8)
	}
	return out
}

// runBinaryBF16 drives a contiguous bf16 binary kernel (vv_<Op>bfloat16) over two
// equal-length bf16 byte buffers. name e.g. "vv_Multiplybfloat16".
func runBinaryBF16(name string, a, b []byte) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(a) != len(b) {
		return nil, core.NewError("native.runBinaryBF16: a and b must be the same length")
	}
	if len(a)%bf16Size != 0 {
		return nil, core.NewError("native.runBinaryBF16: byte length must be a multiple of 2")
	}
	pso, err := pipelineFor(name)
	if err != nil {
		return nil, err
	}
	n := len(a) / bf16Size
	out := make([]byte, len(a))
	if n == 0 {
		return out, nil
	}
	withAutoreleasePool(func() {
		aBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&a[0]), uint(len(a)), metal.MTLResourceStorageModeShared)
		bBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&b[0]), uint(len(b)), metal.MTLResourceStorageModeShared)
		outBuf := device.NewBufferWithLengthOptions(uint(len(a)), metal.MTLResourceStorageModeShared)
		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		enc.SetComputePipelineState(pso)
		enc.SetBufferWithOffsetAtIndex(aBuf, 0, 0)
		enc.SetBufferWithOffsetAtIndex(bBuf, 0, 1)
		enc.SetBufferWithOffsetAtIndex(outBuf, 0, 2)
		setEncInt32(enc, int32(n), 3)
		group := uint(256)
		if uint(n) < group {
			group = uint(n)
		}
		enc.DispatchThreadsThreadsPerThreadgroup(
			metal.MTLSize{Width: uint(n), Height: 1, Depth: 1},
			metal.MTLSize{Width: group, Height: 1, Depth: 1},
		)
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()
		copy(out, unsafe.Slice((*byte)(outBuf.Contents()), len(a)))
	})
	return out, nil
}

// MulBF16 is the bf16 sibling of Mul: element-wise a[i]*b[i] over bf16 bytes
// (kernel vv_Multiplybfloat16) — the MLP gate·up step in the decode dtype.
func MulBF16(a, b []byte) ([]byte, error) { return runBinaryBF16("vv_Multiplybfloat16", a, b) }

// TanhBF16 is the bf16 sibling of Tanh: element-wise tanh over bf16 bytes (kernel
// v_Tanhbfloat16bfloat16) — the nonlinearity inside the gelu approximation.
func TanhBF16(x []byte) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(x)%bf16Size != 0 {
		return nil, core.NewError("native.TanhBF16: byte length must be a multiple of 2")
	}
	pso, err := pipelineFor("v_Tanhbfloat16bfloat16")
	if err != nil {
		return nil, err
	}
	n := len(x) / bf16Size
	out := make([]byte, len(x))
	if n == 0 {
		return out, nil
	}
	withAutoreleasePool(func() {
		inBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&x[0]), uint(len(x)), metal.MTLResourceStorageModeShared)
		outBuf := device.NewBufferWithLengthOptions(uint(len(x)), metal.MTLResourceStorageModeShared)
		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		enc.SetComputePipelineState(pso)
		enc.SetBufferWithOffsetAtIndex(inBuf, 0, 0)
		enc.SetBufferWithOffsetAtIndex(outBuf, 0, 1)
		cnt := uint32(n)
		enc.SetBytesLengthAtIndex(unsafe.Slice((*byte)(unsafe.Pointer(&cnt)), 4), 4, 2)
		group := uint(256)
		if uint(n) < group {
			group = uint(n)
		}
		enc.DispatchThreadsThreadsPerThreadgroup(
			metal.MTLSize{Width: uint(n), Height: 1, Depth: 1},
			metal.MTLSize{Width: group, Height: 1, Depth: 1},
		)
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()
		copy(out, unsafe.Slice((*byte)(outBuf.Contents()), len(x)))
	})
	return out, nil
}

// GeluBF16 is the bf16 sibling of Gelu: the tanh-approximation GELU composed from
// the bf16 primitives, each intermediate rounded to bf16 (the gelu_approx graph
// MLX would run un-fused in bf16). Input/output raw bf16 bytes.
func GeluBF16(x []byte) ([]byte, error) {
	n := len(x) / bf16Size
	mustMul := func(a, b []byte, e error) ([]byte, error) {
		if e != nil {
			return nil, e
		}
		return MulBF16(a, b)
	}
	x2, err := MulBF16(x, x)
	if err != nil {
		return nil, err
	}
	x3, err := mustMul(x2, x, nil)
	if err != nil {
		return nil, err
	}
	x3s, err := mustMul(x3, bf16ConstBytes(n, 0.044715), nil)
	if err != nil {
		return nil, err
	}
	inner, err := AddBF16(x, x3s)
	if err != nil {
		return nil, err
	}
	scaled, err := mustMul(inner, bf16ConstBytes(n, 0.7978845608028654), nil)
	if err != nil {
		return nil, err
	}
	t, err := TanhBF16(scaled)
	if err != nil {
		return nil, err
	}
	onePlus, err := AddBF16(t, bf16ConstBytes(n, 1.0))
	if err != nil {
		return nil, err
	}
	halfX, err := mustMul(x, bf16ConstBytes(n, 0.5), nil)
	if err != nil {
		return nil, err
	}
	return MulBF16(halfX, onePlus)
}

// GeluGateMulBF16 computes gelu(gate)·up in bf16 — gemma's MLP gate in the decode
// dtype. Uses the fused kernel (fp32-internal, one dispatch) when the custom kernels
// metallib is loaded, else the composed bf16 primitive chain. Parity in parity_test.go.
func GeluGateMulBF16(gate, up []byte) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(up) != len(gate) {
		return nil, core.NewError("native.GeluGateMulBF16: gate/up length mismatch")
	}
	if gpuHasGeluKernel() {
		return geluGateMulFused(gate, up, len(gate)/bf16Size)
	}
	g, err := GeluBF16(gate)
	if err != nil {
		return nil, err
	}
	return MulBF16(g, up)
}
