// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

// RunBinary drives a contiguous binary MLX kernel over two equal-length inputs
// and returns a fresh result slice. It targets the vv_<Op>float32 family, whose
// host ABI (from mlx/backend/metal/binary.cpp) is: a → buffer(0), b → buffer(1),
// out → buffer(2), element count → buffer(3), one GPU thread per element. name is
// e.g. "vv_Addfloat32". The byte-for-byte equivalent of the mlx-c contiguous
// binary path — parity is gated in the tests.
func RunBinary(name string, a, b []float32) ([]float32, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(a) != len(b) {
		return nil, core.NewError("native.RunBinary: a and b must be the same length")
	}
	pso, err := pipelineFor(name)
	if err != nil {
		return nil, err
	}
	n := len(a)
	out := make([]float32, n)
	if n == 0 {
		return out, nil
	}

	withAutoreleasePool(func() {
		aBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&a[0]), uint(n*4), metal.MTLResourceStorageModeShared)
		bBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&b[0]), uint(n*4), metal.MTLResourceStorageModeShared)
		outBuf := device.NewBufferWithLengthOptions(uint(n*4), metal.MTLResourceStorageModeShared)

		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		enc.SetComputePipelineState(pso)
		enc.SetBufferWithOffsetAtIndex(aBuf, 0, 0)
		enc.SetBufferWithOffsetAtIndex(bBuf, 0, 1)
		enc.SetBufferWithOffsetAtIndex(outBuf, 0, 2)
		setEncInt32(enc, int32(n), 3) // element count

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

		copy(out, unsafe.Slice((*float32)(outBuf.Contents()), n))
	})
	return out, nil
}

// Add returns the element-wise sum a[i]+b[i] on the GPU via the shared
// mlx.metallib (kernel vv_Addfloat32). This is the residual add used twice per
// decode block. Parity with pkg/metal.Add is gated in parity_test.go.
//
//	out, err := native.Add([]float32{1, 2}, []float32{3, 4}) // out = [4 6]
func Add(a, b []float32) ([]float32, error) {
	return RunBinary("vv_Addfloat32", a, b)
}

// Mul returns the element-wise product a[i]*b[i] on the GPU via the shared
// mlx.metallib (kernel vv_Multiplyfloat32) — the gate·up step of the MLP. Parity
// with pkg/metal.Mul is gated in parity_test.go.
//
//	out, err := native.Mul([]float32{2, 3}, []float32{4, 5}) // out = [8 15]
func Mul(a, b []float32) ([]float32, error) {
	return RunBinary("vv_Multiplyfloat32", a, b)
}
