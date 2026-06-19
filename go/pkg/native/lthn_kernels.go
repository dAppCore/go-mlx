// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"sync"
	"unsafe"

	"github.com/tmc/apple/metal"
)

// lthn_kernels.go is the native engine's own custom-kernel mechanism: kernels MLX's static metallib
// does not have, compiled from kernels/*.metal into a sibling lthn_kernels.metallib that device.go
// loads beside MLX's (customLibrary). The first such kernel is the fused gelu (kernels/
// lthn_gelu_gate_mul.metal). This is the foundation for any fused/novel op the native wants — fused
// activations, the "compute fp32, store bf16" path, future LEK/MTP kernels — independent of whether
// any one of them is wired into the serve decode.

// gpuHasGeluKernel reports whether the fused gelu kernel is available (the custom kernels metallib
// loaded). The composed bf16 chain is the production path; this is the fused capability beside it.
func gpuHasGeluKernel() bool { return customLibraryLoaded }

var (
	geluPSOOnce sync.Once
	geluPSO     metal.MTLComputePipelineState
	geluPSOErr  error
)

// geluPipeline builds (once) the fused gelu pipeline from the custom kernels library.
func geluPipeline() (metal.MTLComputePipelineState, error) {
	geluPSOOnce.Do(func() {
		fn := customLibrary.NewFunctionWithName("lthn_gelu_gate_mul_bf16")
		geluPSO, geluPSOErr = device.NewComputePipelineStateWithFunctionError(fn)
	})
	return geluPSO, geluPSOErr
}

// encGeluGateMulFused encodes gelu(gate)·up via the fused kernel — one dispatch, fp32-internal, one
// bf16 rounding (see the kernel comment for why this differs from the composed production path).
// gate/up/out are contiguous bf16 buffers of n elements. Guard with gpuHasGeluKernel before calling.
func encGeluGateMulFused(enc metal.MTLComputeCommandEncoder, gate, up, out metal.MTLBuffer, n int) error {
	pso, err := geluPipeline()
	if err != nil {
		return err
	}
	enc.SetComputePipelineState(pso)
	enc.SetBufferWithOffsetAtIndex(gate, 0, 0)
	enc.SetBufferWithOffsetAtIndex(up, 0, 1)
	enc.SetBufferWithOffsetAtIndex(out, 0, 2)
	setEncInt32(enc, int32(n), 3)
	group := uint(256)
	if uint(n) < group {
		group = uint(n)
	}
	enc.DispatchThreadsThreadsPerThreadgroup(
		metal.MTLSize{Width: uint(n), Height: 1, Depth: 1},
		metal.MTLSize{Width: group, Height: 1, Depth: 1},
	)
	return nil
}

// geluGateMulFused is the one-shot host wrapper around the fused kernel — gate/up bf16 bytes in,
// bf16 bytes out. The diagnostic + bench exercise it; the decode stays on the composed chain.
func geluGateMulFused(gate, up []byte, n int) ([]byte, error) {
	var out []byte
	var encErr error
	withAutoreleasePool(func() {
		gBuf, uBuf := sharedBytes(gate), sharedBytes(up)
		oBuf := scratchBF16(n)
		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		if encErr = encGeluGateMulFused(enc, gBuf, uBuf, oBuf, n); encErr != nil {
			enc.EndEncoding()
			return
		}
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()
		out = make([]byte, n*bf16Size)
		copy(out, unsafe.Slice((*byte)(oBuf.Contents()), n*bf16Size))
	})
	return out, encErr
}
