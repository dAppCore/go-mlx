// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"sync"
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

var (
	rmsQMVPSOMu    sync.Mutex
	rmsQMVPSOCache = map[string]metal.MTLComputePipelineState{}

	rmsQMVICBPSOMu    sync.Mutex
	rmsQMVICBPSOCache = map[string]metal.MTLComputePipelineState{}
)

// rmsQMVPipelineICB is rmsQMVFastPipeline with indirect-command-buffer support — the variant the
// decode ICB records (and replays per token). Same fused kernel; the descriptor just opts into ICB.
func rmsQMVPipelineICB(groupSize, bits int) (metal.MTLComputePipelineState, error) {
	name := core.Sprintf("lthn_rms_affine_qmv_fast_bfloat16_t_gs_%d_b_%d", groupSize, bits)
	key := name + "|icb"
	rmsQMVICBPSOMu.Lock()
	defer rmsQMVICBPSOMu.Unlock()
	if pso, ok := rmsQMVICBPSOCache[key]; ok {
		return pso, nil
	}
	if customLibrary == nil || customLibrary.GetID() == 0 {
		return nil, core.NewError("native.rmsQMVPipelineICB: custom library unavailable")
	}
	fn := customLibrary.NewFunctionWithName(name)
	if fn == nil || fn.GetID() == 0 {
		return nil, core.NewError("native.rmsQMVPipelineICB: kernel " + name + " not found")
	}
	desc := metal.NewMTLComputePipelineDescriptor()
	desc.SetComputeFunction(fn)
	desc.SetSupportIndirectCommandBuffers(true)
	pso, err := device.NewComputePipelineStateWithDescriptorOptionsReflectionError(desc, 0, nil)
	if err != nil {
		return nil, core.E("native.rmsQMVPipelineICB", key, err)
	}
	rmsQMVICBPSOCache[key] = pso
	return pso, nil
}

// rmsQMVFastPipeline builds (and caches) the fused rms-norm + affine_qmv_fast pipeline for a group
// size / bits from the custom kernels library. Shares the customLibraryLoaded gate with gelu.
func rmsQMVFastPipeline(groupSize, bits int) (metal.MTLComputePipelineState, error) {
	key := core.Sprintf("lthn_rms_affine_qmv_fast_bfloat16_t_gs_%d_b_%d", groupSize, bits)
	rmsQMVPSOMu.Lock()
	defer rmsQMVPSOMu.Unlock()
	if pso, ok := rmsQMVPSOCache[key]; ok {
		return pso, nil
	}
	if customLibrary == nil || customLibrary.GetID() == 0 {
		return nil, core.NewError("native.rmsQMVFastPipeline: custom library unavailable")
	}
	fn := customLibrary.NewFunctionWithName(key)
	if fn == nil || fn.GetID() == 0 {
		return nil, core.NewError("native.rmsQMVFastPipeline: kernel " + key + " not found")
	}
	pso, err := device.NewComputePipelineStateWithFunctionError(fn)
	if err != nil {
		return nil, core.E("native.rmsQMVFastPipeline", key, err)
	}
	rmsQMVPSOCache[key] = pso
	return pso, nil
}

// RMSQMVFastBF16 fuses, in ONE dispatch, the per-row RMSNorm(x, normW) + the 4-bit affine_qmv_fast
// projection: out = (W·RMSNorm(x, normW)). x/normW are inDim bf16 bytes; wq/scales/biases are the
// packed 4-bit weight; out is outDim bf16 bytes. Numerically equal to QMVBF16(RMSNormBF16(x, normW)) —
// the qmv arithmetic is byte-identical (bfloat16_t == native bfloat), only the rms reduction differs
// (~1 ULP, cosine ~1.0). Requires the fast-variant geometry (outDim%8==0, inDim%512==0). Guard with
// gpuHasGeluKernel.
func RMSQMVFastBF16(x, normW, wq, scales, biases []byte, outDim, inDim, groupSize, bits int, eps float32) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(x) != inDim*bf16Size || len(normW) != inDim*bf16Size {
		return nil, core.NewError("native.RMSQMVFastBF16: x and normW must each be inDim bf16 bytes")
	}
	if outDim%8 != 0 || inDim%512 != 0 {
		return nil, core.NewError("native.RMSQMVFastBF16: needs outDim%8==0 and inDim%512==0 (fast variant)")
	}
	pso, err := rmsQMVFastPipeline(groupSize, bits)
	if err != nil {
		return nil, err
	}

	out := make([]byte, outDim*bf16Size)
	withAutoreleasePool(func() {
		wBuf, sBuf, bBuf := sharedBytes(wq), sharedBytes(scales), sharedBytes(biases)
		xBuf, nwBuf := sharedBytes(x), sharedBytes(normW)
		outBuf := device.NewBufferWithLengthOptions(uint(outDim*bf16Size), metal.MTLResourceStorageModeShared)

		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		enc.SetComputePipelineState(pso)
		enc.SetBufferWithOffsetAtIndex(wBuf, 0, 0)
		enc.SetBufferWithOffsetAtIndex(sBuf, 0, 1)
		enc.SetBufferWithOffsetAtIndex(bBuf, 0, 2)
		enc.SetBufferWithOffsetAtIndex(xBuf, 0, 3)
		enc.SetBufferWithOffsetAtIndex(outBuf, 0, 4)
		setEncInt32(enc, int32(inDim), 5)  // K
		setEncInt32(enc, int32(outDim), 6) // N
		enc.SetBufferWithOffsetAtIndex(nwBuf, 0, 7)
		setEncFloat32(enc, eps, 8)
		const bn, bk = 8, 32
		nTgp := (outDim + bn - 1) / bn
		enc.DispatchThreadgroupsThreadsPerThreadgroup(
			metal.MTLSize{Width: 1, Height: uint(nTgp), Depth: 1},
			metal.MTLSize{Width: bk, Height: 2, Depth: 1},
		)
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()
		copy(out, unsafe.Slice((*byte)(outBuf.Contents()), outDim*bf16Size))
	})
	return out, nil
}
