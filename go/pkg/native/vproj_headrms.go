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
	vprojHeadRMSPSOMu    sync.Mutex
	vprojHeadRMSPSOCache = map[string]metal.MTLComputePipelineState{}
)

func vprojHeadRMSPipeline(groupSize, bits int, icb bool) (metal.MTLComputePipelineState, error) {
	name := core.Sprintf("lthn_vproj_headrms_bfloat16_t_gs_%d_b_%d", groupSize, bits)
	key := name
	if icb {
		key += "|icb"
	}
	vprojHeadRMSPSOMu.Lock()
	defer vprojHeadRMSPSOMu.Unlock()
	if pso, ok := vprojHeadRMSPSOCache[key]; ok {
		return pso, nil
	}
	if customLibrary == nil || customLibrary.GetID() == 0 {
		return nil, core.NewError("native.vprojHeadRMSPipeline: custom library unavailable")
	}
	fn := customLibrary.NewFunctionWithName(name)
	if fn == nil || fn.GetID() == 0 {
		return nil, core.NewError("native.vprojHeadRMSPipeline: kernel " + name + " not found")
	}
	if !icb {
		pso, err := device.NewComputePipelineStateWithFunctionError(fn)
		if err != nil {
			return nil, core.E("native.vprojHeadRMSPipeline", key, err)
		}
		vprojHeadRMSPSOCache[key] = pso
		return pso, nil
	}
	desc := metal.NewMTLComputePipelineDescriptor()
	desc.SetComputeFunction(fn)
	desc.SetSupportIndirectCommandBuffers(true)
	pso, err := device.NewComputePipelineStateWithDescriptorOptionsReflectionError(desc, 0, nil)
	if err != nil {
		return nil, core.E("native.vprojHeadRMSPipeline", key, err)
	}
	vprojHeadRMSPSOCache[key] = pso
	return pso, nil
}

// VProjHeadRMSBF16 fuses, in ONE dispatch, the whole gemma4 V path: input-RMSNorm(x, inNormW) → 4-bit
// V projection → per-head value-norm (RMS over headDim). One threadgroup per KV head. Equal (cosine ~1.0,
// lockstep) to RMSNormBF16(QMVBF16(RMSNormBF16(x, inNormW)), ones, nKVHeads, headDim). headDim must be a
// power of two ≤ 1024 (the in-kernel tree reductions). x/inNormW are inDim bf16 bytes; out is
// nKVHeads·headDim bf16 bytes.
func VProjHeadRMSBF16(x, inNormW, wq, scales, biases []byte, nKVHeads, headDim, inDim, groupSize, bits int, eps float32) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(x) != inDim*bf16Size || len(inNormW) != inDim*bf16Size {
		return nil, core.NewError("native.VProjHeadRMSBF16: x and inNormW must each be inDim bf16 bytes")
	}
	if headDim <= 0 || headDim > 1024 || headDim&(headDim-1) != 0 {
		return nil, core.NewError("native.VProjHeadRMSBF16: headDim must be a power of two ≤ 1024")
	}
	pso, err := vprojHeadRMSPipeline(groupSize, bits, false)
	if err != nil {
		return nil, err
	}

	outDim := nKVHeads * headDim
	out := make([]byte, outDim*bf16Size)
	withAutoreleasePool(func() {
		wBuf, sBuf, bBuf := sharedBytes(wq), sharedBytes(scales), sharedBytes(biases)
		xBuf, nwBuf := sharedBytes(x), sharedBytes(inNormW)
		outBuf := device.NewBufferWithLengthOptions(uint(outDim*bf16Size), metal.MTLResourceStorageModeShared)

		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		enc.SetComputePipelineState(pso)
		enc.SetBufferWithOffsetAtIndex(wBuf, 0, 0)
		enc.SetBufferWithOffsetAtIndex(sBuf, 0, 1)
		enc.SetBufferWithOffsetAtIndex(bBuf, 0, 2)
		enc.SetBufferWithOffsetAtIndex(xBuf, 0, 3)
		enc.SetBufferWithOffsetAtIndex(nwBuf, 0, 4)
		enc.SetBufferWithOffsetAtIndex(outBuf, 0, 5)
		setEncInt32(enc, int32(inDim), 6)
		setEncFloat32(enc, eps, 8) // buffer 7 (head_dim) is implicit = threads-per-threadgroup
		enc.DispatchThreadgroupsThreadsPerThreadgroup(
			metal.MTLSize{Width: uint(nKVHeads), Height: 1, Depth: 1},
			metal.MTLSize{Width: uint(headDim), Height: 1, Depth: 1},
		)
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()
		copy(out, unsafe.Slice((*byte)(outBuf.Contents()), outDim*bf16Size))
	})
	return out, nil
}
