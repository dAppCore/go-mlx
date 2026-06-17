// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"sync"
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/foundation"
	"github.com/tmc/apple/metal"
)

// Indirect Command Buffer (ICB) replay: record a fixed op sequence ONCE, then
// re-issue it per step with a single executeCommandsInBuffer call instead of
// re-encoding every op on the host. This is the encode-bypass lever — a decode
// step's command sequence is fixed across tokens, so recording it once skips the
// per-token host re-encode. Two things differ from the regular encode path:
//   - ICB commands bind only BUFFERS, never inline setBytes — so every scalar
//     parameter becomes a tiny persistent buffer (scalarI32/scalarI64/scalarF32);
//   - the ICB has no automatic hazard tracking, so dependent commands need an
//     explicit SetBarrier, and the replay encoder must mark every referenced
//     buffer resident with UseResource.
//
// NormProjectICB de-risks the mechanism on a real dependent 2-op sequence
// (rms→gemv) before it scales to the full DecodeLayer.

// icbPSOCache memoises ICB-capable pipelines (built with
// supportIndirectCommandBuffers=true, required for a kernel to run inside an ICB).
var (
	icbPSOMu    sync.Mutex
	icbPSOCache = map[string]metal.MTLComputePipelineState{}
)

// pipelineForICB builds (and caches) an ICB-capable pipeline for a metallib
// kernel — same kernel as pipelineFor, but the descriptor sets
// supportIndirectCommandBuffers so it can be recorded into an indirect command.
func pipelineForICB(name string) (metal.MTLComputePipelineState, error) {
	icbPSOMu.Lock()
	defer icbPSOMu.Unlock()
	if pso, ok := icbPSOCache[name]; ok {
		return pso, nil
	}
	fn := library.NewFunctionWithName(name)
	desc := metal.NewMTLComputePipelineDescriptor()
	desc.SetComputeFunction(fn)
	desc.SetSupportIndirectCommandBuffers(true)
	pso, err := device.NewComputePipelineStateWithDescriptorOptionsReflectionError(desc, 0, nil)
	if err != nil {
		return nil, core.E("native.pipelineForICB", name, err)
	}
	icbPSOCache[name] = pso
	return pso, nil
}

// squareICB records the contiguous Square kernel once into an ICB and replays it
// — the smallest real ICB (one op, in/out + a scalar count as a buffer) to isolate
// the basic mechanism (ICB-capable PSO + scalar-as-buffer + residency + execute)
// from the multi-op barrier path. Returns in[i]² if the ICB executes correctly.
func squareICB(in []float32) ([]float32, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	pso, err := pipelineForICB("v_Squarefloat32float32")
	if err != nil {
		return nil, err
	}
	n := len(in)
	out := make([]float32, n)
	withAutoreleasePool(func() {
		inBuf := shared(in)
		outBuf := scratch(n)
		sizeBuf := scalarI32(int32(n))

		icbDesc := metal.NewMTLIndirectCommandBufferDescriptor()
		icbDesc.SetCommandTypes(metal.MTLIndirectCommandTypeConcurrentDispatch)
		icbDesc.SetInheritBuffers(false)
		icbDesc.SetInheritPipelineState(false)
		icbDesc.SetMaxKernelBufferBindCount(4)
		icb := device.NewIndirectCommandBufferWithDescriptorMaxCommandCountOptions(icbDesc, 1, metal.MTLResourceStorageModeShared)

		c0 := icb.IndirectComputeCommandAtIndex(0)
		c0.SetComputePipelineState(pso)
		c0.SetKernelBufferOffsetAtIndex(inBuf, 0, 0)
		c0.SetKernelBufferOffsetAtIndex(outBuf, 0, 1)
		c0.SetKernelBufferOffsetAtIndex(sizeBuf, 0, 2)
		group := uint(256)
		if uint(n) < group {
			group = uint(n)
		}
		c0.ConcurrentDispatchThreadsThreadsPerThreadgroup(
			metal.MTLSize{Width: uint(n), Height: 1, Depth: 1},
			metal.MTLSize{Width: group, Height: 1, Depth: 1},
		)

		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		enc.UseResourceUsage(inBuf, metal.MTLResourceUsageRead)
		enc.UseResourceUsage(outBuf, metal.MTLResourceUsageRead|metal.MTLResourceUsageWrite)
		enc.UseResourceUsage(sizeBuf, metal.MTLResourceUsageRead)
		enc.ExecuteCommandsInBufferWithRange(icb, foundation.NSRange{Location: 0, Length: 1})
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()
		copy(out, unsafe.Slice((*float32)(outBuf.Contents()), n))
	})
	return out, nil
}

// gemvICB records the gemv kernel once into an ICB and replays it — isolates
// gemv-in-ICB (threadgroups dispatch, 10 buffer binds incl. scalars) from the
// multi-op path. Returns mat @ vec if correct.
func gemvICB(mat, vec []float32, outDim, inDim int) ([]float32, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	bm, bn, sm, sn, tm, tn := gemvTiles(inDim, outDim)
	pso, err := pipelineForICB(core.Sprintf("gemv_float32_bm%d_bn%d_sm%d_sn%d_tm%d_tn%d_nc0_axpby0", bm, bn, sm, sn, tm, tn))
	if err != nil {
		return nil, err
	}
	out := make([]float32, outDim)
	withAutoreleasePool(func() {
		matBuf, vecBuf := shared(mat), shared(vec)
		outBuf := scratch(outDim)
		inB, outB, ldB := scalarI32(int32(inDim)), scalarI32(int32(outDim)), scalarI32(int32(inDim))
		bndB, bshB, vsB, msB := scalarI32(1), scalarI32(1), scalarI64(0), scalarI64(0)

		icbDesc := metal.NewMTLIndirectCommandBufferDescriptor()
		icbDesc.SetCommandTypes(metal.MTLIndirectCommandTypeConcurrentDispatch)
		icbDesc.SetInheritBuffers(false)
		icbDesc.SetInheritPipelineState(false)
		icbDesc.SetMaxKernelBufferBindCount(16)
		icb := device.NewIndirectCommandBufferWithDescriptorMaxCommandCountOptions(icbDesc, 1, metal.MTLResourceStorageModeShared)

		c0 := icb.IndirectComputeCommandAtIndex(0)
		c0.SetComputePipelineState(pso)
		c0.SetKernelBufferOffsetAtIndex(matBuf, 0, 0)
		c0.SetKernelBufferOffsetAtIndex(vecBuf, 0, 1)
		c0.SetKernelBufferOffsetAtIndex(outBuf, 0, 3)
		c0.SetKernelBufferOffsetAtIndex(inB, 0, 4)
		c0.SetKernelBufferOffsetAtIndex(outB, 0, 5)
		c0.SetKernelBufferOffsetAtIndex(ldB, 0, 6)
		c0.SetKernelBufferOffsetAtIndex(bndB, 0, 9)
		c0.SetKernelBufferOffsetAtIndex(bshB, 0, 10)
		c0.SetKernelBufferOffsetAtIndex(vsB, 0, 11)
		c0.SetKernelBufferOffsetAtIndex(msB, 0, 12)
		nOutPerTgp := bm * sm * tm
		nTgp := (outDim + nOutPerTgp - 1) / nOutPerTgp
		c0.ConcurrentDispatchThreadgroupsThreadsPerThreadgroup(
			metal.MTLSize{Width: uint(nTgp), Height: 1, Depth: 1},
			metal.MTLSize{Width: 32, Height: uint(bn), Depth: uint(bm)},
		)

		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		for _, b := range []metal.MTLBuffer{matBuf, vecBuf, outBuf, inB, outB, ldB, bndB, bshB, vsB, msB} {
			enc.UseResourceUsage(b, metal.MTLResourceUsageRead|metal.MTLResourceUsageWrite)
		}
		enc.ExecuteCommandsInBufferWithRange(icb, foundation.NSRange{Location: 0, Length: 1})
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()
		copy(out, unsafe.Slice((*float32)(outBuf.Contents()), outDim))
	})
	return out, nil
}

// rebindProbeICB records ONE gemv command into an ICB, then replays it nRows
// times — re-setting only the output buffer's OFFSET (SetKernelBufferOffsetAtIndex
// at index 3) between replays so replay r writes row r of a tall output buffer.
// It is the smallest test of the cache-grow lever: an ICB's command bindings are
// recorded once, but re-setting one buffer offset per replay is far cheaper than
// re-encoding, and IS the mechanism the growing KV cache needs (the per-token
// write row advances while the rest of the command stays recorded). Returns the
// nRows×outDim output; every row must equal mat@vec if the rebind takes effect.
func rebindProbeICB(mat, vec []float32, outDim, inDim, nRows int) ([]float32, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	bm, bn, sm, sn, tm, tn := gemvTiles(inDim, outDim)
	pso, err := pipelineForICB(core.Sprintf("gemv_float32_bm%d_bn%d_sm%d_sn%d_tm%d_tn%d_nc0_axpby0", bm, bn, sm, sn, tm, tn))
	if err != nil {
		return nil, err
	}
	out := make([]float32, nRows*outDim)
	withAutoreleasePool(func() {
		matBuf, vecBuf := shared(mat), shared(vec)
		outBuf := scratch(nRows * outDim)
		inB, outDimB, ldB := scalarI32(int32(inDim)), scalarI32(int32(outDim)), scalarI32(int32(inDim))
		bndB, bshB, vsB, msB := scalarI32(1), scalarI32(1), scalarI64(0), scalarI64(0)

		icbDesc := metal.NewMTLIndirectCommandBufferDescriptor()
		icbDesc.SetCommandTypes(metal.MTLIndirectCommandTypeConcurrentDispatch)
		icbDesc.SetInheritBuffers(false)
		icbDesc.SetInheritPipelineState(false)
		icbDesc.SetMaxKernelBufferBindCount(16)
		icb := device.NewIndirectCommandBufferWithDescriptorMaxCommandCountOptions(icbDesc, 1, metal.MTLResourceStorageModeShared)

		c0 := icb.IndirectComputeCommandAtIndex(0)
		c0.SetComputePipelineState(pso)
		c0.SetKernelBufferOffsetAtIndex(matBuf, 0, 0)
		c0.SetKernelBufferOffsetAtIndex(vecBuf, 0, 1)
		c0.SetKernelBufferOffsetAtIndex(outBuf, 0, 3) // offset re-set per replay below
		c0.SetKernelBufferOffsetAtIndex(inB, 0, 4)
		c0.SetKernelBufferOffsetAtIndex(outDimB, 0, 5)
		c0.SetKernelBufferOffsetAtIndex(ldB, 0, 6)
		c0.SetKernelBufferOffsetAtIndex(bndB, 0, 9)
		c0.SetKernelBufferOffsetAtIndex(bshB, 0, 10)
		c0.SetKernelBufferOffsetAtIndex(vsB, 0, 11)
		c0.SetKernelBufferOffsetAtIndex(msB, 0, 12)
		nOutPerTgp := bm * sm * tm
		nTgp := (outDim + nOutPerTgp - 1) / nOutPerTgp

		for r := 0; r < nRows; r++ {
			// the only per-replay change: advance the output row (4 bytes/f32)
			c0.SetKernelBufferOffsetAtIndex(outBuf, uint(r*outDim*4), 3)
			cb := queue.CommandBuffer()
			enc := cb.ComputeCommandEncoder()
			for _, b := range []metal.MTLBuffer{matBuf, vecBuf, outBuf, inB, outDimB, ldB, bndB, bshB, vsB, msB} {
				enc.UseResourceUsage(b, metal.MTLResourceUsageRead|metal.MTLResourceUsageWrite)
			}
			c0.ConcurrentDispatchThreadgroupsThreadsPerThreadgroup(
				metal.MTLSize{Width: uint(nTgp), Height: 1, Depth: 1},
				metal.MTLSize{Width: 32, Height: uint(bn), Depth: uint(bm)},
			)
			enc.ExecuteCommandsInBufferWithRange(icb, foundation.NSRange{Location: 0, Length: 1})
			enc.EndEncoding()
			cb.Commit()
			cb.WaitUntilCompleted()
		}
		copy(out, unsafe.Slice((*float32)(outBuf.Contents()), nRows*outDim))
	})
	return out, nil
}

// qmvICB records the bf16-activation 4-bit qmv ONCE into an ICB and replays it —
// the smallest proof that affine_qmv_bfloat16_t works as an INDIRECT command. It's
// a plain named kernel (no function constants, unlike rope/sdpa), so pipelineForICB
// should build it ICB-capable directly; this confirms that plus the w0 s1 b2 x3
// out4 K5 N6 binding as an ICB command. The qmv projection swap in the cache-grow
// ICB rests on this. Returns out = x @ Wᵀ; must equal QMVBF16 on the same bytes.
func qmvICB(x, wq, scales, biases []byte, outDim, inDim, groupSize, bits int) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	variant := "_qmv_"
	if outDim%8 == 0 && inDim%512 == 0 {
		variant = "_qmv_fast_"
	}
	pso, err := pipelineForICB(core.Sprintf("affine%sbfloat16_t_gs_%d_b_%d_batch_0", variant, groupSize, bits))
	if err != nil {
		return nil, err
	}
	out := make([]byte, outDim*bf16Size)
	withAutoreleasePool(func() {
		wBuf, sBuf, bBuf := sharedBytes(wq), sharedBytes(scales), sharedBytes(biases)
		xBuf := sharedBytes(x)
		outBuf := scratchBF16(outDim)
		kB, nB := scalarI32(int32(inDim)), scalarI32(int32(outDim))

		icbDesc := metal.NewMTLIndirectCommandBufferDescriptor()
		icbDesc.SetCommandTypes(metal.MTLIndirectCommandTypeConcurrentDispatch)
		icbDesc.SetInheritBuffers(false)
		icbDesc.SetInheritPipelineState(false)
		icbDesc.SetMaxKernelBufferBindCount(8)
		icb := device.NewIndirectCommandBufferWithDescriptorMaxCommandCountOptions(icbDesc, 1, metal.MTLResourceStorageModeShared)

		c0 := icb.IndirectComputeCommandAtIndex(0)
		c0.SetComputePipelineState(pso)
		c0.SetKernelBufferOffsetAtIndex(wBuf, 0, 0)
		c0.SetKernelBufferOffsetAtIndex(sBuf, 0, 1)
		c0.SetKernelBufferOffsetAtIndex(bBuf, 0, 2)
		c0.SetKernelBufferOffsetAtIndex(xBuf, 0, 3)
		c0.SetKernelBufferOffsetAtIndex(outBuf, 0, 4)
		c0.SetKernelBufferOffsetAtIndex(kB, 0, 5)
		c0.SetKernelBufferOffsetAtIndex(nB, 0, 6)
		const bn, bk = 8, 32
		nTgp := (outDim + bn - 1) / bn
		c0.ConcurrentDispatchThreadgroupsThreadsPerThreadgroup(
			metal.MTLSize{Width: 1, Height: uint(nTgp), Depth: 1},
			metal.MTLSize{Width: bk, Height: 2, Depth: 1},
		)

		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		for _, b := range []metal.MTLBuffer{wBuf, sBuf, bBuf, xBuf, outBuf, kB, nB} {
			enc.UseResourceUsage(b, metal.MTLResourceUsageRead|metal.MTLResourceUsageWrite)
		}
		enc.ExecuteCommandsInBufferWithRange(icb, foundation.NSRange{Location: 0, Length: 1})
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()
		copy(out, unsafe.Slice((*byte)(outBuf.Contents()), outDim*bf16Size))
	})
	return out, nil
}

// ropePipelineICB / sdpaVectorPipelineICB are the ICB-capable, function-constant
// pipelines — the new wrinkle for the attention ICB: combine the specialised
// function (func consts) with the ICB descriptor (supportIndirectCommandBuffers).
func ropePipelineICB(traditional bool) (metal.MTLComputePipelineState, error) {
	key := core.Sprintf("rope_single_bfloat16|icb|trad=%v", traditional)
	icbPSOMu.Lock()
	defer icbPSOMu.Unlock()
	if pso, ok := icbPSOCache[key]; ok {
		return pso, nil
	}
	fc := metal.NewMTLFunctionConstantValues()
	fwd, trad, transpose := uint8(1), uint8(0), uint8(0)
	if traditional {
		trad = 1
	}
	fc.SetConstantValueTypeAtIndex(unsafe.Pointer(&fwd), metal.MTLDataTypeBool, 1)
	fc.SetConstantValueTypeAtIndex(unsafe.Pointer(&trad), metal.MTLDataTypeBool, 2)
	fc.SetConstantValueTypeAtIndex(unsafe.Pointer(&transpose), metal.MTLDataTypeBool, 3)
	fn, err := library.NewFunctionWithNameConstantValuesError("rope_single_bfloat16", fc)
	if err != nil {
		return nil, core.E("native.ropePipelineICB", "function", err)
	}
	desc := metal.NewMTLComputePipelineDescriptor()
	desc.SetComputeFunction(fn)
	desc.SetSupportIndirectCommandBuffers(true)
	pso, err := device.NewComputePipelineStateWithDescriptorOptionsReflectionError(desc, 0, nil)
	if err != nil {
		return nil, core.E("native.ropePipelineICB", "pipeline", err)
	}
	icbPSOCache[key] = pso
	return pso, nil
}

func sdpaVectorPipelineICB(name string) (metal.MTLComputePipelineState, error) {
	key := name + "|icb"
	icbPSOMu.Lock()
	defer icbPSOMu.Unlock()
	if pso, ok := icbPSOCache[key]; ok {
		return pso, nil
	}
	fc := metal.NewMTLFunctionConstantValues()
	off := uint8(0)
	for _, idx := range []uint{20, 21, 22, 23, 24, 25} {
		fc.SetConstantValueTypeAtIndex(unsafe.Pointer(&off), metal.MTLDataTypeBool, idx)
	}
	fn, err := library.NewFunctionWithNameConstantValuesError(name, fc)
	if err != nil {
		return nil, core.E("native.sdpaVectorPipelineICB", name, err)
	}
	desc := metal.NewMTLComputePipelineDescriptor()
	desc.SetComputeFunction(fn)
	desc.SetSupportIndirectCommandBuffers(true)
	pso, err := device.NewComputePipelineStateWithDescriptorOptionsReflectionError(desc, 0, nil)
	if err != nil {
		return nil, core.E("native.sdpaVectorPipelineICB", "pipeline "+name, err)
	}
	icbPSOCache[key] = pso
	return pso, nil
}

// AttentionBlockICB records the bf16 attention block once into an ICB and replays
// it `replays` times — proving ICB replay across a real func-const multi-op chain
// (rms→gemv→rope→sdpa→gemv→add), every scalar a buffer, a barrier on each
// consumer, residency on every buffer. With replays=1 it must equal AttentionBlock
// byte-for-byte. Inputs/outputs raw bf16 bytes; same shapes as AttentionBlock.
func AttentionBlockICB(x, normWeight, wQ, wO, kCache, vCache []byte, dModel, nHeads, nKVHeads, headDim, kvLen int, base, scale float32, offset int, eps float32, replays int) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if replays < 1 {
		replays = 1
	}
	qDim := nHeads * headDim

	rmsPSO, err := pipelineForICB("rmsbfloat16")
	if err != nil {
		return nil, err
	}
	bmQ, bnQ, smQ, snQ, tmQ, tnQ := gemvTiles(dModel, qDim)
	gemvQPSO, err := pipelineForICB(core.Sprintf("gemv_bfloat16_bm%d_bn%d_sm%d_sn%d_tm%d_tn%d_nc0_axpby0", bmQ, bnQ, smQ, snQ, tmQ, tnQ))
	if err != nil {
		return nil, err
	}
	bmO, bnO, smO, snO, tmO, tnO := gemvTiles(qDim, dModel)
	gemvOPSO, err := pipelineForICB(core.Sprintf("gemv_bfloat16_bm%d_bn%d_sm%d_sn%d_tm%d_tn%d_nc0_axpby0", bmO, bnO, smO, snO, tmO, tnO))
	if err != nil {
		return nil, err
	}
	ropePSO, err := ropePipelineICB(false)
	if err != nil {
		return nil, err
	}
	sdpaPSO, err := sdpaVectorPipelineICB(core.Sprintf("sdpa_vector_bfloat16_t_%d_%d", headDim, headDim))
	if err != nil {
		return nil, err
	}
	addPSO, err := pipelineForICB("vv_Addbfloat16")
	if err != nil {
		return nil, err
	}

	out := make([]byte, dModel*bf16Size)
	withAutoreleasePool(func() {
		// data buffers
		xBuf, nwBuf := sharedBytes(x), sharedBytes(normWeight)
		wqBuf, woBuf := sharedBytes(wQ), sharedBytes(wO)
		kBuf, vBuf := sharedBytes(kCache), sharedBytes(vCache)
		normed := scratchBF16(dModel)
		q, qr, attn := scratchBF16(qDim), scratchBF16(qDim), scratchBF16(qDim)
		attnOut, outBuf := scratchBF16(dModel), scratchBF16(dModel)
		// scalar buffers
		off := int32(offset)
		offBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&off), 4, metal.MTLResourceStorageModeShared)
		epsBuf, axisBuf, wsBuf := scalarF32(eps), scalarI32(int32(dModel)), scalarI32(1)
		// gemv Q scalars
		qInB, qOutB, qLdB := scalarI32(int32(dModel)), scalarI32(int32(qDim)), scalarI32(int32(dModel))
		// gemv O scalars
		oInB, oOutB, oLdB := scalarI32(int32(qDim)), scalarI32(int32(dModel)), scalarI32(int32(qDim))
		bndB, bshB, vsB, msB := scalarI32(1), scalarI32(1), scalarI64(0), scalarI64(0)
		// rope scalars
		ropeScaleB := scalarF32(scale)
		ropeMatB := scalarI64(int64(headDim))
		ropeBaseB := scalarF32(float32(math.Log2(float64(base))))
		// sdpa scalars
		gqaB, nB := scalarI32(int32(nHeads/nKVHeads)), scalarI32(int32(kvLen))
		khsB, kssB := scalarI64(int64(kvLen*headDim)), scalarI64(int64(headDim))
		vhsB, vssB := scalarI64(int64(kvLen*headDim)), scalarI64(int64(headDim))
		sdpaScaleB := scalarF32(scale)
		// add scalar
		addCntB := scalarI32(int32(dModel))

		resident := []metal.MTLBuffer{
			xBuf, nwBuf, wqBuf, woBuf, kBuf, vBuf, normed, q, qr, attn, attnOut, outBuf,
			offBuf, epsBuf, axisBuf, wsBuf, qInB, qOutB, qLdB, oInB, oOutB, oLdB, bndB, bshB, vsB, msB,
			ropeScaleB, ropeMatB, ropeBaseB, gqaB, nB, khsB, kssB, vhsB, vssB, sdpaScaleB, addCntB,
		}

		icbDesc := metal.NewMTLIndirectCommandBufferDescriptor()
		icbDesc.SetCommandTypes(metal.MTLIndirectCommandTypeConcurrentDispatch)
		icbDesc.SetInheritBuffers(false)
		icbDesc.SetInheritPipelineState(false)
		icbDesc.SetMaxKernelBufferBindCount(16)
		icb := device.NewIndirectCommandBufferWithDescriptorMaxCommandCountOptions(icbDesc, 6, metal.MTLResourceStorageModeShared)

		rmsTG := uint(rmsSimdSize * ((((dModel + rmsNReads - 1) / rmsNReads) + rmsSimdSize - 1) / rmsSimdSize))
		gemvGrid := func(outDim, bm, sm, tm int) uint { return uint((outDim + bm*sm*tm - 1) / (bm * sm * tm)) }

		// 0: rms x -> normed
		c := icb.IndirectComputeCommandAtIndex(0)
		c.SetComputePipelineState(rmsPSO)
		c.SetKernelBufferOffsetAtIndex(xBuf, 0, 0)
		c.SetKernelBufferOffsetAtIndex(nwBuf, 0, 1)
		c.SetKernelBufferOffsetAtIndex(normed, 0, 2)
		c.SetKernelBufferOffsetAtIndex(epsBuf, 0, 3)
		c.SetKernelBufferOffsetAtIndex(axisBuf, 0, 4)
		c.SetKernelBufferOffsetAtIndex(wsBuf, 0, 5)
		c.ConcurrentDispatchThreadsThreadsPerThreadgroup(metal.MTLSize{Width: rmsTG, Height: 1, Depth: 1}, metal.MTLSize{Width: rmsTG, Height: 1, Depth: 1})

		// 1: gemv Wq @ normed -> q
		c = icb.IndirectComputeCommandAtIndex(1)
		c.SetBarrier()
		c.SetComputePipelineState(gemvQPSO)
		c.SetKernelBufferOffsetAtIndex(wqBuf, 0, 0)
		c.SetKernelBufferOffsetAtIndex(normed, 0, 1)
		c.SetKernelBufferOffsetAtIndex(q, 0, 3)
		c.SetKernelBufferOffsetAtIndex(qInB, 0, 4)
		c.SetKernelBufferOffsetAtIndex(qOutB, 0, 5)
		c.SetKernelBufferOffsetAtIndex(qLdB, 0, 6)
		c.SetKernelBufferOffsetAtIndex(bndB, 0, 9)
		c.SetKernelBufferOffsetAtIndex(bshB, 0, 10)
		c.SetKernelBufferOffsetAtIndex(vsB, 0, 11)
		c.SetKernelBufferOffsetAtIndex(msB, 0, 12)
		c.ConcurrentDispatchThreadgroupsThreadsPerThreadgroup(metal.MTLSize{Width: gemvGrid(qDim, bmQ, smQ, tmQ), Height: 1, Depth: 1}, metal.MTLSize{Width: 32, Height: uint(bnQ), Depth: uint(bmQ)})

		// 2: rope q -> qr
		c = icb.IndirectComputeCommandAtIndex(2)
		c.SetBarrier()
		c.SetComputePipelineState(ropePSO)
		c.SetKernelBufferOffsetAtIndex(q, 0, 0)
		c.SetKernelBufferOffsetAtIndex(qr, 0, 1)
		c.SetKernelBufferOffsetAtIndex(offBuf, 0, 2)
		c.SetKernelBufferOffsetAtIndex(ropeScaleB, 0, 3)
		c.SetKernelBufferOffsetAtIndex(ropeMatB, 0, 4)
		c.SetKernelBufferOffsetAtIndex(ropeBaseB, 0, 10)
		ropeDim0 := uint(headDim / 2)
		c.ConcurrentDispatchThreadsThreadsPerThreadgroup(metal.MTLSize{Width: ropeDim0, Height: uint(nHeads), Depth: 1}, metal.MTLSize{Width: ropeDim0, Height: 1, Depth: 1})

		// 3: sdpa qr, k, v -> attn
		c = icb.IndirectComputeCommandAtIndex(3)
		c.SetBarrier()
		c.SetComputePipelineState(sdpaPSO)
		c.SetKernelBufferOffsetAtIndex(qr, 0, 0)
		c.SetKernelBufferOffsetAtIndex(kBuf, 0, 1)
		c.SetKernelBufferOffsetAtIndex(vBuf, 0, 2)
		c.SetKernelBufferOffsetAtIndex(attn, 0, 3)
		c.SetKernelBufferOffsetAtIndex(gqaB, 0, 4)
		c.SetKernelBufferOffsetAtIndex(nB, 0, 5)
		c.SetKernelBufferOffsetAtIndex(khsB, 0, 6)
		c.SetKernelBufferOffsetAtIndex(kssB, 0, 7)
		c.SetKernelBufferOffsetAtIndex(vhsB, 0, 8)
		c.SetKernelBufferOffsetAtIndex(vssB, 0, 9)
		c.SetKernelBufferOffsetAtIndex(sdpaScaleB, 0, 10)
		c.ConcurrentDispatchThreadgroupsThreadsPerThreadgroup(metal.MTLSize{Width: uint(nHeads), Height: 1, Depth: 1}, metal.MTLSize{Width: 1024, Height: 1, Depth: 1})

		// 4: gemv Wo @ attn -> attnOut
		c = icb.IndirectComputeCommandAtIndex(4)
		c.SetBarrier()
		c.SetComputePipelineState(gemvOPSO)
		c.SetKernelBufferOffsetAtIndex(woBuf, 0, 0)
		c.SetKernelBufferOffsetAtIndex(attn, 0, 1)
		c.SetKernelBufferOffsetAtIndex(attnOut, 0, 3)
		c.SetKernelBufferOffsetAtIndex(oInB, 0, 4)
		c.SetKernelBufferOffsetAtIndex(oOutB, 0, 5)
		c.SetKernelBufferOffsetAtIndex(oLdB, 0, 6)
		c.SetKernelBufferOffsetAtIndex(bndB, 0, 9)
		c.SetKernelBufferOffsetAtIndex(bshB, 0, 10)
		c.SetKernelBufferOffsetAtIndex(vsB, 0, 11)
		c.SetKernelBufferOffsetAtIndex(msB, 0, 12)
		c.ConcurrentDispatchThreadgroupsThreadsPerThreadgroup(metal.MTLSize{Width: gemvGrid(dModel, bmO, smO, tmO), Height: 1, Depth: 1}, metal.MTLSize{Width: 32, Height: uint(bnO), Depth: uint(bmO)})

		// 5: add x + attnOut -> out
		c = icb.IndirectComputeCommandAtIndex(5)
		c.SetBarrier()
		c.SetComputePipelineState(addPSO)
		c.SetKernelBufferOffsetAtIndex(xBuf, 0, 0)
		c.SetKernelBufferOffsetAtIndex(attnOut, 0, 1)
		c.SetKernelBufferOffsetAtIndex(outBuf, 0, 2)
		c.SetKernelBufferOffsetAtIndex(addCntB, 0, 3)
		addGroup := uint(256)
		if uint(dModel) < addGroup {
			addGroup = uint(dModel)
		}
		c.ConcurrentDispatchThreadsThreadsPerThreadgroup(metal.MTLSize{Width: uint(dModel), Height: 1, Depth: 1}, metal.MTLSize{Width: addGroup, Height: 1, Depth: 1})

		rng := foundation.NSRange{Location: 0, Length: 6}
		for r := 0; r < replays; r++ {
			cb := queue.CommandBuffer()
			enc := cb.ComputeCommandEncoder()
			for _, b := range resident {
				enc.UseResourceUsage(b, metal.MTLResourceUsageRead|metal.MTLResourceUsageWrite)
			}
			enc.ExecuteCommandsInBufferWithRange(icb, rng)
			enc.EndEncoding()
			cb.Commit()
			cb.WaitUntilCompleted()
		}
		copy(out, unsafe.Slice((*byte)(outBuf.Contents()), len(out)))
	})
	return out, nil
}

func scalarI32(v int32) metal.MTLBuffer {
	return device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&v), 4, metal.MTLResourceStorageModeShared)
}

func scalarI64(v int64) metal.MTLBuffer {
	return device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&v), 8, metal.MTLResourceStorageModeShared)
}

func scalarF32(v float32) metal.MTLBuffer {
	return device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&v), 4, metal.MTLResourceStorageModeShared)
}

// NormProjectICB computes the same rms→projection as NormProject, but records the
// two ops once into an ICB and replays it `replays` times (the decode-loop
// pattern). Returns the output of the final replay. With replays=1 it must equal
// NormProject byte-for-byte — same kernels, same data, only the submission path
// differs. Proves ICB replay on a real dependent sequence (scalar-as-buffer +
// SetBarrier + residency). float32.
func NormProjectICB(x, normWeight, projWeight []float32, dIn, dOut int, eps float32, replays int) ([]float32, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(x) != dIn || len(normWeight) != dIn || len(projWeight) != dOut*dIn {
		return nil, core.NewError("native.NormProjectICB: size mismatch")
	}
	if replays < 1 {
		replays = 1
	}

	rmsPSO, err := pipelineForICB("rmsfloat32")
	if err != nil {
		return nil, err
	}
	bm, bn, sm, sn, tm, tn := gemvTiles(dIn, dOut)
	gemvPSO, err := pipelineForICB(core.Sprintf("gemv_float32_bm%d_bn%d_sm%d_sn%d_tm%d_tn%d_nc0_axpby0", bm, bn, sm, sn, tm, tn))
	if err != nil {
		return nil, err
	}

	out := make([]float32, dOut)
	withAutoreleasePool(func() {
		// persistent data buffers
		xBuf, nwBuf, pwBuf := shared(x), shared(normWeight), shared(projWeight)
		tmpBuf, outBuf := scratch(dIn), scratch(dOut)
		// scalar params as buffers (ICB can't setBytes inline)
		epsBuf, axisBuf, wsBuf := scalarF32(eps), scalarI32(int32(dIn)), scalarI32(1)
		inBuf, outdimBuf, ldBuf := scalarI32(int32(dIn)), scalarI32(int32(dOut)), scalarI32(int32(dIn))
		bndBuf, bshBuf, vsBuf, msBuf := scalarI32(1), scalarI32(1), scalarI64(0), scalarI64(0)
		resident := []metal.MTLBuffer{xBuf, nwBuf, pwBuf, tmpBuf, outBuf, epsBuf, axisBuf, wsBuf, inBuf, outdimBuf, ldBuf, bndBuf, bshBuf, vsBuf, msBuf}

		// record the 2-op sequence once
		icbDesc := metal.NewMTLIndirectCommandBufferDescriptor()
		icbDesc.SetCommandTypes(metal.MTLIndirectCommandTypeConcurrentDispatch)
		icbDesc.SetInheritBuffers(false)
		icbDesc.SetInheritPipelineState(false)
		icbDesc.SetMaxKernelBufferBindCount(16) // gemv binds up to index 12
		icb := device.NewIndirectCommandBufferWithDescriptorMaxCommandCountOptions(icbDesc, 2, metal.MTLResourceStorageModeShared)

		// cmd 0: rmsnorm  x -> tmp
		c0 := icb.IndirectComputeCommandAtIndex(0)
		c0.SetComputePipelineState(rmsPSO)
		c0.SetKernelBufferOffsetAtIndex(xBuf, 0, 0)
		c0.SetKernelBufferOffsetAtIndex(nwBuf, 0, 1)
		c0.SetKernelBufferOffsetAtIndex(tmpBuf, 0, 2)
		c0.SetKernelBufferOffsetAtIndex(epsBuf, 0, 3)
		c0.SetKernelBufferOffsetAtIndex(axisBuf, 0, 4)
		c0.SetKernelBufferOffsetAtIndex(wsBuf, 0, 5)
		tg := uint(rmsSimdSize * ((((dIn + rmsNReads - 1) / rmsNReads) + rmsSimdSize - 1) / rmsSimdSize))
		c0.ConcurrentDispatchThreadsThreadsPerThreadgroup(
			metal.MTLSize{Width: tg, Height: 1, Depth: 1},
			metal.MTLSize{Width: tg, Height: 1, Depth: 1},
		)

		// cmd 1: gemv  projW @ tmp -> out
		c1 := icb.IndirectComputeCommandAtIndex(1)
		c1.SetBarrier() // wait for c0's tmp write to be visible before reading it
		c1.SetComputePipelineState(gemvPSO)
		c1.SetKernelBufferOffsetAtIndex(pwBuf, 0, 0)
		c1.SetKernelBufferOffsetAtIndex(tmpBuf, 0, 1)
		c1.SetKernelBufferOffsetAtIndex(outBuf, 0, 3)
		c1.SetKernelBufferOffsetAtIndex(inBuf, 0, 4)
		c1.SetKernelBufferOffsetAtIndex(outdimBuf, 0, 5)
		c1.SetKernelBufferOffsetAtIndex(ldBuf, 0, 6)
		c1.SetKernelBufferOffsetAtIndex(bndBuf, 0, 9)
		c1.SetKernelBufferOffsetAtIndex(bshBuf, 0, 10)
		c1.SetKernelBufferOffsetAtIndex(vsBuf, 0, 11)
		c1.SetKernelBufferOffsetAtIndex(msBuf, 0, 12)
		nOutPerTgp := bm * sm * tm
		nTgp := (dOut + nOutPerTgp - 1) / nOutPerTgp
		c1.ConcurrentDispatchThreadgroupsThreadsPerThreadgroup(
			metal.MTLSize{Width: uint(nTgp), Height: 1, Depth: 1},
			metal.MTLSize{Width: 32, Height: uint(bn), Depth: uint(bm)},
		)

		// replay the recorded sequence
		rng := foundation.NSRange{Location: 0, Length: 2}
		for r := 0; r < replays; r++ {
			cb := queue.CommandBuffer()
			enc := cb.ComputeCommandEncoder()
			for _, b := range resident {
				enc.UseResourceUsage(b, metal.MTLResourceUsageRead|metal.MTLResourceUsageWrite)
			}
			enc.ExecuteCommandsInBufferWithRange(icb, rng)
			enc.EndEncoding()
			cb.Commit()
			cb.WaitUntilCompleted()
		}
		copy(out, unsafe.Slice((*float32)(outBuf.Contents()), dOut))
	})
	return out, nil
}
