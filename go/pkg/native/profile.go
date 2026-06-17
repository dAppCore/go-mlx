// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"time"

	"github.com/tmc/apple/metal"
)

// profileForward, when set, makes DecodeForwardICB accumulate pure GPU execution
// time (Σ per-token GPUEndTime-GPUStartTime) into profForwardGPUSec — a profiling
// side-channel to read the GPU fraction of steady-state per-token cost. Set only
// by the profiling tests (reset profForwardGPUSec first); production leaves it off.
var (
	profileForward    bool
	profForwardGPUSec float64
)

// dispatchProfile breaks the per-dispatch cost of the no-cgo path into its three
// parts by encoding nDispatch trivial bf16 adds (vecLen elements each) into ONE
// command buffer and reading the GPU timestamps:
//
//   - encode  — host time to encode the nDispatch ops (what ICB replay removes)
//   - gpuSec  — pure GPU execution, GPUEndTime-GPUStartTime (what FUSION removes:
//     fewer/bigger dispatches → fewer kernel launches)
//   - run     — Commit→WaitUntilCompleted wall; run-gpuSec is the fixed commit/wait
//     sync (amortised across the whole token, not per dispatch)
//
// The decode forward is ~24 dispatches/layer; at E2B scale ~840/token. This says
// which term dominates that 26 µs/dispatch — i.e. whether fusion (cut GPU launches)
// or the already-built encode-bypass (cut host encode) is the lever, with evidence
// rather than assumption.
func dispatchProfile(nDispatch, vecLen int) (encode, run time.Duration, gpuSec float64, err error) {
	if err = ensureInit(); err != nil {
		return
	}
	pso, e := pipelineFor("vv_Addbfloat16")
	if e != nil {
		err = e
		return
	}
	withAutoreleasePool(func() {
		a, b, o := scratchBF16(vecLen), scratchBF16(vecLen), scratchBF16(vecLen)
		group := uint(256)
		if uint(vecLen) < group {
			group = uint(vecLen)
		}
		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		enc.SetComputePipelineState(pso)
		tEnc := time.Now()
		for i := 0; i < nDispatch; i++ {
			enc.SetBufferWithOffsetAtIndex(a, 0, 0)
			enc.SetBufferWithOffsetAtIndex(b, 0, 1)
			enc.SetBufferWithOffsetAtIndex(o, 0, 2)
			setEncInt32(enc, int32(vecLen), 3)
			enc.DispatchThreadsThreadsPerThreadgroup(
				metal.MTLSize{Width: uint(vecLen), Height: 1, Depth: 1},
				metal.MTLSize{Width: group, Height: 1, Depth: 1},
			)
		}
		enc.EndEncoding()
		encode = time.Since(tEnc)
		tRun := time.Now()
		cb.Commit()
		cb.WaitUntilCompleted()
		run = time.Since(tRun)
		gpuSec = float64(cb.GPUEndTime() - cb.GPUStartTime())
	})
	return
}

// rebindCostProbe records a one-command ICB once, then times M re-sets of its
// output buffer OFFSET (SetKernelBufferOffsetAtIndex) — the per-token cache-grow
// rebind. The decode forward does 2·nLayers of these per token (~70 at E2B);
// if each is expensive (the driver re-validates the command) they, not the GPU,
// dominate the per-token wall. Returns total time for M re-sets.
func rebindCostProbe(M int) (time.Duration, error) {
	if err := ensureInit(); err != nil {
		return 0, err
	}
	pso, err := pipelineForICB("vv_Addbfloat16")
	if err != nil {
		return 0, err
	}
	var dur time.Duration
	withAutoreleasePool(func() {
		a, b, o := scratchBF16(64), scratchBF16(64), scratchBF16(64*8)
		cntB := scalarI32(64)
		icbDesc := metal.NewMTLIndirectCommandBufferDescriptor()
		icbDesc.SetCommandTypes(metal.MTLIndirectCommandTypeConcurrentDispatch)
		icbDesc.SetInheritBuffers(false)
		icbDesc.SetInheritPipelineState(false)
		icbDesc.SetMaxKernelBufferBindCount(4)
		icb := device.NewIndirectCommandBufferWithDescriptorMaxCommandCountOptions(icbDesc, 1, metal.MTLResourceStorageModeShared)
		c0 := icb.IndirectComputeCommandAtIndex(0)
		c0.SetComputePipelineState(pso)
		c0.SetKernelBufferOffsetAtIndex(a, 0, 0)
		c0.SetKernelBufferOffsetAtIndex(b, 0, 1)
		c0.SetKernelBufferOffsetAtIndex(o, 0, 2)
		c0.SetKernelBufferOffsetAtIndex(cntB, 0, 3)
		t0 := time.Now()
		for i := 0; i < M; i++ {
			c0.SetKernelBufferOffsetAtIndex(o, uint((i%8)*64*bf16Size), 2)
		}
		dur = time.Since(t0)
	})
	return dur, nil
}

// gemvProfile measures the GPU time of an (outDim×inDim) bf16 gemv repeated
// nDispatch times in one command buffer — the decode forward's dominant op (a
// matvec reads the whole weight matrix once per token, so decode is weight-read
// bandwidth-bound). Returns total GPU seconds and the bytes read per dispatch
// (the weight matrix), so the caller can compute effective GB/s vs the device
// peak — the evidence for whether the lever is fewer bytes (4-bit weights) rather
// than fewer/fused dispatches.
func gemvProfile(outDim, inDim, nDispatch int) (gpuSec float64, weightBytes int, err error) {
	if err = ensureInit(); err != nil {
		return
	}
	weightBytes = outDim * inDim * bf16Size
	withAutoreleasePool(func() {
		mat, vec, out := scratchBF16(outDim*inDim), scratchBF16(inDim), scratchBF16(outDim)
		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		for i := 0; i < nDispatch; i++ {
			if e := encGemvBF16(enc, mat, vec, out, outDim, inDim); e != nil {
				err = e
				enc.EndEncoding()
				return
			}
		}
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()
		gpuSec = float64(cb.GPUEndTime() - cb.GPUStartTime())
	})
	return
}
