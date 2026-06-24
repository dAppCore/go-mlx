// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"github.com/tmc/apple/metal"
)

// dispatchSink abstracts "record one compute dispatch" over the two Metal targets the decode path
// drives: the live MTLComputeCommandEncoder (re-encode every token) and the MTLIndirectComputeCommand
// (record-once ICB replay). An op written against a sink — its pipeline, buffer bindings, and dispatch
// geometry, i.e. the binding ABI — records into EITHER target from ONE body, instead of the two parallel
// emit-helper sets (the live enc* funcs and the ICB recorder's set*/rec* closures) that drifted. That
// drift is not hypothetical: the 12B/31B kvHeads gate sat closed for a long time on a believed-but-false
// recorder divergence that lived in exactly the gap between the two copies.
//
// The asymmetries the sink hides:
//   - scalars: the encoder binds them inline (SetBytes); an ICB command CANNOT, so it binds a buffer.
//     setI32/I64/F32 inline on encSink, bind a process-memoised scalar buffer (scalarI32/…) on icbSink.
//     The emit bodies are generic over the sink (not interface params) so binding adds NO per-call alloc.
//   - dispatch: DispatchThreads* / DispatchThreadgroups* on the encoder vs the ConcurrentDispatch*
//     variants on the ICB command.
//
// What the sink does NOT hide (caller-provided, because they legitimately differ per target):
//   - the pipeline: ICB ops need a supportIndirectCommandBuffers variant (pipelineForICB); the live path
//     uses pipelineFor — different PSO objects for the same kernel, so the caller passes the right one.
//   - per-token-VARYING scalars (the SDPA live length, the sliding read offset): those are the ICB
//     orchestration's rebindable buffers, passed in as buffers; the sink owns only constant scalars.
type dispatchSink interface {
	setPSO(pso metal.MTLComputePipelineState)
	setBuf(buf metal.MTLBuffer, off, idx uint)
	setI32(v int32, idx uint)
	setI64(v int64, idx uint)
	setF32(v float32, idx uint)
	dispatchThreads(grid, group metal.MTLSize)
	dispatchThreadgroups(grid, group metal.MTLSize)
}

// encSink records into a live compute encoder: scalars inline, plain dispatch.
type encSink struct {
	enc metal.MTLComputeCommandEncoder
}

func (s encSink) setPSO(pso metal.MTLComputePipelineState) { s.enc.SetComputePipelineState(pso) }
func (s encSink) setBuf(buf metal.MTLBuffer, off, idx uint) {
	s.enc.SetBufferWithOffsetAtIndex(buf, off, idx)
}
func (s encSink) setI32(v int32, idx uint)   { setEncInt32(s.enc, v, idx) }
func (s encSink) setI64(v int64, idx uint)   { setEncInt64(s.enc, v, idx) }
func (s encSink) setF32(v float32, idx uint) { setEncFloat32(s.enc, v, idx) }
func (s encSink) dispatchThreads(grid, group metal.MTLSize) {
	s.enc.DispatchThreadsThreadsPerThreadgroup(grid, group)
}
func (s encSink) dispatchThreadgroups(grid, group metal.MTLSize) {
	s.enc.DispatchThreadgroupsThreadsPerThreadgroup(grid, group)
}

// icbSink records into an ICB command: scalars bound as (process-memoised) buffers — an ICB command
// cannot SetBytes inline — and concurrent dispatch. The scalar buffers come from scalarI32/I64/F32, which
// memoise by value, so binding a scalar adds no per-record allocation and reuses the recorder's own
// resident scalar handles (created via the same scalar* helpers).
type icbSink struct {
	cmd metal.MTLIndirectComputeCommand
}

func (s icbSink) setPSO(pso metal.MTLComputePipelineState) { s.cmd.SetComputePipelineState(pso) }
func (s icbSink) setBuf(buf metal.MTLBuffer, off, idx uint) {
	s.cmd.SetKernelBufferOffsetAtIndex(buf, off, idx)
}
func (s icbSink) setI32(v int32, idx uint) { s.cmd.SetKernelBufferOffsetAtIndex(scalarI32(v), 0, idx) }
func (s icbSink) setI64(v int64, idx uint) { s.cmd.SetKernelBufferOffsetAtIndex(scalarI64(v), 0, idx) }
func (s icbSink) setF32(v float32, idx uint) {
	s.cmd.SetKernelBufferOffsetAtIndex(scalarF32(v), 0, idx)
}
func (s icbSink) dispatchThreads(grid, group metal.MTLSize) {
	s.cmd.ConcurrentDispatchThreadsThreadsPerThreadgroup(grid, group)
}
func (s icbSink) dispatchThreadgroups(grid, group metal.MTLSize) {
	s.cmd.ConcurrentDispatchThreadgroupsThreadsPerThreadgroup(grid, group)
}

// emitRMSNorm records a single-row bf16 RMSNorm (out = rmsnorm(x, w@wOff), axisSize ≤ the kernel cap)
// through any sink: the binding ABI (x=0, w=1, out=2, eps=3, axisSize=4, ws=5) + a square single-row
// threadgroup. pso + tg are caller-provided — the ICB needs a supportIndirectCommandBuffers pipeline
// and carries its own tg. This is the ONE body behind both encRMSNormBF16 (live, encSink) and the ICB
// recorder's setRMS (icbSink); byte-parity with the re-encode path is gated by the ICB parity suite.
func emitRMSNorm[S dispatchSink](sink S, pso metal.MTLComputePipelineState, x, w, out metal.MTLBuffer, wOff uint, axisSize int, eps float32, tg uint) {
	sink.setPSO(pso)
	sink.setBuf(x, 0, 0)
	sink.setBuf(w, wOff, 1)
	sink.setBuf(out, 0, 2)
	sink.setF32(eps, 3)
	sink.setI32(int32(axisSize), 4)
	sink.setI32(1, 5) // ws (row stride = 1, single row)
	sink.dispatchThreads(metal.MTLSize{Width: tg, Height: 1, Depth: 1}, metal.MTLSize{Width: tg, Height: 1, Depth: 1})
}

// emitRMSNormRows records a per-row bf16 RMSNorm — `rows` independent rows of axisSize each (each at its
// byte offset) — through any sink: same binding ABI as emitRMSNorm (x=0, w=1, out=2, eps=3, axisSize=4,
// ws=5) but dispatched as rows·tg threads in tg-wide groups. The body behind encRMSNormRowsBF16 (live)
// and the recorder's setRMSRows (gemma4 per-head QK-norm). pso + tg caller-provided.
func emitRMSNormRows[S dispatchSink](sink S, pso metal.MTLComputePipelineState, x, w, out metal.MTLBuffer, xOff, wOff, outOff uint, axisSize int, eps float32, rows int, tg uint) {
	sink.setPSO(pso)
	sink.setBuf(x, xOff, 0)
	sink.setBuf(w, wOff, 1)
	sink.setBuf(out, outOff, 2)
	sink.setF32(eps, 3)
	sink.setI32(int32(axisSize), 4)
	sink.setI32(1, 5)
	sink.dispatchThreads(metal.MTLSize{Width: uint(rows) * tg, Height: 1, Depth: 1}, metal.MTLSize{Width: tg, Height: 1, Depth: 1})
}

// emitRMSNormResidual records the FUSED post-norm tail out = res + rmsnorm(x, w@wOff) in one dispatch
// (lthn_rmsnorm_residual_bf16) through any sink: x=0, w=1, res=2, out=3, eps=4, axisSize=5, ws=6. The
// body behind encRMSNormResidualBF16 (live) and the recorder's setRMSResidual. pso + tg caller-provided.
func emitRMSNormResidual[S dispatchSink](sink S, pso metal.MTLComputePipelineState, x, w, res, out metal.MTLBuffer, wOff uint, axisSize int, eps float32, tg uint) {
	sink.setPSO(pso)
	sink.setBuf(x, 0, 0)
	sink.setBuf(w, wOff, 1)
	sink.setBuf(res, 0, 2)
	sink.setBuf(out, 0, 3)
	sink.setF32(eps, 4)
	sink.setI32(int32(axisSize), 5)
	sink.setI32(1, 6)
	sink.dispatchThreads(metal.MTLSize{Width: tg, Height: 1, Depth: 1}, metal.MTLSize{Width: tg, Height: 1, Depth: 1})
}

// emitBinary records an element-wise binary op (vv_Add/vv_Multiply…) out = a⊙b over n elements through
// any sink: a=0, b=1, out=2 (each at its byte offset), count=3, dispatched as n threads in min(n,256)-wide
// groups. The body behind encBinaryDT (live) and the recorder's setBin. pso caller-provided (the ICB
// needs its supportIndirectCommandBuffers variant); the count routes through the sink — inline on the
// encoder, a memoised (resident) scalar buffer on the ICB.
func emitBinary[S dispatchSink](sink S, pso metal.MTLComputePipelineState, a metal.MTLBuffer, aOff uint, b metal.MTLBuffer, bOff uint, out metal.MTLBuffer, oOff uint, n int) {
	sink.setPSO(pso)
	sink.setBuf(a, aOff, 0)
	sink.setBuf(b, bOff, 1)
	sink.setBuf(out, oOff, 2)
	sink.setI32(int32(n), 3)
	g := uint(256)
	if uint(n) < g {
		g = uint(n)
	}
	sink.dispatchThreads(metal.MTLSize{Width: uint(n), Height: 1, Depth: 1}, metal.MTLSize{Width: g, Height: 1, Depth: 1})
}

// emitRope records partial-rotary RoPE (rotated width rd ≤ headDim) over nHeads heads through any sink:
// in=0, out=1, pos=2 (the per-token position buffer — a VARYING buffer the ICB rebinds, passed in), scale=3,
// headStride=4, then EITHER periods@10 + freqStride@11 (the freqs form, periods != nil) OR log2base@10 (the
// base form). 2D dispatch (rd/2 × nHeads). The body behind encRoPEBF16To / encRoPEFreqsBF16To (live) and
// the recorder's setRope. pso caller-provided — the ICB variant, and base vs freqs are different pipelines.
func emitRope[S dispatchSink](sink S, pso metal.MTLComputePipelineState, x, out metal.MTLBuffer, inOff, outOff uint, pos, periods metal.MTLBuffer, nHeads, rd, headDim int, scale, log2base float32) {
	sink.setPSO(pso)
	sink.setBuf(x, inOff, 0)
	sink.setBuf(out, outOff, 1)
	sink.setBuf(pos, 0, 2)
	sink.setF32(scale, 3)
	sink.setI64(int64(headDim), 4)
	if periods != nil {
		sink.setBuf(periods, 0, 10)
		sink.setI64(1, 11) // freq_stride = 1
	} else {
		sink.setF32(log2base, 10)
	}
	d0 := uint(rd / 2)
	sink.dispatchThreads(metal.MTLSize{Width: d0, Height: uint(nHeads), Depth: 1}, metal.MTLSize{Width: d0, Height: 1, Depth: 1})
}

// emitQKNormRope records the FUSED per-head QK-norm + RoPE (out = RoPE(RMSNorm(in, w))) in ONE op through
// any sink: in=0, w=1, out=2, eps=3, headDim=4, rd=5, scale=6, pos=7 (the per-token position buffer), then
// log2base=8, periods=9 (real or a dummy when periods==nil), useFreqs=10 (1/0). One threadgroup per head
// (headDim threads). The body behind encQKNormRope (live) and the recorder's setQKNormRope. `dummy` is the
// caller's bound-but-unread periods buffer for the base form (each path supplies its own — content ignored
// when useFreqs=0). pso caller-provided (ICB variant).
func emitQKNormRope[S dispatchSink](sink S, pso metal.MTLComputePipelineState, x, w, out metal.MTLBuffer, xOff, wOff, outOff uint, pos, periods, dummy metal.MTLBuffer, nHeads, headDim, rd int, eps, scale, log2base float32) {
	sink.setPSO(pso)
	sink.setBuf(x, xOff, 0)
	sink.setBuf(w, wOff, 1)
	sink.setBuf(out, outOff, 2)
	sink.setF32(eps, 3)
	sink.setI32(int32(headDim), 4)
	sink.setI32(int32(rd), 5)
	sink.setF32(scale, 6)
	sink.setBuf(pos, 0, 7)
	sink.setF32(log2base, 8)
	if periods != nil {
		sink.setBuf(periods, 0, 9)
		sink.setI32(1, 10)
	} else {
		sink.setBuf(dummy, 0, 9)
		sink.setI32(0, 10)
	}
	sink.dispatchThreads(metal.MTLSize{Width: uint(nHeads * headDim), Height: 1, Depth: 1}, metal.MTLSize{Width: uint(headDim), Height: 1, Depth: 1})
}

// emitSDPA records single-query single-pass scaled-dot-product attention (the sdpa_vector kernel) through
// any sink: q=0, k=1 (at kvByteOff — the sliding read offset), v=2 (kvByteOff), out=3, gqa=4, N=5,
// strides=6..9, scale=10, one threadgroup per head (1024-wide). The body behind encSDPAStrided (live) and
// the recorder's SDPA op — the op that STARTED the path-unification (the 2-pass had to be wired twice).
//
// N is the one truly per-token-VARYING scalar: the ICB binds its rebindable nBuf (rebound each token at
// replay), the live path inlines the value. So nBuf != nil binds the buffer at 5; nBuf == nil inlines n.
// Everything else is constant (gqa/strides/scale) and routes through the sink's memoised scalars — the
// recorder's gqaOf/sdpaStrideOf/sdpaScaleB buffers ARE those memoised scalars. pso caller-provided.
func emitSDPA[S dispatchSink](sink S, pso metal.MTLComputePipelineState, q, k, v, out metal.MTLBuffer, kvByteOff uint, nBuf metal.MTLBuffer, nHeads, nKVHeads, n int, kHeadStride, kSeqStride, vHeadStride, vSeqStride int64, scale float32) {
	sink.setPSO(pso)
	sink.setBuf(q, 0, 0)
	sink.setBuf(k, kvByteOff, 1)
	sink.setBuf(v, kvByteOff, 2)
	sink.setBuf(out, 0, 3)
	sink.setI32(int32(nHeads/nKVHeads), 4) // gqa_factor
	if nBuf != nil {
		sink.setBuf(nBuf, 0, 5) // ICB: the N buffer, rebound per token at replay
	} else {
		sink.setI32(int32(n), 5) // live: inline N (the live cache length this token)
	}
	sink.setI64(kHeadStride, 6)
	sink.setI64(kSeqStride, 7)
	sink.setI64(vHeadStride, 8)
	sink.setI64(vSeqStride, 9)
	sink.setF32(scale, 10)
	sink.dispatchThreadgroups(metal.MTLSize{Width: uint(nHeads), Height: 1, Depth: 1}, metal.MTLSize{Width: 1024, Height: 1, Depth: 1})
}
