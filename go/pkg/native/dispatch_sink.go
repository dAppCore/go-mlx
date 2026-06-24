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
//     setI32/I64/F32 inline on encSink, bind a value-keyed pooled buffer on icbSink.
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

// scalarPool memoises constant-scalar buffers by value. An ICB command binds scalars as buffers (it
// cannot SetBytes inline), and the recording reuses ONE buffer per distinct value (eps, an axis size,
// a count) across every op — exactly what the recorder's hand-rolled epsBuf/axisBuf/… buffers did, now
// keyed automatically so a converted op needs no bespoke scalar buffer wired through the closure scope.
type scalarPool struct {
	i32 map[int32]metal.MTLBuffer
	i64 map[int64]metal.MTLBuffer
	f32 map[float32]metal.MTLBuffer
}

func newScalarPool() *scalarPool { return &scalarPool{} } // maps lazily — a pool that only binds i32+f32 never allocates the i64 map

func (p *scalarPool) bufI32(v int32) metal.MTLBuffer {
	if b, ok := p.i32[v]; ok { // nil-map read is safe
		return b
	}
	if p.i32 == nil {
		p.i32 = map[int32]metal.MTLBuffer{}
	}
	b := scalarI32(v)
	p.i32[v] = b
	return b
}

func (p *scalarPool) bufI64(v int64) metal.MTLBuffer {
	if b, ok := p.i64[v]; ok {
		return b
	}
	if p.i64 == nil {
		p.i64 = map[int64]metal.MTLBuffer{}
	}
	b := scalarI64(v)
	p.i64[v] = b
	return b
}

func (p *scalarPool) bufF32(v float32) metal.MTLBuffer {
	if b, ok := p.f32[v]; ok {
		return b
	}
	if p.f32 == nil {
		p.f32 = map[float32]metal.MTLBuffer{}
	}
	b := scalarF32(v)
	p.f32[v] = b
	return b
}

// icbSink records into an ICB command: scalars via the pool, concurrent dispatch.
type icbSink struct {
	cmd  metal.MTLIndirectComputeCommand
	pool *scalarPool
}

func (s icbSink) setPSO(pso metal.MTLComputePipelineState) { s.cmd.SetComputePipelineState(pso) }
func (s icbSink) setBuf(buf metal.MTLBuffer, off, idx uint) {
	s.cmd.SetKernelBufferOffsetAtIndex(buf, off, idx)
}
func (s icbSink) setI32(v int32, idx uint) {
	s.cmd.SetKernelBufferOffsetAtIndex(s.pool.bufI32(v), 0, idx)
}
func (s icbSink) setI64(v int64, idx uint) {
	s.cmd.SetKernelBufferOffsetAtIndex(s.pool.bufI64(v), 0, idx)
}
func (s icbSink) setF32(v float32, idx uint) {
	s.cmd.SetKernelBufferOffsetAtIndex(s.pool.bufF32(v), 0, idx)
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
func emitRMSNorm(sink dispatchSink, pso metal.MTLComputePipelineState, x, w, out metal.MTLBuffer, wOff uint, axisSize int, eps float32, tg uint) {
	sink.setPSO(pso)
	sink.setBuf(x, 0, 0)
	sink.setBuf(w, wOff, 1)
	sink.setBuf(out, 0, 2)
	sink.setF32(eps, 3)
	sink.setI32(int32(axisSize), 4)
	sink.setI32(1, 5) // ws (row stride = 1, single row)
	sink.dispatchThreads(metal.MTLSize{Width: tg, Height: 1, Depth: 1}, metal.MTLSize{Width: tg, Height: 1, Depth: 1})
}
