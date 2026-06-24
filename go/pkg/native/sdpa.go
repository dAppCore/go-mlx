// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"sync"
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

// sdpaPSOCache memoises the sdpa_vector pipeline keyed by kernel name. The decode
// path is always no-mask / non-causal / non-transposed / no-sinks, so the six
// function constants are fixed to false; if other combinations are added later,
// fold them into the key.
var (
	sdpaPSOMu    sync.Mutex
	sdpaPSOCache = map[string]metal.MTLComputePipelineState{}
)

// sdpaVectorPipeline builds (and caches) the sdpa_vector kernel with MLX's six
// attention function constants all false (no mask, query not transposed, not
// causal, no bool/float mask, no sinks) — the decode-time configuration.
func sdpaVectorPipeline(name string) (metal.MTLComputePipelineState, error) {
	sdpaPSOMu.Lock()
	defer sdpaPSOMu.Unlock()
	if pso, ok := sdpaPSOCache[name]; ok {
		return pso, nil
	}
	if library == nil || library.GetID() == 0 {
		return nil, core.NewError("native.sdpaVectorPipeline: library unavailable for " + name)
	}
	fc := metal.NewMTLFunctionConstantValues()
	off := uint8(0)
	// indices: has_mask(20) query_transposed(21) do_causal(22) bool_mask(23)
	// float_mask(24) has_sinks(25)
	for _, idx := range []uint{20, 21, 22, 23, 24, 25} {
		fc.SetConstantValueTypeAtIndex(unsafe.Pointer(&off), metal.MTLDataTypeBool, idx)
	}
	fn, err := library.NewFunctionWithNameConstantValuesError(name, fc)
	if err != nil {
		return nil, core.E("native.sdpaVectorPipeline", name, err)
	}
	if fn == nil || fn.GetID() == 0 {
		return nil, core.NewError("native.sdpaVectorPipeline: kernel " + name + " not found")
	}
	pso, err := device.NewComputePipelineStateWithFunctionError(fn)
	if err != nil {
		return nil, core.E("native.sdpaVectorPipeline", "pipeline "+name, err)
	}
	sdpaPSOCache[name] = pso
	return pso, nil
}

// sdpaVector2Pass1Pipeline builds (and caches) the sdpa_vector_2pass_1 kernel —
// attention function constants 20..25 false (decode-time: no mask/transpose/
// causal/sinks) PLUS function constant 26 = blocks (the cache-split count). blocks
// is baked into the pipeline because the kernel indexes the intermediate by it; the
// PSO is keyed by name+blocks so a new block count is a fresh pipeline, not a clash.
func sdpaVector2Pass1Pipeline(name string, blocks int32) (metal.MTLComputePipelineState, error) {
	key := core.Sprintf("%s:b%d", name, blocks)
	sdpaPSOMu.Lock()
	defer sdpaPSOMu.Unlock()
	if pso, ok := sdpaPSOCache[key]; ok {
		return pso, nil
	}
	if library == nil || library.GetID() == 0 {
		return nil, core.NewError("native.sdpaVector2Pass1Pipeline: library unavailable for " + name)
	}
	fc := metal.NewMTLFunctionConstantValues()
	off := uint8(0)
	for _, idx := range []uint{20, 21, 22, 23, 24, 25} { // has_mask query_transposed do_causal bool_mask float_mask has_sinks
		fc.SetConstantValueTypeAtIndex(unsafe.Pointer(&off), metal.MTLDataTypeBool, idx)
	}
	blk := blocks
	fc.SetConstantValueTypeAtIndex(unsafe.Pointer(&blk), metal.MTLDataTypeInt, 26) // blocks
	fn, err := library.NewFunctionWithNameConstantValuesError(name, fc)
	if err != nil {
		return nil, core.E("native.sdpaVector2Pass1Pipeline", name, err)
	}
	if fn == nil || fn.GetID() == 0 {
		return nil, core.NewError("native.sdpaVector2Pass1Pipeline: kernel " + name + " not found")
	}
	pso, err := device.NewComputePipelineStateWithFunctionError(fn)
	if err != nil {
		return nil, core.E("native.sdpaVector2Pass1Pipeline", "pipeline "+name, err)
	}
	sdpaPSOCache[key] = pso
	return pso, nil
}

// sdpaVector2Pass2Pipeline builds (and caches) the sdpa_vector_2pass_2 combine
// kernel. It carries no function constants (MLX builds it plain) — blocks arrives
// as a runtime buffer — so a name-keyed lookup suffices.
func sdpaVector2Pass2Pipeline(name string) (metal.MTLComputePipelineState, error) {
	sdpaPSOMu.Lock()
	defer sdpaPSOMu.Unlock()
	if pso, ok := sdpaPSOCache[name]; ok {
		return pso, nil
	}
	if library == nil || library.GetID() == 0 {
		return nil, core.NewError("native.sdpaVector2Pass2Pipeline: library unavailable for " + name)
	}
	fn := library.NewFunctionWithName(name)
	if fn == nil || fn.GetID() == 0 {
		return nil, core.NewError("native.sdpaVector2Pass2Pipeline: kernel " + name + " not found")
	}
	pso, err := device.NewComputePipelineStateWithFunctionError(fn)
	if err != nil {
		return nil, core.E("native.sdpaVector2Pass2Pipeline", "pipeline "+name, err)
	}
	sdpaPSOCache[name] = pso
	return pso, nil
}

// sdpa2PassBlocks picks the cache-split count for a kvLen — the number of
// threadgroups that share the softmax reduction. Single-pass uses one threadgroup
// per (b·head) and stalls past ~1024 because that one group reduces the whole
// cache; 2-pass fans the reduction over `blocks` groups, so saturation grows with
// context. Must stay a multiple of BN=32 (the pass-2 combine loops blocks/32).
// The ladder mirrors MLX's own heuristic (more blocks as N climbs).
func sdpa2PassBlocks(kvLen int) int32 {
	switch {
	case kvLen <= 8192:
		return 64
	case kvLen <= 32768:
		return 128
	case kvLen <= 65536:
		return 256
	default:
		return 512
	}
}

// SDPA2Pass computes single-query scaled-dot-product attention over a contiguous KV
// cache via MLX's TWO-pass sdpa_vector kernels — the long-context path. Pass 1
// (sdpa_vector_2pass_1) splits the cache into `blocks` segments across threadgroups,
// each emitting a partial weighted-V sum + that segment's online-softmax (sum, max)
// into intermediate buffers; pass 2 (sdpa_vector_2pass_2) merges the per-block
// partials back into one head output. Same inputs/outputs and byte ABI intent as
// SDPA (raw bf16, q (b,nHeads,1,headDim), k/v (b,nKVHeads,kvLen,headDim) → out
// (b,nHeads,1,headDim)) — but it keeps scaling past kvLen~1024 where SDPA's single
// threadgroup-per-head reduction degrades. Token-identical to SDPA (online softmax,
// same maths); validated cosine~1 vs a host float reference in sdpa_2pass_test.go.
func SDPA2Pass(qb, kb, vb []byte, b, nHeads, nKVHeads, headDim, kvLen int, scale float32) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if nKVHeads == 0 || nHeads%nKVHeads != 0 {
		return nil, core.NewError("native.SDPA2Pass: nHeads must be a multiple of nKVHeads")
	}
	gqa := nHeads / nKVHeads
	blocks := sdpa2PassBlocks(kvLen)
	pso1, err := sdpaVector2Pass1Pipeline(core.Sprintf("sdpa_vector_2pass_1_bfloat16_t_%d_%d", headDim, headDim), blocks)
	if err != nil {
		return nil, err
	}
	pso2, err := sdpaVector2Pass2Pipeline(core.Sprintf("sdpa_vector_2pass_2_bfloat16_t_%d", headDim))
	if err != nil {
		return nil, err
	}

	const bf16Size = 2
	outLen := b * nHeads * headDim * bf16Size
	out := make([]byte, outLen)
	withAutoreleasePool(func() {
		qBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&qb[0]), uint(len(qb)), metal.MTLResourceStorageModeShared)
		kBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&kb[0]), uint(len(kb)), metal.MTLResourceStorageModeShared)
		vBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&vb[0]), uint(len(vb)), metal.MTLResourceStorageModeShared)
		// intermediates: partials [b·nHeads·blocks·headDim] bf16, sums/maxs [b·nHeads·blocks] float32.
		nbh := b * nHeads
		partials := device.NewBufferWithLengthOptions(uint(nbh*int(blocks)*headDim*bf16Size), metal.MTLResourceStorageModeShared)
		sums := device.NewBufferWithLengthOptions(uint(nbh*int(blocks)*4), metal.MTLResourceStorageModeShared)
		maxs := device.NewBufferWithLengthOptions(uint(nbh*int(blocks)*4), metal.MTLResourceStorageModeShared)
		outBuf := device.NewBufferWithLengthOptions(uint(outLen), metal.MTLResourceStorageModeShared)

		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder() // serial dispatch: pass 2 sees pass 1's writes
		// Pass 1: per-block partials. Strides in elements; size_t at 8..11. N int at 7.
		enc.SetComputePipelineState(pso1)
		enc.SetBufferWithOffsetAtIndex(qBuf, 0, 0)
		enc.SetBufferWithOffsetAtIndex(kBuf, 0, 1)
		enc.SetBufferWithOffsetAtIndex(vBuf, 0, 2)
		enc.SetBufferWithOffsetAtIndex(partials, 0, 3)
		enc.SetBufferWithOffsetAtIndex(sums, 0, 4)
		enc.SetBufferWithOffsetAtIndex(maxs, 0, 5)
		setEncInt32(enc, int32(kvLen), 7)          // N
		setEncInt64(enc, int64(kvLen*headDim), 8)  // k_head_stride
		setEncInt64(enc, int64(headDim), 9)        // k_seq_stride
		setEncInt64(enc, int64(kvLen*headDim), 10) // v_head_stride
		setEncInt64(enc, int64(headDim), 11)       // v_seq_stride
		setEncFloat32(enc, scale, 12)
		// grid (nKVHeads, b, blocks); group (32, gqa, qseq=1) — each TG spans the gqa query-heads of one kv-head.
		enc.DispatchThreadgroupsThreadsPerThreadgroup(
			metal.MTLSize{Width: uint(nKVHeads), Height: uint(b), Depth: uint(blocks)},
			metal.MTLSize{Width: 32, Height: uint(gqa), Depth: 1},
		)
		// Pass 2: merge per-block partials into the head output.
		enc.SetComputePipelineState(pso2)
		enc.SetBufferWithOffsetAtIndex(partials, 0, 0)
		enc.SetBufferWithOffsetAtIndex(sums, 0, 1)
		enc.SetBufferWithOffsetAtIndex(maxs, 0, 2)
		enc.SetBufferWithOffsetAtIndex(outBuf, 0, 3)
		setEncInt32(enc, blocks, 4)
		enc.DispatchThreadgroupsThreadsPerThreadgroup(
			metal.MTLSize{Width: uint(b * nHeads), Height: 1, Depth: 1},
			metal.MTLSize{Width: 1024, Height: 1, Depth: 1},
		)
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()

		copy(out, unsafe.Slice((*byte)(outBuf.Contents()), outLen))
	})
	return out, nil
}

// SDPA computes single-query scaled-dot-product attention (the decode path) over
// a contiguous KV cache, driving MLX's sdpa_vector kernel directly (no cgo).
// Inputs are raw bfloat16 bytes — the only dtype the decode attention kernel is
// compiled for — laid out as q (b, nHeads, 1, headDim), k/v (b, nKVHeads, kvLen,
// headDim); the result is the bfloat16 output bytes, shape (b, nHeads, 1,
// headDim). nHeads/nKVHeads gives the GQA factor. Buffer ABI: q(0) k(1) v(2)
// out(3) gqa_factor(4) N(5) k_head_stride(6) k_seq_stride(7) v_head_stride(8)
// v_seq_stride(9) scale(10), strides in elements; one threadgroup per (b·head).
// No mask / not causal. Byte-for-byte parity with pkg/metal.ScaledDotProductAttention
// is gated in parity_test.go.
//
// kvLen must stay under 1024 to keep MLX on the single-pass kernel (the 2-pass
// kernel accumulates the softmax differently); decode against a longer cache is
// the sdpa_vector_2pass follow-up.
func SDPA(qb, kb, vb []byte, b, nHeads, nKVHeads, headDim, kvLen int, scale float32) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if nKVHeads == 0 || nHeads%nKVHeads != 0 {
		return nil, core.NewError("native.SDPA: nHeads must be a multiple of nKVHeads")
	}
	name := core.Sprintf("sdpa_vector_bfloat16_t_%d_%d", headDim, headDim)
	pso, err := sdpaVectorPipeline(name)
	if err != nil {
		return nil, err
	}

	const bf16Size = 2
	outLen := b * nHeads * headDim * bf16Size
	out := make([]byte, outLen)
	withAutoreleasePool(func() {
		qBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&qb[0]), uint(len(qb)), metal.MTLResourceStorageModeShared)
		kBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&kb[0]), uint(len(kb)), metal.MTLResourceStorageModeShared)
		vBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&vb[0]), uint(len(vb)), metal.MTLResourceStorageModeShared)
		outBuf := device.NewBufferWithLengthOptions(uint(outLen), metal.MTLResourceStorageModeShared)

		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		enc.SetComputePipelineState(pso)
		enc.SetBufferWithOffsetAtIndex(qBuf, 0, 0)
		enc.SetBufferWithOffsetAtIndex(kBuf, 0, 1)
		enc.SetBufferWithOffsetAtIndex(vBuf, 0, 2)
		enc.SetBufferWithOffsetAtIndex(outBuf, 0, 3)
		setEncInt32(enc, int32(nHeads/nKVHeads), 4) // gqa_factor
		setEncInt32(enc, int32(kvLen), 5)           // N (kv length)
		setEncInt64(enc, int64(kvLen*headDim), 6)   // k_head_stride (elements)
		setEncInt64(enc, int64(headDim), 7)         // k_seq_stride
		setEncInt64(enc, int64(kvLen*headDim), 8)   // v_head_stride
		setEncInt64(enc, int64(headDim), 9)         // v_seq_stride
		setEncFloat32(enc, scale, 10)

		// one threadgroup per (batch · query-head); group is a full 1024-wide
		// simd team as MLX dispatches it.
		enc.DispatchThreadgroupsThreadsPerThreadgroup(
			metal.MTLSize{Width: uint(b * nHeads), Height: 1, Depth: 1},
			metal.MTLSize{Width: 1024, Height: 1, Depth: 1},
		)
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()

		copy(out, unsafe.Slice((*byte)(outBuf.Contents()), outLen))
	})
	return out, nil
}
