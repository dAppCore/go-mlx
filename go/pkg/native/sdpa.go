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
