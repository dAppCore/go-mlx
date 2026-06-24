// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"sync"
	"unsafe"

	core "dappco.re/go"
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
		if customLibrary == nil || customLibrary.GetID() == 0 {
			geluPSOErr = core.NewError("native.geluPipeline: custom library unavailable")
			return
		}
		fn := customLibrary.NewFunctionWithName("lthn_gelu_gate_mul_bf16")
		if fn == nil || fn.GetID() == 0 {
			geluPSOErr = core.NewError("native.geluPipeline: kernel lthn_gelu_gate_mul_bf16 not found")
			return
		}
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
	// the fused gelu(gate)·up shares the binary-op ABI (in0=0, in1=1, out=2, count=3) — one shared
	// emitBinary body with vv_Add/vv_Multiply and the ICB recorder's gelu op, just a different pipeline.
	emitBinary(encSink{enc}, pso, gate, 0, up, 0, out, 0, n)
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

var (
	bf16MulScalarPSOOnce sync.Once
	bf16MulScalarPSO     metal.MTLComputePipelineState
	bf16MulScalarPSOErr  error
)

func bf16MulScalarPipeline() (metal.MTLComputePipelineState, error) {
	bf16MulScalarPSOOnce.Do(func() {
		if customLibrary == nil || customLibrary.GetID() == 0 {
			bf16MulScalarPSOErr = core.NewError("native.bf16MulScalarPipeline: custom library unavailable")
			return
		}
		fn := customLibrary.NewFunctionWithName("lthn_bf16_mul_scalar")
		if fn == nil || fn.GetID() == 0 {
			bf16MulScalarPSOErr = core.NewError("native.bf16MulScalarPipeline: kernel lthn_bf16_mul_scalar not found")
			return
		}
		bf16MulScalarPSO, bf16MulScalarPSOErr = device.NewComputePipelineStateWithFunctionError(fn)
	})
	return bf16MulScalarPSO, bf16MulScalarPSOErr
}

func encMulScalarBF16(enc metal.MTLComputeCommandEncoder, in, scalar, out metal.MTLBuffer, scalarOffset uint, n int) error {
	if n < 0 {
		return core.NewError("native.encMulScalarBF16: n must be >= 0")
	}
	if n == 0 {
		return nil
	}
	pso, err := bf16MulScalarPipeline()
	if err != nil {
		return err
	}
	enc.SetComputePipelineState(pso)
	enc.SetBufferWithOffsetAtIndex(in, 0, 0)
	enc.SetBufferWithOffsetAtIndex(scalar, scalarOffset, 1)
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

func bf16ScalarBytes(v float32) [bf16Size]byte {
	h := f32ToBF16(v)
	return [bf16Size]byte{byte(h), byte(h >> 8)}
}

func encScaleBF16(enc metal.MTLComputeCommandEncoder, in, scalar, out metal.MTLBuffer, scalarOffset uint, scalarBytes []byte, n int) error {
	if err := encMulScalarBF16(enc, in, scalar, out, scalarOffset, n); err == nil {
		return nil
	}
	return encMulBF16(enc, in, sharedBytes(scalarFillBF16(scalarBytes, n)), out, n)
}

// MulScalarBF16 multiplies each bf16 element in in by one bf16 scalar. When the
// native custom kernels are available it binds the scalar directly, avoiding the
// dense broadcast vector that pkg/metal's scalar bridge also avoids. Without the
// sibling custom metallib it falls back to the existing dense-vector multiply so
// the public operation still works.
func MulScalarBF16(in, scalar []byte) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(in)%bf16Size != 0 {
		return nil, core.NewError("native.MulScalarBF16: input byte length must be a multiple of 2")
	}
	if len(scalar) != bf16Size {
		return nil, core.NewError("native.MulScalarBF16: scalar must be one bf16 value")
	}
	n := len(in) / bf16Size
	out := make([]byte, len(in))
	if n == 0 {
		return out, nil
	}
	var encErr error
	withAutoreleasePool(func() {
		inBuf := sharedBytes(in)
		scalarBuf := sharedBytes(scalar)
		outBuf := scratchBF16(n)
		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		if encErr = encMulScalarBF16(enc, inBuf, scalarBuf, outBuf, 0, n); encErr != nil {
			enc.EndEncoding()
			return
		}
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()
		copy(out, unsafe.Slice((*byte)(outBuf.Contents()), len(out)))
	})
	if encErr == nil {
		return out, nil
	}
	return MulBF16(in, scalarFillBF16(scalar, n))
}

const routerTopKMaxK = 32

var (
	routerTopKPSOOnce sync.Once
	routerTopKPSO     metal.MTLComputePipelineState
	routerTopKPSOErr  error
)

func routerTopKPipeline() (metal.MTLComputePipelineState, error) {
	routerTopKPSOOnce.Do(func() {
		if customLibrary == nil || customLibrary.GetID() == 0 {
			routerTopKPSOErr = core.NewError("native.routerTopKPipeline: custom library unavailable")
			return
		}
		fn := customLibrary.NewFunctionWithName("lthn_moe_router_topk_bf16")
		if fn == nil || fn.GetID() == 0 {
			routerTopKPSOErr = core.NewError("native.routerTopKPipeline: kernel lthn_moe_router_topk_bf16 not found")
			return
		}
		routerTopKPSO, routerTopKPSOErr = device.NewComputePipelineStateWithFunctionError(fn)
	})
	return routerTopKPSO, routerTopKPSOErr
}

func encRouterTopKBF16(enc metal.MTLComputeCommandEncoder, scores, perExpertScale, topIndices, topWeights metal.MTLBuffer, scaleOff uint, numExperts, topK int, hasScale bool) error {
	if topK <= 0 || topK > numExperts || topK > routerTopKMaxK {
		return core.NewError("native.encRouterTopKBF16: topK must be in 1..numExperts and <= 32")
	}
	pso, err := routerTopKPipeline()
	if err != nil {
		return err
	}
	if perExpertScale == nil {
		perExpertScale = scores
	}
	scaleFlag := int32(0)
	if hasScale {
		scaleFlag = 1
	}
	enc.SetComputePipelineState(pso)
	enc.SetBufferWithOffsetAtIndex(scores, 0, 0)
	enc.SetBufferWithOffsetAtIndex(perExpertScale, scaleOff, 1)
	enc.SetBufferWithOffsetAtIndex(topIndices, 0, 2)
	enc.SetBufferWithOffsetAtIndex(topWeights, 0, 3)
	setEncInt32(enc, int32(numExperts), 4)
	setEncInt32(enc, int32(topK), 5)
	setEncInt32(enc, scaleFlag, 6)
	enc.DispatchThreadsThreadsPerThreadgroup(
		metal.MTLSize{Width: 32, Height: 1, Depth: 1},
		metal.MTLSize{Width: 32, Height: 1, Depth: 1},
	)
	return nil
}

func routerTopKBF16(scoresB, perExpertScale []byte, numExperts, topK int) ([]int32, []byte, error) {
	if err := ensureInit(); err != nil {
		return nil, nil, err
	}
	if len(scoresB) != numExperts*bf16Size {
		return nil, nil, core.NewError("native.routerTopKBF16: scores must be numExperts bf16 bytes")
	}
	if perExpertScale != nil && len(perExpertScale) != numExperts*bf16Size {
		return nil, nil, core.NewError("native.routerTopKBF16: perExpertScale must be numExperts bf16 bytes or nil")
	}
	if topK <= 0 || topK > numExperts || topK > routerTopKMaxK {
		return nil, nil, core.NewError("native.routerTopKBF16: topK must be in 1..numExperts and <= 32")
	}
	idx := make([]int32, topK)
	weights := make([]byte, topK*bf16Size)
	var encErr error
	withAutoreleasePool(func() {
		scoresBuf := sharedBytes(scoresB)
		scaleBuf := metal.MTLBuffer(nil)
		if perExpertScale != nil {
			scaleBuf = sharedBytes(perExpertScale)
		}
		idxBuf := device.NewBufferWithLengthOptions(uint(topK*4), metal.MTLResourceStorageModeShared)
		weightBuf := scratchBF16(topK)
		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		if encErr = encRouterTopKBF16(enc, scoresBuf, scaleBuf, idxBuf, weightBuf, 0, numExperts, topK, perExpertScale != nil); encErr != nil {
			enc.EndEncoding()
			return
		}
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()
		copy(idx, unsafe.Slice((*int32)(idxBuf.Contents()), topK))
		copy(weights, unsafe.Slice((*byte)(weightBuf.Contents()), topK*bf16Size))
	})
	if encErr != nil {
		return nil, nil, encErr
	}
	return idx, weights, nil
}

const bf16LMHeadArgmaxRowsPerTile = 8
const bf16LogitsArgmaxRowsPerTile = 256

var (
	bf16LMHeadArgmaxTilesPSOOnce sync.Once
	bf16LMHeadArgmaxTilesPSO     metal.MTLComputePipelineState
	bf16LMHeadArgmaxTilesPSOErr  error
	bf16LogitsArgmaxTilesPSOOnce sync.Once
	bf16LogitsArgmaxTilesPSO     metal.MTLComputePipelineState
	bf16LogitsArgmaxTilesPSOErr  error
	argmaxMergeF32PSOOnce        sync.Once
	argmaxMergeF32PSO            metal.MTLComputePipelineState
	argmaxMergeF32PSOErr         error
)

func bf16LMHeadArgmaxTilesPipeline() (metal.MTLComputePipelineState, error) {
	bf16LMHeadArgmaxTilesPSOOnce.Do(func() {
		if customLibrary == nil || customLibrary.GetID() == 0 {
			bf16LMHeadArgmaxTilesPSOErr = core.NewError("native.bf16LMHeadArgmaxTilesPipeline: custom library unavailable")
			return
		}
		fn := customLibrary.NewFunctionWithName("lthn_bf16_lm_head_argmax_tiles_bf16")
		if fn == nil || fn.GetID() == 0 {
			bf16LMHeadArgmaxTilesPSOErr = core.NewError("native.bf16LMHeadArgmaxTilesPipeline: kernel lthn_bf16_lm_head_argmax_tiles_bf16 not found")
			return
		}
		bf16LMHeadArgmaxTilesPSO, bf16LMHeadArgmaxTilesPSOErr = device.NewComputePipelineStateWithFunctionError(fn)
	})
	return bf16LMHeadArgmaxTilesPSO, bf16LMHeadArgmaxTilesPSOErr
}

func bf16LogitsArgmaxTilesPipeline() (metal.MTLComputePipelineState, error) {
	bf16LogitsArgmaxTilesPSOOnce.Do(func() {
		if customLibrary == nil || customLibrary.GetID() == 0 {
			bf16LogitsArgmaxTilesPSOErr = core.NewError("native.bf16LogitsArgmaxTilesPipeline: custom library unavailable")
			return
		}
		fn := customLibrary.NewFunctionWithName("lthn_bf16_logits_argmax_tiles_bf16")
		if fn == nil || fn.GetID() == 0 {
			bf16LogitsArgmaxTilesPSOErr = core.NewError("native.bf16LogitsArgmaxTilesPipeline: kernel lthn_bf16_logits_argmax_tiles_bf16 not found")
			return
		}
		bf16LogitsArgmaxTilesPSO, bf16LogitsArgmaxTilesPSOErr = device.NewComputePipelineStateWithFunctionError(fn)
	})
	return bf16LogitsArgmaxTilesPSO, bf16LogitsArgmaxTilesPSOErr
}

func argmaxMergeF32Pipeline() (metal.MTLComputePipelineState, error) {
	argmaxMergeF32PSOOnce.Do(func() {
		if customLibrary == nil || customLibrary.GetID() == 0 {
			argmaxMergeF32PSOErr = core.NewError("native.argmaxMergeF32Pipeline: custom library unavailable")
			return
		}
		fn := customLibrary.NewFunctionWithName("lthn_argmax_merge_f32")
		if fn == nil || fn.GetID() == 0 {
			argmaxMergeF32PSOErr = core.NewError("native.argmaxMergeF32Pipeline: kernel lthn_argmax_merge_f32 not found")
			return
		}
		argmaxMergeF32PSO, argmaxMergeF32PSOErr = device.NewComputePipelineStateWithFunctionError(fn)
	})
	return argmaxMergeF32PSO, argmaxMergeF32PSOErr
}

func bf16LMHeadArgmaxUsable(dModel, vocab int) bool {
	if dModel <= 0 || vocab <= 0 {
		return false
	}
	if _, err := bf16LMHeadArgmaxTilesPipeline(); err != nil {
		return false
	}
	if _, err := argmaxMergeF32Pipeline(); err != nil {
		return false
	}
	return true
}

func qmvLogitsArgmaxUsable(dModel, vocab, groupSize, bits int) bool {
	if dModel <= 0 || vocab <= 0 || bits != 4 {
		return false
	}
	if groupSize != 32 && groupSize != 64 && groupSize != 128 {
		return false
	}
	if dModel%groupSize != 0 {
		return false
	}
	if _, err := pipelineFor(qmvBF16KernelName(vocab, dModel, groupSize, bits)); err != nil {
		return false
	}
	if _, err := bf16LogitsArgmaxTilesPipeline(); err != nil {
		return false
	}
	if _, err := argmaxMergeF32Pipeline(); err != nil {
		return false
	}
	return true
}

func encBF16LogitsArgmaxTilesBF16(
	enc metal.MTLComputeCommandEncoder,
	logits, tileValues, tileIndices, suppress metal.MTLBuffer,
	vocab, suppressCount int,
) error {
	if vocab <= 0 {
		return core.NewError("native.encBF16LogitsArgmaxTilesBF16: invalid logits geometry")
	}
	pso, err := bf16LogitsArgmaxTilesPipeline()
	if err != nil {
		return err
	}
	tileCount := (vocab + bf16LogitsArgmaxRowsPerTile - 1) / bf16LogitsArgmaxRowsPerTile
	enc.SetComputePipelineState(pso)
	enc.SetBufferWithOffsetAtIndex(logits, 0, 0)
	enc.SetBufferWithOffsetAtIndex(tileValues, 0, 1)
	enc.SetBufferWithOffsetAtIndex(tileIndices, 0, 2)
	setEncInt32(enc, int32(vocab), 3)
	if suppress == nil {
		suppress = logits
	}
	enc.SetBufferWithOffsetAtIndex(suppress, 0, 4)
	setEncInt32(enc, int32(suppressCount), 5)
	enc.DispatchThreadgroupsThreadsPerThreadgroup(
		metal.MTLSize{Width: uint(tileCount), Height: 1, Depth: 1},
		metal.MTLSize{Width: 32, Height: 1, Depth: 1},
	)
	return nil
}

func encBF16LMHeadArgmaxTilesBF16(
	enc metal.MTLComputeCommandEncoder,
	x, weight, tileValues, tileIndices, suppress metal.MTLBuffer,
	xOff, weightOff uint,
	dModel, vocab, suppressCount int,
) error {
	if dModel <= 0 || vocab <= 0 {
		return core.NewError("native.encBF16LMHeadArgmaxTilesBF16: invalid head geometry")
	}
	pso, err := bf16LMHeadArgmaxTilesPipeline()
	if err != nil {
		return err
	}
	tileCount := (vocab + bf16LMHeadArgmaxRowsPerTile - 1) / bf16LMHeadArgmaxRowsPerTile
	enc.SetComputePipelineState(pso)
	enc.SetBufferWithOffsetAtIndex(x, xOff, 0)
	enc.SetBufferWithOffsetAtIndex(weight, weightOff, 1)
	enc.SetBufferWithOffsetAtIndex(tileValues, 0, 2)
	enc.SetBufferWithOffsetAtIndex(tileIndices, 0, 3)
	setEncInt32(enc, int32(dModel), 4)
	setEncInt32(enc, int32(vocab), 5)
	if suppress == nil {
		suppress = x
	}
	enc.SetBufferWithOffsetAtIndex(suppress, 0, 6)
	setEncInt32(enc, int32(suppressCount), 7)
	enc.DispatchThreadgroupsThreadsPerThreadgroup(
		metal.MTLSize{Width: uint(tileCount), Height: 1, Depth: 1},
		metal.MTLSize{Width: 32, Height: bf16LMHeadArgmaxRowsPerTile, Depth: 1},
	)
	return nil
}

func encArgmaxMergeF32(enc metal.MTLComputeCommandEncoder, values, indices, out metal.MTLBuffer, n int) error {
	if n <= 0 {
		return core.NewError("native.encArgmaxMergeF32: n must be > 0")
	}
	pso, err := argmaxMergeF32Pipeline()
	if err != nil {
		return err
	}
	enc.SetComputePipelineState(pso)
	enc.SetBufferWithOffsetAtIndex(values, 0, 0)
	enc.SetBufferWithOffsetAtIndex(indices, 0, 1)
	enc.SetBufferWithOffsetAtIndex(out, 0, 2)
	setEncInt32(enc, int32(n), 3)
	enc.DispatchThreadsThreadsPerThreadgroup(
		metal.MTLSize{Width: 32, Height: 1, Depth: 1},
		metal.MTLSize{Width: 32, Height: 1, Depth: 1},
	)
	return nil
}
