// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"sync"
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

// ropePSOCache memoises rope pipelines keyed by name + the function-constant
// combination (forward/traditional/transpose), since those specialise the
// kernel at build time and a name alone doesn't identify the variant.
var (
	ropePSOMu    sync.Mutex
	ropePSOCache = map[string]metal.MTLComputePipelineState{}
)

// ropePipeline builds (and caches) a rope kernel specialised by MLX's function
// constants: forward (id 1), traditional (id 2), head_seq_transpose (id 3) —
// set at pipeline-build time via MTLFunctionConstantValues, not as buffers. This
// is the first native kernel to use function constants; the plumbing is reusable.
func ropePipeline(name string, traditional bool) (metal.MTLComputePipelineState, error) {
	key := core.Sprintf("%s|trad=%v", name, traditional)
	ropePSOMu.Lock()
	defer ropePSOMu.Unlock()
	if pso, ok := ropePSOCache[key]; ok {
		return pso, nil
	}
	fc := metal.NewMTLFunctionConstantValues()
	fwd, trad, transpose := uint8(1), uint8(0), uint8(0) // forward, !traditional, !transpose
	if traditional {
		trad = 1
	}
	fc.SetConstantValueTypeAtIndex(unsafe.Pointer(&fwd), metal.MTLDataTypeBool, 1)
	fc.SetConstantValueTypeAtIndex(unsafe.Pointer(&trad), metal.MTLDataTypeBool, 2)
	fc.SetConstantValueTypeAtIndex(unsafe.Pointer(&transpose), metal.MTLDataTypeBool, 3)

	fn, err := library.NewFunctionWithNameConstantValuesError(name, fc)
	if err != nil {
		return nil, core.E("native.ropePipeline", name, err)
	}
	pso, err := device.NewComputePipelineStateWithFunctionError(fn)
	if err != nil {
		return nil, core.E("native.ropePipeline", "pipeline "+name, err)
	}
	ropePSOCache[key] = pso
	return pso, nil
}

// RoPE applies rotary position embedding for the single-token (decode) case: x
// is row-major (b, nHeads, 1, headDim), offset is the absolute position, and the
// full headDim is rotated. It drives MLX's rope_single kernel directly (no cgo):
// in(0) out(1) offset(2) scale(3) out_strides[0](4) base(10), with
// forward/traditional/transpose supplied as function constants and base passed
// pre-logged (log2) exactly as MLX does. float32. Byte-for-byte parity with
// pkg/metal.RoPE is gated in parity_test.go.
func RoPE(x []float32, b, nHeads, headDim int, base, scale float32, offset int, traditional bool) ([]float32, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(x) != b*nHeads*headDim {
		return nil, core.NewError("native.RoPE: len(x) must equal b*nHeads*headDim (T=1)")
	}
	if headDim == 0 || nHeads == 0 || b == 0 {
		return make([]float32, len(x)), nil
	}

	pso, err := ropePipeline("rope_single_float32", traditional)
	if err != nil {
		return nil, err
	}

	out := make([]float32, len(x))
	withAutoreleasePool(func() {
		xBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&x[0]), uint(len(x)*4), metal.MTLResourceStorageModeShared)
		outBuf := device.NewBufferWithLengthOptions(uint(len(x)*4), metal.MTLResourceStorageModeShared)
		off := int32(offset)
		offBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&off), 4, metal.MTLResourceStorageModeShared)
		logBase := float32(math.Log2(float64(base)))
		matSize := int64(headDim) // out_strides[0] = T*D, T==1

		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		enc.SetComputePipelineState(pso)
		enc.SetBufferWithOffsetAtIndex(xBuf, 0, 0)
		enc.SetBufferWithOffsetAtIndex(outBuf, 0, 1)
		enc.SetBufferWithOffsetAtIndex(offBuf, 0, 2)
		setEncFloat32(enc, scale, 3)
		setEncInt64(enc, matSize, 4)
		setEncFloat32(enc, logBase, 10)

		// grid (dims/2, N, 1); rope has no cross-thread reduction so the result is
		// threadgroup-invariant — any covering group works (a 1-D group of dim0).
		dim0 := uint(headDim / 2)
		enc.DispatchThreadsThreadsPerThreadgroup(
			metal.MTLSize{Width: dim0, Height: uint(nHeads), Depth: 1},
			metal.MTLSize{Width: dim0, Height: 1, Depth: 1},
		)
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()

		copy(out, unsafe.Slice((*float32)(outBuf.Contents()), len(x)))
	})
	return out, nil
}
