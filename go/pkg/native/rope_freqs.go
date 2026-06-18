// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"sync"
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

// rope_freqs.go applies rotary position embedding with EXPLICIT per-dimension
// inverse frequencies — the YaRN long-context spectrum the arch carries in
// RopeFreqs — instead of a single uniform base. It drives MLX's
// rope_single_freqs_bfloat16 kernel, the freqs sibling of rope_single_bfloat16:
// identical buffer ABI except buffer(10) is a per-dim frequency array (not the
// log2 base) and buffer(11) its stride. The kernel reads inv_freq = 1/freqs[d],
// so the buffer holds the reciprocal of the inverse frequencies (the periods);
// RoPEFreqsBF16 inverts the caller's inv-freqs into that form.

var (
	ropeFreqsPSOBF16Cache = map[string]metal.MTLComputePipelineState{}
	ropeFreqsPSOBF16Mu    sync.Mutex
)

// ropeFreqsPipelineBF16 builds (and caches) the rope_single_freqs_bfloat16 kernel,
// specialised by the same forward/traditional/transpose function constants as the
// base rope_single_bfloat16 pipeline (both call rope_single_impl).
func ropeFreqsPipelineBF16(traditional bool) (metal.MTLComputePipelineState, error) {
	key := core.Sprintf("rope_single_freqs_bfloat16|trad=%v", traditional)
	ropeFreqsPSOBF16Mu.Lock()
	defer ropeFreqsPSOBF16Mu.Unlock()
	if pso, ok := ropeFreqsPSOBF16Cache[key]; ok {
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
	fn, err := library.NewFunctionWithNameConstantValuesError("rope_single_freqs_bfloat16", fc)
	if err != nil {
		return nil, core.E("native.ropeFreqsPipelineBF16", "rope_single_freqs_bfloat16", err)
	}
	pso, err := device.NewComputePipelineStateWithFunctionError(fn)
	if err != nil {
		return nil, core.E("native.ropeFreqsPipelineBF16", "pipeline rope_single_freqs_bfloat16", err)
	}
	ropeFreqsPSOBF16Cache[key] = pso
	return pso, nil
}

// RoPEFreqsBF16 is the explicit-frequency sibling of RoPEDimsBF16: it applies
// rotary embedding to x (bf16 bytes, row-major (b, nHeads, 1, headDim)) at
// absolute position offset, rotating the first rotaryDim of each head with the
// per-dim inverse frequencies invFreqs (len rotaryDim/2 — the arch's RopeFreqs),
// the tail [rotaryDim:headDim] passing through. The kernel uses inv_freq =
// 1/freqs[d], so invFreqs is inverted into the periods it expects. When invFreqs
// is the plain-rope spectrum (base^(-2d/rotaryDim)) the result is identical to
// RoPEDimsBF16 with that base — gated in rope_freqs_test.go.
//
//	out, err := native.RoPEFreqsBF16(xBytes, 1, 8, 128, 128, yarnInvFreqs, 1, pos, false)
func RoPEFreqsBF16(x []byte, b, nHeads, headDim, rotaryDim int, invFreqs []float32, scale float32, offset int, traditional bool) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(x) != b*nHeads*headDim*bf16Size {
		return nil, core.NewError("native.RoPEFreqsBF16: len(x) must equal b*nHeads*headDim*2 bytes (T=1)")
	}
	if headDim == 0 || nHeads == 0 || b == 0 {
		return make([]byte, len(x)), nil
	}
	if rotaryDim <= 0 || rotaryDim > headDim || rotaryDim%2 != 0 {
		return nil, core.NewError("native.RoPEFreqsBF16: rotaryDim must be even and in (0, headDim]")
	}
	if len(invFreqs) != rotaryDim/2 {
		return nil, core.NewError("native.RoPEFreqsBF16: len(invFreqs) must equal rotaryDim/2")
	}

	pso, err := ropeFreqsPipelineBF16(traditional)
	if err != nil {
		return nil, err
	}

	// the kernel computes inv_freq = 1/freqs[d]; hand it the reciprocal (periods).
	periods := make([]float32, len(invFreqs))
	for i, f := range invFreqs {
		periods[i] = 1.0 / f
	}

	out := make([]byte, len(x))
	withAutoreleasePool(func() {
		xBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&x[0]), uint(len(x)), metal.MTLResourceStorageModeShared)
		var outBuf metal.MTLBuffer
		if rotaryDim < headDim {
			// partial: seed out with x so the non-rotated tail passes through.
			outBuf = device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&x[0]), uint(len(x)), metal.MTLResourceStorageModeShared)
		} else {
			outBuf = device.NewBufferWithLengthOptions(uint(len(x)), metal.MTLResourceStorageModeShared)
		}
		off := int32(offset)
		offBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&off), 4, metal.MTLResourceStorageModeShared)
		freqsBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&periods[0]), uint(len(periods)*4), metal.MTLResourceStorageModeShared)
		matSize := int64(headDim) // out_strides[0] = T*D, T==1 (full head stride; tail lives in this row)

		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		enc.SetComputePipelineState(pso)
		enc.SetBufferWithOffsetAtIndex(xBuf, 0, 0)
		enc.SetBufferWithOffsetAtIndex(outBuf, 0, 1)
		enc.SetBufferWithOffsetAtIndex(offBuf, 0, 2)
		setEncFloat32(enc, scale, 3)
		setEncInt64(enc, matSize, 4)
		enc.SetBufferWithOffsetAtIndex(freqsBuf, 0, 10)
		setEncInt64(enc, 1, 11) // freq_stride = 1 (contiguous, one freq per rotated dim)

		dim0 := uint(rotaryDim / 2)
		enc.DispatchThreadsThreadsPerThreadgroup(
			metal.MTLSize{Width: dim0, Height: uint(nHeads), Depth: 1},
			metal.MTLSize{Width: dim0, Height: 1, Depth: 1},
		)
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()

		copy(out, unsafe.Slice((*byte)(outBuf.Contents()), len(x)))
	})
	return out, nil
}
