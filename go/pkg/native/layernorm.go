// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

const (
	layerNormNReads      = 8
	layerNormLoopedLimit = 6656
	layerNormSimdSize    = 32
)

// LayerNormBF16 is the byte-parity bf16 LayerNorm (kernel layer_normbfloat16): per row over the last
// axis it computes (x-mean)/sqrt(var+eps)·weight + bias, equalling pkg/metal.LayerNorm on the same
// bf16 arrays. The gemma4 audio subsampler uses a scale-only LayerNorm (bias = zeros) after each
// strided conv. ABI (mlx normalization.cpp): x→0, w→1, b→2, out→3, eps→4, axis_size→5, w_stride→6,
// b_stride→7; one threadgroup per row. Axes up to 6656 use the block kernel; longer axes use MLX's
// looped kernel. weight/bias are length-axisSize bf16.
func LayerNormBF16(x, weight, bias []byte, rows, axisSize int, eps float32) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if axisSize == 0 || len(x) != rows*axisSize*bf16Size {
		return nil, core.NewError("native.LayerNormBF16: len(x) must equal rows*axisSize*2 bytes")
	}
	if len(weight) != axisSize*bf16Size || len(bias) != axisSize*bf16Size {
		return nil, core.NewError("native.LayerNormBF16: weight/bias must be length axisSize bf16")
	}
	name := "layer_normbfloat16"
	if axisSize > layerNormLoopedLimit {
		name = "layer_norm_loopedbfloat16"
	}
	pso, err := pipelineFor(name)
	if err != nil {
		return nil, err
	}

	tg := layerNormThreadgroup(axisSize, pso)

	out := make([]byte, len(x))
	if rows == 0 {
		return out, nil
	}
	withAutoreleasePool(func() {
		xBuf, wBuf, bBuf := sharedBytes(x), sharedBytes(weight), sharedBytes(bias)
		outBuf := scratchBF16(rows * axisSize)
		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		enc.SetComputePipelineState(pso)
		enc.SetBufferWithOffsetAtIndex(xBuf, 0, 0)
		enc.SetBufferWithOffsetAtIndex(wBuf, 0, 1)
		enc.SetBufferWithOffsetAtIndex(bBuf, 0, 2)
		enc.SetBufferWithOffsetAtIndex(outBuf, 0, 3)
		setEncFloat32(enc, eps, 4)
		setEncInt32(enc, int32(axisSize), 5) // axis_size (uint32 ABI; 4 bytes)
		setEncInt32(enc, 1, 6)               // w_stride = 1 (contiguous 1-D weight)
		setEncInt32(enc, 1, 7)               // b_stride = 1 (contiguous 1-D bias)
		enc.DispatchThreadsThreadsPerThreadgroup(
			metal.MTLSize{Width: uint(rows) * tg, Height: 1, Depth: 1},
			metal.MTLSize{Width: tg, Height: 1, Depth: 1},
		)
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()
		copy(out, unsafe.Slice((*byte)(outBuf.Contents()), len(x)))
	})
	return out, nil
}

// LayerNormF32 is the fp32 LayerNorm (kernel layer_normfloat32) — the fp32 sibling of LayerNormBF16,
// matching metal.LayerNorm on fp32 arrays (the subsampler's second LayerNorm runs fp32). weight/bias
// are length-axisSize fp32 (the bf16 model weights widened). Axes above 6656 use MLX's looped kernel.
func LayerNormF32(x, weight, bias []float32, rows, axisSize int, eps float32) ([]float32, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if axisSize == 0 || len(x) != rows*axisSize {
		return nil, core.NewError("native.LayerNormF32: len(x) must equal rows*axisSize")
	}
	if len(weight) != axisSize || len(bias) != axisSize {
		return nil, core.NewError("native.LayerNormF32: weight/bias must be length axisSize")
	}
	name := "layer_normfloat32"
	if axisSize > layerNormLoopedLimit {
		name = "layer_norm_loopedfloat32"
	}
	pso, err := pipelineFor(name)
	if err != nil {
		return nil, err
	}

	tg := layerNormThreadgroup(axisSize, pso)

	out := make([]float32, len(x))
	if rows == 0 {
		return out, nil
	}
	withAutoreleasePool(func() {
		xBuf, wBuf, bBuf := shared(x), shared(weight), shared(bias)
		outBuf := scratch(rows * axisSize)
		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		enc.SetComputePipelineState(pso)
		enc.SetBufferWithOffsetAtIndex(xBuf, 0, 0)
		enc.SetBufferWithOffsetAtIndex(wBuf, 0, 1)
		enc.SetBufferWithOffsetAtIndex(bBuf, 0, 2)
		enc.SetBufferWithOffsetAtIndex(outBuf, 0, 3)
		setEncFloat32(enc, eps, 4)
		setEncInt32(enc, int32(axisSize), 5)
		setEncInt32(enc, 1, 6)
		setEncInt32(enc, 1, 7)
		enc.DispatchThreadsThreadsPerThreadgroup(
			metal.MTLSize{Width: uint(rows) * tg, Height: 1, Depth: 1},
			metal.MTLSize{Width: tg, Height: 1, Depth: 1},
		)
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()
		copy(out, unsafe.Slice((*float32)(outBuf.Contents()), len(x)))
	})
	return out, nil
}

func layerNormThreadgroup(axisSize int, pso metal.MTLComputePipelineState) uint {
	if axisSize > layerNormLoopedLimit {
		return pso.MaxTotalThreadsPerThreadgroup()
	}
	tgNeeded := (axisSize + layerNormNReads - 1) / layerNormNReads
	simdsNeeded := (tgNeeded + layerNormSimdSize - 1) / layerNormSimdSize
	return uint(layerNormSimdSize * simdsNeeded)
}
