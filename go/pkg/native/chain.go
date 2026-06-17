// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

// This file assembles the parity-proven kernels into on-device sequences: ops
// feed each other through GPU-resident buffers within ONE command buffer, so a
// whole block runs with a single commit and no per-op host round-trip. Metal's
// default hazard tracking orders dependent dispatches via their shared buffers.
// The encode* helpers each encode exactly one dispatch into a caller-supplied
// encoder — the building blocks both the public ops and these chains share.

// shared makes a host-visible GPU buffer holding the given float32 data.
func shared(data []float32) metal.MTLBuffer {
	return device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&data[0]), uint(len(data)*4), metal.MTLResourceStorageModeShared)
}

// scratch makes an uninitialised host-visible GPU buffer of n float32.
func scratch(n int) metal.MTLBuffer {
	return device.NewBufferWithLengthOptions(uint(n*4), metal.MTLResourceStorageModeShared)
}

// encodeRMSNorm encodes a single-row RMSNorm (x·rsqrt(mean(x²)+eps)·w) over
// axisSize elements into enc. Mirrors RMSNorm's binding.
func encodeRMSNorm(enc metal.MTLComputeCommandEncoder, x, w, out metal.MTLBuffer, axisSize int, eps float32) error {
	pso, err := pipelineFor("rmsfloat32")
	if err != nil {
		return err
	}
	enc.SetComputePipelineState(pso)
	enc.SetBufferWithOffsetAtIndex(x, 0, 0)
	enc.SetBufferWithOffsetAtIndex(w, 0, 1)
	enc.SetBufferWithOffsetAtIndex(out, 0, 2)
	setEncFloat32(enc, eps, 3)
	setEncInt32(enc, int32(axisSize), 4)
	setEncInt32(enc, 1, 5)
	tgNeeded := (axisSize + rmsNReads - 1) / rmsNReads
	simdsNeeded := (tgNeeded + rmsSimdSize - 1) / rmsSimdSize
	tg := uint(rmsSimdSize * simdsNeeded)
	enc.DispatchThreadsThreadsPerThreadgroup(
		metal.MTLSize{Width: tg, Height: 1, Depth: 1},
		metal.MTLSize{Width: tg, Height: 1, Depth: 1},
	)
	return nil
}

// encodeGemv encodes out = mat @ vec (mat row-major outDim×inDim, vec inDim)
// into enc. Mirrors MatVec's binding (single size-1 batch).
func encodeGemv(enc metal.MTLComputeCommandEncoder, mat, vec, out metal.MTLBuffer, outDim, inDim int) error {
	bm, bn, sm, sn, tm, tn := gemvTiles(inDim, outDim)
	pso, err := pipelineFor(core.Sprintf("gemv_float32_bm%d_bn%d_sm%d_sn%d_tm%d_tn%d_nc0_axpby0", bm, bn, sm, sn, tm, tn))
	if err != nil {
		return err
	}
	enc.SetComputePipelineState(pso)
	enc.SetBufferWithOffsetAtIndex(mat, 0, 0)
	enc.SetBufferWithOffsetAtIndex(vec, 0, 1)
	enc.SetBufferWithOffsetAtIndex(out, 0, 3)
	setEncInt32(enc, int32(inDim), 4)
	setEncInt32(enc, int32(outDim), 5)
	setEncInt32(enc, int32(inDim), 6)
	setEncInt32(enc, 1, 9)
	setEncInt32(enc, 1, 10)
	setEncInt64(enc, 0, 11)
	setEncInt64(enc, 0, 12)
	nOutPerTgp := bm * sm * tm
	nTgp := (outDim + nOutPerTgp - 1) / nOutPerTgp
	enc.DispatchThreadgroupsThreadsPerThreadgroup(
		metal.MTLSize{Width: uint(nTgp), Height: 1, Depth: 1},
		metal.MTLSize{Width: 32, Height: uint(bn), Depth: uint(bm)},
	)
	return nil
}

// encodeUnary encodes a contiguous unary kernel (v_<Op>float32float32) over n
// elements into enc. Mirrors RunUnary's binding.
func encodeUnary(enc metal.MTLComputeCommandEncoder, name string, in, out metal.MTLBuffer, n int) error {
	pso, err := pipelineFor(name)
	if err != nil {
		return err
	}
	enc.SetComputePipelineState(pso)
	enc.SetBufferWithOffsetAtIndex(in, 0, 0)
	enc.SetBufferWithOffsetAtIndex(out, 0, 1)
	cnt := uint32(n)
	enc.SetBytesLengthAtIndex(unsafe.Slice((*byte)(unsafe.Pointer(&cnt)), 4), 4, 2)
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

// encodeBinary encodes a contiguous binary kernel (vv_<Op>float32) over n
// elements into enc. Mirrors RunBinary's binding.
func encodeBinary(enc metal.MTLComputeCommandEncoder, name string, a, b, out metal.MTLBuffer, n int) error {
	pso, err := pipelineFor(name)
	if err != nil {
		return err
	}
	enc.SetComputePipelineState(pso)
	enc.SetBufferWithOffsetAtIndex(a, 0, 0)
	enc.SetBufferWithOffsetAtIndex(b, 0, 1)
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

// NormProject runs RMSNorm then a matrix projection as one on-device sequence —
// the normalise-then-project that opens every transformer block, intermediate
// resident. Result equals RMSNorm then MatVec separately.
func NormProject(x, normWeight, projWeight []float32, dIn, dOut int, eps float32) ([]float32, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(x) != dIn || len(normWeight) != dIn || len(projWeight) != dOut*dIn {
		return nil, core.NewError("native.NormProject: size mismatch (x/normWeight=dIn, projWeight=dOut*dIn)")
	}

	out := make([]float32, dOut)
	var encErr error
	withAutoreleasePool(func() {
		xBuf, nwBuf, pwBuf := shared(x), shared(normWeight), shared(projWeight)
		tmpBuf, outBuf := scratch(dIn), scratch(dOut)

		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		if encErr = encodeRMSNorm(enc, xBuf, nwBuf, tmpBuf, dIn, eps); encErr != nil {
			enc.EndEncoding()
			return
		}
		if encErr = encodeGemv(enc, pwBuf, tmpBuf, outBuf, dOut, dIn); encErr != nil {
			enc.EndEncoding()
			return
		}
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()
		copy(out, unsafe.Slice((*float32)(outBuf.Contents()), dOut))
	})
	return out, encErr
}

// MLPBlock runs a full gemma feed-forward block on-device in one command buffer:
//
//	normed = rmsnorm(x, normWeight)
//	gate   = Wgate · normed     up = Wup · normed      (dModel → dFF)
//	gated  = gelu(gate) · up    (gelu_approx composed in-line)
//	down   = Wdown · gated      (dFF → dModel)
//	out    = x + down           (residual)
//
// Every intermediate stays resident; ~16 dispatches, one commit. Wgate/Wup are
// row-major (dFF × dModel), Wdown is (dModel × dFF). The result equals the same
// ops via mlx-c — proven in the tests. This is a real decode sub-block on the
// no-cgo path. float32.
func MLPBlock(x, normWeight, wGate, wUp, wDown []float32, dModel, dFF int, eps float32) ([]float32, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(x) != dModel || len(normWeight) != dModel {
		return nil, core.NewError("native.MLPBlock: x/normWeight must be length dModel")
	}
	if len(wGate) != dFF*dModel || len(wUp) != dFF*dModel || len(wDown) != dModel*dFF {
		return nil, core.NewError("native.MLPBlock: projection weight sizes mismatch")
	}

	out := make([]float32, dModel)
	var encErr error
	withAutoreleasePool(func() {
		xBuf := shared(x)
		nwBuf := shared(normWeight)
		wgBuf, wuBuf, wdBuf := shared(wGate), shared(wUp), shared(wDown)
		// gelu scalar operands as dense dFF-length constant buffers.
		c044 := shared(fillConst(dFF, 0.044715))
		c079 := shared(fillConst(dFF, 0.7978845608028654))
		c1 := shared(fillConst(dFF, 1.0))
		c05 := shared(fillConst(dFF, 0.5))
		// intermediates (resident)
		normed := scratch(dModel)
		gate, up := scratch(dFF), scratch(dFF)
		x2, x3, x3s, inner, scaled, t, onePlus, halfG := scratch(dFF), scratch(dFF), scratch(dFF), scratch(dFF), scratch(dFF), scratch(dFF), scratch(dFF), scratch(dFF)
		gelu, gated := scratch(dFF), scratch(dFF)
		down := scratch(dModel)
		outBuf := scratch(dModel)

		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		steps := []func() error{
			func() error { return encodeRMSNorm(enc, xBuf, nwBuf, normed, dModel, eps) },
			func() error { return encodeGemv(enc, wgBuf, normed, gate, dFF, dModel) },
			func() error { return encodeGemv(enc, wuBuf, normed, up, dFF, dModel) },
			// gelu_approx(gate): x2=g·g; x3=x2·g; x3s=0.044715·x3; inner=g+x3s;
			//                    scaled=0.7978…·inner; t=tanh(scaled);
			//                    onePlus=t+1; halfG=0.5·g; gelu=halfG·onePlus
			func() error { return encodeBinary(enc, "vv_Multiplyfloat32", gate, gate, x2, dFF) },
			func() error { return encodeBinary(enc, "vv_Multiplyfloat32", x2, gate, x3, dFF) },
			func() error { return encodeBinary(enc, "vv_Multiplyfloat32", x3, c044, x3s, dFF) },
			func() error { return encodeBinary(enc, "vv_Addfloat32", gate, x3s, inner, dFF) },
			func() error { return encodeBinary(enc, "vv_Multiplyfloat32", inner, c079, scaled, dFF) },
			func() error { return encodeUnary(enc, "v_Tanhfloat32float32", scaled, t, dFF) },
			func() error { return encodeBinary(enc, "vv_Addfloat32", t, c1, onePlus, dFF) },
			func() error { return encodeBinary(enc, "vv_Multiplyfloat32", gate, c05, halfG, dFF) },
			func() error { return encodeBinary(enc, "vv_Multiplyfloat32", halfG, onePlus, gelu, dFF) },
			// gate·up, down projection, residual
			func() error { return encodeBinary(enc, "vv_Multiplyfloat32", gelu, up, gated, dFF) },
			func() error { return encodeGemv(enc, wdBuf, gated, down, dModel, dFF) },
			func() error { return encodeBinary(enc, "vv_Addfloat32", xBuf, down, outBuf, dModel) },
		}
		for _, step := range steps {
			if encErr = step(); encErr != nil {
				enc.EndEncoding()
				return
			}
		}
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()
		copy(out, unsafe.Slice((*float32)(outBuf.Contents()), dModel))
	})
	return out, encErr
}
