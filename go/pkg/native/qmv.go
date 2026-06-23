// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

// QMV computes out = x @ Wᵀ for a 4-bit (affine) quantised weight matrix — the
// 4-bit decode hot path. wq/scales/biases are the raw packed bytes MLX's
// quantiser produces for a logically (outDim x inDim) weight; x is a length-inDim
// float32 activation vector; the result is length outDim. It drives MLX's
// affine_qmv kernel directly through the no-cgo path: w(0) scales(1) biases(2)
// x(3) out(4) K(5) N(6) — and because this is a single (B<=1) matvec, MLX's
// add_strides_and_shapes early-returns, so there are no batch params to set.
// group_size and bits are baked into the kernel name. float32 activations only.
//
// Byte-for-byte parity with pkg/metal.QuantizedMatmul (transpose=true) on the
// same packed bytes is gated in parity_test.go.
func QMV(x []float32, wq, scales, biases []byte, outDim, inDim, groupSize, bits int) ([]float32, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(x) != inDim {
		return nil, core.NewError("native.QMV: len(x) must equal inDim")
	}
	if outDim == 0 || inDim == 0 {
		return make([]float32, outDim), nil
	}

	// fast variant when the matrix tiles cleanly (mlx: N%bn==0 && K%512==0, bn=8).
	variant := "_qmv_"
	if outDim%8 == 0 && inDim%512 == 0 {
		variant = "_qmv_fast_"
	}
	name := core.Sprintf("affine%sfloat_gs_%d_b_%d_batch_0", variant, groupSize, bits)
	pso, err := pipelineFor(name)
	if err != nil {
		return nil, err
	}

	out := make([]float32, outDim)
	withAutoreleasePool(func() {
		wBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&wq[0]), uint(len(wq)), metal.MTLResourceStorageModeShared)
		sBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&scales[0]), uint(len(scales)), metal.MTLResourceStorageModeShared)
		bBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&biases[0]), uint(len(biases)), metal.MTLResourceStorageModeShared)
		xBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&x[0]), uint(len(x)*4), metal.MTLResourceStorageModeShared)
		outBuf := device.NewBufferWithLengthOptions(uint(outDim*4), metal.MTLResourceStorageModeShared)

		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		enc.SetComputePipelineState(pso)
		enc.SetBufferWithOffsetAtIndex(wBuf, 0, 0)
		enc.SetBufferWithOffsetAtIndex(sBuf, 0, 1)
		enc.SetBufferWithOffsetAtIndex(bBuf, 0, 2)
		enc.SetBufferWithOffsetAtIndex(xBuf, 0, 3)
		enc.SetBufferWithOffsetAtIndex(outBuf, 0, 4)
		setEncInt32(enc, int32(inDim), 5)  // K = in_vector_len
		setEncInt32(enc, int32(outDim), 6) // N = out_vector_len

		const bn, bk = 8, 32
		nTgp := (outDim + bn - 1) / bn
		enc.DispatchThreadgroupsThreadsPerThreadgroup(
			metal.MTLSize{Width: 1, Height: uint(nTgp), Depth: 1}, // grid (M, ceil(N/bn), B)
			metal.MTLSize{Width: bk, Height: 2, Depth: 1},         // group (bk, 2, 1)
		)
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()

		copy(out, unsafe.Slice((*float32)(outBuf.Contents()), outDim))
	})
	return out, nil
}

// QMVBF16 is the bfloat16-activation sibling of QMV: out = x @ Wᵀ for a 4-bit
// (affine) quantised weight matrix, with bf16 activations, scales, biases and
// output — the quantised decode projection. x is inDim bf16 bytes; wq/scales/
// biases are the packed bytes MLX's quantiser produces for a bf16 (outDim x inDim)
// weight (scales and biases bf16, one per group per row); the result is outDim
// bf16 bytes. It drives affine_qmv[_fast]_bfloat16_t_gs_G_b_B_batch_0 — the same
// kernel template and host ABI as QMV (w0 s1 b2 x3 out4 K5 N6; single B<=1 matvec,
// so MLX's add_strides_and_shapes early-returns and there are no batch params),
// only the activation dtype differs. Because the decode path is already bf16, this
// needs NO precision conversion around the projections (unlike float QMV). The bf16
// type token is bfloat16_t. Byte-for-byte parity with pkg/metal.QuantizedMatmul
// (transpose=true) on bf16 inputs + the same packed bytes is gated in parity_test.go.
func QMVBF16(x, wq, scales, biases []byte, outDim, inDim, groupSize, bits int) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(x) != inDim*bf16Size {
		return nil, core.NewError("native.QMVBF16: len(x) must equal inDim bf16 bytes")
	}
	if outDim == 0 || inDim == 0 {
		return make([]byte, outDim*bf16Size), nil
	}

	out := make([]byte, outDim*bf16Size)
	var encErr error
	withAutoreleasePool(func() {
		wBuf, sBuf, bBuf := sharedBytes(wq), sharedBytes(scales), sharedBytes(biases)
		xBuf := sharedBytes(x)
		outBuf := device.NewBufferWithLengthOptions(uint(outDim*bf16Size), metal.MTLResourceStorageModeShared)

		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		if encErr = encQMVBF16(enc, wBuf, sBuf, bBuf, xBuf, outBuf, 0, 0, 0, 0, outDim, inDim, groupSize, bits); encErr != nil {
			enc.EndEncoding()
			return
		}
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()

		copy(out, unsafe.Slice((*byte)(outBuf.Contents()), outDim*bf16Size))
	})
	if encErr != nil {
		return nil, encErr
	}
	return out, nil
}

func qmvBF16Resident(x []byte, w QuantWeight, outDim, inDim, groupSize, bits int) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(x) != inDim*bf16Size {
		return nil, core.NewError("native.qmvBF16Resident: len(x) must equal inDim bf16 bytes")
	}
	if outDim == 0 || inDim == 0 {
		return make([]byte, outDim*bf16Size), nil
	}
	groupSize, bits = quantWeightGeometryForShape(w, outDim, inDim, groupSize, bits)
	if groupSize <= 0 || bits <= 0 || inDim%groupSize != 0 {
		return nil, core.NewError("native.qmvBF16Resident: invalid quant geometry")
	}
	wantPacked := outDim * inDim * bits / 8
	wantSB := outDim * (inDim / groupSize) * bf16Size
	if len(w.Packed) != wantPacked || len(w.Scales) != wantSB || len(w.Biases) != wantSB {
		return nil, core.NewError("native.qmvBF16Resident: quant weight size mismatch")
	}

	out := make([]byte, outDim*bf16Size)
	var encErr error
	withAutoreleasePool(func() {
		wBuf, sBuf, bBuf := quantWeightViews(w)
		xBuf := sharedBytes(x)
		outBuf := device.NewBufferWithLengthOptions(uint(outDim*bf16Size), metal.MTLResourceStorageModeShared)

		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		if encErr = encQMVBF16(enc, wBuf.buf, sBuf.buf, bBuf.buf, xBuf, outBuf, wBuf.off, sBuf.off, bBuf.off, 0, outDim, inDim, groupSize, bits); encErr != nil {
			enc.EndEncoding()
			return
		}
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()

		copy(out, unsafe.Slice((*byte)(outBuf.Contents()), outDim*bf16Size))
	})
	if encErr != nil {
		return nil, encErr
	}
	return out, nil
}

func quantWeightViews(w QuantWeight) (bufView, bufView, bufView) {
	if w.packedView.buf != nil && w.scalesView.buf != nil && w.biasesView.buf != nil {
		return w.packedView, w.scalesView, w.biasesView
	}
	return bufView{buf: residentBytes(w.Packed)}, bufView{buf: residentBytes(w.Scales)}, bufView{buf: residentBytes(w.Biases)}
}

func bf16WeightView(weight []byte, view bufView) bufView {
	if view.buf != nil {
		return view
	}
	return bufView{buf: residentBytes(weight)}
}

func rmsNormBF16View(x, weight []byte, weightView bufView, rows, axisSize int, eps float32) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if len(x) != rows*axisSize*bf16Size {
		return nil, core.NewError("native.rmsNormBF16View: len(x) must equal rows*axisSize*2 bytes")
	}
	if len(weight) != axisSize*bf16Size {
		return nil, core.NewError("native.rmsNormBF16View: len(weight) must equal axisSize*2 bytes")
	}
	if rows == 0 || axisSize == 0 {
		return make([]byte, len(x)), nil
	}

	out := make([]byte, len(x))
	var encErr error
	withAutoreleasePool(func() {
		xBuf := sharedBytes(x)
		w := bf16WeightView(weight, weightView)
		outBuf := device.NewBufferWithLengthOptions(uint(len(out)), metal.MTLResourceStorageModeShared)

		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		if encErr = encRMSNormRowsBF16(enc, xBuf, w.buf, outBuf, 0, w.off, 0, rows, axisSize, eps); encErr != nil {
			enc.EndEncoding()
			return
		}
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()
		copy(out, unsafe.Slice((*byte)(outBuf.Contents()), len(out)))
	})
	if encErr != nil {
		return nil, encErr
	}
	return out, nil
}
