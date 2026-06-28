// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"sync"
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

type plHostScratchKey struct {
	plDim, dModel int
	projScale     [2]byte
}

var plHostScratchPools sync.Map

func plHostScratchPoolForKey(key plHostScratchKey) *sync.Pool {
	if v, ok := plHostScratchPools.Load(key); ok {
		return v.(*sync.Pool)
	}
	pool := &sync.Pool{}
	if v, loaded := plHostScratchPools.LoadOrStore(key, pool); loaded {
		return v.(*sync.Pool)
	}
	return pool
}

type plHostScratch struct {
	hidden, perLayer                        *pinnedNoCopyBytes
	projected, scaled, projNormed, combined metal.MTLBuffer
	out                                     metal.MTLBuffer
	projScaleBuf, combineScaleBuf           metal.MTLBuffer
	projScaleBytes, combineScaleBytes       [2]byte
	outHost                                 []byte
	outHostPinned                           *pinnedNoCopyBytes
	plDim, dModel                           int
}

func newPLHostScratch(plDim, dModel int, projScale float32) (*plHostScratch, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if plDim <= 0 || dModel <= 0 {
		return nil, core.NewError("native.newPLHostScratch: invalid dimensions")
	}
	hidden, err := newPinnedNoCopyBytes(dModel * bf16Size)
	if err != nil {
		return nil, err
	}
	perLayer, err := newPinnedNoCopyBytes(plDim * bf16Size)
	if err != nil {
		hidden.Close()
		return nil, err
	}
	nb := func() metal.MTLBuffer {
		return device.NewBufferWithLengthOptions(uint(plDim*bf16Size), metal.MTLResourceStorageModeShared)
	}
	s := &plHostScratch{
		hidden: hidden, perLayer: perLayer,
		projected: nb(), scaled: nb(), projNormed: nb(), combined: nb(), out: nb(),
		plDim: plDim, dModel: dModel,
	}
	s.projScaleBytes = bf16ScalarBytes(projScale)
	s.combineScaleBytes = bf16ScalarBytes(gemma4PerLayerCombineScale)
	s.projScaleBuf = bf16ConstBuffer(1, projScale)
	s.combineScaleBuf = bf16ConstBuffer(1, gemma4PerLayerCombineScale)
	return s, nil
}

func plHostScratchPoolFor(plDim, dModel int, projScale float32) *sync.Pool {
	return plHostScratchPoolForKey(plHostScratchKey{plDim: plDim, dModel: dModel, projScale: bf16ScalarBytes(projScale)})
}

func getPLHostScratch(plDim, dModel int, projScale float32) (*plHostScratch, error) {
	pool := plHostScratchPoolFor(plDim, dModel, projScale)
	if v := pool.Get(); v != nil {
		s := v.(*plHostScratch)
		if s != nil &&
			s.plDim == plDim &&
			s.dModel == dModel &&
			s.hidden != nil &&
			s.perLayer != nil &&
			s.projected != nil &&
			s.scaled != nil &&
			s.projNormed != nil &&
			s.combined != nil &&
			s.out != nil &&
			s.projScaleBytes == bf16ScalarBytes(projScale) &&
			s.combineScaleBytes == bf16ScalarBytes(gemma4PerLayerCombineScale) {
			return s, nil
		}
		s.Close()
	}
	return newPLHostScratch(plDim, dModel, projScale)
}

func putPLHostScratch(s *plHostScratch) {
	if s != nil && s.plDim > 0 && s.dModel > 0 && s.hidden != nil && s.perLayer != nil && s.out != nil {
		plHostScratchPoolForKey(plHostScratchKey{plDim: s.plDim, dModel: s.dModel, projScale: s.projScaleBytes}).Put(s)
	}
}

func (s *plHostScratch) Close() {
	if s == nil {
		return
	}
	if s.hidden != nil {
		s.hidden.Close()
		s.hidden = nil
	}
	if s.perLayer != nil {
		s.perLayer.Close()
		s.perLayer = nil
	}
	s.projected, s.scaled, s.projNormed, s.combined, s.out = nil, nil, nil, nil, nil
	s.projScaleBuf, s.combineScaleBuf = nil, nil
	s.closeHostReadback()
}

func (s *plHostScratch) closeHostReadback() {
	if s == nil {
		return
	}
	if s.outHostPinned != nil {
		s.outHostPinned.Close()
		s.outHostPinned = nil
	}
	s.outHost = nil
}

func (s *plHostScratch) hostReadbackBuffer(n int) ([]byte, metal.MTLBuffer, error) {
	if s == nil {
		return nil, nil, core.NewError("native.plHostScratch.hostReadbackBuffer: scratch is nil")
	}
	if n <= 0 {
		return nil, nil, core.NewError("native.plHostScratch.hostReadbackBuffer: size must be > 0")
	}
	if s.outHostPinned == nil || len(s.outHostPinned.bytes) != n {
		s.closeHostReadback()
		pinned, err := newPinnedNoCopyBytes(n)
		if err != nil {
			return nil, nil, err
		}
		s.outHostPinned = pinned
	}
	s.outHost = s.outHostPinned.bytes[:n]
	return s.outHost, s.outHostPinned.buf, nil
}

// perLayerProjBatched runs the gemma4 PLE projection chain — steps 2-6 of PerLayerInputs: resident-weight
// matvec → ×projScale → RMSNorm(rows) → +perLayer → ×combineScale — as ONE command buffer: a single
// Commit()+WaitUntilCompleted() instead of five. That collapses five per-token GPU round-trips (~5×199µs ≈
// 1ms/token of host stall, GPU idle between) to one. The ops chain via device buffers (no per-op host
// download), driving the SAME kernels as the host path, so the result is byte-identical to the unbatched
// steps 2-6. Intermediate buffers are autoreleased (pool-freed); the projection weight is the resident
// no-copy shard view (projView). scratch, when supplied by a session, keeps the dynamic hidden/per-layer inputs
// in reusable pinned no-copy staging buffers and reuses intermediates plus the host readback across tokens.
func perLayerProjBatched(projView bufView, hidden, perLayer []byte, projScale float32, projNormW []byte, plDim, numLayers, pliDim, dModel int, eps float32, scratchArg ...*plHostScratch) ([]byte, error) {
	out, _, err := perLayerProjBatchedCore(projView, hidden, nil, perLayer, projScale, projNormW, plDim, numLayers, pliDim, dModel, eps, true, scratchArg...)
	return out, err
}

func perLayerProjBatchedResident(projView bufView, hidden, perLayer []byte, projScale float32, projNormW []byte, plDim, numLayers, pliDim, dModel int, eps float32, scratch *plHostScratch) (metal.MTLBuffer, error) {
	if scratch == nil {
		return nil, core.NewError("native.perLayerProjBatchedResident: scratch is required")
	}
	_, buf, err := perLayerProjBatchedCore(projView, hidden, nil, perLayer, projScale, projNormW, plDim, numLayers, pliDim, dModel, eps, false, scratch)
	return buf, err
}

func perLayerProjBatchedResidentBuffer(projView bufView, hiddenBuf metal.MTLBuffer, perLayer []byte, projScale float32, projNormW []byte, plDim, numLayers, pliDim, dModel int, eps float32, scratch *plHostScratch) (metal.MTLBuffer, error) {
	if scratch == nil {
		return nil, core.NewError("native.perLayerProjBatchedResidentBuffer: scratch is required")
	}
	if hiddenBuf == nil {
		return nil, core.NewError("native.perLayerProjBatchedResidentBuffer: hidden buffer is nil")
	}
	_, buf, err := perLayerProjBatchedCore(projView, nil, hiddenBuf, perLayer, projScale, projNormW, plDim, numLayers, pliDim, dModel, eps, false, scratch)
	return buf, err
}

func perLayerProjBatchedCore(projView bufView, hidden []byte, hiddenBufArg metal.MTLBuffer, perLayer []byte, projScale float32, projNormW []byte, plDim, numLayers, pliDim, dModel int, eps float32, readback bool, scratchArg ...*plHostScratch) ([]byte, metal.MTLBuffer, error) {
	if numLayers <= 0 || pliDim <= 0 || dModel <= 0 || plDim != numLayers*pliDim {
		return nil, nil, core.NewError("native.perLayerProjBatched: invalid dimensions")
	}
	if hiddenBufArg == nil && len(hidden) != dModel*bf16Size {
		return nil, nil, core.NewError("native.perLayerProjBatched: hidden must be dModel bf16 bytes")
	}
	if len(perLayer) != plDim*bf16Size {
		return nil, nil, core.NewError("native.perLayerProjBatched: perLayer must be numLayers*pliDim bf16 bytes")
	}
	if len(projNormW) != pliDim*bf16Size {
		return nil, nil, core.NewError("native.perLayerProjBatched: projNormW must be pliDim bf16 bytes")
	}
	if projView.buf == nil {
		return nil, nil, core.NewError("native.perLayerProjBatched: resident projection buffer is nil")
	}
	var scratch *plHostScratch
	if len(scratchArg) > 0 {
		scratch = scratchArg[0]
	}
	outLen := plDim * bf16Size
	var out []byte
	var residentOut metal.MTLBuffer
	directReadback := false
	var ferr error
	withAutoreleasePool(func() {
		projScaleBytes := bf16ScalarBytes(projScale)
		combineScaleBytes := bf16ScalarBytes(gemma4PerLayerCombineScale)
		var hiddenBuf, perLayerBuf, projNormWBuf, projScaleBuf, combineScaleBuf metal.MTLBuffer
		var projectedBuf, scaledBuf, projNormedBuf, combinedBuf, outBuf metal.MTLBuffer
		if scratch != nil {
			if scratch.plDim != plDim || scratch.dModel != dModel {
				ferr = core.NewError("native.perLayerProjBatched: scratch dimension mismatch")
				return
			}
			if scratch.projScaleBytes != projScaleBytes || scratch.combineScaleBytes != combineScaleBytes {
				ferr = core.NewError("native.perLayerProjBatched: scratch scale mismatch")
				return
			}
			if hiddenBufArg != nil {
				hiddenBuf = hiddenBufArg
			} else {
				if hiddenBuf, ferr = scratch.hidden.copyBuffer(hidden); ferr != nil {
					return
				}
			}
			if len(perLayer) == len(scratch.perLayer.bytes) && len(perLayer) > 0 && unsafe.Pointer(&perLayer[0]) == unsafe.Pointer(&scratch.perLayer.bytes[0]) {
				perLayerBuf = scratch.perLayer.buf
			} else {
				if perLayerBuf, ferr = scratch.perLayer.copyBuffer(perLayer); ferr != nil {
					return
				}
			}
			projNormWBuf = residentBytes(projNormW)
			projScaleBuf, combineScaleBuf = scratch.projScaleBuf, scratch.combineScaleBuf
			projectedBuf, scaledBuf, projNormedBuf, combinedBuf, outBuf = scratch.projected, scratch.scaled, scratch.projNormed, scratch.combined, scratch.out
			if readback {
				if out, outBuf, ferr = scratch.hostReadbackBuffer(outLen); ferr != nil {
					return
				}
				directReadback = true
			} else {
				scratch.closeHostReadback()
			}
		} else {
			mk := func(b []byte) metal.MTLBuffer {
				return device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&b[0]), uint(len(b)), metal.MTLResourceStorageModeShared)
			}
			nb := func() metal.MTLBuffer {
				return device.NewBufferWithLengthOptions(uint(plDim*bf16Size), metal.MTLResourceStorageModeShared)
			}
			if hiddenBufArg != nil {
				hiddenBuf = hiddenBufArg
			} else {
				hiddenBuf = mk(hidden)
			}
			perLayerBuf = mk(perLayer)
			projNormWBuf = residentBytes(projNormW)
			projScaleBuf = mk(projScaleBytes[:])
			combineScaleBuf = mk(combineScaleBytes[:])
			projectedBuf, scaledBuf, projNormedBuf, combinedBuf, outBuf = nb(), nb(), nb(), nb(), nb()
			if readback {
				out = make([]byte, outLen)
			}
		}
		residentOut = outBuf

		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		encode := func() error {
			if err := encGemvBF16To(enc, projView.buf, hiddenBuf, projectedBuf, projView.off, 0, plDim, dModel); err != nil {
				return err
			}
			if err := encScaleBF16(enc, projectedBuf, projScaleBuf, scaledBuf, 0, projScaleBytes[:], plDim); err != nil {
				return err
			}
			if err := encRMSNormRowsBF16(enc, scaledBuf, projNormWBuf, projNormedBuf, 0, 0, 0, numLayers, pliDim, eps); err != nil {
				return err
			}
			if err := encAddBF16(enc, projNormedBuf, perLayerBuf, combinedBuf, plDim); err != nil {
				return err
			}
			return encScaleBF16(enc, combinedBuf, combineScaleBuf, outBuf, 0, combineScaleBytes[:], plDim)
		}
		ferr = encode()
		endEncodingFast(enc)
		if ferr != nil {
			return
		}
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		if readback && !directReadback {
			copy(out, unsafe.Slice((*byte)(outBuf.Contents()), outLen))
		}
	})
	return out, residentOut, ferr
}

func perLayerProjQuantBatched(q QuantWeight, hidden, perLayer []byte, projScale float32, projNormW []byte, plDim, numLayers, pliDim, dModel, groupSize, bits int, eps float32, scratchArg ...*plHostScratch) ([]byte, error) {
	out, _, err := perLayerProjQuantBatchedCore(q, hidden, nil, perLayer, projScale, projNormW, plDim, numLayers, pliDim, dModel, groupSize, bits, eps, true, scratchArg...)
	return out, err
}

func perLayerProjQuantBatchedResident(q QuantWeight, hidden, perLayer []byte, projScale float32, projNormW []byte, plDim, numLayers, pliDim, dModel, groupSize, bits int, eps float32, scratch *plHostScratch) (metal.MTLBuffer, error) {
	if scratch == nil {
		return nil, core.NewError("native.perLayerProjQuantBatchedResident: scratch is required")
	}
	_, buf, err := perLayerProjQuantBatchedCore(q, hidden, nil, perLayer, projScale, projNormW, plDim, numLayers, pliDim, dModel, groupSize, bits, eps, false, scratch)
	return buf, err
}

func perLayerProjQuantBatchedResidentBuffer(q QuantWeight, hiddenBuf metal.MTLBuffer, perLayer []byte, projScale float32, projNormW []byte, plDim, numLayers, pliDim, dModel, groupSize, bits int, eps float32, scratch *plHostScratch) (metal.MTLBuffer, error) {
	if scratch == nil {
		return nil, core.NewError("native.perLayerProjQuantBatchedResidentBuffer: scratch is required")
	}
	if hiddenBuf == nil {
		return nil, core.NewError("native.perLayerProjQuantBatchedResidentBuffer: hidden buffer is nil")
	}
	_, buf, err := perLayerProjQuantBatchedCore(q, nil, hiddenBuf, perLayer, projScale, projNormW, plDim, numLayers, pliDim, dModel, groupSize, bits, eps, false, scratch)
	return buf, err
}

func perLayerProjQuantBatchedCore(q QuantWeight, hidden []byte, hiddenBufArg metal.MTLBuffer, perLayer []byte, projScale float32, projNormW []byte, plDim, numLayers, pliDim, dModel, groupSize, bits int, eps float32, readback bool, scratchArg ...*plHostScratch) ([]byte, metal.MTLBuffer, error) {
	if numLayers <= 0 || pliDim <= 0 || dModel <= 0 || plDim != numLayers*pliDim {
		return nil, nil, core.NewError("native.perLayerProjQuantBatched: invalid dimensions")
	}
	if hiddenBufArg == nil && len(hidden) != dModel*bf16Size {
		return nil, nil, core.NewError("native.perLayerProjQuantBatched: hidden must be dModel bf16 bytes")
	}
	if len(perLayer) != plDim*bf16Size {
		return nil, nil, core.NewError("native.perLayerProjQuantBatched: perLayer must be numLayers*pliDim bf16 bytes")
	}
	if len(projNormW) != pliDim*bf16Size {
		return nil, nil, core.NewError("native.perLayerProjQuantBatched: projNormW must be pliDim bf16 bytes")
	}
	groupSize, bits = quantWeightGeometryForShape(q, plDim, dModel, groupSize, bits)
	if groupSize <= 0 || bits <= 0 || dModel%groupSize != 0 {
		return nil, nil, core.NewError("native.perLayerProjQuantBatched: invalid quant geometry")
	}
	wantPacked := plDim * dModel * bits / 8
	wantSB := plDim * (dModel / groupSize) * bf16Size
	if len(q.Packed) != wantPacked || len(q.Scales) != wantSB || len(q.Biases) != wantSB {
		return nil, nil, core.NewError("native.perLayerProjQuantBatched: quant projection size mismatch")
	}
	var scratch *plHostScratch
	if len(scratchArg) > 0 {
		scratch = scratchArg[0]
	}
	outLen := plDim * bf16Size
	var out []byte
	var residentOut metal.MTLBuffer
	directReadback := false
	var ferr error
	withAutoreleasePool(func() {
		projScaleBytes := bf16ScalarBytes(projScale)
		combineScaleBytes := bf16ScalarBytes(gemma4PerLayerCombineScale)
		var hiddenBuf, perLayerBuf, projNormWBuf, projScaleBuf, combineScaleBuf metal.MTLBuffer
		var projectedBuf, scaledBuf, projNormedBuf, combinedBuf, outBuf metal.MTLBuffer
		if scratch != nil {
			if scratch.plDim != plDim || scratch.dModel != dModel {
				ferr = core.NewError("native.perLayerProjQuantBatched: scratch dimension mismatch")
				return
			}
			if scratch.projScaleBytes != projScaleBytes || scratch.combineScaleBytes != combineScaleBytes {
				ferr = core.NewError("native.perLayerProjQuantBatched: scratch scale mismatch")
				return
			}
			if hiddenBufArg != nil {
				hiddenBuf = hiddenBufArg
			} else {
				if hiddenBuf, ferr = scratch.hidden.copyBuffer(hidden); ferr != nil {
					return
				}
			}
			if len(perLayer) == len(scratch.perLayer.bytes) && len(perLayer) > 0 && unsafe.Pointer(&perLayer[0]) == unsafe.Pointer(&scratch.perLayer.bytes[0]) {
				perLayerBuf = scratch.perLayer.buf
			} else {
				if perLayerBuf, ferr = scratch.perLayer.copyBuffer(perLayer); ferr != nil {
					return
				}
			}
			projNormWBuf = residentBytes(projNormW)
			projScaleBuf, combineScaleBuf = scratch.projScaleBuf, scratch.combineScaleBuf
			projectedBuf, scaledBuf, projNormedBuf, combinedBuf, outBuf = scratch.projected, scratch.scaled, scratch.projNormed, scratch.combined, scratch.out
			if readback {
				if out, outBuf, ferr = scratch.hostReadbackBuffer(outLen); ferr != nil {
					return
				}
				directReadback = true
			} else {
				scratch.closeHostReadback()
			}
		} else {
			mk := func(b []byte) metal.MTLBuffer {
				return device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&b[0]), uint(len(b)), metal.MTLResourceStorageModeShared)
			}
			nb := func() metal.MTLBuffer {
				return device.NewBufferWithLengthOptions(uint(plDim*bf16Size), metal.MTLResourceStorageModeShared)
			}
			if hiddenBufArg != nil {
				hiddenBuf = hiddenBufArg
			} else {
				hiddenBuf = mk(hidden)
			}
			perLayerBuf = mk(perLayer)
			projNormWBuf = residentBytes(projNormW)
			projScaleBuf = mk(projScaleBytes[:])
			combineScaleBuf = mk(combineScaleBytes[:])
			projectedBuf, scaledBuf, projNormedBuf, combinedBuf, outBuf = nb(), nb(), nb(), nb(), nb()
			if readback {
				out = make([]byte, outLen)
			}
		}
		residentOut = outBuf

		wBuf, scalesBuf, biasesBuf := quantWeightViews(q)
		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		encode := func() error {
			if err := encQMVBF16(enc, wBuf.buf, scalesBuf.buf, biasesBuf.buf, hiddenBuf, projectedBuf, wBuf.off, scalesBuf.off, biasesBuf.off, 0, plDim, dModel, groupSize, bits); err != nil {
				return err
			}
			if err := encScaleBF16(enc, projectedBuf, projScaleBuf, scaledBuf, 0, projScaleBytes[:], plDim); err != nil {
				return err
			}
			if err := encRMSNormRowsBF16(enc, scaledBuf, projNormWBuf, projNormedBuf, 0, 0, 0, numLayers, pliDim, eps); err != nil {
				return err
			}
			if err := encAddBF16(enc, projNormedBuf, perLayerBuf, combinedBuf, plDim); err != nil {
				return err
			}
			return encScaleBF16(enc, combinedBuf, combineScaleBuf, outBuf, 0, combineScaleBytes[:], plDim)
		}
		ferr = encode()
		endEncodingFast(enc)
		if ferr != nil {
			return
		}
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		if readback && !directReadback {
			copy(out, unsafe.Slice((*byte)(outBuf.Contents()), outLen))
		}
	})
	return out, residentOut, ferr
}
