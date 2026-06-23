// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

// plGPUScratch is the device-buffer scratch for the on-GPU PLE (one set per in-flight pipeline slot).
type plGPUScratch struct {
	perLayer, projected, scaled, projNormed, combined, out metal.MTLBuffer
	projScaleBuf, combineScaleBuf                          metal.MTLBuffer
	projScaleBytes, combineScaleBytes                      [2]byte
}

func newPLGPUScratch(plDim int, projScale float32) *plGPUScratch {
	nb := func() metal.MTLBuffer {
		return device.NewBufferWithLengthOptions(uint(plDim*bf16Size), metal.MTLResourceStorageModeShared)
	}
	s := &plGPUScratch{
		perLayer: nb(), projected: nb(), scaled: nb(), projNormed: nb(), combined: nb(), out: nb(),
	}
	s.projScaleBytes = bf16ScalarBytes(projScale)
	s.combineScaleBytes = bf16ScalarBytes(gemma4PerLayerCombineScale)
	s.projScaleBuf = device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&s.projScaleBytes[0]), 2, metal.MTLResourceStorageModeShared)
	s.combineScaleBuf = device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&s.combineScaleBytes[0]), 2, metal.MTLResourceStorageModeShared)
	return s
}

// encPerLayerInputsGPU encodes the WHOLE gemma4 PLE for one token into `enc` (no commit): the per-layer
// embedding is gathered+dequantised on the GPU from `tokenBuf` (the LM-head argmax output), the main
// embedding `embBuf` is projected → ×projScale → RMSNorm(rows) → +perLayer → ×combineScale. Output is
// scratch.out ([numLayers·pliDim] bf16). The token never round-trips to host — the seam the submit-ahead
// decode pipeline needs for PLE models (e2b/e4b). bf16 projection (e2b); 4-bit per-layer embedding.
func encPerLayerInputsGPU(enc metal.MTLComputeCommandEncoder, embedGatherPSO metal.MTLComputePipelineState,
	tokenBuf, embBuf metal.MTLBuffer,
	embedPacked, embedScales, embedBiases metal.MTLBuffer, embedPackedOff, embedScalesOff, embedBiasesOff uint,
	projW metal.MTLBuffer, projWOff uint, projNormW metal.MTLBuffer,
	sc *plGPUScratch, numLayers, pliDim, dModel, embGS, embBits int, embScale float32, eps float32) error {
	plDim := numLayers * pliDim
	// (1) per-layer embedding: gather token's [plDim] row × √pliDim on the GPU.
	encEmbedGatherQuant(enc, embedGatherPSO, tokenBuf, embedPacked, embedScales, embedBiases, sc.perLayer, embedPackedOff, embedScalesOff, embedBiasesOff, plDim, embGS, embBits, embScale)
	// (2-6) project the main embedding → ×projScale → RMSNorm(rows) → +perLayer → ×combineScale.
	// (projScale is baked into sc.projScaleBuf by newPLGPUScratch.)
	if err := encGemvBF16To(enc, projW, embBuf, sc.projected, projWOff, 0, plDim, dModel); err != nil {
		return err
	}
	if err := encScaleBF16(enc, sc.projected, sc.projScaleBuf, sc.scaled, 0, sc.projScaleBytes[:], plDim); err != nil {
		return err
	}
	if err := encRMSNormRowsBF16(enc, sc.scaled, projNormW, sc.projNormed, 0, 0, 0, numLayers, pliDim, eps); err != nil {
		return err
	}
	if err := encAddBF16(enc, sc.projNormed, sc.perLayer, sc.combined, plDim); err != nil {
		return err
	}
	return encScaleBF16(enc, sc.combined, sc.combineScaleBuf, sc.out, 0, sc.combineScaleBytes[:], plDim)
}

// nextInputsGPU computes one token's NEXT-step decode inputs — the main embedding (dModel) and the PLE
// tensor (numLayers·pliDim) — fully on the GPU via the session's resident weights, reading both back.
// The host-visible check that encNextInputsGPU matches s.embed + s.perLayerInput. ok=false when the
// session has no GPU PLE seam (non-e2b shape). Single-shot (own command buffer); the pipeline drives
// encNextInputsGPU directly into the ICB input buffers instead.
func (s *ArchSession) nextInputsGPU(tokenID int32) (emb, pli []byte, ok bool, err error) {
	if s.encNextInputsGPU == nil || s.plScratchNew == nil {
		return nil, nil, false, nil
	}
	dModel := s.arch.Hidden
	plDim := len(s.arch.Layer) * s.arch.PerLayerInputHidden
	emb = make([]byte, dModel*bf16Size)
	pli = make([]byte, plDim*bf16Size)
	withAutoreleasePool(func() {
		tokBuf := device.NewBufferWithLengthOptions(4, metal.MTLResourceStorageModeShared)
		*(*int32)(tokBuf.Contents()) = tokenID
		embBuf := device.NewBufferWithLengthOptions(uint(dModel*bf16Size), metal.MTLResourceStorageModeShared)
		sc := s.plScratchNew()
		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		if err = s.encNextInputsGPU(enc, tokBuf, embBuf, sc); err != nil {
			enc.EndEncoding()
			return
		}
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()
		copy(emb, unsafe.Slice((*byte)(embBuf.Contents()), dModel*bf16Size))
		copy(pli, unsafe.Slice((*byte)(sc.out.Contents()), plDim*bf16Size))
	})
	if err != nil {
		return nil, nil, false, err
	}
	return emb, pli, true, nil
}

// PerLayerInputsGPU is the standalone host entry over encPerLayerInputsGPU: computes one token's PLE
// tensor fully on the GPU (token id + main embedding in, [numLayers·pliDim] bf16 out). bf16 projection
// (e2b). Byte/cosine-tracks the host PerLayerInputs.
func PerLayerInputsGPU(tokenID int32, emb []byte, embedPacked, embedScales, embedBiases, projW, projNormW []byte, vocabPLI, numLayers, pliDim, dModel, embGS, embBits int, eps float32) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if embBits != 4 {
		return nil, core.NewError("native.PerLayerInputsGPU: per-layer embedding must be 4-bit")
	}
	plDim := numLayers * pliDim
	gpso, err := embedGatherPipeline()
	if err != nil {
		return nil, err
	}
	embScale := float32(math.Sqrt(float64(pliDim)))
	projScale := float32(1.0 / math.Sqrt(float64(dModel)))
	out := make([]byte, plDim*bf16Size)
	var ferr error
	withAutoreleasePool(func() {
		tokBuf := device.NewBufferWithLengthOptions(4, metal.MTLResourceStorageModeShared)
		*(*int32)(tokBuf.Contents()) = tokenID
		embBuf := sharedBytes(emb)
		ePacked, eScales, eBiases := sharedBytes(embedPacked), sharedBytes(embedScales), sharedBytes(embedBiases)
		projWBuf, projNormWBuf := sharedBytes(projW), sharedBytes(projNormW)
		sc := newPLGPUScratch(plDim, projScale)
		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		if ferr = encPerLayerInputsGPU(enc, gpso, tokBuf, embBuf, ePacked, eScales, eBiases, 0, 0, 0, projWBuf, 0, projNormWBuf, sc, numLayers, pliDim, dModel, embGS, embBits, embScale, eps); ferr != nil {
			enc.EndEncoding()
			return
		}
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()
		copy(out, unsafe.Slice((*byte)(sc.out.Contents()), plDim*bf16Size))
	})
	return out, ferr
}
