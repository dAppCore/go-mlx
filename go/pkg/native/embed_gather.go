// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"sync"
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

var (
	embedGatherPSOMu sync.Mutex
	embedGatherPSO   metal.MTLComputePipelineState
	embedGatherErr   error
	embedGatherOnce  sync.Once
)

func embedGatherPipeline() (metal.MTLComputePipelineState, error) {
	embedGatherOnce.Do(func() {
		if customLibrary == nil || customLibrary.GetID() == 0 {
			embedGatherErr = core.NewError("native.embedGatherPipeline: custom library unavailable")
			return
		}
		fn := customLibrary.NewFunctionWithName("lthn_embed_gather_bf16")
		if fn == nil || fn.GetID() == 0 {
			embedGatherErr = core.NewError("native.embedGatherPipeline: kernel lthn_embed_gather_bf16 not found")
			return
		}
		embedGatherPSO, embedGatherErr = device.NewComputePipelineStateWithFunctionError(fn)
	})
	embedGatherPSOMu.Lock()
	defer embedGatherPSOMu.Unlock()
	return embedGatherPSO, embedGatherErr
}

// encEmbedGatherQuant encodes the GPU dequant-gather of the token in `tokenBuf` (a device int buffer — the
// LM-head argmax output) into `out` (dModel bf16): the 4-bit affine embedding row × embedScale. Lets the
// chained decode step compute the NEXT step's input embedding without a host round-trip. 4-bit only.
func encEmbedGatherQuant(enc metal.MTLComputeCommandEncoder, pso metal.MTLComputePipelineState, tokenBuf, packed, scales, biases, out metal.MTLBuffer, packedOff, scalesOff, biasesOff uint, dModel, groupSize, bits int, embedScale float32) {
	rowPacked := dModel * bits / 8
	rowSB := dModel / groupSize
	enc.SetComputePipelineState(pso)
	enc.SetBufferWithOffsetAtIndex(tokenBuf, 0, 0)
	enc.SetBufferWithOffsetAtIndex(packed, packedOff, 1)
	enc.SetBufferWithOffsetAtIndex(scales, scalesOff, 2)
	enc.SetBufferWithOffsetAtIndex(biases, biasesOff, 3)
	enc.SetBufferWithOffsetAtIndex(out, 0, 4)
	setEncInt32(enc, int32(dModel), 5)
	setEncInt32(enc, int32(groupSize), 6)
	setEncFloat32(enc, embedScale, 7)
	setEncInt32(enc, int32(rowPacked), 8)
	setEncInt32(enc, int32(rowSB), 9)
	enc.DispatchThreadsThreadsPerThreadgroup(
		metal.MTLSize{Width: uint(dModel), Height: 1, Depth: 1},
		metal.MTLSize{Width: uint(elemGroupTG(dModel)), Height: 1, Depth: 1},
	)
}

func elemGroupTG(n int) int {
	if n < 256 {
		return n
	}
	return 256
}

// EmbedGatherQuantBF16 gathers + dequantises one token's 4-bit embedding row on the GPU — the standalone
// host entry (creates a token buffer, dispatches, reads out). Byte-tracks embedTokenQuant. dModel bf16.
func EmbedGatherQuantBF16(tokenID int32, packed, scales, biases []byte, dModel, groupSize, bits int, embedScale float32) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if bits != 4 {
		return nil, core.NewError("native.EmbedGatherQuantBF16: only 4-bit supported")
	}
	pso, err := embedGatherPipeline()
	if err != nil {
		return nil, err
	}
	out := make([]byte, dModel*bf16Size)
	withAutoreleasePool(func() {
		tokBuf := device.NewBufferWithLengthOptions(4, metal.MTLResourceStorageModeShared)
		*(*int32)(tokBuf.Contents()) = tokenID
		pBuf, sBuf, bBuf := sharedBytes(packed), sharedBytes(scales), sharedBytes(biases)
		outBuf := device.NewBufferWithLengthOptions(uint(dModel*bf16Size), metal.MTLResourceStorageModeShared)
		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		encEmbedGatherQuant(enc, pso, tokBuf, pBuf, sBuf, bBuf, outBuf, 0, 0, 0, dModel, groupSize, bits, embedScale)
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()
		copy(out, unsafe.Slice((*byte)(outBuf.Contents()), dModel*bf16Size))
	})
	return out, nil
}
