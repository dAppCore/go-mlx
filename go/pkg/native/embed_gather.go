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

	embedGatherScratchPools sync.Map
)

type embedGatherScratch struct {
	dModel     int
	token, out *pinnedNoCopyBytes
}

func embedGatherScratchPoolFor(dModel int) *sync.Pool {
	if v, ok := embedGatherScratchPools.Load(dModel); ok {
		return v.(*sync.Pool)
	}
	pool := new(sync.Pool)
	if v, loaded := embedGatherScratchPools.LoadOrStore(dModel, pool); loaded {
		return v.(*sync.Pool)
	}
	return pool
}

func embedGatherScratchReady(s *embedGatherScratch, dModel int) bool {
	return s != nil &&
		s.dModel == dModel &&
		s.token != nil &&
		s.token.buf != nil &&
		len(s.token.bytes) == 4 &&
		s.out != nil &&
		s.out.buf != nil &&
		len(s.out.bytes) == dModel*bf16Size
}

func newEmbedGatherScratch(dModel int) (*embedGatherScratch, error) {
	if dModel <= 0 {
		return nil, core.NewError("native.newEmbedGatherScratch: dModel must be > 0")
	}
	token, err := newPinnedNoCopyBytes(4)
	if err != nil {
		return nil, err
	}
	out, err := newPinnedNoCopyBytes(dModel * bf16Size)
	if err != nil {
		token.Close()
		return nil, err
	}
	return &embedGatherScratch{dModel: dModel, token: token, out: out}, nil
}

func getEmbedGatherScratch(dModel int) (*embedGatherScratch, error) {
	pool := embedGatherScratchPoolFor(dModel)
	if v := pool.Get(); v != nil {
		s := v.(*embedGatherScratch)
		if embedGatherScratchReady(s, dModel) {
			return s, nil
		}
		s.Close()
	}
	return newEmbedGatherScratch(dModel)
}

func putEmbedGatherScratch(s *embedGatherScratch) {
	if s == nil {
		return
	}
	if embedGatherScratchReady(s, s.dModel) {
		embedGatherScratchPoolFor(s.dModel).Put(s)
	}
}

func (s *embedGatherScratch) Close() {
	if s == nil {
		return
	}
	if s.token != nil {
		s.token.Close()
		s.token = nil
	}
	if s.out != nil {
		s.out.Close()
		s.out = nil
	}
	s.dModel = 0
}

func (s *embedGatherScratch) buffers(tokenID int32, dModel int) (metal.MTLBuffer, metal.MTLBuffer, error) {
	if s == nil || s.token == nil || s.out == nil {
		return nil, nil, core.NewError("native.embedGatherScratch.buffers: scratch is nil")
	}
	if s.dModel != dModel || len(s.token.bytes) != 4 || len(s.out.bytes) != dModel*bf16Size {
		return nil, nil, core.NewError("native.embedGatherScratch.buffers: dimension mismatch")
	}
	*(*int32)(unsafe.Pointer(&s.token.bytes[0])) = tokenID
	return s.token.buf, s.out.buf, nil
}

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
	emitEmbedGatherQuant(encSink{enc}, pso, tokenBuf, packed, scales, biases, out, packedOff, scalesOff, biasesOff, dModel, groupSize, bits, embedScale)
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
	if dModel == 0 {
		return out, nil
	}
	var encErr error
	withAutoreleasePool(func() {
		scratch, err := getEmbedGatherScratch(dModel)
		if err != nil {
			encErr = err
			return
		}
		defer putEmbedGatherScratch(scratch)
		tokBuf, outBuf, err := scratch.buffers(tokenID, dModel)
		if err != nil {
			encErr = err
			return
		}
		pBuf, sBuf, bBuf := residentBytes(packed), residentBytes(scales), residentBytes(biases)
		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		encEmbedGatherQuant(enc, pso, tokBuf, pBuf, sBuf, bBuf, outBuf, 0, 0, 0, dModel, groupSize, bits, embedScale)
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		copy(out, scratch.out.bytes[:len(out)])
	})
	if encErr != nil {
		return nil, encErr
	}
	return out, nil
}
