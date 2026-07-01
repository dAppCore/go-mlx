// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"runtime"
	"sync"
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

type sdpaPagedKernelParams struct {
	NHeads      uint32
	NKVHeads    uint32
	HeadDim     uint32
	PageLen     uint32
	KHeadStride uint32
	KSeqStride  uint32
	VHeadStride uint32
	VSeqStride  uint32
	Scale       float32
}

type sdpaPagedDecodeScratch struct {
	nHeads, headDim   int
	maxs, denoms, acc metal.MTLBuffer
	maxPtr, denomPtr  []float32
	accPtr            []float32
}

var (
	sdpaPagedUpdatePSOOnce sync.Once
	sdpaPagedUpdatePSO     metal.MTLComputePipelineState
	sdpaPagedUpdatePSOErr  error

	sdpaPagedFinalisePSOOnce sync.Once
	sdpaPagedFinalisePSO     metal.MTLComputePipelineState
	sdpaPagedFinalisePSOErr  error
)

func newSDPAPagedDecodeScratch(nHeads, headDim int) (*sdpaPagedDecodeScratch, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if nHeads <= 0 || headDim <= 0 {
		return nil, core.NewError("native.newSDPAPagedDecodeScratch: dimensions must be > 0")
	}
	maxs := scratchF32(nHeads)
	denoms := scratchF32(nHeads)
	acc := scratchF32(nHeads * headDim)
	if maxs == nil || denoms == nil || acc == nil ||
		maxs.GetID() == 0 || denoms.GetID() == 0 || acc.GetID() == 0 {
		return nil, core.NewError("native.newSDPAPagedDecodeScratch: failed to allocate scratch buffers")
	}
	return &sdpaPagedDecodeScratch{
		nHeads:   nHeads,
		headDim:  headDim,
		maxs:     maxs,
		denoms:   denoms,
		acc:      acc,
		maxPtr:   unsafe.Slice((*float32)(maxs.Contents()), nHeads),
		denomPtr: unsafe.Slice((*float32)(denoms.Contents()), nHeads),
		accPtr:   unsafe.Slice((*float32)(acc.Contents()), nHeads*headDim),
	}, nil
}

func (s *sdpaPagedDecodeScratch) reset(nHeads, headDim int) error {
	if s == nil || s.maxs == nil || s.denoms == nil || s.acc == nil {
		return core.NewError("native.sdpaPagedDecodeScratch.reset: nil scratch")
	}
	if s.nHeads != nHeads || s.headDim != headDim ||
		len(s.maxPtr) != nHeads || len(s.denomPtr) != nHeads || len(s.accPtr) != nHeads*headDim {
		return core.NewError("native.sdpaPagedDecodeScratch.reset: dimension mismatch")
	}
	for i := range s.maxPtr {
		s.maxPtr[i] = -3.0e38
	}
	clear(s.denomPtr)
	clear(s.accPtr)
	return nil
}

func sdpaPagedUpdatePipeline() (metal.MTLComputePipelineState, error) {
	sdpaPagedUpdatePSOOnce.Do(func() {
		if customLibrary == nil || customLibrary.GetID() == 0 {
			sdpaPagedUpdatePSOErr = core.NewError("native.sdpaPagedUpdatePipeline: custom library unavailable")
			return
		}
		fn := customLibrary.NewFunctionWithName("lthn_sdpa_paged_update_bf16")
		if fn == nil || fn.GetID() == 0 {
			sdpaPagedUpdatePSOErr = core.NewError("native.sdpaPagedUpdatePipeline: kernel lthn_sdpa_paged_update_bf16 not found")
			return
		}
		sdpaPagedUpdatePSO, sdpaPagedUpdatePSOErr = device.NewComputePipelineStateWithFunctionError(fn)
	})
	return sdpaPagedUpdatePSO, sdpaPagedUpdatePSOErr
}

func sdpaPagedFinalisePipeline() (metal.MTLComputePipelineState, error) {
	sdpaPagedFinalisePSOOnce.Do(func() {
		if customLibrary == nil || customLibrary.GetID() == 0 {
			sdpaPagedFinalisePSOErr = core.NewError("native.sdpaPagedFinalisePipeline: custom library unavailable")
			return
		}
		fn := customLibrary.NewFunctionWithName("lthn_sdpa_paged_finalise_bf16")
		if fn == nil || fn.GetID() == 0 {
			sdpaPagedFinalisePSOErr = core.NewError("native.sdpaPagedFinalisePipeline: kernel lthn_sdpa_paged_finalise_bf16 not found")
			return
		}
		sdpaPagedFinalisePSO, sdpaPagedFinalisePSOErr = device.NewComputePipelineStateWithFunctionError(fn)
	})
	return sdpaPagedFinalisePSO, sdpaPagedFinalisePSOErr
}

func encSDPAPagedDecode(
	enc metal.MTLComputeCommandEncoder,
	q metal.MTLBuffer,
	keyPages, valuePages []metal.MTLBuffer,
	pageLens, pageSpans []int,
	out metal.MTLBuffer,
	scratch *sdpaPagedDecodeScratch,
	nHeads, nKVHeads, headDim int,
	scale float32,
) error {
	if len(pageLens) != len(keyPages) || len(pageSpans) != len(keyPages) {
		return core.NewError("native.encSDPAPagedDecode: page lengths and spans must match page buffers")
	}
	keyHeadStrides := make([]int, len(pageSpans))
	keySeqStrides := make([]int, len(pageSpans))
	valueHeadStrides := make([]int, len(pageSpans))
	valueSeqStrides := make([]int, len(pageSpans))
	for i, span := range pageSpans {
		if span < pageLens[i] {
			return core.NewError("native.encSDPAPagedDecode: visible page length must fit physical span")
		}
		keyHeadStrides[i] = span * headDim
		keySeqStrides[i] = headDim
		valueHeadStrides[i] = span * headDim
		valueSeqStrides[i] = headDim
	}
	return encSDPAPagedDecodeStrided(enc, q, keyPages, valuePages, pageLens, keyHeadStrides, keySeqStrides, valueHeadStrides, valueSeqStrides, out, scratch, nHeads, nKVHeads, headDim, scale)
}

func encSDPAPagedDecodeStrided(
	enc metal.MTLComputeCommandEncoder,
	q metal.MTLBuffer,
	keyPages, valuePages []metal.MTLBuffer,
	pageLens, keyHeadStrides, keySeqStrides, valueHeadStrides, valueSeqStrides []int,
	out metal.MTLBuffer,
	scratch *sdpaPagedDecodeScratch,
	nHeads, nKVHeads, headDim int,
	scale float32,
) error {
	if nHeads <= 0 || nKVHeads <= 0 || headDim <= 0 {
		return core.NewError("native.encSDPAPagedDecodeStrided: dimensions must be > 0")
	}
	if nHeads%nKVHeads != 0 {
		return core.NewError("native.encSDPAPagedDecodeStrided: nHeads must be a multiple of nKVHeads")
	}
	if q == nil || q.GetID() == 0 || out == nil || out.GetID() == 0 {
		return core.NewError("native.encSDPAPagedDecodeStrided: nil input/output buffer")
	}
	if len(keyPages) == 0 || len(keyPages) != len(valuePages) || len(keyPages) != len(pageLens) ||
		len(keyPages) != len(keyHeadStrides) || len(keyPages) != len(keySeqStrides) ||
		len(keyPages) != len(valueHeadStrides) || len(keyPages) != len(valueSeqStrides) {
		return core.NewError("native.encSDPAPagedDecodeStrided: page buffers and strides must be non-empty and matched")
	}
	for i := range keyPages {
		if keyPages[i] == nil || keyPages[i].GetID() == 0 || valuePages[i] == nil || valuePages[i].GetID() == 0 {
			return core.NewError("native.encSDPAPagedDecodeStrided: nil page buffer")
		}
		if pageLens[i] <= 0 || keyHeadStrides[i] <= 0 || keySeqStrides[i] <= 0 || valueHeadStrides[i] <= 0 || valueSeqStrides[i] <= 0 {
			return core.NewError("native.encSDPAPagedDecodeStrided: page lengths and strides must be > 0")
		}
	}
	if err := scratch.reset(nHeads, headDim); err != nil {
		return err
	}
	updatePSO, err := sdpaPagedUpdatePipeline()
	if err != nil {
		return err
	}
	finalisePSO, err := sdpaPagedFinalisePipeline()
	if err != nil {
		return err
	}

	for i := range keyPages {
		params := sdpaPagedKernelParams{
			NHeads:      uint32(nHeads),
			NKVHeads:    uint32(nKVHeads),
			HeadDim:     uint32(headDim),
			PageLen:     uint32(pageLens[i]),
			KHeadStride: uint32(keyHeadStrides[i]),
			KSeqStride:  uint32(keySeqStrides[i]),
			VHeadStride: uint32(valueHeadStrides[i]),
			VSeqStride:  uint32(valueSeqStrides[i]),
			Scale:       scale,
		}
		setPSO(enc, updatePSO)
		setBuf(enc, q, 0, 0)
		setBuf(enc, keyPages[i], 0, 1)
		setBuf(enc, valuePages[i], 0, 2)
		setBuf(enc, scratch.maxs, 0, 3)
		setBuf(enc, scratch.denoms, 0, 4)
		setBuf(enc, scratch.acc, 0, 5)
		setBytes(enc, unsafe.Pointer(&params), uint(unsafe.Sizeof(params)), 6)
		dispatchThreadgroups(enc,
			metal.MTLSize{Width: uint(nHeads), Height: 1, Depth: 1},
			metal.MTLSize{Width: 64, Height: 1, Depth: 1},
		)
	}
	total := nHeads * headDim
	setPSO(enc, finalisePSO)
	setBuf(enc, scratch.denoms, 0, 0)
	setBuf(enc, scratch.acc, 0, 1)
	setBuf(enc, out, 0, 2)
	setBytesI32(enc, int32(headDim), 3)
	setBytesI32(enc, int32(total), 4)
	dispatchThreadgroups(enc,
		metal.MTLSize{Width: uint(total), Height: 1, Depth: 1},
		metal.MTLSize{Width: 256, Height: 1, Depth: 1},
	)
	return nil
}

func sdpaPagedTransientBuffer(b []byte, pinners *[]*runtime.Pinner) metal.MTLBuffer {
	if buf, ok := registeredPinnedNoCopyBytes(b); ok {
		return buf
	}
	buf, pinner, noCopy := residentNoCopyBytes(b)
	if noCopy && pinner != nil {
		*pinners = append(*pinners, pinner)
	}
	return buf
}

func sdpaPagedOutputBuffer(out []byte) (metal.MTLBuffer, *runtime.Pinner, bool) {
	if buf, ok := registeredPinnedNoCopyBytes(out); ok {
		return buf, nil, true
	}
	buf, pinner, noCopy := residentNoCopyBytes(out)
	if !noCopy {
		if pinner != nil {
			pinner.Unpin()
		}
		return nil, nil, false
	}
	return buf, pinner, true
}

func sdpaPagedValidate(qb []byte, keyPages, valuePages [][]byte, pageLens []int, nHeads, nKVHeads, headDim int) ([]int, int, error) {
	if nHeads <= 0 || nKVHeads <= 0 || headDim <= 0 {
		return nil, 0, core.NewError("native.SDPAPagedBF16: dimensions must be > 0")
	}
	if nHeads%nKVHeads != 0 {
		return nil, 0, core.NewError("native.SDPAPagedBF16: nHeads must be a multiple of nKVHeads")
	}
	if len(qb) != nHeads*headDim*bf16Size {
		return nil, 0, core.NewError("native.SDPAPagedBF16: query length mismatch")
	}
	if len(keyPages) == 0 || len(keyPages) != len(valuePages) {
		return nil, 0, core.NewError("native.SDPAPagedBF16: key/value pages must be non-empty and matched")
	}
	if pageLens != nil && len(pageLens) != len(keyPages) {
		return nil, 0, core.NewError("native.SDPAPagedBF16: page lens must match key/value pages")
	}
	pageStride := nKVHeads * headDim * bf16Size
	lens := make([]int, len(keyPages))
	total := 0
	for i := range keyPages {
		if len(keyPages[i]) == 0 || len(valuePages[i]) == 0 {
			return nil, 0, core.NewError("native.SDPAPagedBF16: page length must be > 0")
		}
		if len(keyPages[i]) != len(valuePages[i]) {
			return nil, 0, core.NewError("native.SDPAPagedBF16: key/value page byte lengths differ")
		}
		if len(keyPages[i])%pageStride != 0 {
			return nil, 0, core.NewError("native.SDPAPagedBF16: page byte length is not aligned to KV heads and headDim")
		}
		pageLen := len(keyPages[i]) / pageStride
		if pageLens != nil {
			pageLen = pageLens[i]
			physicalLen := len(keyPages[i]) / pageStride
			if pageLen <= 0 || pageLen > physicalLen {
				return nil, 0, core.NewError("native.SDPAPagedBF16: page lens must fit the physical page")
			}
		}
		lens[i] = pageLen
		total += pageLen
	}
	return lens, total, nil
}

// SDPAPagedBF16 computes single-token scaled-dot-product attention over paged BF16
// KV cache rows without concatenating the pages on the host. Page layout is
// head-major [nKVHeads, pageLen, headDim], matching pkg/metal's paged-cache ABI.
func SDPAPagedBF16(qb []byte, keyPages, valuePages [][]byte, nHeads, nKVHeads, headDim int, scale float32) ([]byte, error) {
	return SDPAPagedBF16Into(nil, qb, keyPages, valuePages, nHeads, nKVHeads, headDim, scale)
}

func SDPAPagedBF16Into(out []byte, qb []byte, keyPages, valuePages [][]byte, nHeads, nKVHeads, headDim int, scale float32) ([]byte, error) {
	return sdpaPagedBF16IntoPageLens(out, qb, keyPages, valuePages, nil, nHeads, nKVHeads, headDim, scale)
}

func sdpaPagedBF16IntoPageLens(out []byte, qb []byte, keyPages, valuePages [][]byte, pageLens []int, nHeads, nKVHeads, headDim int, scale float32) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	pageLens, _, err := sdpaPagedValidate(qb, keyPages, valuePages, pageLens, nHeads, nKVHeads, headDim)
	if err != nil {
		return nil, err
	}

	outLen := nHeads * headDim * bf16Size
	callerOut := cap(out) >= outLen
	if callerOut {
		out = out[:outLen]
	} else {
		out = make([]byte, outLen)
	}

	var encErr error
	withAutoreleasePool(func() {
		outBuf := scratchBF16(nHeads * headDim)
		if outBuf == nil || outBuf.GetID() == 0 {
			encErr = core.NewError("native.SDPAPagedBF16: failed to allocate scratch buffers")
			return
		}
		scratch, err := newSDPAPagedDecodeScratch(nHeads, headDim)
		if err != nil {
			encErr = err
			return
		}

		var outPinner *runtime.Pinner
		directOut := false
		if callerOut {
			if tmp, pinner, ok := sdpaPagedOutputBuffer(out); ok {
				outBuf = tmp
				outPinner = pinner
				directOut = true
			}
		}
		defer func() {
			if outPinner != nil {
				outPinner.Unpin()
			}
		}()

		pinners := make([]*runtime.Pinner, 0, 1+len(keyPages)*2)
		defer func() {
			for _, pinner := range pinners {
				if pinner != nil {
					pinner.Unpin()
				}
			}
		}()
		qBuf := sdpaPagedTransientBuffer(qb, &pinners)
		keyBufs := make([]metal.MTLBuffer, len(keyPages))
		valueBufs := make([]metal.MTLBuffer, len(valuePages))
		pageSpans := make([]int, len(keyPages))
		for i := range keyPages {
			keyBufs[i] = sdpaPagedTransientBuffer(keyPages[i], &pinners)
			valueBufs[i] = sdpaPagedTransientBuffer(valuePages[i], &pinners)
			pageSpans[i] = len(keyPages[i]) / (nKVHeads * headDim * bf16Size)
		}

		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		encErr = encSDPAPagedDecode(enc, qBuf, keyBufs, valueBufs, pageLens, pageSpans, outBuf, scratch, nHeads, nKVHeads, headDim, scale)
		endEncodingFast(enc)
		if encErr != nil {
			return
		}
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)

		if !directOut {
			copy(out, unsafe.Slice((*byte)(outBuf.Contents()), outLen))
		}
		runtime.KeepAlive(qb)
		runtime.KeepAlive(keyPages)
		runtime.KeepAlive(valuePages)
		runtime.KeepAlive(out)
	})
	if encErr != nil {
		return nil, encErr
	}
	return out, nil
}
