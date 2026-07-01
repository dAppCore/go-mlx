// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"unsafe"

	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/model"
	"github.com/tmc/apple/metal"
)

type denseBatchScratch struct {
	inRowsStack        [16]metal.MTLBuffer
	outRowsStack       [16]metal.MTLBuffer
	readRowsStack      [16]metal.MTLBuffer
	directOutRowsStack [16]metal.MTLBuffer
	offBufStack        [16]metal.MTLBuffer
	offPtrStack        [16]*int32
	offOffStack        [16]uint
	rowOffStack        [16]uint
	readOffStack       [16]uint
	directOutOffStack  [16]uint
	inputViewStack     [16]cachedNoCopyBytesView
	outputViewStack    [16]cachedNoCopyBytesView
	inRows             []metal.MTLBuffer
	outRows            []metal.MTLBuffer
	readRows           []metal.MTLBuffer
	directOutRows      []metal.MTLBuffer
	offBuf             []metal.MTLBuffer
	offPtr             []*int32
	offOff             []uint
	rowOff             []uint
	readOff            []uint
	directOutOff       []uint
	inputViews         []cachedNoCopyBytesView
	outputViews        []cachedNoCopyBytesView
	lastOutView        cachedNoCopyBytesView
	offPacked          metal.MTLBuffer
	offPackedCap       int
	inPacked           metal.MTLBuffer
	outPacked          metal.MTLBuffer
	rowPackedCap       int
	rowBytes           int
	lastRows           metal.MTLBuffer
	lastRowOff         []uint
	lastK              int
	lastResult         [1][]byte
}

func (s *denseBatchScratch) Close() {
	if s == nil {
		return
	}
	for i := range s.inputViewStack {
		s.inputViewStack[i].Close()
	}
	for i := range s.outputViewStack {
		s.outputViewStack[i].Close()
	}
	for i := range s.inputViews {
		s.inputViews[i].Close()
	}
	for i := range s.outputViews {
		s.outputViews[i].Close()
	}
	s.lastOutView.Close()
	*s = denseBatchScratch{}
}

func (s *denseBatchScratch) inputViewsFor(k int) []cachedNoCopyBytesView {
	if k <= len(s.inputViewStack) {
		return s.inputViewStack[:k]
	}
	if cap(s.inputViews) < k {
		for i := range s.inputViews {
			s.inputViews[i].Close()
		}
		s.inputViews = make([]cachedNoCopyBytesView, k)
	} else {
		s.inputViews = s.inputViews[:k]
	}
	return s.inputViews
}

func (s *denseBatchScratch) outputViewsFor(k int) []cachedNoCopyBytesView {
	if k <= len(s.outputViewStack) {
		return s.outputViewStack[:k]
	}
	if cap(s.outputViews) < k {
		for i := range s.outputViews {
			s.outputViews[i].Close()
		}
		s.outputViews = make([]cachedNoCopyBytesView, k)
	} else {
		s.outputViews = s.outputViews[:k]
	}
	return s.outputViews
}

func (s *denseBatchScratch) rows(k, dModel int) (inRows, outRows, offBuf []metal.MTLBuffer, offPtr []*int32, offOff, rowOff []uint) {
	if k <= len(s.inRowsStack) {
		s.inRows = s.inRowsStack[:k]
		s.outRows = s.outRowsStack[:k]
		s.offBuf = s.offBufStack[:k]
		s.offPtr = s.offPtrStack[:k]
		s.offOff = s.offOffStack[:k]
		s.rowOff = s.rowOffStack[:k]
	} else if cap(s.inRows) < k || cap(s.outRows) < k || cap(s.offBuf) < k || cap(s.offPtr) < k || cap(s.offOff) < k || cap(s.rowOff) < k {
		s.inRows = make([]metal.MTLBuffer, k)
		s.outRows = make([]metal.MTLBuffer, k)
		s.offBuf = make([]metal.MTLBuffer, k)
		s.offPtr = make([]*int32, k)
		s.offOff = make([]uint, k)
		s.rowOff = make([]uint, k)
	} else {
		s.inRows = s.inRows[:k]
		s.outRows = s.outRows[:k]
		s.offBuf = s.offBuf[:k]
		s.offPtr = s.offPtr[:k]
		s.offOff = s.offOff[:k]
		s.rowOff = s.rowOff[:k]
	}
	if s.offPacked == nil || s.offPackedCap < k {
		s.offPacked = device.NewBufferWithLengthOptions(uint(k*4), metal.MTLResourceStorageModeShared)
		s.offPackedCap = k
	}
	rowBytes := dModel * bf16Size
	if s.inPacked == nil || s.outPacked == nil || s.rowPackedCap < k || s.rowBytes != rowBytes {
		s.inPacked = scratchBF16(k * dModel)
		s.outPacked = scratchBF16(k * dModel)
		s.rowPackedCap = k
		s.rowBytes = rowBytes
	}
	offsets := unsafe.Slice((*int32)(s.offPacked.Contents()), k)
	for i := 0; i < k; i++ {
		s.inRows[i] = s.inPacked
		s.outRows[i] = s.outPacked
		s.offBuf[i] = s.offPacked
		s.offPtr[i] = &offsets[i]
		s.offOff[i] = uint(i * 4)
		s.rowOff[i] = uint(i * rowBytes)
	}
	return s.inRows, s.outRows, s.offBuf, s.offPtr, s.offOff, s.rowOff
}

func (s *denseBatchScratch) readRowsFor(k int) ([]metal.MTLBuffer, []uint) {
	if k <= len(s.readRowsStack) {
		s.readRows = s.readRowsStack[:k]
		s.readOff = s.readOffStack[:k]
	} else if cap(s.readRows) < k || cap(s.readOff) < k {
		s.readRows = make([]metal.MTLBuffer, k)
		s.readOff = make([]uint, k)
	} else {
		s.readRows = s.readRows[:k]
		s.readOff = s.readOff[:k]
	}
	return s.readRows, s.readOff
}

func (s *denseBatchScratch) directOutputRowsFor(k int) ([]metal.MTLBuffer, []uint) {
	if k <= len(s.directOutRowsStack) {
		s.directOutRows = s.directOutRowsStack[:k]
		s.directOutOff = s.directOutOffStack[:k]
	} else if cap(s.directOutRows) < k || cap(s.directOutOff) < k {
		s.directOutRows = make([]metal.MTLBuffer, k)
		s.directOutOff = make([]uint, k)
	} else {
		s.directOutRows = s.directOutRows[:k]
		s.directOutOff = s.directOutOff[:k]
	}
	return s.directOutRows, s.directOutOff
}

func (s *denseBatchScratch) lastOutputView(out []byte) (metal.MTLBuffer, bool) {
	if s == nil || len(out) == 0 {
		return nil, false
	}
	return s.lastOutView.buffer(out)
}

// decode_batched_session.go — the session-level MTP batched verify: K query tokens through the WHOLE
// resident decode stack in as few command buffers as possible, reusing the resident layer weights and
// caches (no re-upload). Each row i decodes at position basePos+i, writes its K/V into every layer's
// cache at row basePos+i, and attends [0..basePos+i] with the SAME single-query kernels stepToken
// uses — so the K returned hiddens are BYTE-IDENTICAL to calling stepToken K times at basePos..
// basePos+K-1 (proven in decode_batched_session_test.go). This is what lets MTPDecode verify a whole
// K-token draft block against the resident cache in one batched pass instead of K stepGreedy rounds.
//
// v1 covers the dense uniform path (every layer owns its cache; per-layer output scalar handled
// on-device). Layers needing a host flush per row — MoE FFN, the PLE input gate, shared-KV, the trace
// hooks — are out of scope here; stepTokensBatchedDense reports !ok so MTPDecode falls back to the
// byte-identical sequential verify for those models. Folding the K per-row projections into one steel
// GEMM (weight reuse) is the further speedup that trades byte- for token-identity (metal-MTP parity).

// stepTokensBatchedDense runs K tokens at positions [basePos, basePos+K) through the resident layer
// stack and returns their K output hiddens ([]([]byte), each dModel bf16). It writes each token's K/V
// into the per-layer caches at row basePos+i. ok is false (no work done, no cache mutation) when the
// model is outside the dense uniform path — the caller then steps sequentially. Single-goroutine, like
// every ArchSession decode. Must run inside a withAutoreleasePool.
func (s *archDecodeState) stepTokensBatchedDense(embs [][]byte, basePos int) (out [][]byte, ok bool, err error) {
	return s.stepTokensBatchedDenseResult(embs, basePos, true, false, nil, nil)
}

func (s *archDecodeState) stepTokensBatchedDenseNoResult(embs [][]byte, basePos int) (ok bool, err error) {
	_, ok, err = s.stepTokensBatchedDenseResult(embs, basePos, false, false, nil, nil)
	return ok, err
}

func (s *archDecodeState) stepTokensBatchedDenseLastInto(embs [][]byte, basePos int, dst []byte) (last []byte, ok bool, err error) {
	out, ok, err := s.stepTokensBatchedDenseResult(embs, basePos, true, true, dst, nil)
	if err != nil || !ok {
		return nil, ok, err
	}
	if len(out) != 1 {
		return nil, true, core.NewError("native.stepTokensBatchedDenseLast: hidden result count mismatch")
	}
	return out[0], true, nil
}

func (s *archDecodeState) stepTokensBatchedDenseInto(embs [][]byte, basePos int, dstRows [][]byte) (out [][]byte, ok bool, err error) {
	return s.stepTokensBatchedDenseResult(embs, basePos, true, false, nil, dstRows)
}

func (s *archDecodeState) stepTokensBatchedDenseResult(embs [][]byte, basePos int, readResult, readLastOnly bool, lastDst []byte, dstRows [][]byte) (out [][]byte, ok bool, err error) {
	K := len(embs)
	if K == 0 {
		return nil, false, core.NewError("native.stepTokensBatchedDense: empty batch")
	}
	// dense uniform guard: every layer owns its cache + is non-MoE; no PLE gate, no trace, no recorded
	// ICB (whose replay holds its OWN caches, not s.lb). These need a per-row host flush / a different
	// cache — the sequential verify already covers them, byte-identically.
	if s.trace || len(s.ple) > 0 || s.icb != nil {
		return nil, false, nil
	}
	for li := range s.specs {
		if !s.specs[li].OwnsCache() || s.specs[li].MoE {
			return nil, false, nil
		}
		if li < len(s.moeWeights) && s.moeWeights[li] != nil {
			return nil, false, nil
		}
		if li < len(s.moeQuant) && s.moeQuant[li] != nil {
			return nil, false, nil
		}
	}
	for i := range embs {
		if len(embs[i]) != s.dModel*bf16Size {
			return nil, false, core.NewError("native.stepTokensBatchedDense: emb must be dModel bf16 bytes")
		}
	}
	if err := s.syncLinearKVFromDevicePaged(basePos); err != nil {
		return nil, false, err
	}

	rowBytes := s.dModel * bf16Size
	var (
		lastOutBuf    metal.MTLBuffer
		directLastOut bool
	)
	if readResult && readLastOnly {
		if cap(lastDst) < rowBytes {
			lastDst = make([]byte, rowBytes)
		} else {
			lastDst = lastDst[:rowBytes]
		}
		if tmp, ok := s.denseBatch.lastOutputView(lastDst); ok {
			lastOutBuf = tmp
			directLastOut = true
		}
	}
	var (
		directOutputRows      []metal.MTLBuffer
		directOutputOff       []uint
		usingDirectOutputRows bool
	)
	// K-wide working rows (ping-ponged across layers) + per-row position buffers, retained on the state.
	inRows, outRows, offBuf, offPtr, offOff, rowOff := s.denseBatch.rows(K, s.dModel)
	readRows, readOff := inRows, rowOff
	directInputRows, directInputOff := s.denseBatch.readRowsFor(K)
	inputViews := s.denseBatch.inputViewsFor(K)
	usingDirectInputRows := false
	for i := 0; i < K; i++ {
		*offPtr[i] = int32(basePos + i)
		if buf, direct := inputViews[i].buffer(embs[i]); direct {
			directInputRows[i] = buf
			directInputOff[i] = 0
			usingDirectInputRows = true
			continue
		}
		directInputRows[i] = inRows[i]
		directInputOff[i] = rowOff[i]
		off := int(rowOff[i])
		copy(unsafe.Slice((*byte)(inRows[i].Contents()), off+rowBytes)[off:], embs[i])
	}
	if usingDirectInputRows {
		readRows, readOff = directInputRows, directInputOff
	}
	if readResult && !readLastOnly && len(dstRows) >= K {
		directOutputRows, directOutputOff = s.denseBatch.directOutputRowsFor(K)
		outputViews := s.denseBatch.outputViewsFor(K)
		usingDirectOutputRows = true
		for i := 0; i < K; i++ {
			if cap(dstRows[i]) < rowBytes {
				usingDirectOutputRows = false
				break
			}
			dstRows[i] = dstRows[i][:rowBytes]
			buf, direct := outputViews[i].buffer(dstRows[i])
			if !direct {
				usingDirectOutputRows = false
				break
			}
			directOutputRows[i] = buf
			directOutputOff[i] = 0
		}
	}

	cb := commandBufferFast(queue)
	enc := computeCommandEncoderFast(cb)
	for li := 0; li < len(s.specs); li++ {
		lhd, lkv := headDimOf(s.specs[li], s.headDim), kvHeadsOf(s.specs[li], s.nKVHeads)
		slideW, rbase, rotDim := 0, s.base, s.rotaryDim
		layerRopeFreqs := s.ropeFreqs
		if s.specs[li].Attention == model.SlidingAttention {
			slideW, rbase, rotDim = s.slidingWindow, s.localBase, s.rotaryDimLocal
		} else if s.globalRopeFreqs != nil {
			layerRopeFreqs, rotDim = s.globalRopeFreqs, lhd
		}
		lff := s.dFF
		if s.lb[li].dFF > 0 {
			lff = s.lb[li].dFF
		}
		// each row in turn: attention half (writes its K/V row, attends [0..basePos+i]) then MLP half.
		// Metal's buffer hazard tracking orders the cross-row cache write→read, so row i+1 attends row
		// i's freshly written K/V — exactly the sequential per-token causal structure.
		for i := 0; i < K; i++ {
			outBuf, outOff := outRows[i], rowOff[i]
			if directLastOut && li == len(s.specs)-1 && i == K-1 {
				outBuf, outOff = lastOutBuf, 0
			} else if usingDirectOutputRows && li == len(s.specs)-1 {
				outBuf, outOff = directOutputRows[i], directOutputOff[i]
			}
			if err = encAttnHalfKVInputAt(enc, readRows[i], readOff[i], s.lb[li].kCache, s.lb[li].vCache, offBuf[i], s.hBuf, offOff[i],
				s.lb[li].anw, s.lb[li].postAttnNorm, s.lb[li].qNorm, s.lb[li].kNorm, s.valueNormOnes, s.asc, s.lb[li].proj,
				s.dModel, s.nHeads, lkv, lhd, basePos+i, slideW, rotDim, rbase, s.scale, s.eps, layerRopeFreqs); err != nil {
				endEncodingFast(enc)
				return nil, false, err
			}
			if err = encMLPHalfBF16At(enc, s.hBuf, outBuf, outOff, s.lb[li].mnw, s.lb[li].postFFNorm, s.msc, s.lb[li].proj, s.dModel, lff, s.eps); err != nil {
				endEncodingFast(enc)
				return nil, false, err
			}
			if s.lb[li].layerScalar != nil { // gemma4 per-layer output scalar (on-device)
				if err = encMulBF16To(enc, outBuf, s.lb[li].layerScalar, outBuf, outOff, 0, outOff, s.dModel); err != nil {
					endEncodingFast(enc)
					return nil, false, err
				}
			}
		}
		readRows, outRows = outRows, inRows // this layer's outputs feed the next layer
		readOff = rowOff
	}
	endEncodingFast(enc)
	commitCommandBufferFast(cb)
	waitUntilCompletedFast(cb)
	if K > 0 {
		if usingDirectOutputRows {
			s.denseBatch.lastRows = directOutputRows[0]
			s.denseBatch.lastRowOff = directOutputOff[:K]
			s.denseBatch.lastK = K
		} else if directLastOut && readLastOnly {
			s.denseBatch.readOffStack[0] = 0
			s.denseBatch.lastRows = lastOutBuf
			s.denseBatch.lastRowOff = s.denseBatch.readOffStack[:1]
			s.denseBatch.lastK = 1
		} else {
			s.denseBatch.lastRows = readRows[0]
			s.denseBatch.lastRowOff = readOff[:K]
			s.denseBatch.lastK = K
		}
	}
	if err := s.reloadDevicePagedKVFromLinear(basePos + K); err != nil {
		return nil, false, err
	}

	if readResult {
		if readLastOnly {
			out = s.denseBatch.lastResult[:1]
			out[0] = lastDst
			if !directLastOut {
				off := int(readOff[K-1])
				copy(out[0], unsafe.Slice((*byte)(readRows[K-1].Contents()), off+rowBytes)[off:]) // readRows = final layer out
			}
			return out, true, nil
		}
		if len(dstRows) >= K {
			out = dstRows[:K]
		} else {
			out = make([][]byte, K)
		}
		for i := 0; i < K; i++ {
			if usingDirectOutputRows {
				out[i] = out[i][:rowBytes]
				continue
			}
			if cap(out[i]) < rowBytes {
				out[i] = make([]byte, rowBytes)
			} else {
				out[i] = out[i][:rowBytes]
			}
			off := int(readOff[i])
			copy(out[i], unsafe.Slice((*byte)(readRows[i].Contents()), off+rowBytes)[off:]) // readRows = final layer out
		}
	}
	return out, true, nil
}

// verifyBatched is the MTP verify's batched fast path: it embeds the K ids, runs them through the
// resident stack in ONE pass at positions [pos, pos+K), writes their K/V into the caches, and returns
// each id's NEXT-token greedy (greedys[i] = the target's greedy of the hidden after ids[i]). It does
// NOT advance s.pos — MTPDecode sets the position to the committed length after accept/reject, exactly
// as the sequential verify leaves it. ok is false (no work, no cache mutation) for models outside the
// dense path (PLE / MoE / recorded-ICB / shared-KV), where MTPDecode steps sequentially instead — both
// paths produce the identical greedys, so the token stream is unchanged either way.
func (s *ArchSession) verifyBatched(ids []int32) (greedys []int32, ok bool, err error) {
	return s.verifyBatchedInto(ids, nil)
}

func (s *ArchSession) verifyBatchedHiddens(ids []int32) ([][]byte, bool, error) {
	if len(ids) == 0 {
		return nil, false, core.NewError("native.verifyBatchedHiddens: empty batch")
	}
	if s.perLayerInput != nil || s.pos+len(ids) > s.maxLen {
		return nil, false, nil
	}
	var embStack [16][]byte
	var embs [][]byte
	if len(ids) <= len(embStack) {
		embs = embStack[:len(ids)]
	} else {
		embs = make([][]byte, len(ids))
	}
	if s.canUseEmbedScratch() {
		rowBytes := s.arch.Hidden * bf16Size
		need := len(ids) * rowBytes
		if cap(s.embedScratch) < need {
			s.embedScratch = make([]byte, need)
		} else {
			s.embedScratch = s.embedScratch[:need]
		}
		for i, id := range ids {
			dst := s.embedScratch[i*rowBytes : (i+1)*rowBytes]
			emb, eerr := s.embedInto(dst, id)
			if eerr != nil {
				return nil, false, eerr
			}
			if len(emb) != rowBytes {
				return nil, false, core.NewError("native.verifyBatchedHiddens: embedInto returned wrong hidden size")
			}
			embs[i] = emb
		}
	} else {
		for i, id := range ids {
			emb, eerr := s.embed(id)
			if eerr != nil {
				return nil, false, eerr
			}
			embs[i] = emb
		}
	}
	var (
		hiddens [][]byte
		ok      bool
		err     error
	)
	withAutoreleasePool(func() {
		if rows, rowsOK := s.mtpVerifyHiddenRowsScratch(len(ids), s.arch.Hidden*bf16Size); rowsOK {
			hiddens, ok, err = s.state.stepTokensBatchedDenseInto(embs, s.pos, rows)
		} else {
			hiddens, ok, err = s.state.stepTokensBatchedDense(embs, s.pos)
		}
	})
	if err != nil || !ok {
		return nil, ok, err
	}
	return hiddens, true, nil
}

func (s *ArchSession) verifyBatchedInto(ids []int32, greedys []int32) ([]int32, bool, error) {
	if len(ids) == 0 {
		return nil, false, core.NewError("native.verifyBatched: empty batch")
	}
	if s.perLayerInput != nil || s.pos+len(ids) > s.maxLen {
		return nil, false, nil // PLE models / no cache headroom → sequential fallback
	}
	var embStack [16][]byte
	var embs [][]byte
	if len(ids) <= len(embStack) {
		embs = embStack[:len(ids)]
	} else {
		embs = make([][]byte, len(ids))
	}
	if s.canUseEmbedScratch() {
		rowBytes := s.arch.Hidden * bf16Size
		need := len(ids) * rowBytes
		if cap(s.embedScratch) < need {
			s.embedScratch = make([]byte, need)
		} else {
			s.embedScratch = s.embedScratch[:need]
		}
		for i, id := range ids {
			dst := s.embedScratch[i*rowBytes : (i+1)*rowBytes]
			emb, eerr := s.embedInto(dst, id)
			if eerr != nil {
				return nil, false, eerr
			}
			if len(emb) != rowBytes {
				return nil, false, core.NewError("native.verifyBatched: embedInto returned wrong hidden size")
			}
			embs[i] = emb
		}
	} else {
		for i, id := range ids {
			e, eerr := s.embed(id)
			if eerr != nil {
				return nil, false, eerr
			}
			embs[i] = e
		}
	}
	if s.canUseDirectHeadGreedy() {
		if len(greedys) < len(ids) {
			greedys = make([]int32, len(ids))
		} else {
			greedys = greedys[:len(ids)]
		}
		var (
			ok  bool
			err error
		)
		withAutoreleasePool(func() {
			ok, err = s.state.stepTokensBatchedDenseNoResult(embs, s.pos)
			if err != nil || !ok {
				return
			}
			err = s.encodePackedGreedyRowsInto(s.state.denseBatch.lastRows, s.state.denseBatch.lastRowOff, len(ids), greedys)
		})
		if err != nil || !ok {
			return nil, ok, err
		}
		return greedys, true, nil
	}
	var (
		hiddens [][]byte
		ok      bool
		err     error
	)
	withAutoreleasePool(func() {
		if rows, rowsOK := s.mtpVerifyHiddenRowsScratch(len(ids), s.arch.Hidden*bf16Size); rowsOK {
			hiddens, ok, err = s.state.stepTokensBatchedDenseInto(embs, s.pos, rows)
		} else {
			hiddens, ok, err = s.state.stepTokensBatchedDense(embs, s.pos)
		}
	})
	if err != nil || !ok {
		return nil, ok, err
	}
	if len(greedys) < len(hiddens) {
		greedys = make([]int32, len(hiddens))
	} else {
		greedys = greedys[:len(hiddens)]
	}
	for i, h := range hiddens {
		g, gerr := s.greedyOf(h)
		if gerr != nil {
			return nil, false, gerr
		}
		greedys[i] = g
	}
	return greedys, true, nil
}

func (s *ArchSession) verifyBatchedCrossesSlidingRingWrap(n int) bool {
	if s == nil || n <= 0 || s.arch.SlidingWindow <= 0 || s.arch.SlidingWindow >= s.maxLen {
		return false
	}
	window := s.arch.SlidingWindow
	if s.pos%window+n <= window {
		return false
	}
	for _, spec := range s.state.specs {
		if spec.OwnsCache() && spec.Attention != model.GlobalAttention {
			return true
		}
	}
	return false
}

func (s *ArchSession) rememberDenseBatchRetainedHidden(row int) error {
	if s == nil || row < 0 || row >= len(s.state.denseBatch.lastRowOff) || s.state.denseBatch.lastRows == nil {
		return core.NewError("native.verifyBatched: retained hidden row is unavailable")
	}
	rowBytes := s.arch.Hidden * bf16Size
	off := int(s.state.denseBatch.lastRowOff[row])
	if off < 0 || off+rowBytes > int(bufferLengthFast(s.state.denseBatch.lastRows)) {
		return core.NewError("native.verifyBatched: retained hidden row is out of range")
	}
	base := unsafe.Pointer((*byte)(s.state.denseBatch.lastRows.Contents()))
	s.rememberRetainedHiddenFrom((*byte)(unsafe.Add(base, off)))
	return nil
}

func (s *ArchSession) encodePackedGreedyRowsInto(rows metal.MTLBuffer, rowOff []uint, n int, greedys []int32) error {
	if rows == nil || len(rowOff) < n || len(greedys) < n {
		return core.NewError("native.verifyBatched: missing packed dense rows")
	}
	var scratchStack [16]*headGreedyScratch
	scratches := scratchStack[:0]
	if n > len(scratchStack) {
		scratches = make([]*headGreedyScratch, 0, n)
	}
	cb := commandBufferFast(queue)
	enc := computeCommandEncoderFast(cb)
	for i := 0; i < n; i++ {
		scratch, ok, err := s.headEnc.encodeGreedyAt(enc, rows, rowOff[i], nil)
		if err != nil || !ok {
			endEncodingFast(enc)
			for _, sc := range scratches {
				s.headEnc.putGreedyScratch(sc)
			}
			if err != nil {
				return err
			}
			return core.NewError("native.verifyBatched: direct head greedy unavailable")
		}
		scratches = append(scratches, scratch)
	}
	endEncodingFast(enc)
	commitCommandBufferFast(cb)
	waitUntilCompletedFast(cb)
	for i, scratch := range scratches {
		greedys[i] = scratch.token()
		s.headEnc.putGreedyScratch(scratch)
	}
	for i, token := range greedys[:n] {
		if token < 0 || int(token) >= s.arch.Vocab {
			return core.NewError(core.Sprintf("native.verifyBatched: greedy row %d returned invalid token %d for vocab %d", i, token, s.arch.Vocab))
		}
	}
	return nil
}
