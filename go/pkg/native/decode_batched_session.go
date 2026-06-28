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
	inRowsStack  [16]metal.MTLBuffer
	outRowsStack [16]metal.MTLBuffer
	offBufStack  [16]metal.MTLBuffer
	offPtrStack  [16]*int32
	offOffStack  [16]uint
	rowOffStack  [16]uint
	inRows       []metal.MTLBuffer
	outRows      []metal.MTLBuffer
	offBuf       []metal.MTLBuffer
	offPtr       []*int32
	offOff       []uint
	rowOff       []uint
	offPacked    metal.MTLBuffer
	offPackedCap int
	inPacked     metal.MTLBuffer
	outPacked    metal.MTLBuffer
	rowPackedCap int
	rowBytes     int
	lastRows     metal.MTLBuffer
	lastRowOff   []uint
	lastK        int
	lastResult   [1][]byte
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
	return s.stepTokensBatchedDenseResult(embs, basePos, true, false, nil)
}

func (s *archDecodeState) stepTokensBatchedDenseNoResult(embs [][]byte, basePos int) (ok bool, err error) {
	_, ok, err = s.stepTokensBatchedDenseResult(embs, basePos, false, false, nil)
	return ok, err
}

func (s *archDecodeState) stepTokensBatchedDenseLastInto(embs [][]byte, basePos int, dst []byte) (last []byte, ok bool, err error) {
	out, ok, err := s.stepTokensBatchedDenseResult(embs, basePos, true, true, dst)
	if err != nil || !ok {
		return nil, ok, err
	}
	if len(out) != 1 {
		return nil, true, core.NewError("native.stepTokensBatchedDenseLast: hidden result count mismatch")
	}
	return out[0], true, nil
}

func (s *archDecodeState) stepTokensBatchedDenseResult(embs [][]byte, basePos int, readResult, readLastOnly bool, lastDst []byte) (out [][]byte, ok bool, err error) {
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

	rowBytes := s.dModel * bf16Size
	// K-wide working rows (ping-ponged across layers) + per-row position buffers, retained on the state.
	inRows, outRows, offBuf, offPtr, offOff, rowOff := s.denseBatch.rows(K, s.dModel)
	for i := 0; i < K; i++ {
		off := int(rowOff[i])
		copy(unsafe.Slice((*byte)(inRows[i].Contents()), off+rowBytes)[off:], embs[i])
		*offPtr[i] = int32(basePos + i)
	}

	cb := queue.CommandBuffer()
	enc := cb.ComputeCommandEncoder()
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
			if err = encAttnHalfKVInputAt(enc, inRows[i], rowOff[i], s.lb[li].kCache, s.lb[li].vCache, offBuf[i], s.hBuf, offOff[i],
				s.lb[li].anw, s.lb[li].postAttnNorm, s.lb[li].qNorm, s.lb[li].kNorm, s.valueNormOnes, s.asc, s.lb[li].proj,
				s.dModel, s.nHeads, lkv, lhd, basePos+i, slideW, rotDim, rbase, s.scale, s.eps, layerRopeFreqs); err != nil {
				enc.EndEncoding()
				return nil, false, err
			}
			if err = encMLPHalfBF16At(enc, s.hBuf, outRows[i], rowOff[i], s.lb[li].mnw, s.lb[li].postFFNorm, s.msc, s.lb[li].proj, s.dModel, lff, s.eps); err != nil {
				enc.EndEncoding()
				return nil, false, err
			}
			if s.lb[li].layerScalar != nil { // gemma4 per-layer output scalar (on-device)
				if err = encMulBF16To(enc, outRows[i], s.lb[li].layerScalar, outRows[i], rowOff[i], 0, rowOff[i], s.dModel); err != nil {
					enc.EndEncoding()
					return nil, false, err
				}
			}
		}
		inRows, outRows = outRows, inRows // this layer's outputs feed the next layer
	}
	enc.EndEncoding()
	cb.Commit()
	cb.WaitUntilCompleted()
	if K > 0 {
		s.denseBatch.lastRows = inRows[0]
		s.denseBatch.lastRowOff = rowOff[:K]
		s.denseBatch.lastK = K
	}

	if readResult {
		if readLastOnly {
			out = s.denseBatch.lastResult[:1]
			if cap(lastDst) < rowBytes {
				lastDst = make([]byte, rowBytes)
			} else {
				lastDst = lastDst[:rowBytes]
			}
			out[0] = lastDst
			off := int(rowOff[K-1])
			copy(out[0], unsafe.Slice((*byte)(inRows[K-1].Contents()), off+rowBytes)[off:]) // inRows = last swap = final layer out
			return out, true, nil
		}
		out = make([][]byte, K)
		for i := 0; i < K; i++ {
			out[i] = make([]byte, rowBytes)
			off := int(rowOff[i])
			copy(out[i], unsafe.Slice((*byte)(inRows[i].Contents()), off+rowBytes)[off:]) // inRows = last swap = final layer out
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
	for i, id := range ids {
		e, eerr := s.embed(id)
		if eerr != nil {
			return nil, false, eerr
		}
		embs[i] = e
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
		hiddens, ok, err = s.state.stepTokensBatchedDense(embs, s.pos)
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

func (s *ArchSession) encodePackedGreedyRowsInto(rows metal.MTLBuffer, rowOff []uint, n int, greedys []int32) error {
	if rows == nil || len(rowOff) < n || len(greedys) < n {
		return core.NewError("native.verifyBatched: missing packed dense rows")
	}
	var scratchStack [16]*headGreedyScratch
	scratches := scratchStack[:0]
	if n > len(scratchStack) {
		scratches = make([]*headGreedyScratch, 0, n)
	}
	cb := queue.CommandBuffer()
	enc := cb.ComputeCommandEncoder()
	for i := 0; i < n; i++ {
		scratch, ok, err := s.headEnc.encodeGreedyAt(enc, rows, rowOff[i], nil)
		if err != nil || !ok {
			enc.EndEncoding()
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
	enc.EndEncoding()
	cb.Commit()
	cb.WaitUntilCompleted()
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
