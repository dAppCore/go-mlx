// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"unsafe"

	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/model"
	"github.com/tmc/apple/metal"
)

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
	return s.stepTokensBatchedDenseResult(embs, basePos, true)
}

func (s *archDecodeState) stepTokensBatchedDenseNoResult(embs [][]byte, basePos int) (ok bool, err error) {
	_, ok, err = s.stepTokensBatchedDenseResult(embs, basePos, false)
	return ok, err
}

func (s *archDecodeState) stepTokensBatchedDenseResult(embs [][]byte, basePos int, readResult bool) (out [][]byte, ok bool, err error) {
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
	// K-wide working rows (ping-ponged across layers) + per-row position buffers, allocated once.
	var inRowsStack [16]metal.MTLBuffer
	var outRowsStack [16]metal.MTLBuffer
	var offBufStack [16]metal.MTLBuffer
	var inRows []metal.MTLBuffer
	var outRows []metal.MTLBuffer
	var offBuf []metal.MTLBuffer
	if K <= len(inRowsStack) {
		inRows = inRowsStack[:K]
		outRows = outRowsStack[:K]
		offBuf = offBufStack[:K]
	} else {
		inRows = make([]metal.MTLBuffer, K)
		outRows = make([]metal.MTLBuffer, K)
		offBuf = make([]metal.MTLBuffer, K)
	}
	for i := 0; i < K; i++ {
		inRows[i] = scratchBF16(s.dModel)
		copy(unsafe.Slice((*byte)(inRows[i].Contents()), rowBytes), embs[i])
		outRows[i] = scratchBF16(s.dModel)
		off := int32(basePos + i)
		offBuf[i] = device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&off), 4, metal.MTLResourceStorageModeShared)
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
			if err = encAttnHalfKV(enc, inRows[i], s.lb[li].kCache, s.lb[li].vCache, offBuf[i], s.hBuf,
				s.lb[li].anw, s.lb[li].postAttnNorm, s.lb[li].qNorm, s.lb[li].kNorm, s.valueNormOnes, s.asc, s.lb[li].proj,
				s.dModel, s.nHeads, lkv, lhd, basePos+i, slideW, rotDim, rbase, s.scale, s.eps, layerRopeFreqs); err != nil {
				enc.EndEncoding()
				return nil, false, err
			}
			if err = encMLPHalfBF16(enc, s.hBuf, outRows[i], s.lb[li].mnw, s.lb[li].postFFNorm, s.msc, s.lb[li].proj, s.dModel, lff, s.eps); err != nil {
				enc.EndEncoding()
				return nil, false, err
			}
			if s.lb[li].layerScalar != nil { // gemma4 per-layer output scalar (on-device)
				if err = encMulBF16(enc, outRows[i], s.lb[li].layerScalar, outRows[i], s.dModel); err != nil {
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

	if readResult {
		out = make([][]byte, K)
		for i := 0; i < K; i++ {
			out[i] = make([]byte, rowBytes)
			copy(out[i], unsafe.Slice((*byte)(inRows[i].Contents()), rowBytes)) // inRows = last swap = final layer out
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
