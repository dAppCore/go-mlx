// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

// decode_step_batched.go — the MTP batched verify forward: K query tokens through one decode layer
// in ONE command buffer over the resident KV cache, with the layer weights uploaded ONCE. This is
// the speculative-decode speedup the sequential mtp.go verify (K separate steps = K command buffers
// + K weight uploads) leaves on the table: the target verifies the whole K-token draft block in a
// single submit. Each query row i decodes at position basePos+i, writes its K/V into the cache at
// row basePos+i, and attends [0..basePos+i] with the SAME single-query kernels the per-token step
// uses — so the layer output is BYTE-IDENTICAL to K sequential DecodeStepKV calls (proven in
// decode_step_batched_test.go), only without the per-token command-buffer + weight-upload overhead.
//
// Why byte-identical (not merely close): the heavy projections still run as per-row gemv (the exact
// kernel a single-token step uses), and the attention runs per-row single-query encSDPAStrided over
// the cache window [0..basePos+i] — identical dispatches to the sequential path, just encoded into
// one command buffer with the weights resident once. The cache write→read ordering across rows is
// Metal's automatic buffer hazard tracking (row i+1's attention reads the cache after row i's K/V
// write), so the K-position causal structure is exact. The remaining speedup — folding the K per-row
// projections into one steel GEMM (weight reuse across rows) — trades this byte-identity for
// token-identity (a GEMM reduces over the contraction in a different order than K gemvs); that is the
// metal-MTP-parity follow-up. This v1 keeps byte-identity and still wins the submit + upload overhead.

// DecodeLayerBatchedKV runs one full decode layer (attention half + gemma MLP half, both residuals)
// for K query tokens at positions [basePos, basePos+K) in one command buffer, growing the seq-major
// KV caches at rows basePos..basePos+K-1. xs is the K input hiddens [K, dModel] bf16; the result is
// the K output hiddens [K, dModel] bf16. kCache/vCache are updated in place. Byte-identical to
// stepping the same K rows one at a time with DecodeStepKV (same kernels, same cache evolution).
func DecodeLayerBatchedKV(
	xs, attnNormW, wQ, wK, wV, wO, kCache, vCache, mlpNormW, wGate, wUp, wDown []byte,
	dModel, nHeads, nKVHeads, headDim, maxLen, dFF, basePos, K int,
	base, scale, eps float32,
) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if K <= 0 {
		return nil, core.NewError("native.DecodeLayerBatchedKV: K must be > 0")
	}
	if basePos < 0 || basePos+K > maxLen {
		return nil, core.NewError("native.DecodeLayerBatchedKV: [basePos, basePos+K) out of [0,maxLen)")
	}
	rowBytes := dModel * bf16Size
	if len(xs) != K*rowBytes {
		return nil, core.NewError("native.DecodeLayerBatchedKV: xs must be K*dModel bf16 bytes")
	}
	// the per-row shape contract for the weights + caches (validated at the first row's position).
	if err := validateStepKV(xs[:rowBytes], attnNormW, wQ, wK, wV, wO, kCache, vCache, dModel, nHeads, nKVHeads, headDim, maxLen, basePos); err != nil {
		return nil, err
	}
	if len(mlpNormW) != dModel*bf16Size {
		return nil, core.NewError("native.DecodeLayerBatchedKV: mlpNormW must be dModel bf16 bytes")
	}
	if len(wGate) != dFF*dModel*bf16Size || len(wUp) != dFF*dModel*bf16Size || len(wDown) != dModel*dFF*bf16Size {
		return nil, core.NewError("native.DecodeLayerBatchedKV: MLP weight size mismatch")
	}
	qDim, kvDim := nHeads*headDim, nKVHeads*headDim
	out := make([]byte, K*rowBytes)
	var encErr error
	withAutoreleasePool(func() {
		// the layer weights are uploaded ONCE and reused across all K rows — the win over K separate
		// DecodeStepKV calls, each of which re-uploads every weight.
		proj := bf16Projector{
			wQ: copyView(wQ), wK: copyView(wK), wV: copyView(wV), wO: copyView(wO),
			wGate: copyView(wGate), wUp: copyView(wUp), wDown: copyView(wDown),
			dModel: dModel, qDim: qDim, kvDim: kvDim, dFF: dFF,
		}
		nwBuf, mnwBuf := sharedBytes(attnNormW), sharedBytes(mlpNormW)
		kBuf, vBuf := sharedBytes(kCache), sharedBytes(vCache)
		asc := newAttnScratch(dModel, qDim, kvDim)
		msc := newMLPScratch(dModel, dFF)
		hBuf := scratchBF16(dModel)
		xRow := make([]metal.MTLBuffer, K)
		outRow := make([]metal.MTLBuffer, K)
		offBuf := make([]metal.MTLBuffer, K)
		for i := 0; i < K; i++ {
			xRow[i] = sharedBytes(xs[i*rowBytes : (i+1)*rowBytes])
			outRow[i] = scratchBF16(dModel)
			off := int32(basePos + i)
			offBuf[i] = device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&off), 4, metal.MTLResourceStorageModeShared)
		}

		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		for i := 0; i < K; i++ {
			// attention half: project q/k/v from row i, write k/v into the cache at row basePos+i,
			// attend [0..basePos+i] (single-query, the exact per-token kernel) → h.
			if encErr = encAttnHalfKV(enc, xRow[i], kBuf, vBuf, offBuf[i], hBuf,
				bufView{buf: nwBuf}, bufView{}, bufView{}, bufView{}, nil, asc, proj,
				dModel, nHeads, nKVHeads, headDim, basePos+i, 0, headDim, base, scale, eps, nil); encErr != nil {
				enc.EndEncoding()
				return
			}
			// MLP half on h → row i's output.
			if encErr = encMLPHalfBF16(enc, hBuf, outRow[i], bufView{buf: mnwBuf}, bufView{}, msc, proj, dModel, dFF, eps); encErr != nil {
				enc.EndEncoding()
				return
			}
		}
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()
		for i := 0; i < K; i++ {
			copy(out[i*rowBytes:(i+1)*rowBytes], unsafe.Slice((*byte)(outRow[i].Contents()), rowBytes))
		}
		copy(kCache, unsafe.Slice((*byte)(kBuf.Contents()), len(kCache)))
		copy(vCache, unsafe.Slice((*byte)(vBuf.Contents()), len(vCache)))
	})
	return out, encErr
}
