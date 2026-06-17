// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"unsafe"

	"github.com/tmc/apple/metal"
)

// attentionReEncode runs the bf16 attention block `reps` times the REGULAR way —
// persistent buffers, but the 6 ops re-encoded into a fresh command buffer every
// rep (the host re-encode the ICB path replaces). Buffers are created once so the
// measurement isolates per-rep host ENCODE cost, not buffer churn; the A/B
// against AttentionBlockICB(reps) is the encode-bypass number. Returns after the
// last rep completes.
func attentionReEncode(x, normWeight, wQ, wO, kCache, vCache []byte, dModel, nHeads, nKVHeads, headDim, kvLen int, base, scale float32, offset int, eps float32, reps int) error {
	if err := ensureInit(); err != nil {
		return err
	}
	qDim := nHeads * headDim
	var encErr error
	withAutoreleasePool(func() {
		xBuf, nwBuf := sharedBytes(x), sharedBytes(normWeight)
		wqBuf, woBuf := sharedBytes(wQ), sharedBytes(wO)
		kBuf, vBuf := sharedBytes(kCache), sharedBytes(vCache)
		off := int32(offset)
		offBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&off), 4, metal.MTLResourceStorageModeShared)
		normed := scratchBF16(dModel)
		q, qr, attn := scratchBF16(qDim), scratchBF16(qDim), scratchBF16(qDim)
		attnOut, outBuf := scratchBF16(dModel), scratchBF16(dModel)
		_ = outBuf

		for r := 0; r < reps; r++ {
			cb := queue.CommandBuffer()
			enc := cb.ComputeCommandEncoder()
			if encErr = encRMSNormBF16(enc, xBuf, nwBuf, normed, dModel, eps); encErr != nil {
				enc.EndEncoding()
				return
			}
			_ = encGemvBF16(enc, wqBuf, normed, q, qDim, dModel)
			_ = encRoPEBF16(enc, q, qr, offBuf, nHeads, headDim, base, scale)
			_ = encSDPA(enc, qr, kBuf, vBuf, attn, nHeads, nKVHeads, headDim, kvLen, scale)
			_ = encGemvBF16(enc, woBuf, attn, attnOut, dModel, qDim)
			_ = encAddBF16(enc, xBuf, attnOut, outBuf, dModel)
			enc.EndEncoding()
			cb.Commit()
			cb.WaitUntilCompleted()
		}
	})
	return encErr
}
