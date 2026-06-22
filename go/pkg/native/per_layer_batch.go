// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"unsafe"

	"github.com/tmc/apple/metal"
)

// perLayerProjBatched runs the gemma4 PLE projection chain — steps 2-6 of PerLayerInputs: resident-weight
// matvec → ×projScale → RMSNorm(rows) → +perLayer → ×combineScale — as ONE command buffer: a single
// Commit()+WaitUntilCompleted() instead of five. That collapses five per-token GPU round-trips (~5×199µs ≈
// 1ms/token of host stall, GPU idle between) to one. The ops chain via device buffers (no per-op host
// download), driving the SAME kernels as the host path, so the result is byte-identical to the unbatched
// steps 2-6. Intermediate buffers are autoreleased (pool-freed); the projection weight is the resident
// no-copy shard view (projView).
func perLayerProjBatched(projView bufView, hidden, perLayer []byte, projScale float32, projNormW []byte, plDim, numLayers, pliDim, dModel int, eps float32) ([]byte, error) {
	out := make([]byte, plDim*bf16Size)
	var ferr error
	withAutoreleasePool(func() {
		mk := func(b []byte) metal.MTLBuffer {
			return device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&b[0]), uint(len(b)), metal.MTLResourceStorageModeShared)
		}
		nb := func() metal.MTLBuffer {
			return device.NewBufferWithLengthOptions(uint(plDim*bf16Size), metal.MTLResourceStorageModeShared)
		}
		hiddenBuf := mk(hidden)
		perLayerBuf := mk(perLayer)
		projNormWBuf := mk(projNormW)
		projScaleBuf := mk(bf16ConstBytes(plDim, projScale))
		combineScaleBuf := mk(bf16ConstBytes(plDim, gemma4PerLayerCombineScale))
		projectedBuf, scaledBuf, projNormedBuf, combinedBuf, outBuf := nb(), nb(), nb(), nb(), nb()

		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		encode := func() error {
			if err := encGemvBF16To(enc, projView.buf, hiddenBuf, projectedBuf, projView.off, 0, plDim, dModel); err != nil {
				return err
			}
			if err := encMulBF16(enc, projectedBuf, projScaleBuf, scaledBuf, plDim); err != nil {
				return err
			}
			if err := encRMSNormRowsBF16(enc, scaledBuf, projNormWBuf, projNormedBuf, 0, 0, 0, numLayers, pliDim, eps); err != nil {
				return err
			}
			if err := encAddBF16(enc, projNormedBuf, perLayerBuf, combinedBuf, plDim); err != nil {
				return err
			}
			return encMulBF16(enc, combinedBuf, combineScaleBuf, outBuf, plDim)
		}
		ferr = encode()
		enc.EndEncoding()
		if ferr != nil {
			return
		}
		cb.Commit()
		cb.WaitUntilCompleted()
		copy(out, unsafe.Slice((*byte)(outBuf.Contents()), plDim*bf16Size))
	})
	return out, ferr
}
