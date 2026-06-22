// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

// nocopy_matvec.go is the resident-weight sibling of MatVecBF16: the matrix is supplied as a bufView — a
// no-copy view into a resident shard buffer (from shardBuffers.bufFor) — instead of host bytes, so a
// per-token op over a FIXED weight binds the weight at its shard offset rather than re-uploading the whole
// matrix to a fresh Metal buffer every call. This is the gemma4 PLE projection's hot path: PerLayerInputs
// runs once per token over a model-level [numLayers·pliDim, dModel] projection that never changes, and the
// host-bytes MatVecBF16 was the single biggest per-token tail cost — the NewBufferWithBytes balloon the
// CPU profile flagged (52% cum). Byte-identical to MatVecBF16(mat, vec) when matView backs the same bytes.

// pleResidentDisabled forces the PLE projection back onto the host-bytes MatVecBF16 path (a test hook for
// the resident-vs-host byte-identity check; always false in production).
var pleResidentDisabled bool

// resetResidentBufsForTest clears the address-keyed resident-weight cache. residentBytes assumes ONE model
// per process (keys by &weight[0] in the stable mmap); a test that loads several models reuses freed mmap
// addresses after Close → munmap, so a stale cache hit returns a prior model's buffer. Tests that load more
// than one model must reset between loads. Never called in production (a served process loads one model).
func resetResidentBufsForTest() {
	residentBufMu.Lock()
	residentBufs = map[uintptr]residentBuf{}
	residentBufMu.Unlock()
}

// MatVecBF16Buf computes out[outDim] = mat[outDim,inDim] @ vec[inDim] in bf16, with the matrix bound from a
// resident no-copy buffer view (matView) at its offset; vec/out stay per-call (small). A nil matView.buf is
// an error — the caller falls back to MatVecBF16 when no shard buffer is available.
func MatVecBF16Buf(matView bufView, vec []byte, outDim, inDim int) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if matView.buf == nil {
		return nil, core.NewError("native.MatVecBF16Buf: nil resident matrix buffer")
	}
	if len(vec) != inDim*bf16Size {
		return nil, core.NewError("native.MatVecBF16Buf: len(vec) must equal inDim*2 bytes")
	}
	if outDim == 0 || inDim == 0 {
		return make([]byte, outDim*bf16Size), nil
	}
	bm, bn, sm, sn, tm, tn := gemvTiles(inDim, outDim)
	name := core.Sprintf("gemv_bfloat16_bm%d_bn%d_sm%d_sn%d_tm%d_tn%d_nc0_axpby0", bm, bn, sm, sn, tm, tn)
	pso, err := pipelineFor(name)
	if err != nil {
		return nil, err
	}
	outLen := outDim * bf16Size
	out := make([]byte, outLen)
	withAutoreleasePool(func() {
		vecBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&vec[0]), uint(len(vec)), metal.MTLResourceStorageModeShared)
		outBuf := device.NewBufferWithLengthOptions(uint(outLen), metal.MTLResourceStorageModeShared)

		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		enc.SetComputePipelineState(pso)
		enc.SetBufferWithOffsetAtIndex(matView.buf, matView.off, 0) // resident matrix bound at its shard offset
		enc.SetBufferWithOffsetAtIndex(vecBuf, 0, 1)
		enc.SetBufferWithOffsetAtIndex(outBuf, 0, 3)
		setEncInt32(enc, int32(inDim), 4)
		setEncInt32(enc, int32(outDim), 5)
		setEncInt32(enc, int32(inDim), 6)
		setEncInt32(enc, 1, 9)
		setEncInt32(enc, 1, 10)
		setEncInt64(enc, 0, 11)
		setEncInt64(enc, 0, 12)

		nOutPerTgp := bm * sm * tm
		nTgp := (outDim + nOutPerTgp - 1) / nOutPerTgp
		enc.DispatchThreadgroupsThreadsPerThreadgroup(
			metal.MTLSize{Width: uint(nTgp), Height: 1, Depth: 1},
			metal.MTLSize{Width: 32, Height: uint(bn), Depth: uint(bm)},
		)
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()

		copy(out, unsafe.Slice((*byte)(outBuf.Contents()), outLen))
	})
	return out, nil
}
