// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"unsafe"

	g4 "dappco.re/go/mlx/pkg/model/gemma4"
	"github.com/tmc/apple/metal"
)

// nativeDecoder is the no-cgo backend's implementation of the shared gemma4 compute
// seam (model/gemma4.Decoder): each primitive RECORDS one op into a Metal command
// encoder, and Commit/Read flush that batch once. So the gemma4 decode SEQUENCE
// (which lives once in pkg/model/gemma4) drives native's proven enc* ops with
// native's whole-token-in-one-command-buffer speed — no dispatch-per-op, no bytes
// crossing the seam mid-decode. This replaces native's home-grown stepToken
// orchestration (the code that drifted from metal) with the shared one.
//
// MUST be constructed + driven inside a withAutoreleasePool (newMLPScratch + the
// command buffers are pool-scoped, like every native decode path). Single-goroutine.
type nativeDecoder struct {
	cb     metal.MTLCommandBuffer
	enc    metal.MTLComputeCommandEncoder
	active bool          // an encoder is open + recording
	mlpSc  mlpScratch    // the SwiGLU gelu-chain intermediates (sized once for dFF)
	err    error         // first recording error; surfaced at Commit/Read (record-style)
}

// nativeBuf is native's opaque g4.Buffer: a resident Metal buffer plus its byte
// length (so Read knows how much to copy back).
type nativeBuf struct {
	mb metal.MTLBuffer
	n  int
}

var _ g4.Decoder = (*nativeDecoder)(nil)

// newNativeDecoder builds a decoder for a model of the given dims (dModel/dFF size
// the SwiGLU scratch). MUST be called inside a withAutoreleasePool.
func newNativeDecoder(dModel, dFF int) *nativeDecoder {
	return &nativeDecoder{mlpSc: newMLPScratch(dModel, dFF)}
}

func (d *nativeDecoder) ensure() {
	if !d.active {
		d.cb = queue.CommandBuffer()
		d.enc = d.cb.ComputeCommandEncoder()
		d.active = true
	}
}

func (d *nativeDecoder) flush() {
	if d.active {
		d.enc.EndEncoding()
		d.cb.Commit()
		d.cb.WaitUntilCompleted()
		d.active = false
	}
}

func (d *nativeDecoder) setErr(err error) {
	if err != nil && d.err == nil {
		d.err = err
	}
}

// b unwraps an opaque g4.Buffer to native's nativeBuf.
func b(buf g4.Buffer) nativeBuf { return buf.(nativeBuf) }

func (d *nativeDecoder) Alloc(nBytes int) g4.Buffer {
	return nativeBuf{mb: device.NewBufferWithLengthOptions(uint(nBytes), metal.MTLResourceStorageModeShared), n: nBytes}
}

func (d *nativeDecoder) Upload(data []byte) g4.Buffer {
	return nativeBuf{mb: sharedBytes(data), n: len(data)}
}

func (d *nativeDecoder) Read(buf g4.Buffer) ([]byte, error) {
	d.flush()
	if d.err != nil {
		return nil, d.err
	}
	nb := b(buf)
	out := make([]byte, nb.n)
	copy(out, unsafe.Slice((*byte)(nb.mb.Contents()), nb.n))
	return out, nil
}

func (d *nativeDecoder) RMSNorm(out, src, weight g4.Buffer, srcOff, outOff, rows, axisSize int, eps float32) {
	d.ensure()
	d.setErr(encRMSNormRowsBF16(d.enc, b(src).mb, b(weight).mb, b(out).mb, uint(srcOff*bf16Size), uint(outOff*bf16Size), rows, axisSize, eps))
}

func (d *nativeDecoder) Proj(out, src, weight g4.Buffer, outOff, outDim, inDim int) {
	d.ensure()
	d.setErr(encGemvBF16To(d.enc, b(weight).mb, b(src).mb, b(out).mb, uint(outOff*bf16Size), outDim, inDim))
}

func (d *nativeDecoder) QuantProj(out, src, packed, scales, biases g4.Buffer, outOff, outDim, inDim, groupSize, bits int) {
	d.ensure()
	d.setErr(encQMVBF16(d.enc, b(packed).mb, b(scales).mb, b(biases).mb, b(src).mb, b(out).mb, uint(outOff*bf16Size), outDim, inDim, groupSize, bits))
}

func (d *nativeDecoder) RoPE(out, src, pos, freqs g4.Buffer, srcOff, outOff, nHeads, headDim, rotaryDim int, base, scale float32) {
	d.ensure()
	var freqBuf metal.MTLBuffer
	if freqs != nil {
		freqBuf = b(freqs).mb
	}
	d.setErr(encRopeDecode(d.enc, b(src).mb, b(out).mb, uint(srcOff*bf16Size), uint(outOff*bf16Size), b(pos).mb, freqBuf, nHeads, headDim, rotaryDim, base, scale))
}

func (d *nativeDecoder) SDPA(out, q, kCache, vCache g4.Buffer, nHeads, nKVHeads, headDim, n int, scale float32, kvElemOff int) {
	d.ensure()
	kvDim := nKVHeads * headDim
	d.setErr(encSDPAStrided(d.enc, b(q).mb, b(kCache).mb, b(vCache).mb, b(out).mb,
		nHeads, nKVHeads, headDim, n,
		int64(headDim), int64(kvDim), int64(headDim), int64(kvDim), scale, uint(kvElemOff*bf16Size)))
}

func (d *nativeDecoder) SwiGLU(out, gate, up g4.Buffer, n int) {
	d.ensure()
	encGeluGateMul(d.enc, b(gate).mb, b(up).mb, b(out).mb, d.mlpSc, n)
}

func (d *nativeDecoder) Add(out, a, bb g4.Buffer, n int) {
	d.ensure()
	d.setErr(encAddBF16(d.enc, b(a).mb, b(bb).mb, b(out).mb, n))
}

func (d *nativeDecoder) Mul(out, a, scalar g4.Buffer, n int) {
	d.ensure()
	d.setErr(encMulBF16(d.enc, b(a).mb, b(scalar).mb, b(out).mb, n))
}

func (d *nativeDecoder) Commit() error {
	d.flush()
	return d.err
}
