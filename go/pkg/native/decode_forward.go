// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

// DecodeForward runs a real multi-layer, multi-token decode forward on the no-cgo
// path: each token flows through every layer (residual stream layer→layer), each
// layer APPENDS its K/V to its OWN growing cache at the token's position, and the
// whole N-layer stack for a token is submitted in ONE command buffer + commit
// (how a real decode step submits). It is DecodeStepKV (the parity-proven real
// layer) wired into the autoregressive loop with resident per-layer caches and
// shared scratch — no per-token/per-layer buffer churn, so the per-token cost is
// the encode + the growing-window GPU work, nothing else.
//
// inputs are the T token hidden vectors (each dModel bf16) — the embedding/lm_head
// /sampler are separate concerns (a real model load, Snider's call); this exercises
// the transformer stack + KV growth. Returns the T per-token output vectors. With
// the same weights/inputs it equals stepping DecodeStepKV token-by-token,
// layer-by-layer (gated byte-for-byte in the tests). All raw bf16.

// DecodeLayerWeights is one decode layer's weights (raw bf16 bytes): attention
// norm, Q/K/V/O projections, MLP norm, gate/up/down. wQ is (nHeads·headDim ×
// dModel), wK/wV are (nKVHeads·headDim × dModel), wO is (dModel × nHeads·headDim),
// wGate/wUp are (dFF × dModel), wDown is (dModel × dFF).
type DecodeLayerWeights struct {
	AttnNormW, WQ, WK, WV, WO   []byte
	MLPNormW, WGate, WUp, WDown []byte
	// MoE, when non-nil, replaces the dense MLP half with the gemma4 dual-branch MoE
	// feed-forward (MoEBlockBF16) for this layer. The dense MLPNormW/WGate/WUp/WDown
	// are then unused (the local MLP lives in MoE.WGate/WUp/WDown). Only honoured by
	// the arch executor (DecodeForwardArch) when the layer's spec.MoE is set.
	MoE *MoELayerWeights
	// gemma4 norms the loader populates but the decode does NOT consume yet: QK-norm
	// (per-head RMSNorm on Q/K before RoPE), post-attention norm, post-feed-forward
	// norm. The native dense decode currently does pre-attn + pre-FF only; wiring these
	// four into encAttnHalfKV/encMLPHalfBF16 is the "gemma4 norm reconciliation" slice.
	// nil when the checkpoint omits them. (MLPNormW is the pre-feed-forward norm.)
	QNormW, KNormW, PostAttnNormW, PostFFNormW []byte
	// LayerScalarW is gemma4's per-layer output scalar (shape [1] bf16): the layer's final
	// hidden is multiplied by it before the next layer (applied by the arch executor). nil
	// when the checkpoint omits it.
	LayerScalarW []byte
	// gemma4 per-layer-input tower (E2B/E4B), bf16: the per-layer-input gate + projection and the
	// post-per-layer-input norm, applied host-side by PerLayerInputGateBF16 (the bf16 sibling of
	// the quant path). nil when the model has no PLE tower.
	PerLayerGate, PerLayerProjection, PostPerLayerInputNormW []byte
	// DFF is the per-layer MatFormer FFN width (E2B/E4B vary it, 6144/12288); 0 ⇒ the arch default.
	// The bf16 decode reads it so the MLP projector matches each layer's actual gate/up/down width.
	DFF int
}

type decodeForwardStepScratch struct {
	hBuf, xA, xB metal.MTLBuffer
	offBuf       metal.MTLBuffer
	offPtr       *int32
	hBufPtr      *byte
	xAPtr, xBPtr *byte
	dModel       int
}

func newDecodeForwardStepScratch(dModel int) decodeForwardStepScratch {
	off := int32(0)
	hBuf := scratchBF16(dModel)
	xA, xB := scratchBF16(dModel), scratchBF16(dModel)
	offBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&off), 4, metal.MTLResourceStorageModeShared)
	return decodeForwardStepScratch{
		hBuf:    hBuf,
		xA:      xA,
		xB:      xB,
		offBuf:  offBuf,
		offPtr:  (*int32)(offBuf.Contents()),
		hBufPtr: (*byte)(hBuf.Contents()),
		xAPtr:   (*byte)(xA.Contents()),
		xBPtr:   (*byte)(xB.Contents()),
		dModel:  dModel,
	}
}

func (s *decodeForwardStepScratch) bufferPtr(buf metal.MTLBuffer) *byte {
	if s == nil || buf == nil {
		return nil
	}
	switch buf {
	case s.hBuf:
		if s.hBufPtr != nil {
			return s.hBufPtr
		}
	case s.xA:
		if s.xAPtr != nil {
			return s.xAPtr
		}
	case s.xB:
		if s.xBPtr != nil {
			return s.xBPtr
		}
	}
	return (*byte)(buf.Contents())
}

func (s *decodeForwardStepScratch) bufferBytes(buf metal.MTLBuffer) []byte {
	return unsafe.Slice(s.bufferPtr(buf), s.dModel*bf16Size)
}

func (s *decodeForwardStepScratch) seed(pos int, input []byte) {
	*s.offPtr = int32(pos)
	copy(s.bufferBytes(s.xA), input)
}

func (s *decodeForwardStepScratch) copyBuffer(dst []byte, src metal.MTLBuffer) {
	copy(dst, s.bufferBytes(src))
}

// DecodeForward — see file header.
func DecodeForward(
	inputs [][]byte, layers []DecodeLayerWeights,
	dModel, nHeads, nKVHeads, headDim, maxLen, dFF int,
	base, scale, eps float32,
) ([][]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	nLayers := len(layers)
	if nLayers == 0 {
		return nil, core.NewError("native.DecodeForward: no layers")
	}
	T := len(inputs)
	if T == 0 {
		return nil, core.NewError("native.DecodeForward: no inputs")
	}
	if T > maxLen {
		return nil, core.NewError("native.DecodeForward: more tokens than maxLen cache rows")
	}
	qDim, kvDim := nHeads*headDim, nKVHeads*headDim
	for i := range inputs {
		if len(inputs[i]) != dModel*bf16Size {
			return nil, core.NewError("native.DecodeForward: each input must be dModel bf16 bytes")
		}
	}
	for li := range layers {
		w := layers[li]
		if len(w.AttnNormW) != dModel*bf16Size || len(w.MLPNormW) != dModel*bf16Size {
			return nil, core.NewError("native.DecodeForward: layer norm weight size mismatch")
		}
		if len(w.WQ) != qDim*dModel*bf16Size || len(w.WO) != dModel*qDim*bf16Size {
			return nil, core.NewError("native.DecodeForward: layer wQ/wO size mismatch")
		}
		if len(w.WK) != kvDim*dModel*bf16Size || len(w.WV) != kvDim*dModel*bf16Size {
			return nil, core.NewError("native.DecodeForward: layer wK/wV size mismatch")
		}
		if len(w.WGate) != dFF*dModel*bf16Size || len(w.WUp) != dFF*dModel*bf16Size || len(w.WDown) != dModel*dFF*bf16Size {
			return nil, core.NewError("native.DecodeForward: layer MLP weight size mismatch")
		}
	}

	outputs := make([][]byte, T)
	for i := range outputs {
		outputs[i] = make([]byte, dModel*bf16Size)
	}
	var encErr error
	withAutoreleasePool(func() {
		// resident per-layer weight buffers + per-layer caches (caches zeroed; rows
		// fill as tokens append). Created once for the whole forward.
		type layerBufs struct {
			anw, wq, wk, wv, wo, mnw, wg, wu, wd metal.MTLBuffer
			pan, pfn                             metal.MTLBuffer // gemma4 post-attn/post-FF norms (nil = skip)
			qn, kn                               metal.MTLBuffer // gemma4 per-head QK-norm (nil = skip)
			kCache, vCache                       metal.MTLBuffer
		}
		lb := make([]layerBufs, nLayers)
		cacheBytes := uint(maxLen * kvDim * bf16Size)
		residentOrNil := func(b []byte) metal.MTLBuffer {
			if len(b) == 0 {
				return nil
			}
			return residentBytes(b)
		}
		for li := range layers {
			w := layers[li]
			lb[li] = layerBufs{
				anw: residentBytes(w.AttnNormW), wq: residentBytes(w.WQ), wk: residentBytes(w.WK),
				wv: residentBytes(w.WV), wo: residentBytes(w.WO), mnw: residentBytes(w.MLPNormW),
				wg: residentBytes(w.WGate), wu: residentBytes(w.WUp), wd: residentBytes(w.WDown),
				pan: residentOrNil(w.PostAttnNormW), pfn: residentOrNil(w.PostFFNormW),
				qn: residentOrNil(w.QNormW), kn: residentOrNil(w.KNormW),
				kCache: device.NewBufferWithLengthOptions(cacheBytes, metal.MTLResourceStorageModeShared),
				vCache: device.NewBufferWithLengthOptions(cacheBytes, metal.MTLResourceStorageModeShared),
			}
		}

		// one bf16 projector per layer (holds that layer's 7 weight buffers); the
		// half-encoders project through it, so a quantised forward differs only in
		// building qmvProjectors here.
		projs := make([]bf16Projector, nLayers)
		for li := range lb {
			l := lb[li]
			projs[li] = bf16Projector{
				wQ: bufView{buf: l.wq}, wK: bufView{buf: l.wk}, wV: bufView{buf: l.wv}, wO: bufView{buf: l.wo},
				wGate: bufView{buf: l.wg}, wUp: bufView{buf: l.wu}, wDown: bufView{buf: l.wd},
				dModel: dModel, qDim: qDim, kvDim: kvDim, dFF: dFF,
			}
		}

		// shared scratch (reused across every layer and token; serial dispatch +
		// per-token commit make reuse safe) and the residual-stream ping-pong.
		asc := newAttnScratch(dModel, qDim, kvDim, nHeads, 0)
		msc := newMLPScratch(dModel, dFF)
		sc := newDecodeForwardStepScratch(dModel)

		for t := 0; t < T; t++ {
			sc.seed(t, inputs[t])

			cb := queue.CommandBuffer()
			enc := cb.ComputeCommandEncoder()
			in, out := sc.xA, sc.xB
			for li := 0; li < nLayers; li++ {
				l := lb[li]
				if encErr = encAttnHalfKV(enc, in, l.kCache, l.vCache, sc.offBuf, sc.hBuf, bufView{buf: l.anw}, bufView{buf: l.pan}, bufView{buf: l.qn}, bufView{buf: l.kn}, nil, asc, projs[li], dModel, nHeads, nKVHeads, headDim, t, 0, headDim, base, scale, eps, nil); encErr != nil {
					enc.EndEncoding()
					return
				}
				if encErr = encMLPHalfBF16(enc, sc.hBuf, out, bufView{buf: l.mnw}, bufView{buf: l.pfn}, msc, projs[li], dModel, dFF, eps); encErr != nil {
					enc.EndEncoding()
					return
				}
				in, out = out, in // next layer reads this layer's output
			}
			enc.EndEncoding()
			cb.Commit()
			cb.WaitUntilCompleted()
			sc.copyBuffer(outputs[t], in) // `in` holds the last layer's output after the final swap
		}
	})
	return outputs, encErr
}
