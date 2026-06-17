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
			kCache, vCache                       metal.MTLBuffer
		}
		lb := make([]layerBufs, nLayers)
		cacheBytes := uint(maxLen * kvDim * bf16Size)
		for li := range layers {
			w := layers[li]
			lb[li] = layerBufs{
				anw: sharedBytes(w.AttnNormW), wq: sharedBytes(w.WQ), wk: sharedBytes(w.WK),
				wv: sharedBytes(w.WV), wo: sharedBytes(w.WO), mnw: sharedBytes(w.MLPNormW),
				wg: sharedBytes(w.WGate), wu: sharedBytes(w.WUp), wd: sharedBytes(w.WDown),
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
				wQ: l.wq, wK: l.wk, wV: l.wv, wO: l.wo, wGate: l.wg, wUp: l.wu, wDown: l.wd,
				dModel: dModel, qDim: qDim, kvDim: kvDim, dFF: dFF,
			}
		}

		// shared scratch (reused across every layer and token; serial dispatch +
		// per-token commit make reuse safe) and the residual-stream ping-pong.
		asc := newAttnScratch(dModel, qDim, kvDim)
		msc := newMLPScratch(dModel, dFF)
		hBuf := scratchBF16(dModel)
		xA, xB := scratchBF16(dModel), scratchBF16(dModel)
		off := int32(0)
		offBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&off), 4, metal.MTLResourceStorageModeShared)

		for t := 0; t < T; t++ {
			// position buffer for this token (safe to mutate: committed per token)
			*(*int32)(offBuf.Contents()) = int32(t)
			// seed the residual stream with this token's input
			copy(unsafe.Slice((*byte)(xA.Contents()), dModel*bf16Size), inputs[t])

			cb := queue.CommandBuffer()
			enc := cb.ComputeCommandEncoder()
			in, out := xA, xB
			for li := 0; li < nLayers; li++ {
				l := lb[li]
				if encErr = encAttnHalfKV(enc, in, l.anw, l.kCache, l.vCache, offBuf, hBuf, asc, projs[li], dModel, nHeads, nKVHeads, headDim, t, 0, base, scale, eps); encErr != nil {
					enc.EndEncoding()
					return
				}
				if encErr = encMLPHalfBF16(enc, hBuf, l.mnw, out, msc, projs[li], dModel, dFF, eps); encErr != nil {
					enc.EndEncoding()
					return
				}
				in, out = out, in // next layer reads this layer's output
			}
			enc.EndEncoding()
			cb.Commit()
			cb.WaitUntilCompleted()
			copy(outputs[t], unsafe.Slice((*byte)(in.Contents()), dModel*bf16Size)) // `in` holds the last layer's output after the final swap
		}
	})
	return outputs, encErr
}
