// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

// QuantWeight is one projection's affine-quantised weight: MLX's packed codes + bf16 scales +
// bf16 biases (one scale/bias per group per row). GroupSize/Bits are the weight's OWN affine
// geometry — mixed-precision packs (e4b-qat: the MLP is 8-bit while attention is 4-bit) vary it
// per weight; 0 ⇒ fall back to the projector's layer-default groupSize/bits (uniform packs).
type QuantWeight struct {
	Packed, Scales, Biases []byte
	GroupSize, Bits        int
}

// QuantizedLayerWeights is one decode layer with 4-bit projections: the two
// RMSNorm weights stay bf16 (norms aren't quantised — tiny vectors), the seven
// matmuls are quantised. GroupSize ∈ {32,64,128}, Bits = 4 for the models we serve.
type QuantizedLayerWeights struct {
	AttnNormW, MLPNormW        []byte
	Q, K, V, O, Gate, Up, Down QuantWeight
	GroupSize, Bits            int
	// DFF is this layer's FFN width — gemma4 E2B/E4B (MatFormer) vary it per layer, so the decode
	// can't assume a single arch.FF. 0 ⇒ use the arch default (uniform models).
	DFF int
	// gemma4 norms (bf16, not quantised), applied when non-nil: PostAttnNormW /
	// PostFFNormW before their residual add; QNormW / KNormW per-head on Q/K before RoPE.
	PostAttnNormW, PostFFNormW []byte
	QNormW, KNormW             []byte
	// LayerScalarW is gemma4's per-layer output scalar (shape [1] bf16, not quantised); the
	// arch executor multiplies the layer's final hidden by it. nil when omitted.
	LayerScalarW []byte
	// per-layer-input gate (gemma4 E2B/E4B): the 4-bit gate (pliDim×dModel) + projection
	// (dModel×pliDim) and the bf16 post-norm (dModel). All nil for models without the PLE
	// tower (the dense 12B). Applied at the layer tail by PerLayerInputGateQuant.
	PerLayerGate, PerLayerProjection QuantWeight
	PostPerLayerInputNormW           []byte
	// MoE, when non-nil (gemma4 26B-A4B), replaces the dense MLP half with the 4-bit dual-branch
	// MoEBlockQuant for this layer; the dense MLPNormW/Gate/Up/Down are then unused.
	MoE *MoEQuantLayerWeights
}

// DecodeForwardQuant is DecodeForward with 4-bit-quantised projections: identical
// in every other respect (bf16 activations, growing seq-major KV cache, one
// commit+wait per token, residual stream layer→layer), because the only thing that
// changes is the projector — qmvProjector (affine_qmv_bfloat16_t) instead of
// bf16Projector. This is the whole 4-bit decode forward running with NO mlx-c at
// runtime. With the same logical weights it equals DecodeForward up to quantisation
// (gated against the parity-proven standalone ops in the tests). All raw bf16 I/O.
func DecodeForwardQuant(
	inputs [][]byte, qlayers []QuantizedLayerWeights,
	dModel, nHeads, nKVHeads, headDim, maxLen, dFF int,
	base, scale, eps float32,
) ([][]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	nLayers, T := len(qlayers), len(inputs)
	if nLayers == 0 || T == 0 {
		return nil, core.NewError("native.DecodeForwardQuant: need layers and inputs")
	}
	if T > maxLen {
		return nil, core.NewError("native.DecodeForwardQuant: more tokens than maxLen cache rows")
	}
	qDim, kvDim := nHeads*headDim, nKVHeads*headDim
	for i := range inputs {
		if len(inputs[i]) != dModel*bf16Size {
			return nil, core.NewError("native.DecodeForwardQuant: each input must be dModel bf16 bytes")
		}
	}
	// validate per-layer: norms bf16; each projection's packed/scales/biases sizes
	type pj struct {
		w           QuantWeight
		outDim, inD int
	}
	for li := range qlayers {
		ql := qlayers[li]
		if ql.GroupSize == 0 || ql.Bits == 0 {
			return nil, core.NewError("native.DecodeForwardQuant: GroupSize/Bits unset")
		}
		if len(ql.AttnNormW) != dModel*bf16Size || len(ql.MLPNormW) != dModel*bf16Size {
			return nil, core.NewError("native.DecodeForwardQuant: norm weight size mismatch")
		}
		for _, p := range []pj{
			{ql.Q, qDim, dModel}, {ql.K, kvDim, dModel}, {ql.V, kvDim, dModel}, {ql.O, dModel, qDim},
			{ql.Gate, dFF, dModel}, {ql.Up, dFF, dModel}, {ql.Down, dModel, dFF},
		} {
			if p.inD%ql.GroupSize != 0 {
				return nil, core.NewError("native.DecodeForwardQuant: inDim not a multiple of GroupSize")
			}
			wantPacked := p.outDim * p.inD * ql.Bits / 8
			wantSB := p.outDim * (p.inD / ql.GroupSize) * bf16Size
			if len(p.w.Packed) != wantPacked || len(p.w.Scales) != wantSB || len(p.w.Biases) != wantSB {
				return nil, core.NewError("native.DecodeForwardQuant: quantised weight size mismatch")
			}
		}
	}

	outputs := make([][]byte, T)
	for i := range outputs {
		outputs[i] = make([]byte, dModel*bf16Size)
	}
	var encErr error
	withAutoreleasePool(func() {
		// per-layer resident: bf16 norms + the quantised projector + growing caches
		type layerBufs struct{ anw, mnw, pan, pfn, qn, kn, kCache, vCache metal.MTLBuffer }
		lb := make([]layerBufs, nLayers)
		projs := make([]qmvProjector, nLayers)
		cacheBytes := uint(maxLen * kvDim * bf16Size)
		mkW := func(qw QuantWeight) qmvWeight {
			return qmvWeight{wq: copyView(qw.Packed), scales: copyView(qw.Scales), biases: copyView(qw.Biases)}
		}
		for li := range qlayers {
			ql := qlayers[li]
			lb[li] = layerBufs{
				anw: sharedBytes(ql.AttnNormW), mnw: sharedBytes(ql.MLPNormW),
				pan: sharedOrNil(ql.PostAttnNormW), pfn: sharedOrNil(ql.PostFFNormW),
				qn: sharedOrNil(ql.QNormW), kn: sharedOrNil(ql.KNormW),
				kCache: device.NewBufferWithLengthOptions(cacheBytes, metal.MTLResourceStorageModeShared),
				vCache: device.NewBufferWithLengthOptions(cacheBytes, metal.MTLResourceStorageModeShared),
			}
			projs[li] = qmvProjector{
				q: mkW(ql.Q), k: mkW(ql.K), v: mkW(ql.V), o: mkW(ql.O),
				gate: mkW(ql.Gate), up: mkW(ql.Up), down: mkW(ql.Down),
				dModel: dModel, qDim: qDim, kvDim: kvDim, dFF: dFF,
				groupSize: ql.GroupSize, bits: ql.Bits,
			}
		}

		asc := newAttnScratch(dModel, qDim, kvDim)
		msc := newMLPScratch(dModel, dFF)
		hBuf := scratchBF16(dModel)
		xA, xB := scratchBF16(dModel), scratchBF16(dModel)
		off := int32(0)
		offBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&off), 4, metal.MTLResourceStorageModeShared)

		for t := 0; t < T; t++ {
			*(*int32)(offBuf.Contents()) = int32(t)
			copy(unsafe.Slice((*byte)(xA.Contents()), dModel*bf16Size), inputs[t])

			cb := queue.CommandBuffer()
			enc := cb.ComputeCommandEncoder()
			in, out := xA, xB
			for li := 0; li < nLayers; li++ {
				l := lb[li]
				if encErr = encAttnHalfKV(enc, in, l.kCache, l.vCache, offBuf, hBuf, bufView{buf: l.anw}, bufView{buf: l.pan}, bufView{buf: l.qn}, bufView{buf: l.kn}, nil, asc, projs[li], dModel, nHeads, nKVHeads, headDim, t, 0, headDim, base, scale, eps, nil); encErr != nil {
					enc.EndEncoding()
					return
				}
				if encErr = encMLPHalfBF16(enc, hBuf, out, bufView{buf: l.mnw}, bufView{buf: l.pfn}, msc, projs[li], dModel, dFF, eps); encErr != nil {
					enc.EndEncoding()
					return
				}
				in, out = out, in
			}
			enc.EndEncoding()
			cb.Commit()
			cb.WaitUntilCompleted()
			copy(outputs[t], unsafe.Slice((*byte)(in.Contents()), dModel*bf16Size))
		}
	})
	return outputs, encErr
}
