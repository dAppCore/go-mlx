// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	core "dappco.re/go"
	g4 "dappco.re/go/mlx/pkg/model/gemma4"
	"github.com/tmc/apple/metal"
)

// DecodeForwardArchQuant is the 4-bit arch-driven decode forward: DecodeForwardArch
// with quantised projections. It runs the SAME arch-driven loop (runArchDecode) over
// the SAME cache-topology + sliding-window the bf16 path does — the projector seam is
// the only difference (qmvProjector / affine_qmv_bfloat16_t instead of bf16Projector),
// so KV-sharing and sliding layers get 4-bit weights for free. With an all-owner,
// all-global arch it equals DecodeForwardQuant byte-for-byte (gated). The norms stay
// bf16 (not quantised). MoE layers are NOT supported yet — quantised experts are a
// deeper slice — so a spec.MoE layer is rejected. All raw bf16 activations.
func DecodeForwardArchQuant(
	inputs [][]byte, qlayers []QuantizedLayerWeights, specs []g4.LayerSpec,
	dModel, nHeads, nKVHeads, headDim, maxLen, dFF, slidingWindow int,
	base, scale, eps float32,
) ([][]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	nLayers, T := len(qlayers), len(inputs)
	if nLayers == 0 || T == 0 {
		return nil, core.NewError("native.DecodeForwardArchQuant: need layers and inputs")
	}
	if len(specs) != nLayers {
		return nil, core.NewError("native.DecodeForwardArchQuant: specs length must equal layers")
	}
	if T > maxLen {
		return nil, core.NewError("native.DecodeForwardArchQuant: more tokens than maxLen cache rows")
	}
	qDim, kvDim := nHeads*headDim, nKVHeads*headDim
	for i := range inputs {
		if len(inputs[i]) != dModel*bf16Size {
			return nil, core.NewError("native.DecodeForwardArchQuant: each input must be dModel bf16 bytes")
		}
	}
	for li := range specs {
		o := specs[li].KVShareFrom
		if o < 0 || o > li || (o != li && !specs[o].OwnsCache()) {
			return nil, core.NewError("native.DecodeForwardArchQuant: KVShareFrom must reference an earlier owner layer")
		}
		if specs[li].MoE {
			return nil, core.NewError("native.DecodeForwardArchQuant: MoE layers are not supported on the quant path yet")
		}
	}
	// validate each layer's quant weight shapes (norms bf16; the seven projections).
	type pj struct {
		w           QuantWeight
		outDim, inD int
	}
	for li := range qlayers {
		ql := qlayers[li]
		if ql.GroupSize == 0 || ql.Bits == 0 {
			return nil, core.NewError("native.DecodeForwardArchQuant: GroupSize/Bits unset")
		}
		if len(ql.AttnNormW) != dModel*bf16Size || len(ql.MLPNormW) != dModel*bf16Size {
			return nil, core.NewError("native.DecodeForwardArchQuant: norm weight size mismatch")
		}
		for _, p := range []pj{
			{ql.Q, qDim, dModel}, {ql.K, kvDim, dModel}, {ql.V, kvDim, dModel}, {ql.O, dModel, qDim},
			{ql.Gate, dFF, dModel}, {ql.Up, dFF, dModel}, {ql.Down, dModel, dFF},
		} {
			if p.inD%ql.GroupSize != 0 {
				return nil, core.NewError("native.DecodeForwardArchQuant: inDim not a multiple of GroupSize")
			}
			wantPacked := p.outDim * p.inD * ql.Bits / 8
			wantSB := p.outDim * (p.inD / ql.GroupSize) * bf16Size
			if len(p.w.Packed) != wantPacked || len(p.w.Scales) != wantSB || len(p.w.Biases) != wantSB {
				return nil, core.NewError("native.DecodeForwardArchQuant: quantised weight size mismatch")
			}
		}
	}

	var outputs [][]byte
	var err error
	withAutoreleasePool(func() {
		lb, _ := buildQuantArchLayerBufs(qlayers, specs, dModel, nHeads, nKVHeads, headDim, dFF, maxLen) // whole-seq forward rejects MoE
		moeWeights := make([]*MoELayerWeights, nLayers)                                                  // all nil — DecodeForwardArchQuant is non-MoE
		outputs, err = runArchDecode(inputs, specs, lb, moeWeights, dModel, nHeads, nKVHeads, headDim, dFF, slidingWindow, headDim, headDim, base, base, scale, eps)
	})
	return outputs, err
}

// buildQuantArchLayerBufs builds the per-layer archLayerBufs for the 4-bit path: bf16 norm
// buffers (the norms aren't quantised), owner-layer KV caches, and a qmvProjector per layer —
// the only difference from buildBF16ArchLayerBufs. Shared by DecodeForwardArchQuant and
// NewGemma4QuantSession. MUST be called inside a withAutoreleasePool.
func buildQuantArchLayerBufs(qlayers []QuantizedLayerWeights, specs []g4.LayerSpec, dModel, nHeads, nKVHeads, headDim, dFF, maxLen int) ([]archLayerBufs, []*MoEQuantLayerWeights) {
	lb := make([]archLayerBufs, len(qlayers))
	moeQuant := make([]*MoEQuantLayerWeights, len(qlayers))
	mkW := func(qw QuantWeight) qmvWeight {
		return qmvWeight{sharedBytes(qw.Packed), sharedBytes(qw.Scales), sharedBytes(qw.Biases)}
	}
	for li := range qlayers {
		ql := qlayers[li]
		// per-attention-type geometry: full layers use the larger global head_dim.
		lhd, lkv := headDimOf(specs[li], headDim), kvHeadsOf(specs[li], nKVHeads)
		qDim, kvDim := nHeads*lhd, lkv*lhd
		cacheBytes := uint(maxLen * kvDim * bf16Size)
		lb[li].anw = sharedBytes(ql.AttnNormW)
		lb[li].postAttnNorm = sharedOrNil(ql.PostAttnNormW)
		lb[li].postFFNorm = sharedOrNil(ql.PostFFNormW)
		lb[li].qNorm = sharedOrNil(ql.QNormW)
		lb[li].kNorm = sharedOrNil(ql.KNormW)
		lb[li].layerScalar = layerScalarBuf(ql.LayerScalarW, dModel)
		if specs[li].OwnsCache() {
			lb[li].kCache = device.NewBufferWithLengthOptions(cacheBytes, metal.MTLResourceStorageModeShared)
			lb[li].vCache = device.NewBufferWithLengthOptions(cacheBytes, metal.MTLResourceStorageModeShared)
		}
		proj := qmvProjector{
			q: mkW(ql.Q), k: mkW(ql.K), v: mkW(ql.V), o: mkW(ql.O),
			dModel: dModel, qDim: qDim, kvDim: kvDim, dFF: dFF,
			groupSize: ql.GroupSize, bits: ql.Bits,
		}
		// MoE layers run MoEBlockQuant (host-orchestrated) instead of the dense MLP, so the
		// projector binds only attention; the dense MLP weights/norm are unused (and nil).
		if ql.MoE != nil {
			moeQuant[li] = ql.MoE
		} else {
			lb[li].mnw = sharedBytes(ql.MLPNormW)
			proj.gate, proj.up, proj.down = mkW(ql.Gate), mkW(ql.Up), mkW(ql.Down)
		}
		lb[li].proj = proj
	}
	return lb, moeQuant
}
