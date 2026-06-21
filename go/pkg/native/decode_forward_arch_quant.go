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
	base, scale, eps float32, valueNorm bool,
	pleArgs ...ArchPLEQuant,
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
		// per-layer FFN width (gemma4 E2B/E4B MatFormer varies it): validate Gate/Up/Down against
		// THIS layer's lff, not the uniform dFF — buildQuantArchLayerBufs already runs the decode at
		// ql.DFF, so a uniform-dFF check would reject the heterogeneous layer it can correctly execute.
		// lff==dFF for uniform callers ⇒ byte-identical validation.
		lff := dFF
		if ql.DFF > 0 {
			lff = ql.DFF
		}
		projChecks := []pj{
			{ql.Q, qDim, dModel}, {ql.K, kvDim, dModel}, {ql.O, dModel, qDim},
			{ql.Gate, lff, dModel}, {ql.Up, lff, dModel}, {ql.Down, dModel, lff},
		}
		if len(ql.V.Packed) > 0 { // gemma4 K==V layers carry no v_proj — V rides the k-proj output
			projChecks = append(projChecks, pj{ql.V, kvDim, dModel})
		}
		for _, p := range projChecks {
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
	plePayload, err := singleArchPLEQuant("native.DecodeForwardArchQuant", pleArgs)
	if err != nil {
		return nil, err
	}
	pleRuntime, pliDim, err := archPLEQuantRuntime("native.DecodeForwardArchQuant", plePayload, nLayers, T, dModel, eps)
	if err != nil {
		return nil, err
	}
	var pleLayers []pleLayer
	if pleRuntime != nil {
		pleLayers, err = quantPLELayers("native.DecodeForwardArchQuant", qlayers, dModel, pliDim, plePayload.GroupSize, plePayload.Bits)
		if err != nil {
			return nil, err
		}
	}

	var outputs [][]byte
	withAutoreleasePool(func() {
		lb, _, berr := buildQuantArchLayerBufs(qlayers, specs, dModel, nHeads, nKVHeads, headDim, dFF, maxLen, slidingWindow, nil) // whole-seq forward rejects MoE
		if berr != nil {
			err = berr
			return
		}
		moeWeights := make([]*MoELayerWeights, nLayers) // all nil — DecodeForwardArchQuant is non-MoE
		if pleRuntime != nil {
			state := newArchDecodeState(specs, lb, moeWeights, dModel, nHeads, nKVHeads, headDim, dFF, slidingWindow, headDim, headDim, base, base, scale, eps, valueNorm)
			state.ple, state.pliDim = pleLayers, pliDim
			outputs, err = runArchDecodeState(inputs, &state, pleRuntime)
			return
		}
		outputs, err = runArchDecode(inputs, specs, lb, moeWeights, dModel, nHeads, nKVHeads, headDim, dFF, slidingWindow, headDim, headDim, base, base, scale, eps, valueNorm)
	})
	return outputs, err
}

// buildQuantArchLayerBufs builds the per-layer archLayerBufs for the 4-bit path: bf16 norm
// buffers (the norms aren't quantised), owner-layer KV caches, and a qmvProjector per layer —
// the only difference from buildBF16ArchLayerBufs. Shared by DecodeForwardArchQuant and
// NewArchQuantSession. sb is the zero-copy weight source (see buildBF16ArchLayerBufs): non-nil
// binds every weight (norms + the quant triples) as no-copy shard views; nil uploads owned copies.
// MUST be called inside a withAutoreleasePool.
func buildQuantArchLayerBufs(qlayers []QuantizedLayerWeights, specs []g4.LayerSpec, dModel, nHeads, nKVHeads, headDim, dFF, maxLen, slidingWindow int, sb *shardBuffers) ([]archLayerBufs, []*MoEQuantLayerWeights, error) {
	lb := make([]archLayerBufs, len(qlayers))
	moeQuant := make([]*MoEQuantLayerWeights, len(qlayers))
	var ferr error
	view := func(b []byte) bufView {
		if sb != nil {
			return sb.mustBufFor(b, &ferr)
		}
		return copyView(b)
	}
	view4 := func(b []byte) bufView { // 4-bit packed uint32 weights need 4-byte alignment (affine_qmv reads uint32)
		if sb != nil {
			return sb.mustBufFor4(b, &ferr)
		}
		return copyView(b)
	}
	viewOrNil := func(b []byte) bufView {
		if len(b) == 0 {
			return bufView{}
		}
		return view(b)
	}
	// mkW resolves one 4-bit triple to bufViews (no-copy shard views or copies); an absent
	// projection (gemma4 K==V: no v_proj) ⇒ the zero qmvWeight, hasV()==false.
	mkW := func(qw QuantWeight) qmvWeight {
		if len(qw.Packed) == 0 {
			return qmvWeight{}
		}
		return qmvWeight{wq: view4(qw.Packed), scales: view(qw.Scales), biases: view(qw.Biases), gs: qw.GroupSize, bits: qw.Bits}
	}
	for li := range qlayers {
		ql := qlayers[li]
		// per-attention-type geometry: full layers use the larger global head_dim.
		lhd, lkv := headDimOf(specs[li], headDim), kvHeadsOf(specs[li], nKVHeads)
		qDim, kvDim := nHeads*lhd, lkv*lhd
		// sliding layers RING at slidingWindow rows (the full-context KV memory fix) — see the bf16
		// build for the rationale; global (full_attention) layers keep maxLen.
		cacheLen := maxLen
		if slidingWindow > 0 && slidingWindow < maxLen && specs[li].Attention != g4.GlobalAttention {
			cacheLen = slidingWindow
		}
		cacheBytes := uint(cacheLen * kvDim * bf16Size)
		lb[li].anw = view(ql.AttnNormW)
		lb[li].postAttnNorm = viewOrNil(ql.PostAttnNormW)
		lb[li].postFFNorm = viewOrNil(ql.PostFFNormW)
		lb[li].qNorm = viewOrNil(ql.QNormW)
		lb[li].kNorm = viewOrNil(ql.KNormW)
		lb[li].layerScalar = layerScalarBuf(ql.LayerScalarW, dModel) // synthesised broadcast (not a shard view)
		if specs[li].OwnsCache() {
			lb[li].kCache = device.NewBufferWithLengthOptions(cacheBytes, metal.MTLResourceStorageModeShared)
			lb[li].vCache = device.NewBufferWithLengthOptions(cacheBytes, metal.MTLResourceStorageModeShared)
		}
		lFF := dFF // per-layer FFN width (gemma4 E2B/E4B vary it); 0 ⇒ arch default
		if ql.DFF > 0 {
			lFF = ql.DFF
		}
		lb[li].dFF = lFF
		proj := qmvProjector{
			q: mkW(ql.Q), k: mkW(ql.K), v: mkW(ql.V), o: mkW(ql.O),
			dModel: dModel, qDim: qDim, kvDim: kvDim, dFF: lFF,
			groupSize: ql.GroupSize, bits: ql.Bits,
		}
		// MoE layers run MoEBlockQuant (host-orchestrated) instead of the dense MLP, so the
		// projector binds only attention; the dense MLP weights/norm are unused (and nil).
		if ql.MoE != nil {
			moeQuant[li] = ql.MoE
		} else {
			lb[li].mnw = view(ql.MLPNormW)
			proj.gate, proj.up, proj.down = mkW(ql.Gate), mkW(ql.Up), mkW(ql.Down)
		}
		lb[li].proj = proj
	}
	return lb, moeQuant, ferr
}
