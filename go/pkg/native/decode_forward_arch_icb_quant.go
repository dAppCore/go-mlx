// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	core "dappco.re/go"
	g4 "dappco.re/go/mlx/pkg/model/gemma4"
	"github.com/tmc/apple/metal"
)

// DecodeForwardArchICBQuant is the arch-driven decode with BOTH fast-path levers
// stacked: 4-bit qmv weights (cut the GPU read) AND the ICB encode-bypass replay (cut
// the per-token host re-encode), DRIVEN by the declared arch (KV-share + sliding). It
// is DecodeForwardArchICB with a qmv `recordProj` (affine_qmv_bfloat16_t) instead of
// gemv, running the same arch-aware decodeForwardArchICBCore — the V projection binds at
// index 4 (qmv) not 3 (gemv), so vOutBind=4. Byte-for-byte equal to DecodeForwardArchQuant
// on the same arch (gated). MoE layers are rejected (the router's host top-k can't sit in
// a recorded/replayed command buffer). All raw bf16 activations.
// recordArchICBQuant records the 4-bit arch ICB and returns the held *archICBReplay — the
// recorder shared by the batch DecodeForwardArchICBQuant (record + runBatch) and the
// Gemma4Session (record once at open, stepBody per token). Caches + the PLE runtime are
// parameters: the batch passes fresh caches + a batch-token-id runtime; the session passes its
// own lb caches (so prefill's KV is visible) + {nil, s.perLayerInput}. pleRuntime nil ⇒ no PLE;
// pleGS/pleBits are the PLE gate/proj quant geometry for quantPLELayers.
func recordArchICBQuant(
	qlayers []QuantizedLayerWeights, specs []g4.LayerSpec,
	kCaches, vCaches []metal.MTLBuffer,
	pleRuntime *archDecodePLEInputs, pliDim, pleGS, pleBits int,
	dModel, nHeads, nKVHeads, headDim, maxLen, dFF, slidingWindow int,
	rope icbRope, scale, eps float32, valueNorm bool,
) (*archICBReplay, error) {
	nLayers := len(qlayers)
	qDim, kvDim := nHeads*headDim, nKVHeads*headDim
	_ = kvDim
	gs, bits := qlayers[0].GroupSize, qlayers[0].Bits
	if gs == 0 || bits == 0 {
		return nil, core.NewError("native.recordArchICBQuant: GroupSize/Bits unset")
	}
	for li := range specs {
		o := specs[li].KVShareFrom
		if o < 0 || o > li || (o != li && !specs[o].OwnsCache()) {
			return nil, core.NewError("native.DecodeForwardArchICBQuant: KVShareFrom must reference an earlier owner layer")
		}
		if specs[li].MoE {
			return nil, core.NewError("native.DecodeForwardArchICBQuant: MoE layers are not supported on the ICB path")
		}
	}
	// per-layer FFN width (gemma4 E2B/E4B MatFormer): lFF[li] (from ql.DFF, fallback dFF) —
	// drives the Gate/Up/Down size validation, the per-width PSO/scalar keying, and the core.
	lFF := make([]int, nLayers)
	for li := range qlayers {
		lFF[li] = dFF
		if qlayers[li].DFF > 0 {
			lFF[li] = qlayers[li].DFF
		}
	}
	type pj struct {
		w           QuantWeight
		outDim, inD int
	}
	for li := range qlayers {
		ql := qlayers[li]
		if ql.GroupSize != gs || ql.Bits != bits {
			return nil, core.NewError("native.DecodeForwardArchICBQuant: layers must share GroupSize/Bits")
		}
		if len(ql.AttnNormW) != dModel*bf16Size || len(ql.MLPNormW) != dModel*bf16Size {
			return nil, core.NewError("native.DecodeForwardArchICBQuant: norm weight size mismatch")
		}
		lff := lFF[li]
		lhd := headDimOf(specs[li], headDim) // per-layer head dim (gemma4 full_attention > sliding)
		lqDim, lkvDim := nHeads*lhd, nKVHeads*lhd
		projChecks := []pj{
			{ql.Q, lqDim, dModel}, {ql.K, lkvDim, dModel}, {ql.O, dModel, lqDim},
			{ql.Gate, lff, dModel}, {ql.Up, lff, dModel}, {ql.Down, dModel, lff},
		}
		if len(ql.V.Packed) > 0 { // gemma4 K==V layers carry no v_proj — V rides the k-proj output
			projChecks = append(projChecks, pj{ql.V, lkvDim, dModel})
		}
		for _, p := range projChecks {
			if p.inD%gs != 0 {
				return nil, core.NewError("native.DecodeForwardArchICBQuant: inDim not a multiple of GroupSize")
			}
			if len(p.w.Packed) != p.outDim*p.inD*bits/8 ||
				len(p.w.Scales) != p.outDim*(p.inD/gs)*bf16Size || len(p.w.Biases) != p.outDim*(p.inD/gs)*bf16Size {
				return nil, core.NewError("native.DecodeForwardArchICBQuant: quantised weight size mismatch")
			}
		}
	}
	var pleLayers []pleLayer
	var err error
	if pleRuntime != nil {
		pleLayers, err = quantPLELayers("native.recordArchICBQuant", qlayers, dModel, pliDim, pleGS, pleBits)
		if err != nil {
			return nil, err
		}
	}

	// qmv ICB pipelines, one per distinct (outDim,inDim) shape (built before the pool).
	qmvPSO := func(outDim, inDim int) (metal.MTLComputePipelineState, error) {
		variant := "_qmv_"
		if outDim%8 == 0 && inDim%512 == 0 {
			variant = "_qmv_fast_"
		}
		return pipelineForICB(core.Sprintf("affine%sbfloat16_t_gs_%d_b_%d_batch_0", variant, gs, bits))
	}
	psoQ, err := qmvPSO(qDim, dModel)
	if err != nil {
		return nil, err
	}
	psoKV, err := qmvPSO(kvDim, dModel)
	if err != nil {
		return nil, err
	}
	psoO, err := qmvPSO(dModel, qDim)
	if err != nil {
		return nil, err
	}
	// gate/up (lff←dModel) and down (dModel←lff) qmv PSOs, one per distinct FFN width.
	psoFByDFF := make(map[int]metal.MTLComputePipelineState)
	psoDByDFF := make(map[int]metal.MTLComputePipelineState)
	for li := range lFF {
		lff := lFF[li]
		if _, ok := psoFByDFF[lff]; !ok {
			pf, e := qmvPSO(lff, dModel)
			if e != nil {
				return nil, e
			}
			pd, e2 := qmvPSO(dModel, lff)
			if e2 != nil {
				return nil, e2
			}
			psoFByDFF[lff], psoDByDFF[lff] = pf, pd
		}
	}
	var psoPLEGate, psoPLEProj metal.MTLComputePipelineState
	if pleRuntime != nil {
		if psoPLEGate, err = qmvPSO(pliDim, dModel); err != nil {
			return nil, err
		}
		if psoPLEProj, err = qmvPSO(dModel, pliDim); err != nil {
			return nil, err
		}
	}

	var r *archICBReplay
	var coreErr error
	withAutoreleasePool(func() {
		anwBufs := make([]metal.MTLBuffer, nLayers)
		mnwBufs := make([]metal.MTLBuffer, nLayers)
		qNormBufs := make([]metal.MTLBuffer, nLayers)
		kNormBufs := make([]metal.MTLBuffer, nLayers)
		postAttnBufs := make([]metal.MTLBuffer, nLayers)
		postFFBufs := make([]metal.MTLBuffer, nLayers)
		layerScalarBufs := make([]metal.MTLBuffer, nLayers)
		type lw struct{ q, k, v, o, g, u, d qmvWeight }
		lb := make([]lw, nLayers)
		type plw struct{ gate, proj qmvWeight }
		pleLB := make([]plw, nLayers)
		plePostNorms := make([]metal.MTLBuffer, nLayers)
		mkW := func(w QuantWeight) qmvWeight {
			if len(w.Packed) == 0 { // absent projection (gemma4 K==V: no v_proj) ⇒ nil weight, hasV()==false
				return qmvWeight{}
			}
			return qmvWeight{wq: copyView(w.Packed), scales: copyView(w.Scales), biases: copyView(w.Biases)}
		}
		// presized to the upper bound (every layer's 7 projections × wq/scales/biases, the 5 shared
		// trailing scalar buffers, plus ≤2 FFN dim scalars per distinct dFF width) so the per-forward
		// build never geometrically regrows its backing array — K==V layers simply leave the v-proj
		// slot unused. Byte-identical.
		projResident := make([]metal.MTLBuffer, 0, nLayers*7*3+5+nLayers*2)
		for li := range qlayers {
			ql := qlayers[li]
			anwBufs[li] = sharedBytes(ql.AttnNormW)
			mnwBufs[li] = sharedBytes(ql.MLPNormW)
			qNormBufs[li] = sharedOrNil(ql.QNormW)
			kNormBufs[li] = sharedOrNil(ql.KNormW)
			postAttnBufs[li] = sharedOrNil(ql.PostAttnNormW)
			postFFBufs[li] = sharedOrNil(ql.PostFFNormW)
			layerScalarBufs[li] = layerScalarBuf(ql.LayerScalarW, dModel)
			lb[li] = lw{mkW(ql.Q), mkW(ql.K), mkW(ql.V), mkW(ql.O), mkW(ql.Gate), mkW(ql.Up), mkW(ql.Down)}
			for _, w := range []qmvWeight{lb[li].q, lb[li].k, lb[li].v, lb[li].o, lb[li].g, lb[li].u, lb[li].d} {
				if w.wq.buf == nil { // K==V: no v_proj weight to make resident
					continue
				}
				projResident = append(projResident, w.wq.buf, w.scales.buf, w.biases.buf)
			}
			if pleRuntime != nil {
				pleLB[li] = plw{mkW(pleLayers[li].gate), mkW(pleLayers[li].proj)}
				plePostNorms[li] = sharedBytes(pleLayers[li].postNorm)
			}
		}
		kDModel, nDModel := scalarI32(int32(dModel)), scalarI32(int32(dModel))
		// per-hd qmv dim scalars (gemma4 global vs sliding head dim): nQDim = qDim out (projQ),
		// nKvDim = kvDim out (projK/V), kQDim = qDim in (projO). One set per distinct hd.
		nQDimByHd := make(map[int]metal.MTLBuffer)
		nKvDimByHd := make(map[int]metal.MTLBuffer)
		kQDimByHd := make(map[int]metal.MTLBuffer)
		for li := range specs {
			hd := headDimOf(specs[li], headDim)
			if _, ok := nQDimByHd[hd]; !ok {
				nQDimByHd[hd] = scalarI32(int32(nHeads * hd))
				nKvDimByHd[hd] = scalarI32(int32(nKVHeads * hd))
				kQDimByHd[hd] = scalarI32(int32(nHeads * hd))
			}
		}
		// per-distinct-dFF qmv dim scalars: kDFF (down's K=inDim=lff) and nDFF (gate/up's N=outDim=lff).
		kDFFByW := make(map[int]metal.MTLBuffer)
		nDFFByW := make(map[int]metal.MTLBuffer)
		for li := range lFF {
			lff := lFF[li]
			if _, ok := kDFFByW[lff]; !ok {
				kDFFByW[lff] = scalarI32(int32(lff))
				nDFFByW[lff] = scalarI32(int32(lff))
			}
		}
		projResident = append(projResident, kDModel, nDModel)
		for hd, b := range nQDimByHd {
			projResident = append(projResident, b, nKvDimByHd[hd], kQDimByHd[hd])
		}
		for lff, b := range kDFFByW {
			projResident = append(projResident, b, nDFFByW[lff])
		}

		setQMV := func(c metal.MTLIndirectComputeCommand, pso metal.MTLComputePipelineState, w qmvWeight, vec, out metal.MTLBuffer, outOff uint, kB, nB metal.MTLBuffer, outDim int) {
			c.SetComputePipelineState(pso)
			c.SetKernelBufferOffsetAtIndex(w.wq.buf, w.wq.off, 0)
			c.SetKernelBufferOffsetAtIndex(w.scales.buf, w.scales.off, 1)
			c.SetKernelBufferOffsetAtIndex(w.biases.buf, w.biases.off, 2)
			c.SetKernelBufferOffsetAtIndex(vec, 0, 3)
			c.SetKernelBufferOffsetAtIndex(out, outOff, 4)
			c.SetKernelBufferOffsetAtIndex(kB, 0, 5)
			c.SetKernelBufferOffsetAtIndex(nB, 0, 6)
			const bn, bk = 8, 32
			nTgp := (outDim + bn - 1) / bn
			c.ConcurrentDispatchThreadgroupsThreadsPerThreadgroup(metal.MTLSize{Width: 1, Height: uint(nTgp), Depth: 1}, metal.MTLSize{Width: bk, Height: 2, Depth: 1})
		}
		var plePlan *archICBPLEPlan
		if pleRuntime != nil {
			kPLIDim, nPLIDim := scalarI32(int32(pliDim)), scalarI32(int32(pliDim))
			pleResident := []metal.MTLBuffer{kPLIDim, nPLIDim}
			for li := range pleLB {
				for _, w := range []qmvWeight{pleLB[li].gate, pleLB[li].proj} {
					pleResident = append(pleResident, w.wq.buf, w.scales.buf, w.biases.buf)
				}
			}
			plePlan = &archICBPLEPlan{
				runtime: pleRuntime, pliDim: pliDim, postNormBufs: plePostNorms, resident: pleResident,
			}
			plePlan.recordGate = func(li int, c metal.MTLIndirectComputeCommand, vec, out metal.MTLBuffer) {
				setQMV(c, psoPLEGate, pleLB[li].gate, vec, out, 0, kDModel, nPLIDim, pliDim)
			}
			plePlan.recordProj = func(li int, c metal.MTLIndirectComputeCommand, vec, out metal.MTLBuffer) {
				setQMV(c, psoPLEProj, pleLB[li].proj, vec, out, 0, kPLIDim, nDModel, dModel)
			}
		}
		recordProj := func(li int, c metal.MTLIndirectComputeCommand, vec, out metal.MTLBuffer, outOff uint, p projIndex) {
			l := lb[li]
			hd := headDimOf(specs[li], headDim)
			switch p {
			case projQ:
				setQMV(c, psoQ, l.q, vec, out, outOff, kDModel, nQDimByHd[hd], nHeads*hd)
			case projK:
				setQMV(c, psoKV, l.k, vec, out, outOff, kDModel, nKvDimByHd[hd], nKVHeads*hd)
			case projV:
				setQMV(c, psoKV, l.v, vec, out, outOff, kDModel, nKvDimByHd[hd], nKVHeads*hd)
			case projO:
				setQMV(c, psoO, l.o, vec, out, outOff, kQDimByHd[hd], nDModel, dModel)
			case projGate:
				lff := lFF[li]
				setQMV(c, psoFByDFF[lff], l.g, vec, out, outOff, kDModel, nDFFByW[lff], lff)
			case projUp:
				lff := lFF[li]
				setQMV(c, psoFByDFF[lff], l.u, vec, out, outOff, kDModel, nDFFByW[lff], lff)
			case projDown:
				lff := lFF[li]
				setQMV(c, psoDByDFF[lff], l.d, vec, out, outOff, kDFFByW[lff], nDModel, dModel)
			}
		}
		valueNormOnes := valueNormOnesBuf(valueNorm, headDim)
		vProjIdx := projV
		if len(qlayers[0].V.Packed) == 0 { // gemma4 K==V: V rides the k-proj
			vProjIdx = projK
		}
		r, coreErr = recordArchICB(specs, anwBufs, mnwBufs, kCaches, vCaches, projResident, qNormBufs, kNormBufs, postAttnBufs, postFFBufs, layerScalarBufs, plePlan, recordProj, 4, valueNormOnes, vProjIdx, dModel, nHeads, nKVHeads, headDim, dFF, maxLen, slidingWindow, lFF, rope, scale, eps)
	})
	if coreErr != nil {
		return nil, coreErr
	}
	return r, nil
}

// DecodeForwardArchICBQuant is the batch 4-bit arch ICB: record the stack once + replay it
// across the whole input sequence (the encode-bypass). It is recordArchICBQuant + runBatch,
// byte-identical to the pre-split entry. MoE layers are rejected. All bf16 activations.
func DecodeForwardArchICBQuant(
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
		return nil, core.NewError("native.DecodeForwardArchICBQuant: need layers and inputs")
	}
	if len(specs) != nLayers {
		return nil, core.NewError("native.DecodeForwardArchICBQuant: specs length must equal layers")
	}
	if T > maxLen {
		return nil, core.NewError("native.DecodeForwardArchICBQuant: more tokens than maxLen cache rows")
	}
	for i := range inputs {
		if len(inputs[i]) != dModel*bf16Size {
			return nil, core.NewError("native.DecodeForwardArchICBQuant: each input must be dModel bf16 bytes")
		}
	}
	plePayload, err := singleArchPLEQuant("native.DecodeForwardArchICBQuant", pleArgs)
	if err != nil {
		return nil, err
	}
	pleRuntime, pliDim, err := archPLEQuantRuntime("native.DecodeForwardArchICBQuant", plePayload, nLayers, T, dModel, eps)
	if err != nil {
		return nil, err
	}
	pleGS, pleBits := 0, 0
	if plePayload != nil {
		pleGS, pleBits = plePayload.GroupSize, plePayload.Bits
	}
	kCaches := make([]metal.MTLBuffer, nLayers)
	vCaches := make([]metal.MTLBuffer, nLayers)
	var r *archICBReplay
	var coreErr error
	withAutoreleasePool(func() {
		for li := range specs {
			if specs[li].OwnsCache() { // per-layer linear cache — global layers' rows are wider (larger head_dim)
				cacheBytes := uint(maxLen * nKVHeads * headDimOf(specs[li], headDim) * bf16Size)
				kCaches[li] = device.NewBufferWithLengthOptions(cacheBytes, metal.MTLResourceStorageModeShared)
				vCaches[li] = device.NewBufferWithLengthOptions(cacheBytes, metal.MTLResourceStorageModeShared)
			}
		}
		r, coreErr = recordArchICBQuant(qlayers, specs, kCaches, vCaches, pleRuntime, pliDim, pleGS, pleBits, dModel, nHeads, nKVHeads, headDim, maxLen, dFF, slidingWindow, simpleICBRope(base, headDim), scale, eps, valueNorm)
	})
	if coreErr != nil {
		return nil, coreErr
	}
	return r.runBatch(inputs)
}
