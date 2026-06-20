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
func DecodeForwardArchICBQuant(
	inputs [][]byte, qlayers []QuantizedLayerWeights, specs []g4.LayerSpec,
	dModel, nHeads, nKVHeads, headDim, maxLen, dFF, slidingWindow int,
	base, scale, eps float32, valueNorm bool,
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
	qDim, kvDim := nHeads*headDim, nKVHeads*headDim
	gs, bits := qlayers[0].GroupSize, qlayers[0].Bits
	if gs == 0 || bits == 0 {
		return nil, core.NewError("native.DecodeForwardArchICBQuant: GroupSize/Bits unset")
	}
	for i := range inputs {
		if len(inputs[i]) != dModel*bf16Size {
			return nil, core.NewError("native.DecodeForwardArchICBQuant: each input must be dModel bf16 bytes")
		}
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
		projChecks := []pj{
			{ql.Q, qDim, dModel}, {ql.K, kvDim, dModel}, {ql.O, dModel, qDim},
			{ql.Gate, lff, dModel}, {ql.Up, lff, dModel}, {ql.Down, dModel, lff},
		}
		if len(ql.V.Packed) > 0 { // gemma4 K==V layers carry no v_proj — V rides the k-proj output
			projChecks = append(projChecks, pj{ql.V, kvDim, dModel})
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

	var outputs [][]byte
	var coreErr error
	withAutoreleasePool(func() {
		anwBufs := make([]metal.MTLBuffer, nLayers)
		mnwBufs := make([]metal.MTLBuffer, nLayers)
		qNormBufs := make([]metal.MTLBuffer, nLayers)
		kNormBufs := make([]metal.MTLBuffer, nLayers)
		postAttnBufs := make([]metal.MTLBuffer, nLayers)
		postFFBufs := make([]metal.MTLBuffer, nLayers)
		kCaches := make([]metal.MTLBuffer, nLayers)
		vCaches := make([]metal.MTLBuffer, nLayers)
		type lw struct{ q, k, v, o, g, u, d qmvWeight }
		lb := make([]lw, nLayers)
		cacheBytes := uint(maxLen * kvDim * bf16Size)
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
			if specs[li].OwnsCache() {
				kCaches[li] = device.NewBufferWithLengthOptions(cacheBytes, metal.MTLResourceStorageModeShared)
				vCaches[li] = device.NewBufferWithLengthOptions(cacheBytes, metal.MTLResourceStorageModeShared)
			}
			lb[li] = lw{mkW(ql.Q), mkW(ql.K), mkW(ql.V), mkW(ql.O), mkW(ql.Gate), mkW(ql.Up), mkW(ql.Down)}
			for _, w := range []qmvWeight{lb[li].q, lb[li].k, lb[li].v, lb[li].o, lb[li].g, lb[li].u, lb[li].d} {
				if w.wq.buf == nil { // K==V: no v_proj weight to make resident
					continue
				}
				projResident = append(projResident, w.wq.buf, w.scales.buf, w.biases.buf)
			}
		}
		kDModel, kQDim := scalarI32(int32(dModel)), scalarI32(int32(qDim))
		nQDim, nKvDim, nDModel := scalarI32(int32(qDim)), scalarI32(int32(kvDim)), scalarI32(int32(dModel))
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
		projResident = append(projResident, kDModel, kQDim, nQDim, nKvDim, nDModel)
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
		recordProj := func(li int, c metal.MTLIndirectComputeCommand, vec, out metal.MTLBuffer, outOff uint, p projIndex) {
			l := lb[li]
			switch p {
			case projQ:
				setQMV(c, psoQ, l.q, vec, out, outOff, kDModel, nQDim, qDim)
			case projK:
				setQMV(c, psoKV, l.k, vec, out, outOff, kDModel, nKvDim, kvDim)
			case projV:
				setQMV(c, psoKV, l.v, vec, out, outOff, kDModel, nKvDim, kvDim)
			case projO:
				setQMV(c, psoO, l.o, vec, out, outOff, kQDim, nDModel, dModel)
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
		outputs, coreErr = decodeForwardArchICBCore(inputs, specs, anwBufs, mnwBufs, kCaches, vCaches, projResident, qNormBufs, kNormBufs, postAttnBufs, postFFBufs, recordProj, 4, valueNormOnes, vProjIdx, dModel, nHeads, nKVHeads, headDim, dFF, maxLen, slidingWindow, lFF, base, scale, eps)
	})
	if coreErr != nil {
		return nil, coreErr
	}
	return outputs, nil
}
