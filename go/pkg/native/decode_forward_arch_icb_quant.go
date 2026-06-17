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
	base, scale, eps float32,
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
		for _, p := range []pj{
			{ql.Q, qDim, dModel}, {ql.K, kvDim, dModel}, {ql.V, kvDim, dModel}, {ql.O, dModel, qDim},
			{ql.Gate, dFF, dModel}, {ql.Up, dFF, dModel}, {ql.Down, dModel, dFF},
		} {
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
	psoF, err := qmvPSO(dFF, dModel)
	if err != nil {
		return nil, err
	}
	psoD, err := qmvPSO(dModel, dFF)
	if err != nil {
		return nil, err
	}

	var outputs [][]byte
	var coreErr error
	withAutoreleasePool(func() {
		anwBufs := make([]metal.MTLBuffer, nLayers)
		mnwBufs := make([]metal.MTLBuffer, nLayers)
		kCaches := make([]metal.MTLBuffer, nLayers)
		vCaches := make([]metal.MTLBuffer, nLayers)
		type lw struct{ q, k, v, o, g, u, d qmvWeight }
		lb := make([]lw, nLayers)
		cacheBytes := uint(maxLen * kvDim * bf16Size)
		mkW := func(w QuantWeight) qmvWeight {
			return qmvWeight{sharedBytes(w.Packed), sharedBytes(w.Scales), sharedBytes(w.Biases)}
		}
		var projResident []metal.MTLBuffer
		for li := range qlayers {
			ql := qlayers[li]
			anwBufs[li] = sharedBytes(ql.AttnNormW)
			mnwBufs[li] = sharedBytes(ql.MLPNormW)
			if specs[li].OwnsCache() {
				kCaches[li] = device.NewBufferWithLengthOptions(cacheBytes, metal.MTLResourceStorageModeShared)
				vCaches[li] = device.NewBufferWithLengthOptions(cacheBytes, metal.MTLResourceStorageModeShared)
			}
			lb[li] = lw{mkW(ql.Q), mkW(ql.K), mkW(ql.V), mkW(ql.O), mkW(ql.Gate), mkW(ql.Up), mkW(ql.Down)}
			for _, w := range []qmvWeight{lb[li].q, lb[li].k, lb[li].v, lb[li].o, lb[li].g, lb[li].u, lb[li].d} {
				projResident = append(projResident, w.wq, w.scales, w.biases)
			}
		}
		kDModel, kQDim, kDFF := scalarI32(int32(dModel)), scalarI32(int32(qDim)), scalarI32(int32(dFF))
		nQDim, nKvDim, nDModel, nDFF := scalarI32(int32(qDim)), scalarI32(int32(kvDim)), scalarI32(int32(dModel)), scalarI32(int32(dFF))
		projResident = append(projResident, kDModel, kQDim, kDFF, nQDim, nKvDim, nDModel, nDFF)

		setQMV := func(c metal.MTLIndirectComputeCommand, pso metal.MTLComputePipelineState, w qmvWeight, vec, out metal.MTLBuffer, outOff uint, kB, nB metal.MTLBuffer, outDim int) {
			c.SetComputePipelineState(pso)
			c.SetKernelBufferOffsetAtIndex(w.wq, 0, 0)
			c.SetKernelBufferOffsetAtIndex(w.scales, 0, 1)
			c.SetKernelBufferOffsetAtIndex(w.biases, 0, 2)
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
				setQMV(c, psoF, l.g, vec, out, outOff, kDModel, nDFF, dFF)
			case projUp:
				setQMV(c, psoF, l.u, vec, out, outOff, kDModel, nDFF, dFF)
			case projDown:
				setQMV(c, psoD, l.d, vec, out, outOff, kDFF, nDModel, dModel)
			}
		}
		outputs, coreErr = decodeForwardArchICBCore(inputs, specs, anwBufs, mnwBufs, kCaches, vCaches, projResident, recordProj, 4, dModel, nHeads, nKVHeads, headDim, dFF, maxLen, slidingWindow, base, scale, eps)
	})
	if coreErr != nil {
		return nil, coreErr
	}
	return outputs, nil
}
