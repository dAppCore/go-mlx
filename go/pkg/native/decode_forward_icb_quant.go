// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	core "dappco.re/go"
	"github.com/tmc/apple/metal"
)

// DecodeForwardICBQuant is the 4-bit cache-grow ICB — both levers stacked: 4-bit
// weights (qmv) cut the GPU, ICB replay cuts the per-token host re-encode. It is
// DecodeForwardICB with a qmv `recordProj` (affine_qmv_bfloat16_t) instead of gemv,
// running the same backend-agnostic decodeForwardICBCore. The V projection's output
// binds at index 4 (qmv) not 3 (gemv), so the per-token cache-row rebind uses
// vOutBind=4. This is the whole quantised decode forward, replay-driven, off mlx-c
// at runtime — the production-shaped fast path. Equals DecodeForwardQuant up to
// nothing (same kernels): gated byte-for-byte against it. All raw bf16 activations.
func DecodeForwardICBQuant(
	inputs [][]byte, qlayers []QuantizedLayerWeights,
	dModel, nHeads, nKVHeads, headDim, maxLen, dFF int,
	base, scale, eps float32,
) ([][]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	nLayers, T := len(qlayers), len(inputs)
	if nLayers == 0 || T == 0 {
		return nil, core.NewError("native.DecodeForwardICBQuant: need layers and inputs")
	}
	if T > maxLen {
		return nil, core.NewError("native.DecodeForwardICBQuant: more tokens than maxLen cache rows")
	}
	qDim, kvDim := nHeads*headDim, nKVHeads*headDim
	for i := range inputs {
		if len(inputs[i]) != dModel*bf16Size {
			return nil, core.NewError("native.DecodeForwardICBQuant: each input must be dModel bf16 bytes")
		}
	}
	type pj struct {
		w           QuantWeight
		outDim, inD int
	}
	for li := range qlayers {
		ql := qlayers[li]
		if ql.GroupSize == 0 || ql.Bits == 0 {
			return nil, core.NewError("native.DecodeForwardICBQuant: GroupSize/Bits unset")
		}
		if len(ql.AttnNormW) != dModel*bf16Size || len(ql.MLPNormW) != dModel*bf16Size {
			return nil, core.NewError("native.DecodeForwardICBQuant: norm weight size mismatch")
		}
		for _, p := range []pj{
			{ql.Q, qDim, dModel}, {ql.K, kvDim, dModel}, {ql.V, kvDim, dModel}, {ql.O, dModel, qDim},
			{ql.Gate, dFF, dModel}, {ql.Up, dFF, dModel}, {ql.Down, dModel, dFF},
		} {
			if !quantWeightShapeOK(p.w, p.outDim, p.inD, ql.GroupSize, ql.Bits) {
				return nil, core.NewError("native.DecodeForwardICBQuant: quantised weight size mismatch")
			}
		}
	}

	// qmv ICB pipelines, one per distinct (outDim,inDim,groupSize,bits) shape (built
	// before the pool so errors return cleanly). Mixed-precision packs (for example
	// 8-bit MLP beside 4-bit attention) need distinct recorded pipeline states.
	type qmvPSOKey struct{ outDim, inDim, groupSize, bits int }
	psoByKey := map[qmvPSOKey]metal.MTLComputePipelineState{}
	qmvPSO := func(outDim, inDim, groupSize, bits int) (metal.MTLComputePipelineState, error) {
		key := qmvPSOKey{outDim: outDim, inDim: inDim, groupSize: groupSize, bits: bits}
		if pso, ok := psoByKey[key]; ok {
			return pso, nil
		}
		variant := "_qmv_"
		if outDim%8 == 0 && inDim%512 == 0 {
			variant = "_qmv_fast_"
		}
		pso, err := pipelineForICB(core.Sprintf("affine%sbfloat16_t_gs_%d_b_%d_batch_0", variant, groupSize, bits))
		if err != nil {
			return nil, err
		}
		psoByKey[key] = pso
		return pso, nil
	}
	ensureQMVPSO := func(w QuantWeight, outDim, inDim, groupSize, bits int) error {
		groupSize, bits = quantWeightGeometry(w, groupSize, bits)
		_, err := qmvPSO(outDim, inDim, groupSize, bits)
		return err
	}
	for li := range qlayers {
		ql := qlayers[li]
		for _, p := range []pj{
			{ql.Q, qDim, dModel}, {ql.K, kvDim, dModel}, {ql.V, kvDim, dModel}, {ql.O, dModel, qDim},
			{ql.Gate, dFF, dModel}, {ql.Up, dFF, dModel}, {ql.Down, dModel, dFF},
		} {
			if err := ensureQMVPSO(p.w, p.outDim, p.inD, ql.GroupSize, ql.Bits); err != nil {
				return nil, err
			}
		}
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
		residentView := func(b []byte) bufView { return bufView{buf: residentBytes(b)} }
		mkW := func(w QuantWeight, groupSize, bits int) qmvWeight {
			groupSize, bits = quantWeightGeometry(w, groupSize, bits)
			return qmvWeight{wq: residentView(w.Packed), scales: residentView(w.Scales), biases: residentView(w.Biases), gs: groupSize, bits: bits}
		}
		psoFor := func(w qmvWeight, outDim, inDim int) metal.MTLComputePipelineState {
			return psoByKey[qmvPSOKey{outDim: outDim, inDim: inDim, groupSize: w.gs, bits: w.bits}]
		}
		// presized to the upper bound (every layer's 7 projections × wq/scales/biases, plus the
		// 7 trailing scalar buffers) so the per-forward build never geometrically regrows its
		// backing array. Byte-identical.
		projResident := make([]metal.MTLBuffer, 0, nLayers*7*3+7)
		for li := range qlayers {
			ql := qlayers[li]
			anwBufs[li] = residentBytes(ql.AttnNormW)
			mnwBufs[li] = residentBytes(ql.MLPNormW)
			kCaches[li] = device.NewBufferWithLengthOptions(cacheBytes, metal.MTLResourceStorageModeShared)
			vCaches[li] = device.NewBufferWithLengthOptions(cacheBytes, metal.MTLResourceStorageModeShared)
			lb[li] = lw{
				mkW(ql.Q, ql.GroupSize, ql.Bits), mkW(ql.K, ql.GroupSize, ql.Bits),
				mkW(ql.V, ql.GroupSize, ql.Bits), mkW(ql.O, ql.GroupSize, ql.Bits),
				mkW(ql.Gate, ql.GroupSize, ql.Bits), mkW(ql.Up, ql.GroupSize, ql.Bits),
				mkW(ql.Down, ql.GroupSize, ql.Bits),
			}
			for _, w := range []qmvWeight{lb[li].q, lb[li].k, lb[li].v, lb[li].o, lb[li].g, lb[li].u, lb[li].d} {
				projResident = append(projResident, w.wq.buf, w.scales.buf, w.biases.buf)
			}
		}
		// qmv K(=inDim) / N(=outDim) scalar params per shape (shared across layers)
		kDModel, kQDim, kDFF := scalarI32(int32(dModel)), scalarI32(int32(qDim)), scalarI32(int32(dFF))
		nQDim, nKvDim, nDModel, nDFF := scalarI32(int32(qDim)), scalarI32(int32(kvDim)), scalarI32(int32(dModel)), scalarI32(int32(dFF))
		projResident = append(projResident, kDModel, kQDim, kDFF, nQDim, nKvDim, nDModel, nDFF)

		// 4-bit qmv through the SHARED emitQMV body (with encQMVBF16); K/N bind the memoised count scalars.
		setQMV := func(c metal.MTLIndirectComputeCommand, pso metal.MTLComputePipelineState, w qmvWeight, vec, out metal.MTLBuffer, outOff uint, inDim, outDim int) {
			emitQMV(icbSink{c}, pso, w.wq.buf, w.wq.off, w.scales.buf, w.scales.off, w.biases.buf, w.biases.off, vec, out, outOff, inDim, outDim)
		}
		recordProj := func(li int, c metal.MTLIndirectComputeCommand, vec, out metal.MTLBuffer, outOff uint, p projIndex) {
			l := lb[li]
			switch p {
			case projQ:
				setQMV(c, psoFor(l.q, qDim, dModel), l.q, vec, out, outOff, dModel, qDim)
			case projK:
				setQMV(c, psoFor(l.k, kvDim, dModel), l.k, vec, out, outOff, dModel, kvDim)
			case projV:
				setQMV(c, psoFor(l.v, kvDim, dModel), l.v, vec, out, outOff, dModel, kvDim)
			case projO:
				setQMV(c, psoFor(l.o, dModel, qDim), l.o, vec, out, outOff, qDim, dModel)
			case projGate:
				setQMV(c, psoFor(l.g, dFF, dModel), l.g, vec, out, outOff, dModel, dFF)
			case projUp:
				setQMV(c, psoFor(l.u, dFF, dModel), l.u, vec, out, outOff, dModel, dFF)
			case projDown:
				setQMV(c, psoFor(l.d, dModel, dFF), l.d, vec, out, outOff, dFF, dModel)
			}
		}
		outputs, coreErr = decodeForwardICBCore(inputs, anwBufs, mnwBufs, kCaches, vCaches, projResident, recordProj, 4, dModel, nHeads, nKVHeads, headDim, dFF, maxLen, base, scale, eps)
	})
	if coreErr != nil {
		return nil, coreErr
	}
	return outputs, nil
}
