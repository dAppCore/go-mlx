// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"unsafe"

	core "dappco.re/go"
	g4 "dappco.re/go/mlx/pkg/model/gemma4"
	"github.com/tmc/apple/foundation"
	"github.com/tmc/apple/metal"
)

// decodeForwardArchICBCore is the ARCH-AWARE cache-grow ICB recorder + replay: like
// decodeForwardICBCore it records the decode stack ONCE and replays per token over a
// growing seq-major KV cache with cheap per-token offset rebinds, but it is DRIVEN by
// the declared arch (specs) — honouring the KV-cache topology (sharer layers attend an
// earlier owner's cache instead of their own) and per-layer sliding-window attention
// (the SDPA reads only the last W rows). MoE is NOT supported here (the router's host
// top-k can't live inside a single recorded/replayed command buffer).
//
// Layout: a uniform 24 ops/layer (base = 24·li) keeps indexing simple. A SHARER layer
// still records its K/V projections (ops 3-5) but to THROWAWAY scratch — its SDPA (op
// 6) reads the OWNER's cache. (Truly eliding the sharer's K/V matmuls would need a
// variable op layout; that's a perf micro-opt, not correctness — the output is identical.)
//
// Per-token rebind: offBuf (rope position), the two window-length buffers (nGlobalBuf =
// t+1, nSlidingBuf = min(t+1,W)), each OWNER layer's two cache-WRITE offsets (advancing
// row t), and each SLIDING layer's SDPA K/V READ offset (the window start). recordProj
// records the seven projections (gemv or qmv) exactly as the non-arch core; vOutBind is
// the projection output's bind index (gemv 3 / qmv 4).
func decodeForwardArchICBCore(
	inputs [][]byte, specs []g4.LayerSpec,
	anwBufs, mnwBufs, kCaches, vCaches, projResident []metal.MTLBuffer,
	recordProj func(li int, c metal.MTLIndirectComputeCommand, vec, out metal.MTLBuffer, outOff uint, p projIndex),
	vOutBind uint,
	dModel, nHeads, nKVHeads, headDim, dFF, maxLen, slidingWindow int,
	base, scale, eps float32,
) ([][]byte, error) {
	nLayers, T := len(anwBufs), len(inputs)
	qDim, kvDim := nHeads*headDim, nKVHeads*headDim

	rmsPSO, err := pipelineForICB("rmsbfloat16")
	if err != nil {
		return nil, err
	}
	ropePSO, err := ropePipelineICB(false)
	if err != nil {
		return nil, err
	}
	sdpaPSO, err := sdpaVectorPipelineICB(core.Sprintf("sdpa_vector_bfloat16_t_%d_%d", headDim, headDim))
	if err != nil {
		return nil, err
	}
	addPSO, err := pipelineForICB("vv_Addbfloat16")
	if err != nil {
		return nil, err
	}
	mulPSO, err := pipelineForICB("vv_Multiplybfloat16")
	if err != nil {
		return nil, err
	}
	tanhPSO, err := pipelineForICB("v_Tanhbfloat16bfloat16")
	if err != nil {
		return nil, err
	}

	outputs := make([][]byte, T)
	for i := range outputs {
		outputs[i] = make([]byte, dModel*bf16Size)
	}
	var coreErr error
	withAutoreleasePool(func() {
		normed := scratchBF16(dModel)
		q, qr, kProj, attn := scratchBF16(qDim), scratchBF16(qDim), scratchBF16(kvDim), scratchBF16(qDim)
		attnOut := scratchBF16(dModel)
		kThrow, vThrow := scratchBF16(kvDim), scratchBF16(kvDim) // sharer's discarded K/V
		mlpNormed := scratchBF16(dModel)
		gate, up := scratchBF16(dFF), scratchBF16(dFF)
		x2, x3, x3s, inner := scratchBF16(dFF), scratchBF16(dFF), scratchBF16(dFF), scratchBF16(dFF)
		scaled, tnh, onePlus, halfG := scratchBF16(dFF), scratchBF16(dFF), scratchBF16(dFF), scratchBF16(dFF)
		gelu, gated, down := scratchBF16(dFF), scratchBF16(dFF), scratchBF16(dModel)
		c044 := sharedBytes(bf16ConstBytes(dFF, 0.044715))
		c079 := sharedBytes(bf16ConstBytes(dFF, 0.7978845608028654))
		c1c := sharedBytes(bf16ConstBytes(dFF, 1.0))
		c05 := sharedBytes(bf16ConstBytes(dFF, 0.5))
		ping := [2]metal.MTLBuffer{scratchBF16(dModel), scratchBF16(dModel)}
		hBufs := make([]metal.MTLBuffer, nLayers)
		for i := range hBufs {
			hBufs[i] = scratchBF16(dModel)
		}

		off := int32(0)
		offBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&off), 4, metal.MTLResourceStorageModeShared)
		nGlobal := int32(1)
		nGlobalBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&nGlobal), 4, metal.MTLResourceStorageModeShared)
		nSliding := int32(1)
		nSlidingBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&nSliding), 4, metal.MTLResourceStorageModeShared)
		epsBuf, axisBuf, wsBuf := scalarF32(eps), scalarI32(int32(dModel)), scalarI32(1)
		ropeScaleB := scalarF32(scale)
		ropeMatB := scalarI64(int64(headDim))
		ropeBaseB := scalarF32(float32(math.Log2(float64(base))))
		gqaB := scalarI32(int32(nHeads / nKVHeads))
		khsB, kssB := scalarI64(int64(headDim)), scalarI64(int64(kvDim))
		vhsB, vssB := scalarI64(int64(headDim)), scalarI64(int64(kvDim))
		sdpaScaleB := scalarF32(scale)
		addModelB, cntFFB, tanhCntB := scalarI32(int32(dModel)), scalarI32(int32(dFF)), scalarI32(int32(dFF))

		resident := []metal.MTLBuffer{
			ping[0], ping[1], normed, q, qr, kProj, attn, attnOut, kThrow, vThrow, mlpNormed,
			gate, up, x2, x3, x3s, inner, scaled, tnh, onePlus, halfG, gelu, gated, down,
			c044, c079, c1c, c05,
			offBuf, nGlobalBuf, nSlidingBuf, epsBuf, axisBuf, wsBuf,
			ropeScaleB, ropeMatB, ropeBaseB, gqaB, khsB, kssB, vhsB, vssB, sdpaScaleB, addModelB, cntFFB, tanhCntB,
		}
		resident = append(resident, projResident...)
		resident = append(resident, anwBufs...)
		resident = append(resident, mnwBufs...)
		for _, b := range kCaches {
			if b != nil {
				resident = append(resident, b)
			}
		}
		for _, b := range vCaches {
			if b != nil {
				resident = append(resident, b)
			}
		}
		resident = append(resident, hBufs...)

		const opsPerLayer = 24
		total := opsPerLayer * nLayers
		icbDesc := metal.NewMTLIndirectCommandBufferDescriptor()
		icbDesc.SetCommandTypes(metal.MTLIndirectCommandTypeConcurrentDispatch)
		icbDesc.SetInheritBuffers(false)
		icbDesc.SetInheritPipelineState(false)
		icbDesc.SetMaxKernelBufferBindCount(16)
		icb := device.NewIndirectCommandBufferWithDescriptorMaxCommandCountOptions(icbDesc, uint(total), metal.MTLResourceStorageModeShared)

		rmsTG := uint(rmsSimdSize * ((((dModel + rmsNReads - 1) / rmsNReads) + rmsSimdSize - 1) / rmsSimdSize))
		elemGroup := func(n int) uint {
			if uint(n) < 256 {
				return uint(n)
			}
			return 256
		}
		setRMS := func(c metal.MTLIndirectComputeCommand, in, w, o metal.MTLBuffer) {
			c.SetComputePipelineState(rmsPSO)
			c.SetKernelBufferOffsetAtIndex(in, 0, 0)
			c.SetKernelBufferOffsetAtIndex(w, 0, 1)
			c.SetKernelBufferOffsetAtIndex(o, 0, 2)
			c.SetKernelBufferOffsetAtIndex(epsBuf, 0, 3)
			c.SetKernelBufferOffsetAtIndex(axisBuf, 0, 4)
			c.SetKernelBufferOffsetAtIndex(wsBuf, 0, 5)
			c.ConcurrentDispatchThreadsThreadsPerThreadgroup(metal.MTLSize{Width: rmsTG, Height: 1, Depth: 1}, metal.MTLSize{Width: rmsTG, Height: 1, Depth: 1})
		}
		setBin := func(c metal.MTLIndirectComputeCommand, pso metal.MTLComputePipelineState, a, b, o, cntB metal.MTLBuffer, n int) {
			c.SetComputePipelineState(pso)
			c.SetKernelBufferOffsetAtIndex(a, 0, 0)
			c.SetKernelBufferOffsetAtIndex(b, 0, 1)
			c.SetKernelBufferOffsetAtIndex(o, 0, 2)
			c.SetKernelBufferOffsetAtIndex(cntB, 0, 3)
			c.ConcurrentDispatchThreadsThreadsPerThreadgroup(metal.MTLSize{Width: uint(n), Height: 1, Depth: 1}, metal.MTLSize{Width: elemGroup(n), Height: 1, Depth: 1})
		}
		setRope := func(c metal.MTLIndirectComputeCommand, in, out metal.MTLBuffer, heads int) {
			c.SetComputePipelineState(ropePSO)
			c.SetKernelBufferOffsetAtIndex(in, 0, 0)
			c.SetKernelBufferOffsetAtIndex(out, 0, 1)
			c.SetKernelBufferOffsetAtIndex(offBuf, 0, 2)
			c.SetKernelBufferOffsetAtIndex(ropeScaleB, 0, 3)
			c.SetKernelBufferOffsetAtIndex(ropeMatB, 0, 4)
			c.SetKernelBufferOffsetAtIndex(ropeBaseB, 0, 10)
			d0 := uint(headDim / 2)
			c.ConcurrentDispatchThreadsThreadsPerThreadgroup(metal.MTLSize{Width: d0, Height: uint(heads), Depth: 1}, metal.MTLSize{Width: d0, Height: 1, Depth: 1})
		}

		// per-layer commands whose bindings advance per token
		kRopeCmd := make([]metal.MTLIndirectComputeCommand, nLayers) // owner cache-write (K)
		vCmd := make([]metal.MTLIndirectComputeCommand, nLayers)     // owner cache-write (V)
		sdpaCmd := make([]metal.MTLIndirectComputeCommand, nLayers)  // every layer (sliding: read offset)

		for li := 0; li < nLayers; li++ {
			b := opsPerLayer * li
			owns := specs[li].OwnsCache()
			ownerIdx := specs[li].KVShareFrom
			sliding := specs[li].Attention == g4.SlidingAttention
			attendK, attendV := kCaches[ownerIdx], vCaches[ownerIdx]
			nBufForLayer := nGlobalBuf
			if sliding {
				nBufForLayer = nSlidingBuf
			}
			inBuf, outBuf := ping[li%2], ping[(li+1)%2]
			hBuf := hBufs[li]
			cmd := func(op int) metal.MTLIndirectComputeCommand {
				c := icb.IndirectComputeCommandAtIndex(uint(b + op))
				if b+op != 0 {
					c.SetBarrier()
				}
				return c
			}
			// --- attention half (ops 0-8) ---
			setRMS(cmd(0), inBuf, anwBufs[li], normed)
			recordProj(li, cmd(1), normed, q, 0, projQ)
			setRope(cmd(2), q, qr, nHeads)
			// K/V projection: owner writes its cache; sharer writes throwaway scratch.
			recordProj(li, cmd(3), normed, kProj, 0, projK)
			if owns {
				ck := cmd(4)
				setRope(ck, kProj, kCaches[li], nKVHeads) // -> kCache @ row pos (rebound/token)
				kRopeCmd[li] = ck
				cv := cmd(5)
				recordProj(li, cv, normed, vCaches[li], 0, projV) // -> vCache @ row pos (rebound/token)
				vCmd[li] = cv
			} else {
				setRope(cmd(4), kProj, kThrow, nKVHeads)         // discarded
				recordProj(li, cmd(5), normed, vThrow, 0, projV) // discarded
			}
			// SDPA over the owner's cache; sliding layers read the windowed slice.
			cs := cmd(6)
			cs.SetComputePipelineState(sdpaPSO)
			cs.SetKernelBufferOffsetAtIndex(qr, 0, 0)
			cs.SetKernelBufferOffsetAtIndex(attendK, 0, 1) // read offset rebound/token if sliding
			cs.SetKernelBufferOffsetAtIndex(attendV, 0, 2)
			cs.SetKernelBufferOffsetAtIndex(attn, 0, 3)
			cs.SetKernelBufferOffsetAtIndex(gqaB, 0, 4)
			cs.SetKernelBufferOffsetAtIndex(nBufForLayer, 0, 5)
			cs.SetKernelBufferOffsetAtIndex(khsB, 0, 6)
			cs.SetKernelBufferOffsetAtIndex(kssB, 0, 7)
			cs.SetKernelBufferOffsetAtIndex(vhsB, 0, 8)
			cs.SetKernelBufferOffsetAtIndex(vssB, 0, 9)
			cs.SetKernelBufferOffsetAtIndex(sdpaScaleB, 0, 10)
			cs.ConcurrentDispatchThreadgroupsThreadsPerThreadgroup(metal.MTLSize{Width: uint(nHeads), Height: 1, Depth: 1}, metal.MTLSize{Width: 1024, Height: 1, Depth: 1})
			sdpaCmd[li] = cs
			recordProj(li, cmd(7), attn, attnOut, 0, projO)
			setBin(cmd(8), addPSO, inBuf, attnOut, hBuf, addModelB, dModel)

			// --- MLP half (ops 9-23) ---
			setRMS(cmd(9), hBuf, mnwBufs[li], mlpNormed)
			recordProj(li, cmd(10), mlpNormed, gate, 0, projGate)
			recordProj(li, cmd(11), mlpNormed, up, 0, projUp)
			setBin(cmd(12), mulPSO, gate, gate, x2, cntFFB, dFF)
			setBin(cmd(13), mulPSO, x2, gate, x3, cntFFB, dFF)
			setBin(cmd(14), mulPSO, x3, c044, x3s, cntFFB, dFF)
			setBin(cmd(15), addPSO, gate, x3s, inner, cntFFB, dFF)
			setBin(cmd(16), mulPSO, inner, c079, scaled, cntFFB, dFF)
			ct := cmd(17)
			ct.SetComputePipelineState(tanhPSO)
			ct.SetKernelBufferOffsetAtIndex(scaled, 0, 0)
			ct.SetKernelBufferOffsetAtIndex(tnh, 0, 1)
			ct.SetKernelBufferOffsetAtIndex(tanhCntB, 0, 2)
			ct.ConcurrentDispatchThreadsThreadsPerThreadgroup(metal.MTLSize{Width: uint(dFF), Height: 1, Depth: 1}, metal.MTLSize{Width: elemGroup(dFF), Height: 1, Depth: 1})
			setBin(cmd(18), addPSO, tnh, c1c, onePlus, cntFFB, dFF)
			setBin(cmd(19), mulPSO, gate, c05, halfG, cntFFB, dFF)
			setBin(cmd(20), mulPSO, halfG, onePlus, gelu, cntFFB, dFF)
			setBin(cmd(21), mulPSO, gelu, up, gated, cntFFB, dFF)
			recordProj(li, cmd(22), gated, down, 0, projDown)
			setBin(cmd(23), addPSO, hBuf, down, outBuf, addModelB, dModel)
		}

		lastOut := ping[nLayers%2]
		residentRes := make([]metal.MTLResource, len(resident))
		for i, bb := range resident {
			residentRes[i] = bb
		}
		rng := foundation.NSRange{Location: 0, Length: uint(total)}

		optCb := queue.CommandBuffer()
		blit := optCb.BlitCommandEncoder()
		blit.OptimizeIndirectCommandBufferWithRange(icb, rng)
		blit.EndEncoding()
		optCb.Commit()
		optCb.WaitUntilCompleted()

		rowBytes := kvDim * bf16Size
		for t := 0; t < T; t++ {
			*(*int32)(offBuf.Contents()) = int32(t)
			*(*int32)(nGlobalBuf.Contents()) = int32(t + 1)
			win := t + 1
			start := 0
			if slidingWindow > 0 && win > slidingWindow {
				start = win - slidingWindow
				win = slidingWindow
			}
			*(*int32)(nSlidingBuf.Contents()) = int32(win)
			rowOff := uint(t * rowBytes)
			slideOff := uint(start * rowBytes)
			for li := 0; li < nLayers; li++ {
				if specs[li].OwnsCache() {
					kRopeCmd[li].SetKernelBufferOffsetAtIndex(kCaches[li], rowOff, 1)
					vCmd[li].SetKernelBufferOffsetAtIndex(vCaches[li], rowOff, vOutBind)
				}
				if specs[li].Attention == g4.SlidingAttention {
					own := specs[li].KVShareFrom
					sdpaCmd[li].SetKernelBufferOffsetAtIndex(kCaches[own], slideOff, 1)
					sdpaCmd[li].SetKernelBufferOffsetAtIndex(vCaches[own], slideOff, 2)
				}
			}
			copy(unsafe.Slice((*byte)(ping[0].Contents()), dModel*bf16Size), inputs[t])

			cb := queue.CommandBuffer()
			enc := cb.ComputeCommandEncoder()
			enc.UseResourcesCountUsage(residentRes, uint(len(residentRes)), metal.MTLResourceUsageRead|metal.MTLResourceUsageWrite)
			enc.ExecuteCommandsInBufferWithRange(icb, rng)
			enc.EndEncoding()
			cb.Commit()
			cb.WaitUntilCompleted()
			copy(outputs[t], unsafe.Slice((*byte)(lastOut.Contents()), dModel*bf16Size))
		}
	})
	if coreErr != nil {
		return nil, coreErr
	}
	return outputs, nil
}

// DecodeForwardArchICB is the bf16 ARCH-driven cache-grow ICB: the encode-bypass replay
// of DecodeForwardArch (KV-share + sliding-window), recorded once and replayed per token.
// It builds a gemv recorder + the per-layer weight/cache buffers (caches for OWNER layers
// only) and runs decodeForwardArchICBCore. Byte-for-byte equal to DecodeForwardArch on
// the same arch (gated). MoE layers are not supported (rejected). All bf16.
func DecodeForwardArchICB(
	inputs [][]byte, layers []DecodeLayerWeights, specs []g4.LayerSpec,
	dModel, nHeads, nKVHeads, headDim, maxLen, dFF, slidingWindow int,
	base, scale, eps float32,
) ([][]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	nLayers, T := len(layers), len(inputs)
	if nLayers == 0 || T == 0 {
		return nil, core.NewError("native.DecodeForwardArchICB: need layers and inputs")
	}
	if len(specs) != nLayers {
		return nil, core.NewError("native.DecodeForwardArchICB: specs length must equal layers")
	}
	if T > maxLen {
		return nil, core.NewError("native.DecodeForwardArchICB: more tokens than maxLen cache rows")
	}
	qDim, kvDim := nHeads*headDim, nKVHeads*headDim
	for i := range inputs {
		if len(inputs[i]) != dModel*bf16Size {
			return nil, core.NewError("native.DecodeForwardArchICB: each input must be dModel bf16 bytes")
		}
	}
	for li := range specs {
		o := specs[li].KVShareFrom
		if o < 0 || o > li || (o != li && !specs[o].OwnsCache()) {
			return nil, core.NewError("native.DecodeForwardArchICB: KVShareFrom must reference an earlier owner layer")
		}
		if specs[li].MoE {
			return nil, core.NewError("native.DecodeForwardArchICB: MoE layers are not supported on the ICB path")
		}
	}

	gemvPSO := func(inDim, outDim int) (metal.MTLComputePipelineState, int, int, int, int, error) {
		bm, bn, sm, sn, tm, tn := gemvTiles(inDim, outDim)
		p, e := pipelineForICB(core.Sprintf("gemv_bfloat16_bm%d_bn%d_sm%d_sn%d_tm%d_tn%d_nc0_axpby0", bm, bn, sm, sn, tm, tn))
		return p, bm, bn, sm, tm, e
	}
	psoQ, bmQ, bnQ, smQ, tmQ, err := gemvPSO(dModel, qDim)
	if err != nil {
		return nil, err
	}
	psoKV, bmKV, bnKV, smKV, tmKV, err := gemvPSO(dModel, kvDim)
	if err != nil {
		return nil, err
	}
	psoO, bmO, bnO, smO, tmO, err := gemvPSO(qDim, dModel)
	if err != nil {
		return nil, err
	}
	psoF, bmF, bnF, smF, tmF, err := gemvPSO(dModel, dFF)
	if err != nil {
		return nil, err
	}
	psoD, bmD, bnD, smD, tmD, err := gemvPSO(dFF, dModel)
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
		type lw struct{ wq, wk, wv, wo, wg, wu, wd metal.MTLBuffer }
		lb := make([]lw, nLayers)
		cacheBytes := uint(maxLen * kvDim * bf16Size)
		var projResident []metal.MTLBuffer
		for li := range layers {
			w := layers[li]
			anwBufs[li] = sharedBytes(w.AttnNormW)
			mnwBufs[li] = sharedBytes(w.MLPNormW)
			if specs[li].OwnsCache() {
				kCaches[li] = device.NewBufferWithLengthOptions(cacheBytes, metal.MTLResourceStorageModeShared)
				vCaches[li] = device.NewBufferWithLengthOptions(cacheBytes, metal.MTLResourceStorageModeShared)
			}
			lb[li] = lw{sharedBytes(w.WQ), sharedBytes(w.WK), sharedBytes(w.WV), sharedBytes(w.WO), sharedBytes(w.WGate), sharedBytes(w.WUp), sharedBytes(w.WDown)}
			projResident = append(projResident, lb[li].wq, lb[li].wk, lb[li].wv, lb[li].wo, lb[li].wg, lb[li].wu, lb[li].wd)
		}
		qInB, qOutB, qLdB := scalarI32(int32(dModel)), scalarI32(int32(qDim)), scalarI32(int32(dModel))
		kvInB, kvOutB, kvLdB := scalarI32(int32(dModel)), scalarI32(int32(kvDim)), scalarI32(int32(dModel))
		oInB, oOutB, oLdB := scalarI32(int32(qDim)), scalarI32(int32(dModel)), scalarI32(int32(qDim))
		fInB, fOutB, fLdB := scalarI32(int32(dModel)), scalarI32(int32(dFF)), scalarI32(int32(dModel))
		dInB, dOutB, dLdB := scalarI32(int32(dFF)), scalarI32(int32(dModel)), scalarI32(int32(dFF))
		bndB, bshB, vsB, msB := scalarI32(1), scalarI32(1), scalarI64(0), scalarI64(0)
		projResident = append(projResident, qInB, qOutB, qLdB, kvInB, kvOutB, kvLdB, oInB, oOutB, oLdB, fInB, fOutB, fLdB, dInB, dOutB, dLdB, bndB, bshB, vsB, msB)

		gemvGrid := func(outDim, bm, sm, tm int) uint { return uint((outDim + bm*sm*tm - 1) / (bm * sm * tm)) }
		setGemv := func(c metal.MTLIndirectComputeCommand, pso metal.MTLComputePipelineState, mat, vec, o, inB, outB, ldB metal.MTLBuffer, outOff uint, outDim, bm, bn, sm, tm int) {
			c.SetComputePipelineState(pso)
			c.SetKernelBufferOffsetAtIndex(mat, 0, 0)
			c.SetKernelBufferOffsetAtIndex(vec, 0, 1)
			c.SetKernelBufferOffsetAtIndex(o, outOff, 3)
			c.SetKernelBufferOffsetAtIndex(inB, 0, 4)
			c.SetKernelBufferOffsetAtIndex(outB, 0, 5)
			c.SetKernelBufferOffsetAtIndex(ldB, 0, 6)
			c.SetKernelBufferOffsetAtIndex(bndB, 0, 9)
			c.SetKernelBufferOffsetAtIndex(bshB, 0, 10)
			c.SetKernelBufferOffsetAtIndex(vsB, 0, 11)
			c.SetKernelBufferOffsetAtIndex(msB, 0, 12)
			c.ConcurrentDispatchThreadgroupsThreadsPerThreadgroup(metal.MTLSize{Width: gemvGrid(outDim, bm, sm, tm), Height: 1, Depth: 1}, metal.MTLSize{Width: 32, Height: uint(bn), Depth: uint(bm)})
		}
		recordProj := func(li int, c metal.MTLIndirectComputeCommand, vec, out metal.MTLBuffer, outOff uint, p projIndex) {
			l := lb[li]
			switch p {
			case projQ:
				setGemv(c, psoQ, l.wq, vec, out, qInB, qOutB, qLdB, outOff, qDim, bmQ, bnQ, smQ, tmQ)
			case projK:
				setGemv(c, psoKV, l.wk, vec, out, kvInB, kvOutB, kvLdB, outOff, kvDim, bmKV, bnKV, smKV, tmKV)
			case projV:
				setGemv(c, psoKV, l.wv, vec, out, kvInB, kvOutB, kvLdB, outOff, kvDim, bmKV, bnKV, smKV, tmKV)
			case projO:
				setGemv(c, psoO, l.wo, vec, out, oInB, oOutB, oLdB, outOff, dModel, bmO, bnO, smO, tmO)
			case projGate:
				setGemv(c, psoF, l.wg, vec, out, fInB, fOutB, fLdB, outOff, dFF, bmF, bnF, smF, tmF)
			case projUp:
				setGemv(c, psoF, l.wu, vec, out, fInB, fOutB, fLdB, outOff, dFF, bmF, bnF, smF, tmF)
			case projDown:
				setGemv(c, psoD, l.wd, vec, out, dInB, dOutB, dLdB, outOff, dModel, bmD, bnD, smD, tmD)
			}
		}
		outputs, coreErr = decodeForwardArchICBCore(inputs, specs, anwBufs, mnwBufs, kCaches, vCaches, projResident, recordProj, 3, dModel, nHeads, nKVHeads, headDim, dFF, maxLen, slidingWindow, base, scale, eps)
	})
	if coreErr != nil {
		return nil, coreErr
	}
	return outputs, nil
}
