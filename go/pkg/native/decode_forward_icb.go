// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/foundation"
	"github.com/tmc/apple/metal"
)

// DecodeForwardICB is DecodeForward over a GROWING KV cache via ICB replay — the
// cache-grow ICB. It records the full N-layer real decode stack (24 ops/layer:
// the AttentionStepKV half with K/V projections + cache write, then the MLP half)
// ONCE into one indirect command buffer, then replays the whole stack per token.
//
// The crux a fixed ICB can't express directly is the cache WRITE row, which
// advances every token. The lever (proven in TestICBRebindOffset): an ICB
// command's bindings are recorded once, but re-setting ONE buffer offset between
// replays is cheap and takes effect. So per token only four things change — the
// position buffer (offBuf), the attention window length (nBuf), and each layer's
// two cache-write OUTPUT offsets (the K-RoPE and V-gemv that write into the
// seq-major cache row) — i.e. 2*nLayers offset re-sets + 2 buffer writes, versus
// re-encoding 24*nLayers ops. Everything else (pipelines, weights, scratch,
// residual ping-pong, seq-major SDPA strides) stays recorded.
//
// Same signature/semantics as DecodeForward; with the same weights/inputs it
// equals it byte-for-byte (gated in the tests). All raw bf16.
func DecodeForwardICB(
	inputs [][]byte, layers []DecodeLayerWeights,
	dModel, nHeads, nKVHeads, headDim, maxLen, dFF int,
	base, scale, eps float32,
) ([][]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	nLayers, T := len(layers), len(inputs)
	if nLayers == 0 || T == 0 {
		return nil, core.NewError("native.DecodeForwardICB: need layers and inputs")
	}
	if T > maxLen {
		return nil, core.NewError("native.DecodeForwardICB: more tokens than maxLen cache rows")
	}
	qDim, kvDim := nHeads*headDim, nKVHeads*headDim
	for i := range inputs {
		if len(inputs[i]) != dModel*bf16Size {
			return nil, core.NewError("native.DecodeForwardICB: each input must be dModel bf16 bytes")
		}
	}
	for li := range layers {
		w := layers[li]
		if len(w.AttnNormW) != dModel*bf16Size || len(w.MLPNormW) != dModel*bf16Size ||
			len(w.WQ) != qDim*dModel*bf16Size || len(w.WO) != dModel*qDim*bf16Size ||
			len(w.WK) != kvDim*dModel*bf16Size || len(w.WV) != kvDim*dModel*bf16Size ||
			len(w.WGate) != dFF*dModel*bf16Size || len(w.WUp) != dFF*dModel*bf16Size || len(w.WDown) != dModel*dFF*bf16Size {
			return nil, core.NewError("native.DecodeForwardICB: layer weight size mismatch")
		}
	}

	// ICB-capable pipelines (one per distinct gemv tile shape + the rest).
	rmsPSO, err := pipelineForICB("rmsbfloat16")
	if err != nil {
		return nil, err
	}
	gemvPSO := func(inDim, outDim int) (metal.MTLComputePipelineState, int, int, int, int, error) {
		bm, bn, sm, sn, tm, tn := gemvTiles(inDim, outDim)
		p, e := pipelineForICB(core.Sprintf("gemv_bfloat16_bm%d_bn%d_sm%d_sn%d_tm%d_tn%d_nc0_axpby0", bm, bn, sm, sn, tm, tn))
		return p, bm, bn, sm, tm, e
	}
	gemvQPSO, bmQ, bnQ, smQ, tmQ, err := gemvPSO(dModel, qDim)
	if err != nil {
		return nil, err
	}
	gemvKVPSO, bmKV, bnKV, smKV, tmKV, err := gemvPSO(dModel, kvDim)
	if err != nil {
		return nil, err
	}
	gemvOPSO, bmO, bnO, smO, tmO, err := gemvPSO(qDim, dModel)
	if err != nil {
		return nil, err
	}
	gemvFPSO, bmF, bnF, smF, tmF, err := gemvPSO(dModel, dFF)
	if err != nil {
		return nil, err
	}
	gemvDPSO, bmD, bnD, smD, tmD, err := gemvPSO(dFF, dModel)
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
	withAutoreleasePool(func() {
		// per-layer resident weights + caches (caches zeroed; rows fill as tokens append)
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

		// shared scratch + residual ping-pong
		normed := scratchBF16(dModel)
		q, qr, kProj, attn := scratchBF16(qDim), scratchBF16(qDim), scratchBF16(kvDim), scratchBF16(qDim)
		attnOut := scratchBF16(dModel)
		mlpNormed := scratchBF16(dModel)
		gate, up := scratchBF16(dFF), scratchBF16(dFF)
		x2, x3, x3s, inner := scratchBF16(dFF), scratchBF16(dFF), scratchBF16(dFF), scratchBF16(dFF)
		scaled, tnh, onePlus, halfG := scratchBF16(dFF), scratchBF16(dFF), scratchBF16(dFF), scratchBF16(dFF)
		gelu, gated, down := scratchBF16(dFF), scratchBF16(dFF), scratchBF16(dModel)
		c044 := sharedBytes(bf16ConstBytes(dFF, 0.044715))
		c079 := sharedBytes(bf16ConstBytes(dFF, 0.7978845608028654))
		c1c := sharedBytes(bf16ConstBytes(dFF, 1.0))
		c05 := sharedBytes(bf16ConstBytes(dFF, 0.5))
		// h is per-layer but reused (serial barriers); ping-pong residual stream
		ping := [2]metal.MTLBuffer{scratchBF16(dModel), scratchBF16(dModel)}
		hBufs := make([]metal.MTLBuffer, nLayers)
		for i := range hBufs {
			hBufs[i] = scratchBF16(dModel)
		}

		// scalar buffers (shared; offBuf + nBuf are bumped per token)
		off := int32(0)
		offBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&off), 4, metal.MTLResourceStorageModeShared)
		nWin := int32(1)
		nBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&nWin), 4, metal.MTLResourceStorageModeShared)
		epsBuf, axisBuf, wsBuf := scalarF32(eps), scalarI32(int32(dModel)), scalarI32(1)
		qInB, qOutB, qLdB := scalarI32(int32(dModel)), scalarI32(int32(qDim)), scalarI32(int32(dModel))
		kvInB, kvOutB, kvLdB := scalarI32(int32(dModel)), scalarI32(int32(kvDim)), scalarI32(int32(dModel))
		oInB, oOutB, oLdB := scalarI32(int32(qDim)), scalarI32(int32(dModel)), scalarI32(int32(qDim))
		fInB, fOutB, fLdB := scalarI32(int32(dModel)), scalarI32(int32(dFF)), scalarI32(int32(dModel))
		dInB, dOutB, dLdB := scalarI32(int32(dFF)), scalarI32(int32(dModel)), scalarI32(int32(dFF))
		bndB, bshB, vsB, msB := scalarI32(1), scalarI32(1), scalarI64(0), scalarI64(0)
		ropeScaleB := scalarF32(scale)
		ropeMatB := scalarI64(int64(headDim))
		ropeBaseB := scalarF32(float32(math.Log2(float64(base))))
		gqaB := scalarI32(int32(nHeads / nKVHeads))
		// seq-major cache strides: head jumps headDim, seq jumps kvDim (one row)
		khsB, kssB := scalarI64(int64(headDim)), scalarI64(int64(kvDim))
		vhsB, vssB := scalarI64(int64(headDim)), scalarI64(int64(kvDim))
		sdpaScaleB := scalarF32(scale)
		addModelB, cntFFB, tanhCntB := scalarI32(int32(dModel)), scalarI32(int32(dFF)), scalarI32(int32(dFF))

		resident := []metal.MTLBuffer{
			ping[0], ping[1], normed, q, qr, kProj, attn, attnOut, mlpNormed,
			gate, up, x2, x3, x3s, inner, scaled, tnh, onePlus, halfG, gelu, gated, down,
			c044, c079, c1c, c05,
			offBuf, nBuf, epsBuf, axisBuf, wsBuf,
			qInB, qOutB, qLdB, kvInB, kvOutB, kvLdB, oInB, oOutB, oLdB, fInB, fOutB, fLdB, dInB, dOutB, dLdB,
			bndB, bshB, vsB, msB, ropeScaleB, ropeMatB, ropeBaseB,
			gqaB, khsB, kssB, vhsB, vssB, sdpaScaleB, addModelB, cntFFB, tanhCntB,
		}
		for li := range lb {
			l := lb[li]
			resident = append(resident, l.anw, l.wq, l.wk, l.wv, l.wo, l.mnw, l.wg, l.wu, l.wd, l.kCache, l.vCache)
		}
		for i := range hBufs {
			resident = append(resident, hBufs[i])
		}

		const opsPerLayer = 24
		total := opsPerLayer * nLayers
		icbDesc := metal.NewMTLIndirectCommandBufferDescriptor()
		icbDesc.SetCommandTypes(metal.MTLIndirectCommandTypeConcurrentDispatch)
		icbDesc.SetInheritBuffers(false)
		icbDesc.SetInheritPipelineState(false)
		icbDesc.SetMaxKernelBufferBindCount(16)
		icb := device.NewIndirectCommandBufferWithDescriptorMaxCommandCountOptions(icbDesc, uint(total), metal.MTLResourceStorageModeShared)

		rmsTG := uint(rmsSimdSize * ((((dModel + rmsNReads - 1) / rmsNReads) + rmsSimdSize - 1) / rmsSimdSize))
		gemvGrid := func(outDim, bm, sm, tm int) uint { return uint((outDim + bm*sm*tm - 1) / (bm * sm * tm)) }
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
		setGemv := func(c metal.MTLIndirectComputeCommand, pso metal.MTLComputePipelineState, mat, vec, o, inB, outB, ldB metal.MTLBuffer, outDim, bm, bn, sm, tm int, outByteOff uint) {
			c.SetComputePipelineState(pso)
			c.SetKernelBufferOffsetAtIndex(mat, 0, 0)
			c.SetKernelBufferOffsetAtIndex(vec, 0, 1)
			c.SetKernelBufferOffsetAtIndex(o, outByteOff, 3)
			c.SetKernelBufferOffsetAtIndex(inB, 0, 4)
			c.SetKernelBufferOffsetAtIndex(outB, 0, 5)
			c.SetKernelBufferOffsetAtIndex(ldB, 0, 6)
			c.SetKernelBufferOffsetAtIndex(bndB, 0, 9)
			c.SetKernelBufferOffsetAtIndex(bshB, 0, 10)
			c.SetKernelBufferOffsetAtIndex(vsB, 0, 11)
			c.SetKernelBufferOffsetAtIndex(msB, 0, 12)
			c.ConcurrentDispatchThreadgroupsThreadsPerThreadgroup(metal.MTLSize{Width: gemvGrid(outDim, bm, sm, tm), Height: 1, Depth: 1}, metal.MTLSize{Width: 32, Height: uint(bn), Depth: uint(bm)})
		}
		setBin := func(c metal.MTLIndirectComputeCommand, pso metal.MTLComputePipelineState, a, b, o, cntB metal.MTLBuffer, n int) {
			c.SetComputePipelineState(pso)
			c.SetKernelBufferOffsetAtIndex(a, 0, 0)
			c.SetKernelBufferOffsetAtIndex(b, 0, 1)
			c.SetKernelBufferOffsetAtIndex(o, 0, 2)
			c.SetKernelBufferOffsetAtIndex(cntB, 0, 3)
			c.ConcurrentDispatchThreadsThreadsPerThreadgroup(metal.MTLSize{Width: uint(n), Height: 1, Depth: 1}, metal.MTLSize{Width: elemGroup(n), Height: 1, Depth: 1})
		}

		// per-layer cache-write commands whose OUTPUT offset is re-set per token
		kRopeCmd := make([]metal.MTLIndirectComputeCommand, nLayers)
		vGemvCmd := make([]metal.MTLIndirectComputeCommand, nLayers)

		for li := 0; li < nLayers; li++ {
			l := lb[li]
			base := opsPerLayer * li
			inBuf, outBuf := ping[li%2], ping[(li+1)%2]
			hBuf := hBufs[li]
			cmd := func(op int) metal.MTLIndirectComputeCommand {
				c := icb.IndirectComputeCommandAtIndex(uint(base + op))
				if base+op != 0 {
					c.SetBarrier()
				}
				return c
			}
			// --- attention half with cache write (ops 0-8) ---
			setRMS(cmd(0), inBuf, l.anw, normed)
			setGemv(cmd(1), gemvQPSO, l.wq, normed, q, qInB, qOutB, qLdB, qDim, bmQ, bnQ, smQ, tmQ, 0)
			// 2: rope q -> qr
			c := cmd(2)
			c.SetComputePipelineState(ropePSO)
			c.SetKernelBufferOffsetAtIndex(q, 0, 0)
			c.SetKernelBufferOffsetAtIndex(qr, 0, 1)
			c.SetKernelBufferOffsetAtIndex(offBuf, 0, 2)
			c.SetKernelBufferOffsetAtIndex(ropeScaleB, 0, 3)
			c.SetKernelBufferOffsetAtIndex(ropeMatB, 0, 4)
			c.SetKernelBufferOffsetAtIndex(ropeBaseB, 0, 10)
			ropeQDim0 := uint(headDim / 2)
			c.ConcurrentDispatchThreadsThreadsPerThreadgroup(metal.MTLSize{Width: ropeQDim0, Height: uint(nHeads), Depth: 1}, metal.MTLSize{Width: ropeQDim0, Height: 1, Depth: 1})
			// 3: gemv K -> kProj
			setGemv(cmd(3), gemvKVPSO, l.wk, normed, kProj, kvInB, kvOutB, kvLdB, kvDim, bmKV, bnKV, smKV, tmKV, 0)
			// 4: rope K -> kCache @ row pos  (OUTPUT OFFSET re-set per token)
			c = cmd(4)
			c.SetComputePipelineState(ropePSO)
			c.SetKernelBufferOffsetAtIndex(kProj, 0, 0)
			c.SetKernelBufferOffsetAtIndex(l.kCache, 0, 1)
			c.SetKernelBufferOffsetAtIndex(offBuf, 0, 2)
			c.SetKernelBufferOffsetAtIndex(ropeScaleB, 0, 3)
			c.SetKernelBufferOffsetAtIndex(ropeMatB, 0, 4)
			c.SetKernelBufferOffsetAtIndex(ropeBaseB, 0, 10)
			ropeKDim0 := uint(headDim / 2)
			c.ConcurrentDispatchThreadsThreadsPerThreadgroup(metal.MTLSize{Width: ropeKDim0, Height: uint(nKVHeads), Depth: 1}, metal.MTLSize{Width: ropeKDim0, Height: 1, Depth: 1})
			kRopeCmd[li] = c
			// 5: gemv V -> vCache @ row pos  (OUTPUT OFFSET re-set per token)
			cv := cmd(5)
			setGemv(cv, gemvKVPSO, l.wv, normed, l.vCache, kvInB, kvOutB, kvLdB, kvDim, bmKV, bnKV, smKV, tmKV, 0)
			vGemvCmd[li] = cv
			// 6: sdpa over the grown window (N from nBuf; seq-major strides)
			c = cmd(6)
			c.SetComputePipelineState(sdpaPSO)
			c.SetKernelBufferOffsetAtIndex(qr, 0, 0)
			c.SetKernelBufferOffsetAtIndex(l.kCache, 0, 1)
			c.SetKernelBufferOffsetAtIndex(l.vCache, 0, 2)
			c.SetKernelBufferOffsetAtIndex(attn, 0, 3)
			c.SetKernelBufferOffsetAtIndex(gqaB, 0, 4)
			c.SetKernelBufferOffsetAtIndex(nBuf, 0, 5)
			c.SetKernelBufferOffsetAtIndex(khsB, 0, 6)
			c.SetKernelBufferOffsetAtIndex(kssB, 0, 7)
			c.SetKernelBufferOffsetAtIndex(vhsB, 0, 8)
			c.SetKernelBufferOffsetAtIndex(vssB, 0, 9)
			c.SetKernelBufferOffsetAtIndex(sdpaScaleB, 0, 10)
			c.ConcurrentDispatchThreadgroupsThreadsPerThreadgroup(metal.MTLSize{Width: uint(nHeads), Height: 1, Depth: 1}, metal.MTLSize{Width: 1024, Height: 1, Depth: 1})
			// 7: gemv Wo -> attnOut
			setGemv(cmd(7), gemvOPSO, l.wo, attn, attnOut, oInB, oOutB, oLdB, dModel, bmO, bnO, smO, tmO, 0)
			// 8: add inBuf + attnOut -> h
			setBin(cmd(8), addPSO, inBuf, attnOut, hBuf, addModelB, dModel)

			// --- MLP half (ops 9-23) ---
			setRMS(cmd(9), hBuf, l.mnw, mlpNormed)
			setGemv(cmd(10), gemvFPSO, l.wg, mlpNormed, gate, fInB, fOutB, fLdB, dFF, bmF, bnF, smF, tmF, 0)
			setGemv(cmd(11), gemvFPSO, l.wu, mlpNormed, up, fInB, fOutB, fLdB, dFF, bmF, bnF, smF, tmF, 0)
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
			setGemv(cmd(22), gemvDPSO, l.wd, gated, down, dInB, dOutB, dLdB, dModel, bmD, bnD, smD, tmD, 0)
			setBin(cmd(23), addPSO, hBuf, down, outBuf, addModelB, dModel)
		}

		lastOut := ping[nLayers%2] // residual stream output after N ping-pong swaps
		// residency in ONE batched call per token, not one per buffer: at E2B scale
		// `resident` is ~487 buffers (35 layers × 11 + scratch + scalars), and a
		// per-buffer UseResource loop was ~66% of the per-token wall (487 purego→objc
		// calls/token, GPU idle meanwhile). UseResourcesCountUsage marks them all in
		// a single call. Built once; the set is identical every token.
		residentRes := make([]metal.MTLResource, len(resident))
		for i, b := range resident {
			residentRes[i] = b
		}
		rng := foundation.NSRange{Location: 0, Length: uint(total)}

		// Optimize the recorded ICB once: without this the driver re-processes the
		// whole command buffer on every execute (the per-token cache-write offset
		// rebinds mutate it), which is host work with the GPU idle. Offset-only
		// rebinds after optimize are cheap and don't require re-optimizing.
		optCb := queue.CommandBuffer()
		blit := optCb.BlitCommandEncoder()
		blit.OptimizeIndirectCommandBufferWithRange(icb, rng)
		blit.EndEncoding()
		optCb.Commit()
		optCb.WaitUntilCompleted()
		rowBytes := kvDim * bf16Size
		for t := 0; t < T; t++ {
			*(*int32)(offBuf.Contents()) = int32(t)
			*(*int32)(nBuf.Contents()) = int32(t + 1)
			rowOff := uint(t * rowBytes)
			for li := 0; li < nLayers; li++ {
				// advance this token's cache-write row on the two recorded commands
				kRopeCmd[li].SetKernelBufferOffsetAtIndex(lb[li].kCache, rowOff, 1)
				vGemvCmd[li].SetKernelBufferOffsetAtIndex(lb[li].vCache, rowOff, 3)
			}
			copy(unsafe.Slice((*byte)(ping[0].Contents()), dModel*bf16Size), inputs[t])

			cb := queue.CommandBuffer()
			enc := cb.ComputeCommandEncoder()
			enc.UseResourcesCountUsage(residentRes, uint(len(residentRes)), metal.MTLResourceUsageRead|metal.MTLResourceUsageWrite)
			enc.ExecuteCommandsInBufferWithRange(icb, rng)
			enc.EndEncoding()
			cb.Commit()
			cb.WaitUntilCompleted()
			if profileForward {
				profForwardGPUSec += float64(cb.GPUEndTime() - cb.GPUStartTime())
			}
			copy(outputs[t], unsafe.Slice((*byte)(lastOut.Contents()), dModel*bf16Size))
		}
	})
	return outputs, nil
}
