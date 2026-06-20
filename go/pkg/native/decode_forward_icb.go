// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"slices"
	"unsafe"

	core "dappco.re/go"
	"github.com/tmc/apple/foundation"
	"github.com/tmc/apple/metal"
)

// decodeForwardICBCore is the backend-agnostic cache-grow ICB recorder + replay:
// it records the full N-layer decode stack (24 ops/layer) ONCE and replays it per
// token over a GROWING seq-major KV cache. The seven projections are the only ops
// that differ between a bf16 and a 4-bit layer, so they're recorded through the
// `recordProj` closure (gemv or qmv); everything else — rms, rope, sdpa, the gelu
// chain, the residual adds, the cache layout, the per-token rebind, the optimize
// pass and the single-submit replay — is shared here.
//
// recordProj(li, c, vec, out, outOff, p) records projection p of layer li at the
// already-barriered command c (reading vec, writing out at outOff bytes); vOutBind
// is the projection output's bind index (gemv 3 / qmv 4), re-set per token for the
// V cache row. projResident lists the backend's weight + scalar buffers so they're
// made resident. anwBufs/mnwBufs are the per-layer bf16 norm buffers (norms aren't
// quantised); kCaches/vCaches are the per-layer growing caches the caller created.
//
// The crux a fixed ICB can't express directly is the cache WRITE row, which
// advances every token. The lever (TestICBRebindOffset / TestQMVICB): an ICB
// command's bindings are recorded once, but re-setting ONE buffer offset between
// replays is cheap and takes effect. So per token only offBuf, nBuf and each
// layer's two cache-write offsets (K-RoPE @ idx 1, V projection @ vOutBind) change.
func decodeForwardICBCore(
	inputs [][]byte,
	anwBufs, mnwBufs, kCaches, vCaches, projResident []metal.MTLBuffer,
	recordProj func(li int, c metal.MTLIndirectComputeCommand, vec, out metal.MTLBuffer, outOff uint, p projIndex),
	vOutBind uint,
	dModel, nHeads, nKVHeads, headDim, dFF, maxLen int,
	base, scale, eps float32,
) ([][]byte, error) {
	nLayers, T := len(anwBufs), len(inputs)
	qDim, kvDim := nHeads*headDim, nKVHeads*headDim

	// shared (non-projection) ICB-capable pipelines
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
	var geluICBPSO metal.MTLComputePipelineState
	if gpuHasGeluKernel() {
		if geluICBPSO, err = geluPipelineICB(); err != nil {
			return nil, err
		}
	}

	outputs := make([][]byte, T)
	for i := range outputs {
		outputs[i] = make([]byte, dModel*bf16Size)
	}
	withAutoreleasePool(func() {
		// shared scratch + gelu constants + residual ping-pong
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
		ping := [2]metal.MTLBuffer{scratchBF16(dModel), scratchBF16(dModel)}
		hBufs := make([]metal.MTLBuffer, nLayers)
		for i := range hBufs {
			hBufs[i] = scratchBF16(dModel)
		}

		// shared (non-projection) scalar buffers; offBuf + nBuf bumped per token
		off := int32(0)
		offBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&off), 4, metal.MTLResourceStorageModeShared)
		nWin := int32(1)
		nBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&nWin), 4, metal.MTLResourceStorageModeShared)
		epsBuf, axisBuf, wsBuf := scalarF32(eps), scalarI32(int32(dModel)), scalarI32(1)
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
			ropeScaleB, ropeMatB, ropeBaseB, gqaB, khsB, kssB, vhsB, vssB, sdpaScaleB, addModelB, cntFFB, tanhCntB,
		}
		// reserve the upper-bound capacity for the appends that follow (projResident + 5 per-layer
		// buffer slices = 12 buffers/layer + the 19 projResident scalars) so the resident slice never
		// geometrically regrows. Grow changes capacity only — contents and kernel bindings unchanged.
		resident = slices.Grow(resident, 12*nLayers+20)
		resident = append(resident, projResident...)
		resident = append(resident, anwBufs...)
		resident = append(resident, mnwBufs...)
		resident = append(resident, kCaches...)
		resident = append(resident, vCaches...)
		resident = append(resident, hBufs...)

		opsPerLayer := 24
		if gpuHasGeluKernel() { // fused gelu is 1 command vs the composed chain's 10
			opsPerLayer = 15
		}
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

		// per-layer cache-write commands whose OUTPUT offset is re-set per token
		kRopeCmd := make([]metal.MTLIndirectComputeCommand, nLayers)
		vCmd := make([]metal.MTLIndirectComputeCommand, nLayers)

		for li := 0; li < nLayers; li++ {
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
			setRMS(cmd(0), inBuf, anwBufs[li], normed)
			recordProj(li, cmd(1), normed, q, 0, projQ) // Q
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
			recordProj(li, cmd(3), normed, kProj, 0, projK) // K -> kProj
			// 4: rope K -> kCache @ row pos  (OUTPUT OFFSET re-set per token)
			c = cmd(4)
			c.SetComputePipelineState(ropePSO)
			c.SetKernelBufferOffsetAtIndex(kProj, 0, 0)
			c.SetKernelBufferOffsetAtIndex(kCaches[li], 0, 1)
			c.SetKernelBufferOffsetAtIndex(offBuf, 0, 2)
			c.SetKernelBufferOffsetAtIndex(ropeScaleB, 0, 3)
			c.SetKernelBufferOffsetAtIndex(ropeMatB, 0, 4)
			c.SetKernelBufferOffsetAtIndex(ropeBaseB, 0, 10)
			ropeKDim0 := uint(headDim / 2)
			c.ConcurrentDispatchThreadsThreadsPerThreadgroup(metal.MTLSize{Width: ropeKDim0, Height: uint(nKVHeads), Depth: 1}, metal.MTLSize{Width: ropeKDim0, Height: 1, Depth: 1})
			kRopeCmd[li] = c
			// 5: V projection -> vCache @ row pos  (OUTPUT OFFSET re-set per token)
			cv := cmd(5)
			recordProj(li, cv, normed, vCaches[li], 0, projV)
			vCmd[li] = cv
			// 6: sdpa over the grown window (N from nBuf; seq-major strides)
			c = cmd(6)
			c.SetComputePipelineState(sdpaPSO)
			c.SetKernelBufferOffsetAtIndex(qr, 0, 0)
			c.SetKernelBufferOffsetAtIndex(kCaches[li], 0, 1)
			c.SetKernelBufferOffsetAtIndex(vCaches[li], 0, 2)
			c.SetKernelBufferOffsetAtIndex(attn, 0, 3)
			c.SetKernelBufferOffsetAtIndex(gqaB, 0, 4)
			c.SetKernelBufferOffsetAtIndex(nBuf, 0, 5)
			c.SetKernelBufferOffsetAtIndex(khsB, 0, 6)
			c.SetKernelBufferOffsetAtIndex(kssB, 0, 7)
			c.SetKernelBufferOffsetAtIndex(vhsB, 0, 8)
			c.SetKernelBufferOffsetAtIndex(vssB, 0, 9)
			c.SetKernelBufferOffsetAtIndex(sdpaScaleB, 0, 10)
			c.ConcurrentDispatchThreadgroupsThreadsPerThreadgroup(metal.MTLSize{Width: uint(nHeads), Height: 1, Depth: 1}, metal.MTLSize{Width: 1024, Height: 1, Depth: 1})
			recordProj(li, cmd(7), attn, attnOut, 0, projO) // Wo
			setBin(cmd(8), addPSO, inBuf, attnOut, hBuf, addModelB, dModel)

			// --- MLP half (ops 9-23) ---
			setRMS(cmd(9), hBuf, mnwBufs[li], mlpNormed)
			recordProj(li, cmd(10), mlpNormed, gate, 0, projGate)
			recordProj(li, cmd(11), mlpNormed, up, 0, projUp)
			dpIdx := 22 // down-proj op index — follows the composed gelu (cmd 12-21)
			if gpuHasGeluKernel() {
				cg := cmd(12) // fused gelu(gate)·up — one command (cntFFB = dFF as the n buffer)
				cg.SetComputePipelineState(geluICBPSO)
				cg.SetKernelBufferOffsetAtIndex(gate, 0, 0)
				cg.SetKernelBufferOffsetAtIndex(up, 0, 1)
				cg.SetKernelBufferOffsetAtIndex(gated, 0, 2)
				cg.SetKernelBufferOffsetAtIndex(cntFFB, 0, 3)
				cg.ConcurrentDispatchThreadsThreadsPerThreadgroup(metal.MTLSize{Width: uint(dFF), Height: 1, Depth: 1}, metal.MTLSize{Width: elemGroup(dFF), Height: 1, Depth: 1})
				dpIdx = 13
			} else {
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
			}
			recordProj(li, cmd(dpIdx), gated, down, 0, projDown) // Wdown
			setBin(cmd(dpIdx+1), addPSO, hBuf, down, outBuf, addModelB, dModel)
		}

		lastOut := ping[nLayers%2] // residual stream output after N ping-pong swaps
		residentRes := make([]metal.MTLResource, len(resident))
		for i, b := range resident {
			residentRes[i] = b
		}
		rng := foundation.NSRange{Location: 0, Length: uint(total)}

		// optimize the recorded ICB once (offset-only rebinds after don't re-optimize)
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
				kRopeCmd[li].SetKernelBufferOffsetAtIndex(kCaches[li], rowOff, 1)
				vCmd[li].SetKernelBufferOffsetAtIndex(vCaches[li], rowOff, vOutBind)
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

// DecodeForwardICB is the bf16 cache-grow ICB: it builds a gemv recorder + the
// per-layer weight/cache buffers and runs the shared decodeForwardICBCore. Same
// signature/semantics as DecodeForward; byte-for-byte equal to it (gated). All bf16.
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

	// gemv ICB pipelines, one per distinct tile shape
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
		// presized to the upper bound (every layer's 7 projection buffers, plus the 19 trailing
		// scalar buffers) so the per-forward build never geometrically regrows its backing array.
		// Byte-identical.
		projResident := make([]metal.MTLBuffer, 0, nLayers*7+19)
		for li := range layers {
			w := layers[li]
			anwBufs[li] = sharedBytes(w.AttnNormW)
			mnwBufs[li] = sharedBytes(w.MLPNormW)
			kCaches[li] = device.NewBufferWithLengthOptions(cacheBytes, metal.MTLResourceStorageModeShared)
			vCaches[li] = device.NewBufferWithLengthOptions(cacheBytes, metal.MTLResourceStorageModeShared)
			lb[li] = lw{sharedBytes(w.WQ), sharedBytes(w.WK), sharedBytes(w.WV), sharedBytes(w.WO), sharedBytes(w.WGate), sharedBytes(w.WUp), sharedBytes(w.WDown)}
			projResident = append(projResident, lb[li].wq, lb[li].wk, lb[li].wv, lb[li].wo, lb[li].wg, lb[li].wu, lb[li].wd)
		}
		// gemv scalar params (shared across layers)
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
		outputs, coreErr = decodeForwardICBCore(inputs, anwBufs, mnwBufs, kCaches, vCaches, projResident, recordProj, 3, dModel, nHeads, nKVHeads, headDim, dFF, maxLen, base, scale, eps)
	})
	if coreErr != nil {
		return nil, coreErr
	}
	return outputs, nil
}
