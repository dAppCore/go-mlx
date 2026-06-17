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

// DecodeLayerICB records a full gemma transformer decode layer — the same 21-op
// sequence as DecodeLayer (attention half then MLP half, both residuals) — once
// into an indirect command buffer, then replays it `replays` times. This is the
// full-layer encode-bypass: a decode step's command sequence is fixed across
// tokens, so recording it once skips the per-token host re-encode of all 21 ops.
//
// It extends AttentionBlockICB (ops 0-5) with the MLP block (ops 6-20). The ICB
// rules from AttentionBlockICB hold throughout: every scalar param is a tiny
// persistent buffer (ICB commands cannot setBytes inline), each consumer command
// carries a SetBarrier so its read of a prior op's output is ordered, and the
// replay encoder marks every referenced buffer resident with UseResource. The
// gelu scalar operands are dense dFF-length bf16 constant buffers (bf16ConstBytes),
// exactly as DecodeLayer/MLPBlockBF16 build them, so the in-line gelu is identical.
//
// Same arguments and shapes as DecodeLayer; inputs/outputs raw bf16 bytes. With
// replays=1 it must equal DecodeLayer byte-for-byte — same kernels, same data,
// only the submission path differs.
func DecodeLayerICB(
	x, attnNormW, wQ, wO, kCache, vCache, mlpNormW, wGate, wUp, wDown []byte,
	dModel, nHeads, nKVHeads, headDim, kvLen, dFF int,
	base, scale float32, offset int, eps float32,
	replays int,
) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if replays < 1 {
		replays = 1
	}
	qDim := nHeads * headDim
	if len(x) != dModel*bf16Size || len(attnNormW) != dModel*bf16Size || len(mlpNormW) != dModel*bf16Size {
		return nil, core.NewError("native.DecodeLayerICB: x/attnNormW/mlpNormW must be dModel bf16 bytes")
	}
	if len(wQ) != qDim*dModel*bf16Size || len(wO) != dModel*qDim*bf16Size {
		return nil, core.NewError("native.DecodeLayerICB: wQ/wO size mismatch")
	}
	if len(wGate) != dFF*dModel*bf16Size || len(wUp) != dFF*dModel*bf16Size || len(wDown) != dModel*dFF*bf16Size {
		return nil, core.NewError("native.DecodeLayerICB: MLP weight size mismatch")
	}
	if len(kCache) != nKVHeads*kvLen*headDim*bf16Size || len(vCache) != nKVHeads*kvLen*headDim*bf16Size {
		return nil, core.NewError("native.DecodeLayerICB: kCache/vCache size mismatch")
	}

	// ICB-capable pipelines. gemv tiles depend on (inDim, outDim) so there are
	// four distinct gemv PSOs: Q (dModel→qDim), O (qDim→dModel), gate/up
	// (dModel→dFF, shared) and down (dFF→dModel).
	rmsPSO, err := pipelineForICB("rmsbfloat16")
	if err != nil {
		return nil, err
	}
	bmQ, bnQ, smQ, snQ, tmQ, tnQ := gemvTiles(dModel, qDim)
	gemvQPSO, err := pipelineForICB(core.Sprintf("gemv_bfloat16_bm%d_bn%d_sm%d_sn%d_tm%d_tn%d_nc0_axpby0", bmQ, bnQ, smQ, snQ, tmQ, tnQ))
	if err != nil {
		return nil, err
	}
	bmO, bnO, smO, snO, tmO, tnO := gemvTiles(qDim, dModel)
	gemvOPSO, err := pipelineForICB(core.Sprintf("gemv_bfloat16_bm%d_bn%d_sm%d_sn%d_tm%d_tn%d_nc0_axpby0", bmO, bnO, smO, snO, tmO, tnO))
	if err != nil {
		return nil, err
	}
	bmF, bnF, smF, snF, tmF, tnF := gemvTiles(dModel, dFF)
	gemvFPSO, err := pipelineForICB(core.Sprintf("gemv_bfloat16_bm%d_bn%d_sm%d_sn%d_tm%d_tn%d_nc0_axpby0", bmF, bnF, smF, snF, tmF, tnF))
	if err != nil {
		return nil, err
	}
	bmD, bnD, smD, snD, tmD, tnD := gemvTiles(dFF, dModel)
	gemvDPSO, err := pipelineForICB(core.Sprintf("gemv_bfloat16_bm%d_bn%d_sm%d_sn%d_tm%d_tn%d_nc0_axpby0", bmD, bnD, smD, snD, tmD, tnD))
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

	out := make([]byte, dModel*bf16Size)
	withAutoreleasePool(func() {
		// --- data buffers ---
		xBuf := sharedBytes(x)
		anwBuf, mnwBuf := sharedBytes(attnNormW), sharedBytes(mlpNormW)
		wqBuf, woBuf := sharedBytes(wQ), sharedBytes(wO)
		kBuf, vBuf := sharedBytes(kCache), sharedBytes(vCache)
		wgBuf, wuBuf, wdBuf := sharedBytes(wGate), sharedBytes(wUp), sharedBytes(wDown)
		// gelu scalar operands as dense dFF-length bf16 constant buffers.
		c044 := sharedBytes(bf16ConstBytes(dFF, 0.044715))
		c079 := sharedBytes(bf16ConstBytes(dFF, 0.7978845608028654))
		c1c := sharedBytes(bf16ConstBytes(dFF, 1.0))
		c05 := sharedBytes(bf16ConstBytes(dFF, 0.5))

		// attention intermediates
		attnNormed := scratchBF16(dModel)
		q, qr, attn := scratchBF16(qDim), scratchBF16(qDim), scratchBF16(qDim)
		attnOut, h := scratchBF16(dModel), scratchBF16(dModel)
		// mlp intermediates
		mlpNormed := scratchBF16(dModel)
		gate, up := scratchBF16(dFF), scratchBF16(dFF)
		x2, x3, x3s, inner := scratchBF16(dFF), scratchBF16(dFF), scratchBF16(dFF), scratchBF16(dFF)
		scaled, tnh, onePlus, halfG := scratchBF16(dFF), scratchBF16(dFF), scratchBF16(dFF), scratchBF16(dFF)
		gelu, gated := scratchBF16(dFF), scratchBF16(dFF)
		down, outBuf := scratchBF16(dModel), scratchBF16(dModel)

		// --- scalar buffers ---
		off := int32(offset)
		offBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&off), 4, metal.MTLResourceStorageModeShared)
		epsBuf, axisBuf, wsBuf := scalarF32(eps), scalarI32(int32(dModel)), scalarI32(1)
		// gemv scalars (inDim, outDim, ld vary per op; batch scalars are shared)
		qInB, qOutB, qLdB := scalarI32(int32(dModel)), scalarI32(int32(qDim)), scalarI32(int32(dModel))
		oInB, oOutB, oLdB := scalarI32(int32(qDim)), scalarI32(int32(dModel)), scalarI32(int32(qDim))
		fInB, fOutB, fLdB := scalarI32(int32(dModel)), scalarI32(int32(dFF)), scalarI32(int32(dModel))
		dInB, dOutB, dLdB := scalarI32(int32(dFF)), scalarI32(int32(dModel)), scalarI32(int32(dFF))
		bndB, bshB, vsB, msB := scalarI32(1), scalarI32(1), scalarI64(0), scalarI64(0)
		// rope scalars
		ropeScaleB := scalarF32(scale)
		ropeMatB := scalarI64(int64(headDim))
		ropeBaseB := scalarF32(float32(math.Log2(float64(base))))
		// sdpa scalars
		gqaB, nB := scalarI32(int32(nHeads/nKVHeads)), scalarI32(int32(kvLen))
		khsB, kssB := scalarI64(int64(kvLen*headDim)), scalarI64(int64(headDim))
		vhsB, vssB := scalarI64(int64(kvLen*headDim)), scalarI64(int64(headDim))
		sdpaScaleB := scalarF32(scale)
		// element-wise counts: dModel for the two residual adds, dFF for the
		// MLP element-wise chain. tanh's count is bound as a buffer at index 2
		// (the encode path uses setBytes there; the ICB cannot, so it is a buffer).
		addModelB := scalarI32(int32(dModel))
		cntFFB := scalarI32(int32(dFF))
		tanhCntB := scalarI32(int32(dFF))

		resident := []metal.MTLBuffer{
			xBuf, anwBuf, mnwBuf, wqBuf, woBuf, kBuf, vBuf, wgBuf, wuBuf, wdBuf,
			c044, c079, c1c, c05,
			attnNormed, q, qr, attn, attnOut, h,
			mlpNormed, gate, up, x2, x3, x3s, inner, scaled, tnh, onePlus, halfG, gelu, gated, down, outBuf,
			offBuf, epsBuf, axisBuf, wsBuf,
			qInB, qOutB, qLdB, oInB, oOutB, oLdB, fInB, fOutB, fLdB, dInB, dOutB, dLdB,
			bndB, bshB, vsB, msB,
			ropeScaleB, ropeMatB, ropeBaseB,
			gqaB, nB, khsB, kssB, vhsB, vssB, sdpaScaleB,
			addModelB, cntFFB, tanhCntB,
		}

		icbDesc := metal.NewMTLIndirectCommandBufferDescriptor()
		icbDesc.SetCommandTypes(metal.MTLIndirectCommandTypeConcurrentDispatch)
		icbDesc.SetInheritBuffers(false)
		icbDesc.SetInheritPipelineState(false)
		icbDesc.SetMaxKernelBufferBindCount(16)
		icb := device.NewIndirectCommandBufferWithDescriptorMaxCommandCountOptions(icbDesc, 21, metal.MTLResourceStorageModeShared)

		rmsTG := uint(rmsSimdSize * ((((dModel + rmsNReads - 1) / rmsNReads) + rmsSimdSize - 1) / rmsSimdSize))
		gemvGrid := func(outDim, bm, sm, tm int) uint { return uint((outDim + bm*sm*tm - 1) / (bm * sm * tm)) }
		elemGroup := func(n int) uint {
			g := uint(256)
			if uint(n) < g {
				g = uint(n)
			}
			return g
		}

		// helper closures so each op's binding matches its encode-form exactly.
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
		setGemv := func(c metal.MTLIndirectComputeCommand, pso metal.MTLComputePipelineState, mat, vec, o metal.MTLBuffer, inB, outB, ldB metal.MTLBuffer, outDim, bm, bn, sm, tm int) {
			c.SetComputePipelineState(pso)
			c.SetKernelBufferOffsetAtIndex(mat, 0, 0)
			c.SetKernelBufferOffsetAtIndex(vec, 0, 1)
			c.SetKernelBufferOffsetAtIndex(o, 0, 3)
			c.SetKernelBufferOffsetAtIndex(inB, 0, 4)
			c.SetKernelBufferOffsetAtIndex(outB, 0, 5)
			c.SetKernelBufferOffsetAtIndex(ldB, 0, 6)
			c.SetKernelBufferOffsetAtIndex(bndB, 0, 9)
			c.SetKernelBufferOffsetAtIndex(bshB, 0, 10)
			c.SetKernelBufferOffsetAtIndex(vsB, 0, 11)
			c.SetKernelBufferOffsetAtIndex(msB, 0, 12)
			c.ConcurrentDispatchThreadgroupsThreadsPerThreadgroup(metal.MTLSize{Width: gemvGrid(outDim, bm, sm, tm), Height: 1, Depth: 1}, metal.MTLSize{Width: 32, Height: uint(bn), Depth: uint(bm)})
		}
		setBinary := func(c metal.MTLIndirectComputeCommand, pso metal.MTLComputePipelineState, a, b, o, cntB metal.MTLBuffer, n int) {
			c.SetComputePipelineState(pso)
			c.SetKernelBufferOffsetAtIndex(a, 0, 0)
			c.SetKernelBufferOffsetAtIndex(b, 0, 1)
			c.SetKernelBufferOffsetAtIndex(o, 0, 2)
			c.SetKernelBufferOffsetAtIndex(cntB, 0, 3)
			c.ConcurrentDispatchThreadsThreadsPerThreadgroup(metal.MTLSize{Width: uint(n), Height: 1, Depth: 1}, metal.MTLSize{Width: elemGroup(n), Height: 1, Depth: 1})
		}

		// ===== attention half (ops 0-5): h = x + Wo·sdpa(rope(Wq·rms(x))) =====
		// 0: rms x -> attnNormed
		setRMS(icb.IndirectComputeCommandAtIndex(0), xBuf, anwBuf, attnNormed)

		// 1: gemv Wq @ attnNormed -> q  (dModel -> qDim)
		c := icb.IndirectComputeCommandAtIndex(1)
		c.SetBarrier()
		setGemv(c, gemvQPSO, wqBuf, attnNormed, q, qInB, qOutB, qLdB, qDim, bmQ, bnQ, smQ, tmQ)

		// 2: rope q -> qr
		c = icb.IndirectComputeCommandAtIndex(2)
		c.SetBarrier()
		c.SetComputePipelineState(ropePSO)
		c.SetKernelBufferOffsetAtIndex(q, 0, 0)
		c.SetKernelBufferOffsetAtIndex(qr, 0, 1)
		c.SetKernelBufferOffsetAtIndex(offBuf, 0, 2)
		c.SetKernelBufferOffsetAtIndex(ropeScaleB, 0, 3)
		c.SetKernelBufferOffsetAtIndex(ropeMatB, 0, 4)
		c.SetKernelBufferOffsetAtIndex(ropeBaseB, 0, 10)
		ropeDim0 := uint(headDim / 2)
		c.ConcurrentDispatchThreadsThreadsPerThreadgroup(metal.MTLSize{Width: ropeDim0, Height: uint(nHeads), Depth: 1}, metal.MTLSize{Width: ropeDim0, Height: 1, Depth: 1})

		// 3: sdpa qr, k, v -> attn
		c = icb.IndirectComputeCommandAtIndex(3)
		c.SetBarrier()
		c.SetComputePipelineState(sdpaPSO)
		c.SetKernelBufferOffsetAtIndex(qr, 0, 0)
		c.SetKernelBufferOffsetAtIndex(kBuf, 0, 1)
		c.SetKernelBufferOffsetAtIndex(vBuf, 0, 2)
		c.SetKernelBufferOffsetAtIndex(attn, 0, 3)
		c.SetKernelBufferOffsetAtIndex(gqaB, 0, 4)
		c.SetKernelBufferOffsetAtIndex(nB, 0, 5)
		c.SetKernelBufferOffsetAtIndex(khsB, 0, 6)
		c.SetKernelBufferOffsetAtIndex(kssB, 0, 7)
		c.SetKernelBufferOffsetAtIndex(vhsB, 0, 8)
		c.SetKernelBufferOffsetAtIndex(vssB, 0, 9)
		c.SetKernelBufferOffsetAtIndex(sdpaScaleB, 0, 10)
		c.ConcurrentDispatchThreadgroupsThreadsPerThreadgroup(metal.MTLSize{Width: uint(nHeads), Height: 1, Depth: 1}, metal.MTLSize{Width: 1024, Height: 1, Depth: 1})

		// 4: gemv Wo @ attn -> attnOut  (qDim -> dModel)
		c = icb.IndirectComputeCommandAtIndex(4)
		c.SetBarrier()
		setGemv(c, gemvOPSO, woBuf, attn, attnOut, oInB, oOutB, oLdB, dModel, bmO, bnO, smO, tmO)

		// 5: add x + attnOut -> h
		c = icb.IndirectComputeCommandAtIndex(5)
		c.SetBarrier()
		setBinary(c, addPSO, xBuf, attnOut, h, addModelB, dModel)

		// ===== MLP half (ops 6-20): out = h + Wdown·(gelu(Wgate·rms(h))·(Wup·rms(h))) =====
		// 6: rms h -> mlpNormed
		c = icb.IndirectComputeCommandAtIndex(6)
		c.SetBarrier()
		setRMS(c, h, mnwBuf, mlpNormed)

		// 7: gemv Wgate @ mlpNormed -> gate  (dModel -> dFF)
		c = icb.IndirectComputeCommandAtIndex(7)
		c.SetBarrier()
		setGemv(c, gemvFPSO, wgBuf, mlpNormed, gate, fInB, fOutB, fLdB, dFF, bmF, bnF, smF, tmF)

		// 8: gemv Wup @ mlpNormed -> up  (dModel -> dFF)
		c = icb.IndirectComputeCommandAtIndex(8)
		c.SetBarrier()
		setGemv(c, gemvFPSO, wuBuf, mlpNormed, up, fInB, fOutB, fLdB, dFF, bmF, bnF, smF, tmF)

		// gelu_approx(gate): x2=g·g; x3=x2·g; x3s=0.044715·x3; inner=g+x3s;
		//                    scaled=0.7978…·inner; tnh=tanh(scaled);
		//                    onePlus=tnh+1; halfG=0.5·g; gelu=halfG·onePlus
		// 9: mul gate·gate -> x2
		c = icb.IndirectComputeCommandAtIndex(9)
		c.SetBarrier()
		setBinary(c, mulPSO, gate, gate, x2, cntFFB, dFF)

		// 10: mul x2·gate -> x3
		c = icb.IndirectComputeCommandAtIndex(10)
		c.SetBarrier()
		setBinary(c, mulPSO, x2, gate, x3, cntFFB, dFF)

		// 11: mul x3·c044 -> x3s
		c = icb.IndirectComputeCommandAtIndex(11)
		c.SetBarrier()
		setBinary(c, mulPSO, x3, c044, x3s, cntFFB, dFF)

		// 12: add gate+x3s -> inner
		c = icb.IndirectComputeCommandAtIndex(12)
		c.SetBarrier()
		setBinary(c, addPSO, gate, x3s, inner, cntFFB, dFF)

		// 13: mul inner·c079 -> scaled
		c = icb.IndirectComputeCommandAtIndex(13)
		c.SetBarrier()
		setBinary(c, mulPSO, inner, c079, scaled, cntFFB, dFF)

		// 14: tanh scaled -> tnh  (count buffer at index 2)
		c = icb.IndirectComputeCommandAtIndex(14)
		c.SetBarrier()
		c.SetComputePipelineState(tanhPSO)
		c.SetKernelBufferOffsetAtIndex(scaled, 0, 0)
		c.SetKernelBufferOffsetAtIndex(tnh, 0, 1)
		c.SetKernelBufferOffsetAtIndex(tanhCntB, 0, 2)
		c.ConcurrentDispatchThreadsThreadsPerThreadgroup(metal.MTLSize{Width: uint(dFF), Height: 1, Depth: 1}, metal.MTLSize{Width: elemGroup(dFF), Height: 1, Depth: 1})

		// 15: add tnh+c1 -> onePlus
		c = icb.IndirectComputeCommandAtIndex(15)
		c.SetBarrier()
		setBinary(c, addPSO, tnh, c1c, onePlus, cntFFB, dFF)

		// 16: mul gate·c05 -> halfG
		c = icb.IndirectComputeCommandAtIndex(16)
		c.SetBarrier()
		setBinary(c, mulPSO, gate, c05, halfG, cntFFB, dFF)

		// 17: mul halfG·onePlus -> gelu
		c = icb.IndirectComputeCommandAtIndex(17)
		c.SetBarrier()
		setBinary(c, mulPSO, halfG, onePlus, gelu, cntFFB, dFF)

		// 18: mul gelu·up -> gated
		c = icb.IndirectComputeCommandAtIndex(18)
		c.SetBarrier()
		setBinary(c, mulPSO, gelu, up, gated, cntFFB, dFF)

		// 19: gemv Wdown @ gated -> down  (dFF -> dModel)
		c = icb.IndirectComputeCommandAtIndex(19)
		c.SetBarrier()
		setGemv(c, gemvDPSO, wdBuf, gated, down, dInB, dOutB, dLdB, dModel, bmD, bnD, smD, tmD)

		// 20: add h + down -> outBuf  (residual)
		c = icb.IndirectComputeCommandAtIndex(20)
		c.SetBarrier()
		setBinary(c, addPSO, h, down, outBuf, addModelB, dModel)

		rng := foundation.NSRange{Location: 0, Length: 21}
		for r := 0; r < replays; r++ {
			cb := queue.CommandBuffer()
			enc := cb.ComputeCommandEncoder()
			for _, b := range resident {
				enc.UseResourceUsage(b, metal.MTLResourceUsageRead|metal.MTLResourceUsageWrite)
			}
			enc.ExecuteCommandsInBufferWithRange(icb, rng)
			enc.EndEncoding()
			cb.Commit()
			cb.WaitUntilCompleted()
		}
		copy(out, unsafe.Slice((*byte)(outBuf.Contents()), len(out)))
	})
	return out, nil
}

// DecodeTokenICB records a full nLayers-deep decode TOKEN — nLayers copies of the
// 21-op DecodeLayer sequence chained through a residual-stream ping-pong — once
// into one indirect command buffer (21*nLayers commands), then replays the whole
// stack with a SINGLE executeCommandsInBuffer + commit + wait per token, `replays`
// times. This is the per-token analogue of DecodeLayerICB and the un-diluted
// encode-bypass headline: a real decode step submits its entire layer stack at
// once, so the one commit+wait is amortised across all the layers and the A/B vs
// tokenReEncode(reps) reflects the true per-token host saving (DecodeLayerICB paid
// a commit+wait per layer, diluting the ratio with GPU+submit time).
//
// Same per-layer ICB rules as DecodeLayerICB hold: every scalar is a persistent
// buffer, every command carries a SetBarrier so its reads of prior writes are
// ordered (here that also serialises layer L+1's input rms after layer L's output
// add) EXCEPT the very first command, and the replay encoder marks every buffer
// resident. Layers share weights/scratch/KV (the host encode cost per command is
// independent of which buffer it binds; keeps it AX-11-light); only the residual
// stream ping-pongs between two model-dim buffers. With nLayers=1, replays=1 it is
// DecodeLayerICB; chained it equals nLayers applications of DecodeLayer — both
// gated byte-for-byte in the tests. Inputs/outputs raw bf16 bytes.
func DecodeTokenICB(
	x, attnNormW, wQ, wO, kCache, vCache, mlpNormW, wGate, wUp, wDown []byte,
	dModel, nHeads, nKVHeads, headDim, kvLen, dFF, nLayers int,
	base, scale float32, offset int, eps float32,
	replays int,
) ([]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if nLayers < 1 {
		nLayers = 1
	}
	if replays < 1 {
		replays = 1
	}
	qDim := nHeads * headDim
	if len(x) != dModel*bf16Size || len(attnNormW) != dModel*bf16Size || len(mlpNormW) != dModel*bf16Size {
		return nil, core.NewError("native.DecodeTokenICB: x/attnNormW/mlpNormW must be dModel bf16 bytes")
	}
	if len(wQ) != qDim*dModel*bf16Size || len(wO) != dModel*qDim*bf16Size {
		return nil, core.NewError("native.DecodeTokenICB: wQ/wO size mismatch")
	}
	if len(wGate) != dFF*dModel*bf16Size || len(wUp) != dFF*dModel*bf16Size || len(wDown) != dModel*dFF*bf16Size {
		return nil, core.NewError("native.DecodeTokenICB: MLP weight size mismatch")
	}
	if len(kCache) != nKVHeads*kvLen*headDim*bf16Size || len(vCache) != nKVHeads*kvLen*headDim*bf16Size {
		return nil, core.NewError("native.DecodeTokenICB: kCache/vCache size mismatch")
	}

	rmsPSO, err := pipelineForICB("rmsbfloat16")
	if err != nil {
		return nil, err
	}
	bmQ, bnQ, smQ, snQ, tmQ, tnQ := gemvTiles(dModel, qDim)
	gemvQPSO, err := pipelineForICB(core.Sprintf("gemv_bfloat16_bm%d_bn%d_sm%d_sn%d_tm%d_tn%d_nc0_axpby0", bmQ, bnQ, smQ, snQ, tmQ, tnQ))
	if err != nil {
		return nil, err
	}
	bmO, bnO, smO, snO, tmO, tnO := gemvTiles(qDim, dModel)
	gemvOPSO, err := pipelineForICB(core.Sprintf("gemv_bfloat16_bm%d_bn%d_sm%d_sn%d_tm%d_tn%d_nc0_axpby0", bmO, bnO, smO, snO, tmO, tnO))
	if err != nil {
		return nil, err
	}
	bmF, bnF, smF, snF, tmF, tnF := gemvTiles(dModel, dFF)
	gemvFPSO, err := pipelineForICB(core.Sprintf("gemv_bfloat16_bm%d_bn%d_sm%d_sn%d_tm%d_tn%d_nc0_axpby0", bmF, bnF, smF, snF, tmF, tnF))
	if err != nil {
		return nil, err
	}
	bmD, bnD, smD, snD, tmD, tnD := gemvTiles(dFF, dModel)
	gemvDPSO, err := pipelineForICB(core.Sprintf("gemv_bfloat16_bm%d_bn%d_sm%d_sn%d_tm%d_tn%d_nc0_axpby0", bmD, bnD, smD, snD, tmD, tnD))
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

	out := make([]byte, dModel*bf16Size)
	withAutoreleasePool(func() {
		// --- weight / KV / gelu-const data buffers (shared across layers) ---
		anwBuf, mnwBuf := sharedBytes(attnNormW), sharedBytes(mlpNormW)
		wqBuf, woBuf := sharedBytes(wQ), sharedBytes(wO)
		kBuf, vBuf := sharedBytes(kCache), sharedBytes(vCache)
		wgBuf, wuBuf, wdBuf := sharedBytes(wGate), sharedBytes(wUp), sharedBytes(wDown)
		c044 := sharedBytes(bf16ConstBytes(dFF, 0.044715))
		c079 := sharedBytes(bf16ConstBytes(dFF, 0.7978845608028654))
		c1c := sharedBytes(bf16ConstBytes(dFF, 1.0))
		c05 := sharedBytes(bf16ConstBytes(dFF, 0.5))

		// residual-stream ping-pong: xA seeded with the token input, xB scratch.
		xA, xB := sharedBytes(x), scratchBF16(dModel)

		// shared per-layer intermediates
		attnNormed := scratchBF16(dModel)
		q, qr, attn := scratchBF16(qDim), scratchBF16(qDim), scratchBF16(qDim)
		attnOut, h := scratchBF16(dModel), scratchBF16(dModel)
		mlpNormed := scratchBF16(dModel)
		gate, up := scratchBF16(dFF), scratchBF16(dFF)
		x2, x3, x3s, inner := scratchBF16(dFF), scratchBF16(dFF), scratchBF16(dFF), scratchBF16(dFF)
		scaled, tnh, onePlus, halfG := scratchBF16(dFF), scratchBF16(dFF), scratchBF16(dFF), scratchBF16(dFF)
		gelu, gated := scratchBF16(dFF), scratchBF16(dFF)
		down := scratchBF16(dModel)

		// --- scalar buffers (shared) ---
		off := int32(offset)
		offBuf := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&off), 4, metal.MTLResourceStorageModeShared)
		epsBuf, axisBuf, wsBuf := scalarF32(eps), scalarI32(int32(dModel)), scalarI32(1)
		qInB, qOutB, qLdB := scalarI32(int32(dModel)), scalarI32(int32(qDim)), scalarI32(int32(dModel))
		oInB, oOutB, oLdB := scalarI32(int32(qDim)), scalarI32(int32(dModel)), scalarI32(int32(qDim))
		fInB, fOutB, fLdB := scalarI32(int32(dModel)), scalarI32(int32(dFF)), scalarI32(int32(dModel))
		dInB, dOutB, dLdB := scalarI32(int32(dFF)), scalarI32(int32(dModel)), scalarI32(int32(dFF))
		bndB, bshB, vsB, msB := scalarI32(1), scalarI32(1), scalarI64(0), scalarI64(0)
		ropeScaleB := scalarF32(scale)
		ropeMatB := scalarI64(int64(headDim))
		ropeBaseB := scalarF32(float32(math.Log2(float64(base))))
		gqaB, nB := scalarI32(int32(nHeads/nKVHeads)), scalarI32(int32(kvLen))
		khsB, kssB := scalarI64(int64(kvLen*headDim)), scalarI64(int64(headDim))
		vhsB, vssB := scalarI64(int64(kvLen*headDim)), scalarI64(int64(headDim))
		sdpaScaleB := scalarF32(scale)
		addModelB := scalarI32(int32(dModel))
		cntFFB := scalarI32(int32(dFF))
		tanhCntB := scalarI32(int32(dFF))

		resident := []metal.MTLBuffer{
			xA, xB, anwBuf, mnwBuf, wqBuf, woBuf, kBuf, vBuf, wgBuf, wuBuf, wdBuf,
			c044, c079, c1c, c05,
			attnNormed, q, qr, attn, attnOut, h,
			mlpNormed, gate, up, x2, x3, x3s, inner, scaled, tnh, onePlus, halfG, gelu, gated, down,
			offBuf, epsBuf, axisBuf, wsBuf,
			qInB, qOutB, qLdB, oInB, oOutB, oLdB, fInB, fOutB, fLdB, dInB, dOutB, dLdB,
			bndB, bshB, vsB, msB,
			ropeScaleB, ropeMatB, ropeBaseB,
			gqaB, nB, khsB, kssB, vhsB, vssB, sdpaScaleB,
			addModelB, cntFFB, tanhCntB,
		}

		total := 21 * nLayers
		icbDesc := metal.NewMTLIndirectCommandBufferDescriptor()
		icbDesc.SetCommandTypes(metal.MTLIndirectCommandTypeConcurrentDispatch)
		icbDesc.SetInheritBuffers(false)
		icbDesc.SetInheritPipelineState(false)
		icbDesc.SetMaxKernelBufferBindCount(16)
		icb := device.NewIndirectCommandBufferWithDescriptorMaxCommandCountOptions(icbDesc, uint(total), metal.MTLResourceStorageModeShared)

		rmsTG := uint(rmsSimdSize * ((((dModel + rmsNReads - 1) / rmsNReads) + rmsSimdSize - 1) / rmsSimdSize))
		gemvGrid := func(outDim, bm, sm, tm int) uint { return uint((outDim + bm*sm*tm - 1) / (bm * sm * tm)) }
		elemGroup := func(n int) uint {
			g := uint(256)
			if uint(n) < g {
				g = uint(n)
			}
			return g
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
		setGemv := func(c metal.MTLIndirectComputeCommand, pso metal.MTLComputePipelineState, mat, vec, o metal.MTLBuffer, inB, outB, ldB metal.MTLBuffer, outDim, bm, bn, sm, tm int) {
			c.SetComputePipelineState(pso)
			c.SetKernelBufferOffsetAtIndex(mat, 0, 0)
			c.SetKernelBufferOffsetAtIndex(vec, 0, 1)
			c.SetKernelBufferOffsetAtIndex(o, 0, 3)
			c.SetKernelBufferOffsetAtIndex(inB, 0, 4)
			c.SetKernelBufferOffsetAtIndex(outB, 0, 5)
			c.SetKernelBufferOffsetAtIndex(ldB, 0, 6)
			c.SetKernelBufferOffsetAtIndex(bndB, 0, 9)
			c.SetKernelBufferOffsetAtIndex(bshB, 0, 10)
			c.SetKernelBufferOffsetAtIndex(vsB, 0, 11)
			c.SetKernelBufferOffsetAtIndex(msB, 0, 12)
			c.ConcurrentDispatchThreadgroupsThreadsPerThreadgroup(metal.MTLSize{Width: gemvGrid(outDim, bm, sm, tm), Height: 1, Depth: 1}, metal.MTLSize{Width: 32, Height: uint(bn), Depth: uint(bm)})
		}
		setBinary := func(c metal.MTLIndirectComputeCommand, pso metal.MTLComputePipelineState, a, b, o, cntB metal.MTLBuffer, n int) {
			c.SetComputePipelineState(pso)
			c.SetKernelBufferOffsetAtIndex(a, 0, 0)
			c.SetKernelBufferOffsetAtIndex(b, 0, 1)
			c.SetKernelBufferOffsetAtIndex(o, 0, 2)
			c.SetKernelBufferOffsetAtIndex(cntB, 0, 3)
			c.ConcurrentDispatchThreadsThreadsPerThreadgroup(metal.MTLSize{Width: uint(n), Height: 1, Depth: 1}, metal.MTLSize{Width: elemGroup(n), Height: 1, Depth: 1})
		}

		// recordLayer writes the 21 commands of one layer at [base, base+20],
		// reading inBuf, writing outBuf. Every command barriers except global 0 —
		// which also orders this layer's input rms after the previous layer's
		// output add (the shared scratch is reused, so the stack must serialise).
		recordLayer := func(base int, inBuf, outBuf metal.MTLBuffer) {
			cmd := func(op int) metal.MTLIndirectComputeCommand {
				c := icb.IndirectComputeCommandAtIndex(uint(base + op))
				if base+op != 0 {
					c.SetBarrier()
				}
				return c
			}
			// ===== attention half: h = in + Wo·sdpa(rope(Wq·rms(in))) =====
			setRMS(cmd(0), inBuf, anwBuf, attnNormed)
			setGemv(cmd(1), gemvQPSO, wqBuf, attnNormed, q, qInB, qOutB, qLdB, qDim, bmQ, bnQ, smQ, tmQ)
			c := cmd(2)
			c.SetComputePipelineState(ropePSO)
			c.SetKernelBufferOffsetAtIndex(q, 0, 0)
			c.SetKernelBufferOffsetAtIndex(qr, 0, 1)
			c.SetKernelBufferOffsetAtIndex(offBuf, 0, 2)
			c.SetKernelBufferOffsetAtIndex(ropeScaleB, 0, 3)
			c.SetKernelBufferOffsetAtIndex(ropeMatB, 0, 4)
			c.SetKernelBufferOffsetAtIndex(ropeBaseB, 0, 10)
			ropeDim0 := uint(headDim / 2)
			c.ConcurrentDispatchThreadsThreadsPerThreadgroup(metal.MTLSize{Width: ropeDim0, Height: uint(nHeads), Depth: 1}, metal.MTLSize{Width: ropeDim0, Height: 1, Depth: 1})

			c = cmd(3)
			c.SetComputePipelineState(sdpaPSO)
			c.SetKernelBufferOffsetAtIndex(qr, 0, 0)
			c.SetKernelBufferOffsetAtIndex(kBuf, 0, 1)
			c.SetKernelBufferOffsetAtIndex(vBuf, 0, 2)
			c.SetKernelBufferOffsetAtIndex(attn, 0, 3)
			c.SetKernelBufferOffsetAtIndex(gqaB, 0, 4)
			c.SetKernelBufferOffsetAtIndex(nB, 0, 5)
			c.SetKernelBufferOffsetAtIndex(khsB, 0, 6)
			c.SetKernelBufferOffsetAtIndex(kssB, 0, 7)
			c.SetKernelBufferOffsetAtIndex(vhsB, 0, 8)
			c.SetKernelBufferOffsetAtIndex(vssB, 0, 9)
			c.SetKernelBufferOffsetAtIndex(sdpaScaleB, 0, 10)
			c.ConcurrentDispatchThreadgroupsThreadsPerThreadgroup(metal.MTLSize{Width: uint(nHeads), Height: 1, Depth: 1}, metal.MTLSize{Width: 1024, Height: 1, Depth: 1})

			setGemv(cmd(4), gemvOPSO, woBuf, attn, attnOut, oInB, oOutB, oLdB, dModel, bmO, bnO, smO, tmO)
			setBinary(cmd(5), addPSO, inBuf, attnOut, h, addModelB, dModel)

			// ===== MLP half: out = h + Wdown·(gelu(Wgate·rms(h))·(Wup·rms(h))) =====
			setRMS(cmd(6), h, mnwBuf, mlpNormed)
			setGemv(cmd(7), gemvFPSO, wgBuf, mlpNormed, gate, fInB, fOutB, fLdB, dFF, bmF, bnF, smF, tmF)
			setGemv(cmd(8), gemvFPSO, wuBuf, mlpNormed, up, fInB, fOutB, fLdB, dFF, bmF, bnF, smF, tmF)
			setBinary(cmd(9), mulPSO, gate, gate, x2, cntFFB, dFF)
			setBinary(cmd(10), mulPSO, x2, gate, x3, cntFFB, dFF)
			setBinary(cmd(11), mulPSO, x3, c044, x3s, cntFFB, dFF)
			setBinary(cmd(12), addPSO, gate, x3s, inner, cntFFB, dFF)
			setBinary(cmd(13), mulPSO, inner, c079, scaled, cntFFB, dFF)
			c = cmd(14)
			c.SetComputePipelineState(tanhPSO)
			c.SetKernelBufferOffsetAtIndex(scaled, 0, 0)
			c.SetKernelBufferOffsetAtIndex(tnh, 0, 1)
			c.SetKernelBufferOffsetAtIndex(tanhCntB, 0, 2)
			c.ConcurrentDispatchThreadsThreadsPerThreadgroup(metal.MTLSize{Width: uint(dFF), Height: 1, Depth: 1}, metal.MTLSize{Width: elemGroup(dFF), Height: 1, Depth: 1})
			setBinary(cmd(15), addPSO, tnh, c1c, onePlus, cntFFB, dFF)
			setBinary(cmd(16), mulPSO, gate, c05, halfG, cntFFB, dFF)
			setBinary(cmd(17), mulPSO, halfG, onePlus, gelu, cntFFB, dFF)
			setBinary(cmd(18), mulPSO, gelu, up, gated, cntFFB, dFF)
			setGemv(cmd(19), gemvDPSO, wdBuf, gated, down, dInB, dOutB, dLdB, dModel, bmD, bnD, smD, tmD)
			setBinary(cmd(20), addPSO, h, down, outBuf, addModelB, dModel)
		}

		in, outB := xA, xB
		for L := 0; L < nLayers; L++ {
			recordLayer(21*L, in, outB)
			in, outB = outB, in
		}
		lastOut := in // after the final swap, `in` is the last layer's output

		rng := foundation.NSRange{Location: 0, Length: uint(total)}
		for r := 0; r < replays; r++ {
			cb := queue.CommandBuffer()
			enc := cb.ComputeCommandEncoder()
			for _, b := range resident {
				enc.UseResourceUsage(b, metal.MTLResourceUsageRead|metal.MTLResourceUsageWrite)
			}
			enc.ExecuteCommandsInBufferWithRange(icb, rng)
			enc.EndEncoding()
			cb.Commit()
			cb.WaitUntilCompleted()
		}
		copy(out, unsafe.Slice((*byte)(lastOut.Contents()), len(out)))
	})
	return out, nil
}
