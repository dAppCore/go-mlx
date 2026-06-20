// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"slices"
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
//
// perLayerDFF carries each layer's FFN width (gemma4 E2B/E4B MatFormer varies it per
// layer): the FFN scratch + GeLU-constant buffers are sized to the WIDEST layer and the
// per-layer FFN dispatch widths / element-count buffers read only that layer's lff. A nil
// or short entry (or 0) falls back to the uniform dFF, so the existing uniform callers are
// byte-identical. The recordProj seam keys the gate/up/down PSOs per layer (it already
// receives li), so it must select the matching (outDim,inDim) shape for that layer's lff.
// (Per-layer headDim — gemma4 global layers' larger head_dim — is a later step: it would
// also make kvDim/rowBytes/SDPA-PSO per-layer; this core keeps headDim uniform.)
func decodeForwardArchICBCore(
	inputs [][]byte, specs []g4.LayerSpec,
	anwBufs, mnwBufs, kCaches, vCaches, projResident []metal.MTLBuffer,
	qNormBufs, kNormBufs, postAttnBufs, postFFBufs []metal.MTLBuffer,
	recordProj func(li int, c metal.MTLIndirectComputeCommand, vec, out metal.MTLBuffer, outOff uint, p projIndex),
	vOutBind uint, valueNormOnes metal.MTLBuffer, vProjIdx projIndex,
	dModel, nHeads, nKVHeads, headDim, dFF, maxLen, slidingWindow int,
	perLayerDFF []int,
	base, scale, eps float32,
) ([][]byte, error) {
	nLayers, T := len(anwBufs), len(inputs)
	qDim, kvDim := nHeads*headDim, nKVHeads*headDim
	// per-layer FFN width: lffOf(li) is this layer's FFN dim (gemma4 MatFormer); maxDFF
	// sizes the shared FFN scratch + GeLU constants to the widest layer. Falls back to the
	// uniform dFF when perLayerDFF is absent/0 ⇒ uniform callers are byte-identical.
	lffOf := func(li int) int {
		if li < len(perLayerDFF) && perLayerDFF[li] > 0 {
			return perLayerDFF[li]
		}
		return dFF
	}
	maxDFF := dFF
	for li := 0; li < nLayers; li++ {
		if l := lffOf(li); l > maxDFF {
			maxDFF = l
		}
	}

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
	var coreErr error
	withAutoreleasePool(func() {
		normed := scratchBF16(dModel)
		q, qr, kProj, attn := scratchBF16(qDim), scratchBF16(qDim), scratchBF16(kvDim), scratchBF16(qDim)
		attnOut := scratchBF16(dModel)
		kThrow, vThrow := scratchBF16(kvDim), scratchBF16(kvDim) // sharer's discarded K/V
		mlpNormed := scratchBF16(dModel)
		// FFN scratch + GeLU constants sized to the WIDEST layer (gemma4 MatFormer varies dFF
		// per layer); each layer dispatches only its own lff elements, so a narrower layer reads
		// a prefix of these buffers. Uniform callers (maxDFF==dFF) are byte-identical.
		gate, up := scratchBF16(maxDFF), scratchBF16(maxDFF)
		x2, x3, x3s, inner := scratchBF16(maxDFF), scratchBF16(maxDFF), scratchBF16(maxDFF), scratchBF16(maxDFF)
		scaled, tnh, onePlus, halfG := scratchBF16(maxDFF), scratchBF16(maxDFF), scratchBF16(maxDFF), scratchBF16(maxDFF)
		gelu, gated, down := scratchBF16(maxDFF), scratchBF16(maxDFF), scratchBF16(dModel)
		c044 := sharedBytes(bf16ConstBytes(maxDFF, 0.044715))
		c079 := sharedBytes(bf16ConstBytes(maxDFF, 0.7978845608028654))
		c1c := sharedBytes(bf16ConstBytes(maxDFF, 1.0))
		c05 := sharedBytes(bf16ConstBytes(maxDFF, 0.5))
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
		axisHeadBuf := scalarI32(int32(headDim)) // axis size for the per-head QK-norm
		ropeScaleB := scalarF32(scale)
		ropeMatB := scalarI64(int64(headDim))
		ropeBaseB := scalarF32(float32(math.Log2(float64(base))))
		gqaB := scalarI32(int32(nHeads / nKVHeads))
		khsB, kssB := scalarI64(int64(headDim)), scalarI64(int64(kvDim))
		vhsB, vssB := scalarI64(int64(headDim)), scalarI64(int64(kvDim))
		sdpaScaleB := scalarF32(scale)
		addModelB := scalarI32(int32(dModel))
		// per-distinct-dFF element-count buffers (the FFN binary/gelu/tanh ops take the count
		// as a buffer): one scalar per distinct width, shared across layers of that width. Every
		// one is appended to resident below so the ICB replay's UseResources covers it — a
		// non-resident count buffer is read as garbage on the layer that uses it.
		ffCntBufs := make(map[int]metal.MTLBuffer)
		ffCntOf := func(n int) metal.MTLBuffer {
			b, ok := ffCntBufs[n]
			if !ok {
				b = scalarI32(int32(n))
				ffCntBufs[n] = b
			}
			return b
		}
		for li := 0; li < nLayers; li++ {
			ffCntOf(lffOf(li))
		}

		resident := []metal.MTLBuffer{
			ping[0], ping[1], normed, q, qr, kProj, attn, attnOut, kThrow, vThrow, mlpNormed,
			gate, up, x2, x3, x3s, inner, scaled, tnh, onePlus, halfG, gelu, gated, down,
			c044, c079, c1c, c05,
			offBuf, nGlobalBuf, nSlidingBuf, epsBuf, axisBuf, axisHeadBuf, wsBuf,
			ropeScaleB, ropeMatB, ropeBaseB, gqaB, khsB, kssB, vhsB, vssB, sdpaScaleB, addModelB,
		}
		for _, b := range ffCntBufs { // the per-distinct-dFF FFN count buffers must be resident for the replay
			resident = append(resident, b)
		}
		// reserve the upper-bound capacity for the appends that follow (projResident + the per-layer
		// weight/norm/cache slices, ≤16 buffers/layer + the 19 projResident scalars) so the resident
		// slice never geometrically regrows its backing array. Grow changes capacity only — the
		// literal contents, the appended buffers, and every kernel binding are unchanged.
		resident = slices.Grow(resident, 16*nLayers+20)
		resident = append(resident, projResident...)
		resident = append(resident, anwBufs...)
		resident = append(resident, mnwBufs...)
		// gemma4 norm buffers (uniform presence across layers); add the non-nil ones.
		for _, bufs := range [][]metal.MTLBuffer{qNormBufs, kNormBufs, postAttnBufs, postFFBufs} {
			for _, b := range bufs {
				if b != nil {
					resident = append(resident, b)
				}
			}
		}
		if valueNormOnes != nil {
			resident = append(resident, valueNormOnes)
		}
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

		// gemma4 norm presence (uniform across layers): each present norm adds one op per
		// layer, so the layout grows but stays uniform → a single running op counter.
		hasQN := len(qNormBufs) > 0 && qNormBufs[0] != nil
		hasKN := len(kNormBufs) > 0 && kNormBufs[0] != nil
		hasPA := len(postAttnBufs) > 0 && postAttnBufs[0] != nil
		hasPF := len(postFFBufs) > 0 && postFFBufs[0] != nil
		extra := 0
		for _, h := range []bool{hasQN, hasKN, hasPA, hasPF} {
			if h {
				extra++
			}
		}
		if valueNormOnes != nil { // gemma4 value-norm adds one op/layer (owner: the V row; sharer: discarded)
			extra++
		}
		opsPerLayer := 24 + extra
		if gpuHasGeluKernel() { // fused gelu is 1 command vs the composed chain's 10
			opsPerLayer -= 9
		}
		total := opsPerLayer * nLayers
		icbDesc := metal.NewMTLIndirectCommandBufferDescriptor()
		icbDesc.SetCommandTypes(metal.MTLIndirectCommandTypeConcurrentDispatch)
		icbDesc.SetInheritBuffers(false)
		icbDesc.SetInheritPipelineState(false)
		icbDesc.SetMaxKernelBufferBindCount(16)
		icb := device.NewIndirectCommandBufferWithDescriptorMaxCommandCountOptions(icbDesc, uint(total), metal.MTLResourceStorageModeShared)

		rmsTG := uint(rmsSimdSize * ((((dModel + rmsNReads - 1) / rmsNReads) + rmsSimdSize - 1) / rmsSimdSize))
		headTG := uint(rmsSimdSize * ((((headDim + rmsNReads - 1) / rmsNReads) + rmsSimdSize - 1) / rmsSimdSize))
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
		// setRMSRows records a per-head RMSNorm (gemma4 QK-norm): `rows` threadgroups over
		// headDim each, with the headDim axis size. Mirrors encRMSNormRowsBF16.
		setRMSRows := func(c metal.MTLIndirectComputeCommand, in, w, o metal.MTLBuffer, rows int) {
			c.SetComputePipelineState(rmsPSO)
			c.SetKernelBufferOffsetAtIndex(in, 0, 0)
			c.SetKernelBufferOffsetAtIndex(w, 0, 1)
			c.SetKernelBufferOffsetAtIndex(o, 0, 2)
			c.SetKernelBufferOffsetAtIndex(epsBuf, 0, 3)
			c.SetKernelBufferOffsetAtIndex(axisHeadBuf, 0, 4)
			c.SetKernelBufferOffsetAtIndex(wsBuf, 0, 5)
			c.ConcurrentDispatchThreadsThreadsPerThreadgroup(metal.MTLSize{Width: uint(rows) * headTG, Height: 1, Depth: 1}, metal.MTLSize{Width: headTG, Height: 1, Depth: 1})
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
		vNormCmd := make([]metal.MTLIndirectComputeCommand, nLayers) // owner value-norm on the V row (rebound/token)
		sdpaCmd := make([]metal.MTLIndirectComputeCommand, nLayers)  // every layer (sliding: read offset)

		// one running command index across the whole stack (the conditional norm ops make
		// per-layer offsets uneven, but the count is uniform so the running counter stays
		// aligned). The barrier on every command but the first makes execution sequential.
		opIdx := 0
		emit := func() metal.MTLIndirectComputeCommand {
			c := icb.IndirectComputeCommandAtIndex(uint(opIdx))
			if opIdx != 0 {
				c.SetBarrier()
			}
			opIdx++
			return c
		}

		for li := 0; li < nLayers; li++ {
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

			// --- attention half ---
			setRMS(emit(), inBuf, anwBufs[li], normed)
			recordProj(li, emit(), normed, q, 0, projQ)
			if hasQN { // gemma4 per-head QK-norm on Q before RoPE (in-place)
				setRMSRows(emit(), q, qNormBufs[li], q, nHeads)
			}
			setRope(emit(), q, qr, nHeads)
			recordProj(li, emit(), normed, kProj, 0, projK)
			if hasKN {
				setRMSRows(emit(), kProj, kNormBufs[li], kProj, nKVHeads)
			}
			if owns {
				ck := emit()
				setRope(ck, kProj, kCaches[li], nKVHeads) // -> kCache @ row pos (rebound/token)
				kRopeCmd[li] = ck
				cv := emit()
				recordProj(li, cv, normed, vCaches[li], 0, vProjIdx) // -> vCache @ row pos (rebound/token); K==V projects via wK
				vCmd[li] = cv
				if valueNormOnes != nil { // gemma4 value-norm on the new V row (per head; rebound/token)
					cvn := emit()
					setRMSRows(cvn, vCaches[li], valueNormOnes, vCaches[li], nKVHeads)
					vNormCmd[li] = cvn
				}
			} else {
				setRope(emit(), kProj, kThrow, nKVHeads)            // discarded
				recordProj(li, emit(), normed, vThrow, 0, vProjIdx) // discarded
				if valueNormOnes != nil {
					setRMSRows(emit(), vThrow, valueNormOnes, vThrow, nKVHeads) // discarded (keeps the op layout uniform)
				}
			}
			// SDPA over the owner's cache; sliding layers read the windowed slice.
			cs := emit()
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
			recordProj(li, emit(), attn, attnOut, 0, projO)
			if hasPA { // gemma4 post-attention norm on Wo·attn before the residual (in-place)
				setRMS(emit(), attnOut, postAttnBufs[li], attnOut)
			}
			setBin(emit(), addPSO, inBuf, attnOut, hBuf, addModelB, dModel)

			// --- MLP half --- (lff = this layer's FFN width; the FFN ops dispatch only lff
			// elements + bind this width's count buffer — gemma4 MatFormer varies it per layer)
			lff := lffOf(li)
			ffCntB := ffCntOf(lff)
			setRMS(emit(), hBuf, mnwBufs[li], mlpNormed)
			recordProj(li, emit(), mlpNormed, gate, 0, projGate)
			recordProj(li, emit(), mlpNormed, up, 0, projUp)
			if gpuHasGeluKernel() { // fused gelu(gate)·up — one ICB command (ffCntB = lff as the n buffer)
				cg := emit()
				cg.SetComputePipelineState(geluICBPSO)
				cg.SetKernelBufferOffsetAtIndex(gate, 0, 0)
				cg.SetKernelBufferOffsetAtIndex(up, 0, 1)
				cg.SetKernelBufferOffsetAtIndex(gated, 0, 2)
				cg.SetKernelBufferOffsetAtIndex(ffCntB, 0, 3)
				cg.ConcurrentDispatchThreadsThreadsPerThreadgroup(metal.MTLSize{Width: uint(lff), Height: 1, Depth: 1}, metal.MTLSize{Width: elemGroup(lff), Height: 1, Depth: 1})
			} else {
				setBin(emit(), mulPSO, gate, gate, x2, ffCntB, lff)
				setBin(emit(), mulPSO, x2, gate, x3, ffCntB, lff)
				setBin(emit(), mulPSO, x3, c044, x3s, ffCntB, lff)
				setBin(emit(), addPSO, gate, x3s, inner, ffCntB, lff)
				setBin(emit(), mulPSO, inner, c079, scaled, ffCntB, lff)
				ct := emit()
				ct.SetComputePipelineState(tanhPSO)
				ct.SetKernelBufferOffsetAtIndex(scaled, 0, 0)
				ct.SetKernelBufferOffsetAtIndex(tnh, 0, 1)
				ct.SetKernelBufferOffsetAtIndex(ffCntB, 0, 2)
				ct.ConcurrentDispatchThreadsThreadsPerThreadgroup(metal.MTLSize{Width: uint(lff), Height: 1, Depth: 1}, metal.MTLSize{Width: elemGroup(lff), Height: 1, Depth: 1})
				setBin(emit(), addPSO, tnh, c1c, onePlus, ffCntB, lff)
				setBin(emit(), mulPSO, gate, c05, halfG, ffCntB, lff)
				setBin(emit(), mulPSO, halfG, onePlus, gelu, ffCntB, lff)
				setBin(emit(), mulPSO, gelu, up, gated, ffCntB, lff)
			}
			recordProj(li, emit(), gated, down, 0, projDown)
			if hasPF { // gemma4 post-feed-forward norm on Wdown·… before the residual (in-place)
				setRMS(emit(), down, postFFBufs[li], down)
			}
			setBin(emit(), addPSO, hBuf, down, outBuf, addModelB, dModel)
		}
		// the per-layer op-count is invariant to dFF (the gelu/no-gelu + owner/sharer branches
		// are fixed-count), so the running index must land exactly on `total`. A mismatch means
		// the recorded layout diverged from opsPerLayer·nLayers — a recorder bug, not a numeric
		// drift; fail loud rather than replay a misaligned ICB.
		if opIdx != total {
			coreErr = core.NewError(core.Sprintf("native.decodeForwardArchICBCore: recorded %d ops, expected %d (opsPerLayer=%d × %d layers) — heterogeneous layout misaligned", opIdx, total, opsPerLayer, nLayers))
			return
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
					if valueNormOnes != nil { // value-norm reads+writes the new V row in place
						vNormCmd[li].SetKernelBufferOffsetAtIndex(vCaches[li], rowOff, 0)
						vNormCmd[li].SetKernelBufferOffsetAtIndex(vCaches[li], rowOff, 2)
					}
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
	base, scale, eps float32, valueNorm bool,
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

	// per-layer FFN width (gemma4 E2B/E4B MatFormer): lFF[li] (from w.DFF, fallback dFF).
	lFF := make([]int, nLayers)
	for li := range layers {
		lFF[li] = dFF
		if layers[li].DFF > 0 {
			lFF[li] = layers[li].DFF
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
	// gate/up (dModel→lff) and down (lff→dModel) gemv PSOs + tiles, one per distinct FFN width.
	type gemvShape struct {
		pso            metal.MTLComputePipelineState
		bm, bn, sm, tm int
	}
	ffUp := make(map[int]gemvShape)   // gate/up: dModel→lff
	ffDown := make(map[int]gemvShape) // down: lff→dModel
	for li := range lFF {
		lff := lFF[li]
		if _, ok := ffUp[lff]; !ok {
			p, bm, bn, sm, tm, e := gemvPSO(dModel, lff)
			if e != nil {
				return nil, e
			}
			ffUp[lff] = gemvShape{p, bm, bn, sm, tm}
			p2, bm2, bn2, sm2, tm2, e2 := gemvPSO(lff, dModel)
			if e2 != nil {
				return nil, e2
			}
			ffDown[lff] = gemvShape{p2, bm2, bn2, sm2, tm2}
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
		type lw struct{ wq, wk, wv, wo, wg, wu, wd metal.MTLBuffer }
		lb := make([]lw, nLayers)
		cacheBytes := uint(maxLen * kvDim * bf16Size)
		// presized to the upper bound (every layer's ≤7 projection buffers, the 16 shared trailing
		// scalar buffers, plus ≤3 FFN dim scalars per distinct dFF width) so the per-forward build
		// never geometrically regrows its backing array — K==V layers leave the v-proj slot unused.
		// Byte-identical.
		projResident := make([]metal.MTLBuffer, 0, nLayers*7+16+nLayers*3)
		for li := range layers {
			w := layers[li]
			anwBufs[li] = sharedBytes(w.AttnNormW)
			mnwBufs[li] = sharedBytes(w.MLPNormW)
			qNormBufs[li] = sharedOrNil(w.QNormW)
			kNormBufs[li] = sharedOrNil(w.KNormW)
			postAttnBufs[li] = sharedOrNil(w.PostAttnNormW)
			postFFBufs[li] = sharedOrNil(w.PostFFNormW)
			if specs[li].OwnsCache() {
				kCaches[li] = device.NewBufferWithLengthOptions(cacheBytes, metal.MTLResourceStorageModeShared)
				vCaches[li] = device.NewBufferWithLengthOptions(cacheBytes, metal.MTLResourceStorageModeShared)
			}
			lb[li] = lw{sharedBytes(w.WQ), sharedBytes(w.WK), sharedOrNil(w.WV), sharedBytes(w.WO), sharedBytes(w.WGate), sharedBytes(w.WUp), sharedBytes(w.WDown)}
			projResident = append(projResident, lb[li].wq, lb[li].wk, lb[li].wo, lb[li].wg, lb[li].wu, lb[li].wd)
			if lb[li].wv != nil { // gemma4 K==V layers carry no v_proj
				projResident = append(projResident, lb[li].wv)
			}
		}
		qInB, qOutB, qLdB := scalarI32(int32(dModel)), scalarI32(int32(qDim)), scalarI32(int32(dModel))
		kvInB, kvOutB, kvLdB := scalarI32(int32(dModel)), scalarI32(int32(kvDim)), scalarI32(int32(dModel))
		oInB, oOutB, oLdB := scalarI32(int32(qDim)), scalarI32(int32(dModel)), scalarI32(int32(qDim))
		// FFN gemv dim scalars: the dModel-side (up's in/ld, down's out) are shared; the lff-side
		// (up's out, down's in/ld) is one buffer per distinct width. All appended to projResident.
		fInB, fLdB, dOutB := scalarI32(int32(dModel)), scalarI32(int32(dModel)), scalarI32(int32(dModel))
		fOutByDFF := make(map[int]metal.MTLBuffer) // up out dim = lff
		dInByDFF := make(map[int]metal.MTLBuffer)  // down in dim = lff
		dLdByDFF := make(map[int]metal.MTLBuffer)  // down leading dim = lff
		for li := range lFF {
			lff := lFF[li]
			if _, ok := fOutByDFF[lff]; !ok {
				fOutByDFF[lff] = scalarI32(int32(lff))
				dInByDFF[lff] = scalarI32(int32(lff))
				dLdByDFF[lff] = scalarI32(int32(lff))
			}
		}
		bndB, bshB, vsB, msB := scalarI32(1), scalarI32(1), scalarI64(0), scalarI64(0)
		projResident = append(projResident, qInB, qOutB, qLdB, kvInB, kvOutB, kvLdB, oInB, oOutB, oLdB, fInB, fLdB, dOutB, bndB, bshB, vsB, msB)
		for lff, b := range fOutByDFF {
			projResident = append(projResident, b, dInByDFF[lff], dLdByDFF[lff])
		}

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
				lff := lFF[li]
				u := ffUp[lff]
				setGemv(c, u.pso, l.wg, vec, out, fInB, fOutByDFF[lff], fLdB, outOff, lff, u.bm, u.bn, u.sm, u.tm)
			case projUp:
				lff := lFF[li]
				u := ffUp[lff]
				setGemv(c, u.pso, l.wu, vec, out, fInB, fOutByDFF[lff], fLdB, outOff, lff, u.bm, u.bn, u.sm, u.tm)
			case projDown:
				lff := lFF[li]
				d := ffDown[lff]
				setGemv(c, d.pso, l.wd, vec, out, dInByDFF[lff], dOutB, dLdByDFF[lff], outOff, dModel, d.bm, d.bn, d.sm, d.tm)
			}
		}
		valueNormOnes := valueNormOnesBuf(valueNorm, headDim)
		vProjIdx := projV
		if len(layers[0].WV) == 0 { // gemma4 K==V: V rides the k-proj
			vProjIdx = projK
		}
		outputs, coreErr = decodeForwardArchICBCore(inputs, specs, anwBufs, mnwBufs, kCaches, vCaches, projResident, qNormBufs, kNormBufs, postAttnBufs, postFFBufs, recordProj, 3, valueNormOnes, vProjIdx, dModel, nHeads, nKVHeads, headDim, dFF, maxLen, slidingWindow, lFF, base, scale, eps)
	})
	if coreErr != nil {
		return nil, coreErr
	}
	return outputs, nil
}
