// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"slices"
	"unsafe"

	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/model"
	"github.com/tmc/apple/foundation"
	"github.com/tmc/apple/metal"
)

type archICBPLEPlan struct {
	runtime                *archDecodePLEInputs
	pliDim                 int
	postNormBufs           []metal.MTLBuffer
	resident               []metal.MTLBuffer
	recordGate, recordProj func(li int, c metal.MTLIndirectComputeCommand, vec, out metal.MTLBuffer)
}

func (p *archICBPLEPlan) enabled() bool {
	return p != nil && p.runtime != nil && p.pliDim > 0
}

// archICBReplay is a recorded arch ICB held for incremental replay: recordArchICB builds it ONCE
// (the decode stack baked into icb) and each stepBody replays it for ONE token over the growing
// cache with cheap per-token offset rebinds. The batch core records it + runBatch-loops every
// token (byte-identical to the old single-call core); the ArchSession holds it across StepWithID
// calls for the per-token encode-bypass. Every buffer + the icb is retained (scratchBF16 /
// device.New* return owned objects, like the session's own caches), so the struct survives the
// per-step autorelease pools.
type archICBReplay struct {
	icb                               metal.MTLIndirectCommandBuffer
	rng                               foundation.NSRange
	residentRes                       []metal.MTLResource
	specs                             []model.LayerSpec
	nLayers                           int
	vOutBind                          uint
	hasValueNorm                      bool
	kRopeIdx, vIdx, vNormIdx, sdpaIdx []int
	kCaches, vCaches                  []metal.MTLBuffer
	offBuf, nGlobalBuf, nSlidingBuf   metal.MTLBuffer
	ping0, lastOut, pleInput          metal.MTLBuffer
	hasPLE                            bool
	plePliDim                         int
	pleRuntime                        *archDecodePLEInputs
	rowBytes                          []int // per-layer KV cache row stride (nKVHeads·hd·bf16Size) — gemma4 global layers are wider
	slidingWindow, dModel             int
}

// stepBody replays the recorded ICB for ONE token at position pos over the growing cache. pli is
// this token's [nLayers·pliDim] PerLayerInputs tensor (nil for non-PLE); the caller computes it
// (ArchSession.StepWithID from the token id, runBatch from the batch token ids). Returns a
// fresh hidden copy (read out of the device buffer, so it survives the caller's pool). The caller
// wraps the call in withAutoreleasePool (StepWithID + runBatch both do).
func (r *archICBReplay) stepBody(inputEmb []byte, pos int, pli []byte) []byte {
	*(*int32)(r.offBuf.Contents()) = int32(pos)
	*(*int32)(r.nGlobalBuf.Contents()) = int32(pos + 1)
	win := pos + 1
	start := 0
	if r.slidingWindow > 0 && win > r.slidingWindow {
		start = win - r.slidingWindow
		win = r.slidingWindow
	}
	*(*int32)(r.nSlidingBuf.Contents()) = int32(win)
	for li := 0; li < r.nLayers; li++ {
		if r.specs[li].OwnsCache() {
			// Re-acquire the command from the retained icb each step: the handle from
			// IndirectComputeCommandAtIndex is a pool-scoped view that does NOT survive the
			// record pool's drain, but the icb + its recorded commands persist — so rebind by
			// op index. (The buffers + the icb are device.New*-owned, hence retained.)
			rowOff := uint(pos * r.rowBytes[li]) // per-layer: global layers' rows are wider (larger head_dim)
			r.icb.IndirectComputeCommandAtIndex(uint(r.kRopeIdx[li])).SetKernelBufferOffsetAtIndex(r.kCaches[li], rowOff, 1)
			r.icb.IndirectComputeCommandAtIndex(uint(r.vIdx[li])).SetKernelBufferOffsetAtIndex(r.vCaches[li], rowOff, r.vOutBind)
			if r.hasValueNorm {
				vn := r.icb.IndirectComputeCommandAtIndex(uint(r.vNormIdx[li]))
				vn.SetKernelBufferOffsetAtIndex(r.vCaches[li], rowOff, 0)
				vn.SetKernelBufferOffsetAtIndex(r.vCaches[li], rowOff, 2)
			}
		}
		if r.specs[li].Attention == model.SlidingAttention {
			own := r.specs[li].KVShareFrom
			slideOff := uint(start * r.rowBytes[own]) // read the owner's cache at its row stride
			sd := r.icb.IndirectComputeCommandAtIndex(uint(r.sdpaIdx[li]))
			sd.SetKernelBufferOffsetAtIndex(r.kCaches[own], slideOff, 1)
			sd.SetKernelBufferOffsetAtIndex(r.vCaches[own], slideOff, 2)
		}
	}
	if r.hasPLE && pli != nil {
		want := r.nLayers * r.plePliDim * bf16Size
		copy(unsafe.Slice((*byte)(r.pleInput.Contents()), want), pli)
	}
	copy(unsafe.Slice((*byte)(r.ping0.Contents()), r.dModel*bf16Size), inputEmb)
	cb := queue.CommandBuffer()
	enc := cb.ComputeCommandEncoder()
	enc.UseResourcesCountUsage(r.residentRes, uint(len(r.residentRes)), metal.MTLResourceUsageRead|metal.MTLResourceUsageWrite)
	enc.ExecuteCommandsInBufferWithRange(r.icb, r.rng)
	enc.EndEncoding()
	cb.Commit()
	cb.WaitUntilCompleted()
	out := make([]byte, r.dModel*bf16Size)
	copy(out, unsafe.Slice((*byte)(r.lastOut.Contents()), r.dModel*bf16Size))
	return out
}

// runBatch replays the recorded ICB across a whole T-token sequence — the batch encode-bypass
// (the old core's replay loop), one autorelease pool for the run. PLE tensors are computed per
// token from the recorded runtime's batch token ids.
func (r *archICBReplay) runBatch(inputs [][]byte) ([][]byte, error) {
	if r.hasPLE && len(r.pleRuntime.tokenIDs) != len(inputs) {
		return nil, core.NewError("native.archICBReplay.runBatch: PLE token id count must equal inputs")
	}
	outputs := make([][]byte, len(inputs))
	var coreErr error
	withAutoreleasePool(func() {
		for t := range inputs {
			var pli []byte
			if r.hasPLE {
				p, err := r.pleRuntime.compute(r.pleRuntime.tokenIDs[t], inputs[t])
				if err != nil {
					coreErr = err
					return
				}
				if len(p) != r.nLayers*r.plePliDim*bf16Size {
					coreErr = core.NewError("native.archICBReplay.runBatch: PLE tensor size mismatch")
					return
				}
				pli = p
			}
			outputs[t] = r.stepBody(inputs[t], t, pli)
		}
	})
	if coreErr != nil {
		return nil, coreErr
	}
	return outputs, nil
}

// icbRope bundles the per-layer rope geometry the ICB records: the global theta `base` + the
// sliding theta `localBase`, the partial-rotary dims (`rotaryDim` global, `rotaryDimLocal` sliding),
// the `globalHeadDim` proportional-global layers rope over, and the explicit-periods buffers
// (`globalFreqs` proportional-global, `freqs` YaRN; nil ⇒ base-derived). A uniform model sets
// localBase==base, rotary==headDim, nil freqs ⇒ every layer ropes on `base` (the old single-base
// behaviour, byte-identical).
type icbRope struct {
	base, localBase                          float32
	rotaryDim, rotaryDimLocal, globalHeadDim int
	globalFreqs, freqs                       metal.MTLBuffer
}

// simpleICBRope is the uniform rope (every layer on `base`, full rotary, no freqs) — the
// byte-identical default for callers that carry no per-layer rope (the bf16/quant batch entries).
func simpleICBRope(base float32, headDim int) icbRope {
	return icbRope{base: base, localBase: base, rotaryDim: headDim, rotaryDimLocal: headDim, globalHeadDim: headDim}
}

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
func recordArchICB(
	specs []model.LayerSpec,
	anwBufs, mnwBufs, kCaches, vCaches, projResident []metal.MTLBuffer,
	qNormBufs, kNormBufs, postAttnBufs, postFFBufs []metal.MTLBuffer,
	layerScalarBufs []metal.MTLBuffer, ple *archICBPLEPlan,
	recordProj func(li int, c metal.MTLIndirectComputeCommand, vec, out metal.MTLBuffer, outOff uint, p projIndex),
	vOutBind uint, valueNormOnes metal.MTLBuffer, vProjIdx projIndex,
	dModel, nHeads, nKVHeads, headDim, dFF, maxLen, slidingWindow int,
	perLayerDFF []int,
	rope icbRope, scale, eps float32,
) (*archICBReplay, error) {
	nLayers := len(anwBufs)
	// per-layer head dim (gemma4 full_attention layers attend with a LARGER head_dim than sliding):
	// hdOf(li) is the layer's head dim; maxHd sizes the shared attention scratch; each layer binds a
	// per-hd SDPA PSO + stride/axis/rope-stride. kvHeads stays uniform (the eligibility guarantees it).
	// Uniform models (maxHd==headDim) are byte-identical.
	hdOf := func(li int) int { return headDimOf(specs[li], headDim) }
	kvdOf := func(li int) int { return nKVHeads * hdOf(li) }
	maxHd := headDim
	for li := 0; li < nLayers; li++ {
		if h := hdOf(li); h > maxHd {
			maxHd = h
		}
	}
	maxQd, maxKvd := nHeads*maxHd, nKVHeads*maxHd
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
	hasPLE := ple.enabled()
	if hasPLE {
		if len(ple.postNormBufs) != nLayers {
			return nil, core.NewError("native.recordArchICB: PLE post norm count must equal layers")
		}
	}
	hasLayerScalar := false
	for _, b := range layerScalarBufs {
		if b != nil {
			hasLayerScalar = true
			break
		}
	}
	maxGelu := maxDFF
	if hasPLE && ple.pliDim > maxGelu {
		maxGelu = ple.pliDim
	}

	rmsPSO, err := pipelineForICB("rmsbfloat16")
	if err != nil {
		return nil, err
	}
	ropePSO, err := ropePipelineICB(false)
	if err != nil {
		return nil, err
	}
	var ropeFreqsPSO metal.MTLComputePipelineState
	if rope.globalFreqs != nil || rope.freqs != nil {
		if ropeFreqsPSO, err = ropeFreqsPipelineICB(false); err != nil {
			return nil, err
		}
	}
	// per-hd SDPA PSO (gemma4 global 512 vs sliding 256 head dim) — one per distinct hd, picked per layer.
	sdpaPSOByHd := make(map[int]metal.MTLComputePipelineState)
	for li := 0; li < nLayers; li++ {
		hd := hdOf(li)
		if _, ok := sdpaPSOByHd[hd]; !ok {
			pso, e := sdpaVectorPipelineICB(core.Sprintf("sdpa_vector_bfloat16_t_%d_%d", hd, hd))
			if e != nil {
				return nil, e
			}
			sdpaPSOByHd[hd] = pso
		}
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

	var r *archICBReplay
	var coreErr error
	withAutoreleasePool(func() {
		normed := scratchBF16(dModel)
		q, qr, kProj, attn := scratchBF16(maxQd), scratchBF16(maxQd), scratchBF16(maxKvd), scratchBF16(maxQd)
		attnOut := scratchBF16(dModel)
		kThrow, vThrow := scratchBF16(maxKvd), scratchBF16(maxKvd) // sharer's discarded K/V
		mlpNormed := scratchBF16(dModel)
		// FFN scratch + GeLU constants sized to the WIDEST layer (gemma4 MatFormer varies dFF
		// per layer); each layer dispatches only its own lff elements, so a narrower layer reads
		// a prefix of these buffers. Uniform callers (maxDFF==dFF) are byte-identical.
		gate, up := scratchBF16(maxDFF), scratchBF16(maxDFF)
		x2, x3, x3s, inner := scratchBF16(maxGelu), scratchBF16(maxGelu), scratchBF16(maxGelu), scratchBF16(maxGelu)
		scaled, tnh, onePlus, halfG := scratchBF16(maxGelu), scratchBF16(maxGelu), scratchBF16(maxGelu), scratchBF16(maxGelu)
		gelu, gated, down := scratchBF16(maxGelu), scratchBF16(maxGelu), scratchBF16(dModel)
		c044 := sharedBytes(bf16ConstBytes(maxGelu, 0.044715))
		c079 := sharedBytes(bf16ConstBytes(maxGelu, 0.7978845608028654))
		c1c := sharedBytes(bf16ConstBytes(maxGelu, 1.0))
		c05 := sharedBytes(bf16ConstBytes(maxGelu, 0.5))
		var pleInput, pleGate, pleGated, pleProj, pleNorm metal.MTLBuffer
		if hasPLE {
			pleInput = scratchBF16(nLayers * ple.pliDim)
			pleGate, pleGated = scratchBF16(ple.pliDim), scratchBF16(ple.pliDim)
			pleProj, pleNorm = scratchBF16(dModel), scratchBF16(dModel)
		}
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
		ropeBaseB := scalarF32(float32(math.Log2(float64(rope.base))))
		ropeLocalBaseB := scalarF32(float32(math.Log2(float64(rope.localBase))))
		freqStride1B := scalarI64(1)
		gqaB := scalarI32(int32(nHeads / nKVHeads))
		// per-hd attention scalars (QK-norm axis, rope head-stride, SDPA K/V head+seq strides): one set
		// per distinct head dim, shared across layers of that hd, all made resident below.
		type hdScalars struct{ axisHead, ropeMat, khs, kss, vhs, vss metal.MTLBuffer }
		hdScalarBy := make(map[int]hdScalars)
		hdScalarOf := func(hd int) hdScalars {
			s, ok := hdScalarBy[hd]
			if !ok {
				kvd := nKVHeads * hd
				s = hdScalars{
					axisHead: scalarI32(int32(hd)), ropeMat: scalarI64(int64(hd)),
					khs: scalarI64(int64(hd)), kss: scalarI64(int64(kvd)),
					vhs: scalarI64(int64(hd)), vss: scalarI64(int64(kvd)),
				}
				hdScalarBy[hd] = s
			}
			return s
		}
		for li := 0; li < nLayers; li++ {
			hdScalarOf(hdOf(li))
		}
		sdpaScaleB := scalarF32(scale)
		addModelB := scalarI32(int32(dModel))
		var pleCntB metal.MTLBuffer
		if hasPLE {
			pleCntB = scalarI32(int32(ple.pliDim))
		}
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
			offBuf, nGlobalBuf, nSlidingBuf, epsBuf, axisBuf, wsBuf,
			ropeScaleB, ropeBaseB, ropeLocalBaseB, freqStride1B, gqaB, sdpaScaleB, addModelB,
		}
		for _, s := range hdScalarBy {
			resident = append(resident, s.axisHead, s.ropeMat, s.khs, s.kss, s.vhs, s.vss)
		}
		if rope.globalFreqs != nil {
			resident = append(resident, rope.globalFreqs)
		}
		if rope.freqs != nil {
			resident = append(resident, rope.freqs)
		}
		var layerScalarOnes metal.MTLBuffer
		if hasPLE {
			resident = append(resident, pleInput, pleGate, pleGated, pleProj, pleNorm, pleCntB)
			resident = append(resident, ple.resident...)
			for _, b := range ple.postNormBufs {
				resident = append(resident, b)
			}
		}
		if hasLayerScalar {
			layerScalarOnes = sharedBytes(bf16ConstBytes(dModel, 1.0))
			resident = append(resident, layerScalarOnes)
			for _, b := range layerScalarBufs {
				if b != nil {
					resident = append(resident, b)
				}
			}
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
		if hasPLE {
			if gpuHasGeluKernel() {
				opsPerLayer += 5 // qmv gate, fused gelu*pli, qmv proj, rms, residual add
			} else {
				opsPerLayer += 14 // qmv gate, 10-op gelu*pli chain, qmv proj, rms, residual add
			}
		}
		if hasLayerScalar {
			opsPerLayer++
		}
		total := opsPerLayer * nLayers
		icbDesc := metal.NewMTLIndirectCommandBufferDescriptor()
		icbDesc.SetCommandTypes(metal.MTLIndirectCommandTypeConcurrentDispatch)
		icbDesc.SetInheritBuffers(false)
		icbDesc.SetInheritPipelineState(false)
		icbDesc.SetMaxKernelBufferBindCount(16)
		icb := device.NewIndirectCommandBufferWithDescriptorMaxCommandCountOptions(icbDesc, uint(total), metal.MTLResourceStorageModeShared)

		rmsTG := uint(rmsSimdSize * ((((dModel + rmsNReads - 1) / rmsNReads) + rmsSimdSize - 1) / rmsSimdSize))
		headTGOf := func(hd int) uint {
			return uint(rmsSimdSize * ((((hd + rmsNReads - 1) / rmsNReads) + rmsSimdSize - 1) / rmsSimdSize))
		}
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
		setRMSRows := func(c metal.MTLIndirectComputeCommand, in, w, o metal.MTLBuffer, rows, hd int) {
			htg := headTGOf(hd)
			c.SetComputePipelineState(rmsPSO)
			c.SetKernelBufferOffsetAtIndex(in, 0, 0)
			c.SetKernelBufferOffsetAtIndex(w, 0, 1)
			c.SetKernelBufferOffsetAtIndex(o, 0, 2)
			c.SetKernelBufferOffsetAtIndex(epsBuf, 0, 3)
			c.SetKernelBufferOffsetAtIndex(hdScalarOf(hd).axisHead, 0, 4)
			c.SetKernelBufferOffsetAtIndex(wsBuf, 0, 5)
			c.ConcurrentDispatchThreadsThreadsPerThreadgroup(metal.MTLSize{Width: uint(rows) * htg, Height: 1, Depth: 1}, metal.MTLSize{Width: htg, Height: 1, Depth: 1})
		}
		setBinOffsets := func(c metal.MTLIndirectComputeCommand, pso metal.MTLComputePipelineState, a metal.MTLBuffer, aOff uint, b metal.MTLBuffer, bOff uint, o metal.MTLBuffer, oOff uint, cntB metal.MTLBuffer, n int) {
			c.SetComputePipelineState(pso)
			c.SetKernelBufferOffsetAtIndex(a, aOff, 0)
			c.SetKernelBufferOffsetAtIndex(b, bOff, 1)
			c.SetKernelBufferOffsetAtIndex(o, oOff, 2)
			c.SetKernelBufferOffsetAtIndex(cntB, 0, 3)
			c.ConcurrentDispatchThreadsThreadsPerThreadgroup(metal.MTLSize{Width: uint(n), Height: 1, Depth: 1}, metal.MTLSize{Width: elemGroup(n), Height: 1, Depth: 1})
		}
		setBin := func(c metal.MTLIndirectComputeCommand, pso metal.MTLComputePipelineState, a, b, o, cntB metal.MTLBuffer, n int) {
			setBinOffsets(c, pso, a, 0, b, 0, o, 0, cntB, n)
		}
		setRope := func(c metal.MTLIndirectComputeCommand, in, out metal.MTLBuffer, heads, li int) {
			// per-layer rope, matching the host stepToken pick (decode_forward_arch.go): sliding →
			// localBase/rotaryDimLocal; proportional-global → the globalFreqs spectrum over globalHeadDim;
			// else base/rotaryDim. A uniform icbRope collapses every branch to base/headDim (byte-identical).
			hd := hdOf(li)
			baseBuf, rotDim, freqs := ropeBaseB, rope.rotaryDim, rope.freqs
			if specs[li].Attention == model.SlidingAttention {
				baseBuf, rotDim, freqs = ropeLocalBaseB, rope.rotaryDimLocal, rope.freqs
			} else if rope.globalFreqs != nil {
				rotDim, freqs = rope.globalHeadDim, rope.globalFreqs
			}
			if rotDim <= 0 || rotDim > hd {
				rotDim = hd
			}
			d0 := uint(rotDim / 2)
			c.SetKernelBufferOffsetAtIndex(in, 0, 0)
			c.SetKernelBufferOffsetAtIndex(out, 0, 1)
			c.SetKernelBufferOffsetAtIndex(offBuf, 0, 2)
			c.SetKernelBufferOffsetAtIndex(ropeScaleB, 0, 3)
			c.SetKernelBufferOffsetAtIndex(hdScalarOf(hd).ropeMat, 0, 4)
			if freqs != nil {
				c.SetComputePipelineState(ropeFreqsPSO)
				c.SetKernelBufferOffsetAtIndex(freqs, 0, 10)
				c.SetKernelBufferOffsetAtIndex(freqStride1B, 0, 11)
			} else {
				c.SetComputePipelineState(ropePSO)
				c.SetKernelBufferOffsetAtIndex(baseBuf, 0, 10)
			}
			c.ConcurrentDispatchThreadsThreadsPerThreadgroup(metal.MTLSize{Width: d0, Height: uint(heads), Depth: 1}, metal.MTLSize{Width: d0, Height: 1, Depth: 1})
		}
		layerScalarFor := func(li int) metal.MTLBuffer {
			if li < len(layerScalarBufs) && layerScalarBufs[li] != nil {
				return layerScalarBufs[li]
			}
			return layerScalarOnes
		}

		// per-layer commands whose bindings advance per token
		kRopeIdx := make([]int, nLayers) // owner cache-write (K) op index — re-acquired per token
		vIdx := make([]int, nLayers)     // owner cache-write (V) op index
		vNormIdx := make([]int, nLayers) // owner value-norm op index (rebound/token)
		sdpaIdx := make([]int, nLayers)  // SDPA op index (sliding: read offset rebound/token)

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
			sliding := specs[li].Attention == model.SlidingAttention
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
				setRMSRows(emit(), q, qNormBufs[li], q, nHeads, hdOf(li))
			}
			setRope(emit(), q, qr, nHeads, li)
			recordProj(li, emit(), normed, kProj, 0, projK)
			if hasKN {
				setRMSRows(emit(), kProj, kNormBufs[li], kProj, nKVHeads, hdOf(li))
			}
			if owns {
				ck := emit()
				setRope(ck, kProj, kCaches[li], nKVHeads, li) // -> kCache @ row pos (rebound/token)
				kRopeIdx[li] = opIdx - 1
				cv := emit()
				recordProj(li, cv, normed, vCaches[li], 0, vProjIdx) // -> vCache @ row pos (rebound/token); K==V projects via wK
				vIdx[li] = opIdx - 1
				if valueNormOnes != nil { // gemma4 value-norm on the new V row (per head; rebound/token)
					cvn := emit()
					setRMSRows(cvn, vCaches[li], valueNormOnes, vCaches[li], nKVHeads, hdOf(li))
					vNormIdx[li] = opIdx - 1
				}
			} else {
				setRope(emit(), kProj, kThrow, nKVHeads, li)        // discarded
				recordProj(li, emit(), normed, vThrow, 0, vProjIdx) // discarded
				if valueNormOnes != nil {
					setRMSRows(emit(), vThrow, valueNormOnes, vThrow, nKVHeads, hdOf(li)) // discarded (keeps the op layout uniform)
				}
			}
			// SDPA over the owner's cache; sliding layers read the windowed slice.
			cs := emit()
			sh := hdScalarOf(hdOf(li))
			cs.SetComputePipelineState(sdpaPSOByHd[hdOf(li)])
			cs.SetKernelBufferOffsetAtIndex(qr, 0, 0)
			cs.SetKernelBufferOffsetAtIndex(attendK, 0, 1) // read offset rebound/token if sliding
			cs.SetKernelBufferOffsetAtIndex(attendV, 0, 2)
			cs.SetKernelBufferOffsetAtIndex(attn, 0, 3)
			cs.SetKernelBufferOffsetAtIndex(gqaB, 0, 4)
			cs.SetKernelBufferOffsetAtIndex(nBufForLayer, 0, 5)
			cs.SetKernelBufferOffsetAtIndex(sh.khs, 0, 6)
			cs.SetKernelBufferOffsetAtIndex(sh.kss, 0, 7)
			cs.SetKernelBufferOffsetAtIndex(sh.vhs, 0, 8)
			cs.SetKernelBufferOffsetAtIndex(sh.vss, 0, 9)
			cs.SetKernelBufferOffsetAtIndex(sdpaScaleB, 0, 10)
			cs.ConcurrentDispatchThreadgroupsThreadsPerThreadgroup(metal.MTLSize{Width: uint(nHeads), Height: 1, Depth: 1}, metal.MTLSize{Width: 1024, Height: 1, Depth: 1})
			sdpaIdx[li] = opIdx - 1
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
			if hasPLE {
				ple.recordGate(li, emit(), outBuf, pleGate)
				pleOff := uint(li * ple.pliDim * bf16Size)
				if gpuHasGeluKernel() {
					cg := emit()
					cg.SetComputePipelineState(geluICBPSO)
					cg.SetKernelBufferOffsetAtIndex(pleGate, 0, 0)
					cg.SetKernelBufferOffsetAtIndex(pleInput, pleOff, 1)
					cg.SetKernelBufferOffsetAtIndex(pleGated, 0, 2)
					cg.SetKernelBufferOffsetAtIndex(pleCntB, 0, 3)
					cg.ConcurrentDispatchThreadsThreadsPerThreadgroup(metal.MTLSize{Width: uint(ple.pliDim), Height: 1, Depth: 1}, metal.MTLSize{Width: elemGroup(ple.pliDim), Height: 1, Depth: 1})
				} else {
					setBin(emit(), mulPSO, pleGate, pleGate, x2, pleCntB, ple.pliDim)
					setBin(emit(), mulPSO, x2, pleGate, x3, pleCntB, ple.pliDim)
					setBin(emit(), mulPSO, x3, c044, x3s, pleCntB, ple.pliDim)
					setBin(emit(), addPSO, pleGate, x3s, inner, pleCntB, ple.pliDim)
					setBin(emit(), mulPSO, inner, c079, scaled, pleCntB, ple.pliDim)
					ct := emit()
					ct.SetComputePipelineState(tanhPSO)
					ct.SetKernelBufferOffsetAtIndex(scaled, 0, 0)
					ct.SetKernelBufferOffsetAtIndex(tnh, 0, 1)
					ct.SetKernelBufferOffsetAtIndex(pleCntB, 0, 2)
					ct.ConcurrentDispatchThreadsThreadsPerThreadgroup(metal.MTLSize{Width: uint(ple.pliDim), Height: 1, Depth: 1}, metal.MTLSize{Width: elemGroup(ple.pliDim), Height: 1, Depth: 1})
					setBin(emit(), addPSO, tnh, c1c, onePlus, pleCntB, ple.pliDim)
					setBin(emit(), mulPSO, pleGate, c05, halfG, pleCntB, ple.pliDim)
					setBin(emit(), mulPSO, halfG, onePlus, gelu, pleCntB, ple.pliDim)
					setBinOffsets(emit(), mulPSO, gelu, 0, pleInput, pleOff, pleGated, 0, pleCntB, ple.pliDim)
				}
				ple.recordProj(li, emit(), pleGated, pleProj)
				setRMS(emit(), pleProj, ple.postNormBufs[li], pleNorm)
				setBin(emit(), addPSO, outBuf, pleNorm, outBuf, addModelB, dModel)
			}
			if hasLayerScalar {
				setBin(emit(), mulPSO, outBuf, layerScalarFor(li), outBuf, addModelB, dModel)
			}
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

		plePliDim, pleRuntime := 0, (*archDecodePLEInputs)(nil)
		if hasPLE {
			plePliDim, pleRuntime = ple.pliDim, ple.runtime
		}
		rowBytesByLayer := make([]int, nLayers)
		for li := 0; li < nLayers; li++ {
			rowBytesByLayer[li] = kvdOf(li) * bf16Size
		}
		r = &archICBReplay{
			icb: icb, rng: rng, residentRes: residentRes,
			specs: specs, nLayers: nLayers, vOutBind: vOutBind, hasValueNorm: valueNormOnes != nil,
			kRopeIdx: kRopeIdx, vIdx: vIdx, vNormIdx: vNormIdx, sdpaIdx: sdpaIdx,
			kCaches: kCaches, vCaches: vCaches,
			offBuf: offBuf, nGlobalBuf: nGlobalBuf, nSlidingBuf: nSlidingBuf,
			ping0: ping[0], lastOut: lastOut, pleInput: pleInput,
			hasPLE: hasPLE, plePliDim: plePliDim, pleRuntime: pleRuntime,
			rowBytes: rowBytesByLayer, slidingWindow: slidingWindow, dModel: dModel,
		}
	})
	if coreErr != nil {
		return nil, coreErr
	}
	return r, nil
}

// decodeForwardArchICBCore records the arch ICB then replays it across the whole input sequence —
// the batch encode-bypass. It is recordArchICB + runBatch; byte-identical to the pre-split core.
func decodeForwardArchICBCore(
	inputs [][]byte, specs []model.LayerSpec,
	anwBufs, mnwBufs, kCaches, vCaches, projResident []metal.MTLBuffer,
	qNormBufs, kNormBufs, postAttnBufs, postFFBufs []metal.MTLBuffer,
	layerScalarBufs []metal.MTLBuffer, ple *archICBPLEPlan,
	recordProj func(li int, c metal.MTLIndirectComputeCommand, vec, out metal.MTLBuffer, outOff uint, p projIndex),
	vOutBind uint, valueNormOnes metal.MTLBuffer, vProjIdx projIndex,
	dModel, nHeads, nKVHeads, headDim, dFF, maxLen, slidingWindow int,
	perLayerDFF []int,
	base, scale, eps float32,
) ([][]byte, error) {
	r, err := recordArchICB(specs, anwBufs, mnwBufs, kCaches, vCaches, projResident, qNormBufs, kNormBufs, postAttnBufs, postFFBufs, layerScalarBufs, ple, recordProj, vOutBind, valueNormOnes, vProjIdx, dModel, nHeads, nKVHeads, headDim, dFF, maxLen, slidingWindow, perLayerDFF, simpleICBRope(base, headDim), scale, eps)
	if err != nil {
		return nil, err
	}
	return r.runBatch(inputs)
}

// DecodeForwardArchICB is the bf16 ARCH-driven cache-grow ICB: the encode-bypass replay
// of DecodeForwardArch (KV-share + sliding-window), recorded once and replayed per token.
// It builds a gemv recorder + the per-layer weight/cache buffers (caches for OWNER layers
// only) and runs decodeForwardArchICBCore. Byte-for-byte equal to DecodeForwardArch on
// the same arch (gated). MoE layers are not supported (rejected). All bf16.
func DecodeForwardArchICB(
	inputs [][]byte, layers []DecodeLayerWeights, specs []model.LayerSpec,
	dModel, nHeads, nKVHeads, headDim, maxLen, dFF, slidingWindow int,
	base, scale, eps float32, valueNorm bool,
	pleArgs ...ArchPLEBF16,
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
	plePayload, err := singleArchPLEBF16("native.DecodeForwardArchICB", pleArgs)
	if err != nil {
		return nil, err
	}
	pleRuntime, pliDim, err := archPLEBF16Runtime("native.DecodeForwardArchICB", plePayload, nLayers, T, dModel, eps)
	if err != nil {
		return nil, err
	}
	var pleLayers []pleLayer
	if pleRuntime != nil {
		pleLayers, err = bf16PLELayers("native.DecodeForwardArchICB", layers, dModel, pliDim)
		if err != nil {
			return nil, err
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
	var pleGateShape, pleProjShape gemvShape
	if pleRuntime != nil {
		p, bm, bn, sm, tm, e := gemvPSO(dModel, pliDim)
		if e != nil {
			return nil, e
		}
		pleGateShape = gemvShape{p, bm, bn, sm, tm}
		p, bm, bn, sm, tm, e = gemvPSO(pliDim, dModel)
		if e != nil {
			return nil, e
		}
		pleProjShape = gemvShape{p, bm, bn, sm, tm}
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
		layerScalarBufs := make([]metal.MTLBuffer, nLayers)
		kCaches := make([]metal.MTLBuffer, nLayers)
		vCaches := make([]metal.MTLBuffer, nLayers)
		type lw struct{ wq, wk, wv, wo, wg, wu, wd metal.MTLBuffer }
		lb := make([]lw, nLayers)
		type plw struct{ gate, proj metal.MTLBuffer }
		pleLB := make([]plw, nLayers)
		plePostNorms := make([]metal.MTLBuffer, nLayers)
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
			layerScalarBufs[li] = layerScalarBuf(w.LayerScalarW, dModel)
			if specs[li].OwnsCache() {
				kCaches[li] = device.NewBufferWithLengthOptions(cacheBytes, metal.MTLResourceStorageModeShared)
				vCaches[li] = device.NewBufferWithLengthOptions(cacheBytes, metal.MTLResourceStorageModeShared)
			}
			lb[li] = lw{sharedBytes(w.WQ), sharedBytes(w.WK), sharedOrNil(w.WV), sharedBytes(w.WO), sharedBytes(w.WGate), sharedBytes(w.WUp), sharedBytes(w.WDown)}
			projResident = append(projResident, lb[li].wq, lb[li].wk, lb[li].wo, lb[li].wg, lb[li].wu, lb[li].wd)
			if lb[li].wv != nil { // gemma4 K==V layers carry no v_proj
				projResident = append(projResident, lb[li].wv)
			}
			if pleRuntime != nil {
				pleLB[li] = plw{sharedBytes(pleLayers[li].gate.Packed), sharedBytes(pleLayers[li].proj.Packed)}
				plePostNorms[li] = sharedBytes(pleLayers[li].postNorm)
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
		var plePlan *archICBPLEPlan
		if pleRuntime != nil {
			pleGateInB, pleGateOutB, pleGateLdB := scalarI32(int32(dModel)), scalarI32(int32(pliDim)), scalarI32(int32(dModel))
			pleProjInB, pleProjOutB, pleProjLdB := scalarI32(int32(pliDim)), scalarI32(int32(dModel)), scalarI32(int32(pliDim))
			pleResident := []metal.MTLBuffer{pleGateInB, pleGateOutB, pleGateLdB, pleProjInB, pleProjOutB, pleProjLdB}
			for li := range pleLB {
				pleResident = append(pleResident, pleLB[li].gate, pleLB[li].proj)
			}
			plePlan = &archICBPLEPlan{
				runtime: pleRuntime, pliDim: pliDim, postNormBufs: plePostNorms, resident: pleResident,
			}
			plePlan.recordGate = func(li int, c metal.MTLIndirectComputeCommand, vec, out metal.MTLBuffer) {
				g := pleGateShape
				setGemv(c, g.pso, pleLB[li].gate, vec, out, pleGateInB, pleGateOutB, pleGateLdB, 0, pliDim, g.bm, g.bn, g.sm, g.tm)
			}
			plePlan.recordProj = func(li int, c metal.MTLIndirectComputeCommand, vec, out metal.MTLBuffer) {
				g := pleProjShape
				setGemv(c, g.pso, pleLB[li].proj, vec, out, pleProjInB, pleProjOutB, pleProjLdB, 0, dModel, g.bm, g.bn, g.sm, g.tm)
			}
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
		outputs, coreErr = decodeForwardArchICBCore(inputs, specs, anwBufs, mnwBufs, kCaches, vCaches, projResident, qNormBufs, kNormBufs, postAttnBufs, postFFBufs, layerScalarBufs, plePlan, recordProj, 3, valueNormOnes, vProjIdx, dModel, nHeads, nKVHeads, headDim, dFF, maxLen, slidingWindow, lFF, base, scale, eps)
	})
	if coreErr != nil {
		return nil, coreErr
	}
	return outputs, nil
}
