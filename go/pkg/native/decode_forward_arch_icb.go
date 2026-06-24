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
	kRopeBind                         uint // K cache-write buffer index: 1 for plain rope, 2 for the fused qk-norm+rope op
	hasValueNorm                      bool
	kRopeIdx, vIdx, vNormIdx, sdpaIdx []int
	barrierOps                        []int // fine-grained replay: op indices to insert an encoder memory barrier before
	kCaches, vCaches                  []metal.MTLBuffer
	offBuf, nGlobalBuf, nSlidingBuf   metal.MTLBuffer
	ping                              [2]metal.MTLBuffer
	ping0, lastOut, pleInput          metal.MTLBuffer
	hasPLE                            bool
	plePliDim                         int
	pleRuntime                        *archDecodePLEInputs
	opsPerLayer                       uint
	rowBytes                          []int // per-layer KV cache row stride (nKVHeads·hd·bf16Size) — gemma4 global layers are wider
	slidingWindow, dModel             int
}

// stepBody replays the recorded ICB for ONE token at position pos over the growing cache. pli is
// this token's [nLayers·pliDim] PerLayerInputs tensor (nil for non-PLE); the caller computes it
// (ArchSession.StepWithID from the token id, runBatch from the batch token ids). Returns a
// fresh hidden copy (read out of the device buffer, so it survives the caller's pool). The caller
// wraps the call in withAutoreleasePool (StepWithID + runBatch both do).
func (r *archICBReplay) stepBody(inputEmb []byte, pos int, pli []byte) []byte {
	return r.stepBodyResult(inputEmb, pos, pli, true)
}

func (r *archICBReplay) stepBodyNoResult(inputEmb []byte, pos int, pli []byte) {
	r.stepBodyResult(inputEmb, pos, pli, false)
}

// encodeStepBody records this token's ICB replay into the caller-owned `enc` WITHOUT committing, so the
// caller can append more GPU work (the LM head + argmax) to the SAME command buffer and sync once per
// token instead of twice. Returns the device buffer holding this layer-stack's final hidden (r.lastOut),
// which the caller reads after the command buffer completes. Must run inside an autorelease pool.
func (r *archICBReplay) encodeStepBody(enc metal.MTLComputeCommandEncoder, inputEmb []byte, pos int, pli []byte) metal.MTLBuffer {
	r.prepareStep(inputEmb, pos, pli)
	enc.UseResourcesCountUsage(r.residentRes, uint(len(r.residentRes)), metal.MTLResourceUsageRead|metal.MTLResourceUsageWrite)
	enc.ExecuteCommandsInBufferWithRange(r.icb, r.rng)
	return r.lastOut
}

func (r *archICBReplay) stepBodyResult(inputEmb []byte, pos int, pli []byte, readResult bool) []byte {
	r.prepareStep(inputEmb, pos, pli)
	cb := queue.CommandBuffer()
	enc := cb.ComputeCommandEncoder()
	enc.UseResourcesCountUsage(r.residentRes, uint(len(r.residentRes)), metal.MTLResourceUsageRead|metal.MTLResourceUsageWrite)
	if fineGrainedReplay && len(r.barrierOps) > 0 {
		// replay barrier-free ICB ranges with an encoder memory barrier at each recorded dep point —
		// resource-scoped coherency instead of the coarse all-prior drain.
		start := r.rng.Location
		for _, b := range r.barrierOps {
			bb := uint(b)
			enc.ExecuteCommandsInBufferWithRange(r.icb, foundation.NSRange{Location: start, Length: bb - start})
			enc.MemoryBarrierWithScope(metal.MTLBarrierScopeBuffers)
			start = bb
		}
		enc.ExecuteCommandsInBufferWithRange(r.icb, foundation.NSRange{Location: start, Length: r.rng.Location + r.rng.Length - start})
	} else {
		enc.ExecuteCommandsInBufferWithRange(r.icb, r.rng)
	}
	enc.EndEncoding()
	cb.Commit()
	cb.WaitUntilCompleted()
	if pieceTimingOn { // GPU execution span of the replay — vs the wall, splits GPU-side from host submit/wait
		icbGPUNs += int64(float64(cb.GPUEndTime()-cb.GPUStartTime()) * 1e9)
	}
	if !readResult {
		return nil
	}
	out := make([]byte, r.dModel*bf16Size)
	copy(out, unsafe.Slice((*byte)(r.lastOut.Contents()), r.dModel*bf16Size))
	return out
}

func (r *archICBReplay) stepBodyCapture(inputEmb []byte, pos int, pli []byte) (final []byte, perLayer [][]byte) {
	r.prepareStep(inputEmb, pos, pli)
	perLayer = make([][]byte, r.nLayers)
	for li := 0; li < r.nLayers; li++ {
		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		enc.UseResourcesCountUsage(r.residentRes, uint(len(r.residentRes)), metal.MTLResourceUsageRead|metal.MTLResourceUsageWrite)
		enc.ExecuteCommandsInBufferWithRange(r.icb, foundation.NSRange{
			Location: uint(li) * r.opsPerLayer,
			Length:   r.opsPerLayer,
		})
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()
		row := make([]byte, r.dModel*bf16Size)
		copy(row, unsafe.Slice((*byte)(r.ping[(li+1)%2].Contents()), r.dModel*bf16Size))
		perLayer[li] = row
	}
	if len(perLayer) > 0 {
		final = append([]byte(nil), perLayer[len(perLayer)-1]...)
	}
	return final, perLayer
}

func (r *archICBReplay) prepareStep(inputEmb []byte, pos int, pli []byte) {
	r.prepareStepRebind(pos)
	if r.hasPLE && pli != nil {
		want := r.nLayers * r.plePliDim * bf16Size
		copy(unsafe.Slice((*byte)(r.pleInput.Contents()), want), pli)
	}
	copy(unsafe.Slice((*byte)(r.ping0.Contents()), r.dModel*bf16Size), inputEmb)
}

// prepareStepRebind does the position-dependent ICB rebind for one decode step — the offset/window
// counters + per-layer cache-row offsets — WITHOUT writing the input emb/pli. The chained-GPU decode
// path uses this: the next step's emb (→ping0) and pli (→pleInput) are produced on-GPU by the prior
// step's encNextInputsGPU, so the host must not overwrite them, only re-point the caches for `pos`.
func (r *archICBReplay) prepareStepRebind(pos int) {
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
			r.icb.IndirectComputeCommandAtIndex(uint(r.kRopeIdx[li])).SetKernelBufferOffsetAtIndex(r.kCaches[li], rowOff, r.kRopeBind)
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
}

// encodeStepBodyNoInput replays one decode step with the input emb+pli ALREADY in ping0/pleInput (the
// chained-GPU path: produced on-GPU by the prior step's encNextInputsGPU). It rebinds the caches for
// `pos` and replays — no host emb/pli write — returning lastOut (the post-stack hidden).
func (r *archICBReplay) encodeStepBodyNoInput(enc metal.MTLComputeCommandEncoder, pos int) metal.MTLBuffer {
	r.prepareStepRebind(pos)
	enc.UseResourcesCountUsage(r.residentRes, uint(len(r.residentRes)), metal.MTLResourceUsageRead|metal.MTLResourceUsageWrite)
	if fineGrainedReplay && len(r.barrierOps) > 0 {
		// Replay barrier-free ICB ranges separated by a RESOURCE-SCOPED encoder memory barrier at each
		// true dependency — buffer-coherency sync instead of the coarse all-prior SetBarrier full drain,
		// so the tiny decode kernels can pipeline. The ICB must have been recorded barrier-free.
		start := r.rng.Location
		for _, b := range r.barrierOps {
			bb := uint(b)
			enc.ExecuteCommandsInBufferWithRange(r.icb, foundation.NSRange{Location: start, Length: bb - start})
			enc.MemoryBarrierWithScope(metal.MTLBarrierScopeBuffers)
			start = bb
		}
		enc.ExecuteCommandsInBufferWithRange(r.icb, foundation.NSRange{Location: start, Length: r.rng.Location + r.rng.Length - start})
		return r.lastOut
	}
	enc.ExecuteCommandsInBufferWithRange(r.icb, r.rng)
	return r.lastOut
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

// runBatchPipelined replays the sequence DOUBLE-BUFFERED across r and r2 — two ICBs recorded over the
// SAME KV caches. Token t's host prep+submit on rs[t%2] overlaps token t-1's GPU compute on rs[(t-1)%2],
// reclaiming the per-token WaitUntilCompleted/submit/read idle (~40% of the wall — the GPU sits stalled
// between tokens in the serial runBatch). The shared-cache hazard serialises the GPU side correctly
// (token t's attention waits t-1's KV write), so it's byte-identical to runBatch. r2 must be recorded
// against the same caches/runtime as r. ~1.6× on e2b prefill.
func (r *archICBReplay) runBatchPipelined(r2 *archICBReplay, inputs [][]byte) ([][]byte, error) {
	if r.hasPLE && len(r.pleRuntime.tokenIDs) != len(inputs) {
		return nil, core.NewError("native.archICBReplay.runBatchPipelined: PLE token id count must equal inputs")
	}
	rs := [2]*archICBReplay{r, r2}
	outputs := make([][]byte, len(inputs))
	readOut := func(rr *archICBReplay) []byte {
		o := make([]byte, rr.dModel*bf16Size)
		copy(o, unsafe.Slice((*byte)(rr.lastOut.Contents()), rr.dModel*bf16Size))
		return o
	}
	var coreErr error
	withAutoreleasePool(func() {
		var prev *archICBReplay
		var prevCB metal.MTLCommandBuffer
		var prevT int
		for t := range inputs {
			rr := rs[t%2]
			var pli []byte
			if rr.hasPLE {
				p, err := rr.pleRuntime.compute(rr.pleRuntime.tokenIDs[t], inputs[t])
				if err != nil {
					coreErr = err
					return
				}
				if len(p) != rr.nLayers*rr.plePliDim*bf16Size {
					coreErr = core.NewError("native.archICBReplay.runBatchPipelined: PLE tensor size mismatch")
					return
				}
				pli = p
			}
			rr.prepareStep(inputs[t], t, pli)
			cb := queue.CommandBuffer()
			enc := cb.ComputeCommandEncoder()
			enc.UseResourcesCountUsage(rr.residentRes, uint(len(rr.residentRes)), metal.MTLResourceUsageRead|metal.MTLResourceUsageWrite)
			enc.ExecuteCommandsInBufferWithRange(rr.icb, rr.rng)
			enc.EndEncoding()
			cb.Commit() // submit t WITHOUT waiting — overlaps t-1's GPU compute with this host turn
			if prevCB != nil {
				prevCB.WaitUntilCompleted()
				outputs[prevT] = readOut(prev)
			}
			prev, prevCB, prevT = rr, cb, t
		}
		if prevCB != nil {
			prevCB.WaitUntilCompleted()
			outputs[prevT] = readOut(prev)
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
	recordFusedRMSProj func(li int, c metal.MTLIndirectComputeCommand, rawIn, normW, epsB, out metal.MTLBuffer, outOff uint, p projIndex),
	vOutBind uint, valueNormOnes metal.MTLBuffer, vProjIdx projIndex,
	dModel, nHeads, nKVHeads, headDim, dFF, maxLen, slidingWindow int,
	perLayerDFF []int,
	rope icbRope, scale, eps float32,
) (*archICBReplay, error) {
	nLayers := len(anwBufs)
	// per-layer head dim AND kv heads (gemma4 full_attention layers attend with a LARGER head_dim than
	// sliding, and the 12B/31B global layers use MQA — kvHeads=1 — vs GQA on the sliding layers): hdOf(li)
	// / kvOf(li) are the layer's geometry; maxHd·maxKv size the shared attention scratch; each layer binds a
	// per-hd SDPA PSO + a per-(hd,kv) stride/axis set + a per-kv GQA-ratio buffer. Uniform models
	// (maxHd==headDim, maxKv==nKVHeads) are byte-identical to the pre-per-layer recorder.
	hdOf := func(li int) int { return headDimOf(specs[li], headDim) }
	kvOf := func(li int) int { return kvHeadsOf(specs[li], nKVHeads) }
	kvdOf := func(li int) int { return kvOf(li) * hdOf(li) }
	maxHd, maxKv := headDim, nKVHeads
	for li := 0; li < nLayers; li++ {
		if h := hdOf(li); h > maxHd {
			maxHd = h
		}
		if k := kvOf(li); k > maxKv {
			maxKv = k
		}
	}
	maxQd, maxKvd := nHeads*maxHd, maxKv*maxHd
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
	// Fused residual-RMSNorm: gemma4's post-attn / post-FF norm-then-add (out = res + rms(branch)) collapses
	// from two barriered ICB ops (rms in-place + vv_Add) to ONE — removing 2 full-drain barriers/layer (the
	// no-barrier ceiling probe showed each coarse SetBarrier drain costs ~7.5µs at decode batch=1).
	var rmsResPSO metal.MTLComputePipelineState
	useFusedResRMS := gpuHasGeluKernel()
	if useFusedResRMS {
		if rmsResPSO, err = rmsResidualPipelineICB(); err != nil {
			return nil, err
		}
	}
	// Fused per-head QK-norm + RoPE: qNorm+ropeQ (and kNorm+ropeK) collapse from two barriered ICB ops
	// to one — the high-value element-wise fusion (the probe: per-head norms ~+7.5, rope ~+5.5 tok/s).
	// Soft (fall back to the composed pair on miss). Lockstep with the re-encode encQKNormRope (same
	// kernel) so ICB ≡ re-encode stays byte-equal; ~1 ULP from the old composed path.
	var qkRopeICBPSO metal.MTLComputePipelineState
	useFusedQKRope := false
	if gpuHasGeluKernel() { // same custom library as gelu — if that built, this builds (hard, like gelu)
		if qkRopeICBPSO, err = qkNormRopePipelineICB(); err != nil {
			return nil, err
		}
		useFusedQKRope = true
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
		// scalarI32/F32 memoise by value, so a sink-driven op (emitRMSNorm via icbSink) binds the SAME
		// eps/axis/ws buffers these named handles hold — no duplicate scalar buffers, no per-record alloc.
		epsBuf, axisBuf, wsBuf := scalarF32(eps), scalarI32(int32(dModel)), scalarI32(1)
		ropeScaleB := scalarF32(scale)
		ropeBaseB := scalarF32(float32(math.Log2(float64(rope.base))))
		ropeLocalBaseB := scalarF32(float32(math.Log2(float64(rope.localBase))))
		freqStride1B := scalarI64(1)
		// per-kv GQA ratio buffer (nHeads/kvHeads): one per distinct kvHeads (gemma4 12B/31B mix MQA
		// global layers kv=1 with GQA sliding layers kv=8), shared across layers of that kv, resident below.
		gqaBy := make(map[int]metal.MTLBuffer)
		gqaOf := func(kv int) metal.MTLBuffer {
			b, ok := gqaBy[kv]
			if !ok {
				b = scalarI32(int32(nHeads / kv))
				gqaBy[kv] = b
			}
			return b
		}
		// per-hd axis scalars (QK-norm axis = hd, rope head-stride = hd): hd-only, one per distinct head dim.
		type hdAxis struct{ axisHead, ropeMat metal.MTLBuffer }
		hdAxisBy := make(map[int]hdAxis)
		hdAxisOf := func(hd int) hdAxis {
			a, ok := hdAxisBy[hd]
			if !ok {
				a = hdAxis{axisHead: scalarI32(int32(hd)), ropeMat: scalarI64(int64(hd))} // memoised, so emitRMSNormRows binds this same buffer
				hdAxisBy[hd] = a
			}
			return a
		}
		// per-(hd,kv) SDPA strides: head stride = hd, seq stride = kvHeads·hd — the seq stride varies with kv
		// (12B/31B global layers are MQA, kv=1). One set per distinct (hd,kv), all made resident below.
		type sdpaStrides struct{ khs, kss, vhs, vss metal.MTLBuffer }
		sdpaStrideBy := make(map[[2]int]sdpaStrides)
		sdpaStrideOf := func(hd, kv int) sdpaStrides {
			key := [2]int{hd, kv}
			s, ok := sdpaStrideBy[key]
			if !ok {
				kvd := kv * hd
				s = sdpaStrides{khs: scalarI64(int64(hd)), kss: scalarI64(int64(kvd)), vhs: scalarI64(int64(hd)), vss: scalarI64(int64(kvd))}
				sdpaStrideBy[key] = s
			}
			return s
		}
		for li := 0; li < nLayers; li++ {
			hdAxisOf(hdOf(li))
			sdpaStrideOf(hdOf(li), kvOf(li))
			gqaOf(kvOf(li))
		}
		sdpaScaleB := scalarF32(scale)
		addModelB := scalarI32(int32(dModel)) // memoised, so a sink-driven binary op binds this same resident buffer
		var pleCntB metal.MTLBuffer
		if hasPLE {
			pleCntB = scalarI32(int32(ple.pliDim)) // memoised, so the sink-driven PLE gelu binds this same resident buffer
		}
		// per-distinct-dFF element-count buffers (the FFN binary/gelu/tanh ops take the count
		// as a buffer): one scalar per distinct width, shared across layers of that width. Every
		// one is appended to resident below so the ICB replay's UseResources covers it — a
		// non-resident count buffer is read as garbage on the layer that uses it.
		ffCntBufs := make(map[int]metal.MTLBuffer)
		ffCntOf := func(n int) metal.MTLBuffer {
			b, ok := ffCntBufs[n]
			if !ok {
				b = scalarI32(int32(n)) // memoised; still tracked here for residency
				ffCntBufs[n] = b
			}
			return b
		}
		for li := 0; li < nLayers; li++ {
			ffCntOf(lffOf(li))
		}
		// fused QK-norm+rope per-layer params: ropeParamsOf mirrors setRope's per-layer base/rotDim/freqs
		// pick; a rotary-dim scalar per distinct rotaryDim + the use-freqs flags + a dummy periods buffer,
		// all made resident below (a non-resident param buffer reads garbage on the layer that uses it).
		ropeParamsOf := func(li int) (baseBuf, freqs metal.MTLBuffer, rotDim int) {
			hd := hdOf(li)
			baseBuf, rotDim, freqs = ropeBaseB, rope.rotaryDim, rope.freqs
			if specs[li].Attention == model.SlidingAttention {
				baseBuf, rotDim, freqs = ropeLocalBaseB, rope.rotaryDimLocal, rope.freqs
			} else if rope.globalFreqs != nil {
				rotDim, freqs = rope.globalHeadDim, rope.globalFreqs
			}
			if rotDim <= 0 || rotDim > hd {
				rotDim = hd
			}
			return
		}
		rotDimBufs := make(map[int]metal.MTLBuffer)
		rotDimBufOf := func(rd int) metal.MTLBuffer {
			b, ok := rotDimBufs[rd]
			if !ok {
				b = scalarI32(int32(rd))
				rotDimBufs[rd] = b
			}
			return b
		}
		useFreqs0B, useFreqs1B := scalarI32(0), scalarI32(1)
		qkDummyPeriodsB := qkRopeDummyBuf()
		if useFusedQKRope {
			for li := 0; li < nLayers; li++ {
				_, _, rd := ropeParamsOf(li)
				rotDimBufOf(rd)
			}
		}

		resident := []metal.MTLBuffer{
			ping[0], ping[1], normed, q, qr, kProj, attn, attnOut, kThrow, vThrow, mlpNormed,
			gate, up, x2, x3, x3s, inner, scaled, tnh, onePlus, halfG, gelu, gated, down,
			c044, c079, c1c, c05,
			offBuf, nGlobalBuf, nSlidingBuf, epsBuf, axisBuf, wsBuf,
			ropeScaleB, ropeBaseB, ropeLocalBaseB, freqStride1B, sdpaScaleB, addModelB,
		}
		for _, a := range hdAxisBy {
			resident = append(resident, a.axisHead, a.ropeMat)
		}
		for _, s := range sdpaStrideBy {
			resident = append(resident, s.khs, s.kss, s.vhs, s.vss)
		}
		for _, b := range gqaBy {
			resident = append(resident, b)
		}
		if rope.globalFreqs != nil {
			resident = append(resident, rope.globalFreqs)
		}
		if rope.freqs != nil {
			resident = append(resident, rope.freqs)
		}
		resident = append(resident, useFreqs0B, useFreqs1B, qkDummyPeriodsB)
		for _, b := range rotDimBufs {
			resident = append(resident, b)
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
		// fused QK-norm+rope collapses (qNorm + ropeQ) and (kNorm + ropeK) from 2 ops to 1 each when the
		// layer has QK-norm. The fused K op writes the cache at buffer index 2 (its `out`), not the plain
		// rope's index 1 — so the per-token kRopeIdx rebind (prepareStep) uses kRopeBindIdx.
		kRopeBindIdx := uint(1)
		if useFusedQKRope && hasQN {
			opsPerLayer-- // qNorm+ropeQ
		}
		if useFusedQKRope && hasKN {
			opsPerLayer-- // kNorm+ropeK
			kRopeBindIdx = 2
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
		// fused input-RMSNorm+qmv folds the attn-input rms and the mlp-input rms INTO their following
		// projections (Q/K/V read inBuf+attnNormW; gate/up read hBuf+mlpNormW), removing both setRMS ops.
		if recordFusedRMSProj != nil {
			opsPerLayer -= 2
		}
		// fused residual-RMSNorm folds each post-norm + its residual add into one op (out = res + rms(branch)).
		if useFusedResRMS {
			if hasPA {
				opsPerLayer--
			}
			if hasPF {
				opsPerLayer--
			}
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
		// full-dModel RMSNorm through the SHARED emitRMSNorm body (the same one encRMSNormBF16 drives) via
		// icbSink — the path-unifying dispatchSink, one math recorded into both the encoder and the ICB.
		// icbSink binds eps/axis/ws as the memoised scalar buffers (== epsBuf/axisBuf/wsBuf bound above).
		setRMS := func(c metal.MTLIndirectComputeCommand, in, w, o metal.MTLBuffer) {
			emitRMSNorm(icbSink{c}, rmsPSO, in, w, o, 0, dModel, eps, rmsTG)
		}
		// fused post-norm tail out = res + rmsnorm(x, w) in ONE ICB command (lthn_rmsnorm_residual_bf16,
		// one fewer barrier than RMS + vv_Add) through the SHARED emitRMSNormResidual body.
		setRMSResidual := func(c metal.MTLIndirectComputeCommand, x, w, res, o metal.MTLBuffer) {
			emitRMSNormResidual(icbSink{c}, rmsResPSO, x, w, res, o, 0, dModel, eps, rmsTG)
		}
		// per-head RMSNorm (gemma4 QK-norm: rows of headDim each) through the SHARED emitRMSNormRows body;
		// axisSize = hd binds the same memoised buffer hdAxisOf(hd).axisHead holds.
		setRMSRows := func(c metal.MTLIndirectComputeCommand, in, w, o metal.MTLBuffer, rows, hd int) {
			emitRMSNormRows(icbSink{c}, rmsPSO, in, w, o, 0, 0, 0, hd, eps, rows, headTGOf(hd))
		}
		// element-wise binary op through the SHARED emitBinary body (with encBinaryDT). The count binds the
		// memoised scalar buffer addModelB/ffCntOf hold — no separate count param.
		setBinOffsets := func(c metal.MTLIndirectComputeCommand, pso metal.MTLComputePipelineState, a metal.MTLBuffer, aOff uint, b metal.MTLBuffer, bOff uint, o metal.MTLBuffer, oOff uint, n int) {
			emitBinary(icbSink{c}, pso, a, aOff, b, bOff, o, oOff, n)
		}
		setBin := func(c metal.MTLIndirectComputeCommand, pso metal.MTLComputePipelineState, a, b, o metal.MTLBuffer, n int) {
			setBinOffsets(c, pso, a, 0, b, 0, o, 0, n)
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
			c.SetKernelBufferOffsetAtIndex(hdAxisOf(hd).ropeMat, 0, 4)
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
		// setQKNormRope records the fused per-head QK-norm + RoPE (out = RoPE(RMSNorm(in, w))) in ONE op:
		// per-head rms then rotate, replacing setRMSRows+setRope. One threadgroup per head, hd threads.
		// in/out byte offsets carry the K cache row when fusing K (the projection wrote it there).
		setQKNormRope := func(c metal.MTLIndirectComputeCommand, in metal.MTLBuffer, inOff uint, w metal.MTLBuffer, out metal.MTLBuffer, outOff uint, heads, li int) {
			hd := hdOf(li)
			baseBuf, freqs, rd := ropeParamsOf(li)
			c.SetComputePipelineState(qkRopeICBPSO)
			c.SetKernelBufferOffsetAtIndex(in, inOff, 0)
			c.SetKernelBufferOffsetAtIndex(w, 0, 1)
			c.SetKernelBufferOffsetAtIndex(out, outOff, 2)
			c.SetKernelBufferOffsetAtIndex(epsBuf, 0, 3)
			c.SetKernelBufferOffsetAtIndex(hdAxisOf(hd).axisHead, 0, 4)
			c.SetKernelBufferOffsetAtIndex(rotDimBufOf(rd), 0, 5)
			c.SetKernelBufferOffsetAtIndex(ropeScaleB, 0, 6)
			c.SetKernelBufferOffsetAtIndex(offBuf, 0, 7)
			c.SetKernelBufferOffsetAtIndex(baseBuf, 0, 8)
			if freqs != nil {
				c.SetKernelBufferOffsetAtIndex(freqs, 0, 9)
				c.SetKernelBufferOffsetAtIndex(useFreqs1B, 0, 10)
			} else {
				c.SetKernelBufferOffsetAtIndex(qkDummyPeriodsB, 0, 9)
				c.SetKernelBufferOffsetAtIndex(useFreqs0B, 0, 10)
			}
			c.ConcurrentDispatchThreadsThreadsPerThreadgroup(metal.MTLSize{Width: uint(heads) * uint(hd), Height: 1, Depth: 1}, metal.MTLSize{Width: uint(hd), Height: 1, Depth: 1})
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
		var barrierOps []int // op indices that carry a barrier-before — used by the fine-grained replay
		emit := func() metal.MTLIndirectComputeCommand {
			c := icb.IndirectComputeCommandAtIndex(uint(opIdx))
			if opIdx != 0 {
				if fineGrainedReplay {
					// record barrier-free; the replay enforces the dep with an encoder memory barrier
					// (resource-scoped, may pipeline) instead of the coarse all-prior ICB SetBarrier.
					barrierOps = append(barrierOps, opIdx)
				} else if !allBarriersOffForTest { // allBarriersOff: TIMING-ONLY ceiling probe (output races/garbage)
					c.SetBarrier()
				}
			}
			opIdx++
			return c
		}
		// emitNB records a command WITHOUT a barrier — for an INDEPENDENT SECONDARY consumer of a
		// producer whose FIRST consumer already barriered (and so flushed) it. The op reads the
		// already-visible producer and overlaps its sibling ops instead of draining the pipeline.
		// q/kProj/vProj all read `normed` (q barriers, kProj+vProj ride free); gate/up read
		// `mlpNormed` (gate barriers, up rides free — the big FFN-gemv overlap). Each op that READS
		// one of these (kNorm, kRope, valueNorm, SDPA, gelu) still barriers, so the only relaxed
		// ordering is sibling-vs-sibling, which has no data hazard. Byte-parity-gated.
		emitNB := func() metal.MTLIndirectComputeCommand {
			c := icb.IndirectComputeCommandAtIndex(uint(opIdx))
			opIdx++
			return c
		}
		// emitFFN is emit() in production but emitNB() under ffnBarriersOffForTest — the FFN-only no-barrier
		// ceiling probe (racy output; measures the GPU-span a fused FFN megakernel could reclaim).
		emitFFN := func() metal.MTLIndirectComputeCommand {
			if ffnBarriersOffForTest {
				return emitNB()
			}
			return emit()
		}
		// recInputProj records an input-rms-fed projection (Q/K/V/gate/up): the FUSED rms+qmv (rms folded
		// in, reads rawIn+normW) when available, else the plain projection over the pre-normed buffer. The
		// caller emits the command (emit/emitNB) so the barrier structure stays visible at the call site,
		// and emits-or-skips the matching setRMS itself.
		recInputProj := func(c metal.MTLIndirectComputeCommand, li int, rawIn, normW, normed, out metal.MTLBuffer, outOff uint, p projIndex) {
			if recordFusedRMSProj != nil {
				recordFusedRMSProj(li, c, rawIn, normW, epsBuf, out, outOff, p)
			} else {
				recordProj(li, c, normed, out, outOff, p)
			}
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
			if recordFusedRMSProj == nil { // fused path folds this rms into q/kProj/vProj below
				setRMS(emit(), inBuf, anwBufs[li], normed)
			}
			recInputProj(emit(), li, inBuf, anwBufs[li], normed, q, 0, projQ)
			if useFusedQKRope && hasQN { // fused: qr = RoPE(RMSNorm(q, qNormW)) in one op
				setQKNormRope(emit(), q, 0, qNormBufs[li], qr, 0, nHeads, li)
			} else {
				if hasQN { // gemma4 per-head QK-norm on Q before RoPE (in-place)
					setRMSRows(emit(), q, qNormBufs[li], q, nHeads, hdOf(li))
				}
				setRope(emit(), q, qr, nHeads, li)
			}
			recInputProj(emitNB(), li, inBuf, anwBufs[li], normed, kProj, 0, projK) // 2nd consumer (q barriered it) — overlap
			fuseK := useFusedQKRope && hasKN                                        // fuse kNorm+ropeK into one op (writes the cache at buf 2)
			if owns {
				if fuseK {
					ck := emit()
					setQKNormRope(ck, kProj, 0, kNormBufs[li], kCaches[li], 0, kvOf(li), li) // kNorm+rope -> kCache @ row pos (rebound/token)
					kRopeIdx[li] = opIdx - 1
				} else {
					if hasKN {
						setRMSRows(emit(), kProj, kNormBufs[li], kProj, kvOf(li), hdOf(li))
					}
					ck := emit()
					setRope(ck, kProj, kCaches[li], kvOf(li), li) // -> kCache @ row pos (rebound/token)
					kRopeIdx[li] = opIdx - 1
				}
				cv := emitNB()                                                             // 2nd consumer of `normed` (q barriered it) — overlap
				recInputProj(cv, li, inBuf, anwBufs[li], normed, vCaches[li], 0, vProjIdx) // -> vCache @ row pos (rebound/token); K==V projects via wK
				vIdx[li] = opIdx - 1
				if valueNormOnes != nil { // gemma4 value-norm on the new V row (per head; rebound/token)
					cvn := emit()
					setRMSRows(cvn, vCaches[li], valueNormOnes, vCaches[li], kvOf(li), hdOf(li))
					vNormIdx[li] = opIdx - 1
				}
			} else {
				if fuseK {
					setQKNormRope(emit(), kProj, 0, kNormBufs[li], kThrow, 0, kvOf(li), li) // kNorm+rope -> discard
				} else {
					if hasKN {
						setRMSRows(emit(), kProj, kNormBufs[li], kProj, kvOf(li), hdOf(li))
					}
					setRope(emit(), kProj, kThrow, kvOf(li), li) // discarded
				}
				recInputProj(emitNB(), li, inBuf, anwBufs[li], normed, vThrow, 0, vProjIdx) // discarded; 2nd consumer of `normed` — overlap
				if valueNormOnes != nil {
					setRMSRows(emit(), vThrow, valueNormOnes, vThrow, kvOf(li), hdOf(li)) // discarded (keeps the op layout uniform)
				}
			}
			// SDPA over the owner's cache; sliding layers read the windowed slice.
			cs := emit()
			sh := sdpaStrideOf(hdOf(li), kvOf(li))
			cs.SetComputePipelineState(sdpaPSOByHd[hdOf(li)])
			cs.SetKernelBufferOffsetAtIndex(qr, 0, 0)
			cs.SetKernelBufferOffsetAtIndex(attendK, 0, 1) // read offset rebound/token if sliding
			cs.SetKernelBufferOffsetAtIndex(attendV, 0, 2)
			cs.SetKernelBufferOffsetAtIndex(attn, 0, 3)
			cs.SetKernelBufferOffsetAtIndex(gqaOf(kvOf(li)), 0, 4)
			cs.SetKernelBufferOffsetAtIndex(nBufForLayer, 0, 5)
			cs.SetKernelBufferOffsetAtIndex(sh.khs, 0, 6)
			cs.SetKernelBufferOffsetAtIndex(sh.kss, 0, 7)
			cs.SetKernelBufferOffsetAtIndex(sh.vhs, 0, 8)
			cs.SetKernelBufferOffsetAtIndex(sh.vss, 0, 9)
			cs.SetKernelBufferOffsetAtIndex(sdpaScaleB, 0, 10)
			cs.ConcurrentDispatchThreadgroupsThreadsPerThreadgroup(metal.MTLSize{Width: uint(nHeads), Height: 1, Depth: 1}, metal.MTLSize{Width: 1024, Height: 1, Depth: 1})
			sdpaIdx[li] = opIdx - 1
			recordProj(li, emit(), attn, attnOut, 0, projO)
			if hasPA && useFusedResRMS { // fused: hBuf = inBuf + rms(Wo·attn) — one op, one fewer barrier
				setRMSResidual(emit(), attnOut, postAttnBufs[li], inBuf, hBuf)
			} else {
				if hasPA { // gemma4 post-attention norm on Wo·attn before the residual (in-place)
					setRMS(emit(), attnOut, postAttnBufs[li], attnOut)
				}
				setBin(emit(), addPSO, inBuf, attnOut, hBuf, dModel)
			}

			// --- MLP half --- (lff = this layer's FFN width; the FFN ops dispatch only lff
			// elements + bind this width's count buffer — gemma4 MatFormer varies it per layer)
			lff := lffOf(li)
			ffCntB := ffCntOf(lff)
			if recordFusedRMSProj == nil { // fused path folds this rms into gate/up below
				setRMS(emit(), hBuf, mnwBufs[li], mlpNormed)
			}
			recInputProj(emitFFN(), li, hBuf, mnwBufs[li], mlpNormed, gate, 0, projGate)
			recInputProj(emitNB(), li, hBuf, mnwBufs[li], mlpNormed, up, 0, projUp) // 2nd consumer of `mlpNormed` (gate barriered it) — overlap gate
			if gpuHasGeluKernel() {                                                 // fused gelu(gate)·up — one ICB command (ffCntB = lff as the n buffer)
				cg := emitFFN()
				cg.SetComputePipelineState(geluICBPSO)
				cg.SetKernelBufferOffsetAtIndex(gate, 0, 0)
				cg.SetKernelBufferOffsetAtIndex(up, 0, 1)
				cg.SetKernelBufferOffsetAtIndex(gated, 0, 2)
				cg.SetKernelBufferOffsetAtIndex(ffCntB, 0, 3)
				cg.ConcurrentDispatchThreadsThreadsPerThreadgroup(metal.MTLSize{Width: uint(lff), Height: 1, Depth: 1}, metal.MTLSize{Width: elemGroup(lff), Height: 1, Depth: 1})
			} else {
				setBin(emit(), mulPSO, gate, gate, x2, lff)
				setBin(emit(), mulPSO, x2, gate, x3, lff)
				setBin(emit(), mulPSO, x3, c044, x3s, lff)
				setBin(emit(), addPSO, gate, x3s, inner, lff)
				setBin(emit(), mulPSO, inner, c079, scaled, lff)
				ct := emit()
				ct.SetComputePipelineState(tanhPSO)
				ct.SetKernelBufferOffsetAtIndex(scaled, 0, 0)
				ct.SetKernelBufferOffsetAtIndex(tnh, 0, 1)
				ct.SetKernelBufferOffsetAtIndex(ffCntB, 0, 2)
				ct.ConcurrentDispatchThreadsThreadsPerThreadgroup(metal.MTLSize{Width: uint(lff), Height: 1, Depth: 1}, metal.MTLSize{Width: elemGroup(lff), Height: 1, Depth: 1})
				setBin(emit(), addPSO, tnh, c1c, onePlus, lff)
				setBin(emit(), mulPSO, gate, c05, halfG, lff)
				setBin(emit(), mulPSO, halfG, onePlus, gelu, lff)
				setBin(emit(), mulPSO, gelu, up, gated, lff)
			}
			recordProj(li, emitFFN(), gated, down, 0, projDown)
			if hasPF && useFusedResRMS { // fused: outBuf = hBuf + rms(Wdown·…) — one op, one fewer barrier
				setRMSResidual(emit(), down, postFFBufs[li], hBuf, outBuf)
			} else {
				if hasPF { // gemma4 post-feed-forward norm on Wdown·… before the residual (in-place)
					setRMS(emit(), down, postFFBufs[li], down)
				}
				setBin(emit(), addPSO, hBuf, down, outBuf, dModel)
			}
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
					setBin(emit(), mulPSO, pleGate, pleGate, x2, ple.pliDim)
					setBin(emit(), mulPSO, x2, pleGate, x3, ple.pliDim)
					setBin(emit(), mulPSO, x3, c044, x3s, ple.pliDim)
					setBin(emit(), addPSO, pleGate, x3s, inner, ple.pliDim)
					setBin(emit(), mulPSO, inner, c079, scaled, ple.pliDim)
					ct := emit()
					ct.SetComputePipelineState(tanhPSO)
					ct.SetKernelBufferOffsetAtIndex(scaled, 0, 0)
					ct.SetKernelBufferOffsetAtIndex(tnh, 0, 1)
					ct.SetKernelBufferOffsetAtIndex(pleCntB, 0, 2)
					ct.ConcurrentDispatchThreadsThreadsPerThreadgroup(metal.MTLSize{Width: uint(ple.pliDim), Height: 1, Depth: 1}, metal.MTLSize{Width: elemGroup(ple.pliDim), Height: 1, Depth: 1})
					setBin(emit(), addPSO, tnh, c1c, onePlus, ple.pliDim)
					setBin(emit(), mulPSO, pleGate, c05, halfG, ple.pliDim)
					setBin(emit(), mulPSO, halfG, onePlus, gelu, ple.pliDim)
					setBinOffsets(emit(), mulPSO, gelu, 0, pleInput, pleOff, pleGated, 0, ple.pliDim)
				}
				ple.recordProj(li, emit(), pleGated, pleProj)
				// (the PLE post-norm residual stays un-fused: the fused kernel diverges ~2 ULP from the
				// PerLayerInputGate* re-encode / its CPU reference on the dModel axis — byte-parity-hostile.)
				setRMS(emit(), pleProj, ple.postNormBufs[li], pleNorm)
				setBin(emit(), addPSO, outBuf, pleNorm, outBuf, dModel)
			}
			if hasLayerScalar {
				setBin(emit(), mulPSO, outBuf, layerScalarFor(li), outBuf, dModel)
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
			specs: specs, nLayers: nLayers, vOutBind: vOutBind, kRopeBind: kRopeBindIdx, hasValueNorm: valueNormOnes != nil,
			kRopeIdx: kRopeIdx, vIdx: vIdx, vNormIdx: vNormIdx, sdpaIdx: sdpaIdx, barrierOps: barrierOps,
			kCaches: kCaches, vCaches: vCaches,
			offBuf: offBuf, nGlobalBuf: nGlobalBuf, nSlidingBuf: nSlidingBuf,
			ping: ping, ping0: ping[0], lastOut: lastOut, pleInput: pleInput,
			hasPLE: hasPLE, plePliDim: plePliDim, pleRuntime: pleRuntime,
			opsPerLayer: uint(opsPerLayer),
			rowBytes:    rowBytesByLayer, slidingWindow: slidingWindow, dModel: dModel,
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
	recordFusedRMSProj func(li int, c metal.MTLIndirectComputeCommand, rawIn, normW, epsB, out metal.MTLBuffer, outOff uint, p projIndex),
	vOutBind uint, valueNormOnes metal.MTLBuffer, vProjIdx projIndex,
	dModel, nHeads, nKVHeads, headDim, dFF, maxLen, slidingWindow int,
	perLayerDFF []int,
	base, scale, eps float32,
) ([][]byte, error) {
	r, err := recordArchICB(specs, anwBufs, mnwBufs, kCaches, vCaches, projResident, qNormBufs, kNormBufs, postAttnBufs, postFFBufs, layerScalarBufs, ple, recordProj, recordFusedRMSProj, vOutBind, valueNormOnes, vProjIdx, dModel, nHeads, nKVHeads, headDim, dFF, maxLen, slidingWindow, perLayerDFF, simpleICBRope(base, headDim), scale, eps)
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
	hasMoE, mixedHeadDim := false, false
	for li := range specs {
		o := specs[li].KVShareFrom
		if o < 0 || o > li || (o != li && !specs[o].OwnsCache()) {
			return nil, core.NewError("native.DecodeForwardArchICB: KVShareFrom must reference an earlier owner layer")
		}
		if specs[li].MoE {
			hasMoE = true
		}
		if headDimOf(specs[li], headDim) != headDim {
			mixedHeadDim = true // gemma4 global layers are WIDER (e.g. 512 vs sliding 256)
		}
	}
	// This whole-sequence recorder records ONE uniform projection shape + a single base-rope spectrum
	// for every layer (qDim/kvDim/psoQ/psoKV and simpleICBRope are computed once below). It therefore
	// cannot represent MoE (host router) NOR gemma4's per-layer head dim (the global layers' wider
	// head_dim + proportional partial rope). For those, fall back to the per-layer-correct re-encode
	// forward — byte-identical, just not the ICB fast path for this (cold, batch) call. The SESSION
	// path keeps the fast per-hd ICB (it records per-head-dim); this is only the whole-seq batch API.
	if hasMoE || mixedHeadDim {
		return DecodeForwardArch(inputs, layers, specs, dModel, nHeads, nKVHeads, headDim, maxLen, dFF, slidingWindow, base, scale, eps, valueNorm, pleArgs...)
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
		valueNormOnes := valueNormOnesBuf(valueNorm, maxHeadDimOf(specs, headDim))
		vProjIdx := projV
		if len(layers[0].WV) == 0 { // gemma4 K==V: V rides the k-proj
			vProjIdx = projK
		}
		outputs, coreErr = decodeForwardArchICBCore(inputs, specs, anwBufs, mnwBufs, kCaches, vCaches, projResident, qNormBufs, kNormBufs, postAttnBufs, postFFBufs, layerScalarBufs, plePlan, recordProj, nil, 3, valueNormOnes, vProjIdx, dModel, nHeads, nKVHeads, headDim, dFF, maxLen, slidingWindow, lFF, base, scale, eps)
	})
	if coreErr != nil {
		return nil, coreErr
	}
	return outputs, nil
}
