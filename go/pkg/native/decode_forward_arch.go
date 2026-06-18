// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"unsafe"

	core "dappco.re/go"
	g4 "dappco.re/go/mlx/pkg/model/gemma4"
	"github.com/tmc/apple/metal"
)

// attnScaleOf is the SDPA scale the model DECLARES (the engine applies it, never
// assumes): gemma4 = 1.0 (its per-head QK-norm is the scaling), standard transformers
// = 1/√headDim. Falls back to 1/√headDim for a hand-built Arch that predates the
// declared field (AttnScale == 0), so existing paths are byte-identical.
func attnScaleOf(arch g4.Arch) float32 {
	if arch.AttnScale != 0 {
		return arch.AttnScale
	}
	return float32(1.0 / math.Sqrt(float64(arch.HeadDim)))
}

// headDimOf / kvHeadsOf are a layer's RESOLVED attention geometry: gemma4 full_attention
// layers use a larger head_dim (global_head_dim) and may differ in KV heads, declared per
// layer on the spec (pkg/model/gemma4). They fall back to the uniform arch value for a spec
// that predates the per-type resolution (a hand-built Arch), so existing uniform paths are
// byte-identical.
func headDimOf(spec g4.LayerSpec, fallback int) int {
	if spec.HeadDim > 0 {
		return spec.HeadDim
	}
	return fallback
}

func kvHeadsOf(spec g4.LayerSpec, fallback int) int {
	if spec.KVHeads > 0 {
		return spec.KVHeads
	}
	return fallback
}

// encAttnHalfShared is the KV-SHARING attention half: a layer that shares another
// layer's KV cache projects ONLY its query (from its own input) and attends over
// the owner's cache — no K/V projection, no K-RoPE, no cache write. attendK/attendV
// are the owner's seq-major caches; the window N=pos+1 is the owner's live length
// (the owner wrote row pos earlier this token). Writes x + Wo·attn -> h.
func encAttnHalfShared(
	enc metal.MTLComputeCommandEncoder,
	x, attnNormW, attendK, attendV, offBuf, h, postAttnNorm, qNorm metal.MTLBuffer,
	sc attnScratch, proj projector,
	dModel, nHeads, nKVHeads, headDim, pos, slideW, rotaryDim int, base, scale, eps float32,
	ropeFreqs metal.MTLBuffer,
) error {
	kvDim := nKVHeads * headDim
	if err := encRMSNormBF16(enc, x, attnNormW, sc.normed, 0, dModel, eps); err != nil {
		return err
	}
	if err := proj.project(enc, sc.normed, sc.q, 0, projQ); err != nil {
		return err
	}
	if qNorm != nil { // gemma4 per-head QK-norm before RoPE (sharers project only Q)
		if err := encRMSNormRowsBF16(enc, sc.q, qNorm, sc.q, 0, 0, 0, nHeads, headDim, eps); err != nil {
			return err
		}
	}
	// RoPE Q in place so partial rotary's untouched tail keeps the projected value.
	if err := encRopeDecode(enc, sc.q, sc.q, 0, 0, offBuf, ropeFreqs, nHeads, headDim, rotaryDim, base, scale); err != nil {
		return err
	}
	// attend the OWNER's cache, windowed (global: all; sliding: last slideW), no write
	start, n := slideWindow(pos, slideW)
	if err := encSDPAStrided(enc, sc.q, attendK, attendV, sc.attn,
		nHeads, nKVHeads, headDim, n,
		int64(headDim), int64(kvDim), int64(headDim), int64(kvDim), scale, uint(start*kvDim*bf16Size)); err != nil {
		return err
	}
	if err := proj.project(enc, sc.attn, sc.attnOut, 0, projO); err != nil {
		return err
	}
	return encResidualMaybeNorm(enc, x, sc.attnOut, sc.normed, h, postAttnNorm, dModel, eps)
}

// archLayerBufs holds one layer's resident buffers for runArchDecode: bf16 norms +
// the (bf16 or 4-bit qmv) projector + the growing KV caches. kCache/vCache are nil for
// sharer layers (they attend the owner's); mnw and the projector's MLP weights are
// unbound for MoE layers (MoEBlockBF16 owns that FFN).
type archLayerBufs struct {
	anw, mnw                 metal.MTLBuffer
	postAttnNorm, postFFNorm metal.MTLBuffer // gemma4 post-attn/post-FF norms (nil = skip)
	qNorm, kNorm             metal.MTLBuffer // gemma4 per-head QK-norm (nil = skip)
	layerScalar              metal.MTLBuffer // gemma4 per-layer output scalar, broadcast to dModel (nil = skip)
	kCache, vCache           metal.MTLBuffer
	proj                     projector
}

// archDecodeState holds the resident buffers of an arch decode — the per-layer weights/
// caches (lb), shared scratch, and the position buffer — so a single token can be stepped
// repeatedly over a PERSISTENT, growing KV cache. Both the whole-sequence runArchDecode and
// the incremental generation loop build one (inside a withAutoreleasePool) and call
// stepToken per token; the caches in lb persist across calls within that pool, which is
// what turns the O(N²) re-decode into O(1)/token incremental decode.
type archDecodeState struct {
	specs        []g4.LayerSpec
	lb           []archLayerBufs
	moeWeights   []*MoELayerWeights
	asc          attnScratch
	msc          mlpScratch
	hBuf, xA, xB metal.MTLBuffer
	offBuf       metal.MTLBuffer
	ropeFreqs    metal.MTLBuffer // resident periods (1/inv_freq) for YaRN long-context rope; nil = base-derived rope
	valueNormOnes metal.MTLBuffer // gemma4 value-norm: [maxHeadDim] ones weight for the no-scale per-head RMSNorm on V; nil = no value-norm (Mistral)

	dModel, nHeads, nKVHeads, headDim, dFF, slidingWindow int
	rotaryDim, rotaryDimLocal                             int     // partial-rotary dims (global / sliding); == headDim is full
	base, localBase, scale, eps                           float32 // localBase = sliding-layer RoPE theta

	// gemma4 per-layer-input tower (E2B/E4B): when ple is non-nil, each layer's output is gated
	// by PerLayerInputGateQuant before layer_scalar, fed its pliDim slice of perLayerInput (the
	// PerLayerInputs tensor, set per token). nil = no PLE tower (dense models — byte-identical).
	ple           []pleLayer
	perLayerInput []byte // [numLayers·pliDim] bf16, set before each token's stepToken
	pliDim        int

	// gemma4 4-bit MoE (26B-A4B): moeQuant[li] != nil runs MoEBlockQuant for that layer's FFN
	// (host-orchestrated like the bf16 MoE). nil entries use the dense MLP / bf16 moeWeights.
	moeQuant []*MoEQuantLayerWeights
}

// pleLayer is one layer's per-layer-input gate weights: the 4-bit gate + projection and the
// bf16 post-norm. A nil postNorm marks a layer with no gate (so a mixed model is fine).
type pleLayer struct {
	gate, proj      QuantWeight
	postNorm        []byte
	groupSize, bits int
}

// newArchDecodeState builds the shared scratch + position buffer over the caller's
// per-layer buffers. MUST be called inside a withAutoreleasePool.
func newArchDecodeState(specs []g4.LayerSpec, lb []archLayerBufs, moeWeights []*MoELayerWeights, dModel, nHeads, nKVHeads, headDim, dFF, slidingWindow, rotaryDim, rotaryDimLocal int, base, localBase, scale, eps float32, valueNorm bool) archDecodeState {
	// scratch must fit the LARGEST layer's q/kv (gemma4 full_attention layers use a
	// bigger head_dim than sliding) — the shared scratch is reused across all layers.
	maxQDim, maxKvDim, maxHeadDim := nHeads*headDim, nKVHeads*headDim, headDim
	for _, sp := range specs {
		lhd, lkv := headDimOf(sp, headDim), kvHeadsOf(sp, nKVHeads)
		if q := nHeads * lhd; q > maxQDim {
			maxQDim = q
		}
		if kv := lkv * lhd; kv > maxKvDim {
			maxKvDim = kv
		}
		if lhd > maxHeadDim {
			maxHeadDim = lhd
		}
	}
	// gemma4 value-norm weight: ones of the largest head_dim, shared across heads + layers
	// (the per-head value RMSNorm reads axisSize=headDim of it). nil ⇒ no value-norm.
	var valueNormOnes metal.MTLBuffer
	if valueNorm {
		valueNormOnes = sharedBytes(bf16ConstBytes(maxHeadDim, 1.0))
	}
	off := int32(0)
	return archDecodeState{
		specs: specs, lb: lb, moeWeights: moeWeights,
		asc: newAttnScratch(dModel, maxQDim, maxKvDim), msc: newMLPScratch(dModel, dFF),
		hBuf: scratchBF16(dModel), xA: scratchBF16(dModel), xB: scratchBF16(dModel),
		offBuf:         device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&off), 4, metal.MTLResourceStorageModeShared),
		valueNormOnes:  valueNormOnes,
		dModel:         dModel,
		nHeads:         nHeads,
		nKVHeads:       nKVHeads,
		headDim:        headDim,
		dFF:            dFF,
		slidingWindow:  slidingWindow,
		rotaryDim:      rotaryDim,
		rotaryDimLocal: rotaryDimLocal,
		base:           base, localBase: localBase, scale: scale, eps: eps,
	}
}

// stepToken decodes ONE token (its embedding) at sequence position pos, writing this
// token's K/V into the growing cache, and returns its output hidden state. The projector
// seam keeps it weight-representation-agnostic (bf16 / 4-bit qmv); it honours owner/sharer
// KV-sharing, sliding-window, the gemma4 norms, and MoE (the mid-token command-buffer flush
// because the router does host top-k). The caches persist across calls, so successive
// positions extend the same sequence. MUST be called inside a withAutoreleasePool.
func (s *archDecodeState) stepToken(inputEmb []byte, pos int) ([]byte, error) {
	*(*int32)(s.offBuf.Contents()) = int32(pos)
	copy(unsafe.Slice((*byte)(s.xA.Contents()), s.dModel*bf16Size), inputEmb)
	cb := queue.CommandBuffer()
	enc := cb.ComputeCommandEncoder()
	in, out := s.xA, s.xB
	for li := 0; li < len(s.specs); li++ {
		// sliding layers window the SDPA AND use the local RoPE theta + rotary dim; global use the global.
		slideW, rbase, rotDim := 0, s.base, s.rotaryDim
		if s.specs[li].Attention == g4.SlidingAttention {
			slideW, rbase, rotDim = s.slidingWindow, s.localBase, s.rotaryDimLocal
		}
		// per-attention-type head geometry (gemma4 full layers use the larger global head_dim);
		// the SDPA scale stays s.scale — the model DECLARED it (gemma4 1.0, not 1/√headDim).
		lhd, lkv := headDimOf(s.specs[li], s.headDim), kvHeadsOf(s.specs[li], s.nKVHeads)
		if s.specs[li].OwnsCache() {
			if err := encAttnHalfKV(enc, in, s.lb[li].anw, s.lb[li].kCache, s.lb[li].vCache, s.offBuf, s.hBuf, s.lb[li].postAttnNorm, s.lb[li].qNorm, s.lb[li].kNorm, s.valueNormOnes, s.asc, s.lb[li].proj, s.dModel, s.nHeads, lkv, lhd, pos, slideW, rotDim, rbase, s.scale, s.eps, s.ropeFreqs); err != nil {
				enc.EndEncoding()
				return nil, err
			}
		} else {
			own := s.specs[li].KVShareFrom
			if err := encAttnHalfShared(enc, in, s.lb[li].anw, s.lb[own].kCache, s.lb[own].vCache, s.offBuf, s.hBuf, s.lb[li].postAttnNorm, s.lb[li].qNorm, s.asc, s.lb[li].proj, s.dModel, s.nHeads, lkv, lhd, pos, slideW, rotDim, rbase, s.scale, s.eps, s.ropeFreqs); err != nil {
				enc.EndEncoding()
				return nil, err
			}
		}
		var moeQ *MoEQuantLayerWeights
		if li < len(s.moeQuant) {
			moeQ = s.moeQuant[li]
		}
		if moeW := s.moeWeights[li]; moeQ != nil || moeW != nil {
			// the MoE FFN needs h on the host (the router does host top-k): flush the
			// attention half, run the dual-branch block host-side, resume a fresh encoder.
			enc.EndEncoding()
			cb.Commit()
			cb.WaitUntilCompleted()
			hostH := make([]byte, s.dModel*bf16Size)
			copy(hostH, unsafe.Slice((*byte)(s.hBuf.Contents()), s.dModel*bf16Size))
			var res []byte
			var err error
			if moeQ != nil {
				res, err = MoEBlockQuant(hostH, *moeQ, s.dModel, s.dFF, s.eps)
			} else {
				res, err = MoEBlockBF16(hostH, *moeW, s.dModel, s.dFF, s.eps)
			}
			if err != nil {
				return nil, err
			}
			copy(unsafe.Slice((*byte)(out.Contents()), s.dModel*bf16Size), res)
			cb = queue.CommandBuffer()
			enc = cb.ComputeCommandEncoder()
		} else if err := encMLPHalfBF16(enc, s.hBuf, s.lb[li].mnw, out, s.lb[li].postFFNorm, s.msc, s.lb[li].proj, s.dModel, s.dFF, s.eps); err != nil {
			enc.EndEncoding()
			return nil, err
		}
		// gemma4 per-layer-input gate (E2B/E4B): host-orchestrated (QMV+gelu+QMV+norm+add, no
		// fused encoder op), so flush the layer, gate out host-side, resume — mirrors the MoE
		// flush. Applied to the layer output before the per-layer scalar.
		if len(s.ple) > li && len(s.ple[li].postNorm) > 0 {
			enc.EndEncoding()
			cb.Commit()
			cb.WaitUntilCompleted()
			outHost := make([]byte, s.dModel*bf16Size)
			copy(outHost, unsafe.Slice((*byte)(out.Contents()), s.dModel*bf16Size))
			pli := s.perLayerInput[li*s.pliDim*bf16Size : (li+1)*s.pliDim*bf16Size]
			gated, gerr := PerLayerInputGateQuant(outHost, s.ple[li].gate, pli, s.ple[li].proj, s.ple[li].postNorm, s.dModel, s.pliDim, s.ple[li].groupSize, s.ple[li].bits, s.eps)
			if gerr != nil {
				return nil, gerr
			}
			copy(unsafe.Slice((*byte)(out.Contents()), s.dModel*bf16Size), gated)
			cb = queue.CommandBuffer()
			enc = cb.ComputeCommandEncoder()
		}
		// gemma4 per-layer output scalar: multiply the layer's hidden before the next layer.
		if s.lb[li].layerScalar != nil {
			if err := encMulBF16(enc, out, s.lb[li].layerScalar, out, s.dModel); err != nil {
				enc.EndEncoding()
				return nil, err
			}
		}
		in, out = out, in
	}
	enc.EndEncoding()
	cb.Commit()
	cb.WaitUntilCompleted()
	res := make([]byte, s.dModel*bf16Size)
	copy(res, unsafe.Slice((*byte)(in.Contents()), s.dModel*bf16Size))
	return res, nil
}

// runArchDecode is the whole-sequence arch decode: it builds a state and steps each input
// token at its position over a fresh growing cache. See archDecodeState/stepToken — the
// bf16 (DecodeForwardArch) and 4-bit qmv (DecodeForwardArchQuant) forwards share this. MUST
// be called inside a withAutoreleasePool.
func runArchDecode(
	inputs [][]byte, specs []g4.LayerSpec, lb []archLayerBufs, moeWeights []*MoELayerWeights,
	dModel, nHeads, nKVHeads, headDim, dFF, slidingWindow, rotaryDim, rotaryDimLocal int, base, localBase, scale, eps float32, valueNorm bool,
) ([][]byte, error) {
	s := newArchDecodeState(specs, lb, moeWeights, dModel, nHeads, nKVHeads, headDim, dFF, slidingWindow, rotaryDim, rotaryDimLocal, base, localBase, scale, eps, valueNorm)
	outputs := make([][]byte, len(inputs))
	for t := range inputs {
		out, err := s.stepToken(inputs[t], t)
		if err != nil {
			return nil, err
		}
		outputs[t] = out
	}
	return outputs, nil
}

// DecodeForwardArch is the bf16 arch-driven decode forward: it runs a decode DRIVEN by
// a declared gemma4 arch (specs, one LayerSpec per layer) rather than treating every
// layer uniformly. It honours the full cache-topology (owner/sharer KV), the per-layer
// attention type (sliding window), and MoE layers (the dual-branch MoEBlockBF16). With
// an all-owner, all-global, dense arch it equals DecodeForward byte-for-byte (gated).
// bf16 re-encode path (one commit+wait/token; MoE layers flush mid-token). The 4-bit
// variant DecodeForwardArchQuant shares the loop (runArchDecode) via the projector seam.
func DecodeForwardArch(
	inputs [][]byte, layers []DecodeLayerWeights, specs []g4.LayerSpec,
	dModel, nHeads, nKVHeads, headDim, maxLen, dFF, slidingWindow int,
	base, scale, eps float32, valueNorm bool,
) ([][]byte, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	nLayers, T := len(layers), len(inputs)
	if nLayers == 0 || T == 0 {
		return nil, core.NewError("native.DecodeForwardArch: need layers and inputs")
	}
	if len(specs) != nLayers {
		return nil, core.NewError("native.DecodeForwardArch: specs length must equal layers")
	}
	if T > maxLen {
		return nil, core.NewError("native.DecodeForwardArch: more tokens than maxLen cache rows")
	}
	for i := range inputs {
		if len(inputs[i]) != dModel*bf16Size {
			return nil, core.NewError("native.DecodeForwardArch: each input must be dModel bf16 bytes")
		}
	}
	for li := range specs {
		o := specs[li].KVShareFrom
		if o < 0 || o > li || (o != li && !specs[o].OwnsCache()) {
			return nil, core.NewError("native.DecodeForwardArch: KVShareFrom must reference an earlier owner layer")
		}
		if specs[li].MoE != (layers[li].MoE != nil) {
			return nil, core.NewError("native.DecodeForwardArch: spec.MoE must match the presence of layer MoE weights")
		}
	}

	var outputs [][]byte
	var err error
	withAutoreleasePool(func() {
		lb, moeWeights := buildBF16ArchLayerBufs(layers, specs, dModel, nHeads, nKVHeads, headDim, dFF, maxLen)
		outputs, err = runArchDecode(inputs, specs, lb, moeWeights, dModel, nHeads, nKVHeads, headDim, dFF, slidingWindow, headDim, headDim, base, base, scale, eps, valueNorm)
	})
	return outputs, err
}

// buildBF16ArchLayerBufs builds the per-layer resident buffers for a bf16 arch decode:
// bf16 norms + the bf16 projector + the growing KV caches (owner layers only), and the
// per-layer MoE weights (moeWeights[li] != nil ⟺ a MoE layer, whose dense MLP norm +
// gate/up/down stay unbound — MoEBlockBF16 owns that FFN). Shared by the whole-sequence
// forward and the incremental generation loop. MUST be called inside a withAutoreleasePool.
func buildBF16ArchLayerBufs(layers []DecodeLayerWeights, specs []g4.LayerSpec, dModel, nHeads, nKVHeads, headDim, dFF, maxLen int) ([]archLayerBufs, []*MoELayerWeights) {
	nLayers := len(layers)
	lb := make([]archLayerBufs, nLayers)
	moeWeights := make([]*MoELayerWeights, nLayers)
	for li := range layers {
		w := layers[li]
		// per-attention-type geometry: gemma4 full_attention layers use a larger head_dim
		// (global_head_dim), so the projection dims + KV-cache row size are per layer.
		lhd, lkv := headDimOf(specs[li], headDim), kvHeadsOf(specs[li], nKVHeads)
		qDim, kvDim := nHeads*lhd, lkv*lhd
		cacheBytes := uint(maxLen * kvDim * bf16Size)
		lb[li].anw = sharedBytes(w.AttnNormW)
		lb[li].postAttnNorm = sharedOrNil(w.PostAttnNormW)
		lb[li].postFFNorm = sharedOrNil(w.PostFFNormW)
		lb[li].qNorm = sharedOrNil(w.QNormW)
		lb[li].kNorm = sharedOrNil(w.KNormW)
		lb[li].layerScalar = layerScalarBuf(w.LayerScalarW, dModel)
		if specs[li].OwnsCache() {
			lb[li].kCache = device.NewBufferWithLengthOptions(cacheBytes, metal.MTLResourceStorageModeShared)
			lb[li].vCache = device.NewBufferWithLengthOptions(cacheBytes, metal.MTLResourceStorageModeShared)
		}
		p := bf16Projector{
			wQ: sharedBytes(w.WQ), wK: sharedBytes(w.WK), wV: sharedOrNil(w.WV), wO: sharedBytes(w.WO),
			dModel: dModel, qDim: qDim, kvDim: kvDim, dFF: dFF,
		}
		if layers[li].MoE == nil {
			lb[li].mnw = sharedBytes(w.MLPNormW)
			p.wGate = sharedBytes(w.WGate)
			p.wUp = sharedBytes(w.WUp)
			p.wDown = sharedBytes(w.WDown)
		} else {
			moeWeights[li] = layers[li].MoE
		}
		lb[li].proj = p
	}
	return lb, moeWeights
}

// layerScalarBuf broadcasts a gemma4 per-layer output scalar (shape [1] bf16) to a dModel-length
// bf16 buffer for the per-layer output multiply, or nil when absent. The [1]→dModel fill matches
// metal.Mul(hidden, scalar) (broadcast); bf16→f32→bf16 round-trips the scalar value exactly.
func layerScalarBuf(scalarW []byte, dModel int) metal.MTLBuffer {
	if len(scalarW) != bf16Size {
		return nil
	}
	return sharedBytes(bf16ConstBytes(dModel, bf16ToF32(scalarW[0], scalarW[1])))
}

// valueNormOnesBuf is the gemma4 value-norm weight: a [headDim] bf16 ones vector so the
// proven RMSNorm-rows kernel computes the no-scale per-head RMSNorm on V (metal's
// RMSNormNoScale). Returns nil when off (non-gemma4) ⇒ the decode skips value-norm.
// MUST be called inside a withAutoreleasePool. Used by the ICB wrappers (the re-encode
// arch path builds its own at the largest head_dim in newArchDecodeState).
func valueNormOnesBuf(on bool, headDim int) metal.MTLBuffer {
	if !on {
		return nil
	}
	return sharedBytes(bf16ConstBytes(headDim, 1.0))
}
