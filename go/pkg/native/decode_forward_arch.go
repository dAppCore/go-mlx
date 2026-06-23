// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"unsafe"

	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/model"
	"github.com/tmc/apple/metal"
)

// attnScaleOf is the SDPA scale the model DECLARES (the engine applies it, never
// assumes): gemma4 = 1.0 (its per-head QK-norm is the scaling), standard transformers
// = 1/√headDim. Falls back to 1/√headDim for a hand-built Arch that predates the
// declared field (AttnScale == 0), so existing paths are byte-identical.
func attnScaleOf(arch model.Arch) float32 {
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
func headDimOf(spec model.LayerSpec, fallback int) int {
	if spec.HeadDim > 0 {
		return spec.HeadDim
	}
	return fallback
}

func kvHeadsOf(spec model.LayerSpec, fallback int) int {
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
	x, attendK, attendV, offBuf, h metal.MTLBuffer,
	attnNormW, postAttnNorm, qNorm bufView,
	sc attnScratch, proj projector,
	dModel, nHeads, nKVHeads, headDim, pos, slideW, rotaryDim int, base, scale, eps float32,
	ropeFreqs metal.MTLBuffer,
) error {
	kvDim := nKVHeads * headDim
	if err := encRMSNormBF16(enc, x, attnNormW.buf, sc.normed, attnNormW.off, dModel, eps); err != nil {
		return err
	}
	if err := proj.project(enc, sc.normed, sc.q, 0, projQ); err != nil {
		return err
	}
	if gpuHasGeluKernel() && qNorm.buf != nil {
		// fused: sc.q = RoPE(RMSNorm(sc.q, qNorm)) in one op — lockstep with the ICB setQKNormRope
		if err := encQKNormRope(enc, sc.q, qNorm.buf, sc.q, 0, qNorm.off, 0, offBuf, ropeFreqs, nHeads, headDim, rotaryDim, base, scale, eps); err != nil {
			return err
		}
	} else {
		if qNorm.buf != nil { // gemma4 per-head QK-norm before RoPE (sharers project only Q)
			if err := encRMSNormRowsBF16(enc, sc.q, qNorm.buf, sc.q, 0, qNorm.off, 0, nHeads, headDim, eps); err != nil {
				return err
			}
		}
		// RoPE Q in place so partial rotary's untouched tail keeps the projected value.
		if err := encRopeDecode(enc, sc.q, sc.q, 0, 0, offBuf, ropeFreqs, nHeads, headDim, rotaryDim, base, scale); err != nil {
			return err
		}
	}
	// attend the OWNER's cache (no write): the whole seq-major cache (global) or the whole live ring
	// (sliding, slideW>0) — n live rows from offset 0, matching the owner's ring write in encAttnHalfKV.
	n := pos + 1
	if slideW > 0 && n > slideW {
		n = slideW
	}
	if err := encSDPAStrided(enc, sc.q, attendK, attendV, sc.attn,
		nHeads, nKVHeads, headDim, n,
		int64(headDim), int64(kvDim), int64(headDim), int64(kvDim), scale, 0); err != nil {
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
	anw, mnw                 bufView
	postAttnNorm, postFFNorm bufView         // gemma4 post-attn/post-FF norms (nil buf = skip)
	qNorm, kNorm             bufView         // gemma4 per-head QK-norm (nil buf = skip)
	layerScalar              metal.MTLBuffer // gemma4 per-layer output scalar, broadcast to dModel (synthesised, nil = skip)
	kCache, vCache           metal.MTLBuffer
	proj                     projector
	dFF                      int // this layer's FFN width (gemma4 E2B/E4B vary it per layer)
}

// archDecodeState holds the resident buffers of an arch decode — the per-layer weights/
// caches (lb), shared scratch, and the position buffer — so a single token can be stepped
// repeatedly over a PERSISTENT, growing KV cache. Both the whole-sequence runArchDecode and
// the incremental generation loop build one (inside a withAutoreleasePool) and call
// stepToken per token; the caches in lb persist across calls within that pool, which is
// what turns the O(N²) re-decode into O(1)/token incremental decode.
type archDecodeState struct {
	specs        []model.LayerSpec
	lb           []archLayerBufs
	moeWeights   []*MoELayerWeights
	asc          attnScratch
	msc          mlpScratch
	hBuf, xA, xB metal.MTLBuffer
	offBuf       metal.MTLBuffer
	ropeFreqs    metal.MTLBuffer // resident periods (1/inv_freq) for YaRN long-context rope; nil = base-derived rope
	// gemma4 global (proportional+partial) rope: the period spectrum over the FULL head dim
	// (metal's gemma4ProportionalFreqs) for GlobalAttention layers, so rope pairs (d, d+globalHeadDim/2)
	// over the whole head — NOT (d, d+rotaryDim/2). nil ⇒ no proportional global layers.
	globalRopeFreqs metal.MTLBuffer
	globalHeadDim   int             // the full head dim global layers rope over (passed as rotaryDim to the freqs path)
	valueNormOnes   metal.MTLBuffer // gemma4 value-norm: [maxHeadDim] ones weight for the no-scale per-head RMSNorm on V; nil = no value-norm (Mistral)

	dModel, nHeads, nKVHeads, headDim, dFF, slidingWindow int
	rotaryDim, rotaryDimLocal                             int     // partial-rotary dims (global / sliding); == headDim is full
	base, localBase, scale, eps                           float32 // localBase = sliding-layer RoPE theta

	// gemma4 per-layer-input tower (E2B/E4B): when ple is non-nil, each layer's output is gated
	// by PerLayerInputGateQuant before layer_scalar, fed its pliDim slice of perLayerInput (the
	// PerLayerInputs tensor, set per token). nil = no PLE tower (dense models — byte-identical).
	ple           []pleLayer
	perLayerInput []byte // [numLayers·pliDim] bf16, set before each token's stepToken
	pliDim        int
	hostScratch   []byte // reusable dModel bf16 host handoff for MoE/PLE host-orchestrated branches

	// gemma4 4-bit MoE (26B-A4B): moeQuant[li] != nil runs MoEBlockQuant for that layer's FFN
	// (host-orchestrated like the bf16 MoE). nil entries use the dense MLP / bf16 moeWeights.
	moeQuant []*MoEQuantLayerWeights

	// trace (LTHN_NATIVE_TRACE): when set, stepToken flushes + reads back each layer's output
	// hidden and logs the per-token worst max-abs + NaN layer — the decode-degradation probe.
	trace bool

	// icb, when non-nil, is the recorded arch ICB the session replays per token (the encode-bypass)
	// instead of re-encoding via stepToken. Set at session build when icbEligible (no MoE, no trace,
	// uniform head geometry + simple uniform rope — the ICB core's assumptions). It holds its OWN
	// maxLen-linear caches (NOT the state's lb ring caches), so an ICB session decodes EVERY token
	// (prefill + decode) through it. nil ⇒ stepToken.
	icb *archICBReplay
}

func (s *archDecodeState) hostHiddenScratch(dModel int) []byte {
	n := dModel * bf16Size
	if cap(s.hostScratch) < n {
		s.hostScratch = make([]byte, n)
	}
	return s.hostScratch[:n]
}

// pleLayer is one layer's per-layer-input gate weights: the 4-bit gate + projection and the
// bf16 post-norm. A nil postNorm marks a layer with no gate (so a mixed model is fine).
type pleLayer struct {
	gate, proj      QuantWeight
	postNorm        []byte
	groupSize, bits int
}

// ArchPLEBF16 is the token-id-aware PLE payload for a bf16 whole-sequence arch decode.
// TokenIDs line up with the input embeddings passed to DecodeForwardArch/ICB; the PLE
// tensor is computed as PerLayerInputs(id, inputEmbedding) before each token is decoded.
type ArchPLEBF16 struct {
	TokenIDs           []int32
	EmbedPerLayer      []byte
	PerLayerModelProjW []byte
	PerLayerProjNormW  []byte
	VocabPLI, PliDim   int
}

// ArchPLEQuant is the token-id-aware PLE payload for a quant whole-sequence arch decode.
// The embed-per-layer and optional model projection triples are the bookend weights
// consumed by PerLayerInputs; the per-layer gate/projection weights live on qlayers.
type ArchPLEQuant struct {
	TokenIDs []int32

	EmbedPerLayer, EmbedPerLayerScales, EmbedPerLayerBiases              []byte
	PerLayerModelProjW, PerLayerModelProjScales, PerLayerModelProjBiases []byte
	PerLayerProjNormW                                                    []byte

	VocabPLI, PliDim        int
	GroupSize, Bits         int
	ProjGroupSize, ProjBits int
}

type archDecodePLEInputs struct {
	tokenIDs []int32
	compute  func(id int32, emb []byte) ([]byte, error)
}

func singleArchPLEBF16(fn string, ple []ArchPLEBF16) (*ArchPLEBF16, error) {
	if len(ple) == 0 {
		return nil, nil
	}
	if len(ple) > 1 {
		return nil, core.NewError(fn + ": at most one PLE payload is supported")
	}
	return &ple[0], nil
}

func singleArchPLEQuant(fn string, ple []ArchPLEQuant) (*ArchPLEQuant, error) {
	if len(ple) == 0 {
		return nil, nil
	}
	if len(ple) > 1 {
		return nil, core.NewError(fn + ": at most one PLE payload is supported")
	}
	return &ple[0], nil
}

func archPLEBF16Runtime(fn string, p *ArchPLEBF16, nLayers, T, dModel int, eps float32) (*archDecodePLEInputs, int, error) {
	if p == nil {
		return nil, 0, nil
	}
	if len(p.TokenIDs) != T {
		return nil, 0, core.NewError(fn + ": PLE token id count must equal inputs")
	}
	if p.VocabPLI <= 0 || p.PliDim <= 0 {
		return nil, 0, core.NewError(fn + ": PLE vocab and hidden dims must be > 0")
	}
	if len(p.PerLayerProjNormW) != p.PliDim*bf16Size {
		return nil, 0, core.NewError(fn + ": PLE projection norm must be pliDim bf16 bytes")
	}
	return &archDecodePLEInputs{
		tokenIDs: p.TokenIDs,
		compute: func(id int32, emb []byte) ([]byte, error) {
			return PerLayerInputs(p.EmbedPerLayer, nil, nil, p.PerLayerModelProjW, nil, nil, p.PerLayerProjNormW, id, emb, p.VocabPLI, nLayers, p.PliDim, dModel, 0, 0, 0, 0, eps, bufView{})
		},
	}, p.PliDim, nil
}

func archPLEQuantRuntime(fn string, p *ArchPLEQuant, nLayers, T, dModel int, eps float32) (*archDecodePLEInputs, int, error) {
	if p == nil {
		return nil, 0, nil
	}
	if len(p.TokenIDs) != T {
		return nil, 0, core.NewError(fn + ": PLE token id count must equal inputs")
	}
	if p.VocabPLI <= 0 || p.PliDim <= 0 || p.GroupSize <= 0 || p.Bits <= 0 {
		return nil, 0, core.NewError(fn + ": PLE quant geometry must be set")
	}
	if len(p.PerLayerProjNormW) != p.PliDim*bf16Size {
		return nil, 0, core.NewError(fn + ": PLE projection norm must be pliDim bf16 bytes")
	}
	return &archDecodePLEInputs{
		tokenIDs: p.TokenIDs,
		compute: func(id int32, emb []byte) ([]byte, error) {
			return PerLayerInputs(p.EmbedPerLayer, p.EmbedPerLayerScales, p.EmbedPerLayerBiases, p.PerLayerModelProjW, p.PerLayerModelProjScales, p.PerLayerModelProjBiases, p.PerLayerProjNormW, id, emb, p.VocabPLI, nLayers, p.PliDim, dModel, p.GroupSize, p.Bits, p.ProjGroupSize, p.ProjBits, eps, bufView{})
		},
	}, p.PliDim, nil
}

func quantWeightBytesOK(w QuantWeight, outDim, inDim, groupSize, bits int) bool {
	return inDim%groupSize == 0 &&
		len(w.Packed) == outDim*inDim*bits/8 &&
		len(w.Scales) == outDim*(inDim/groupSize)*bf16Size &&
		len(w.Biases) == outDim*(inDim/groupSize)*bf16Size
}

func bf16PLELayers(fn string, layers []DecodeLayerWeights, dModel, pliDim int) ([]pleLayer, error) {
	ple := make([]pleLayer, len(layers))
	for li := range layers {
		w := layers[li]
		if len(w.PerLayerGate) != pliDim*dModel*bf16Size ||
			len(w.PerLayerProjection) != dModel*pliDim*bf16Size ||
			len(w.PostPerLayerInputNormW) != dModel*bf16Size {
			return nil, core.NewError(core.Sprintf("%s: PLE bf16 layer %d weight size mismatch", fn, li))
		}
		ple[li] = pleLayer{
			gate:     QuantWeight{Packed: w.PerLayerGate},
			proj:     QuantWeight{Packed: w.PerLayerProjection},
			postNorm: w.PostPerLayerInputNormW,
		}
	}
	return ple, nil
}

func quantPLELayers(fn string, qlayers []QuantizedLayerWeights, dModel, pliDim, groupSize, bits int) ([]pleLayer, error) {
	ple := make([]pleLayer, len(qlayers))
	for li := range qlayers {
		w := qlayers[li]
		if !quantWeightBytesOK(w.PerLayerGate, pliDim, dModel, groupSize, bits) ||
			!quantWeightBytesOK(w.PerLayerProjection, dModel, pliDim, groupSize, bits) ||
			len(w.PostPerLayerInputNormW) != dModel*bf16Size {
			return nil, core.NewError(core.Sprintf("%s: PLE quant layer %d weight size mismatch", fn, li))
		}
		ple[li] = pleLayer{
			gate: w.PerLayerGate, proj: w.PerLayerProjection,
			postNorm: w.PostPerLayerInputNormW, groupSize: groupSize, bits: bits,
		}
	}
	return ple, nil
}

// newArchDecodeState builds the shared scratch + position buffer over the caller's
// per-layer buffers. MUST be called inside a withAutoreleasePool.
func newArchDecodeState(specs []model.LayerSpec, lb []archLayerBufs, moeWeights []*MoELayerWeights, dModel, nHeads, nKVHeads, headDim, dFF, slidingWindow, rotaryDim, rotaryDimLocal int, base, localBase, scale, eps float32, valueNorm bool) archDecodeState {
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
	// per-layer FFN width (gemma4 E2B/E4B MatFormer): the shared MLP scratch must fit the WIDEST layer.
	maxDFF := dFF
	for i := range lb {
		if lb[i].dFF > maxDFF {
			maxDFF = lb[i].dFF
		}
	}
	// gemma4 value-norm weight: ones of the largest head_dim, shared across heads + layers
	// (the per-head value RMSNorm reads axisSize=headDim of it). nil ⇒ no value-norm.
	var valueNormOnes metal.MTLBuffer
	if valueNorm {
		valueNormOnes = sharedBytes(bf16ConstBytes(maxHeadDim, 1.0))
	}
	// gemma4 global proportional+partial rope spectrum (see gemma4ProportionalPeriods): built once
	// for GlobalAttention layers so their rope pairs over the FULL head dim. Sliding (full rotary)
	// keeps the base-derived path.
	var globalRopeFreqs metal.MTLBuffer
	globalHeadDim := 0
	for _, sp := range specs {
		if sp.Attention == model.GlobalAttention {
			globalHeadDim = headDimOf(sp, headDim)
			break
		}
	}
	if globalHeadDim > 0 && rotaryDim > 0 && rotaryDim < globalHeadDim {
		periods := gemma4ProportionalPeriods(globalHeadDim, rotaryDim, base)
		globalRopeFreqs = device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&periods[0]), uint(len(periods)*4), metal.MTLResourceStorageModeShared)
	}
	off := int32(0)
	return archDecodeState{
		specs: specs, lb: lb, moeWeights: moeWeights,
		globalRopeFreqs: globalRopeFreqs, globalHeadDim: globalHeadDim,
		asc: newAttnScratch(dModel, maxQDim, maxKvDim), msc: newMLPScratch(dModel, maxDFF),
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
		trace: nativeTraceEnabled(),
	}
}

// bufMaxAbsNaN reads a dModel-length bf16 buffer back to host and returns the largest finite
// absolute value plus the count of NaN/Inf-scale elements — the per-layer trace signal. A
// blow-up or NaN, and the token/layer it first appears at, localise where a decode degrades.
// Debug-path only (the readback forces a commit+wait).
func bufMaxAbsNaN(buf metal.MTLBuffer, dModel int) (maxAbs float32, bad int) {
	b := unsafe.Slice((*byte)(buf.Contents()), dModel*bf16Size)
	for i := 0; i < dModel; i++ {
		v := bf16ToF32(b[i*bf16Size], b[i*bf16Size+1])
		if v != v || v > 3.0e38 || v < -3.0e38 { // NaN or Inf-scale
			bad++
			continue
		}
		if v < 0 {
			v = -v
		}
		if v > maxAbs {
			maxAbs = v
		}
	}
	return maxAbs, bad
}

// captureLayerHiddens, when set by the cross-engine test, makes stepToken append each
// layer's output hidden (dModel bf16 bytes) to capturedLayerHiddens — the native half of
// the per-layer cross-engine diff. Reset capturedLayerHiddens to nil before the step.
var (
	captureLayerHiddens  bool
	capturedLayerHiddens [][]byte
	capturedAttnHiddens  [][]byte // post-attention hidden (x + Wo·attn) per layer — isolates attention from MLP
)

// stepToken decodes ONE token (its embedding) at sequence position pos, writing this
// token's K/V into the growing cache, and returns its output hidden state. The projector
// seam keeps it weight-representation-agnostic (bf16 / 4-bit qmv); it honours owner/sharer
// KV-sharing, sliding-window, the gemma4 norms, and MoE (the mid-token command-buffer flush
// because the router does host top-k). The caches persist across calls, so successive
// positions extend the same sequence. MUST be called inside a withAutoreleasePool.
func (s *archDecodeState) stepToken(inputEmb []byte, pos int) ([]byte, error) {
	return s.stepTokenResult(inputEmb, pos, true)
}

func (s *archDecodeState) stepTokenNoResult(inputEmb []byte, pos int) error {
	_, err := s.stepTokenResult(inputEmb, pos, false)
	return err
}

func (s *archDecodeState) stepTokenResult(inputEmb []byte, pos int, readResult bool) ([]byte, error) {
	*(*int32)(s.offBuf.Contents()) = int32(pos)
	copy(unsafe.Slice((*byte)(s.xA.Contents()), s.dModel*bf16Size), inputEmb)
	cb := queue.CommandBuffer()
	enc := cb.ComputeCommandEncoder()
	in, out := s.xA, s.xB
	var trWorstAbs float32
	trWorstLayer, trFirstBad, trBadLayers := -1, -1, 0
	for li := 0; li < len(s.specs); li++ {
		// per-attention-type head geometry (gemma4 full layers use the larger global head_dim);
		// the SDPA scale stays s.scale — the model DECLARED it (gemma4 1.0, not 1/√headDim).
		lhd, lkv := headDimOf(s.specs[li], s.headDim), kvHeadsOf(s.specs[li], s.nKVHeads)
		// sliding layers window the SDPA AND use the local RoPE theta + rotary dim; global use the
		// global. gemma4 global rope is proportional + PARTIAL: drive the freqs path over the FULL
		// head (rotDim=lhd) with the Inf-padded spectrum so it pairs (d, d+headDim/2) — the base
		// path's (d, d+rotaryDim/2) pairing is wrong for partial rotary (see globalRopeFreqs).
		slideW, rbase, rotDim := 0, s.base, s.rotaryDim
		layerRopeFreqs := s.ropeFreqs
		if s.specs[li].Attention == model.SlidingAttention {
			slideW, rbase, rotDim = s.slidingWindow, s.localBase, s.rotaryDimLocal
		} else if s.globalRopeFreqs != nil {
			layerRopeFreqs, rotDim = s.globalRopeFreqs, lhd
		}
		if s.specs[li].OwnsCache() {
			if err := encAttnHalfKV(enc, in, s.lb[li].kCache, s.lb[li].vCache, s.offBuf, s.hBuf, s.lb[li].anw, s.lb[li].postAttnNorm, s.lb[li].qNorm, s.lb[li].kNorm, s.valueNormOnes, s.asc, s.lb[li].proj, s.dModel, s.nHeads, lkv, lhd, pos, slideW, rotDim, rbase, s.scale, s.eps, layerRopeFreqs); err != nil {
				enc.EndEncoding()
				return nil, err
			}
		} else {
			own := s.specs[li].KVShareFrom
			if err := encAttnHalfShared(enc, in, s.lb[own].kCache, s.lb[own].vCache, s.offBuf, s.hBuf, s.lb[li].anw, s.lb[li].postAttnNorm, s.lb[li].qNorm, s.asc, s.lb[li].proj, s.dModel, s.nHeads, lkv, lhd, pos, slideW, rotDim, rbase, s.scale, s.eps, layerRopeFreqs); err != nil {
				enc.EndEncoding()
				return nil, err
			}
		}
		if captureLayerHiddens { // post-attention hidden (x + Wo·attn) — isolates attention from MLP
			enc.EndEncoding()
			cb.Commit()
			cb.WaitUntilCompleted()
			capturedAttnHiddens = append(capturedAttnHiddens, append([]byte(nil), unsafe.Slice((*byte)(s.hBuf.Contents()), s.dModel*bf16Size)...))
			cb = queue.CommandBuffer()
			enc = cb.ComputeCommandEncoder()
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
			hostH := s.hostHiddenScratch(s.dModel)
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
		} else {
			lff := s.dFF // per-layer FFN width (gemma4 E2B/E4B); falls back to the arch default
			if s.lb[li].dFF > 0 {
				lff = s.lb[li].dFF
			}
			if err := encMLPHalfBF16(enc, s.hBuf, out, s.lb[li].mnw, s.lb[li].postFFNorm, s.msc, s.lb[li].proj, s.dModel, lff, s.eps); err != nil {
				enc.EndEncoding()
				return nil, err
			}
		}
		// gemma4 per-layer-input gate (E2B/E4B): host-orchestrated (QMV+gelu+QMV+norm+add, no
		// fused encoder op), so flush the layer, gate out host-side, resume — mirrors the MoE
		// flush. Applied to the layer output before the per-layer scalar.
		if len(s.ple) > li && len(s.ple[li].postNorm) > 0 {
			enc.EndEncoding()
			cb.Commit()
			cb.WaitUntilCompleted()
			outHost := s.hostHiddenScratch(s.dModel)
			copy(outHost, unsafe.Slice((*byte)(out.Contents()), s.dModel*bf16Size))
			pli := s.perLayerInput[li*s.pliDim*bf16Size : (li+1)*s.pliDim*bf16Size]
			var gated []byte
			var gerr error
			if s.ple[li].bits == 0 { // bf16 PLE gate (the quant path sets bits 4/8 ⇒ the qmv)
				gated, gerr = PerLayerInputGateBF16(outHost, s.ple[li].gate.Packed, pli, s.ple[li].proj.Packed, s.ple[li].postNorm, s.dModel, s.pliDim, s.eps)
			} else {
				gated, gerr = PerLayerInputGateQuant(outHost, s.ple[li].gate, pli, s.ple[li].proj, s.ple[li].postNorm, s.dModel, s.pliDim, s.ple[li].groupSize, s.ple[li].bits, s.eps)
			}
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
		if s.trace { // per-layer diagnostic: flush, read this layer's output hidden, accumulate
			enc.EndEncoding()
			cb.Commit()
			cb.WaitUntilCompleted()
			ma, bad := bufMaxAbsNaN(out, s.dModel)
			if bad > 0 {
				trBadLayers++
				if trFirstBad < 0 {
					trFirstBad = li
				}
			}
			if ma > trWorstAbs {
				trWorstAbs, trWorstLayer = ma, li
			}
			cb = queue.CommandBuffer()
			enc = cb.ComputeCommandEncoder()
		}
		if captureLayerHiddens { // cross-engine per-layer diff: store this layer's output hidden
			enc.EndEncoding()
			cb.Commit()
			cb.WaitUntilCompleted()
			capturedLayerHiddens = append(capturedLayerHiddens, append([]byte(nil), unsafe.Slice((*byte)(out.Contents()), s.dModel*bf16Size)...))
			cb = queue.CommandBuffer()
			enc = cb.ComputeCommandEncoder()
		}
		in, out = out, in
	}
	enc.EndEncoding()
	cb.Commit()
	cb.WaitUntilCompleted()
	var res []byte
	if readResult {
		res = make([]byte, s.dModel*bf16Size)
		copy(res, unsafe.Slice((*byte)(in.Contents()), s.dModel*bf16Size))
	}
	if s.trace {
		wt := "-"
		if trWorstLayer >= 0 {
			wt = "sliding"
			if s.specs[trWorstLayer].Attention == model.GlobalAttention {
				wt = "GLOBAL"
			}
		}
		fm, fb := bufMaxAbsNaN(in, s.dModel)
		var ieAbs float32 // input-embedding magnitude — flags a bad token-embed (e.g. a control token's 4-bit dequant)
		for i := 0; i+1 < len(inputEmb); i += 2 {
			if v := bf16ToF32(inputEmb[i], inputEmb[i+1]); v > ieAbs {
				ieAbs = v
			} else if -v > ieAbs {
				ieAbs = -v
			}
		}
		nativeTraceLog(core.Sprintf("native-trace tok=%d inEmbAbs=%.4g worstAbs=%.4g@L%d(%s) badLayers=%d firstBad=L%d finalAbs=%.4g finalBad=%d\n",
			pos, ieAbs, trWorstAbs, trWorstLayer, wt, trBadLayers, trFirstBad, fm, fb))
	}
	return res, nil
}

// runArchDecode is the whole-sequence arch decode: it builds a state and steps each input
// token at its position over a fresh growing cache. See archDecodeState/stepToken — the
// bf16 (DecodeForwardArch) and 4-bit qmv (DecodeForwardArchQuant) forwards share this. MUST
// be called inside a withAutoreleasePool.
func runArchDecode(
	inputs [][]byte, specs []model.LayerSpec, lb []archLayerBufs, moeWeights []*MoELayerWeights,
	dModel, nHeads, nKVHeads, headDim, dFF, slidingWindow, rotaryDim, rotaryDimLocal int, base, localBase, scale, eps float32, valueNorm bool,
) ([][]byte, error) {
	s := newArchDecodeState(specs, lb, moeWeights, dModel, nHeads, nKVHeads, headDim, dFF, slidingWindow, rotaryDim, rotaryDimLocal, base, localBase, scale, eps, valueNorm)
	return runArchDecodeState(inputs, &s, nil)
}

func runArchDecodeState(inputs [][]byte, s *archDecodeState, ple *archDecodePLEInputs) ([][]byte, error) {
	outputs := make([][]byte, len(inputs))
	for t := range inputs {
		if ple != nil {
			pli, err := ple.compute(ple.tokenIDs[t], inputs[t])
			if err != nil {
				return nil, err
			}
			if len(pli) != len(s.specs)*s.pliDim*bf16Size {
				return nil, core.NewError("native.runArchDecodeState: PLE tensor size mismatch")
			}
			s.perLayerInput = pli
		}
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
	plePayload, err := singleArchPLEBF16("native.DecodeForwardArch", pleArgs)
	if err != nil {
		return nil, err
	}
	pleRuntime, pliDim, err := archPLEBF16Runtime("native.DecodeForwardArch", plePayload, nLayers, T, dModel, eps)
	if err != nil {
		return nil, err
	}
	var pleLayers []pleLayer
	if pleRuntime != nil {
		pleLayers, err = bf16PLELayers("native.DecodeForwardArch", layers, dModel, pliDim)
		if err != nil {
			return nil, err
		}
	}

	var outputs [][]byte
	withAutoreleasePool(func() {
		lb, moeWeights, berr := buildBF16ArchLayerBufs(layers, specs, dModel, nHeads, nKVHeads, headDim, dFF, maxLen, slidingWindow, nil)
		if berr != nil {
			err = berr
			return
		}
		if pleRuntime != nil {
			state := newArchDecodeState(specs, lb, moeWeights, dModel, nHeads, nKVHeads, headDim, dFF, slidingWindow, headDim, headDim, base, base, scale, eps, valueNorm)
			state.ple, state.pliDim = pleLayers, pliDim
			outputs, err = runArchDecodeState(inputs, &state, pleRuntime)
			return
		}
		outputs, err = runArchDecode(inputs, specs, lb, moeWeights, dModel, nHeads, nKVHeads, headDim, dFF, slidingWindow, headDim, headDim, base, base, scale, eps, valueNorm)
	})
	return outputs, err
}

// buildBF16ArchLayerBufs builds the per-layer resident buffers for a bf16 arch decode:
// bf16 norms + the bf16 projector + the growing KV caches (owner layers only), and the
// per-layer MoE weights (moeWeights[li] != nil ⟺ a MoE layer, whose dense MLP norm +
// gate/up/down stay unbound — MoEBlockBF16 owns that FFN). Shared by the whole-sequence
// forward and the incremental generation loop.
//
// sb is the zero-copy weight source: when non-nil, every weight is bound as a no-copy view into
// the shared shard mmap at its byte offset (no upload, no second resident copy); when nil (the
// in-memory weight bytes of DecodeForwardArch or a session built from a parsed blob), each weight
// is uploaded into a fresh owned buffer at offset 0 — byte-identical, just a heap+GPU copy. A
// non-nil sb errors if a weight is not a view into its mapping (a programming error). MUST be
// called inside a withAutoreleasePool.
func buildBF16ArchLayerBufs(layers []DecodeLayerWeights, specs []model.LayerSpec, dModel, nHeads, nKVHeads, headDim, dFF, maxLen, slidingWindow int, sb *shardBuffers) ([]archLayerBufs, []*MoELayerWeights, error) {
	nLayers := len(layers)
	lb := make([]archLayerBufs, nLayers)
	moeWeights := make([]*MoELayerWeights, nLayers)
	var ferr error
	// view resolves a required weight: a no-copy shard view (sb != nil) or an uploaded copy.
	view := func(b []byte) bufView {
		if sb != nil {
			return sb.mustBufFor(b, &ferr)
		}
		return copyView(b)
	}
	// viewOrNil is view for an optional weight (absent ⇒ zero bufView, the "skip" sentinel).
	viewOrNil := func(b []byte) bufView {
		if len(b) == 0 {
			return bufView{}
		}
		return view(b)
	}
	for li := range layers {
		w := layers[li]
		// per-attention-type geometry: gemma4 full_attention layers use a larger head_dim
		// (global_head_dim), so the projection dims + KV-cache row size are per layer.
		lhd, lkv := headDimOf(specs[li], headDim), kvHeadsOf(specs[li], nKVHeads)
		qDim, kvDim := nHeads*lhd, lkv*lhd
		// sliding layers RING at slidingWindow rows (they only ever attend the last slidingWindow), so
		// they need slidingWindow rows of cache, not maxLen — the full-context KV memory fix. Global
		// (full_attention) layers attend everything, so they keep maxLen. min() keeps short contexts
		// (maxLen ≤ window) at maxLen (no benefit, no wrap). encAttnHalfKV does the matching ring write.
		cacheLen := maxLen
		if slidingWindow > 0 && slidingWindow < maxLen && specs[li].Attention != model.GlobalAttention {
			cacheLen = slidingWindow
		}
		cacheBytes := uint(cacheLen * kvDim * bf16Size)
		lb[li].anw = view(w.AttnNormW)
		lb[li].postAttnNorm = viewOrNil(w.PostAttnNormW)
		lb[li].postFFNorm = viewOrNil(w.PostFFNormW)
		lb[li].qNorm = viewOrNil(w.QNormW)
		lb[li].kNorm = viewOrNil(w.KNormW)
		lb[li].layerScalar = layerScalarBuf(w.LayerScalarW, dModel) // synthesised broadcast (not a shard view)
		if specs[li].OwnsCache() {
			lb[li].kCache = device.NewBufferWithLengthOptions(cacheBytes, metal.MTLResourceStorageModeShared)
			lb[li].vCache = device.NewBufferWithLengthOptions(cacheBytes, metal.MTLResourceStorageModeShared)
		}
		lFF := dFF // per-layer FFN width — gemma4 E2B/E4B MatFormer varies it (6144/12288); 0 ⇒ arch default
		if w.DFF > 0 {
			lFF = w.DFF
		}
		lb[li].dFF = lFF
		p := bf16Projector{
			wQ: view(w.WQ), wK: view(w.WK), wV: viewOrNil(w.WV), wO: view(w.WO),
			dModel: dModel, qDim: qDim, kvDim: kvDim, dFF: lFF,
		}
		if layers[li].MoE == nil {
			lb[li].mnw = view(w.MLPNormW)
			p.wGate = view(w.WGate)
			p.wUp = view(w.WUp)
			p.wDown = view(w.WDown)
		} else {
			moeWeights[li] = layers[li].MoE
		}
		lb[li].proj = p
	}
	return lb, moeWeights, ferr
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
// MUST be called inside a withAutoreleasePool.
//
// headDim MUST be the LARGEST per-layer head dim (maxHeadDimOf), not the base/uniform one:
// gemma4 E2B global layers use head_dim 512 vs sliding 256, and the value-norm op reads
// axisSize=hdOf(li) (512 on a global layer). A buffer sized at the base 256 makes that read
// run off the end of the ones vector → the upper half of every global head's V is normed by
// garbage weights, diverging from the host path at the first global layer (proven by the
// q4 ICB per-layer localiser). The re-encode arch path already sizes it at maxHeadDim in
// newArchDecodeState; the ICB wrappers must do the same.
func valueNormOnesBuf(on bool, headDim int) metal.MTLBuffer {
	if !on {
		return nil
	}
	return sharedBytes(bf16ConstBytes(headDim, 1.0))
}

// maxHeadDimOf returns the largest per-layer head dim over specs (falling back to the base
// headDim) — the size the shared value-norm ones vector + any per-head-dim scratch must use so
// a wider global layer's read stays in bounds. Mirrors newArchDecodeState's maxHeadDim.
func maxHeadDimOf(specs []model.LayerSpec, headDim int) int {
	m := headDim
	for _, sp := range specs {
		if hd := headDimOf(sp, headDim); hd > m {
			m = hd
		}
	}
	return m
}
