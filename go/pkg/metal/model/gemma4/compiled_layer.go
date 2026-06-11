// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"runtime/debug"
	"sync"
	"sync/atomic"

	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/metal"
)

// Whole-layer compiled decode — one Gemma 4 decoder layer's single-token step
// as ONE mlx_compile'd closure. The trace spans input norm → Q/K/V projections
// → Q/K norms → dynamic-offset RoPE → the fixed-cache attention step (cache
// write, mask, SDPA — the same C++-compiled segment the uncompiled path
// dispatches; nested compiled functions inline during tracing) → O projection
// → the feed-forward block → the per-layer-input gate. Replaying it collapses
// the layer's per-token graph build and schedule to a single closure apply.
//
// Capture discipline (the perlayer.go lesson, inverted): layer weights enter
// as closure INPUTS so one trace serves every layer that shares a config, and
// every shape-bearing parameter lives in the trace key — nothing shape- or
// position-dependent freezes into a reused trace. Position, cache contents,
// and sliding shift indices enter as arrays.
//
// Regimes (selected host-side per token, one trace each):
//
//	ownerPreCap  — cache below capacity: offset-indexed write + causal mask
//	ownerPostCap — cache at capacity: rotate-and-write via shift indices
//	consumer     — shared-KV layer: attention over the owner's fixed state
//
// Decode only (B==1, L==1, no explicit masks); anything else falls through to
// the uncompiled paths, as does any layer the closure declines (LoRA attached,
// MoE, FFN memory augmenter, non-affine quant). A panic from compile or replay
// poisons that trace key for the process.

type gemma4LayerRegime uint8

const (
	gemma4LayerOwnerPreCap gemma4LayerRegime = iota + 1
	gemma4LayerOwnerPostCap
	gemma4LayerConsumer
)

// gemma4LinearSig is one projection's quantisation signature inside the trace
// key. Index order: q, k, v, o, gate, up, down, pliGate, pliProj — absent
// projections stay zero.
type gemma4LinearSig struct {
	bits      int32
	groupSize int32
	mode      string
}

type gemma4CompiledLayerKey struct {
	regime   gemma4LayerRegime
	hidden   int32
	qHeads   int32
	kvHeads  int32
	headDim  int32
	capacity int32
	// attnBand is the power-of-two length band (floor 256, ≤ capacity) the
	// attention actually reads. The cache WRITE targets the full storage, but
	// the SDPA attends a sliced view covering offset+1 — without it the
	// masked positions of the whole storage band are still read every token,
	// which scales the read with CAPACITY instead of fill (catastrophic on
	// wide-head global layers: 31B = 16 KV heads × 512 dims). Crossing a
	// band re-keys the trace, like capacity.
	attnBand int32
	// seqLen is the step's token count: 1 = the decode step; 2..5 = the MTP
	// verify block (draft + carry written and attended in one pass). Each
	// seqLen traces separately, like any other shape in the key.
	seqLen     int32
	pliDim     int32 // 0 = layer has no per-layer-input block
	useKEqV    bool
	hasFreqs   bool
	hasScalar  bool
	ropeDims   int32
	ropeBase   float32
	scale      float32
	eps        float32
	xDType     metal.DType
	cacheDType metal.DType
	quant      [9]gemma4LinearSig

	// MoE signature — zero unless the layer routes experts. The dual-branch
	// FFN (local MLP + routed experts) traces per configuration: expert
	// count and top-k shape the gather geometry; the switch-linear quant
	// sigs shape the GatherQMM calls; the router flags pick the score path.
	moe                    bool
	experts                int32
	moeTopK                int32
	gateUpFused            bool
	hasPerExpertScale      bool
	routerScalePrecomputed bool
	routerProjDense        bool
	routerRootSize         float32
	routerEps              float32
	moeQuant               [4]gemma4LinearSig // routerProj, expertGateUp|expertGate, expertUp, expertDown
}

// gemma4AttnBandFor returns the power-of-two attention length band (floor
// 256) covering needed tokens, clamped to the storage capacity.
func gemma4AttnBandFor(needed, capacity int32) int32 {
	band := int32(256)
	for band < needed {
		band <<= 1
	}
	if band > capacity {
		band = capacity
	}
	return band
}

var (
	gemma4CompiledLayerFns    sync.Map // gemma4CompiledLayerKey -> *metal.CompiledFunc
	gemma4CompiledLayerPoison sync.Map // gemma4CompiledLayerKey -> true

	compiledLayerDecodeHits atomic.Uint64

	compiledLayerDeclineOnce sync.Once
)

// compiledLayerDecline reports (once per process) why the first layer decode
// declined the compiled closure while the gate was on — the difference between
// "compiled and byte-equal" and "declined everywhere and byte-equal" is
// invisible from output alone.
func compiledLayerDecline(layerIdx int32, reason string) (*metal.Array, sharedKV, bool) {
	compiledLayerDeclineOnce.Do(func() {
		core.Info("mlx: compiled layer decode declining", "layer", layerIdx, "reason", reason)
	})
	return nil, sharedKV{}, false
}

// CompiledLayerDecodeHits reports how many layer decode steps ran through the
// compiled closure this process. Tests use it to prove the compiled path
// actually served (byte-equal output alone cannot distinguish "compiled and
// correct" from "declined everywhere").
func CompiledLayerDecodeHits() uint64 { return compiledLayerDecodeHits.Load() }

// gemma4CompiledLayerState is the per-layer compile-eligibility result, built
// once on first eligible decode: the canonical weight input list (borrowed
// layer weights) and the key fields that never change per layer. Per-token
// fields (regime, capacity, dtypes) are completed per call.
type gemma4CompiledLayerState struct {
	declined bool
	consumer bool // weight list built for the consumer regime (no K/V path)
	key      gemma4CompiledLayerKey
	weights  []*metal.Array
	linears  []*metal.Linear // for the per-call LoRA re-check
}

const (
	gemma4LayerOutHidden = 0
	gemma4LayerOutKeys   = 1
	gemma4LayerOutValues = 2

	// gemma4MaxCompiledSeqLen bounds the compiled step's token count: 1 is
	// the decode step; 2..5 covers the MTP verify block (draft 2-4 + carry).
	// Longer blocks are prefill-shaped and keep the uncompiled chunked path.
	gemma4MaxCompiledSeqLen = 5
)

// compiledDecodeForward runs the whole-layer compiled decode step when the
// gate is on and the layer, cache regime, and inputs are trace-eligible.
// ok=false means the caller runs the normal uncompiled path.
func (l *Gemma4DecoderLayer) compiledDecodeForward(x *metal.Array, c metal.Cache, B, L int32, mask, pli *metal.Array, prev sharedKV, cfg *Gemma4TextConfig) (out *metal.Array, kv sharedKV, ok bool) {
	if !metal.CompiledLayerDecodeEnabled() {
		return nil, sharedKV{}, false
	}
	if B != 1 || L < 1 || L > gemma4MaxCompiledSeqLen || mask != nil {
		// Prefill and masked passes decline silently — only decode-shaped
		// declines are worth the one-shot diagnostic below. L 2..5 is the MTP
		// verify block (mask arrives nil there: forwardHidden only builds
		// layer masks when L exceeds the sliding window).
		return nil, sharedKV{}, false
	}
	if x == nil || !x.Valid() || cfg == nil {
		return compiledLayerDecline(l.LayerIdx, "invalid input")
	}
	if l.FFNMemory != nil {
		return compiledLayerDecline(l.LayerIdx, "FFN-memory layer")
	}

	consumer := prev.HasState()
	if consumer && (prev.HasPages() || !prev.Fixed || !gemma4ValidKV(prev.Keys, prev.Values)) {
		return compiledLayerDecline(l.LayerIdx, "shared KV state is not fixed")
	}
	var fixed *metal.FixedKVCache
	if !consumer {
		var isFixed bool
		fixed, isFixed = c.(*metal.FixedKVCache)
		if !isFixed || fixed == nil || fixed.MaxSize() <= 0 {
			return compiledLayerDecline(l.LayerIdx, core.Sprintf("cache is %T, not a sized FixedKVCache", c))
		}
		// Wide (512-dim) heads need no pre-decline here: the pre-cap step is
		// composed in-trace over a fill-band slice (the 512-wide sdpa_vector
		// kernel is present and byte-exact), so wide global layers compile in
		// every mode without the capacity-wide read the guarded native call
		// protected against.
	}

	state := l.compiledLayerState(consumer, cfg)
	if state == nil || state.declined || state.consumer != consumer {
		return compiledLayerDecline(l.LayerIdx, "layer weights are not trace-eligible")
	}
	// LoRA attaches after load; the trace carries base weights only.
	for _, linear := range state.linears {
		if linear.LoRA != nil {
			return compiledLayerDecline(l.LayerIdx, "LoRA adapter attached")
		}
	}
	if (state.key.pliDim > 0) != (pli != nil && pli.Valid()) {
		return compiledLayerDecline(l.LayerIdx, "per-layer input presence mismatch")
	}

	key := state.key
	key.xDType = x.Dtype()

	var cacheK, cacheV *metal.Array
	var offset int
	var shift, last *metal.Array
	if consumer {
		key.regime = gemma4LayerConsumer
		cacheK, cacheV = prev.Keys, prev.Values
		offset = prev.Offset
	} else {
		// Band-stepped storage: grow before borrowing when the next token
		// would cross the current band (no-op otherwise; re-keys the trace
		// for the new capacity).
		fixed.EnsureDecodeCapacityFor(int(L))
		fixedState := fixed.BorrowedFixedState()
		if !gemma4ValidKV(fixedState.Keys, fixedState.Values) {
			return compiledLayerDecline(l.LayerIdx, "fixed cache storage not allocated yet")
		}
		cacheK, cacheV = fixedState.Keys, fixedState.Values
		offset = fixed.Offset()
		switch {
		case offset+int(L) <= fixed.MaxSize():
			key.regime = gemma4LayerOwnerPreCap
		case L == 1 && metal.NativeFixedSlidingAttentionEnabled() && fixed.Len() >= fixed.MaxSize():
			key.regime = gemma4LayerOwnerPostCap
			shift, last = fixed.SlidingUpdateInputs()
			if shift == nil || last == nil || !shift.Valid() || !last.Valid() {
				return compiledLayerDecline(l.LayerIdx, "sliding update inputs unavailable")
			}
		default:
			return compiledLayerDecline(l.LayerIdx, "no compiled regime for cache fill state")
		}
	}
	if cacheK.NumDims() != 4 || cacheV.NumDims() != 4 ||
		int32(cacheK.Dim(1)) != key.kvHeads || int32(cacheK.Dim(3)) != key.headDim {
		return compiledLayerDecline(l.LayerIdx, "cache geometry mismatch")
	}
	key.capacity = int32(cacheK.Dim(2))
	key.cacheDType = cacheK.Dtype()
	switch key.regime {
	case gemma4LayerOwnerPostCap:
		// The sliding window is the read set — already fill-bounded.
		key.attnBand = key.capacity
	default:
		key.attnBand = gemma4AttnBandFor(int32(offset)+L, key.capacity)
	}
	key.seqLen = L

	if _, poisoned := gemma4CompiledLayerPoison.Load(key); poisoned {
		return nil, sharedKV{}, false
	}

	offsetArr := metal.FromValue(offset)
	defer func() {
		if recovered := recover(); recovered != nil {
			core.Error("mlx: compiled Gemma 4 layer decode failed; falling back to uncompiled paths",
				"layer", l.LayerIdx, "regime", key.regime, "error", recovered, "stack", string(debug.Stack()))
			gemma4CompiledLayerPoison.Store(key, true)
			metal.Free(offsetArr)
			out, kv, ok = nil, sharedKV{}, false
		}
	}()

	// Dynamic inputs in canonical order, then the layer's cached weight list.
	inputs := make([]*metal.Array, 0, 6+len(state.weights))
	inputs = append(inputs, x, cacheK, cacheV, offsetArr)
	if key.regime == gemma4LayerOwnerPostCap {
		inputs = append(inputs, shift, last)
	}
	if key.pliDim > 0 {
		inputs = append(inputs, pli)
	}
	inputs = append(inputs, state.weights...)

	outs := gemma4CompiledLayerFn(key).Call(inputs...)

	if consumer {
		if len(outs) != 1 || outs[0] == nil || !outs[0].Valid() {
			metal.Free(outs...)
			metal.Free(offsetArr)
			gemma4CompiledLayerPoison.Store(key, true)
			return nil, sharedKV{}, false
		}
		metal.Free(offsetArr)
		compiledLayerDecodeHits.Add(1)
		return outs[0], prev, true
	}

	if len(outs) != 3 || !gemma4CompiledLayerOutputsValid(outs, cacheK, cacheV) {
		metal.Free(outs...)
		metal.Free(offsetArr)
		gemma4CompiledLayerPoison.Store(key, true)
		return nil, sharedKV{}, false
	}
	var fixedState metal.FixedKVState
	if key.regime == gemma4LayerOwnerPreCap {
		// Masked-write adoption: the in-trace write landed at index offset,
		// hidden by the causal mask until the offset advances. The old
		// storage handle is freed (not retired) so MLX donates the buffer
		// and the write is in place — no full-band copy per token.
		fixedState = fixed.ReplaceFixedWriteThroughBorrowed(outs[gemma4LayerOutKeys], outs[gemma4LayerOutValues], int(L))
	} else {
		// Post-cap sliding rotates the window physically — keep the staged
		// (copy-on-adopt) lane so a speculated rotate stays discardable.
		fixedState = fixed.ReplaceFixedFromNativeBorrowed(outs[gemma4LayerOutKeys], outs[gemma4LayerOutValues], int(L))
	}
	if !gemma4ValidKV(fixedState.Keys, fixedState.Values) {
		metal.Free(outs[gemma4LayerOutHidden])
		metal.Free(offsetArr)
		gemma4CompiledLayerPoison.Store(key, true)
		return nil, sharedKV{}, false
	}
	fixed.RetireAfterNextEval(offsetArr)
	kv = sharedKV{Keys: fixedState.Keys, Values: fixedState.Values, Offset: offset, Fixed: true, Borrowed: true}
	compiledLayerDecodeHits.Add(1)
	return outs[gemma4LayerOutHidden], kv, true
}

// gemma4CompiledLayerOutputsValid checks the closure's updated K/V keep the
// cache storage geometry before the cache adopts them.
func gemma4CompiledLayerOutputsValid(outs []*metal.Array, cacheK, cacheV *metal.Array) bool {
	for _, arr := range outs {
		if arr == nil || !arr.Valid() {
			return false
		}
	}
	newK, newV := outs[gemma4LayerOutKeys], outs[gemma4LayerOutValues]
	if newK.NumDims() != cacheK.NumDims() || newV.NumDims() != cacheV.NumDims() {
		return false
	}
	for axis := 0; axis < 4; axis++ {
		if newK.Dim(axis) != cacheK.Dim(axis) || newV.Dim(axis) != cacheV.Dim(axis) {
			return false
		}
	}
	return newK.Dtype() == cacheK.Dtype() && newV.Dtype() == cacheV.Dtype()
}

// compiledLayerState resolves (building once) the layer's compile eligibility,
// canonical weight inputs, and the per-layer-constant key fields.
func (l *Gemma4DecoderLayer) compiledLayerState(consumer bool, cfg *Gemma4TextConfig) *gemma4CompiledLayerState {
	if cached := l.compiledDecode.Load(); cached != nil {
		return cached
	}
	state := buildGemma4CompiledLayerState(l, consumer, cfg)
	if l.compiledDecode.CompareAndSwap(nil, state) {
		return state
	}
	return l.compiledDecode.Load()
}

func buildGemma4CompiledLayerState(l *Gemma4DecoderLayer, consumer bool, cfg *Gemma4TextConfig) *gemma4CompiledLayerState {
	declined := &gemma4CompiledLayerState{declined: true, consumer: consumer}
	a := l.Attention
	if a == nil || a.HeadDim <= 0 || a.NKVHeads <= 0 || cfg.NumAttentionHeads <= 0 {
		return declined
	}
	if l.InputNormScaled == nil || l.PostAttnNormScaled == nil || l.PreFFNormScaled == nil || l.PostFFNormScaled == nil ||
		a.QNormScaled == nil || a.KNormScaled == nil || l.MLP == nil {
		return declined
	}
	pliPresent := l.PerLayerInputGate != nil && l.PerLayerProjection != nil && l.PostPerLayerInputNormScaled != nil
	if pliPresent && cfg.HiddenSizePerLayerInput <= 0 {
		return declined
	}

	state := &gemma4CompiledLayerState{consumer: consumer}
	key := &state.key
	key.hidden = cfg.HiddenSize
	key.qHeads = cfg.NumAttentionHeads
	key.kvHeads = a.NKVHeads
	key.headDim = a.HeadDim
	key.useKEqV = a.UseKEqV
	key.hasFreqs = a.RopeFreqs != nil && a.RopeFreqs.Valid()
	key.hasScalar = l.LayerScalar != nil && l.LayerScalar.Valid()
	key.ropeDims = a.RopeRotatedDim
	key.ropeBase = a.RopeBase
	key.scale = a.Scale
	key.eps = cfg.RMSNormEps
	if pliPresent {
		key.pliDim = cfg.HiddenSizePerLayerInput
	}

	// Canonical weight order. gemma4CompiledLayerStep's reader consumes inputs
	// in exactly this order — change both together.
	addLinear := func(slot int, linear *metal.Linear) bool {
		if linear == nil || linear.LoRA != nil || linear.Weight == nil || !linear.Weight.Valid() ||
			linear.Scales == nil || !linear.Scales.Valid() || linear.Biases == nil || !linear.Biases.Valid() {
			return false
		}
		if linear.Bias != nil && linear.Bias.Valid() {
			return false
		}
		if !metal.IsAffineQuantizationMode(linear.QuantizationMode) {
			return false
		}
		key.quant[slot] = gemma4LinearSig{bits: int32(linear.Bits), groupSize: int32(linear.GroupSize), mode: linear.QuantizationMode}
		state.weights = append(state.weights, linear.Weight, linear.Scales, linear.Biases)
		state.linears = append(state.linears, linear)
		return true
	}

	state.weights = append(state.weights, l.InputNormScaled)
	if !addLinear(0, a.QProj) {
		return declined
	}
	state.weights = append(state.weights, a.QNormScaled)
	if !consumer {
		if !addLinear(1, a.KProj) {
			return declined
		}
		state.weights = append(state.weights, a.KNormScaled)
		if !key.useKEqV {
			if !addLinear(2, a.VProj) {
				return declined
			}
		} else if a.VProj != nil {
			return declined
		}
	}
	if !addLinear(3, a.OProj) {
		return declined
	}
	state.weights = append(state.weights, l.PostAttnNormScaled, l.PreFFNormScaled)
	if !addLinear(4, l.MLP.GateProj) || !addLinear(5, l.MLP.UpProj) || !addLinear(6, l.MLP.DownProj) {
		return declined
	}
	state.weights = append(state.weights, l.PostFFNormScaled)

	if l.EnableMoE || l.Router != nil || l.Experts != nil {
		// The MoE dual branch: local MLP (already collected above) plus the
		// routed experts. Everything the traced body needs enters as inputs.
		router, experts := l.Router, l.Experts
		if !l.EnableMoE || router == nil || experts == nil || router.Proj == nil ||
			l.PreFFNorm2Scaled == nil || l.PostFFNorm1Scaled == nil || l.PostFFNorm2Scaled == nil {
			return declined
		}
		if router.TopK <= 0 {
			return declined
		}
		addSwitch := func(slot int, sw *metal.SwitchLinear) bool {
			if sw == nil || sw.Weight == nil || !sw.Weight.Valid() ||
				sw.Scales == nil || !sw.Scales.Valid() || sw.Biases == nil || !sw.Biases.Valid() {
				return false
			}
			if sw.Bias != nil && sw.Bias.Valid() {
				return false
			}
			if !metal.IsAffineQuantizationMode(sw.QuantizationMode) ||
				metal.RequiresDenseQuantizedMatmulFallback(sw.QuantizationMode) {
				return false
			}
			key.moeQuant[slot] = gemma4LinearSig{bits: int32(sw.Bits), groupSize: int32(sw.GroupSize), mode: sw.QuantizationMode}
			state.weights = append(state.weights, sw.Weight, sw.Scales, sw.Biases)
			return true
		}
		key.moe = true
		key.moeTopK = router.TopK
		key.routerEps = router.Eps
		state.weights = append(state.weights, l.PreFFNorm2Scaled, l.PostFFNorm1Scaled, l.PostFFNorm2Scaled)
		if router.ScaleScaled != nil && router.ScaleScaled.Valid() {
			key.routerScalePrecomputed = true
			state.weights = append(state.weights, router.ScaleScaled)
		} else if router.Scale != nil && router.Scale.Valid() {
			key.routerRootSize = router.RootSize
			state.weights = append(state.weights, router.Scale)
		} else {
			return declined
		}
		if router.PerExpertScale != nil && router.PerExpertScale.Valid() {
			key.hasPerExpertScale = true
			state.weights = append(state.weights, router.PerExpertScale)
		}
		proj := router.Proj
		if proj.LoRA != nil || (proj.Bias != nil && proj.Bias.Valid()) {
			return declined
		}
		if proj.Scales != nil && proj.Scales.Valid() {
			if proj.Weight == nil || !proj.Weight.Valid() || proj.Biases == nil || !proj.Biases.Valid() ||
				!metal.IsAffineQuantizationMode(proj.QuantizationMode) {
				return declined
			}
			key.moeQuant[0] = gemma4LinearSig{bits: int32(proj.Bits), groupSize: int32(proj.GroupSize), mode: proj.QuantizationMode}
			key.experts = int32(proj.Weight.Dim(0))
			state.weights = append(state.weights, proj.Weight, proj.Scales, proj.Biases)
			state.linears = append(state.linears, proj)
		} else if proj.Weight != nil && proj.Weight.Valid() {
			key.routerProjDense = true
			key.moeQuant[0] = gemma4LinearSig{mode: "dense"}
			key.experts = int32(proj.Weight.Dim(0))
			state.weights = append(state.weights, proj.Weight)
			state.linears = append(state.linears, proj)
		} else {
			return declined
		}
		if experts.GateUpProj != nil && experts.GateUpProj.Weight != nil && experts.GateUpProj.Weight.Valid() {
			key.gateUpFused = true
			if !addSwitch(1, experts.GateUpProj) {
				return declined
			}
		} else {
			if !addSwitch(1, experts.GateProj) || !addSwitch(2, experts.UpProj) {
				return declined
			}
		}
		if !addSwitch(3, experts.DownProj) {
			return declined
		}
	}
	if pliPresent {
		if !addLinear(7, l.PerLayerInputGate) || !addLinear(8, l.PerLayerProjection) {
			return declined
		}
		state.weights = append(state.weights, l.PostPerLayerInputNormScaled)
	}
	if key.hasScalar {
		state.weights = append(state.weights, l.LayerScalar)
	}
	if key.hasFreqs {
		state.weights = append(state.weights, a.RopeFreqs)
	}
	return state
}

// gemma4CompiledLayerFn returns (building on first use) the compiled layer
// closure for a trace key.
func gemma4CompiledLayerFn(key gemma4CompiledLayerKey) *metal.CompiledFunc {
	if cached, found := gemma4CompiledLayerFns.Load(key); found {
		return cached.(*metal.CompiledFunc)
	}
	// shapeless=false: the trace key pins every input shape already, and the
	// layer graph contains AsStrided, whose output shape MLX cannot re-infer
	// under shapeless replay.
	fn := metal.CompileShapeless(gemma4CompiledLayerStep(key), false)
	cached, _ := gemma4CompiledLayerFns.LoadOrStore(key, fn)
	return cached.(*metal.CompiledFunc)
}

// gemma4LayerInputReader consumes closure inputs in the canonical order
// buildGemma4CompiledLayerState appends them.
type gemma4LayerInputReader struct {
	in  []*metal.Array
	pos int
}

func (r *gemma4LayerInputReader) next() *metal.Array {
	arr := r.in[r.pos]
	r.pos++
	return arr
}

func (r *gemma4LayerInputReader) linear(sig gemma4LinearSig) *metal.Linear {
	weight, scales, biases := r.next(), r.next(), r.next()
	return &metal.Linear{Weight: weight, Scales: scales, Biases: biases, QuantizationMode: sig.mode, GroupSize: int(sig.groupSize), Bits: int(sig.bits)}
}

// gemma4CompiledLayerStep builds the closure body for a trace key: the exact
// op sequence of the uncompiled decode path, composed from closure inputs.
func gemma4CompiledLayerStep(key gemma4CompiledLayerKey) func([]*metal.Array) []*metal.Array {
	return func(in []*metal.Array) []*metal.Array {
		r := &gemma4LayerInputReader{in: in}
		x := r.next()
		cacheK := r.next()
		cacheV := r.next()
		offset := r.next()
		var shift, last *metal.Array
		if key.regime == gemma4LayerOwnerPostCap {
			shift, last = r.next(), r.next()
		}
		var pli *metal.Array
		if key.pliDim > 0 {
			pli = r.next()
		}
		inputNorm := r.next()
		qProj := r.linear(key.quant[0])
		qNormScaled := r.next()
		var kProj *metal.Linear
		var kNormScaled *metal.Array
		var vProj *metal.Linear
		if key.regime != gemma4LayerConsumer {
			kProj = r.linear(key.quant[1])
			kNormScaled = r.next()
			if !key.useKEqV {
				vProj = r.linear(key.quant[2])
			}
		}
		oProj := r.linear(key.quant[3])
		postAttnNorm := r.next()
		preFFNorm := r.next()
		gateProj := r.linear(key.quant[4])
		upProj := r.linear(key.quant[5])
		downProj := r.linear(key.quant[6])
		postFFNorm := r.next()
		var preFFNorm2, postFFNorm1, postFFNorm2 *metal.Array
		var routerScale, perExpertScale *metal.Array
		var routerProj *metal.Linear
		var expertGateUp, expertGate, expertUp, expertDown *metal.SwitchLinear
		if key.moe {
			preFFNorm2 = r.next()
			postFFNorm1 = r.next()
			postFFNorm2 = r.next()
			routerScale = r.next()
			if key.hasPerExpertScale {
				perExpertScale = r.next()
			}
			if key.routerProjDense {
				routerProj = &metal.Linear{Weight: r.next()}
			} else {
				routerProj = r.linear(key.moeQuant[0])
			}
			mkSwitch := func(sig gemma4LinearSig) *metal.SwitchLinear {
				return &metal.SwitchLinear{
					Weight: r.next(), Scales: r.next(), Biases: r.next(),
					QuantizationMode: sig.mode, GroupSize: int(sig.groupSize), Bits: int(sig.bits),
				}
			}
			if key.gateUpFused {
				expertGateUp = mkSwitch(key.moeQuant[1])
			} else {
				expertGate = mkSwitch(key.moeQuant[1])
				expertUp = mkSwitch(key.moeQuant[2])
			}
			expertDown = mkSwitch(key.moeQuant[3])
		}
		var pliGate, pliProj *metal.Linear
		var pliNorm *metal.Array
		if key.pliDim > 0 {
			pliGate = r.linear(key.quant[7])
			pliProj = r.linear(key.quant[8])
			pliNorm = r.next()
		}
		var layerScalar *metal.Array
		if key.hasScalar {
			layerScalar = r.next()
		}
		var ropeFreqs *metal.Array
		if key.hasFreqs {
			ropeFreqs = r.next()
		}

		applyRoPE := func(t *metal.Array) *metal.Array {
			if key.hasFreqs {
				return metal.RoPEWithOffsetArray(t, int(key.headDim), false, 0, 1.0, offset, ropeFreqs)
			}
			return metal.RoPEWithOffsetArray(t, int(key.ropeDims), false, key.ropeBase, 1.0, offset, nil)
		}

		// Attention: norm → Q (+K/V for owners) → Q/K norms → RoPE → the
		// fixed-cache attention step → O projection.
		normed := metal.RMSNorm(x, inputNorm, key.eps)
		qp := qProj.Forward(normed)
		// [1, L, H*D] -> [1, H, L, D] view: head stride D, row stride H*D.
		q := metal.AsStrided(qp, []int32{1, key.qHeads, key.seqLen, key.headDim},
			[]int64{int64(key.seqLen) * int64(key.qHeads*key.headDim), int64(key.headDim), int64(key.qHeads * key.headDim), 1}, 0)
		metal.Free(qp)
		qn := metal.RMSNorm(q, qNormScaled, key.eps)
		metal.Free(q)
		qr := applyRoPE(qn)
		metal.Free(qn)

		// bandView slices the attention read set to the fill band — a view,
		// no copy. The full storage stays the write target and the output.
		bandView := func(t *metal.Array) *metal.Array {
			if key.attnBand >= key.capacity {
				return nil
			}
			return metal.Slice4(t, 0, 0, 0, 0, 1, key.kvHeads, key.attnBand, key.headDim)
		}

		var attnOut, newK, newV *metal.Array
		if key.regime == gemma4LayerConsumer {
			attnQ := qr
			var ownedAttnQ *metal.Array
			if qr.Dtype() != key.cacheDType && (key.cacheDType == metal.DTypeFloat16 || key.cacheDType == metal.DTypeBFloat16) {
				ownedAttnQ = metal.AsType(qr, key.cacheDType)
				attnQ = ownedAttnQ
			}
			kAttn, vAttn := cacheK, cacheV
			kBand, vBand := bandView(cacheK), bandView(cacheV)
			if kBand != nil {
				kAttn, vAttn = kBand, vBand
			}
			mask := metal.MultiTokenCausalMask(int(key.attnBand), offset, int(key.seqLen))
			attnOut = metal.ScaledDotProductAttentionWithMask(attnQ, kAttn, vAttn, mask, key.scale)
			metal.Free(mask, ownedAttnQ, kBand, vBand)
		} else {
			kp := kProj.Forward(normed)
			k := metal.AsStrided(kp, []int32{1, key.kvHeads, key.seqLen, key.headDim},
				[]int64{int64(key.seqLen) * int64(key.kvHeads*key.headDim), int64(key.headDim), int64(key.kvHeads * key.headDim), 1}, 0)
			metal.Free(kp)
			var v *metal.Array
			if key.useKEqV {
				// K=V shares the projection source, not the final tensors: K
				// takes KNorm+RoPE, V takes the unscaled value RMSNorm.
				v = k.Clone()
			} else {
				vp := vProj.Forward(normed)
				v = metal.AsStrided(vp, []int32{1, key.kvHeads, key.seqLen, key.headDim},
					[]int64{int64(key.seqLen) * int64(key.kvHeads*key.headDim), int64(key.headDim), int64(key.kvHeads * key.headDim), 1}, 0)
				metal.Free(vp)
			}
			kn := metal.RMSNorm(k, kNormScaled, key.eps)
			metal.Free(k)
			kr := applyRoPE(kn)
			metal.Free(kn)
			vn := metal.RMSNormNoScale(v, key.eps)
			metal.Free(v)

			// Storage-dtype follow: convert the new K/V (and the query the
			// SDPA reads against them) to the cache dtype before the write —
			// the uncompiled paths convert via storageKVPair. Covers both
			// directions: half-precision storage under an fp32 stream, and a
			// restored fp32 cache (an older sleep state) under the bf16
			// stream.
			if kr.Dtype() != key.cacheDType {
				castK := metal.AsType(kr, key.cacheDType)
				metal.Free(kr)
				kr = castK
				castV := metal.AsType(vn, key.cacheDType)
				metal.Free(vn)
				vn = castV
			}
			if qr.Dtype() != key.cacheDType {
				castQ := metal.AsType(qr, key.cacheDType)
				metal.Free(qr)
				qr = castQ
			}

			if key.regime == gemma4LayerOwnerPostCap {
				var stepOK bool
				var stepErr error
				attnOut, newK, newV, stepOK, stepErr = metal.NativeFixedSlidingSingleTokenAttention(qr, cacheK, cacheV, kr, vn, shift, last, key.scale)
				if stepErr != nil {
					metal.Free(kr, vn)
					panic(stepErr)
				}
				if !stepOK {
					shapes := core.Sprintf("q %v · cacheK %v · cacheV %v · k %v · v %v",
						qr.Shape(), cacheK.Shape(), cacheV.Shape(), kr.Shape(), vn.Shape())
					metal.Free(kr, vn)
					panic("mlx: fixed sliding attention declined inside the compiled layer trace (" + shapes + ")")
				}
			} else {
				// Pre-cap step composed in-trace, mirroring the C++ compiled
				// fixed single-token attention op for op (offset-indexed
				// write, offset causal mask, q×scale then SDPA at 1.0) — with
				// the attention read sliced to the fill band. The write
				// targets the full storage; the band views are the read set.
				newK = metal.MultiTokenCacheUpdate(cacheK, kr, offset, int(key.seqLen))
				newV = metal.MultiTokenCacheUpdate(cacheV, vn, offset, int(key.seqLen))
				kAttn, vAttn := newK, newV
				kBand, vBand := bandView(newK), bandView(newV)
				if kBand != nil {
					kAttn, vAttn = kBand, vBand
				}
				mask := metal.MultiTokenCausalMask(int(key.attnBand), offset, int(key.seqLen))
				scaledQ := metal.MulScalar(qr, key.scale)
				attnOut = metal.ScaledDotProductAttentionWithMask(scaledQ, kAttn, vAttn, mask, 1.0)
				metal.Free(mask, scaledQ, kBand, vBand)
			}
			metal.Free(kr, vn)
		}
		metal.Free(qr, normed)

		transposed := metal.Transpose4(attnOut, 0, 2, 1, 3)
		metal.Free(attnOut)
		reshaped := metal.Reshape(transposed, 1, key.seqLen, key.qHeads*key.headDim)
		metal.Free(transposed)
		oOut := oProj.Forward(reshaped)
		metal.Free(reshaped)

		// Residual + feed-forward, mirroring Gemma4DecoderLayer.forward.
		attnNormed := metal.RMSNorm(oOut, postAttnNorm, key.eps)
		metal.Free(oOut)
		h := metal.Add(x, attnNormed)
		metal.Free(attnNormed)

		var ffResidual *metal.Array
		if key.moe {
			// Dual-branch FFN, mirroring Gemma4DecoderLayer.forward's MoE arm:
			// the local MLP lane and the routed expert lane normalise
			// independently, combine, then take the standard post-FF norm.
			h1In := metal.RMSNorm(h, preFFNorm, key.eps)
			h1 := metal.TracedGELUMLPForward(h1In, gateProj, upProj, downProj)
			metal.Free(h1In)

			h2In := metal.RMSNorm(h, preFFNorm2, key.eps)
			// Router (mirrors Gemma4Router.forward, scores from h):
			var scaled *metal.Array
			if key.routerScalePrecomputed {
				scaled = routerScale
			} else {
				scaled = metal.MulScalar(routerScale, key.routerRootSize)
				defer metal.Free(scaled)
			}
			routerNormed := metal.RMSNorm(h, scaled, key.routerEps)
			expertScores := routerProj.Forward(routerNormed)
			metal.Free(routerNormed)
			var topKIndices, topKWeights *metal.Array
			if idx, w, ok, err := metal.NativeMoERouterTopK(expertScores, perExpertScale, int(key.moeTopK)); ok && err == nil {
				topKIndices, topKWeights = idx, w
				metal.Free(expertScores)
			} else {
				kth := key.experts - key.moeTopK
				allIdx := metal.Argpartition(expertScores, int(kth), -1)
				topKIndices = metal.SliceAxis(allIdx, -1, kth, key.experts)
				metal.Free(allIdx)
				raw := metal.TakeAlongAxis(expertScores, topKIndices, -1)
				metal.Free(expertScores)
				softmaxed := metal.Softmax(raw)
				metal.Free(raw)
				if perExpertScale != nil {
					scale := metal.Take(perExpertScale, topKIndices, 0)
					topKWeights = metal.Mul(softmaxed, scale)
					metal.Free(softmaxed, scale)
				} else {
					topKWeights = softmaxed
				}
			}

			// Experts (mirrors Gemma4Experts.forward — GatherQMM lanes):
			expanded1 := metal.ExpandDims(h2In, 2)
			expanded := metal.ExpandDims(expanded1, 2)
			metal.Free(expanded1, h2In)
			var gateE, upE *metal.Array
			if key.gateUpFused {
				gateUp := expertGateUp.Forward(expanded, topKIndices)
				var ok bool
				gateE, upE, ok = splitLastDimArray(gateUp)
				metal.Free(gateUp)
				if !ok {
					panic("mlx: compiled MoE layer: fused gate/up split failed")
				}
			} else {
				upE = expertUp.Forward(expanded, topKIndices)
				gateE = expertGate.Forward(expanded, topKIndices)
			}
			metal.Free(expanded)
			activatedE := metal.GeluGateMul(gateE, upE)
			metal.Free(gateE, upE)
			downE := expertDown.Forward(activatedE, topKIndices)
			metal.Free(activatedE)
			downSq := metal.Squeeze(downE, 3)
			metal.Free(downE)
			wExp := metal.ExpandDims(topKWeights, 3)
			weighted := metal.Mul(wExp, downSq)
			metal.Free(wExp, downSq, topKIndices, topKWeights)
			h2 := metal.Sum(weighted, -2, false)
			metal.Free(weighted)

			h1Normed := metal.RMSNorm(h1, postFFNorm1, key.eps)
			h2Normed := metal.RMSNorm(h2, postFFNorm2, key.eps)
			metal.Free(h1, h2)
			combined := metal.Add(h1Normed, h2Normed)
			metal.Free(h1Normed, h2Normed)
			ffResidual = metal.RMSNorm(combined, postFFNorm, key.eps)
			metal.Free(combined)
		} else {
			ffIn := metal.RMSNorm(h, preFFNorm, key.eps)
			ff := metal.TracedGELUMLPForward(ffIn, gateProj, upProj, downProj)
			metal.Free(ffIn)
			ffResidual = metal.RMSNorm(ff, postFFNorm, key.eps)
			metal.Free(ff)
		}
		hNext := metal.Add(h, ffResidual)
		metal.Free(h, ffResidual)

		if key.pliDim > 0 {
			gate := pliGate.Forward(hNext)
			multiplied := metal.GeluGateMul(gate, pli)
			metal.Free(gate)
			projected := pliProj.Forward(multiplied)
			metal.Free(multiplied)
			projectedNormed := metal.RMSNorm(projected, pliNorm, key.eps)
			metal.Free(projected)
			gated := metal.Add(hNext, projectedNormed)
			metal.Free(hNext, projectedNormed)
			hNext = gated
		}
		if key.hasScalar {
			scaled := metal.Mul(hNext, layerScalar)
			metal.Free(hNext)
			hNext = scaled
		}

		if key.regime == gemma4LayerConsumer {
			return []*metal.Array{hNext}
		}
		return []*metal.Array{hNext, newK, newV}
	}
}
