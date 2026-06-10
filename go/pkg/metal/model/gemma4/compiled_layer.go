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
	regime     gemma4LayerRegime
	hidden     int32
	qHeads     int32
	kvHeads    int32
	headDim    int32
	capacity   int32
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
)

// compiledDecodeForward runs the whole-layer compiled decode step when the
// gate is on and the layer, cache regime, and inputs are trace-eligible.
// ok=false means the caller runs the normal uncompiled path.
func (l *Gemma4DecoderLayer) compiledDecodeForward(x *metal.Array, c metal.Cache, B, L int32, mask, pli *metal.Array, prev sharedKV, cfg *Gemma4TextConfig) (out *metal.Array, kv sharedKV, ok bool) {
	if !metal.CompiledLayerDecodeEnabled() {
		return nil, sharedKV{}, false
	}
	if B != 1 || L != 1 || mask != nil {
		// Prefill and masked passes decline silently — only decode-shaped
		// declines are worth the one-shot diagnostic below.
		return nil, sharedKV{}, false
	}
	if x == nil || !x.Valid() || cfg == nil {
		return compiledLayerDecline(l.LayerIdx, "invalid input")
	}
	if l.EnableMoE || l.Router != nil || l.Experts != nil || l.FFNMemory != nil {
		return compiledLayerDecline(l.LayerIdx, "MoE or FFN-memory layer")
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
		// The bundled metallib has no vector SDPA kernel for 512-wide fixed
		// single-token heads; the fixed attention step declines them, so the
		// closure pre-declines (mirrors nativeFixedSingleTokenAttentionAvailable).
		if l.Attention != nil && l.Attention.HeadDim >= 512 &&
			!metal.FixedWideSDPAAttentionEnabled() && !metal.FixedWideMatmulAttentionEnabled() {
			return compiledLayerDecline(l.LayerIdx, "wide attention head without a wide fixed kernel")
		}
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
		fixedState := fixed.BorrowedFixedState()
		if !gemma4ValidKV(fixedState.Keys, fixedState.Values) {
			return compiledLayerDecline(l.LayerIdx, "fixed cache storage not allocated yet")
		}
		cacheK, cacheV = fixedState.Keys, fixedState.Values
		offset = fixed.Offset()
		switch {
		case offset+1 <= fixed.MaxSize():
			key.regime = gemma4LayerOwnerPreCap
		case metal.NativeFixedSlidingAttentionEnabled() && fixed.Len() >= fixed.MaxSize():
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
	fixedState := fixed.ReplaceFixedFromNativeBorrowed(outs[gemma4LayerOutKeys], outs[gemma4LayerOutValues], int(L))
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
		q := metal.AsStrided(qp, []int32{1, key.qHeads, 1, key.headDim},
			[]int64{int64(key.qHeads * key.headDim), int64(key.headDim), int64(key.qHeads * key.headDim), 1}, 0)
		metal.Free(qp)
		qn := metal.RMSNorm(q, qNormScaled, key.eps)
		metal.Free(q)
		qr := applyRoPE(qn)
		metal.Free(qn)

		var attnOut, newK, newV *metal.Array
		if key.regime == gemma4LayerConsumer {
			attnQ := qr
			var ownedAttnQ *metal.Array
			if qr.Dtype() != key.cacheDType && (key.cacheDType == metal.DTypeFloat16 || key.cacheDType == metal.DTypeBFloat16) {
				ownedAttnQ = metal.AsType(qr, key.cacheDType)
				attnQ = ownedAttnQ
			}
			mask := metal.SingleTokenCausalMask(int(key.capacity), offset)
			attnOut = metal.ScaledDotProductAttentionWithMask(attnQ, cacheK, cacheV, mask, key.scale)
			metal.Free(mask, ownedAttnQ)
		} else {
			kp := kProj.Forward(normed)
			k := metal.AsStrided(kp, []int32{1, key.kvHeads, 1, key.headDim},
				[]int64{int64(key.kvHeads * key.headDim), int64(key.headDim), int64(key.kvHeads * key.headDim), 1}, 0)
			metal.Free(kp)
			var v *metal.Array
			if key.useKEqV {
				// K=V shares the projection source, not the final tensors: K
				// takes KNorm+RoPE, V takes the unscaled value RMSNorm.
				v = k.Clone()
			} else {
				vp := vProj.Forward(normed)
				v = metal.AsStrided(vp, []int32{1, key.kvHeads, 1, key.headDim},
					[]int64{int64(key.kvHeads * key.headDim), int64(key.headDim), int64(key.kvHeads * key.headDim), 1}, 0)
				metal.Free(vp)
			}
			kn := metal.RMSNorm(k, kNormScaled, key.eps)
			metal.Free(k)
			kr := applyRoPE(kn)
			metal.Free(kn)
			vn := metal.RMSNormNoScale(v, key.eps)
			metal.Free(v)

			var stepOK bool
			var stepErr error
			if key.regime == gemma4LayerOwnerPostCap {
				attnOut, newK, newV, stepOK, stepErr = metal.NativeFixedSlidingSingleTokenAttention(qr, cacheK, cacheV, kr, vn, shift, last, key.scale)
			} else {
				attnOut, newK, newV, stepOK, stepErr = metal.NativeFixedSingleTokenAttention(qr, cacheK, cacheV, kr, vn, offset, nil, key.scale)
			}
			if stepErr != nil {
				metal.Free(kr, vn)
				panic(stepErr)
			}
			if !stepOK {
				shapes := core.Sprintf("q %v · cacheK %v · cacheV %v · k %v · v %v",
					qr.Shape(), cacheK.Shape(), cacheV.Shape(), kr.Shape(), vn.Shape())
				metal.Free(kr, vn)
				panic("mlx: fixed single-token attention declined inside the compiled layer trace (" + shapes + ")")
			}
			metal.Free(kr, vn)
		}
		metal.Free(qr, normed)

		transposed := metal.Transpose4(attnOut, 0, 2, 1, 3)
		metal.Free(attnOut)
		reshaped := metal.Reshape(transposed, 1, 1, key.qHeads*key.headDim)
		metal.Free(transposed)
		oOut := oProj.Forward(reshaped)
		metal.Free(reshaped)

		// Residual + feed-forward, mirroring Gemma4DecoderLayer.forward.
		attnNormed := metal.RMSNorm(oOut, postAttnNorm, key.eps)
		metal.Free(oOut)
		h := metal.Add(x, attnNormed)
		metal.Free(attnNormed)

		ffIn := metal.RMSNorm(h, preFFNorm, key.eps)
		ff := metal.TracedGELUMLPForward(ffIn, gateProj, upProj, downProj)
		metal.Free(ffIn)
		ffResidual := metal.RMSNorm(ff, postFFNorm, key.eps)
		metal.Free(ff)
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
