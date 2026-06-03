// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/metal"
)

func nativeGemma4FixedOwnerAttentionBlock(x *metal.Array, fixed *metal.FixedKVCache, fixedMask *metal.Array, attn *Gemma4Attention, cfg *Gemma4TextConfig) (*metal.Array, sharedKV, bool, error) {
	if !nativeGemma4FixedOwnerAttentionBlockAvailable(x, fixed, fixedMask, attn, cfg) {
		return nil, sharedKV{}, false, nil
	}
	return metal.NativeGemma4FixedOwnerAttention(gemma4FixedAttentionRequest(x, nil, fixedMask, attn, nil, cfg), fixed)
}

func nativeGemma4FixedOwnerAttentionResidualBlock(residual, x *metal.Array, fixed *metal.FixedKVCache, fixedMask *metal.Array, attn *Gemma4Attention, postAttnNorm *metal.Array, cfg *Gemma4TextConfig) (*metal.Array, sharedKV, bool, error) {
	if !nativeGemma4FixedOwnerAttentionResidualBlockAvailable(residual, x, fixed, fixedMask, attn, postAttnNorm, cfg) {
		return nil, sharedKV{}, false, nil
	}
	return metal.NativeGemma4FixedOwnerAttentionResidual(gemma4FixedAttentionRequest(x, residual, fixedMask, attn, postAttnNorm, cfg), fixed)
}

// gemma4FixedAttentionRequest fills the metal fused fixed-attention request from
// the model architecture's attention block. KeyCache/ValueCache/Offset/Scale are
// resolved metal-side from the live FixedKVCache, so they stay nil here.
func gemma4FixedAttentionRequest(x, residual, fixedMask *metal.Array, attn *Gemma4Attention, postAttnNorm *metal.Array, cfg *Gemma4TextConfig) metal.Gemma4FixedAttentionRequest {
	return metal.Gemma4FixedAttentionRequest{
		X:                 x,
		Residual:          residual,
		Mask:              fixedMask,
		QProj:             attn.QProj,
		KProj:             attn.KProj,
		VProj:             attn.VProj,
		OProj:             attn.OProj,
		QNorm:             attn.QNormScaled,
		KNorm:             attn.KNormScaled,
		PostAttnNorm:      postAttnNorm,
		RopeFreqs:         attn.RopeFreqs,
		Scale:             attn.Scale,
		NumAttentionHeads: cfg.NumAttentionHeads,
		NumKeyValueHeads:  attn.NKVHeads,
		HeadDim:           attn.HeadDim,
		RopeDims:          attn.RopeRotatedDim,
		RopeBase:          attn.RopeBase,
	}
}

func nativeGemma4FixedOwnerAttentionBlockAvailable(x *metal.Array, fixed *metal.FixedKVCache, fixedMask *metal.Array, attn *Gemma4Attention, cfg *Gemma4TextConfig) bool {
	if x == nil || !x.Valid() || fixed == nil || attn == nil || cfg == nil {
		return false
	}
	if x.NumDims() != 3 || x.Dim(0) <= 0 || x.Dim(1) != 1 || fixed.maxSize <= 0 || fixed.Offset()+1 > fixed.maxSize {
		return false
	}
	if cfg.RMSNormEps != 1e-6 || cfg.NumAttentionHeads <= 0 || attn.NKVHeads <= 0 || attn.HeadDim <= 0 || attn.RopeRotatedDim <= 0 {
		return false
	}
	if attn.UseKEqV || cfg.NumAttentionHeads%attn.NKVHeads != 0 || x.Dim(2) != int(cfg.NumAttentionHeads*attn.HeadDim) {
		return false
	}
	if !nativeGemma4AttentionAvailable(attn) {
		return false
	}
	if fixedMask != nil && fixedMask.Valid() {
		if fixedMask.NumDims() != 4 ||
			fixedMask.Dim(0) != x.Dim(0) ||
			fixedMask.Dim(1) != 1 ||
			fixedMask.Dim(2) != 1 ||
			fixedMask.Dim(3) != fixed.maxSize {
			return false
		}
	}
	if attn.HeadDim >= 512 &&
		core.Env("GO_MLX_ENABLE_FIXED_WIDE_SDPA_ATTENTION") != "1" &&
		core.Env("GO_MLX_ENABLE_FIXED_WIDE_MATMUL_ATTENTION") != "1" {
		return false
	}
	return true
}

func nativeGemma4FixedOwnerAttentionResidualBlockAvailable(residual, x *metal.Array, fixed *metal.FixedKVCache, fixedMask *metal.Array, attn *Gemma4Attention, postAttnNorm *metal.Array, cfg *Gemma4TextConfig) bool {
	if !nativeGemma4FixedOwnerAttentionBlockAvailable(x, fixed, fixedMask, attn, cfg) {
		return false
	}
	if residual == nil || postAttnNorm == nil || !residual.Valid() || !postAttnNorm.Valid() {
		return false
	}
	if residual.NumDims() != x.NumDims() || postAttnNorm.NumDims() != 1 {
		return false
	}
	for i := 0; i < residual.NumDims(); i++ {
		if residual.Dim(i) != x.Dim(i) {
			return false
		}
	}
	return postAttnNorm.Dim(0) == x.Dim(x.NumDims()-1)
}

func nativeGemma4DecodeLayer(x *metal.Array, c metal.Cache, B, L int32, mask *metal.Array, perLayerInput *metal.Array, prev sharedKV, layer *Gemma4DecoderLayer, cfg *Gemma4TextConfig, fixedMask *metal.Array) (*metal.Array, sharedKV, bool, error) {
	if !nativeGemma4DecodeLayerAvailable(x, c, B, L, mask, perLayerInput, prev, layer, cfg) {
		return nil, sharedKV{}, false, nil
	}
	return metal.NativeGemma4DecodeLayer(gemma4LayerRequest(layer, cfg), x, c, B, L, perLayerInput, prev, fixedMask)
}

// gemma4LayerRequest fills the metal fused decode-layer request from a model
// decoder layer + text config. It carries the static per-layer weights as
// *metal.Linear / *metal.SwitchLinear bundles + norms + scalars; the kernel
// resolves the cache state per call.
func gemma4LayerRequest(layer *Gemma4DecoderLayer, cfg *Gemma4TextConfig) metal.Gemma4LayerRequest {
	attn := layer.Attention
	req := metal.Gemma4LayerRequest{
		InputNorm:             layer.InputNormScaled,
		PostAttnNorm:          layer.PostAttnNormScaled,
		PreFFNorm:             layer.PreFFNormScaled,
		PreFFNorm2:            layer.PreFFNorm2Scaled,
		PostFFNorm1:           layer.PostFFNorm1Scaled,
		PostFFNorm2:           layer.PostFFNorm2Scaled,
		PostFFNorm:            layer.PostFFNormScaled,
		PostPerLayerInputNorm: layer.PostPerLayerInputNormScaled,
		LayerScalar:           layer.LayerScalar,
		QProj:                 attn.QProj,
		KProj:                 attn.KProj,
		VProj:                 attn.VProj,
		OProj:                 attn.OProj,
		QNorm:                 attn.QNormScaled,
		KNorm:                 attn.KNormScaled,
		RopeFreqs:             attn.RopeFreqs,
		MLPGate:               layer.MLP.GateProj,
		MLPUp:                 layer.MLP.UpProj,
		MLPDown:               layer.MLP.DownProj,
		PerLayerInputGate:     layer.PerLayerInputGate,
		PerLayerProjection:    layer.PerLayerProjection,
		EnableMoE:             layer.EnableMoE && layer.Router != nil && layer.Experts != nil,
		UseKEqV:               attn.UseKEqV,
		NumAttentionHeads:     cfg.NumAttentionHeads,
		NumKeyValueHeads:      attn.NKVHeads,
		HeadDim:               attn.HeadDim,
		RopeDims:              attn.RopeRotatedDim,
		RopeBase:              attn.RopeBase,
		AttentionScale:        attn.Scale,
	}
	if req.EnableMoE {
		router := layer.Router
		experts := layer.Experts
		req.RouterProj = router.Proj
		req.RouterScale = router.Scale
		req.RouterScaled = router.ScaleScaled
		req.PerExpertScale = router.PerExpertScale
		req.RouterTopK = router.TopK
		req.RouterEps = router.Eps
		req.RouterRootSize = router.RootSize
		req.ExpertGate = experts.GateProj
		req.ExpertUp = experts.UpProj
		req.ExpertGateUp = experts.GateUpProj
		req.ExpertDown = experts.DownProj
	}
	return req
}

func nativeGemma4FixedGreedyToken(h *metal.Array, perLayerInputs []*metal.Array, caches []metal.Cache, model *Gemma4Model, fixedMasks *fixedGemma4AttentionMaskSet, suppressTokens ...int32) (*metal.Array, bool, error) {
	return nativeGemma4FixedGreedyTokenWithArray(h, perLayerInputs, caches, model, fixedMasks, nil, suppressTokens...)
}

func nativeGemma4FixedGreedyTokenWithArray(h *metal.Array, perLayerInputs []*metal.Array, caches []metal.Cache, model *Gemma4Model, fixedMasks *fixedGemma4AttentionMaskSet, suppress *metal.Array, suppressTokens ...int32) (*metal.Array, bool, error) {
	if reason := nativeGemma4FixedGreedyTokenUnavailableReason(h, perLayerInputs, caches, model, fixedMasks); reason != "" {
		metal.TraceNativeSkip("gemma4.model.greedy_token.skip", reason)
		return nil, false, nil
	}
	layers := make([]metal.Gemma4LayerRequest, len(model.Layers))
	for i, layer := range model.Layers {
		layers[i] = gemma4LayerRequest(layer, model.Cfg)
	}
	return metal.NativeGemma4FixedGreedyToken(metal.Gemma4GreedyRequest{
		Hidden:            h,
		Layers:            layers,
		PreviousKVs:       model.PreviousKVs,
		CacheIndexByLayer: model.CacheIndexByLayer,
		Caches:            caches,
		PerLayerInputs:    perLayerInputs,
		FixedMasks:        fixedMasks,
		FinalNorm:         model.NormScaled,
		Output:            model.Output,
	}, suppress, suppressTokens...)
}

func nativeGemma4FixedGreedyTokenAvailable(h *metal.Array, perLayerInputs []*metal.Array, caches []metal.Cache, model *Gemma4Model, fixedMasks *fixedGemma4AttentionMaskSet) bool {
	return nativeGemma4FixedGreedyTokenUnavailableReason(h, perLayerInputs, caches, model, fixedMasks) == ""
}

func nativeGemma4FixedGreedyTokenUnavailableReason(h *metal.Array, perLayerInputs []*metal.Array, caches []metal.Cache, model *Gemma4Model, fixedMasks *fixedGemma4AttentionMaskSet) string {
	if !metal.NativeGemma4ModelGreedyEnabled() {
		return "model metal.Greedy gate is disabled"
	}
	if h == nil || !h.Valid() || model == nil || model.Cfg == nil || fixedMasks == nil || model.Output == nil || model.NormScaled == nil || !model.NormScaled.Valid() {
		return "model metal.Greedy inputs are invalid"
	}
	if h.NumDims() != 3 || h.Dim(0) <= 0 || h.Dim(1) != 1 || h.Dim(2) != int(model.Cfg.HiddenSize) {
		return "hidden state is not a single-token decode row"
	}
	if !metal.NativeLastTokenGreedyTokenAvailable(h, model.NormScaled, model.Output, model.Cfg.RMSNormEps) {
		return "native last-token metal.Greedy output is unavailable"
	}
	layerCount := len(model.Layers)
	if layerCount == 0 {
		return "model has no layers"
	}
	if perLayerInputs != nil && len(perLayerInputs) < layerCount {
		return core.Sprintf("per-layer input metadata is incomplete: got %d want %d", len(perLayerInputs), layerCount)
	}
	if len(model.PreviousKVs) != layerCount || len(model.CacheIndexByLayer) != layerCount {
		return core.Sprintf(
			"cache layout metadata is incomplete: layers=%d previous_kvs=%d cache_index=%d",
			layerCount,
			len(model.PreviousKVs),
			len(model.CacheIndexByLayer),
		)
	}
	B, L := int32(h.Dim(0)), int32(h.Dim(1))
	for i, layer := range model.Layers {
		var perLayerInput *metal.Array
		if perLayerInputs != nil {
			perLayerInput = perLayerInputs[i]
		}
		if reason := gemma4DecodeLayerCommonUnavailableReason(h, B, L, nil, perLayerInput, layer, model.Cfg); reason != "" {
			return core.Sprintf("layer %02d: %s", i, reason)
		}
		prevIdx := int(model.PreviousKVs[i])
		if prevIdx < 0 || prevIdx >= layerCount || prevIdx > i {
			return core.Sprintf("layer %02d: previous kv index is invalid", i)
		}
		if prevIdx == i {
			cacheIdx := int(model.CacheIndexByLayer[i])
			if cacheIdx < 0 || cacheIdx >= len(caches) {
				return core.Sprintf("layer %02d: cache index is invalid", i)
			}
			fixed, ok := caches[cacheIdx].(*metal.FixedKVCache)
			if !ok || fixed == nil || fixed.maxSize <= 0 || fixed.Offset()+1 > fixed.maxSize {
				return core.Sprintf("layer %02d: fixed cache is unavailable", i)
			}
			continue
		}
		if model.PreviousKVs[prevIdx] != int32(prevIdx) {
			return core.Sprintf("layer %02d: shared kv owner is invalid", i)
		}
	}
	return ""
}

func compiledGemma4DecodeLayer(x *metal.Array, c metal.Cache, B, L int32, mask *metal.Array, perLayerInput *metal.Array, prev sharedKV, layer *Gemma4DecoderLayer, cfg *Gemma4TextConfig, fixedMask *metal.Array) (*metal.Array, sharedKV, bool, error) {
	if !metal.CompiledGemma4LayerEnabled() {
		return nil, sharedKV{}, false, nil
	}
	if !gemma4CompiledDecodeLayerBoundaryAvailable(x, c, B, L, mask, perLayerInput, prev, layer, cfg) {
		return nil, sharedKV{}, false, nil
	}

	offset := 0
	var prevKeys, prevValues *metal.Array
	var pageState metal.PagedKVState
	var fixedState metal.FixedKVState
	ownsKV := !prev.HasState()
	fixedKV := prev.Fixed
	if ownsKV {
		switch cache := c.(type) {
		case *metal.PagedKVCache:
			offset = cache.Offset()
			pageState = cache.PageState()
			if len(pageState.Keys) != 1 || len(pageState.Values) != 1 {
				pageState.Free()
				return nil, sharedKV{}, false, nil
			}
			prevKeys = pageState.Keys[0]
			prevValues = pageState.Values[0]
			defer pageState.Free()
		case *metal.FixedKVCache:
			offset = cache.Offset()
			fixedState = cache.BorrowedFixedState()
			if fixedState.Keys == nil || fixedState.Values == nil {
				return nil, sharedKV{}, false, nil
			}
			prevKeys = fixedState.Keys
			prevValues = fixedState.Values
			fixedKV = true
		default:
			return nil, sharedKV{}, false, nil
		}
	} else {
		offset = prev.Offset
		switch {
		case prev.Keys != nil && prev.Values != nil:
			prevKeys, prevValues = prev.Keys, prev.Values
		case prev.HasPages() && len(prev.Pages.Keys) == 1 && len(prev.Pages.Values) == 1:
			prevKeys, prevValues = prev.Pages.Keys[0], prev.Pages.Values[0]
		default:
			return nil, sharedKV{}, false, nil
		}
	}
	if prevKeys == nil || prevValues == nil || !prevKeys.Valid() || !prevValues.Valid() {
		return nil, sharedKV{}, false, nil
	}

	compiled := layer.compiledNativeSharedDecode
	failed := &layer.compiledNativeSharedFailed
	slot := &layer.compiledNativeSharedDecode
	useFixedMask := fixedKV && fixedMask != nil && fixedMask.Valid()
	if fixedKV {
		compiled = layer.compiledNativeFixedSharedDecode
		failed = &layer.compiledNativeFixedSharedFailed
		slot = &layer.compiledNativeFixedSharedDecode
		if useFixedMask {
			compiled = layer.compiledNativeFixedMaskedSharedDecode
			failed = &layer.compiledNativeFixedMaskedSharedFailed
			slot = &layer.compiledNativeFixedMaskedSharedDecode
		}
	}
	if *failed {
		return nil, sharedKV{}, false, nil
	}
	if ownsKV {
		if fixedKV {
			compiled = layer.compiledNativeFixedOwnerDecode
			failed = &layer.compiledNativeFixedOwnerFailed
			slot = &layer.compiledNativeFixedOwnerDecode
			if useFixedMask {
				compiled = layer.compiledNativeFixedMaskedOwnerDecode
				failed = &layer.compiledNativeFixedMaskedOwnerFailed
				slot = &layer.compiledNativeFixedMaskedOwnerDecode
			}
		} else {
			compiled = layer.compiledNativeOwnerDecode
			failed = &layer.compiledNativeOwnerFailed
			slot = &layer.compiledNativeOwnerDecode
		}
		if *failed {
			return nil, sharedKV{}, false, nil
		}
	}
	if compiled == nil || !compiled.Valid() {
		compiled = compileGemma4DecodeLayer(layer, cfg, ownsKV, fixedKV, useFixedMask)
		*slot = compiled
	}

	offsetArray := metal.FromValue(offset)
	defer metal.Free(offsetArray)
	inputs := []*metal.Array{x, prevKeys, prevValues, perLayerInput, offsetArray}
	if useFixedMask {
		inputs = append(inputs, fixedMask)
	}
	outs, callErr := metal.CallCompiledGemma4DecodeLayer(compiled, inputs...)
	if callErr != nil {
		*failed = true
		if *slot != nil {
			(*slot).Free()
			*slot = nil
		}
		return nil, sharedKV{}, true, callErr
	}
	if err := metal.ValidateGemma4LayerOutputs("mlx.compiledGemma4DecodeLayer", outs, ownsKV); err != nil {
		*failed = true
		if *slot != nil {
			(*slot).Free()
			*slot = nil
		}
		metal.Free(outs...)
		return nil, sharedKV{}, true, err
	}
	if err := metal.ValidateGemma4LayerOutputShapes("mlx.compiledGemma4DecodeLayer", x, outs[0], metal.OutputAt(outs, 1), metal.OutputAt(outs, 2), prevKeys, prevValues, ownsKV, fixedKV); err != nil {
		*failed = true
		if *slot != nil {
			(*slot).Free()
			*slot = nil
		}
		metal.Free(outs...)
		return nil, sharedKV{}, true, err
	}
	if ownsKV {
		if fixedKV {
			fixed, _ := c.(*metal.FixedKVCache)
			state := fixed.ReplaceFixedFromNativeBorrowed(outs[1], outs[2], int(L))
			return outs[0], sharedKV{Keys: state.Keys, Values: state.Values, Offset: offset, Fixed: true, Borrowed: true}, true, nil
		}
		paged, _ := c.(*metal.PagedKVCache)
		pages := paged.ReplaceSinglePageFromNative(outs[1], outs[2], int(L))
		return outs[0], sharedKV{Pages: pages, Offset: offset}, true, nil
	}
	return outs[0], prev, true, nil
}

func compileGemma4DecodeLayer(layer *Gemma4DecoderLayer, cfg *Gemma4TextConfig, ownsKV, fixedKV, fixedMask bool) *metal.CompiledFunc {
	return metal.CompileShapeless(func(inputs []*metal.Array) []*metal.Array {
		if len(inputs) < 5 {
			return nil
		}
		var mask *metal.Array
		if fixedMask {
			if len(inputs) < 6 {
				return nil
			}
			mask = inputs[5]
		}
		out, keys, values := gemma4DecodeLayerGraph(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], mask, layer, cfg, ownsKV, fixedKV)
		if ownsKV {
			return []*metal.Array{out, keys, values}
		}
		return []*metal.Array{out}
	}, true)
}

func gemma4DecodeLayerGraph(x, prevKeys, prevValues, perLayerInput, offset, fixedMask *metal.Array, layer *Gemma4DecoderLayer, cfg *Gemma4TextConfig, ownsKV, fixedKV bool) (*metal.Array, *metal.Array, *metal.Array) {
	residual := x
	normed := metal.RMSNorm(x, layer.InputNormScaled, cfg.RMSNormEps)
	attnOut, keys, values := gemma4AttentionGraph(normed, prevKeys, prevValues, offset, fixedMask, layer.Attention, cfg, ownsKV, fixedKV)
	metal.Free(normed)
	attnNormed := metal.RMSNorm(attnOut, layer.PostAttnNormScaled, cfg.RMSNormEps)
	metal.Free(attnOut)
	h := metal.Add(residual, attnNormed)
	metal.Free(attnNormed)

	ffResidual := gemma4DecodeFFNGraph(h, layer, cfg)

	hNext := metal.Add(h, ffResidual)
	metal.Free(h, ffResidual)

	gate := layer.PerLayerInputGate.Forward(hNext)
	multiplied := metal.GeluGateMul(gate, perLayerInput)
	metal.Free(gate)
	projected := layer.PerLayerProjection.Forward(multiplied)
	metal.Free(multiplied)
	projectedNormed := metal.RMSNorm(projected, layer.PostPerLayerInputNormScaled, cfg.RMSNormEps)
	metal.Free(projected)
	gated := metal.Add(hNext, projectedNormed)
	metal.Free(hNext, projectedNormed)
	hNext = gated

	scaled := metal.Mul(hNext, layer.LayerScalar)
	metal.Free(hNext)
	return scaled, keys, values
}

func gemma4DecodeFFNGraph(h *metal.Array, layer *Gemma4DecoderLayer, cfg *Gemma4TextConfig) *metal.Array {
	if layer.EnableMoE && layer.Router != nil && layer.Experts != nil {
		h1In := metal.RMSNorm(h, layer.PreFFNormScaled, cfg.RMSNormEps)
		h1 := metal.Gemma4MLPGraph(h1In, layer.MLP)
		metal.Free(h1In)
		h1Normed := metal.RMSNorm(h1, layer.PostFFNorm1Scaled, cfg.RMSNormEps)
		metal.Free(h1)

		h2In := metal.RMSNorm(h, layer.PreFFNorm2Scaled, cfg.RMSNormEps)
		topKIndices, topKWeights := layer.Router.forward(h)
		h2 := layer.Experts.forward(h2In, topKIndices, topKWeights, "")
		metal.Free(h2In, topKIndices, topKWeights)
		h2Normed := metal.RMSNorm(h2, layer.PostFFNorm2Scaled, cfg.RMSNormEps)
		metal.Free(h2)

		combined := metal.Add(h1Normed, h2Normed)
		metal.Free(h1Normed, h2Normed)
		ffResidual := metal.RMSNorm(combined, layer.PostFFNormScaled, cfg.RMSNormEps)
		metal.Free(combined)
		return ffResidual
	}

	ffIn := metal.RMSNorm(h, layer.PreFFNormScaled, cfg.RMSNormEps)
	ff := metal.Gemma4MLPGraph(ffIn, layer.MLP)
	metal.Free(ffIn)
	ffResidual := metal.RMSNorm(ff, layer.PostFFNormScaled, cfg.RMSNormEps)
	metal.Free(ff)
	return ffResidual
}

func gemma4AttentionGraph(x, prevKeys, prevValues, offset, fixedMask *metal.Array, attn *Gemma4Attention, cfg *Gemma4TextConfig, ownsKV, fixedKV bool) (*metal.Array, *metal.Array, *metal.Array) {
	B, L := int32(x.Dim(0)), int32(x.Dim(1))
	qProj := attn.QProj.Forward(x)
	qReshaped := metal.Reshape(qProj, B, L, cfg.NumAttentionHeads, attn.HeadDim)
	metal.Free(qProj)
	q := metal.Transpose(qReshaped, 0, 2, 1, 3)
	metal.Free(qReshaped)
	oldQ := q
	q = metal.RMSNorm(q, attn.QNormScaled, cfg.RMSNormEps)
	metal.Free(oldQ)

	var keys, values *metal.Array
	var out *metal.Array
	qHasRoPE := false
	if ownsKV {
		kProj := attn.KProj.Forward(x)
		kReshaped := metal.Reshape(kProj, B, L, attn.NKVHeads, attn.HeadDim)
		metal.Free(kProj)
		k := metal.Transpose(kReshaped, 0, 2, 1, 3)
		metal.Free(kReshaped)

		var v *metal.Array
		if attn.UseKEqV {
			v = k.Clone()
		} else {
			vProj := attn.VProj.Forward(x)
			vReshaped := metal.Reshape(vProj, B, L, attn.NKVHeads, attn.HeadDim)
			metal.Free(vProj)
			v = metal.Transpose(vReshaped, 0, 2, 1, 3)
			metal.Free(vReshaped)
		}

		oldK := k
		k = metal.RMSNorm(k, attn.KNormScaled, cfg.RMSNormEps)
		metal.Free(oldK)
		k = gemma4ApplyRoPEDynamic(attn, k, offset)

		vNormed := metal.RMSNormNoScale(v, cfg.RMSNormEps)
		metal.Free(v)
		v = vNormed

		if fixedKV {
			q = gemma4ApplyRoPEDynamic(attn, q, offset)
			qHasRoPE = true
			if nativeOut, nativeKeys, nativeValues, ok, err := metal.NativeFixedSingleTokenAttention(q, prevKeys, prevValues, k, v, offset, fixedMask, attn.Scale); ok {
				out = nativeOut
				keys = nativeKeys
				values = nativeValues
			} else {
				if err != nil {
					core.Error("mlx: native fixed single-token attention failed; falling back to Go graph", "error", err)
				}
				keys = metal.SingleTokenCacheUpdate(prevKeys, k, offset)
				values = metal.SingleTokenCacheUpdate(prevValues, v, offset)
			}
			metal.Free(k, v)
		} else {
			keys = metal.Concatenate2(prevKeys, k, 2)
			values = metal.Concatenate2(prevValues, v, 2)
			metal.Free(k, v)
		}
	} else {
		keys = prevKeys
		values = prevValues
	}

	if !qHasRoPE {
		q = gemma4ApplyRoPEDynamic(attn, q, offset)
	}
	if out == nil {
		if fixedKV {
			mask := fixedMask
			if mask == nil || !mask.Valid() {
				mask = metal.SingleTokenCausalMask(int(keys.Dim(2)), offset)
				defer metal.Free(mask)
			}
			out = metal.ScaledDotProductAttentionWithMask(q, keys, values, mask, attn.Scale)
		} else {
			out = metal.ScaledDotProductAttention(q, keys, values, attn.Scale, false)
		}
	}
	metal.Free(q)

	transposed := metal.Transpose(out, 0, 2, 1, 3)
	metal.Free(out)
	reshaped := metal.Reshape(transposed, B, L, cfg.NumAttentionHeads*attn.HeadDim)
	metal.Free(transposed)
	result := attn.OProj.Forward(reshaped)
	metal.Free(reshaped)
	if !ownsKV {
		return result, nil, nil
	}
	return result, keys, values
}

func gemma4ApplyRoPEDynamic(attn *Gemma4Attention, x, offset *metal.Array) *metal.Array {
	old := x
	if attn.RopeFreqs != nil {
		x = metal.RoPEWithOffsetArray(x, int(attn.HeadDim), false, 0, 1.0, offset, attn.RopeFreqs)
	} else {
		x = metal.RoPEWithOffsetArray(x, int(attn.RopeRotatedDim), false, attn.RopeBase, 1.0, offset, nil)
	}
	metal.Free(old)
	return x
}

func nativeGemma4DecodeLayerAvailable(x *metal.Array, c metal.Cache, B, L int32, mask *metal.Array, perLayerInput *metal.Array, prev sharedKV, layer *Gemma4DecoderLayer, cfg *Gemma4TextConfig) bool {
	if !metal.NativeGemma4LayerEnabled() {
		return false
	}
	if reason := gemma4DecodeLayerBoundaryUnavailableReason(x, c, B, L, mask, perLayerInput, prev, layer, cfg); reason != "" {
		metal.TraceNativeSkip(nativeGemma4LayerSkipTraceName(layer), reason)
		return false
	}
	if reason := gemma4PerLayerDecodeLayerUnavailableReason(layer, cfg); reason != "" {
		metal.TraceNativeSkip(nativeGemma4LayerSkipTraceName(layer), reason)
		return false
	}
	return true
}

func gemma4DecodeLayerBoundaryAvailable(x *metal.Array, c metal.Cache, B, L int32, mask *metal.Array, perLayerInput *metal.Array, prev sharedKV, layer *Gemma4DecoderLayer, cfg *Gemma4TextConfig) bool {
	return gemma4DecodeLayerBoundaryUnavailableReason(x, c, B, L, mask, perLayerInput, prev, layer, cfg) == ""
}

func gemma4DecodeLayerBoundaryUnavailableReason(x *metal.Array, c metal.Cache, B, L int32, mask *metal.Array, perLayerInput *metal.Array, prev sharedKV, layer *Gemma4DecoderLayer, cfg *Gemma4TextConfig) string {
	if reason := gemma4DecodeLayerCommonUnavailableReason(x, B, L, mask, perLayerInput, layer, cfg); reason != "" {
		return reason
	}
	if gemma4PagedDecodeLayerBoundaryAvailable(c, L, prev) {
		return ""
	}
	if prev.HasState() {
		if prev.Fixed && nativeGemma4SharedKVAvailable(prev) {
			return ""
		}
		return "shared-kv state is not native-compatible"
	}
	fixed, ok := c.(*metal.FixedKVCache)
	if !ok {
		return "cache is not fixed and not a native-compatible paged cache"
	}
	if fixed.maxSize <= 0 {
		return "fixed cache has no capacity"
	}
	if fixed.Offset()+int(L) > fixed.maxSize {
		return "fixed cache has insufficient remaining capacity"
	}
	return ""
}

func gemma4DecodeLayerCommonAvailable(x *metal.Array, B, L int32, mask *metal.Array, perLayerInput *metal.Array, layer *Gemma4DecoderLayer, cfg *Gemma4TextConfig) bool {
	return gemma4DecodeLayerCommonUnavailableReason(x, B, L, mask, perLayerInput, layer, cfg) == ""
}

func gemma4DecodeLayerCommonUnavailableReason(x *metal.Array, B, L int32, mask *metal.Array, perLayerInput *metal.Array, layer *Gemma4DecoderLayer, cfg *Gemma4TextConfig) string {
	if x == nil || !x.Valid() {
		return "input is invalid"
	}
	if cfg == nil {
		return "config is nil"
	}
	if layer == nil {
		return "layer is nil"
	}
	if layer.Attention == nil {
		return "attention is nil"
	}
	if layer.MLP == nil {
		return "mlp is nil"
	}
	if layer.EnableMoE && layer.Router != nil && layer.Experts != nil && !metal.NativeGemma4MoELayerEnabled() {
		return "moe native layer is disabled"
	}
	if B <= 0 || L != 1 {
		return "not a single-token decode step"
	}
	if mask != nil {
		return "non-fixed mask is present"
	}
	if cfg.RMSNormEps != 1e-6 {
		return "unsupported rms norm epsilon"
	}
	if cfg.NumAttentionHeads <= 0 || layer.Attention.NKVHeads <= 0 {
		return "attention head counts are invalid"
	}
	if !nativeGemma4NormsAvailable(layer) {
		return "layer norm weights are invalid"
	}
	if reason := nativeGemma4LayerAttentionUnavailableReason(layer.Attention); reason != "" {
		return reason
	}
	if reason := metal.NativeGemma4LayerMLPUnavailableReason(layer.MLP); reason != "" {
		return reason
	}
	if layer.EnableMoE {
		if reason := gemma4DecodeLayerMoEUnavailableReason(layer); reason != "" {
			return reason
		}
	}
	if perLayerInput != nil && perLayerInput.Valid() {
		if layer.PerLayerInputGate == nil || layer.PerLayerProjection == nil {
			return "per-layer input projection is missing"
		}
		if layer.PostPerLayerInputNormScaled == nil || !layer.PostPerLayerInputNormScaled.Valid() {
			return "post per-layer input norm is invalid"
		}
		if reason := metal.NativeGemma4LayerLinearUnavailableReason(layer.PerLayerInputGate, "per-layer gate"); reason != "" {
			return reason
		}
		if reason := metal.NativeGemma4LayerLinearUnavailableReason(layer.PerLayerProjection, "per-layer projection"); reason != "" {
			return reason
		}
	}
	if layer.LayerScalar == nil || !layer.LayerScalar.Valid() {
		return "layer scalar is invalid"
	}
	return ""
}

func gemma4PerLayerDecodeLayerUnavailableReason(layer *Gemma4DecoderLayer, cfg *Gemma4TextConfig) string {
	if layer == nil || layer.Attention == nil || cfg == nil {
		return ""
	}
	if layer.LayerType != "full_attention" {
		return ""
	}
	if cfg.HeadDim <= 0 || cfg.GlobalHeadDim <= 0 || cfg.GlobalHeadDim == cfg.HeadDim {
		return ""
	}
	if layer.Attention.HeadDim == cfg.GlobalHeadDim {
		return "full-attention global head dim requires model-level native boundary"
	}
	return ""
}

func nativeGemma4LayerSkipTraceName(layer *Gemma4DecoderLayer) string {
	if layer == nil {
		return "gemma4.layer.unknown.native_layer.skip"
	}
	return core.Sprintf("gemma4.layer.%02d.native_layer.skip", layer.LayerIdx)
}

func gemma4CompiledDecodeLayerBoundaryAvailable(x *metal.Array, c metal.Cache, B, L int32, mask *metal.Array, perLayerInput *metal.Array, prev sharedKV, layer *Gemma4DecoderLayer, cfg *Gemma4TextConfig) bool {
	if !gemma4DecodeLayerCommonAvailable(x, B, L, mask, perLayerInput, layer, cfg) {
		return false
	}
	if gemma4PerLayerDecodeLayerUnavailableReason(layer, cfg) != "" {
		return false
	}
	if gemma4PagedDecodeLayerBoundaryAvailable(c, L, prev) {
		return true
	}
	if prev.HasState() {
		return prev.Fixed && nativeGemma4SharedKVAvailable(prev)
	}
	fixed, ok := c.(*metal.FixedKVCache)
	return ok && fixed.maxSize > 0 && fixed.Offset()+int(L) <= fixed.maxSize
}

func gemma4DecodeLayerMoEAvailable(layer *Gemma4DecoderLayer) bool {
	return gemma4DecodeLayerMoEUnavailableReason(layer) == ""
}

func gemma4DecodeLayerMoEUnavailableReason(layer *Gemma4DecoderLayer) string {
	if layer == nil || layer.Router == nil || layer.Experts == nil {
		return "moe router or experts are missing"
	}
	if layer.PreFFNorm2Scaled == nil || !layer.PreFFNorm2Scaled.Valid() {
		return "moe pre-ffn2 norm is invalid"
	}
	if layer.PostFFNorm1Scaled == nil || !layer.PostFFNorm1Scaled.Valid() {
		return "moe post-ffn1 norm is invalid"
	}
	if layer.PostFFNorm2Scaled == nil || !layer.PostFFNorm2Scaled.Valid() {
		return "moe post-ffn2 norm is invalid"
	}
	router := layer.Router
	if reason := metal.NativeGemma4LayerLinearUnavailableReason(router.Proj, "router"); reason != "" {
		return reason
	}
	if (router.ScaleScaled == nil || !router.ScaleScaled.Valid()) && (router.Scale == nil || !router.Scale.Valid()) {
		return "router scale is invalid"
	}
	experts := layer.Experts
	if reason := metal.Gemma4DecodeSwitchLinearUnavailableReason(experts.DownProj, "expert down"); reason != "" {
		return reason
	}
	if metal.Gemma4DecodeSwitchLinearAvailable(experts.GateUpProj) {
		return ""
	}
	if reason := metal.Gemma4DecodeSwitchLinearUnavailableReason(experts.GateProj, "expert gate"); reason != "" {
		return reason
	}
	if reason := metal.Gemma4DecodeSwitchLinearUnavailableReason(experts.UpProj, "expert up"); reason != "" {
		return reason
	}
	return ""
}

func gemma4PagedDecodeLayerBoundaryAvailable(c metal.Cache, L int32, prev sharedKV) bool {
	if prev.HasState() {
		return !prev.Fixed && nativeGemma4SharedKVAvailable(prev)
	}
	paged, ok := c.(*metal.PagedKVCache)
	if !ok {
		return false
	}
	if paged.maxSize > 0 && paged.Len()+int(L) > paged.maxSize {
		return false
	}
	if len(paged.kPages) == 1 && metal.PagedArrayLen(paged.kPages[0]) >= paged.pageSize {
		return false
	}
	return len(paged.kPages) <= 1 && len(paged.vPages) <= 1
}

func nativeGemma4NormsAvailable(layer *Gemma4DecoderLayer) bool {
	norms := []*metal.Array{
		layer.InputNormScaled,
		layer.PostAttnNormScaled,
		layer.PreFFNormScaled,
		layer.PostFFNormScaled,
	}
	for _, norm := range norms {
		if norm == nil || !norm.Valid() {
			return false
		}
	}
	return true
}

func nativeGemma4LayerAttentionAvailable(attn *Gemma4Attention) bool {
	return nativeGemma4LayerAttentionUnavailableReason(attn) == ""
}

func nativeGemma4LayerAttentionUnavailableReason(attn *Gemma4Attention) string {
	if attn == nil || attn.HeadDim <= 0 || attn.RopeRotatedDim <= 0 || attn.NKVHeads <= 0 {
		return "attention metadata is invalid"
	}
	if reason := metal.NativeGemma4LayerLinearUnavailableReason(attn.QProj, "attention q"); reason != "" {
		return reason
	}
	if reason := metal.NativeGemma4LayerLinearUnavailableReason(attn.KProj, "attention k"); reason != "" {
		return reason
	}
	if !attn.UseKEqV {
		if reason := metal.NativeGemma4LayerLinearUnavailableReason(attn.VProj, "attention v"); reason != "" {
			return reason
		}
	}
	if reason := metal.NativeGemma4LayerLinearUnavailableReason(attn.OProj, "attention o"); reason != "" {
		return reason
	}
	if attn.QNormScaled == nil || !attn.QNormScaled.Valid() {
		return "attention q norm is invalid"
	}
	if attn.KNormScaled == nil || !attn.KNormScaled.Valid() {
		return "attention k norm is invalid"
	}
	return ""
}

func nativeGemma4AttentionAvailable(attn *Gemma4Attention) bool {
	if attn == nil || attn.HeadDim <= 0 || attn.RopeRotatedDim <= 0 || attn.NKVHeads <= 0 {
		return false
	}
	return metal.NativeMLPLinearAvailable(attn.QProj) &&
		metal.NativeMLPLinearAvailable(attn.KProj) &&
		metal.NativeMLPLinearAvailable(attn.VProj) &&
		metal.NativeMLPLinearAvailable(attn.OProj) &&
		attn.QNormScaled != nil && attn.QNormScaled.Valid() &&
		attn.KNormScaled != nil && attn.KNormScaled.Valid()
}

func nativeGemma4SharedKVAvailable(prev sharedKV) bool {
	switch {
	case prev.Keys != nil && prev.Keys.Valid() && prev.Values != nil && prev.Values.Valid():
		return true
	case prev.HasPages() && len(prev.Pages.Keys) == 1 && len(prev.Pages.Values) == 1:
		return prev.Pages.Keys[0] != nil && prev.Pages.Keys[0].Valid() &&
			prev.Pages.Values[0] != nil && prev.Pages.Values[0].Valid()
	default:
		return false
	}
}
