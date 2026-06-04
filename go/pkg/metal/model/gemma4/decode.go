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
	if x.NumDims() != 3 || x.Dim(0) <= 0 || x.Dim(1) != 1 || fixed.MaxSize() <= 0 || fixed.Offset()+1 > fixed.MaxSize() {
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
			fixedMask.Dim(3) != fixed.MaxSize() {
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
			if !ok || fixed == nil || fixed.MaxSize() <= 0 || fixed.Offset()+1 > fixed.MaxSize() {
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
