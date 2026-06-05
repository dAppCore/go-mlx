// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import "dappco.re/go/mlx/pkg/metal"

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
		!metal.FixedWideSDPAAttentionEnabled() &&
		!metal.FixedWideMatmulAttentionEnabled() {
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
