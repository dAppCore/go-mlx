// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/metal"
)

// Native/compiled decode capability gates — the *Available / *UnavailableReason checks.

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
	if fixed.MaxSize() <= 0 {
		return "fixed cache has no capacity"
	}
	if fixed.Offset()+int(L) > fixed.MaxSize() {
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
	if layer.FFNMemory != nil {
		return "ffn memory augmenter requires graph layer path"
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
	if layer != nil && layer.EnableMoE {
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
	return ok && fixed.MaxSize() > 0 && fixed.Offset()+int(L) <= fixed.MaxSize()
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
	if paged.MaxSize() > 0 && paged.Len()+int(L) > paged.MaxSize() {
		return false
	}
	if len(paged.KPages()) == 1 && metal.PagedArrayLen(paged.KPages()[0]) >= paged.PageSize() {
		return false
	}
	return len(paged.KPages()) <= 1 && len(paged.VPages()) <= 1
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
