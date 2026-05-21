// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

/*
#include <stdlib.h>
#include "decode_bridge.h"

int go_mlx_compiled_greedy_decode_token(mlx_array* res, const mlx_array logits, const mlx_stream stream);
int go_mlx_compiled_dense_last_logits_softcap30(
	mlx_array* res,
	const mlx_array hidden,
	const mlx_array norm_weight,
	const mlx_array output_weight,
	const mlx_stream stream);
int go_mlx_compiled_q4_g64_last_logits_softcap30(
	mlx_array* res,
	const mlx_array hidden,
	const mlx_array norm_weight,
	const mlx_array output_weight,
	const mlx_array output_scales,
	const mlx_array output_biases,
	const mlx_stream stream);
int go_mlx_compiled_dense_last_token(
	mlx_array* res,
	const mlx_array hidden,
	const mlx_array norm_weight,
	const mlx_array output_weight,
	const mlx_stream stream);
int go_mlx_compiled_dense_last_token_suppressed(
	mlx_array* res,
	const mlx_array hidden,
	const mlx_array norm_weight,
	const mlx_array output_weight,
	const mlx_array suppress_token_ids,
	const mlx_stream stream);
int go_mlx_compiled_q4_g64_last_token(
	mlx_array* res,
	const mlx_array hidden,
	const mlx_array norm_weight,
	const mlx_array output_weight,
	const mlx_array output_scales,
	const mlx_array output_biases,
	const mlx_stream stream);
int go_mlx_compiled_q4_g64_last_token_suppressed(
	mlx_array* res,
	const mlx_array hidden,
	const mlx_array norm_weight,
	const mlx_array output_weight,
	const mlx_array output_scales,
	const mlx_array output_biases,
	const mlx_array suppress_token_ids,
	const mlx_stream stream);
int go_mlx_compiled_dense_mlp_gelu(
	mlx_array* res,
	const mlx_array input,
	const mlx_array gate_weight,
	const mlx_array up_weight,
	const mlx_array down_weight,
	const mlx_stream stream);
int go_mlx_compiled_q4_g64_mlp_gelu(
	mlx_array* res,
	const mlx_array input,
	const mlx_array gate_weight,
	const mlx_array gate_scales,
	const mlx_array gate_biases,
	const mlx_array up_weight,
	const mlx_array up_scales,
	const mlx_array up_biases,
	const mlx_array down_weight,
	const mlx_array down_scales,
	const mlx_array down_biases,
	const mlx_stream stream);
int go_mlx_gemma4_fixed_owner_attention(
	mlx_array* out,
	mlx_array* new_keys,
	mlx_array* new_values,
	const go_mlx_gemma4_fixed_attention_args* args,
	const mlx_stream stream);
int go_mlx_gemma4_fixed_owner_attention_residual(
	mlx_array* out,
	mlx_array* new_keys,
	mlx_array* new_values,
	const go_mlx_gemma4_fixed_attention_args* args,
	const mlx_stream stream);
int go_mlx_compiled_rms_norm_residual(
	mlx_array* out,
	const mlx_array residual,
	const mlx_array input,
	const mlx_array norm_weight,
	const mlx_stream stream);
int go_mlx_compiled_fixed_single_token_attention(
	mlx_array* out,
	mlx_array* new_keys,
	mlx_array* new_values,
	const mlx_array query,
	const mlx_array key_cache,
	const mlx_array value_cache,
	const mlx_array key,
	const mlx_array value,
	const mlx_array offset,
	const mlx_array scale,
	const mlx_array mask,
	const int has_mask,
	const mlx_stream stream);
int go_mlx_compiled_fixed_sliding_single_token_attention(
	mlx_array* out,
	mlx_array* new_keys,
	mlx_array* new_values,
	const mlx_array query,
	const mlx_array key_cache,
	const mlx_array value_cache,
	const mlx_array key,
	const mlx_array value,
	const mlx_array scale,
	const mlx_array shift_indices,
	const mlx_array last_index,
	const mlx_stream stream);
*/
import "C"

import (
	"unsafe"

	"dappco.re/go"
)

var (
	enableNativeGemma4Layer                       = core.Env("GO_MLX_ENABLE_NATIVE_GEMMA4_LAYER") == "1"
	enableNativeGemma4MoELayer                    = core.Env("GO_MLX_ENABLE_NATIVE_GEMMA4_MOE_LAYER") == "1"
	enableNativeGemma4ModelGreedy                 = core.Env("GO_MLX_ENABLE_NATIVE_GEMMA4_MODEL_GREEDY") == "1"
	enableCompiledGemma4Layer                     = core.Env("GO_MLX_ENABLE_COMPILED_GEMMA4_LAYER") == "1"
	enableFixedGemma4Cache                        = core.Env("GO_MLX_ENABLE_FIXED_GEMMA4_CACHE") == "1"
	enableFixedGemma4SlidingCacheBound            = core.Env("GO_MLX_ENABLE_FIXED_GEMMA4_SLIDING_CACHE_BOUND") == "1"
	enableFixedGemma4SharedMask                   = core.Env("GO_MLX_ENABLE_FIXED_GEMMA4_SHARED_MASK") == "1"
	enableDirectGreedyToken                       = core.Env("GO_MLX_ENABLE_DIRECT_GREEDY_TOKEN") == "1"
	enableNativeGemma4FixedOwnerAttention         = core.Env("GO_MLX_ENABLE_NATIVE_GEMMA4_FIXED_OWNER_ATTENTION") == "1"
	enableNativeGemma4FixedOwnerAttentionResidual = core.Env("GO_MLX_ENABLE_NATIVE_GEMMA4_FIXED_OWNER_ATTENTION_RESIDUAL") == "1"
	enableNativeGemma4AttentionOMatVec            = core.Env("GO_MLX_ENABLE_NATIVE_GEMMA4_ATTENTION_O_MATVEC") == "1"
	enableNativeGemma4ResidualNorm                = core.Env("GO_MLX_ENABLE_NATIVE_GEMMA4_RESIDUAL_NORM") == "1"
	enableNativeFixedSlidingAttention             = core.Env("GO_MLX_ENABLE_NATIVE_FIXED_SLIDING_ATTENTION") == "1"
)

func nativeGemma4LayerEnabled() bool {
	return enableNativeGemma4Layer || nativeGemma4LayerRuntimeEnabled()
}

func nativeGemma4MoELayerEnabled() bool {
	return enableNativeGemma4MoELayer || nativeGemma4MoELayerRuntimeEnabled()
}

func nativeGemma4ModelGreedyEnabled() bool {
	return enableNativeGemma4ModelGreedy || nativeGemma4ModelGreedyRuntimeEnabled()
}

func compiledGemma4LayerEnabled() bool {
	return enableCompiledGemma4Layer || compiledGemma4LayerRuntimeEnabled()
}

func fixedGemma4CacheEnabled() bool {
	return enableFixedGemma4Cache || fixedGemma4CacheRuntimeEnabled()
}

func fixedGemma4SlidingCacheBoundEnabled() bool {
	return enableFixedGemma4SlidingCacheBound || fixedGemma4SlidingCacheBoundRuntimeEnabled()
}

func fixedGemma4SharedMaskEnabled() bool {
	return enableFixedGemma4SharedMask || fixedGemma4SharedMaskRuntimeEnabled()
}

func directGreedyTokenEnabled() bool {
	return enableDirectGreedyToken || directGreedyTokenRuntimeEnabled()
}

func nativeGemma4FixedOwnerAttentionEnabled() bool {
	return enableNativeGemma4FixedOwnerAttention || nativeGemma4FixedOwnerAttentionRuntimeEnabled()
}

func nativeGemma4FixedOwnerAttentionResidualEnabled() bool {
	return enableNativeGemma4FixedOwnerAttentionResidual || nativeGemma4FixedOwnerAttentionResidualRuntimeEnabled()
}

func nativeGemma4AttentionOMatVecEnabled() bool {
	return enableNativeGemma4AttentionOMatVec || nativeGemma4AttentionOMatVecRuntimeEnabled()
}

func nativeGemma4ResidualNormEnabled() bool {
	return enableNativeGemma4ResidualNorm || nativeGemma4ResidualNormRuntimeEnabled()
}

func nativeFixedSlidingAttentionEnabled() bool {
	return enableNativeFixedSlidingAttention
}

func cArray(a *Array) C.mlx_array {
	if a == nil {
		var empty C.mlx_array
		return empty
	}
	return a.ctx
}

func nativeGreedyDecodeToken(logits *Array) (*Array, error) {
	if logits == nil || !logits.Valid() {
		return nil, core.NewError("mlx: logits are empty")
	}
	out := newArray("FAST_GREEDY_DECODE_TOKEN", logits)
	rc := C.go_mlx_compiled_greedy_decode_token(&out.ctx, logits.ctx, DefaultStream().ctx)
	if rc != 0 {
		Free(out)
		if err := lastError(); err != nil {
			return nil, err
		}
		return nil, core.E("mlx.nativeGreedyDecodeToken", core.Sprintf("native wrapper failed (rc=%d)", rc), nil)
	}
	return out, nil
}

func nativeGreedyDecodeAvailable(cfg GenerateConfig, history []int32, logits *Array) bool {
	return cfg.ProbeSink == nil &&
		cfg.Temperature == 0 &&
		cfg.TopP == 0 &&
		cfg.MinP == 0 &&
		cfg.TopK == 0 &&
		len(cfg.SuppressTokens) == 0 &&
		(cfg.RepeatPenalty <= 1 || len(history) == 0) &&
		logitsSingleStep(logits)
}

func logitsSingleStep(logits *Array) bool {
	if logits == nil || !logits.Valid() {
		return false
	}
	ndim := logits.NumDims()
	switch {
	case ndim == 1:
		return true
	case ndim == 2:
		return logits.Dim(0) == 1
	case ndim > 2:
		return logits.Dim(ndim-2) == 1
	default:
		return false
	}
}

func nativeLastTokenOutputLogits(hidden, normWeight *Array, output *Linear, eps, softcap float32) (*Array, bool, error) {
	if !nativeLastTokenOutputAvailable(hidden, normWeight, output, eps, softcap) {
		return nil, false, nil
	}
	out := newArray("FAST_LAST_TOKEN_OUTPUT_LOGITS", hidden, normWeight, output.Weight, output.Scales, output.Biases)
	var rc C.int
	if output.Scales != nil {
		rc = C.go_mlx_compiled_q4_g64_last_logits_softcap30(
			&out.ctx,
			hidden.ctx,
			normWeight.ctx,
			output.Weight.ctx,
			output.Scales.ctx,
			output.Biases.ctx,
			DefaultStream().ctx,
		)
	} else {
		rc = C.go_mlx_compiled_dense_last_logits_softcap30(
			&out.ctx,
			hidden.ctx,
			normWeight.ctx,
			output.Weight.ctx,
			DefaultStream().ctx,
		)
	}
	if rc != 0 {
		Free(out)
		if err := lastError(); err != nil {
			return nil, true, err
		}
		return nil, true, core.E("mlx.nativeLastTokenOutputLogits", core.Sprintf("native wrapper failed (rc=%d)", rc), nil)
	}
	return out, true, nil
}

func nativeLastTokenOutputAvailable(hidden, normWeight *Array, output *Linear, eps, softcap float32) bool {
	if hidden == nil || !hidden.Valid() || normWeight == nil || !normWeight.Valid() {
		return false
	}
	if output == nil || output.LoRA != nil || output.Weight == nil || !output.Weight.Valid() {
		return false
	}
	if eps != 1e-6 || softcap != 30 {
		return false
	}
	if output.Bias != nil && output.Bias.Valid() {
		return false
	}
	if output.Scales == nil {
		return true
	}
	return output.Scales.Valid() &&
		output.Biases != nil &&
		output.Biases.Valid() &&
		output.GroupSize == 64 &&
		output.Bits == 4
}

func nativeLastTokenGreedyToken(hidden, normWeight *Array, output *Linear, eps float32, suppressTokens ...int32) (*Array, bool, error) {
	if !nativeLastTokenGreedyTokenAvailable(hidden, normWeight, output, eps) {
		return nil, false, nil
	}
	out := newArray("FAST_LAST_TOKEN_GREEDY", hidden, normWeight, output.Weight, output.Scales, output.Biases)
	var rc C.int
	suppress := suppressTokenArray(suppressTokens)
	defer Free(suppress)
	if output.Scales != nil {
		if suppress != nil {
			rc = C.go_mlx_compiled_q4_g64_last_token_suppressed(
				&out.ctx,
				hidden.ctx,
				normWeight.ctx,
				output.Weight.ctx,
				output.Scales.ctx,
				output.Biases.ctx,
				suppress.ctx,
				DefaultStream().ctx,
			)
		} else {
			rc = C.go_mlx_compiled_q4_g64_last_token(
				&out.ctx,
				hidden.ctx,
				normWeight.ctx,
				output.Weight.ctx,
				output.Scales.ctx,
				output.Biases.ctx,
				DefaultStream().ctx,
			)
		}
	} else {
		if suppress != nil {
			rc = C.go_mlx_compiled_dense_last_token_suppressed(
				&out.ctx,
				hidden.ctx,
				normWeight.ctx,
				output.Weight.ctx,
				suppress.ctx,
				DefaultStream().ctx,
			)
		} else {
			rc = C.go_mlx_compiled_dense_last_token(
				&out.ctx,
				hidden.ctx,
				normWeight.ctx,
				output.Weight.ctx,
				DefaultStream().ctx,
			)
		}
	}
	if rc != 0 {
		Free(out)
		if err := lastError(); err != nil {
			return nil, true, err
		}
		return nil, true, core.E("mlx.nativeLastTokenGreedyToken", core.Sprintf("native wrapper failed (rc=%d)", rc), nil)
	}
	return out, true, nil
}

func suppressTokenArray(ids []int32) *Array {
	if len(ids) == 0 {
		return nil
	}
	return FromValues(ids, len(ids))
}

func nativeLastTokenGreedyTokenAvailable(hidden, normWeight *Array, output *Linear, eps float32) bool {
	if hidden == nil || !hidden.Valid() || normWeight == nil || !normWeight.Valid() {
		return false
	}
	if output == nil || output.LoRA != nil || output.Weight == nil || !output.Weight.Valid() {
		return false
	}
	if eps != 1e-6 {
		return false
	}
	if output.Bias != nil && output.Bias.Valid() {
		return false
	}
	if output.Scales == nil {
		return true
	}
	return output.Scales.Valid() &&
		output.Biases != nil &&
		output.Biases.Valid() &&
		output.GroupSize == 64 &&
		output.Bits == 4
}

func nativeMLPGELU(input *Array, mlp *MLP) (*Array, bool, error) {
	if !nativeMLPGELUAvailable(input, mlp) {
		return nil, false, nil
	}
	out := newArray("FAST_MLP_GELU", input, mlp.GateProj.Weight, mlp.GateProj.Scales, mlp.GateProj.Biases, mlp.UpProj.Weight, mlp.UpProj.Scales, mlp.UpProj.Biases, mlp.DownProj.Weight, mlp.DownProj.Scales, mlp.DownProj.Biases)
	var rc C.int
	if mlp.GateProj.Scales != nil {
		rc = C.go_mlx_compiled_q4_g64_mlp_gelu(
			&out.ctx,
			input.ctx,
			mlp.GateProj.Weight.ctx,
			mlp.GateProj.Scales.ctx,
			mlp.GateProj.Biases.ctx,
			mlp.UpProj.Weight.ctx,
			mlp.UpProj.Scales.ctx,
			mlp.UpProj.Biases.ctx,
			mlp.DownProj.Weight.ctx,
			mlp.DownProj.Scales.ctx,
			mlp.DownProj.Biases.ctx,
			DefaultStream().ctx,
		)
	} else {
		rc = C.go_mlx_compiled_dense_mlp_gelu(
			&out.ctx,
			input.ctx,
			mlp.GateProj.Weight.ctx,
			mlp.UpProj.Weight.ctx,
			mlp.DownProj.Weight.ctx,
			DefaultStream().ctx,
		)
	}
	if rc != 0 {
		Free(out)
		if err := lastError(); err != nil {
			return nil, true, err
		}
		return nil, true, core.E("mlx.nativeMLPGELU", core.Sprintf("native wrapper failed (rc=%d)", rc), nil)
	}
	return out, true, nil
}

func nativeMLPGELUAvailable(input *Array, mlp *MLP) bool {
	if core.Env("GO_MLX_ENABLE_NATIVE_MLP_GELU") != "1" {
		return false
	}
	if input == nil || !input.Valid() || mlp == nil {
		return false
	}
	if !nativeMLPLinearAvailable(mlp.GateProj) ||
		!nativeMLPLinearAvailable(mlp.UpProj) ||
		!nativeMLPLinearAvailable(mlp.DownProj) {
		return false
	}
	gateQuantized := mlp.GateProj.Scales != nil
	upQuantized := mlp.UpProj.Scales != nil
	downQuantized := mlp.DownProj.Scales != nil
	if gateQuantized != upQuantized || gateQuantized != downQuantized {
		return false
	}
	return true
}

func nativeMLPLinearAvailable(linear *Linear) bool {
	if linear == nil || linear.LoRA != nil || linear.Weight == nil || !linear.Weight.Valid() {
		return false
	}
	if linear.Bias != nil && linear.Bias.Valid() {
		return false
	}
	if linear.Scales == nil {
		return linear.Biases == nil || !linear.Biases.Valid()
	}
	return linear.Scales.Valid() &&
		linear.Biases != nil &&
		linear.Biases.Valid() &&
		linear.GroupSize == 64 &&
		linear.Bits == 4
}

func nativeResidualNormAdd(residual, input, norm *Array, eps float32) (*Array, bool, error) {
	if !nativeResidualNormAddAvailable(residual, input, norm, eps) {
		return nil, false, nil
	}
	out := newArray("FAST_RMS_NORM_RESIDUAL", residual, input, norm)
	rc := C.go_mlx_compiled_rms_norm_residual(&out.ctx, residual.ctx, input.ctx, norm.ctx, DefaultStream().ctx)
	if rc != 0 {
		Free(out)
		if err := lastError(); err != nil {
			return nil, true, err
		}
		return nil, true, core.E("mlx.nativeResidualNormAdd", core.Sprintf("native wrapper failed (rc=%d)", rc), nil)
	}
	if !out.Valid() {
		Free(out)
		return nil, true, core.E("mlx.nativeResidualNormAdd", "native wrapper returned invalid output", nil)
	}
	return out, true, nil
}

func nativeResidualNormAddAvailable(residual, input, norm *Array, eps float32) bool {
	if residual == nil || input == nil || norm == nil || !residual.Valid() || !input.Valid() || !norm.Valid() {
		return false
	}
	if eps != 1e-6 || residual.NumDims() != input.NumDims() || residual.NumDims() == 0 || norm.NumDims() != 1 {
		return false
	}
	if residual.Size() != input.Size() {
		return false
	}
	for i := 0; i < residual.NumDims(); i++ {
		if residual.Dim(i) != input.Dim(i) {
			return false
		}
	}
	return norm.Dim(0) == input.Dim(input.NumDims()-1)
}

func nativeGemma4FixedOwnerAttentionBlock(x *Array, fixed *FixedKVCache, fixedMask *Array, attn *Gemma4Attention, cfg *Gemma4TextConfig) (*Array, sharedKV, bool, error) {
	if !nativeGemma4FixedOwnerAttentionBlockAvailable(x, fixed, fixedMask, attn, cfg) {
		return nil, sharedKV{}, false, nil
	}
	fixed.ensureShape(int32(x.Dim(0)), attn.NKVHeads, attn.HeadDim, attn.HeadDim, x.Dtype(), x.Dtype())
	state := fixed.BorrowedFixedState()
	if state.Keys == nil || state.Values == nil {
		return nil, sharedKV{}, false, nil
	}
	offset := fixed.Offset()
	offsetArray := FromValue(offset)
	scaleArray := FromValue(attn.Scale)
	defer Free(offsetArray, scaleArray)

	out := newArray("FAST_GEMMA4_FIXED_OWNER_ATTENTION", x, state.Keys, state.Values)
	newKeys := newArray("FAST_GEMMA4_FIXED_OWNER_ATTENTION_K", state.Keys)
	newValues := newArray("FAST_GEMMA4_FIXED_OWNER_ATTENTION_V", state.Values)
	args := nativeGemma4FixedOwnerAttentionArgs(x, nil, state.Keys, state.Values, offsetArray, scaleArray, fixedMask, attn, nil, cfg)
	rc := C.go_mlx_gemma4_fixed_owner_attention(&out.ctx, &newKeys.ctx, &newValues.ctx, &args, DefaultStream().ctx)
	if rc != 0 {
		Free(out, newKeys, newValues)
		if err := lastError(); err != nil {
			return nil, sharedKV{}, true, err
		}
		return nil, sharedKV{}, true, core.E("mlx.nativeGemma4FixedOwnerAttentionBlock", core.Sprintf("native wrapper failed (rc=%d)", rc), nil)
	}
	if !out.Valid() || !newKeys.Valid() || !newValues.Valid() {
		Free(out, newKeys, newValues)
		return nil, sharedKV{}, true, core.E("mlx.nativeGemma4FixedOwnerAttentionBlock", "native wrapper returned invalid outputs", nil)
	}
	fixedState := fixed.ReplaceFixedFromNativeBorrowed(newKeys, newValues, 1)
	return out, sharedKV{Keys: fixedState.Keys, Values: fixedState.Values, Offset: offset, Fixed: true}, true, nil
}

func nativeGemma4FixedOwnerAttentionResidualBlock(residual, x *Array, fixed *FixedKVCache, fixedMask *Array, attn *Gemma4Attention, postAttnNorm *Array, cfg *Gemma4TextConfig) (*Array, sharedKV, bool, error) {
	if !nativeGemma4FixedOwnerAttentionResidualBlockAvailable(residual, x, fixed, fixedMask, attn, postAttnNorm, cfg) {
		return nil, sharedKV{}, false, nil
	}
	fixed.ensureShape(int32(x.Dim(0)), attn.NKVHeads, attn.HeadDim, attn.HeadDim, x.Dtype(), x.Dtype())
	state := fixed.BorrowedFixedState()
	if state.Keys == nil || state.Values == nil {
		return nil, sharedKV{}, false, nil
	}
	offset := fixed.Offset()
	offsetArray := FromValue(offset)
	scaleArray := FromValue(attn.Scale)
	defer Free(offsetArray, scaleArray)

	out := newArray("FAST_GEMMA4_FIXED_OWNER_ATTENTION_RESIDUAL", residual, x, state.Keys, state.Values)
	newKeys := newArray("FAST_GEMMA4_FIXED_OWNER_ATTENTION_RESIDUAL_K", state.Keys)
	newValues := newArray("FAST_GEMMA4_FIXED_OWNER_ATTENTION_RESIDUAL_V", state.Values)
	args := nativeGemma4FixedOwnerAttentionArgs(x, residual, state.Keys, state.Values, offsetArray, scaleArray, fixedMask, attn, postAttnNorm, cfg)
	rc := C.go_mlx_gemma4_fixed_owner_attention_residual(&out.ctx, &newKeys.ctx, &newValues.ctx, &args, DefaultStream().ctx)
	if rc != 0 {
		Free(out, newKeys, newValues)
		if err := lastError(); err != nil {
			return nil, sharedKV{}, true, err
		}
		return nil, sharedKV{}, true, core.E("mlx.nativeGemma4FixedOwnerAttentionResidualBlock", core.Sprintf("native wrapper failed (rc=%d)", rc), nil)
	}
	if !out.Valid() || !newKeys.Valid() || !newValues.Valid() {
		Free(out, newKeys, newValues)
		return nil, sharedKV{}, true, core.E("mlx.nativeGemma4FixedOwnerAttentionResidualBlock", "native wrapper returned invalid outputs", nil)
	}
	fixedState := fixed.ReplaceFixedFromNativeBorrowed(newKeys, newValues, 1)
	return out, sharedKV{Keys: fixedState.Keys, Values: fixedState.Values, Offset: offset, Fixed: true}, true, nil
}

func nativeGemma4FixedOwnerAttentionArgs(x, residual, keyCache, valueCache, offset, scale, fixedMask *Array, attn *Gemma4Attention, postAttnNorm *Array, cfg *Gemma4TextConfig) C.go_mlx_gemma4_fixed_attention_args {
	args := C.go_mlx_gemma4_fixed_attention_args{
		x:                   cArray(x),
		residual:            cArray(residual),
		key_cache:           cArray(keyCache),
		value_cache:         cArray(valueCache),
		offset:              cArray(offset),
		scale:               cArray(scale),
		mask:                cArray(fixedMask),
		q_weight:            cArray(attn.QProj.Weight),
		q_scales:            cArray(attn.QProj.Scales),
		q_biases:            cArray(attn.QProj.Biases),
		k_weight:            cArray(attn.KProj.Weight),
		k_scales:            cArray(attn.KProj.Scales),
		k_biases:            cArray(attn.KProj.Biases),
		v_weight:            cArray(attn.VProj.Weight),
		v_scales:            cArray(attn.VProj.Scales),
		v_biases:            cArray(attn.VProj.Biases),
		o_weight:            cArray(attn.OProj.Weight),
		o_scales:            cArray(attn.OProj.Scales),
		o_biases:            cArray(attn.OProj.Biases),
		q_norm:              cArray(attn.QNormScaled),
		k_norm:              cArray(attn.KNormScaled),
		post_attn_norm:      cArray(postAttnNorm),
		rope_freqs:          cArray(attn.RopeFreqs),
		num_attention_heads: C.int(cfg.NumAttentionHeads),
		num_key_value_heads: C.int(attn.NKVHeads),
		head_dim:            C.int(attn.HeadDim),
		rope_dims:           C.int(attn.RopeRotatedDim),
		rope_base:           C.float(attn.RopeBase),
	}
	if fixedMask != nil && fixedMask.Valid() {
		args.has_mask = 1
	}
	if attn.RopeFreqs != nil && attn.RopeFreqs.Valid() {
		args.has_rope_freqs = 1
	}
	return args
}

func nativeGemma4FixedOwnerAttentionBlockAvailable(x *Array, fixed *FixedKVCache, fixedMask *Array, attn *Gemma4Attention, cfg *Gemma4TextConfig) bool {
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

func nativeGemma4FixedOwnerAttentionResidualBlockAvailable(residual, x *Array, fixed *FixedKVCache, fixedMask *Array, attn *Gemma4Attention, postAttnNorm *Array, cfg *Gemma4TextConfig) bool {
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

func nativeFixedSingleTokenAttention(query, keyCache, valueCache, key, value, offset, mask *Array, scale float32) (*Array, *Array, *Array, bool, error) {
	if !nativeFixedSingleTokenAttentionAvailable(query, keyCache, valueCache, key, value, offset, mask) {
		return nil, nil, nil, false, nil
	}
	scaleArray := FromValue(scale)
	defer Free(scaleArray)
	outInputs := []*Array{query, keyCache, valueCache, key, value, offset, scaleArray}
	hasMask := C.int(0)
	if mask != nil && mask.Valid() {
		outInputs = append(outInputs, mask)
		hasMask = 1
	}
	out := newArray("FAST_FIXED_SINGLE_TOKEN_ATTENTION", outInputs...)
	newKeys := newArray("FAST_FIXED_SINGLE_TOKEN_ATTENTION_K", keyCache, key, offset)
	newValues := newArray("FAST_FIXED_SINGLE_TOKEN_ATTENTION_V", valueCache, value, offset)
	rc := C.go_mlx_compiled_fixed_single_token_attention(
		&out.ctx,
		&newKeys.ctx,
		&newValues.ctx,
		query.ctx,
		keyCache.ctx,
		valueCache.ctx,
		key.ctx,
		value.ctx,
		offset.ctx,
		scaleArray.ctx,
		cArray(mask),
		hasMask,
		DefaultStream().ctx,
	)
	if rc != 0 {
		Free(out, newKeys, newValues)
		if err := lastError(); err != nil {
			return nil, nil, nil, true, err
		}
		return nil, nil, nil, true, core.E("mlx.nativeFixedSingleTokenAttention", core.Sprintf("native wrapper failed (rc=%d)", rc), nil)
	}
	return out, newKeys, newValues, true, nil
}

func nativeFixedSingleTokenAttentionAvailable(query, keyCache, valueCache, key, value, offset, mask *Array) bool {
	arrays := []*Array{query, keyCache, valueCache, key, value, offset}
	for _, arr := range arrays {
		if arr == nil || !arr.Valid() {
			return false
		}
	}
	if query.NumDims() != 4 || keyCache.NumDims() != 4 || valueCache.NumDims() != 4 || key.NumDims() != 4 || value.NumDims() != 4 {
		return false
	}
	if query.Dim(2) != 1 || key.Dim(2) != 1 || value.Dim(2) != 1 {
		return false
	}
	if query.Dim(0) != keyCache.Dim(0) || query.Dim(0) != valueCache.Dim(0) ||
		key.Dim(0) != keyCache.Dim(0) || value.Dim(0) != valueCache.Dim(0) {
		return false
	}
	if keyCache.Dim(1) != valueCache.Dim(1) || key.Dim(1) != keyCache.Dim(1) || value.Dim(1) != valueCache.Dim(1) {
		return false
	}
	if query.Dim(1)%keyCache.Dim(1) != 0 {
		return false
	}
	if keyCache.Dim(2) != valueCache.Dim(2) {
		return false
	}
	if mask != nil && mask.Valid() {
		if mask.NumDims() != 4 ||
			mask.Dim(0) != query.Dim(0) ||
			mask.Dim(1) != 1 ||
			mask.Dim(2) != 1 ||
			mask.Dim(3) != keyCache.Dim(2) {
			return false
		}
	}
	// The current bundled MLX metallib does not provide the vector SDPA kernel
	// selected for 512-wide fixed single-token heads. A native matmul fallback
	// exists for diagnostics, but it is slower than the guarded fallback path.
	if keyCache.Dim(3) >= 512 &&
		core.Env("GO_MLX_ENABLE_FIXED_WIDE_SDPA_ATTENTION") != "1" &&
		core.Env("GO_MLX_ENABLE_FIXED_WIDE_MATMUL_ATTENTION") != "1" {
		return false
	}
	return query.Dim(3) == keyCache.Dim(3) &&
		key.Dim(3) == keyCache.Dim(3) &&
		value.Dim(3) == valueCache.Dim(3)
}

func nativeFixedSlidingSingleTokenAttention(query, keyCache, valueCache, key, value, shiftIndices, lastIndex *Array, scale float32) (*Array, *Array, *Array, bool, error) {
	if !nativeFixedSlidingSingleTokenAttentionAvailable(query, keyCache, valueCache, key, value, shiftIndices, lastIndex) {
		return nil, nil, nil, false, nil
	}
	scaleArray := FromValue(scale)
	defer Free(scaleArray)
	out := newArray("FAST_FIXED_SLIDING_ATTENTION_OUT", query, keyCache, valueCache, key, value, scaleArray, shiftIndices, lastIndex)
	newKeys := newArray("FAST_FIXED_SLIDING_ATTENTION_K", keyCache, key)
	newValues := newArray("FAST_FIXED_SLIDING_ATTENTION_V", valueCache, value)
	rc := C.go_mlx_compiled_fixed_sliding_single_token_attention(
		&out.ctx,
		&newKeys.ctx,
		&newValues.ctx,
		query.ctx,
		keyCache.ctx,
		valueCache.ctx,
		key.ctx,
		value.ctx,
		scaleArray.ctx,
		shiftIndices.ctx,
		lastIndex.ctx,
		DefaultStream().ctx,
	)
	if rc != 0 {
		Free(out, newKeys, newValues)
		if err := lastError(); err != nil {
			return nil, nil, nil, true, err
		}
		return nil, nil, nil, true, core.E("mlx.nativeFixedSlidingSingleTokenAttention", core.Sprintf("native wrapper failed (rc=%d)", rc), nil)
	}
	if !out.Valid() || !newKeys.Valid() || !newValues.Valid() {
		Free(out, newKeys, newValues)
		return nil, nil, nil, true, core.E("mlx.nativeFixedSlidingSingleTokenAttention", "native wrapper returned invalid outputs", nil)
	}
	return out, newKeys, newValues, true, nil
}

func nativeFixedSlidingSingleTokenAttentionAvailable(query, keyCache, valueCache, key, value, shiftIndices, lastIndex *Array) bool {
	arrays := []*Array{query, keyCache, valueCache, key, value, shiftIndices, lastIndex}
	for _, arr := range arrays {
		if arr == nil || !arr.Valid() {
			return false
		}
	}
	if query.NumDims() != 4 || keyCache.NumDims() != 4 || valueCache.NumDims() != 4 || key.NumDims() != 4 || value.NumDims() != 4 {
		return false
	}
	if shiftIndices.NumDims() != 1 || shiftIndices.Dim(0) != keyCache.Dim(2) || lastIndex.NumDims() > 0 {
		return false
	}
	if query.Dim(2) != 1 || key.Dim(2) != 1 || value.Dim(2) != 1 || keyCache.Dim(2) <= 0 || valueCache.Dim(2) != keyCache.Dim(2) {
		return false
	}
	if query.Dim(0) != keyCache.Dim(0) || query.Dim(0) != valueCache.Dim(0) ||
		key.Dim(0) != keyCache.Dim(0) || value.Dim(0) != valueCache.Dim(0) {
		return false
	}
	if keyCache.Dim(1) != valueCache.Dim(1) || key.Dim(1) != keyCache.Dim(1) || value.Dim(1) != valueCache.Dim(1) {
		return false
	}
	if query.Dim(1)%keyCache.Dim(1) != 0 {
		return false
	}
	return query.Dim(3) == keyCache.Dim(3) &&
		key.Dim(3) == keyCache.Dim(3) &&
		value.Dim(3) == valueCache.Dim(3)
}

func nativeGemma4DecodeLayer(x *Array, c Cache, B, L int32, mask *Array, perLayerInput *Array, prev sharedKV, layer *Gemma4DecoderLayer, cfg *Gemma4TextConfig, fixedMask *Array) (*Array, sharedKV, bool, error) {
	if !nativeGemma4DecodeLayerAvailable(x, c, B, L, mask, perLayerInput, prev, layer, cfg) {
		return nil, sharedKV{}, false, nil
	}

	offset := 0
	var prevKeys, prevValues *Array
	var pageState PagedKVState
	var fixedState FixedKVState
	ownsKV := !prev.hasState()
	fixedKV := prev.Fixed
	if ownsKV {
		switch cache := c.(type) {
		case *PagedKVCache:
			offset = cache.Offset()
			pageState = cache.PageState()
			if len(pageState.Keys) == 1 && len(pageState.Values) == 1 {
				prevKeys = pageState.Keys[0]
				prevValues = pageState.Values[0]
			}
			defer pageState.Free()
		case *FixedKVCache:
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
		case prev.hasPages() && len(prev.Pages.Keys) == 1 && len(prev.Pages.Values) == 1:
			prevKeys, prevValues = prev.Pages.Keys[0], prev.Pages.Values[0]
		default:
			return nil, sharedKV{}, false, nil
		}
	}

	out := newArray("FAST_GEMMA4_DECODE_LAYER", x, prevKeys, prevValues, perLayerInput)
	newK := newArray("FAST_GEMMA4_DECODE_LAYER_K", x)
	newV := newArray("FAST_GEMMA4_DECODE_LAYER_V", x)
	args := nativeGemma4LayerArgs(x, prevKeys, prevValues, perLayerInput, fixedMask, layer, cfg, ownsKV, fixedKV, offset)
	rc := C.go_mlx_gemma4_decode_layer(&out.ctx, &newK.ctx, &newV.ctx, &args, DefaultStream().ctx)
	if rc != 0 {
		Free(out, newK, newV)
		if err := lastError(); err != nil {
			return nil, sharedKV{}, true, err
		}
		return nil, sharedKV{}, true, core.E("mlx.nativeGemma4DecodeLayer", core.Sprintf("native wrapper failed (rc=%d)", rc), nil)
	}

	if ownsKV {
		if fixedKV {
			fixed, _ := c.(*FixedKVCache)
			state := fixed.ReplaceFixedFromNativeBorrowed(newK, newV, int(L))
			return out, sharedKV{Keys: state.Keys, Values: state.Values, Offset: offset, Fixed: true}, true, nil
		}
		paged, _ := c.(*PagedKVCache)
		pages := paged.ReplaceSinglePageFromNative(newK, newV, int(L))
		return out, sharedKV{Pages: pages, Offset: offset}, true, nil
	}
	Free(newK, newV)
	return out, prev, true, nil
}

func nativeGemma4FixedGreedyToken(h *Array, perLayerInputs []*Array, caches []Cache, model *Gemma4Model, fixedMasks *fixedGemma4AttentionMaskSet, suppressTokens ...int32) (*Array, bool, error) {
	if reason := nativeGemma4FixedGreedyTokenUnavailableReason(h, perLayerInputs, caches, model, fixedMasks); reason != "" {
		traceNativeSkip("gemma4.model.greedy_token.skip", reason)
		return nil, false, nil
	}

	layerCount := len(model.Layers)
	layerArgsPtr := (*C.go_mlx_gemma4_layer_args)(C.calloc(C.size_t(layerCount), C.size_t(unsafe.Sizeof(C.go_mlx_gemma4_layer_args{}))))
	previousKVsPtr := (*C.int)(C.calloc(C.size_t(layerCount), C.size_t(unsafe.Sizeof(C.int(0)))))
	newKCtxPtr := (*C.mlx_array)(C.calloc(C.size_t(layerCount), C.size_t(unsafe.Sizeof(C.mlx_array{}))))
	newVCtxPtr := (*C.mlx_array)(C.calloc(C.size_t(layerCount), C.size_t(unsafe.Sizeof(C.mlx_array{}))))
	if layerArgsPtr == nil || previousKVsPtr == nil || newKCtxPtr == nil || newVCtxPtr == nil {
		if layerArgsPtr != nil {
			C.free(unsafe.Pointer(layerArgsPtr))
		}
		if previousKVsPtr != nil {
			C.free(unsafe.Pointer(previousKVsPtr))
		}
		if newKCtxPtr != nil {
			C.free(unsafe.Pointer(newKCtxPtr))
		}
		if newVCtxPtr != nil {
			C.free(unsafe.Pointer(newVCtxPtr))
		}
		return nil, true, core.NewError("mlx.nativeGemma4FixedGreedyToken: allocate C argument buffers failed")
	}
	defer C.free(unsafe.Pointer(layerArgsPtr))
	defer C.free(unsafe.Pointer(previousKVsPtr))
	defer C.free(unsafe.Pointer(newKCtxPtr))
	defer C.free(unsafe.Pointer(newVCtxPtr))
	layerArgs := unsafe.Slice(layerArgsPtr, layerCount)
	previousKVs := unsafe.Slice(previousKVsPtr, layerCount)
	newKCtx := unsafe.Slice(newKCtxPtr, layerCount)
	newVCtx := unsafe.Slice(newVCtxPtr, layerCount)
	fixedByLayer := make([]*FixedKVCache, layerCount)
	states := make([]FixedKVState, layerCount)
	offsets := make([]int, layerCount)
	defer func() {
		for i := range states {
			states[i].Free()
		}
	}()

	B := int32(h.Dim(0))
	for i, layer := range model.Layers {
		prevIdx := int(model.PreviousKVs[i])
		previousKVs[i] = C.int(prevIdx)
		ownsKV := prevIdx == i
		var fixed *FixedKVCache
		var prev sharedKV
		var prevKeys, prevValues *Array
		var offset int
		if ownsKV {
			cacheIdx := int(model.CacheIndexByLayer[i])
			fixed = caches[cacheIdx].(*FixedKVCache)
			fixed.ensureShape(B, layer.Attention.NKVHeads, layer.Attention.HeadDim, layer.Attention.HeadDim, h.Dtype(), h.Dtype())
			state := fixed.BorrowedFixedState()
			if state.Keys == nil || state.Values == nil {
				return nil, false, nil
			}
			states[i] = state
			fixedByLayer[i] = fixed
			prevKeys, prevValues = state.Keys, state.Values
			offset = fixed.Offset()
			offsets[i] = offset
		} else {
			state := states[prevIdx]
			if state.Keys == nil || state.Values == nil {
				return nil, false, nil
			}
			prevKeys, prevValues = state.Keys, state.Values
			offset = offsets[prevIdx]
			prev = sharedKV{Keys: prevKeys, Values: prevValues, Offset: offset, Fixed: true}
		}
		var perLayerInput *Array
		if perLayerInputs != nil {
			perLayerInput = perLayerInputs[i]
		}
		fixedMask := fixedMasks.ForLayer(fixed, prev)
		layerArgs[i] = nativeGemma4LayerArgs(h, prevKeys, prevValues, perLayerInput, fixedMask, layer, model.Cfg, ownsKV, true, offset)
	}

	out := newArray("FAST_GEMMA4_MODEL_GREEDY_TOKEN", h, model.NormScaled, model.Output.Weight, model.Output.Scales, model.Output.Biases)
	args := C.go_mlx_gemma4_model_greedy_args{
		hidden:           cArray(h),
		layers:           layerArgsPtr,
		previous_kvs:     previousKVsPtr,
		layer_count:      C.int(layerCount),
		final_norm:       cArray(model.NormScaled),
		output_weight:    cArray(model.Output.Weight),
		output_scales:    cArray(model.Output.Scales),
		output_biases:    cArray(model.Output.Biases),
		output_quantized: 0,
	}
	suppress := suppressTokenArray(suppressTokens)
	defer Free(suppress)
	if suppress != nil {
		args.suppress_token_ids = suppress.ctx
		args.has_suppress_token_ids = 1
	}
	if model.Output.Scales != nil && model.Output.Scales.Valid() {
		args.output_quantized = 1
	}
	rc := C.go_mlx_gemma4_fixed_greedy_token(
		&out.ctx,
		newKCtxPtr,
		newVCtxPtr,
		&args,
		DefaultStream().ctx,
	)
	if rc != 0 {
		Free(out)
		freeCArrayHandles(newKCtx)
		freeCArrayHandles(newVCtx)
		if err := lastError(); err != nil {
			return nil, true, err
		}
		return nil, true, core.E("mlx.nativeGemma4FixedGreedyToken", core.Sprintf("native wrapper failed (rc=%d)", rc), nil)
	}
	if !out.Valid() {
		Free(out)
		freeCArrayHandles(newKCtx)
		freeCArrayHandles(newVCtx)
		return nil, true, core.E("mlx.nativeGemma4FixedGreedyToken", "native wrapper returned invalid token", nil)
	}

	for i, fixed := range fixedByLayer {
		if fixed == nil {
			continue
		}
		newKeys := newArray("FAST_GEMMA4_MODEL_GREEDY_K", h)
		newValues := newArray("FAST_GEMMA4_MODEL_GREEDY_V", h)
		newKeys.ctx = newKCtx[i]
		newValues.ctx = newVCtx[i]
		if !newKeys.Valid() || !newValues.Valid() {
			Free(out, newKeys, newValues)
			return nil, true, core.E("mlx.nativeGemma4FixedGreedyToken", "native wrapper returned invalid KV outputs", nil)
		}
		Free(fixed.keys, fixed.values)
		fixed.keys = newKeys
		fixed.values = newValues
		fixed.offset++
		fixed.length = min(fixed.offset, fixed.maxSize)
	}
	return out, true, nil
}

func nativeGemma4FixedGreedyTokenAvailable(h *Array, perLayerInputs []*Array, caches []Cache, model *Gemma4Model, fixedMasks *fixedGemma4AttentionMaskSet) bool {
	return nativeGemma4FixedGreedyTokenUnavailableReason(h, perLayerInputs, caches, model, fixedMasks) == ""
}

func nativeGemma4FixedGreedyTokenUnavailableReason(h *Array, perLayerInputs []*Array, caches []Cache, model *Gemma4Model, fixedMasks *fixedGemma4AttentionMaskSet) string {
	if !nativeGemma4ModelGreedyEnabled() {
		return "model greedy gate is disabled"
	}
	if h == nil || !h.Valid() || model == nil || model.Cfg == nil || fixedMasks == nil || model.Output == nil || model.NormScaled == nil || !model.NormScaled.Valid() {
		return "model greedy inputs are invalid"
	}
	if h.NumDims() != 3 || h.Dim(0) <= 0 || h.Dim(1) != 1 || h.Dim(2) != int(model.Cfg.HiddenSize) {
		return "hidden state is not a single-token decode row"
	}
	if !nativeLastTokenGreedyTokenAvailable(h, model.NormScaled, model.Output, model.Cfg.RMSNormEps) {
		return "native last-token greedy output is unavailable"
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
		var perLayerInput *Array
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
			fixed, ok := caches[cacheIdx].(*FixedKVCache)
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

func freeCArrayHandles(handles []C.mlx_array) {
	for _, handle := range handles {
		if handle.ctx != nil {
			C.mlx_array_free(handle)
		}
	}
}

func compiledGemma4DecodeLayer(x *Array, c Cache, B, L int32, mask *Array, perLayerInput *Array, prev sharedKV, layer *Gemma4DecoderLayer, cfg *Gemma4TextConfig, fixedMask *Array) (*Array, sharedKV, bool, error) {
	if !compiledGemma4LayerEnabled() {
		return nil, sharedKV{}, false, nil
	}
	if !gemma4CompiledDecodeLayerBoundaryAvailable(x, c, B, L, mask, perLayerInput, prev, layer, cfg) {
		return nil, sharedKV{}, false, nil
	}

	offset := 0
	var prevKeys, prevValues *Array
	var pageState PagedKVState
	var fixedState FixedKVState
	ownsKV := !prev.hasState()
	fixedKV := prev.Fixed
	if ownsKV {
		switch cache := c.(type) {
		case *PagedKVCache:
			offset = cache.Offset()
			pageState = cache.PageState()
			if len(pageState.Keys) != 1 || len(pageState.Values) != 1 {
				pageState.Free()
				return nil, sharedKV{}, false, nil
			}
			prevKeys = pageState.Keys[0]
			prevValues = pageState.Values[0]
			defer pageState.Free()
		case *FixedKVCache:
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
		case prev.hasPages() && len(prev.Pages.Keys) == 1 && len(prev.Pages.Values) == 1:
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

	offsetArray := FromValue(offset)
	defer Free(offsetArray)
	inputs := []*Array{x, prevKeys, prevValues, perLayerInput, offsetArray}
	if useFixedMask {
		inputs = append(inputs, fixedMask)
	}
	outs, callErr := callCompiledGemma4DecodeLayer(compiled, inputs...)
	if callErr != nil {
		*failed = true
		if *slot != nil {
			(*slot).Free()
			*slot = nil
		}
		return nil, sharedKV{}, true, callErr
	}
	if ownsKV {
		if len(outs) != 3 {
			Free(outs...)
			return nil, sharedKV{}, true, core.E("mlx.compiledGemma4DecodeLayer", "owner closure returned invalid outputs", nil)
		}
		if fixedKV {
			fixed, _ := c.(*FixedKVCache)
			state := fixed.ReplaceFixedFromNativeBorrowed(outs[1], outs[2], int(L))
			return outs[0], sharedKV{Keys: state.Keys, Values: state.Values, Offset: offset, Fixed: true}, true, nil
		}
		paged, _ := c.(*PagedKVCache)
		pages := paged.ReplaceSinglePageFromNative(outs[1], outs[2], int(L))
		return outs[0], sharedKV{Pages: pages, Offset: offset}, true, nil
	}
	if len(outs) != 1 {
		Free(outs...)
		return nil, sharedKV{}, true, core.E("mlx.compiledGemma4DecodeLayer", "shared closure returned invalid outputs", nil)
	}
	return outs[0], prev, true, nil
}

func callCompiledGemma4DecodeLayer(compiled *CompiledFunc, inputs ...*Array) (outs []*Array, err error) {
	defer func() {
		if r := recover(); r != nil {
			outs = nil
			err = core.E("mlx.compiledGemma4DecodeLayer", core.Sprintf("compiled closure failed: %v", r), nil)
		}
	}()
	return compiled.Call(inputs...), nil
}

func compileGemma4DecodeLayer(layer *Gemma4DecoderLayer, cfg *Gemma4TextConfig, ownsKV, fixedKV, fixedMask bool) *CompiledFunc {
	return CompileShapeless(func(inputs []*Array) []*Array {
		if len(inputs) < 5 {
			return nil
		}
		var mask *Array
		if fixedMask {
			if len(inputs) < 6 {
				return nil
			}
			mask = inputs[5]
		}
		out, keys, values := gemma4DecodeLayerGraph(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], mask, layer, cfg, ownsKV, fixedKV)
		if ownsKV {
			return []*Array{out, keys, values}
		}
		return []*Array{out}
	}, true)
}

func gemma4DecodeLayerGraph(x, prevKeys, prevValues, perLayerInput, offset, fixedMask *Array, layer *Gemma4DecoderLayer, cfg *Gemma4TextConfig, ownsKV, fixedKV bool) (*Array, *Array, *Array) {
	residual := x
	normed := RMSNorm(x, layer.InputNormScaled, cfg.RMSNormEps)
	attnOut, keys, values := gemma4AttentionGraph(normed, prevKeys, prevValues, offset, fixedMask, layer.Attention, cfg, ownsKV, fixedKV)
	Free(normed)
	attnNormed := RMSNorm(attnOut, layer.PostAttnNormScaled, cfg.RMSNormEps)
	Free(attnOut)
	h := Add(residual, attnNormed)
	Free(attnNormed)

	ffResidual := gemma4DecodeFFNGraph(h, layer, cfg)

	hNext := Add(h, ffResidual)
	Free(h, ffResidual)

	gate := layer.PerLayerInputGate.Forward(hNext)
	multiplied := geluGateMul(gate, perLayerInput)
	Free(gate)
	projected := layer.PerLayerProjection.Forward(multiplied)
	Free(multiplied)
	projectedNormed := RMSNorm(projected, layer.PostPerLayerInputNormScaled, cfg.RMSNormEps)
	Free(projected)
	gated := Add(hNext, projectedNormed)
	Free(hNext, projectedNormed)
	hNext = gated

	scaled := Mul(hNext, layer.LayerScalar)
	Free(hNext)
	return scaled, keys, values
}

func gemma4DecodeFFNGraph(h *Array, layer *Gemma4DecoderLayer, cfg *Gemma4TextConfig) *Array {
	if layer.EnableMoE && layer.Router != nil && layer.Experts != nil {
		h1In := RMSNorm(h, layer.PreFFNormScaled, cfg.RMSNormEps)
		h1 := gemma4MLPGraph(h1In, layer.MLP)
		Free(h1In)
		h1Normed := RMSNorm(h1, layer.PostFFNorm1Scaled, cfg.RMSNormEps)
		Free(h1)

		h2In := RMSNorm(h, layer.PreFFNorm2Scaled, cfg.RMSNormEps)
		topKIndices, topKWeights := layer.Router.forward(h)
		h2 := layer.Experts.forward(h2In, topKIndices, topKWeights, "")
		Free(h2In, topKIndices, topKWeights)
		h2Normed := RMSNorm(h2, layer.PostFFNorm2Scaled, cfg.RMSNormEps)
		Free(h2)

		combined := Add(h1Normed, h2Normed)
		Free(h1Normed, h2Normed)
		ffResidual := RMSNorm(combined, layer.PostFFNormScaled, cfg.RMSNormEps)
		Free(combined)
		return ffResidual
	}

	ffIn := RMSNorm(h, layer.PreFFNormScaled, cfg.RMSNormEps)
	ff := gemma4MLPGraph(ffIn, layer.MLP)
	Free(ffIn)
	ffResidual := RMSNorm(ff, layer.PostFFNormScaled, cfg.RMSNormEps)
	Free(ff)
	return ffResidual
}

func gemma4MLPGraph(x *Array, mlp *MLP) *Array {
	gate := mlp.GateProj.Forward(x)
	up := mlp.UpProj.Forward(x)
	activated := geluGateMul(gate, up)
	Free(gate, up)
	out := mlp.DownProj.Forward(activated)
	Free(activated)
	return out
}

func gemma4AttentionGraph(x, prevKeys, prevValues, offset, fixedMask *Array, attn *Gemma4Attention, cfg *Gemma4TextConfig, ownsKV, fixedKV bool) (*Array, *Array, *Array) {
	B, L := int32(x.Dim(0)), int32(x.Dim(1))
	qProj := attn.QProj.Forward(x)
	qReshaped := Reshape(qProj, B, L, cfg.NumAttentionHeads, attn.HeadDim)
	Free(qProj)
	q := Transpose(qReshaped, 0, 2, 1, 3)
	Free(qReshaped)
	oldQ := q
	q = RMSNorm(q, attn.QNormScaled, cfg.RMSNormEps)
	Free(oldQ)

	var keys, values *Array
	var out *Array
	qHasRoPE := false
	if ownsKV {
		kProj := attn.KProj.Forward(x)
		kReshaped := Reshape(kProj, B, L, attn.NKVHeads, attn.HeadDim)
		Free(kProj)
		k := Transpose(kReshaped, 0, 2, 1, 3)
		Free(kReshaped)
		oldK := k
		k = RMSNorm(k, attn.KNormScaled, cfg.RMSNormEps)
		Free(oldK)
		k = gemma4ApplyRoPEDynamic(attn, k, offset)

		vProj := attn.VProj.Forward(x)
		vReshaped := Reshape(vProj, B, L, attn.NKVHeads, attn.HeadDim)
		Free(vProj)
		v := Transpose(vReshaped, 0, 2, 1, 3)
		Free(vReshaped)
		vNormed := RMSNormNoScale(v, cfg.RMSNormEps)
		Free(v)
		v = vNormed

		if fixedKV {
			q = gemma4ApplyRoPEDynamic(attn, q, offset)
			qHasRoPE = true
			if nativeOut, nativeKeys, nativeValues, ok, err := nativeFixedSingleTokenAttention(q, prevKeys, prevValues, k, v, offset, fixedMask, attn.Scale); ok {
				out = nativeOut
				keys = nativeKeys
				values = nativeValues
			} else {
				if err != nil {
					core.Error("mlx: native fixed single-token attention failed; falling back to Go graph", "error", err)
				}
				keys = singleTokenCacheUpdate(prevKeys, k, offset)
				values = singleTokenCacheUpdate(prevValues, v, offset)
			}
			Free(k, v)
		} else {
			keys = Concatenate([]*Array{prevKeys, k}, 2)
			values = Concatenate([]*Array{prevValues, v}, 2)
			Free(k, v)
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
				mask = singleTokenCausalMask(int(keys.Dim(2)), offset)
				defer Free(mask)
			}
			out = ScaledDotProductAttentionWithMask(q, keys, values, mask, attn.Scale)
		} else {
			out = ScaledDotProductAttention(q, keys, values, attn.Scale, false)
		}
	}
	Free(q)

	transposed := Transpose(out, 0, 2, 1, 3)
	Free(out)
	reshaped := Reshape(transposed, B, L, cfg.NumAttentionHeads*attn.HeadDim)
	Free(transposed)
	result := attn.OProj.Forward(reshaped)
	Free(reshaped)
	if !ownsKV {
		return result, nil, nil
	}
	return result, keys, values
}

func gemma4ApplyRoPEDynamic(attn *Gemma4Attention, x, offset *Array) *Array {
	old := x
	if attn.RopeFreqs != nil {
		x = RoPEWithOffsetArray(x, int(attn.HeadDim), false, 0, 1.0, offset, attn.RopeFreqs)
	} else {
		x = RoPEWithOffsetArray(x, int(attn.RopeRotatedDim), false, attn.RopeBase, 1.0, offset, nil)
	}
	Free(old)
	return x
}

func nativeGemma4LayerArgs(x, prevKeys, prevValues, perLayerInput, fixedMask *Array, layer *Gemma4DecoderLayer, cfg *Gemma4TextConfig, ownsKV, fixedKV bool, offset int) C.go_mlx_gemma4_layer_args {
	attn := layer.Attention
	args := C.go_mlx_gemma4_layer_args{
		x:                         cArray(x),
		prev_keys:                 cArray(prevKeys),
		prev_values:               cArray(prevValues),
		per_layer_input:           cArray(perLayerInput),
		fixed_mask:                cArray(fixedMask),
		input_norm:                cArray(layer.InputNormScaled),
		post_attn_norm:            cArray(layer.PostAttnNormScaled),
		pre_ff_norm:               cArray(layer.PreFFNormScaled),
		pre_ff_norm2:              cArray(layer.PreFFNorm2Scaled),
		post_ff_norm1:             cArray(layer.PostFFNorm1Scaled),
		post_ff_norm2:             cArray(layer.PostFFNorm2Scaled),
		post_ff_norm:              cArray(layer.PostFFNormScaled),
		post_per_layer_input_norm: cArray(layer.PostPerLayerInputNormScaled),
		layer_scalar:              cArray(layer.LayerScalar),
		q_weight:                  cArray(attn.QProj.Weight),
		q_scales:                  cArray(attn.QProj.Scales),
		q_biases:                  cArray(attn.QProj.Biases),
		k_weight:                  cArray(attn.KProj.Weight),
		k_scales:                  cArray(attn.KProj.Scales),
		k_biases:                  cArray(attn.KProj.Biases),
		o_weight:                  cArray(attn.OProj.Weight),
		o_scales:                  cArray(attn.OProj.Scales),
		o_biases:                  cArray(attn.OProj.Biases),
		q_norm:                    cArray(attn.QNormScaled),
		k_norm:                    cArray(attn.KNormScaled),
		rope_freqs:                cArray(attn.RopeFreqs),
		q_group_size:              C.int(attn.QProj.GroupSize),
		q_bits:                    C.int(attn.QProj.Bits),
		k_group_size:              C.int(attn.KProj.GroupSize),
		k_bits:                    C.int(attn.KProj.Bits),
		o_group_size:              C.int(attn.OProj.GroupSize),
		o_bits:                    C.int(attn.OProj.Bits),
		mlp_gate_weight:           cArray(layer.MLP.GateProj.Weight),
		mlp_gate_scales:           cArray(layer.MLP.GateProj.Scales),
		mlp_gate_biases:           cArray(layer.MLP.GateProj.Biases),
		mlp_gate_group_size:       C.int(layer.MLP.GateProj.GroupSize),
		mlp_gate_bits:             C.int(layer.MLP.GateProj.Bits),
		mlp_up_weight:             cArray(layer.MLP.UpProj.Weight),
		mlp_up_scales:             cArray(layer.MLP.UpProj.Scales),
		mlp_up_biases:             cArray(layer.MLP.UpProj.Biases),
		mlp_up_group_size:         C.int(layer.MLP.UpProj.GroupSize),
		mlp_up_bits:               C.int(layer.MLP.UpProj.Bits),
		mlp_down_weight:           cArray(layer.MLP.DownProj.Weight),
		mlp_down_scales:           cArray(layer.MLP.DownProj.Scales),
		mlp_down_biases:           cArray(layer.MLP.DownProj.Biases),
		mlp_down_group_size:       C.int(layer.MLP.DownProj.GroupSize),
		mlp_down_bits:             C.int(layer.MLP.DownProj.Bits),
		num_attention_heads:       C.int(cfg.NumAttentionHeads),
		num_key_value_heads:       C.int(attn.NKVHeads),
		head_dim:                  C.int(attn.HeadDim),
		rope_dims:                 C.int(attn.RopeRotatedDim),
		offset:                    C.int(offset),
		rope_base:                 C.float(attn.RopeBase),
		attention_scale:           C.float(attn.Scale),
	}
	if prevKeys != nil && prevValues != nil {
		args.has_prev = 1
	}
	if perLayerInput != nil && perLayerInput.Valid() {
		args.has_per_layer_input = 1
		args.per_layer_gate_weight = cArray(layer.PerLayerInputGate.Weight)
		args.per_layer_gate_scales = cArray(layer.PerLayerInputGate.Scales)
		args.per_layer_gate_biases = cArray(layer.PerLayerInputGate.Biases)
		args.per_layer_gate_group_size = C.int(layer.PerLayerInputGate.GroupSize)
		args.per_layer_gate_bits = C.int(layer.PerLayerInputGate.Bits)
		args.per_layer_projection_weight = cArray(layer.PerLayerProjection.Weight)
		args.per_layer_projection_scales = cArray(layer.PerLayerProjection.Scales)
		args.per_layer_projection_biases = cArray(layer.PerLayerProjection.Biases)
		args.per_layer_projection_group_size = C.int(layer.PerLayerProjection.GroupSize)
		args.per_layer_projection_bits = C.int(layer.PerLayerProjection.Bits)
	}
	if ownsKV {
		args.owns_kv = 1
	}
	if fixedKV {
		args.fixed_kv = 1
	}
	if fixedMask != nil && fixedMask.Valid() {
		args.has_fixed_mask = 1
	}
	if attn.RopeFreqs != nil && attn.RopeFreqs.Valid() {
		args.has_rope_freqs = 1
	}
	if attn.UseKEqV {
		args.use_k_eq_v = 1
	} else if attn.VProj != nil {
		args.v_weight = cArray(attn.VProj.Weight)
		args.v_scales = cArray(attn.VProj.Scales)
		args.v_biases = cArray(attn.VProj.Biases)
		args.v_group_size = C.int(attn.VProj.GroupSize)
		args.v_bits = C.int(attn.VProj.Bits)
	}
	if layer.EnableMoE && layer.Router != nil && layer.Experts != nil {
		router := layer.Router
		experts := layer.Experts
		args.has_moe = 1
		args.router_weight = cArray(router.Proj.Weight)
		args.router_scales = cArray(router.Proj.Scales)
		args.router_biases = cArray(router.Proj.Biases)
		args.router_group_size = C.int(router.Proj.GroupSize)
		args.router_bits = C.int(router.Proj.Bits)
		if router.ScaleScaled != nil && router.ScaleScaled.Valid() {
			args.router_scale = cArray(router.ScaleScaled)
			args.has_router_scale_scaled = 1
		} else {
			args.router_scale = cArray(router.Scale)
		}
		args.router_per_expert_scale = cArray(router.PerExpertScale)
		args.router_top_k = C.int(router.TopK)
		args.router_eps = C.float(router.Eps)
		args.router_root_size = C.float(router.RootSize)

		if experts.GateProj != nil {
			args.expert_gate_weight = cArray(experts.GateProj.Weight)
			args.expert_gate_scales = cArray(experts.GateProj.Scales)
			args.expert_gate_biases = cArray(experts.GateProj.Biases)
			args.expert_gate_bias = cArray(experts.GateProj.Bias)
			args.expert_gate_group_size = C.int(experts.GateProj.GroupSize)
			args.expert_gate_bits = C.int(experts.GateProj.Bits)
		}
		if experts.UpProj != nil {
			args.expert_up_weight = cArray(experts.UpProj.Weight)
			args.expert_up_scales = cArray(experts.UpProj.Scales)
			args.expert_up_biases = cArray(experts.UpProj.Biases)
			args.expert_up_bias = cArray(experts.UpProj.Bias)
			args.expert_up_group_size = C.int(experts.UpProj.GroupSize)
			args.expert_up_bits = C.int(experts.UpProj.Bits)
		}
		if experts.GateUpProj != nil {
			args.expert_gate_up_weight = cArray(experts.GateUpProj.Weight)
			args.expert_gate_up_scales = cArray(experts.GateUpProj.Scales)
			args.expert_gate_up_biases = cArray(experts.GateUpProj.Biases)
			args.expert_gate_up_bias = cArray(experts.GateUpProj.Bias)
			args.expert_gate_up_group_size = C.int(experts.GateUpProj.GroupSize)
			args.expert_gate_up_bits = C.int(experts.GateUpProj.Bits)
		}
		args.expert_down_weight = cArray(experts.DownProj.Weight)
		args.expert_down_scales = cArray(experts.DownProj.Scales)
		args.expert_down_biases = cArray(experts.DownProj.Biases)
		args.expert_down_bias = cArray(experts.DownProj.Bias)
		args.expert_down_group_size = C.int(experts.DownProj.GroupSize)
		args.expert_down_bits = C.int(experts.DownProj.Bits)
	}
	return args
}

func nativeGemma4DecodeLayerAvailable(x *Array, c Cache, B, L int32, mask *Array, perLayerInput *Array, prev sharedKV, layer *Gemma4DecoderLayer, cfg *Gemma4TextConfig) bool {
	if !nativeGemma4LayerEnabled() {
		return false
	}
	if reason := gemma4DecodeLayerBoundaryUnavailableReason(x, c, B, L, mask, perLayerInput, prev, layer, cfg); reason != "" {
		traceNativeSkip(nativeGemma4LayerSkipTraceName(layer), reason)
		return false
	}
	return true
}

func gemma4DecodeLayerBoundaryAvailable(x *Array, c Cache, B, L int32, mask *Array, perLayerInput *Array, prev sharedKV, layer *Gemma4DecoderLayer, cfg *Gemma4TextConfig) bool {
	return gemma4DecodeLayerBoundaryUnavailableReason(x, c, B, L, mask, perLayerInput, prev, layer, cfg) == ""
}

func gemma4DecodeLayerBoundaryUnavailableReason(x *Array, c Cache, B, L int32, mask *Array, perLayerInput *Array, prev sharedKV, layer *Gemma4DecoderLayer, cfg *Gemma4TextConfig) string {
	if reason := gemma4DecodeLayerCommonUnavailableReason(x, B, L, mask, perLayerInput, layer, cfg); reason != "" {
		return reason
	}
	if gemma4PagedDecodeLayerBoundaryAvailable(c, L, prev) {
		return ""
	}
	if prev.hasState() {
		if prev.Fixed && nativeGemma4SharedKVAvailable(prev) {
			return ""
		}
		return "shared-kv state is not native-compatible"
	}
	fixed, ok := c.(*FixedKVCache)
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

func gemma4DecodeLayerCommonAvailable(x *Array, B, L int32, mask *Array, perLayerInput *Array, layer *Gemma4DecoderLayer, cfg *Gemma4TextConfig) bool {
	return gemma4DecodeLayerCommonUnavailableReason(x, B, L, mask, perLayerInput, layer, cfg) == ""
}

func gemma4DecodeLayerCommonUnavailableReason(x *Array, B, L int32, mask *Array, perLayerInput *Array, layer *Gemma4DecoderLayer, cfg *Gemma4TextConfig) string {
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
	if layer.EnableMoE && layer.Router != nil && layer.Experts != nil && !nativeGemma4MoELayerEnabled() {
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
	if reason := nativeGemma4LayerMLPUnavailableReason(layer.MLP); reason != "" {
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
		if reason := nativeGemma4LayerLinearUnavailableReason(layer.PerLayerInputGate, "per-layer gate"); reason != "" {
			return reason
		}
		if reason := nativeGemma4LayerLinearUnavailableReason(layer.PerLayerProjection, "per-layer projection"); reason != "" {
			return reason
		}
	}
	if layer.LayerScalar == nil || !layer.LayerScalar.Valid() {
		return "layer scalar is invalid"
	}
	return ""
}

func nativeGemma4LayerSkipTraceName(layer *Gemma4DecoderLayer) string {
	if layer == nil {
		return "gemma4.layer.unknown.native_layer.skip"
	}
	return core.Sprintf("gemma4.layer.%02d.native_layer.skip", layer.LayerIdx)
}

func gemma4CompiledDecodeLayerBoundaryAvailable(x *Array, c Cache, B, L int32, mask *Array, perLayerInput *Array, prev sharedKV, layer *Gemma4DecoderLayer, cfg *Gemma4TextConfig) bool {
	if !gemma4DecodeLayerCommonAvailable(x, B, L, mask, perLayerInput, layer, cfg) {
		return false
	}
	if gemma4PagedDecodeLayerBoundaryAvailable(c, L, prev) {
		return true
	}
	if prev.hasState() {
		return prev.Fixed && nativeGemma4SharedKVAvailable(prev)
	}
	fixed, ok := c.(*FixedKVCache)
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
	if reason := nativeGemma4LayerLinearUnavailableReason(router.Proj, "router"); reason != "" {
		return reason
	}
	if (router.ScaleScaled == nil || !router.ScaleScaled.Valid()) && (router.Scale == nil || !router.Scale.Valid()) {
		return "router scale is invalid"
	}
	experts := layer.Experts
	if reason := gemma4DecodeSwitchLinearUnavailableReason(experts.DownProj, "expert down"); reason != "" {
		return reason
	}
	if gemma4DecodeSwitchLinearAvailable(experts.GateUpProj) {
		return ""
	}
	if reason := gemma4DecodeSwitchLinearUnavailableReason(experts.GateProj, "expert gate"); reason != "" {
		return reason
	}
	if reason := gemma4DecodeSwitchLinearUnavailableReason(experts.UpProj, "expert up"); reason != "" {
		return reason
	}
	return ""
}

func gemma4DecodeSwitchLinearAvailable(linear *SwitchLinear) bool {
	return gemma4DecodeSwitchLinearUnavailableReason(linear, "switch") == ""
}

func gemma4DecodeSwitchLinearUnavailableReason(linear *SwitchLinear, name string) string {
	if linear == nil || linear.Weight == nil || !linear.Weight.Valid() {
		return name + " switch linear is invalid"
	}
	if linear.Scales != nil && !linear.Scales.Valid() {
		return name + " switch scales are invalid"
	}
	if linear.Biases != nil && !linear.Biases.Valid() {
		return name + " switch biases are invalid"
	}
	if linear.Bias != nil && !linear.Bias.Valid() {
		return name + " switch bias is invalid"
	}
	if linear.Scales == nil {
		return ""
	}
	if !isAffineQuantizationMode(linear.QuantizationMode) {
		return name + " switch quantization mode is unsupported"
	}
	if linear.Biases == nil || !linear.Biases.Valid() {
		return name + " switch quantization biases are invalid"
	}
	if !validGemma4LayerQuantization(linear.GroupSize, linear.Bits) {
		return core.Sprintf("%s switch quantization is unsupported: group_size=%d bits=%d", name, linear.GroupSize, linear.Bits)
	}
	return ""
}

func gemma4PagedDecodeLayerBoundaryAvailable(c Cache, L int32, prev sharedKV) bool {
	if prev.hasState() {
		return !prev.Fixed && nativeGemma4SharedKVAvailable(prev)
	}
	paged, ok := c.(*PagedKVCache)
	if !ok {
		return false
	}
	if paged.maxSize > 0 && paged.Len()+int(L) > paged.maxSize {
		return false
	}
	if len(paged.kPages) == 1 && pagedArrayLen(paged.kPages[0]) >= paged.pageSize {
		return false
	}
	return len(paged.kPages) <= 1 && len(paged.vPages) <= 1
}

func nativeGemma4NormsAvailable(layer *Gemma4DecoderLayer) bool {
	norms := []*Array{
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
	if reason := nativeGemma4LayerLinearUnavailableReason(attn.QProj, "attention q"); reason != "" {
		return reason
	}
	if reason := nativeGemma4LayerLinearUnavailableReason(attn.KProj, "attention k"); reason != "" {
		return reason
	}
	if !attn.UseKEqV {
		if reason := nativeGemma4LayerLinearUnavailableReason(attn.VProj, "attention v"); reason != "" {
			return reason
		}
	}
	if reason := nativeGemma4LayerLinearUnavailableReason(attn.OProj, "attention o"); reason != "" {
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

func nativeGemma4LayerMLPAvailable(mlp *MLP) bool {
	return nativeGemma4LayerMLPUnavailableReason(mlp) == ""
}

func nativeGemma4LayerMLPUnavailableReason(mlp *MLP) string {
	if mlp == nil {
		return "mlp is nil"
	}
	if reason := nativeGemma4LayerLinearUnavailableReason(mlp.GateProj, "mlp gate"); reason != "" {
		return reason
	}
	if reason := nativeGemma4LayerLinearUnavailableReason(mlp.UpProj, "mlp up"); reason != "" {
		return reason
	}
	if reason := nativeGemma4LayerLinearUnavailableReason(mlp.DownProj, "mlp down"); reason != "" {
		return reason
	}
	return ""
}

func nativeGemma4LayerLinearAvailable(linear *Linear) bool {
	return nativeGemma4LayerLinearUnavailableReason(linear, "linear") == ""
}

func nativeGemma4LayerLinearUnavailableReason(linear *Linear, name string) string {
	if linear == nil || linear.LoRA != nil || linear.Weight == nil || !linear.Weight.Valid() {
		return name + " linear is invalid"
	}
	if linear.Bias != nil && linear.Bias.Valid() {
		return name + " linear has unsupported bias"
	}
	if linear.Scales == nil {
		if linear.Biases == nil || !linear.Biases.Valid() {
			return ""
		}
		return name + " dense linear has quantization biases"
	}
	if !isAffineQuantizationMode(linear.QuantizationMode) {
		return name + " quantization mode is unsupported"
	}
	if !linear.Scales.Valid() || linear.Biases == nil || !linear.Biases.Valid() {
		return name + " quantization sidecars are invalid"
	}
	if !validGemma4LayerQuantization(linear.GroupSize, linear.Bits) {
		return core.Sprintf("%s quantization is unsupported: group_size=%d bits=%d", name, linear.GroupSize, linear.Bits)
	}
	return ""
}

func nativeGemma4AttentionAvailable(attn *Gemma4Attention) bool {
	if attn == nil || attn.HeadDim <= 0 || attn.RopeRotatedDim <= 0 || attn.NKVHeads <= 0 {
		return false
	}
	return nativeMLPLinearAvailable(attn.QProj) &&
		nativeMLPLinearAvailable(attn.KProj) &&
		nativeMLPLinearAvailable(attn.VProj) &&
		nativeMLPLinearAvailable(attn.OProj) &&
		attn.QNormScaled != nil && attn.QNormScaled.Valid() &&
		attn.KNormScaled != nil && attn.KNormScaled.Valid()
}

func nativeGemma4MLPAvailable(mlp *MLP) bool {
	if mlp == nil {
		return false
	}
	return nativeMLPLinearAvailable(mlp.GateProj) &&
		nativeMLPLinearAvailable(mlp.UpProj) &&
		nativeMLPLinearAvailable(mlp.DownProj)
}

func validGemma4LayerQuantization(groupSize, bits int) bool {
	if groupSize <= 0 {
		return false
	}
	switch bits {
	case 2, 4, 8:
		return true
	default:
		return false
	}
}

func nativeGemma4SharedKVAvailable(prev sharedKV) bool {
	switch {
	case prev.Keys != nil && prev.Keys.Valid() && prev.Values != nil && prev.Values.Valid():
		return true
	case prev.hasPages() && len(prev.Pages.Keys) == 1 && len(prev.Pages.Values) == 1:
		return prev.Pages.Keys[0] != nil && prev.Pages.Keys[0].Valid() &&
			prev.Pages.Values[0] != nil && prev.Pages.Values[0].Valid()
	default:
		return false
	}
}
