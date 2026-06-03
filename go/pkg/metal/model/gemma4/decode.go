// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

/*
#cgo CFLAGS: -mmacosx-version-min=26.0
#cgo darwin CFLAGS: -x objective-c
#cgo CPPFLAGS: -I${SRCDIR}/../..
#cgo CPPFLAGS: -I${SRCDIR}/../../../../../external/go-cgo/go
#cgo CPPFLAGS: -I${SRCDIR}/../../../../../lib/mlx
#cgo CPPFLAGS: -I${SRCDIR}/../../../../../lib/mlx-c
#cgo CPPFLAGS: -I${SRCDIR}/../../../../../lib/fmt/include
#cgo CPPFLAGS: -I${SRCDIR}/../../../../../lib/gguflib
#cgo CPPFLAGS: -I${SRCDIR}/../../../../../lib/json/single_include/nlohmann
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
int go_mlx_compiled_q8_g64_last_token(
	mlx_array* res,
	const mlx_array hidden,
	const mlx_array norm_weight,
	const mlx_array output_weight,
	const mlx_array output_scales,
	const mlx_array output_biases,
	const mlx_stream stream);
int go_mlx_compiled_q8_g64_last_token_suppressed(
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
	"runtime"
	"unsafe"

	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/metal"
)

func nativeGemma4FixedOwnerAttentionBlock(x *metal.Array, fixed *metal.FixedKVCache, fixedMask *metal.Array, attn *Gemma4Attention, cfg *Gemma4TextConfig) (*metal.Array, sharedKV, bool, error) {
	if !nativeGemma4FixedOwnerAttentionBlockAvailable(x, fixed, fixedMask, attn, cfg) {
		return nil, sharedKV{}, false, nil
	}
	fixed.ensureShape(int32(x.Dim(0)), attn.NKVHeads, attn.HeadDim, attn.HeadDim, x.Dtype(), x.Dtype())
	state := fixed.BorrowedFixedState()
	if state.Keys == nil || state.Values == nil {
		return nil, sharedKV{}, false, nil
	}
	offset := fixed.Offset()
	offsetArray := metal.FromValue(offset)
	scaleArray := metal.FromValue(attn.Scale)
	defer metal.Free(offsetArray, scaleArray)

	out := metal.NewArray("FAST_GEMMA4_FIXED_OWNER_ATTENTION", x, state.Keys, state.Values)
	newKeys := metal.NewArray("FAST_GEMMA4_FIXED_OWNER_ATTENTION_K", state.Keys)
	newValues := metal.NewArray("FAST_GEMMA4_FIXED_OWNER_ATTENTION_V", state.Values)
	args := nativeGemma4FixedOwnerAttentionArgs(x, nil, state.Keys, state.Values, offsetArray, scaleArray, fixedMask, attn, nil, cfg)
	rc := C.go_mlx_gemma4_fixed_owner_attention(&out.ctx, &newKeys.ctx, &newValues.ctx, &args, gemma4DefaultStream())
	if rc != 0 {
		metal.Free(out, newKeys, newValues)
		if err := metal.LastError(); err != nil {
			return nil, sharedKV{}, true, err
		}
		return nil, sharedKV{}, true, core.E("mlx.nativeGemma4FixedOwnerAttentionBlock", core.Sprintf("native wrapper failed (rc=%d)", rc), nil)
	}
	if err := metal.ValidateGemma4LayerOutputs("mlx.nativeGemma4FixedOwnerAttentionBlock", []*metal.Array{out, newKeys, newValues}, true); err != nil {
		metal.Free(out, newKeys, newValues)
		return nil, sharedKV{}, true, err
	}
	if err := metal.ValidateGemma4LayerOutputShapes("mlx.nativeGemma4FixedOwnerAttentionBlock", x, out, newKeys, newValues, state.Keys, state.Values, true, true); err != nil {
		metal.Free(out, newKeys, newValues)
		return nil, sharedKV{}, true, err
	}
	fixedState := fixed.ReplaceFixedFromNativeBorrowed(newKeys, newValues, 1)
	if !gemma4ValidKV(fixedState.Keys, fixedState.Values) {
		metal.Free(out)
		return nil, sharedKV{}, true, core.E("mlx.nativeGemma4FixedOwnerAttentionBlock", "native wrapper updated cache without valid K/V state", nil)
	}
	return out, sharedKV{Keys: fixedState.Keys, Values: fixedState.Values, Offset: offset, Fixed: true, Borrowed: true}, true, nil
}

func nativeGemma4FixedOwnerAttentionResidualBlock(residual, x *metal.Array, fixed *metal.FixedKVCache, fixedMask *metal.Array, attn *Gemma4Attention, postAttnNorm *metal.Array, cfg *Gemma4TextConfig) (*metal.Array, sharedKV, bool, error) {
	if !nativeGemma4FixedOwnerAttentionResidualBlockAvailable(residual, x, fixed, fixedMask, attn, postAttnNorm, cfg) {
		return nil, sharedKV{}, false, nil
	}
	fixed.ensureShape(int32(x.Dim(0)), attn.NKVHeads, attn.HeadDim, attn.HeadDim, x.Dtype(), x.Dtype())
	state := fixed.BorrowedFixedState()
	if state.Keys == nil || state.Values == nil {
		return nil, sharedKV{}, false, nil
	}
	offset := fixed.Offset()
	offsetArray := metal.FromValue(offset)
	scaleArray := metal.FromValue(attn.Scale)
	defer metal.Free(offsetArray, scaleArray)

	out := metal.NewArray("FAST_GEMMA4_FIXED_OWNER_ATTENTION_RESIDUAL", residual, x, state.Keys, state.Values)
	newKeys := metal.NewArray("FAST_GEMMA4_FIXED_OWNER_ATTENTION_RESIDUAL_K", state.Keys)
	newValues := metal.NewArray("FAST_GEMMA4_FIXED_OWNER_ATTENTION_RESIDUAL_V", state.Values)
	args := nativeGemma4FixedOwnerAttentionArgs(x, residual, state.Keys, state.Values, offsetArray, scaleArray, fixedMask, attn, postAttnNorm, cfg)
	rc := C.go_mlx_gemma4_fixed_owner_attention_residual(&out.ctx, &newKeys.ctx, &newValues.ctx, &args, gemma4DefaultStream())
	if rc != 0 {
		metal.Free(out, newKeys, newValues)
		if err := metal.LastError(); err != nil {
			return nil, sharedKV{}, true, err
		}
		return nil, sharedKV{}, true, core.E("mlx.nativeGemma4FixedOwnerAttentionResidualBlock", core.Sprintf("native wrapper failed (rc=%d)", rc), nil)
	}
	if err := metal.ValidateGemma4LayerOutputs("mlx.nativeGemma4FixedOwnerAttentionResidualBlock", []*metal.Array{out, newKeys, newValues}, true); err != nil {
		metal.Free(out, newKeys, newValues)
		return nil, sharedKV{}, true, err
	}
	if err := metal.ValidateGemma4LayerOutputShapes("mlx.nativeGemma4FixedOwnerAttentionResidualBlock", residual, out, newKeys, newValues, state.Keys, state.Values, true, true); err != nil {
		metal.Free(out, newKeys, newValues)
		return nil, sharedKV{}, true, err
	}
	fixedState := fixed.ReplaceFixedFromNativeBorrowed(newKeys, newValues, 1)
	if !gemma4ValidKV(fixedState.Keys, fixedState.Values) {
		metal.Free(out)
		return nil, sharedKV{}, true, core.E("mlx.nativeGemma4FixedOwnerAttentionResidualBlock", "native wrapper updated cache without valid K/V state", nil)
	}
	return out, sharedKV{Keys: fixedState.Keys, Values: fixedState.Values, Offset: offset, Fixed: true, Borrowed: true}, true, nil
}

func nativeGemma4FixedOwnerAttentionArgs(x, residual, keyCache, valueCache, offset, scale, fixedMask *metal.Array, attn *Gemma4Attention, postAttnNorm *metal.Array, cfg *Gemma4TextConfig) C.go_mlx_gemma4_fixed_attention_args {
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

	offset := 0
	var prevKeys, prevValues *metal.Array
	var pageState metal.PagedKVState
	var fixedState metal.FixedKVState
	ownsKV := !prev.hasState()
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
		case prev.hasPages() && len(prev.Pages.Keys) == 1 && len(prev.Pages.Values) == 1:
			prevKeys, prevValues = prev.Pages.Keys[0], prev.Pages.Values[0]
		default:
			return nil, sharedKV{}, false, nil
		}
	}
	if prevKeys == nil || prevValues == nil || !prevKeys.Valid() || !prevValues.Valid() {
		return nil, sharedKV{}, false, nil
	}

	out := metal.NewArray("FAST_GEMMA4_DECODE_LAYER", x, prevKeys, prevValues, perLayerInput)
	newK := metal.NewArray("FAST_GEMMA4_DECODE_LAYER_K", x)
	newV := metal.NewArray("FAST_GEMMA4_DECODE_LAYER_V", x)
	args := nativeGemma4LayerArgs(x, prevKeys, prevValues, perLayerInput, fixedMask, layer, cfg, ownsKV, fixedKV, offset)
	rc := C.go_mlx_gemma4_decode_layer(&out.ctx, &newK.ctx, &newV.ctx, &args, gemma4DefaultStream())
	if rc != 0 {
		metal.Free(out, newK, newV)
		if err := metal.LastError(); err != nil {
			return nil, sharedKV{}, true, err
		}
		return nil, sharedKV{}, true, core.E("mlx.nativeGemma4DecodeLayer", core.Sprintf("native wrapper failed (rc=%d)", rc), nil)
	}

	if ownsKV {
		if err := metal.ValidateGemma4LayerOutputs("mlx.nativeGemma4DecodeLayer", []*metal.Array{out, newK, newV}, true); err != nil {
			metal.Free(out, newK, newV)
			return nil, sharedKV{}, true, err
		}
		if err := metal.ValidateGemma4LayerOutputShapes("mlx.nativeGemma4DecodeLayer", x, out, newK, newV, prevKeys, prevValues, true, fixedKV); err != nil {
			metal.Free(out, newK, newV)
			return nil, sharedKV{}, true, err
		}
		if fixedKV {
			fixed, _ := c.(*metal.FixedKVCache)
			state := fixed.ReplaceFixedFromNativeBorrowed(newK, newV, int(L))
			return out, sharedKV{Keys: state.Keys, Values: state.Values, Offset: offset, Fixed: true, Borrowed: true}, true, nil
		}
		paged, _ := c.(*metal.PagedKVCache)
		pages := paged.ReplaceSinglePageFromNative(newK, newV, int(L))
		return out, sharedKV{Pages: pages, Offset: offset}, true, nil
	}
	if err := metal.ValidateGemma4LayerOutputs("mlx.nativeGemma4DecodeLayer", []*metal.Array{out}, false); err != nil {
		metal.Free(out, newK, newV)
		return nil, sharedKV{}, true, err
	}
	if err := metal.ValidateGemma4LayerOutputShapes("mlx.nativeGemma4DecodeLayer", x, out, nil, nil, prevKeys, prevValues, false, fixedKV); err != nil {
		metal.Free(out, newK, newV)
		return nil, sharedKV{}, true, err
	}
	metal.Free(newK, newV)
	return out, prev, true, nil
}

func nativeGemma4FixedGreedyToken(h *metal.Array, perLayerInputs []*metal.Array, caches []metal.Cache, model *Gemma4Model, fixedMasks *fixedGemma4AttentionMaskSet, suppressTokens ...int32) (*metal.Array, bool, error) {
	return nativeGemma4FixedGreedyTokenWithArray(h, perLayerInputs, caches, model, fixedMasks, nil, suppressTokens...)
}

func nativeGemma4FixedGreedyTokenWithArray(h *metal.Array, perLayerInputs []*metal.Array, caches []metal.Cache, model *Gemma4Model, fixedMasks *fixedGemma4AttentionMaskSet, suppress *metal.Array, suppressTokens ...int32) (*metal.Array, bool, error) {
	if reason := nativeGemma4FixedGreedyTokenUnavailableReason(h, perLayerInputs, caches, model, fixedMasks); reason != "" {
		metal.TraceNativeSkip("gemma4.model.greedy_token.skip", reason)
		return nil, false, nil
	}

	layerCount := len(model.Layers)
	var layerArgsStack [64]C.go_mlx_gemma4_layer_args
	var previousKVsStack [64]C.int
	var newKCtxStack [64]C.mlx_array
	var newVCtxStack [64]C.mlx_array
	var layerArgs []C.go_mlx_gemma4_layer_args
	var previousKVs []C.int
	var newKCtx []C.mlx_array
	var newVCtx []C.mlx_array
	var layerArgsPtr *C.go_mlx_gemma4_layer_args
	var previousKVsPtr *C.int
	var newKCtxPtr *C.mlx_array
	var newVCtxPtr *C.mlx_array
	var cgoPinner runtime.Pinner
	defer cgoPinner.Unpin()
	if layerCount <= len(layerArgsStack) {
		layerArgs = layerArgsStack[:layerCount]
		previousKVs = previousKVsStack[:layerCount]
		newKCtx = newKCtxStack[:layerCount]
		newVCtx = newVCtxStack[:layerCount]
		layerArgsPtr = &layerArgs[0]
		previousKVsPtr = &previousKVs[0]
		newKCtxPtr = &newKCtx[0]
		newVCtxPtr = &newVCtx[0]
		cgoPinner.Pin(layerArgsPtr)
		cgoPinner.Pin(previousKVsPtr)
		cgoPinner.Pin(newKCtxPtr)
		cgoPinner.Pin(newVCtxPtr)
	} else {
		layerArgsPtr = (*C.go_mlx_gemma4_layer_args)(C.calloc(C.size_t(layerCount), C.size_t(unsafe.Sizeof(C.go_mlx_gemma4_layer_args{}))))
		previousKVsPtr = (*C.int)(C.calloc(C.size_t(layerCount), C.size_t(unsafe.Sizeof(C.int(0)))))
		newKCtxPtr = (*C.mlx_array)(C.calloc(C.size_t(layerCount), C.size_t(unsafe.Sizeof(C.mlx_array{}))))
		newVCtxPtr = (*C.mlx_array)(C.calloc(C.size_t(layerCount), C.size_t(unsafe.Sizeof(C.mlx_array{}))))
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
		layerArgs = unsafe.Slice(layerArgsPtr, layerCount)
		previousKVs = unsafe.Slice(previousKVsPtr, layerCount)
		newKCtx = unsafe.Slice(newKCtxPtr, layerCount)
		newVCtx = unsafe.Slice(newVCtxPtr, layerCount)
	}
	var fixedByLayerStack [64]*metal.FixedKVCache
	var statesStack [64]metal.FixedKVState
	var offsetsStack [64]int
	var fixedByLayer []*metal.FixedKVCache
	var states []metal.FixedKVState
	var offsets []int
	if layerCount <= len(statesStack) {
		fixedByLayer = fixedByLayerStack[:layerCount]
		states = statesStack[:layerCount]
		offsets = offsetsStack[:layerCount]
	} else {
		fixedByLayer = make([]*metal.FixedKVCache, layerCount)
		states = make([]metal.FixedKVState, layerCount)
		offsets = make([]int, layerCount)
	}
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
		var fixed *metal.FixedKVCache
		var prev sharedKV
		var prevKeys, prevValues *metal.Array
		var offset int
		if ownsKV {
			cacheIdx := int(model.CacheIndexByLayer[i])
			fixed = caches[cacheIdx].(*metal.FixedKVCache)
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
			prev = sharedKV{Keys: prevKeys, Values: prevValues, Offset: offset, Fixed: true, Borrowed: true}
		}
		var perLayerInput *metal.Array
		if perLayerInputs != nil {
			perLayerInput = perLayerInputs[i]
		}
		fixedMask := fixedMasks.ForLayer(fixed, prev)
		layerArgs[i] = nativeGemma4LayerArgs(h, prevKeys, prevValues, perLayerInput, fixedMask, layer, model.Cfg, ownsKV, true, offset)
	}

	out := metal.NewArray("FAST_GEMMA4_MODEL_GREEDY_TOKEN", h, model.NormScaled, model.Output.Weight, model.Output.Scales, model.Output.Biases)
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
	ownsSuppress := false
	if len(suppressTokens) == 0 {
		suppress = nil
	} else if suppress == nil || !suppress.Valid() {
		suppress = metal.SuppressTokenArray(suppressTokens)
		ownsSuppress = true
	}
	if ownsSuppress {
		defer metal.Free(suppress)
	}
	if suppress != nil {
		args.suppress_token_ids = suppress.ctx
		args.has_suppress_token_ids = 1
	}
	if model.Output.Scales != nil && model.Output.Scales.Valid() {
		args.output_quantized = 1
	}
	cgoPinner.Pin(&args)
	rc := C.go_mlx_gemma4_fixed_greedy_token(
		&out.ctx,
		newKCtxPtr,
		newVCtxPtr,
		&args,
		gemma4DefaultStream(),
	)
	if rc != 0 {
		metal.Free(out)
		metal.FreeCArrayHandles(newKCtx)
		metal.FreeCArrayHandles(newVCtx)
		if err := metal.LastError(); err != nil {
			return nil, true, err
		}
		return nil, true, core.E("mlx.nativeGemma4FixedGreedyToken", core.Sprintf("native wrapper failed (rc=%d)", rc), nil)
	}
	if !out.Valid() {
		metal.Free(out)
		metal.FreeCArrayHandles(newKCtx)
		metal.FreeCArrayHandles(newVCtx)
		return nil, true, core.E("mlx.nativeGemma4FixedGreedyToken", "native wrapper returned invalid token", nil)
	}

	for i, fixed := range fixedByLayer {
		if fixed == nil {
			continue
		}
		newKeys := metal.NewArray("FAST_GEMMA4_MODEL_GREEDY_K", h)
		newValues := metal.NewArray("FAST_GEMMA4_MODEL_GREEDY_V", h)
		newKeys.ctx = newKCtx[i]
		newValues.ctx = newVCtx[i]
		if !newKeys.Valid() || !newValues.Valid() {
			metal.Free(out, newKeys, newValues)
			return nil, true, core.E("mlx.nativeGemma4FixedGreedyToken", "native wrapper returned invalid KV outputs", nil)
		}
		metal.Free(fixed.keys, fixed.values)
		fixed.keys = newKeys
		fixed.values = newValues
		fixed.offset++
		fixed.length = min(fixed.offset, fixed.maxSize)
	}
	return out, true, nil
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
	ownsKV := !prev.hasState()
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

func nativeGemma4LayerArgs(x, prevKeys, prevValues, perLayerInput, fixedMask *metal.Array, layer *Gemma4DecoderLayer, cfg *Gemma4TextConfig, ownsKV, fixedKV bool, offset int) C.go_mlx_gemma4_layer_args {
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
	if prev.hasState() {
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
	if prev.hasState() {
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
	if prev.hasState() {
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
	case prev.hasPages() && len(prev.Pages.Keys) == 1 && len(prev.Pages.Values) == 1:
		return prev.Pages.Keys[0] != nil && prev.Pages.Keys[0].Valid() &&
			prev.Pages.Values[0] != nil && prev.Pages.Values[0].Valid()
	default:
		return false
	}
}

// cArray rebuilds this package's C.mlx_array from a metal *Array's opaque handle
// (cgo C types are package-private, so we can't share metal's C.mlx_array).
func cArray(a *metal.Array) C.mlx_array {
	var r C.mlx_array
	if a != nil {
		r.ctx = metal.ArrayHandle(a)
	}
	return r
}

// gemma4DefaultStream rebuilds this package's C.mlx_stream from metal's default-stream handle.
func gemma4DefaultStream() C.mlx_stream {
	var s C.mlx_stream
	s.ctx = metal.DefaultStreamHandle()
	return s
}
