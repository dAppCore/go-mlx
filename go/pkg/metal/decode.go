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
int go_mlx_compiled_q6_g64_last_token(
	mlx_array* res,
	const mlx_array hidden,
	const mlx_array norm_weight,
	const mlx_array output_weight,
	const mlx_array output_scales,
	const mlx_array output_biases,
	const mlx_stream stream);
int go_mlx_compiled_q6_g64_last_token_suppressed(
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
	"sync/atomic"

	core "dappco.re/go"
)

// FixedWide SDPA/matmul + row-cache-update are diagnostic-only attention knobs,
// reachable only via SetFixedAttentionDiagnostics so ambient env cannot select
// them. Every other fast-path is a typed runtime Gate the loaded model's
// EngineFeatures.Apply sets (runtime_gate.go), so a later clear is honoured
// rather than frozen at boot. (#55 slice 3b)
var (
	enableFixedWideSDPAAttention   atomic.Bool
	enableFixedWideMatmulAttention atomic.Bool
	enableFixedRowCacheUpdate      atomic.Bool
)

func SetFixedAttentionDiagnostics(wideSDPA, wideMatmul, rowCacheUpdate bool) func() {
	previousWideSDPA := enableFixedWideSDPAAttention.Load()
	previousWideMatmul := enableFixedWideMatmulAttention.Load()
	previousRowCacheUpdate := enableFixedRowCacheUpdate.Load()
	setFixedAttentionDiagnostics(wideSDPA, wideMatmul, rowCacheUpdate)
	return func() {
		setFixedAttentionDiagnostics(previousWideSDPA, previousWideMatmul, previousRowCacheUpdate)
	}
}

func setFixedAttentionDiagnostics(wideSDPA, wideMatmul, rowCacheUpdate bool) {
	enableFixedWideSDPAAttention.Store(wideSDPA)
	enableFixedWideMatmulAttention.Store(wideMatmul)
	enableFixedRowCacheUpdate.Store(rowCacheUpdate)
	C.go_mlx_set_fixed_attention_diagnostics(boolToCInt(wideMatmul), boolToCInt(rowCacheUpdate))
}

func FixedWideSDPAAttentionEnabled() bool {
	return enableFixedWideSDPAAttention.Load()
}

func FixedWideMatmulAttentionEnabled() bool {
	return enableFixedWideMatmulAttention.Load()
}

func FixedRowCacheUpdateEnabled() bool {
	return enableFixedRowCacheUpdate.Load()
}

func boolToCInt(v bool) C.int {
	if v {
		return 1
	}
	return 0
}

func fixedSlidingCacheEnabled() bool { return fixedSlidingCacheRuntimeEnabled() }

func fixedSlidingCacheBoundEnabled() bool { return fixedSlidingCacheBoundRuntimeEnabled() }

func FixedSharedMaskEnabled() bool { return fixedSharedMaskRuntimeEnabled() }

func directGreedyTokenEnabled() bool {
	return directGreedyTokenRuntimeEnabled()
}

func NativeAttentionOMatVecEnabled() bool {
	return nativeAttentionOMatVecRuntimeEnabled()
}

func NativeFixedSlidingAttentionEnabled() bool { return nativeFixedSlidingAttentionRuntimeEnabled() }

func CArray(a *Array) C.mlx_array {
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
	out := NewArray("FAST_GREEDY_DECODE_TOKEN", logits)
	rc := C.go_mlx_compiled_greedy_decode_token(&out.ctx, logits.ctx, DefaultStream().ctx)
	if rc != 0 {
		Free(out)
		if err := LastError(); err != nil {
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

func NativeLastTokenOutputLogits(hidden, normWeight *Array, output *Linear, eps, softcap float32) (*Array, bool, error) {
	if !nativeLastTokenOutputAvailable(hidden, normWeight, output, eps, softcap) {
		return nil, false, nil
	}
	out := NewArray("FAST_LAST_TOKEN_OUTPUT_LOGITS", hidden, normWeight, output.Weight, output.Scales, output.Biases)
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
		if err := LastError(); err != nil {
			return nil, true, err
		}
		return nil, true, core.E("mlx.NativeLastTokenOutputLogits", core.Sprintf("native wrapper failed (rc=%d)", rc), nil)
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
	return NativeLastTokenGreedyTokenWithArray(hidden, normWeight, output, eps, nil, suppressTokens...)
}

func NativeLastTokenGreedyTokenWithArray(hidden, normWeight *Array, output *Linear, eps float32, suppress *Array, suppressTokens ...int32) (*Array, bool, error) {
	if !NativeLastTokenGreedyTokenAvailable(hidden, normWeight, output, eps) {
		return nil, false, nil
	}
	out := NewArray("FAST_LAST_TOKEN_GREEDY", hidden, normWeight, output.Weight, output.Scales, output.Biases)
	var rc C.int
	ownsSuppress := false
	if len(suppressTokens) == 0 {
		suppress = nil
	} else if suppress == nil || !suppress.Valid() {
		suppress = SuppressTokenArray(suppressTokens)
		ownsSuppress = true
	}
	if ownsSuppress {
		defer Free(suppress)
	}
	if output.Scales != nil {
		if output.Bits == 4 && suppress != nil {
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
		} else if output.Bits == 4 {
			rc = C.go_mlx_compiled_q4_g64_last_token(
				&out.ctx,
				hidden.ctx,
				normWeight.ctx,
				output.Weight.ctx,
				output.Scales.ctx,
				output.Biases.ctx,
				DefaultStream().ctx,
			)
		} else if output.Bits == 6 && suppress != nil {
			rc = C.go_mlx_compiled_q6_g64_last_token_suppressed(
				&out.ctx,
				hidden.ctx,
				normWeight.ctx,
				output.Weight.ctx,
				output.Scales.ctx,
				output.Biases.ctx,
				suppress.ctx,
				DefaultStream().ctx,
			)
		} else if output.Bits == 6 {
			rc = C.go_mlx_compiled_q6_g64_last_token(
				&out.ctx,
				hidden.ctx,
				normWeight.ctx,
				output.Weight.ctx,
				output.Scales.ctx,
				output.Biases.ctx,
				DefaultStream().ctx,
			)
		} else if output.Bits == 8 && suppress != nil {
			rc = C.go_mlx_compiled_q8_g64_last_token_suppressed(
				&out.ctx,
				hidden.ctx,
				normWeight.ctx,
				output.Weight.ctx,
				output.Scales.ctx,
				output.Biases.ctx,
				suppress.ctx,
				DefaultStream().ctx,
			)
		} else if output.Bits == 8 {
			rc = C.go_mlx_compiled_q8_g64_last_token(
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
		if err := LastError(); err != nil {
			return nil, true, err
		}
		return nil, true, core.E("mlx.nativeLastTokenGreedyToken", core.Sprintf("native wrapper failed (rc=%d)", rc), nil)
	}
	return out, true, nil
}

func SuppressTokenArray(ids []int32) *Array {
	if len(ids) == 0 {
		return nil
	}
	return FromValues(ids, len(ids))
}

func NativeLastTokenGreedyTokenAvailable(hidden, normWeight *Array, output *Linear, eps float32) bool {
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
		nativeLastTokenQuantizedOutputBitsAvailable(output.Bits)
}

func nativeLastTokenQuantizedOutputBitsAvailable(bits int) bool {
	switch bits {
	case 4, 6, 8:
		return true
	default:
		return false
	}
}

func nativeMLPGELU(input *Array, mlp *MLP) (*Array, bool, error) {
	if !nativeMLPGELUAvailable(input, mlp) {
		return nil, false, nil
	}
	out := NewArray("FAST_MLP_GELU", input, mlp.GateProj.Weight, mlp.GateProj.Scales, mlp.GateProj.Biases, mlp.UpProj.Weight, mlp.UpProj.Scales, mlp.UpProj.Biases, mlp.DownProj.Weight, mlp.DownProj.Scales, mlp.DownProj.Biases)
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
		if err := LastError(); err != nil {
			return nil, true, err
		}
		return nil, true, core.E("mlx.nativeMLPGELU", core.Sprintf("native wrapper failed (rc=%d)", rc), nil)
	}
	return out, true, nil
}

func nativeMLPGELUAvailable(input *Array, mlp *MLP) bool {
	if !enableNativeMLPGELU {
		return false
	}
	if input == nil || !input.Valid() || mlp == nil {
		return false
	}
	if !NativeMLPLinearAvailable(mlp.GateProj) ||
		!NativeMLPLinearAvailable(mlp.UpProj) ||
		!NativeMLPLinearAvailable(mlp.DownProj) {
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

func NativeMLPLinearAvailable(linear *Linear) bool {
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

func NativeFixedSingleTokenAttention(query, keyCache, valueCache, key, value, offset, mask *Array, scale float32) (*Array, *Array, *Array, bool, error) {
	scaleArray := FromValue(scale)
	defer Free(scaleArray)
	if !nativeFixedSingleTokenAttentionAvailable(query, keyCache, valueCache, key, value, offset, mask) {
		return nil, nil, nil, false, nil
	}
	outInputs := []*Array{query, keyCache, valueCache, key, value, offset, scaleArray}
	hasMask := C.int(0)
	if mask != nil && mask.Valid() {
		outInputs = append(outInputs, mask)
		hasMask = 1
	}
	out := NewArray("FAST_FIXED_SINGLE_TOKEN_ATTENTION", outInputs...)
	newKeys := NewArray("FAST_FIXED_SINGLE_TOKEN_ATTENTION_K", keyCache, key, offset)
	newValues := NewArray("FAST_FIXED_SINGLE_TOKEN_ATTENTION_V", valueCache, value, offset)
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
		CArray(mask),
		hasMask,
		DefaultStream().ctx,
	)
	if rc != 0 {
		Free(out, newKeys, newValues)
		if err := LastError(); err != nil {
			return nil, nil, nil, true, err
		}
		return nil, nil, nil, true, core.E("mlx.NativeFixedSingleTokenAttention", core.Sprintf("native wrapper failed (rc=%d)", rc), nil)
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
	if keyCache.Dim(3) >= 512 && !FixedWideSDPAAttentionEnabled() && !FixedWideMatmulAttentionEnabled() {
		return false
	}
	return query.Dim(3) == keyCache.Dim(3) &&
		key.Dim(3) == keyCache.Dim(3) &&
		value.Dim(3) == valueCache.Dim(3)
}

func NativeFixedSlidingSingleTokenAttention(query, keyCache, valueCache, key, value, shiftIndices, lastIndex *Array, scale float32) (*Array, *Array, *Array, bool, error) {
	if !nativeFixedSlidingSingleTokenAttentionAvailable(query, keyCache, valueCache, key, value, shiftIndices, lastIndex) {
		return nil, nil, nil, false, nil
	}
	scaleArray := FromValue(scale)
	defer Free(scaleArray)
	out := NewArray("FAST_FIXED_SLIDING_ATTENTION_OUT", query, keyCache, valueCache, key, value, scaleArray, shiftIndices, lastIndex)
	newKeys := NewArray("FAST_FIXED_SLIDING_ATTENTION_K", keyCache, key)
	newValues := NewArray("FAST_FIXED_SLIDING_ATTENTION_V", valueCache, value)
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
		if err := LastError(); err != nil {
			return nil, nil, nil, true, err
		}
		return nil, nil, nil, true, core.E("mlx.NativeFixedSlidingSingleTokenAttention", core.Sprintf("native wrapper failed (rc=%d)", rc), nil)
	}
	if !out.Valid() || !newKeys.Valid() || !newValues.Valid() {
		Free(out, newKeys, newValues)
		return nil, nil, nil, true, core.E("mlx.NativeFixedSlidingSingleTokenAttention", "native wrapper returned invalid outputs", nil)
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

func FreeCArrayHandles(handles []C.mlx_array) {
	for _, handle := range handles {
		if handle.ctx != nil {
			C.mlx_array_free(handle)
		}
	}
}

func OutputAt(outs []*Array, i int) *Array {
	if i < 0 || i >= len(outs) {
		return nil
	}
	return outs[i]
}

func ValidateLayerOutputShapes(name string, x, out, newK, newV, prevKeys, prevValues *Array, ownsKV, fixedKV bool) error {
	if !sameArrayShape(out, x) {
		return core.E(name, "returned output shape does not match input hidden shape", nil)
	}
	if !ownsKV {
		return nil
	}
	if newK == nil || newV == nil || prevKeys == nil || prevValues == nil ||
		newK.NumDims() != 4 || newV.NumDims() != 4 || prevKeys.NumDims() != 4 || prevValues.NumDims() != 4 {
		return core.E(name, "returned K/V shape is not rank-4", nil)
	}
	if newK.Dim(0) != prevKeys.Dim(0) || newK.Dim(1) != prevKeys.Dim(1) || newK.Dim(3) != prevKeys.Dim(3) ||
		newV.Dim(0) != prevValues.Dim(0) || newV.Dim(1) != prevValues.Dim(1) || newV.Dim(3) != prevValues.Dim(3) {
		return core.E(name, "returned K/V shape is incompatible with previous cache", nil)
	}
	if fixedKV {
		if newK.Dim(2) != prevKeys.Dim(2) || newV.Dim(2) != prevValues.Dim(2) {
			return core.E(name, "returned fixed K/V cache does not preserve capacity", nil)
		}
		return nil
	}
	if newK.Dim(2) <= 0 || newV.Dim(2) <= 0 {
		return core.E(name, "returned paged K/V cache has empty sequence dimension", nil)
	}
	return nil
}

func sameArrayShape(left, right *Array) bool {
	if left == nil || right == nil || !left.Valid() || !right.Valid() {
		return false
	}
	dims := left.NumDims()
	if dims != right.NumDims() {
		return false
	}
	for i := 0; i < dims; i++ {
		if left.Dim(i) != right.Dim(i) {
			return false
		}
	}
	return true
}
