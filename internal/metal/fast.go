//go:build darwin && arm64

package metal

/*
#include <stdlib.h>
#include "mlx/c/mlx.h"
*/
import "C"

import "unsafe"

// RMSNorm applies Root Mean Square normalization using a fused Metal kernel.
//
//	normed := metal.RMSNorm(x, layer.InputNormScaled, 1e-6) // pre-attention normalisation
func RMSNorm(x, weight *Array, eps float32) *Array {
	out := newArray("FAST_RMSNORM", x)
	var cWeight C.mlx_array
	if weight != nil {
		cWeight = weight.ctx
	}
	C.mlx_fast_rms_norm(&out.ctx, x.ctx, cWeight, C.float(eps), DefaultStream().ctx)
	return out
}

// RMSNormNoScale applies RMS normalization without a learnable scale.
func RMSNormNoScale(x *Array, eps float32) *Array {
	return RMSNorm(x, nil, eps)
}

// LayerNorm applies Layer normalization using a fused Metal kernel.
//
//	normed := metal.LayerNorm(x, weight, bias, 1e-5) // standard layer norm with affine params
func LayerNorm(x, weight, bias *Array, eps float32) *Array {
	out := newArray("FAST_LAYERNORM", x)
	C.mlx_fast_layer_norm(&out.ctx, x.ctx, weight.ctx, bias.ctx, C.float(eps), DefaultStream().ctx)
	return out
}

// RoPE applies Rotary Position Embeddings using a fused Metal kernel.
//
//	q = metal.RoPE(q, int(cfg.HeadDim), false, cfg.RopeTheta, 1.0, cache.Offset())
func RoPE(x *Array, dims int, traditional bool, base float32, scale float32, offset int) *Array {
	return RoPEWithFreqs(x, dims, traditional, base, scale, offset, nil)
}

// RoPEWithFreqs applies Rotary Position Embeddings using an explicit frequency tensor.
func RoPEWithFreqs(x *Array, dims int, traditional bool, base float32, scale float32, offset int, freqs *Array) *Array {
	out := newArray("FAST_ROPE", x)
	var cFreqs C.mlx_array
	if freqs != nil {
		cFreqs = freqs.ctx
	}
	C.mlx_fast_rope(
		&out.ctx,
		x.ctx,
		C.int(dims),
		C._Bool(traditional),
		C.mlx_optional_float{
			value:     C.float(base),
			has_value: C._Bool(base != 0),
		},
		C.float(scale),
		C.int(offset),
		cFreqs,
		DefaultStream().ctx,
	)
	return out
}

// ScaledDotProductAttention computes attention using a fused Metal kernel.
//
//	out := metal.ScaledDotProductAttention(q, k, v, cfg.Scale, L > 1) // causal when seqLen > 1
func ScaledDotProductAttention(query, key, value *Array, scale float32, causal bool) *Array {
	mode := ""
	if causal {
		mode = "causal"
	}
	cMode := C.CString(mode)
	defer C.free(unsafe.Pointer(cMode))

	maskArr := C.mlx_array_new()
	defer C.mlx_array_free(maskArr)
	sinksArr := C.mlx_array_new()
	defer C.mlx_array_free(sinksArr)

	out := newArray("FAST_SDPA", query, key, value)
	C.mlx_fast_scaled_dot_product_attention(&out.ctx, query.ctx, key.ctx, value.ctx, C.float(scale), cMode, maskArr, sinksArr, DefaultStream().ctx)
	return out
}

// ScaledDotProductAttentionWithMask computes attention with an explicit mask.
//
//	out := metal.ScaledDotProductAttentionWithMask(q, k, v, batchMask, cfg.Scale)
func ScaledDotProductAttentionWithMask(query, key, value, mask *Array, scale float32) *Array {
	cMode := C.CString("array")
	defer C.free(unsafe.Pointer(cMode))

	sinksArr := C.mlx_array_new()
	defer C.mlx_array_free(sinksArr)

	out := newArray("FAST_SDPA", query, key, value, mask)
	C.mlx_fast_scaled_dot_product_attention(&out.ctx, query.ctx, key.ctx, value.ctx, C.float(scale), cMode, mask.ctx, sinksArr, DefaultStream().ctx)
	return out
}
