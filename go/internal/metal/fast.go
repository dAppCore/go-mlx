// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

/*
#include <stdlib.h>
#include "mlx/c/mlx.h"

int go_mlx_gelu_gate_mul(mlx_array* res, const mlx_array gate, const mlx_array up, const mlx_stream stream);
int go_mlx_silu_gate_mul(mlx_array* res, const mlx_array gate, const mlx_array up, const mlx_stream stream);
int go_mlx_native_paged_single_token_attention(mlx_array* res, const mlx_array query, const mlx_array* key_pages, const mlx_array* value_pages, int page_count, float scale, const mlx_stream stream);
*/
import "C"

import (
	"runtime"
	"sync"

	"dappco.re/go"
)

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

// GELUGateMul computes GELU(gate) * up inside the native MLX wrapper.
func GELUGateMul(gate, up *Array) *Array {
	out := newArray("FAST_GELU_GATE_MUL", gate, up)
	rc := C.go_mlx_gelu_gate_mul(&out.ctx, gate.ctx, up.ctx, DefaultStream().ctx)
	if rc != 0 {
		if err := lastError(); err != nil {
			panic(err)
		}
		panic(core.E("mlx.GELUGateMul", core.Sprintf("native wrapper failed (rc=%d)", rc), nil))
	}
	return out
}

// SiLUGateMul computes SiLU(gate) * up inside the native MLX wrapper.
func SiLUGateMul(gate, up *Array) *Array {
	out := newArray("FAST_SILU_GATE_MUL", gate, up)
	rc := C.go_mlx_silu_gate_mul(&out.ctx, gate.ctx, up.ctx, DefaultStream().ctx)
	if rc != 0 {
		if err := lastError(); err != nil {
			panic(err)
		}
		panic(core.E("mlx.SiLUGateMul", core.Sprintf("native wrapper failed (rc=%d)", rc), nil))
	}
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

func RoPEWithOffsetArray(x *Array, dims int, traditional bool, base float32, scale float32, offset *Array, freqs *Array) *Array {
	out := newArray("FAST_ROPE_DYNAMIC", x, offset)
	var cFreqs C.mlx_array
	if freqs != nil {
		cFreqs = freqs.ctx
	}
	C.mlx_fast_rope_dynamic(
		&out.ctx,
		x.ctx,
		C.int(dims),
		C._Bool(traditional),
		C.mlx_optional_float{
			value:     C.float(base),
			has_value: C._Bool(base != 0),
		},
		C.float(scale),
		offset.ctx,
		cFreqs,
		DefaultStream().ctx,
	)
	return out
}

// SDPA mode strings are the only three values ever passed to
// mlx_fast_scaled_dot_product_attention's mask_mode argument:
// "" (default), "causal", and "array". Allocate the corresponding
// C strings once at package load and reuse them for every call —
// the MLX C wrapper copies the string into its op-tree on each
// invocation, so the cached strings are read-only and never
// freed. Drops the per-call C.CString / defer C.free pair that
// every SDPA call was paying out of the decode hot path.
var (
	sdpaModeDefault = C.CString("")
	sdpaModeCausal  = C.CString("causal")
	sdpaModeArray   = C.CString("array")
)

// ScaledDotProductAttention computes attention using a fused Metal kernel.
//
//	out := metal.ScaledDotProductAttention(q, k, v, cfg.Scale, L > 1) // causal when seqLen > 1
func ScaledDotProductAttention(query, key, value *Array, scale float32, causal bool) *Array {
	cMode := sdpaModeDefault
	if causal {
		cMode = sdpaModeCausal
	}

	maskArr := C.mlx_array_new()
	defer C.mlx_array_free(maskArr)
	sinksArr := C.mlx_array_new()
	defer C.mlx_array_free(sinksArr)

	out := newArray("FAST_SDPA", query, key, value)
	C.mlx_fast_scaled_dot_product_attention(&out.ctx, query.ctx, key.ctx, value.ctx, C.float(scale), cMode, maskArr, sinksArr, DefaultStream().ctx)
	return out
}

// ScaledDotProductAttentionPaged computes decode-time attention over K/V pages
// without concatenating the cached K/V tensors. It is intended for non-causal
// single-token decode; prefill and masked paths should use the fused kernels.
// scorePagesPool reuses the per-page score *Array buffer used by
// ScaledDotProductAttentionPaged.  The slice is drained before the call
// returns, so it can go back to the pool without ABA hazards.  This converts
// the 1 alloc / N×8 bytes per multi-page SDPA call (136B/1alloc at 16 pages,
// the SDPAPaged_16Pages residual) into a pool Get/Put amortised across calls.
var scorePagesPool = sync.Pool{
	New: func() any {
		buf := make([]*Array, 0, 16)
		return &buf
	},
}

// nativePagedCtxPool is a sync.Pool of *[]C.mlx_array slices used by
// nativePagedSingleTokenAttention to hand a contiguous run of mlx_array
// handles across the cgo boundary without paying two C.calloc/C.free
// trips per decode step. The native wrapper consumes the buffer
// synchronously, so the slice can be returned to the pool once the cgo
// call returns, the same way W11-T pools scorePages. The 16-capacity
// matches typical PagedKVCache page counts during decode; larger page
// counts grow the backing array and the pool reuses the grown slot.
var nativePagedCtxPool = sync.Pool{
	New: func() any {
		buf := make([]C.mlx_array, 0, 16)
		return &buf
	},
}

func ScaledDotProductAttentionPaged(query *Array, keyPages, valuePages []*Array, scale float32) *Array {
	if len(keyPages) == 0 || len(keyPages) != len(valuePages) {
		return nil
	}
	if len(keyPages) == 1 {
		return ScaledDotProductAttention(query, keyPages[0], valuePages[0], scale, false)
	}

	scorePagesPtr := scorePagesPool.Get().(*[]*Array)
	scorePages := (*scorePagesPtr)[:0]
	if cap(scorePages) < len(keyPages) {
		scorePages = make([]*Array, 0, len(keyPages))
	}
	var globalMax *Array
	for _, key := range keyPages {
		keyT := Transpose4(key, 0, 1, 3, 2)
		score := Matmul(query, keyT)
		Free(keyT)
		if scale != 1 {
			scaled := MulScalar(score, scale)
			Free(score)
			score = scaled
		}
		pageMax := MaxAxis(score, -1, true)
		if globalMax == nil {
			globalMax = pageMax
		} else {
			nextMax := Maximum(globalMax, pageMax)
			Free(globalMax, pageMax)
			globalMax = nextMax
		}
		scorePages = append(scorePages, score)
	}

	var denom *Array
	var weighted *Array
	for i, score := range scorePages {
		shifted := Subtract(score, globalMax)
		expScore := Exp(shifted)
		Free(shifted)
		pageDenom := Sum(expScore, -1, true)
		pageWeighted := Matmul(expScore, valuePages[i])
		Free(expScore)
		if denom == nil {
			denom = pageDenom
			weighted = pageWeighted
			continue
		}
		nextDenom := Add(denom, pageDenom)
		nextWeighted := Add(weighted, pageWeighted)
		Free(denom, pageDenom, weighted, pageWeighted)
		denom = nextDenom
		weighted = nextWeighted
	}
	out := Divide(weighted, denom)
	Free(globalMax, denom, weighted)
	Free(scorePages...)
	// Reset to zero length and return the (possibly grown) slice header to the
	// pool so subsequent calls reuse the same backing array.
	scorePages = scorePages[:0]
	*scorePagesPtr = scorePages
	scorePagesPool.Put(scorePagesPtr)
	return out
}

func nativePagedSingleTokenAttention(query *Array, keyPages, valuePages []*Array, scale float32) (*Array, bool, error) {
	if query == nil || !query.Valid() || len(keyPages) < 2 || len(keyPages) != len(valuePages) {
		return nil, false, nil
	}
	pageCount := len(keyPages)

	// Pooled C-pointer scratch — the native wrapper copies the
	// page-handle run synchronously, so the slice goes back into
	// nativePagedCtxPool once the cgo call returns. This removes
	// the 2 × C.calloc / C.free trips per decode step that the
	// SDPAPaged_*_Pages benches still show as the residual cost.
	keysBufPtr := nativePagedCtxPool.Get().(*[]C.mlx_array)
	valuesBufPtr := nativePagedCtxPool.Get().(*[]C.mlx_array)
	keysBuf := *keysBufPtr
	valuesBuf := *valuesBufPtr
	if cap(keysBuf) < pageCount {
		keysBuf = make([]C.mlx_array, pageCount)
	} else {
		keysBuf = keysBuf[:pageCount]
	}
	if cap(valuesBuf) < pageCount {
		valuesBuf = make([]C.mlx_array, pageCount)
	} else {
		valuesBuf = valuesBuf[:pageCount]
	}
	for i := 0; i < pageCount; i++ {
		if keyPages[i] == nil || valuePages[i] == nil || !keyPages[i].Valid() || !valuePages[i].Valid() {
			// Restore pool slot before bailing out (no growth survives the
			// abort branch beyond the local re-slicing).
			keysBuf = keysBuf[:0]
			valuesBuf = valuesBuf[:0]
			*keysBufPtr = keysBuf
			*valuesBufPtr = valuesBuf
			nativePagedCtxPool.Put(keysBufPtr)
			nativePagedCtxPool.Put(valuesBufPtr)
			return nil, false, nil
		}
		keysBuf[i] = keyPages[i].ctx
		valuesBuf[i] = valuePages[i].ctx
	}

	out := newArray("NATIVE_PAGED_ATTENTION", query)
	rc := C.go_mlx_native_paged_single_token_attention(&out.ctx, query.ctx, &keysBuf[0], &valuesBuf[0], C.int(pageCount), C.float(scale), DefaultStream().ctx)
	runtime.KeepAlive(query)
	runtime.KeepAlive(keyPages)
	runtime.KeepAlive(valuePages)
	runtime.KeepAlive(keysBuf)
	runtime.KeepAlive(valuesBuf)

	// Reset to zero length and put back the (possibly grown) slice so
	// subsequent calls reuse the same backing array.
	keysBuf = keysBuf[:0]
	valuesBuf = valuesBuf[:0]
	*keysBufPtr = keysBuf
	*valuesBufPtr = valuesBuf
	nativePagedCtxPool.Put(keysBufPtr)
	nativePagedCtxPool.Put(valuesBufPtr)

	if rc != 0 {
		Free(out)
		if err := lastError(); err != nil {
			return nil, true, err
		}
		return nil, true, core.NewError("mlx.nativePagedSingleTokenAttention: native wrapper failed")
	}
	return out, true, nil
}

func singleTokenCausalMask(capacity int, offset *Array) *Array {
	idx := Arange(0, float64(capacity), 1, DTypeInt32)
	reshaped := Reshape(idx, 1, 1, 1, int32(capacity))
	valid := lessEqual(reshaped, offset)
	zero := FromValue(float32(0))
	negInf := FromValue(float32(-1e9))
	mask := Where(valid, zero, negInf)
	Free(idx, reshaped, valid, zero, negInf)
	return mask
}

func singleTokenCacheUpdate(cache, token, offset *Array) *Array {
	shape := token.Shape()
	offsetIndex := Reshape(offset, 1, 1, 1, 1)
	indices := BroadcastTo(offsetIndex, shape)
	updated := PutAlongAxis(cache, indices, token, 2)
	Free(offsetIndex, indices)
	return updated
}

func fixedSingleTokenAttention(query, keyCache, valueCache, key, value, offset *Array, scale float32) (*Array, *Array, *Array) {
	updatedKeys := singleTokenCacheUpdate(keyCache, key, offset)
	updatedValues := singleTokenCacheUpdate(valueCache, value, offset)
	mask := singleTokenCausalMask(int(updatedKeys.Dim(2)), offset)
	out := ScaledDotProductAttentionWithMask(query, updatedKeys, updatedValues, mask, scale)
	Free(mask)
	return out, updatedKeys, updatedValues
}

// ScaledDotProductAttentionWithMask computes attention with an explicit mask.
//
//	out := metal.ScaledDotProductAttentionWithMask(q, k, v, batchMask, cfg.Scale)
func ScaledDotProductAttentionWithMask(query, key, value, mask *Array, scale float32) *Array {
	sinksArr := C.mlx_array_new()
	defer C.mlx_array_free(sinksArr)

	out := newArray("FAST_SDPA", query, key, value, mask)
	C.mlx_fast_scaled_dot_product_attention(&out.ctx, query.ctx, key.ctx, value.ctx, C.float(scale), sdpaModeArray, mask.ctx, sinksArr, DefaultStream().ctx)
	return out
}
