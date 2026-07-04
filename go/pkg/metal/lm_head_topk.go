// SPDX-Licence-Identifier: EUPL-1.2

package metal

/*
#include "lm_head_topk_bridge.h"
*/
import "C"

// Q4LMHeadTopKDefaults mirror the tuned MTPLX launch shape: 4 simdgroups x
// 8 subtiles = 128 head rows per tile.
const (
	Q4LMHeadTopKMaxK           = 64
	q4LMHeadTopKSimdgroups     = 4
	q4LMHeadTopKSubtiles       = 8
	q4LMHeadTopKBlockSize      = 512
	q4LMHeadTopKPackedPerInt32 = 8
)

// Q4LMHeadTopKEligible reports whether the fused q4 lm-head top-k kernel can
// serve this head. Gates mirror the kernel's hard requirements: a single
// bf16/f16/f32 row whose width is a positive multiple of 512 (the tile loop
// has no K tail guard), a 4-bit affine head packed [N, K/8] with group_size
// 32/64/128 scales/biases [N, K/group], and 1 <= topK <= 64 <= N.
//
//	if metal.Q4LMHeadTopKEligible(x, w, scales, biases, 64, topK) {
//	    values, indices, err := metal.NativeQ4LMHeadTopK(x, w, scales, biases, 64, topK)
//	    ...
//	}
func Q4LMHeadTopKEligible(x, w, scales, biases *Array, groupSize, topK int) bool {
	if x == nil || w == nil || scales == nil || biases == nil {
		return false
	}
	if !x.Valid() || !w.Valid() || !scales.Valid() || !biases.Valid() {
		return false
	}
	if topK < 1 || topK > Q4LMHeadTopKMaxK {
		return false
	}
	if groupSize != 32 && groupSize != 64 && groupSize != 128 {
		return false
	}
	switch x.Dtype() {
	case DTypeBFloat16, DTypeFloat16, DTypeFloat32:
	default:
		return false
	}
	k := 0
	switch x.NumDims() {
	case 1:
		k = x.Dim(0)
	case 2:
		if x.Dim(0) != 1 {
			return false
		}
		k = x.Dim(1)
	default:
		return false
	}
	if k <= 0 || k%q4LMHeadTopKBlockSize != 0 {
		return false
	}
	if w.NumDims() != 2 || w.Dim(1)*q4LMHeadTopKPackedPerInt32 != k {
		return false
	}
	n := w.Dim(0)
	if n < topK {
		return false
	}
	groups := k / groupSize
	if scales.NumDims() != 2 || scales.Dim(0) != n || scales.Dim(1) != groups {
		return false
	}
	if biases.NumDims() != 2 || biases.Dim(0) != n || biases.Dim(1) != groups {
		return false
	}
	return true
}

// NativeQ4LMHeadTopK computes the fused q4 lm-head top-k: the quantized
// matrix-vector product and the global top-k in one Metal pass plus an
// in-graph tile merge — the full vocab logits row is never materialised.
// Returns values [topK] float32 descending and indices [topK] int32.
// Callers gate with Q4LMHeadTopKEligible first.
//
//	values, indices, err := metal.NativeQ4LMHeadTopK(hidden, head.W, head.Scales, head.Biases, 64, 64)
//	if err == nil { defer Free(values, indices) }
func NativeQ4LMHeadTopK(x, w, scales, biases *Array, groupSize, topK int) (*Array, *Array, error) {
	values := NewArray("Q4_LM_HEAD_TOPK_VALUES", x, w, scales, biases)
	indices := NewArray("Q4_LM_HEAD_TOPK_INDICES", x, w, scales, biases)
	rc := C.go_mlx_q4_lm_head_topk(
		&values.ctx,
		&indices.ctx,
		x.ctx,
		w.ctx,
		scales.ctx,
		biases.ctx,
		C.int(groupSize),
		C.int(topK),
		C.int(q4LMHeadTopKSimdgroups),
		C.int(q4LMHeadTopKSubtiles),
		DefaultStream().ctx,
	)
	if rc != 0 {
		err := LastError()
		Free(values, indices)
		return nil, nil, err
	}
	return values, indices, nil
}
