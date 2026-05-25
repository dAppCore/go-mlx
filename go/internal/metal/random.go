// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

/*
#include "mlx/c/mlx.h"

// mlx_random_uniform_inline narrows the int32 shape into an 8-slot stack
// int buffer on the C side so the Go-side []C.int copy is unnecessary.
// Rank is bounded by maxTensorRank = 8 (ops.go).
static inline int mlx_random_uniform_inline(
    mlx_array* res, mlx_array low, mlx_array high,
    const int32_t* shape_in, size_t shape_num,
    mlx_dtype dtype, mlx_array key, mlx_stream s) {
    int shape_buf[8];
    for (size_t i = 0; i < shape_num; ++i) shape_buf[i] = (int)shape_in[i];
    return mlx_random_uniform(res, low, high, shape_buf, shape_num, dtype, key, s);
}
*/
import "C"

import (
	"unsafe"

	core "dappco.re/go"
)

// SeedRandom resets MLX's default random key sequence.
func SeedRandom(seed uint64) error {
	Init()
	if rc := C.mlx_random_seed(C.uint64_t(seed)); rc != 0 {
		if err := lastError(); err != nil {
			return err
		}
		return core.E("mlx.random.seed", core.Sprintf("seed failed (rc=%d)", rc), nil)
	}
	return nil
}

// RandomCategorical samples from a categorical distribution defined by logprobs.
// Returns indices sampled according to the log-probability distribution along the last axis.
//
//	tokenID := metal.RandomCategorical(scaledLogits) // sample next token
func RandomCategorical(logprobs *Array) *Array {
	out := newArray("RANDOM_CATEGORICAL", logprobs)
	key := C.mlx_array_new()
	defer C.mlx_array_free(key)
	C.mlx_random_categorical(
		&out.ctx,
		logprobs.ctx,
		C.int(-1), // axis
		// The MLX C API also accepts a zero-value key handle for the default
		// RNG, but the retained request-context probe regressed with that
		// shape. Keep the explicit empty key handle on the production path.
		key,
		DefaultStream().ctx,
	)
	return out
}

// RandomUniform generates uniform random values in [low, high).
// Routes through mlx_random_uniform_inline so the per-call shape array is
// stack-allocated on the C side.
//
//	noise := metal.RandomUniform(0, 1, []int32{batchSize, hiddenSize}, DTypeFloat32)
func RandomUniform(low, high float32, shape []int32, dtype DType) *Array {
	if len(shape) > maxTensorRank {
		panic("RandomUniform: rank exceeds maxTensorRank")
	}
	out := newArray("RANDOM_UNIFORM")
	lo := FromValue(low)
	hi := FromValue(high)
	key := C.mlx_array_new()
	defer C.mlx_array_free(key)
	var shapePtr *C.int32_t
	if len(shape) > 0 {
		shapePtr = (*C.int32_t)(unsafe.Pointer(&shape[0]))
	}
	C.mlx_random_uniform_inline(
		&out.ctx,
		lo.ctx, hi.ctx,
		shapePtr, C.size_t(len(shape)),
		C.mlx_dtype(dtype),
		key,
		DefaultStream().ctx,
	)
	return out
}
