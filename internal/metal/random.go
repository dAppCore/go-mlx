//go:build darwin && arm64 && !nomlx

package metal

/*
#include "mlx/c/mlx.h"
*/
import "C"

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
		key,       // null key = use default RNG
		DefaultStream().ctx,
	)
	return out
}

// RandomUniform generates uniform random values in [low, high).
//
//	noise := metal.RandomUniform(0, 1, []int32{batchSize, hiddenSize}, DTypeFloat32)
func RandomUniform(low, high float32, shape []int32, dtype DType) *Array {
	out := newArray("RANDOM_UNIFORM")
	cShape := make([]C.int, len(shape))
	for i, s := range shape {
		cShape[i] = C.int(s)
	}
	lo := FromValue(low)
	hi := FromValue(high)
	key := C.mlx_array_new()
	defer C.mlx_array_free(key)
	C.mlx_random_uniform(
		&out.ctx,
		lo.ctx, hi.ctx,
		&cShape[0], C.size_t(len(cShape)),
		C.mlx_dtype(dtype),
		key,
		DefaultStream().ctx,
	)
	return out
}
