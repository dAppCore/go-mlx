// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

/*
#include <stdlib.h>
#include "mlx/c/mlx.h"
*/
import "C"

import "unsafe"

func optionalInt(v int) C.mlx_optional_int {
	return C.mlx_optional_int{
		value:     C.int(v),
		has_value: C._Bool(v > 0),
	}
}

// Add returns element-wise a + b.
func Add(a, b *Array) *Array {
	out := newArray("ADD", a, b)
	C.mlx_add(&out.ctx, a.ctx, b.ctx, DefaultStream().ctx)
	return out
}

// AddScalar returns a + scalar (broadcast).
func AddScalar(a *Array, s float32) *Array {
	scalar := FromValue(s)
	res := Add(a, scalar)
	Free(scalar)
	return res
}

// Mul returns element-wise a * b.
func Mul(a, b *Array) *Array {
	out := newArray("MUL", a, b)
	C.mlx_multiply(&out.ctx, a.ctx, b.ctx, DefaultStream().ctx)
	return out
}

// MulScalar returns a * scalar (broadcast).
func MulScalar(a *Array, s float32) *Array {
	scalar := FromValue(s)
	res := Mul(a, scalar)
	Free(scalar)
	return res
}

// Divide returns element-wise a / b.
func Divide(a, b *Array) *Array {
	out := newArray("DIV", a, b)
	C.mlx_divide(&out.ctx, a.ctx, b.ctx, DefaultStream().ctx)
	return out
}

// Subtract returns element-wise a - b.
func Subtract(a, b *Array) *Array {
	out := newArray("SUB", a, b)
	C.mlx_subtract(&out.ctx, a.ctx, b.ctx, DefaultStream().ctx)
	return out
}

// Negative returns element-wise -a.
func Negative(a *Array) *Array {
	out := newArray("NEG", a)
	C.mlx_negative(&out.ctx, a.ctx, DefaultStream().ctx)
	return out
}

// Copy creates a deep copy of an array, breaking the computation graph chain.
// The returned array has the same data but no references to parent graph nodes,
// allowing Metal memory from prior graph operations to be freed.
//
//	snapshot := metal.Copy(activations) // preserve values, release graph parents
func Copy(a *Array) *Array {
	out := newArray("COPY", a)
	C.mlx_copy(&out.ctx, a.ctx, DefaultStream().ctx)
	return out
}

// Exp returns element-wise exp(a).
func Exp(a *Array) *Array {
	out := newArray("EXP", a)
	C.mlx_exp(&out.ctx, a.ctx, DefaultStream().ctx)
	return out
}

// Sigmoid returns element-wise 1/(1+exp(-a)).
func Sigmoid(a *Array) *Array {
	out := newArray("SIGMOID", a)
	C.mlx_sigmoid(&out.ctx, a.ctx, DefaultStream().ctx)
	return out
}

// SiLU returns element-wise x * sigmoid(x) (Swish activation).
func SiLU(a *Array) *Array {
	s := Sigmoid(a)
	res := Mul(a, s)
	Free(s)
	return res
}

// Tanh returns element-wise tanh(a).
func Tanh(a *Array) *Array {
	out := newArray("TANH", a)
	C.mlx_tanh(&out.ctx, a.ctx, DefaultStream().ctx)
	return out
}

// Sqrt returns element-wise sqrt(a).
func Sqrt(a *Array) *Array {
	out := newArray("SQRT", a)
	C.mlx_sqrt(&out.ctx, a.ctx, DefaultStream().ctx)
	return out
}

// Rsqrt returns element-wise 1/sqrt(a).
func Rsqrt(a *Array) *Array {
	out := newArray("RSQRT", a)
	C.mlx_rsqrt(&out.ctx, a.ctx, DefaultStream().ctx)
	return out
}

// Reciprocal returns element-wise 1/a.
func Reciprocal(a *Array) *Array {
	out := newArray("RECIPROCAL", a)
	C.mlx_reciprocal(&out.ctx, a.ctx, DefaultStream().ctx)
	return out
}

// Square returns element-wise a^2.
func Square(a *Array) *Array {
	out := newArray("SQUARE", a)
	C.mlx_square(&out.ctx, a.ctx, DefaultStream().ctx)
	return out
}

// Power returns element-wise a^b.
func Power(a, b *Array) *Array {
	out := newArray("POWER", a, b)
	C.mlx_power(&out.ctx, a.ctx, b.ctx, DefaultStream().ctx)
	return out
}

// Maximum returns element-wise max(a, b).
func Maximum(a, b *Array) *Array {
	out := newArray("MAX", a, b)
	C.mlx_maximum(&out.ctx, a.ctx, b.ctx, DefaultStream().ctx)
	return out
}

// Minimum returns element-wise min(a, b).
func Minimum(a, b *Array) *Array {
	out := newArray("MIN", a, b)
	C.mlx_minimum(&out.ctx, a.ctx, b.ctx, DefaultStream().ctx)
	return out
}

// Matmul returns the matrix product of a and b.
//
//	out := metal.Matmul(x, wT) // [B, L, hidden] @ [hidden, out] → [B, L, out]
func Matmul(a, b *Array) *Array {
	out := newArray("MATMUL", a, b)
	C.mlx_matmul(&out.ctx, a.ctx, b.ctx, DefaultStream().ctx)
	return out
}

// Conv2d performs a 2D convolution using MLX's NHWC input layout and
// [out_channels, kernel_h, kernel_w, in_channels] weight layout.
func Conv2d(input, weight *Array, strideH, strideW, padH, padW, dilationH, dilationW, groups int) *Array {
	out := newArray("CONV2D", input, weight)
	C.mlx_conv2d(
		&out.ctx,
		input.ctx,
		weight.ctx,
		C.int(strideH),
		C.int(strideW),
		C.int(padH),
		C.int(padW),
		C.int(dilationH),
		C.int(dilationW),
		C.int(groups),
		DefaultStream().ctx,
	)
	return out
}

// QuantizedMatmul performs quantized matrix multiplication.
func QuantizedMatmul(x, w, scales, biases *Array, transpose bool, groupSize, bits int) *Array {
	out := newArray("QMATMUL", x, w, scales, biases)
	gs := optionalInt(groupSize)
	b := optionalInt(bits)
	mode := C.CString("affine")
	defer C.free(unsafe.Pointer(mode))
	C.mlx_quantized_matmul(
		&out.ctx, x.ctx, w.ctx, scales.ctx, biases.ctx,
		C._Bool(transpose), gs, b, mode,
		DefaultStream().ctx,
	)
	return out
}

// GatherMM performs expert-indexed matrix multiplication.
func GatherMM(a, b, lhsIndices, rhsIndices *Array, sorted bool) *Array {
	out := newArray("GATHER_MM", a, b, lhsIndices, rhsIndices)
	var cLHS, cRHS C.mlx_array
	if lhsIndices != nil {
		cLHS = lhsIndices.ctx
	}
	if rhsIndices != nil {
		cRHS = rhsIndices.ctx
	}
	C.mlx_gather_mm(&out.ctx, a.ctx, b.ctx, cLHS, cRHS, C._Bool(sorted), DefaultStream().ctx)
	return out
}

// GatherQMM performs expert-indexed quantized matrix multiplication.
func GatherQMM(x, w, scales, biases, lhsIndices, rhsIndices *Array, transpose bool, groupSize, bits int, mode string, sorted bool) *Array {
	out := newArray("GATHER_QMM", x, w, scales, biases, lhsIndices, rhsIndices)
	gs := optionalInt(groupSize)
	b := optionalInt(bits)
	cMode := C.CString(mode)
	defer C.free(unsafe.Pointer(cMode))

	var cBiases, cLHS, cRHS C.mlx_array
	if biases != nil {
		cBiases = biases.ctx
	}
	if lhsIndices != nil {
		cLHS = lhsIndices.ctx
	}
	if rhsIndices != nil {
		cRHS = rhsIndices.ctx
	}
	C.mlx_gather_qmm(
		&out.ctx,
		x.ctx,
		w.ctx,
		scales.ctx,
		cBiases,
		cLHS,
		cRHS,
		C._Bool(transpose),
		gs,
		b,
		cMode,
		C._Bool(sorted),
		DefaultStream().ctx,
	)
	return out
}

// Softmax returns softmax along the last axis.
//
//	probs := metal.Softmax(logits) // convert raw logits to probability distribution
func Softmax(a *Array) *Array {
	out := newArray("SOFTMAX", a)
	axis := []C.int{C.int(-1)}
	C.mlx_softmax_axes(&out.ctx, a.ctx, &axis[0], C.size_t(1), C._Bool(false), DefaultStream().ctx)
	return out
}

// Argmax returns the index of the maximum value along an axis.
//
//	tokenID := metal.Argmax(logits, -1, false) // greedy decoding: pick most likely token
func Argmax(a *Array, axis int, keepDims bool) *Array {
	out := newArray("ARGMAX", a)
	C.mlx_argmax_axis(&out.ctx, a.ctx, C.int(axis), C._Bool(keepDims), DefaultStream().ctx)
	return out
}

// TopK returns the top k values along the last axis.
func TopK(a *Array, k int) *Array {
	out := newArray("TOPK", a)
	C.mlx_topk_axis(&out.ctx, a.ctx, C.int(k), C.int(-1), DefaultStream().ctx)
	return out
}

// Sum reduces by summation along the given axis.
func Sum(a *Array, axis int, keepDims bool) *Array {
	out := newArray("SUM", a)
	axes := []C.int{C.int(axis)}
	C.mlx_sum_axes(&out.ctx, a.ctx, &axes[0], C.size_t(1), C._Bool(keepDims), DefaultStream().ctx)
	return out
}

// Mean reduces by averaging along the given axis.
func Mean(a *Array, axis int, keepDims bool) *Array {
	out := newArray("MEAN", a)
	axes := []C.int{C.int(axis)}
	C.mlx_mean_axes(&out.ctx, a.ctx, &axes[0], C.size_t(1), C._Bool(keepDims), DefaultStream().ctx)
	return out
}

// Reshape changes the shape of an array.
//
//	input := metal.Reshape(tokens, 1, int32(len(tokens))) // add batch dim: [L] → [1, L]
func Reshape(a *Array, shape ...int32) *Array {
	out := newArray("RESHAPE", a)
	cShape := make([]C.int, len(shape))
	for i, s := range shape {
		cShape[i] = C.int(s)
	}
	C.mlx_reshape(&out.ctx, a.ctx, &cShape[0], C.size_t(len(cShape)), DefaultStream().ctx)
	return out
}

// Transpose permutes dimensions. If no axes given, reverses all dims.
func Transpose(a *Array, axes ...int) *Array {
	out := newArray("TRANSPOSE", a)
	if len(axes) == 0 {
		C.mlx_transpose(&out.ctx, a.ctx, DefaultStream().ctx)
	} else {
		cAxes := make([]C.int, len(axes))
		for i, ax := range axes {
			cAxes[i] = C.int(ax)
		}
		C.mlx_transpose_axes(&out.ctx, a.ctx, &cAxes[0], C.size_t(len(cAxes)), DefaultStream().ctx)
	}
	return out
}

// ExpandDims inserts a new axis at the given position.
func ExpandDims(a *Array, axis int) *Array {
	out := newArray("EXPAND_DIMS", a)
	C.mlx_expand_dims(&out.ctx, a.ctx, C.int(axis), DefaultStream().ctx)
	return out
}

// Squeeze removes dimensions of size 1.
func Squeeze(a *Array, axes ...int) *Array {
	out := newArray("SQUEEZE", a)
	cAxes := make([]C.int, len(axes))
	for i, ax := range axes {
		cAxes[i] = C.int(ax)
	}
	C.mlx_squeeze_axes(&out.ctx, a.ctx, &cAxes[0], C.size_t(len(cAxes)), DefaultStream().ctx)
	return out
}

// Concatenate joins arrays along the given axis.
func Concatenate(arrays []*Array, axis int) *Array {
	vector := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(vector)

	inputs := make([]*Array, len(arrays))
	for i, a := range arrays {
		C.mlx_vector_array_append_value(vector, a.ctx)
		inputs[i] = a
	}

	out := newArray("CONCAT", inputs...)
	C.mlx_concatenate_axis(&out.ctx, vector, C.int(axis), DefaultStream().ctx)
	return out
}

// BroadcastTo broadcasts an array to the given shape.
func BroadcastTo(a *Array, shape []int32) *Array {
	out := newArray("BROADCAST", a)
	cShape := make([]C.int, len(shape))
	for i, s := range shape {
		cShape[i] = C.int(s)
	}
	C.mlx_broadcast_to(&out.ctx, a.ctx, &cShape[0], C.size_t(len(cShape)), DefaultStream().ctx)
	return out
}

// AsType casts an array to a different dtype.
func AsType(a *Array, dtype DType) *Array {
	out := newArray("ASTYPE", a)
	C.mlx_astype(&out.ctx, a.ctx, C.mlx_dtype(dtype), DefaultStream().ctx)
	return out
}

// AsStrided creates a view with custom strides.
func AsStrided(a *Array, shape []int32, strides []int64, offset int64) *Array {
	out := newArray("AS_STRIDED", a)
	cShape := make([]C.int, len(shape))
	for i, s := range shape {
		cShape[i] = C.int(s)
	}
	cStrides := make([]C.int64_t, len(strides))
	for i, s := range strides {
		cStrides[i] = C.int64_t(s)
	}
	C.mlx_as_strided(&out.ctx, a.ctx, &cShape[0], C.size_t(len(cShape)), &cStrides[0], C.size_t(len(cStrides)), C.size_t(offset), DefaultStream().ctx)
	return out
}

// Take gathers elements from a along axis using indices.
func Take(a, indices *Array, axis int) *Array {
	out := newArray("TAKE", a, indices)
	C.mlx_take_axis(&out.ctx, a.ctx, indices.ctx, C.int(axis), DefaultStream().ctx)
	return out
}

// Where selects elements from a or b based on condition.
func Where(condition, a, b *Array) *Array {
	out := newArray("WHERE", condition, a, b)
	C.mlx_where(&out.ctx, condition.ctx, a.ctx, b.ctx, DefaultStream().ctx)
	return out
}

// Argpartition partially sorts and returns indices for top-k selection.
func Argpartition(a *Array, kth, axis int) *Array {
	out := newArray("ARGPARTITION", a)
	C.mlx_argpartition_axis(&out.ctx, a.ctx, C.int(kth), C.int(axis), DefaultStream().ctx)
	return out
}

// Dequantize restores a quantized array to full precision.
//
//	fullW := metal.Dequantize(w, scales, biases, 64, 4) // 4-bit weights, group=64
func Dequantize(w, scales, biases *Array, groupSize, bits int) *Array {
	out := newArray("DEQUANTIZE", w, scales, biases)
	gs := optionalInt(groupSize)
	b := optionalInt(bits)
	mode := C.CString("affine")
	defer C.free(unsafe.Pointer(mode))
	noDtype := C.mlx_optional_dtype{has_value: C._Bool(false)}
	C.mlx_dequantize(&out.ctx, w.ctx, scales.ctx, biases.ctx, gs, b, mode, noDtype, DefaultStream().ctx)
	return out
}

// PutAlongAxis places values into array at indices along axis.
func PutAlongAxis(a, indices, values *Array, axis int) *Array {
	out := newArray("PUT_ALONG_AXIS", a, indices, values)
	// Use scatter approach: src[indices] = values
	C.mlx_put_along_axis(&out.ctx, a.ctx, indices.ctx, values.ctx, C.int(axis), DefaultStream().ctx)
	return out
}

// TakeAlongAxis gathers elements from a along axis using indices.
// Unlike Take, this uses the same number of dimensions for indices and input.
func TakeAlongAxis(a, indices *Array, axis int) *Array {
	out := newArray("TAKE_ALONG_AXIS", a, indices)
	C.mlx_take_along_axis(&out.ctx, a.ctx, indices.ctx, C.int(axis), DefaultStream().ctx)
	return out
}

// LogSumExp computes log(sum(exp(a))) along the given axis.
// Numerically stable reduction for cross-entropy loss.
func LogSumExp(a *Array, axis int, keepDims bool) *Array {
	out := newArray("LOGSUMEXP", a)
	C.mlx_logsumexp_axis(&out.ctx, a.ctx, C.int(axis), C._Bool(keepDims), DefaultStream().ctx)
	return out
}

// CumSum returns the cumulative sum along the given axis.
// reverse=false for forward, inclusive=true to include the current element.
func CumSum(a *Array, axis int, reverse, inclusive bool) *Array {
	out := newArray("CUMSUM", a)
	C.mlx_cumsum(&out.ctx, a.ctx, C.int(axis), C._Bool(reverse), C._Bool(inclusive), DefaultStream().ctx)
	return out
}

// Sort returns the array sorted along the given axis.
//
//	sortedProbs := metal.Sort(probs, -1) // sort probability distribution ascending
func Sort(a *Array, axis int) *Array {
	out := newArray("SORT", a)
	C.mlx_sort_axis(&out.ctx, a.ctx, C.int(axis), DefaultStream().ctx)
	return out
}

// Argsort returns the indices that would sort the array along the given axis.
//
//	sortIdx := metal.Argsort(negProbs, -1) // descending sort for top-p nucleus sampling
func Argsort(a *Array, axis int) *Array {
	out := newArray("ARGSORT", a)
	C.mlx_argsort_axis(&out.ctx, a.ctx, C.int(axis), DefaultStream().ctx)
	return out
}

// Greater returns element-wise a > b as a bool array.
func Greater(a, b *Array) *Array {
	out := newArray("GREATER", a, b)
	C.mlx_greater(&out.ctx, a.ctx, b.ctx, DefaultStream().ctx)
	return out
}

// MaxAxis returns the maximum value along the given axis.
func MaxAxis(a *Array, axis int, keepDims bool) *Array {
	out := newArray("MAX_AXIS", a)
	C.mlx_max_axis(&out.ctx, a.ctx, C.int(axis), C._Bool(keepDims), DefaultStream().ctx)
	return out
}

// Any reduces with logical OR over all elements. Returns a scalar bool array.
// Set keepDims to preserve the reduced dimension as size 1.
//
//	hasTrues := metal.Any(mask, false) // check if any element is true
func Any(a *Array, keepDims bool) *Array {
	out := newArray("ANY", a)
	C.mlx_any(&out.ctx, a.ctx, C._Bool(keepDims), DefaultStream().ctx)
	return out
}

// AnyAxis reduces with logical OR along the given axis.
//
//	rowHasTrue := metal.AnyAxis(mask, 1, false) // per-row OR reduction
func AnyAxis(a *Array, axis int, keepDims bool) *Array {
	out := newArray("ANY_AXIS", a)
	C.mlx_any_axis(&out.ctx, a.ctx, C.int(axis), C._Bool(keepDims), DefaultStream().ctx)
	return out
}

// Arange creates a 1-D array with evenly spaced values in [start, stop) with the given step.
// Similar to numpy.arange.
//
//	indices := metal.Arange(0, 10, 1, DTypeInt32)   // [0, 1, 2, ..., 9]
//	halves  := metal.Arange(0, 3, 0.5, DTypeFloat32) // [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
func Arange(start, stop, step float64, dtype DType) *Array {
	Init()
	out := newArray("ARANGE")
	C.mlx_arange(&out.ctx, C.double(start), C.double(stop), C.double(step), C.mlx_dtype(dtype), DefaultStream().ctx)
	return out
}

// IsNaN returns a boolean array indicating which elements are NaN.
//
//	nanMask := metal.IsNaN(logits) // detect NaN values before sampling
func IsNaN(a *Array) *Array {
	out := newArray("ISNAN", a)
	C.mlx_isnan(&out.ctx, a.ctx, DefaultStream().ctx)
	return out
}
