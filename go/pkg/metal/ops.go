// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

/*
#include <stdlib.h>
#include "mlx/c/mlx.h"

// mlx_as_strided_inline materialises the cgo shape + strides arrays inside
// the C frame so callers can pass int32 / int64 values directly without
// allocating Go-side []C.int / []C.int64_t backing arrays.  MLX caps tensor
// rank at 8, and the metal model code tops out at rank 5 (Gemma 4 vision);
// fixed-arity 8-slot C stack arrays cover both with headroom and avoid the
// per-call cgo pointer-checker forcing the backing slice onto the Go heap.
static inline int mlx_as_strided_inline(
    mlx_array* res, mlx_array a,
    const int32_t* shape_in, size_t shape_num,
    const int64_t* strides_in, size_t strides_num,
    size_t offset, mlx_stream s) {
    int shape_buf[8];
    int64_t strides_buf[8];
    for (size_t i = 0; i < shape_num; ++i) shape_buf[i] = (int)shape_in[i];
    for (size_t i = 0; i < strides_num; ++i) strides_buf[i] = strides_in[i];
    return mlx_as_strided(res, a, shape_buf, shape_num, strides_buf, strides_num, offset, s);
}

// mlx_reshape_inline / mlx_broadcast_to_inline / mlx_transpose_axes_inline /
// mlx_squeeze_axes_inline / mlx_sum_axes_inline / mlx_mean_axes_inline /
// mlx_softmax_axes_inline take a single int32 (or int) array and copy into
// a 8-slot stack buffer before forwarding to MLX, eliminating the per-call
// Go heap alloc for the cgo int array.
static inline int mlx_reshape_inline(
    mlx_array* res, mlx_array a,
    const int32_t* shape_in, size_t shape_num,
    mlx_stream s) {
    int shape_buf[8];
    for (size_t i = 0; i < shape_num; ++i) shape_buf[i] = (int)shape_in[i];
    return mlx_reshape(res, a, shape_buf, shape_num, s);
}

static inline int mlx_broadcast_to_inline(
    mlx_array* res, mlx_array a,
    const int32_t* shape_in, size_t shape_num,
    mlx_stream s) {
    int shape_buf[8];
    for (size_t i = 0; i < shape_num; ++i) shape_buf[i] = (int)shape_in[i];
    return mlx_broadcast_to(res, a, shape_buf, shape_num, s);
}

// mlx_transpose_axes_inline / mlx_squeeze_axes_inline accept a pointer to the
// caller's int64 slice (Go's `int` on darwin/arm64) and narrow into a stack
// int buffer on the C side.  Lets Transpose([]int) / Squeeze([]int) stay
// alloc-free while still using a single inline wrapper per call.
static inline int mlx_transpose_axes_inline(
    mlx_array* res, mlx_array a,
    const long long* axes_in, size_t axes_num,
    mlx_stream s) {
    int axes_buf[8];
    for (size_t i = 0; i < axes_num; ++i) axes_buf[i] = (int)axes_in[i];
    return mlx_transpose_axes(res, a, axes_buf, axes_num, s);
}

static inline int mlx_squeeze_axes_inline(
    mlx_array* res, mlx_array a,
    const long long* axes_in, size_t axes_num,
    mlx_stream s) {
    int axes_buf[8];
    for (size_t i = 0; i < axes_num; ++i) axes_buf[i] = (int)axes_in[i];
    return mlx_squeeze_axes(res, a, axes_buf, axes_num, s);
}

// mlx_transpose_axes_inline_4 is the rank-4 scalar-pass form — eliminates the
// Go-side `[]int` materialisation of the variadic axes parameter. Used by
// the attention paths (Transpose(k, 0,1,3,2) appears in SDPAPaged and the
// model attention kernels). 4 axes register-passed; C stack-materialises.
static inline int mlx_transpose_axes_inline_4(
    mlx_array* res, mlx_array a,
    int a0, int a1, int a2, int a3,
    mlx_stream s) {
    int axes_buf[4] = {a0, a1, a2, a3};
    return mlx_transpose_axes(res, a, axes_buf, 4, s);
}

// mlx_reshape_inline_1 / mlx_reshape_inline_2 / mlx_reshape_inline_3 are the rank-1 / rank-2 / rank-3
// scalar-pass forms of mlx_reshape_inline — completes the W11-AC
// Reshape/Slice rank-1/2/3 scalar-pass family alongside Reshape and the
// existing slice rank-4 variants. The Q4 quantise/dequantise paths
// (packQ4Cached, unpackQ4, maxAll) currently call
// `Reshape(arr, int32(n))` or `Reshape(arr, int32(pairs), int32(2))`
// where the variadic []int32 escapes to heap on every call. Passing the
// 1, 2, or 3 register-passed scalars directly to MLX eliminates the slice
// literal entirely. Same W10-J / W11-A pattern, lower rank.
static inline int mlx_reshape_inline_1(
    mlx_array* res, mlx_array a,
    int32_t n,
    mlx_stream s) {
    int shape_buf[1] = {(int)n};
    return mlx_reshape(res, a, shape_buf, 1, s);
}

static inline int mlx_reshape_inline_2(
    mlx_array* res, mlx_array a,
    int32_t h, int32_t w,
    mlx_stream s) {
    int shape_buf[2] = {(int)h, (int)w};
    return mlx_reshape(res, a, shape_buf, 2, s);
}

static inline int mlx_reshape_inline_3(
    mlx_array* res, mlx_array a,
    int32_t d0, int32_t d1, int32_t d2,
    mlx_stream s) {
    int shape_buf[3] = {(int)d0, (int)d1, (int)d2};
    return mlx_reshape(res, a, shape_buf, 3, s);
}

// mlx_*_single_axis_inline materialise the single-element axis array on the
// C stack so the per-call Go side stops allocating a 1-int slice.  Sum /
// Mean each take a single int axis from the Go API; Softmax pins axis = -1
// (last dim).  Used on the sampler / loss / reduction hot paths.
static inline int mlx_softmax_single_axis_inline(
    mlx_array* res, mlx_array a, int axis, bool precise, mlx_stream s) {
    int axes_buf[1] = { axis };
    return mlx_softmax_axes(res, a, axes_buf, 1, precise, s);
}

static inline int mlx_sum_single_axis_inline(
    mlx_array* res, mlx_array a, int axis, bool keepdims, mlx_stream s) {
    int axes_buf[1] = { axis };
    return mlx_sum_axes(res, a, axes_buf, 1, keepdims, s);
}

static inline int mlx_mean_single_axis_inline(
    mlx_array* res, mlx_array a, int axis, bool keepdims, mlx_stream s) {
    int axes_buf[1] = { axis };
    return mlx_mean_axes(res, a, axes_buf, 1, keepdims, s);
}

// mlx_add_scalar_inline / mlx_multiply_scalar_inline collapse the
// FromValue(s) + Add/Mul(a, scalar) + Free(scalar) sequence used by the
// Go-side AddScalar / MulScalar into a single cgo crossing.  MLX does not
// expose mlx_add_scalar / mlx_multiply_scalar primitives, so the scalar
// mlx_array is created on the C frame, fed into the binary op, and freed
// before return.  Net effect: 3 cgo crossings + 1 Go *Array wrapper for
// the scalar collapse into 1 cgo crossing and 0 extra Go allocs.  Used by
// every model file that scales / shifts / softcaps an activation tensor
// (gemma3/4 attention scale, embedding scale, router scale, RoPE rescale,
// gemma4_vision pixel rescale, LoRA delta scale, etc).
// mlx_scalar_like builds the scalar at a's floating dtype — MLX python's
// weak-scalar promotion (h * 2.0 keeps h.dtype), which mlx_array_new_float32
// breaks: a strong float32 scalar upcasts a bf16/fp16 activation stream to
// float32 at every scale/shift, doubling activation bytes through the whole
// forward. Half conversions are host-side (arm64 native __fp16; bf16 via
// round-to-nearest-even truncation). Non-float inputs keep the float32
// scalar, preserving integer promotion.
static inline mlx_array mlx_scalar_like(mlx_array a, float scalar) {
    mlx_dtype dt = mlx_array_dtype(a);
    if (dt == MLX_FLOAT16) {
        __fp16 h = (__fp16)scalar;
        return mlx_array_new_data(&h, NULL, 0, MLX_FLOAT16);
    }
    if (dt == MLX_BFLOAT16) {
        union { float f; unsigned int u; } c;
        c.f = scalar;
        unsigned int lsb = (c.u >> 16) & 1u;
        unsigned short b = (unsigned short)((c.u + 0x7FFFu + lsb) >> 16);
        return mlx_array_new_data(&b, NULL, 0, MLX_BFLOAT16);
    }
    return mlx_array_new_float32(scalar);
}

static inline int mlx_add_scalar_inline(
    mlx_array* res, mlx_array a, float scalar, mlx_stream s) {
    mlx_array sc = mlx_scalar_like(a, scalar);
    int rc = mlx_add(res, a, sc, s);
    mlx_array_free(sc);
    return rc;
}

static inline int mlx_multiply_scalar_inline(
    mlx_array* res, mlx_array a, float scalar, mlx_stream s) {
    mlx_array sc = mlx_scalar_like(a, scalar);
    int rc = mlx_multiply(res, a, sc, s);
    mlx_array_free(sc);
    return rc;
}

// mlx_greater_scalar_inline collapses FromValue(scalar) + Greater(a, scalar)
// + Free(scalar) into a single cgo crossing — used by the sampler hot path
// (TopP threshold compare, MinP threshold compare) where the right-hand side
// of Greater is a per-call float32 constant.
static inline int mlx_greater_scalar_inline(
    mlx_array* res, mlx_array a, float scalar, mlx_stream s) {
    mlx_array sc = mlx_array_new_float32(scalar);
    int rc = mlx_greater(res, a, sc, s);
    mlx_array_free(sc);
    return rc;
}

// mlx_scalar_greater_inline = scalar > a (reversed operand order).  Used by
// MinPSampler.Sample where the scalar threshold is the left-hand side of the
// comparison.  Same single-cgo-crossing rationale as greater_scalar.
static inline int mlx_scalar_greater_inline(
    mlx_array* res, mlx_array a, float scalar, mlx_stream s) {
    mlx_array sc = mlx_array_new_float32(scalar);
    int rc = mlx_greater(res, sc, a, s);
    mlx_array_free(sc);
    return rc;
}

// mlx_subtract_scalar_inline = a - scalar — broadcast subtract of a per-call
// constant.  Currently unused but the symmetric of add_scalar; lands here so
// TopP-style "shift then compare" idioms stay one-call.
static inline int mlx_subtract_scalar_inline(
    mlx_array* res, mlx_array a, float scalar, mlx_stream s) {
    mlx_array sc = mlx_array_new_float32(scalar);
    int rc = mlx_subtract(res, a, sc, s);
    mlx_array_free(sc);
    return rc;
}

// mlx_where_scalar_scalar_inline = where(condition, a_scalar, b_scalar) —
// collapses the FromValue+FromValue+Where+Free×2 sequence used by TopP /
// TopKSampler masking ("set to -inf where excluded, else 0") into a single
// cgo crossing.  Both scalars are materialised on the C frame.
static inline int mlx_where_scalar_scalar_inline(
    mlx_array* res, mlx_array cond, float a_scalar, float b_scalar, mlx_stream s) {
    mlx_array a_sc = mlx_array_new_float32(a_scalar);
    mlx_array b_sc = mlx_array_new_float32(b_scalar);
    int rc = mlx_where(res, cond, a_sc, b_sc, s);
    mlx_array_free(a_sc);
    mlx_array_free(b_sc);
    return rc;
}

// mlx_where_scalar_array_inline = where(condition, a_scalar, b) — collapses
// FromValue(a_scalar) + Where + Free(a_scalar) for the "mask with constant,
// pass-through otherwise" idiom used by the final TopP / MinP mask-apply
// step ("set to -inf where excluded, original logit otherwise").
static inline int mlx_where_scalar_array_inline(
    mlx_array* res, mlx_array cond, float a_scalar, mlx_array b, mlx_stream s) {
    mlx_array a_sc = mlx_array_new_float32(a_scalar);
    int rc = mlx_where(res, cond, a_sc, b, s);
    mlx_array_free(a_sc);
    return rc;
}

// mlx_concatenate_axis_2 builds the temporary MLX vector on the C stack for the
// common two-array concat path. Multi-page concat keeps the append-vector path:
// passing a Go handle array into C makes it escape and regresses Go heap use.
static inline int mlx_concatenate_axis_2(
    mlx_array* res,
    mlx_array left,
    mlx_array right,
    int axis,
    mlx_stream s) {
    mlx_array arrays[2] = {left, right};
    mlx_vector_array vector = mlx_vector_array_new_data(arrays, 2);
    int rc = mlx_concatenate_axis(res, vector, axis, s);
    int free_rc = mlx_vector_array_free(vector);
    return rc != 0 ? rc : free_rc;
}

*/
import "C"

import "unsafe"

// MaxTensorRank is the largest tensor rank supported by MLX (and by the model
// code in this package — Gemma 4 vision tops out at rank 5, Gemma 4 text +
// Qwen 3 + Llama 3 attention top out at rank 4).  Sized at 8 to provide
// headroom for future ops while still fitting comfortably on a goroutine
// stack frame, so per-call cgo int arrays can be materialised inline rather
// than allocated on the heap.
const MaxTensorRank = 8

func optionalInt(v int) C.mlx_optional_int {
	return C.mlx_optional_int{
		value:     C.int(v),
		has_value: C._Bool(v > 0),
	}
}

func optionalArray(a *Array) C.mlx_array {
	if a == nil || !a.Valid() {
		return C.mlx_array{}
	}
	return a.ctx
}

// Add returns element-wise a + b.
func Add(a, b *Array) *Array {
	out := NewArray("ADD", a, b)
	C.mlx_add(&out.ctx, a.ctx, b.ctx, DefaultStream().ctx)
	return out
}

// AddScalar returns a + scalar (broadcast).
//
// Routes through the mlx_add_scalar_inline bridge so the scalar mlx_array
// is materialised on the C stack — single cgo crossing covers scalar
// creation + binary op + scalar release.  Avoids the legacy FromValue +
// Add + Free triple-crossing.
func AddScalar(a *Array, s float32) *Array {
	out := NewArray("ADD_SCALAR", a)
	C.mlx_add_scalar_inline(&out.ctx, a.ctx, C.float(s), DefaultStream().ctx)
	return out
}

// Mul returns element-wise a * b.
func Mul(a, b *Array) *Array {
	out := NewArray("MUL", a, b)
	C.mlx_multiply(&out.ctx, a.ctx, b.ctx, DefaultStream().ctx)
	return out
}

// MulScalar returns a * scalar (broadcast).
//
// Routes through the mlx_multiply_scalar_inline bridge so the scalar
// mlx_array is materialised on the C stack — single cgo crossing covers
// scalar creation + binary op + scalar release.  Avoids the legacy
// FromValue + Mul + Free triple-crossing.
func MulScalar(a *Array, s float32) *Array {
	out := NewArray("MUL_SCALAR", a)
	C.mlx_multiply_scalar_inline(&out.ctx, a.ctx, C.float(s), DefaultStream().ctx)
	return out
}

// Divide returns element-wise a / b.
func Divide(a, b *Array) *Array {
	out := NewArray("DIV", a, b)
	C.mlx_divide(&out.ctx, a.ctx, b.ctx, DefaultStream().ctx)
	return out
}

func FloorDivide(a, b *Array) *Array {
	out := NewArray("FLOOR_DIVIDE", a, b)
	C.mlx_floor_divide(&out.ctx, a.ctx, b.ctx, DefaultStream().ctx)
	return out
}

// Subtract returns element-wise a - b.
func Subtract(a, b *Array) *Array {
	out := NewArray("SUB", a, b)
	C.mlx_subtract(&out.ctx, a.ctx, b.ctx, DefaultStream().ctx)
	return out
}

// Negative returns element-wise -a.
func Negative(a *Array) *Array {
	out := NewArray("NEG", a)
	C.mlx_negative(&out.ctx, a.ctx, DefaultStream().ctx)
	return out
}

// Abs returns element-wise absolute value.
func Abs(a *Array) *Array {
	out := NewArray("ABS", a)
	C.mlx_abs(&out.ctx, a.ctx, DefaultStream().ctx)
	return out
}

// Copy creates a deep copy of an array, breaking the computation graph chain.
// The returned array has the same data but no references to parent graph nodes,
// allowing Metal memory from prior graph operations to be freed.
//
//	snapshot := metal.Copy(activations) // preserve values, release graph parents
func Copy(a *Array) *Array {
	out := NewArray("COPY", a)
	C.mlx_copy(&out.ctx, a.ctx, DefaultStream().ctx)
	return out
}

// Exp returns element-wise exp(a).
func Exp(a *Array) *Array {
	out := NewArray("EXP", a)
	C.mlx_exp(&out.ctx, a.ctx, DefaultStream().ctx)
	return out
}

// Sigmoid returns element-wise 1/(1+exp(-a)).
func Sigmoid(a *Array) *Array {
	out := NewArray("SIGMOID", a)
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
	out := NewArray("TANH", a)
	C.mlx_tanh(&out.ctx, a.ctx, DefaultStream().ctx)
	return out
}

// Sqrt returns element-wise sqrt(a).
func Sqrt(a *Array) *Array {
	out := NewArray("SQRT", a)
	C.mlx_sqrt(&out.ctx, a.ctx, DefaultStream().ctx)
	return out
}

// Rsqrt returns element-wise 1/sqrt(a).
func Rsqrt(a *Array) *Array {
	out := NewArray("RSQRT", a)
	C.mlx_rsqrt(&out.ctx, a.ctx, DefaultStream().ctx)
	return out
}

// Reciprocal returns element-wise 1/a.
func Reciprocal(a *Array) *Array {
	out := NewArray("RECIPROCAL", a)
	C.mlx_reciprocal(&out.ctx, a.ctx, DefaultStream().ctx)
	return out
}

// Square returns element-wise a^2.
func Square(a *Array) *Array {
	out := NewArray("SQUARE", a)
	C.mlx_square(&out.ctx, a.ctx, DefaultStream().ctx)
	return out
}

// Power returns element-wise a^b.
func Power(a, b *Array) *Array {
	out := NewArray("POWER", a, b)
	C.mlx_power(&out.ctx, a.ctx, b.ctx, DefaultStream().ctx)
	return out
}

// Maximum returns element-wise max(a, b).
func Maximum(a, b *Array) *Array {
	out := NewArray("MAX", a, b)
	C.mlx_maximum(&out.ctx, a.ctx, b.ctx, DefaultStream().ctx)
	return out
}

// Minimum returns element-wise min(a, b).
func Minimum(a, b *Array) *Array {
	out := NewArray("MIN", a, b)
	C.mlx_minimum(&out.ctx, a.ctx, b.ctx, DefaultStream().ctx)
	return out
}

// Clip clamps values to the supplied min/max arrays. Nil leaves a bound open.
func Clip(a, minValue, maxValue *Array) *Array {
	out := NewArray("CLIP", a, minValue, maxValue)
	var cMin, cMax C.mlx_array
	if minValue != nil {
		cMin = minValue.ctx
	}
	if maxValue != nil {
		cMax = maxValue.ctx
	}
	C.mlx_clip(&out.ctx, a.ctx, cMin, cMax, DefaultStream().ctx)
	return out
}

// BitwiseAnd returns element-wise bitwise AND.
func BitwiseAnd(a, b *Array) *Array {
	out := NewArray("BITWISE_AND", a, b)
	C.mlx_bitwise_and(&out.ctx, a.ctx, b.ctx, DefaultStream().ctx)
	return out
}

// BitwiseOr returns element-wise bitwise OR.
func BitwiseOr(a, b *Array) *Array {
	out := NewArray("BITWISE_OR", a, b)
	C.mlx_bitwise_or(&out.ctx, a.ctx, b.ctx, DefaultStream().ctx)
	return out
}

// LeftShift shifts integer values left by b.
func LeftShift(a, b *Array) *Array {
	out := NewArray("LEFT_SHIFT", a, b)
	C.mlx_left_shift(&out.ctx, a.ctx, b.ctx, DefaultStream().ctx)
	return out
}

// RightShift shifts integer values right by b.
func RightShift(a, b *Array) *Array {
	out := NewArray("RIGHT_SHIFT", a, b)
	C.mlx_right_shift(&out.ctx, a.ctx, b.ctx, DefaultStream().ctx)
	return out
}

// Matmul returns the matrix product of a and b.
//
//	out := metal.Matmul(x, wT) // [B, L, hidden] @ [hidden, out] → [B, L, out]
func Matmul(a, b *Array) *Array {
	out := NewArray("MATMUL", a, b)
	C.mlx_matmul(&out.ctx, a.ctx, b.ctx, DefaultStream().ctx)
	return out
}

// Conv2d performs a 2D convolution using MLX's NHWC input layout and
// [out_channels, kernel_h, kernel_w, in_channels] weight layout.
func Conv2d(input, weight *Array, strideH, strideW, padH, padW, dilationH, dilationW, groups int) *Array {
	out := NewArray("CONV2D", input, weight)
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

// Conv1d performs a 1D convolution using MLX's NLC input layout and
// [out_channels, kernel, in_channels/groups] weight layout. Depthwise
// convolution (the Conformer audio lconv1d) sets groups == channels.
//
//	out := metal.Conv1d(x, w, 1, 0, 1, channels) // depthwise, caller pre-pads
func Conv1d(input, weight *Array, stride, padding, dilation, groups int) *Array {
	out := NewArray("CONV1D", input, weight)
	C.mlx_conv1d(
		&out.ctx,
		input.ctx,
		weight.ctx,
		C.int(stride),
		C.int(padding),
		C.int(dilation),
		C.int(groups),
		DefaultStream().ctx,
	)
	return out
}

// PadAxis zero-pads one axis of an array by low/high elements.
//
//	padded := metal.PadAxis(x, 1, 12, 11) // pad the time axis: 12 left, 11 right
func PadAxis(a *Array, axis, low, high int) *Array {
	out := NewArray("PAD", a)
	zero := FromValue(float32(0))
	if dtype := a.Dtype(); dtype != DTypeFloat32 {
		cast := AsType(zero, dtype)
		Free(zero)
		zero = cast
	}
	axes := [1]C.int{C.int(axis)}
	lows := [1]C.int{C.int(low)}
	highs := [1]C.int{C.int(high)}
	mode := C.CString("constant")
	defer C.free(unsafe.Pointer(mode))
	C.mlx_pad(
		&out.ctx,
		a.ctx,
		&axes[0], 1,
		&lows[0], 1,
		&highs[0], 1,
		zero.ctx,
		mode,
		DefaultStream().ctx,
	)
	Free(zero)
	return out
}

// QuantizedMatmul performs quantized matrix multiplication.
func QuantizedMatmul(x, w, scales, biases *Array, transpose bool, groupSize, bits int) *Array {
	return quantizedMatmulMode(x, w, scales, biases, transpose, groupSize, bits, "affine")
}

// quantizedMatmulMode performs quantized matrix multiplication using the given
// MLX quantization mode.
func quantizedMatmulMode(x, w, scales, biases *Array, transpose bool, groupSize, bits int, mode string) *Array {
	out := NewArray("QMATMUL", x, w, scales, biases)
	gs := optionalInt(groupSize)
	b := optionalInt(bits)
	cMode := C.CString(NormalizeQuantizationMode(mode))
	defer C.free(unsafe.Pointer(cMode))
	C.mlx_quantized_matmul(
		&out.ctx, x.ctx, w.ctx, scales.ctx, optionalArray(biases),
		C._Bool(transpose), gs, b, cMode,
		DefaultStream().ctx,
	)
	return out
}

// GatherMM performs expert-indexed matrix multiplication.
func GatherMM(a, b, lhsIndices, rhsIndices *Array, sorted bool) *Array {
	out := NewArray("GATHER_MM", a, b, lhsIndices, rhsIndices)
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
	out := NewArray("GATHER_QMM", x, w, scales, biases, lhsIndices, rhsIndices)
	gs := optionalInt(groupSize)
	b := optionalInt(bits)
	cMode := C.CString(NormalizeQuantizationMode(mode))
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

// Softmax returns softmax along the last axis.  Routes through
// mlx_softmax_single_axis_inline so the single-element axis array is C-stack
// allocated rather than a per-call Go []C.int{}.
//
//	probs := metal.Softmax(logits) // convert raw logits to probability distribution
func Softmax(a *Array) *Array {
	out := NewArray("SOFTMAX", a)
	C.mlx_softmax_single_axis_inline(&out.ctx, a.ctx, C.int(-1), C.bool(false), DefaultStream().ctx)
	return out
}

// Argmax returns the index of the maximum value along an axis.
//
//	tokenID := metal.Argmax(logits, -1, false) // Greedy decoding: pick most likely token
func Argmax(a *Array, axis int, keepDims bool) *Array {
	out := NewArray("ARGMAX", a)
	C.mlx_argmax_axis(&out.ctx, a.ctx, C.int(axis), C._Bool(keepDims), DefaultStream().ctx)
	return out
}

// TopK returns the top k values along the last axis.
func TopK(a *Array, k int) *Array {
	out := NewArray("TOPK", a)
	C.mlx_topk_axis(&out.ctx, a.ctx, C.int(k), C.int(-1), DefaultStream().ctx)
	return out
}

// Sum reduces by summation along the given axis.  Routes through
// mlx_sum_single_axis_inline so the single-element axis array stays on the
// C stack and the per-call Go alloc is removed.
func Sum(a *Array, axis int, keepDims bool) *Array {
	out := NewArray("SUM", a)
	C.mlx_sum_single_axis_inline(&out.ctx, a.ctx, C.int(axis), C.bool(keepDims), DefaultStream().ctx)
	return out
}

// Mean reduces by averaging along the given axis.  Routes through
// mlx_mean_single_axis_inline so the single-element axis array stays on the
// C stack and the per-call Go alloc is removed.
func Mean(a *Array, axis int, keepDims bool) *Array {
	out := NewArray("MEAN", a)
	C.mlx_mean_single_axis_inline(&out.ctx, a.ctx, C.int(axis), C.bool(keepDims), DefaultStream().ctx)
	return out
}

// Reshape changes the shape of an array.  Routes through the
// mlx_reshape_inline cgo wrapper so the per-call C.int shape array is
// stack-allocated in C rather than heap-allocated in Go.
//
//	input := metal.Reshape(tokens, 1, int32(len(tokens))) // add batch dim: [L] → [1, L]
func Reshape(a *Array, shape ...int32) *Array {
	if len(shape) > MaxTensorRank {
		panic("Reshape: rank exceeds MaxTensorRank")
	}
	out := NewArray("RESHAPE", a)
	// Copy the variadic shape into a pooled C buffer instead of passing
	// &shape[0] to cgo. The direct address escapes the variadic []int32 the
	// caller builds (Reshape(x, B, L, …) on the per-token attention out-proj
	// + PLE path), heap-allocating it every layer; the copy keeps the param
	// non-escaping so the literal stays on the caller's stack. (Reshape1 already
	// covers the rank-1 scalar case.)
	var shapePtr *C.int32_t
	var shapeBuf *[MaxTensorRank]C.int32_t
	if len(shape) > 0 {
		shapeBuf = metalKernelShapeScratch.Get().(*[MaxTensorRank]C.int32_t)
		for i, v := range shape {
			shapeBuf[i] = C.int32_t(v)
		}
		shapePtr = &shapeBuf[0]
	}
	C.mlx_reshape_inline(&out.ctx, a.ctx, shapePtr, C.size_t(len(shape)), DefaultStream().ctx)
	if shapeBuf != nil {
		metalKernelShapeScratch.Put(shapeBuf)
	}
	return out
}

// Reshape1 is the rank-1 scalar-pass form of Reshape — eliminates the
// variadic-slice escape that `Reshape(arr, int32(n))` pays on every call.
// Used by packQ4Cached's `Reshape(q, int32(n))` + `Reshape(packed2D,
// int32(pairs))` and unpackQ4's `Reshape(stacked, int32(flatLen))` +
// maxAll's `Reshape(a, int32(n))` — every Q4 K/V Update + every
// quantise/maxAll boundary previously paid one slice escape per call.
// Routes through mlx_reshape_inline_1 which materialises the 1-element
// shape buffer on the C stack directly from the register-passed scalar.
//
//	flat := metal.Reshape1(q, int32(n))
func Reshape1(a *Array, n int32) *Array {
	out := NewArray("RESHAPE", a)
	C.mlx_reshape_inline_1(&out.ctx, a.ctx, C.int32_t(n), DefaultStream().ctx)
	return out
}

// Reshape2 is the rank-2 scalar-pass form of Reshape — eliminates the
// variadic-slice escape that `Reshape(arr, int32(h), int32(w))` pays on
// every call. Used by packQ4Cached's `Reshape(padded, int32(pairs),
// int32(2))` — the [pairs, 2] view that powers the low/high nibble
// extraction. Routes through mlx_reshape_inline_2 which materialises the
// 2-element shape buffer on the C stack directly from register-passed
// scalars. W11-AC complement to Slice2 / SliceUpdateInplace2 on the
// rank-2 frontier of the substrate.
//
//	paired := metal.Reshape2(padded, int32(pairs), 2)
func Reshape2(a *Array, h, w int32) *Array {
	out := NewArray("RESHAPE", a)
	C.mlx_reshape_inline_2(&out.ctx, a.ctx, C.int32_t(h), C.int32_t(w), DefaultStream().ctx)
	return out
}

// Reshape3 is the rank-3 scalar-pass form of Reshape — eliminates the
// variadic-slice escape that `Reshape(arr, d0, d1, d2)` pays in per-layer
// Gemma 4 PLE view streaming.
func Reshape3(a *Array, d0, d1, d2 int32) *Array {
	out := NewArray("RESHAPE", a)
	C.mlx_reshape_inline_3(&out.ctx, a.ctx, C.int32_t(d0), C.int32_t(d1), C.int32_t(d2), DefaultStream().ctx)
	return out
}

// Transpose permutes dimensions. If no axes given, reverses all dims.
// Routes through mlx_transpose_axes_inline so the caller's []int axes are
// narrowed to C int on the C stack rather than via a Go-side cgo-int slice.
func Transpose(a *Array, axes ...int) *Array {
	if len(axes) > MaxTensorRank {
		panic("Transpose: rank exceeds MaxTensorRank")
	}
	out := NewArray("TRANSPOSE", a)
	if len(axes) == 0 {
		C.mlx_transpose(&out.ctx, a.ctx, DefaultStream().ctx)
	} else {
		axesPtr := (*C.longlong)(unsafe.Pointer(&axes[0]))
		C.mlx_transpose_axes_inline(&out.ctx, a.ctx, axesPtr, C.size_t(len(axes)), DefaultStream().ctx)
	}
	return out
}

// Transpose4 is the rank-4 scalar-pass form of Transpose — eliminates the
// `[]int` allocation that the variadic axes parameter forces on cgo (escape
// analysis: -gcflags='-m' shows `... argument escapes to heap` on every
// rank-4 transpose call). Used by attention kernels' Transpose(k, 0,1,3,2)
// pattern across SDPAPaged + per-page transposes (Gemma 3/4, Qwen 3, etc.).
//
//	keyT := metal.Transpose4(key, 0, 1, 3, 2)
func Transpose4(a *Array, a0, a1, a2, a3 int) *Array {
	out := NewArray("TRANSPOSE", a)
	C.mlx_transpose_axes_inline_4(&out.ctx, a.ctx,
		C.int(a0), C.int(a1), C.int(a2), C.int(a3),
		DefaultStream().ctx)
	return out
}

// ExpandDims inserts a new axis at the given position.
func ExpandDims(a *Array, axis int) *Array {
	out := NewArray("EXPAND_DIMS", a)
	C.mlx_expand_dims(&out.ctx, a.ctx, C.int(axis), DefaultStream().ctx)
	return out
}

// Squeeze removes dimensions of size 1.  Routes through
// mlx_squeeze_axes_inline so the caller's []int axes are narrowed to C int
// on the C stack rather than via a Go-side cgo-int slice.
func Squeeze(a *Array, axes ...int) *Array {
	if len(axes) > MaxTensorRank {
		panic("Squeeze: rank exceeds MaxTensorRank")
	}
	out := NewArray("SQUEEZE", a)
	var axesPtr *C.longlong
	if len(axes) > 0 {
		axesPtr = (*C.longlong)(unsafe.Pointer(&axes[0]))
	}
	C.mlx_squeeze_axes_inline(&out.ctx, a.ctx, axesPtr, C.size_t(len(axes)), DefaultStream().ctx)
	return out
}

// Concatenate joins arrays along the given axis.
func Concatenate(arrays []*Array, axis int) *Array {
	if len(arrays) == 2 {
		return Concatenate2(arrays[0], arrays[1], axis)
	}
	vector := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(vector)

	for _, a := range arrays {
		C.mlx_vector_array_append_value(vector, a.ctx)
	}

	out := NewArray("CONCAT")
	C.mlx_concatenate_axis(&out.ctx, vector, C.int(axis), DefaultStream().ctx)
	return out
}

func Concatenate2(left, right *Array, axis int) *Array {
	out := NewArray("CONCAT")
	C.mlx_concatenate_axis_2(&out.ctx, left.ctx, right.ctx, C.int(axis), DefaultStream().ctx)
	return out
}

// BroadcastTo broadcasts an array to the given shape.  Routes through
// mlx_broadcast_to_inline so the per-call C.int shape array is materialised
// on the C stack rather than the Go heap.
func BroadcastTo(a *Array, shape []int32) *Array {
	if len(shape) > MaxTensorRank {
		panic("BroadcastTo: rank exceeds MaxTensorRank")
	}
	out := NewArray("BROADCAST", a)
	var shapePtr *C.int32_t
	if len(shape) > 0 {
		shapePtr = (*C.int32_t)(unsafe.Pointer(&shape[0]))
	}
	C.mlx_broadcast_to_inline(&out.ctx, a.ctx, shapePtr, C.size_t(len(shape)), DefaultStream().ctx)
	return out
}

// AsType casts an array to a different dtype.
func AsType(a *Array, dtype DType) *Array {
	out := NewArray("ASTYPE", a)
	C.mlx_astype(&out.ctx, a.ctx, C.mlx_dtype(dtype), DefaultStream().ctx)
	return out
}

// AsStrided creates a view with custom strides.  Transformer attention paths
// call this with rank-4 shape + strides three times per layer (Q/K/V) on the
// per-token forward pass, so this routes through mlx_as_strided_inline — the
// shape/strides arrays are materialised on the C stack rather than the Go
// heap, eliminating two cgo allocs per call (one for cShape, one for cStrides).
func AsStrided(a *Array, shape []int32, strides []int64, offset int64) *Array {
	if len(shape) > MaxTensorRank || len(strides) > MaxTensorRank {
		panic("AsStrided: rank exceeds MaxTensorRank")
	}
	out := NewArray("AS_STRIDED", a)
	// Copy shape/strides into pooled C buffers instead of passing &shape[0] /
	// &strides[0] straight to cgo: the direct address escapes the caller's
	// slice to the heap, and the per-token attention path builds these as
	// []int32{…}/[]int64{…} literals for q/k/v every layer. Pooled copies keep
	// the params non-escaping so the caller's literals stay on the stack.
	var shapePtr *C.int32_t
	var shapeBuf *[MaxTensorRank]C.int32_t
	if len(shape) > 0 {
		shapeBuf = metalKernelShapeScratch.Get().(*[MaxTensorRank]C.int32_t)
		for i, v := range shape {
			shapeBuf[i] = C.int32_t(v)
		}
		shapePtr = &shapeBuf[0]
	}
	var stridesPtr *C.int64_t
	var stridesBuf *[MaxTensorRank]C.int64_t
	if len(strides) > 0 {
		stridesBuf = metalStridesScratch.Get().(*[MaxTensorRank]C.int64_t)
		for i, v := range strides {
			stridesBuf[i] = C.int64_t(v)
		}
		stridesPtr = &stridesBuf[0]
	}
	C.mlx_as_strided_inline(&out.ctx, a.ctx, shapePtr, C.size_t(len(shape)), stridesPtr, C.size_t(len(strides)), C.size_t(offset), DefaultStream().ctx)
	if shapeBuf != nil {
		metalKernelShapeScratch.Put(shapeBuf)
	}
	if stridesBuf != nil {
		metalStridesScratch.Put(stridesBuf)
	}
	return out
}

// Take gathers elements from a along axis using indices.
func Take(a, indices *Array, axis int) *Array {
	out := NewArray("TAKE", a, indices)
	C.mlx_take_axis(&out.ctx, a.ctx, indices.ctx, C.int(axis), DefaultStream().ctx)
	return out
}

// Where selects elements from a or b based on condition.
func Where(condition, a, b *Array) *Array {
	out := NewArray("WHERE", condition, a, b)
	C.mlx_where(&out.ctx, condition.ctx, a.ctx, b.ctx, DefaultStream().ctx)
	return out
}

// Argpartition partially sorts and returns indices for top-k selection.
func Argpartition(a *Array, kth, axis int) *Array {
	out := NewArray("ARGPARTITION", a)
	C.mlx_argpartition_axis(&out.ctx, a.ctx, C.int(kth), C.int(axis), DefaultStream().ctx)
	return out
}

// Dequantize restores a quantized array to full precision.
//
//	fullW := metal.Dequantize(w, scales, biases, 64, 4) // 4-bit weights, group=64
func Dequantize(w, scales, biases *Array, groupSize, bits int) *Array {
	return DequantizeMode(w, scales, biases, groupSize, bits, "affine")
}

// DequantizeMode restores a quantized array to full precision using the given
// MLX quantization mode.
func DequantizeMode(w, scales, biases *Array, groupSize, bits int, mode string) *Array {
	out := NewArray("DEQUANTIZE", w, scales, biases)
	gs := optionalInt(groupSize)
	b := optionalInt(bits)
	cMode := C.CString(NormalizeQuantizationMode(mode))
	defer C.free(unsafe.Pointer(cMode))
	noDtype := C.mlx_optional_dtype{has_value: C._Bool(false)}
	C.mlx_dequantize(&out.ctx, w.ctx, scales.ctx, optionalArray(biases), gs, b, cMode, optionalArray(nil), noDtype, DefaultStream().ctx)
	return out
}

// PutAlongAxis places values into array at indices along axis.
func PutAlongAxis(a, indices, values *Array, axis int) *Array {
	out := NewArray("PUT_ALONG_AXIS", a, indices, values)
	// Use scatter approach: src[indices] = values
	C.mlx_put_along_axis(&out.ctx, a.ctx, indices.ctx, values.ctx, C.int(axis), DefaultStream().ctx)
	return out
}

// TakeAlongAxis gathers elements from a along axis using indices.
// Unlike Take, this uses the same number of dimensions for indices and input.
func TakeAlongAxis(a, indices *Array, axis int) *Array {
	out := NewArray("TAKE_ALONG_AXIS", a, indices)
	C.mlx_take_along_axis(&out.ctx, a.ctx, indices.ctx, C.int(axis), DefaultStream().ctx)
	return out
}

// LogSumExp computes log(sum(exp(a))) along the given axis.
// Numerically stable reduction for cross-entropy loss.
func LogSumExp(a *Array, axis int, keepDims bool) *Array {
	out := NewArray("LOGSUMEXP", a)
	C.mlx_logsumexp_axis(&out.ctx, a.ctx, C.int(axis), C._Bool(keepDims), DefaultStream().ctx)
	return out
}

// CumSum returns the cumulative sum along the given axis.
// reverse=false for forward, inclusive=true to include the current element.
func CumSum(a *Array, axis int, reverse, inclusive bool) *Array {
	out := NewArray("CUMSUM", a)
	C.mlx_cumsum(&out.ctx, a.ctx, C.int(axis), C._Bool(reverse), C._Bool(inclusive), DefaultStream().ctx)
	return out
}

// Sort returns the array sorted along the given axis.
//
//	sortedProbs := metal.Sort(probs, -1) // sort probability distribution ascending
func Sort(a *Array, axis int) *Array {
	out := NewArray("SORT", a)
	C.mlx_sort_axis(&out.ctx, a.ctx, C.int(axis), DefaultStream().ctx)
	return out
}

// Argsort returns the indices that would sort the array along the given axis.
//
//	sortIdx := metal.Argsort(negProbs, -1) // descending sort for top-p nucleus sampling
func Argsort(a *Array, axis int) *Array {
	out := NewArray("ARGSORT", a)
	C.mlx_argsort_axis(&out.ctx, a.ctx, C.int(axis), DefaultStream().ctx)
	return out
}

// Round returns element-wise rounding to the nearest integer value.
func Round(a *Array) *Array {
	out := NewArray("ROUND", a)
	C.mlx_round(&out.ctx, a.ctx, C.int(0), DefaultStream().ctx)
	return out
}

// Greater returns element-wise a > b as a bool array.
func Greater(a, b *Array) *Array {
	out := NewArray("GREATER", a, b)
	C.mlx_greater(&out.ctx, a.ctx, b.ctx, DefaultStream().ctx)
	return out
}

// GreaterEqual returns element-wise a >= b as a bool array.
//
// The NSA / MoBA block-selection masks compare per-block scores against the
// n-th-largest TopK threshold to build the keep set; both packages previously
// composed a>=b locally as ¬(b>a) (Greater + Equal-with-false). This is the
// direct MLX primitive.
//
//	keep := metal.GreaterEqual(blockScores, threshold) // top-n keep mask
func GreaterEqual(a, b *Array) *Array {
	out := NewArray("GREATER_EQUAL", a, b)
	C.mlx_greater_equal(&out.ctx, a.ctx, b.ctx, DefaultStream().ctx)
	return out
}

// Equal returns element-wise a == b as a bool array.
func Equal(a, b *Array) *Array {
	out := NewArray("EQUAL", a, b)
	C.mlx_equal(&out.ctx, a.ctx, b.ctx, DefaultStream().ctx)
	return out
}

// greaterScalar returns element-wise a > scalar.
//
// Routes through mlx_greater_scalar_inline — single cgo crossing covers
// scalar creation + comparison + scalar release.  Used by the sampler
// per-token hot path (TopP threshold compare) where the rhs is a Go
// float32 constant.
func greaterScalar(a *Array, scalar float32) *Array {
	out := NewArray("GREATER_SCALAR", a)
	C.mlx_greater_scalar_inline(&out.ctx, a.ctx, C.float(scalar), DefaultStream().ctx)
	return out
}

// whereScalarScalar returns element-wise where(cond, a_scalar, b_scalar).
//
// Routes through mlx_where_scalar_scalar_inline — single cgo crossing covers
// both scalar creations + ternary select + both scalar releases.  Used by
// the sampler per-token hot path (TopP mask-build: -inf where excluded,
// else 0).
func whereScalarScalar(cond *Array, aScalar, bScalar float32) *Array {
	out := NewArray("WHERE_SCALAR_SCALAR", cond)
	C.mlx_where_scalar_scalar_inline(&out.ctx, cond.ctx, C.float(aScalar), C.float(bScalar), DefaultStream().ctx)
	return out
}

// WhereScalarArray returns element-wise where(cond, a_scalar, b).
//
// Routes through mlx_where_scalar_array_inline — single cgo crossing covers
// scalar creation + ternary select + scalar release.  Used by the sampler
// per-token hot path (TopP / MinP mask-apply: -inf where excluded, original
// logit otherwise).
func WhereScalarArray(cond *Array, aScalar float32, b *Array) *Array {
	out := NewArray("WHERE_SCALAR_ARRAY", cond, b)
	C.mlx_where_scalar_array_inline(&out.ctx, cond.ctx, C.float(aScalar), b.ctx, DefaultStream().ctx)
	return out
}

// scalarGreater returns element-wise scalar > a (reversed operand order).
//
// Routes through mlx_scalar_greater_inline — single cgo crossing covers
// scalar creation + comparison + scalar release.  Used by MinPSampler
// where the threshold scalar is the LHS of the comparison.
func scalarGreater(scalar float32, a *Array) *Array {
	out := NewArray("SCALAR_GREATER", a)
	C.mlx_scalar_greater_inline(&out.ctx, a.ctx, C.float(scalar), DefaultStream().ctx)
	return out
}

func lessEqual(a, b *Array) *Array {
	out := NewArray("LESS_EQUAL", a, b)
	C.mlx_less_equal(&out.ctx, a.ctx, b.ctx, DefaultStream().ctx)
	return out
}

// MaxAxis returns the maximum value along the given axis.
func MaxAxis(a *Array, axis int, keepDims bool) *Array {
	out := NewArray("MAX_AXIS", a)
	C.mlx_max_axis(&out.ctx, a.ctx, C.int(axis), C._Bool(keepDims), DefaultStream().ctx)
	return out
}

// Any reduces with logical OR over all elements. Returns a scalar bool array.
// Set keepDims to preserve the reduced dimension as size 1.
//
//	hasTrues := metal.Any(mask, false) // check if any element is true
func Any(a *Array, keepDims bool) *Array {
	out := NewArray("ANY", a)
	C.mlx_any(&out.ctx, a.ctx, C._Bool(keepDims), DefaultStream().ctx)
	return out
}

// AnyAxis reduces with logical OR along the given axis.
//
//	rowHasTrue := metal.AnyAxis(mask, 1, false) // per-row OR reduction
func AnyAxis(a *Array, axis int, keepDims bool) *Array {
	out := NewArray("ANY_AXIS", a)
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
	out := NewArray("ARANGE")
	C.mlx_arange(&out.ctx, C.double(start), C.double(stop), C.double(step), C.mlx_dtype(dtype), DefaultStream().ctx)
	return out
}

// IsNaN returns a boolean array indicating which elements are NaN.
//
//	nanMask := metal.IsNaN(logits) // detect NaN values before sampling
func IsNaN(a *Array) *Array {
	out := NewArray("ISNAN", a)
	C.mlx_isnan(&out.ctx, a.ctx, DefaultStream().ctx)
	return out
}
