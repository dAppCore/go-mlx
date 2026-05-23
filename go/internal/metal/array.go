// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

/*
#include <stdlib.h>
#include "mlx/c/mlx.h"

static const void* go_mlx_array_data_float16(mlx_array arr) {
	return (const void*)mlx_array_data_float16(arr);
}

static const void* go_mlx_array_data_bfloat16(mlx_array arr) {
	return (const void*)mlx_array_data_bfloat16(arr);
}

static const void* go_mlx_array_data_complex64(mlx_array arr) {
	return (const void*)mlx_array_data_complex64(arr);
}

// mlx_zeros_inline / mlx_array_new_data_inline materialise the shape array
// on the C stack so the Go side passes &shape[0] from the caller-owned slice
// without forcing the cgo escape analyser to heap-allocate a []C.int copy.
// Rank is bounded by maxTensorRank = 8 in ops.go.
static inline int mlx_zeros_inline(
    mlx_array* res, const int32_t* shape_in, size_t shape_num,
    mlx_dtype dtype, mlx_stream s) {
    int shape_buf[8];
    for (size_t i = 0; i < shape_num; ++i) shape_buf[i] = (int)shape_in[i];
    return mlx_zeros(res, shape_buf, shape_num, dtype, s);
}

// mlx_zeros_inline_4 is the rank-4 scalar-pass form — eliminates the
// []int32{...} literal allocation by passing the 4 dims as scalars.  KV
// cache page-grow paths construct []int32{B,H,pageSize,D} on every new-page
// call; passing the four register-passed scalars eliminates the slice
// literal escape entirely.  Same W11-A pattern as mlx_slice_inline_4.
static inline int mlx_zeros_inline_4(
    mlx_array* res, int32_t s0, int32_t s1, int32_t s2, int32_t s3,
    mlx_dtype dtype, mlx_stream s) {
    int shape_buf[4] = {(int)s0, (int)s1, (int)s2, (int)s3};
    return mlx_zeros(res, shape_buf, 4, dtype, s);
}

// mlx_array_new_data_inline_i / _ll variants accept the caller's int32 (for
// raw-tensor APIs) or long long (for Go-int variadic FromValues) shape slice
// and copy into a 8-slot stack int buffer before forwarding.
static inline mlx_array mlx_array_new_data_inline_i(
    const void* data, const int32_t* shape_in, int shape_num, mlx_dtype dtype) {
    int shape_buf[8];
    for (int i = 0; i < shape_num; ++i) shape_buf[i] = (int)shape_in[i];
    return mlx_array_new_data(data, shape_buf, shape_num, dtype);
}

static inline mlx_array mlx_array_new_data_inline_ll(
    const void* data, const long long* shape_in, int shape_num, mlx_dtype dtype) {
    int shape_buf[8];
    for (int i = 0; i < shape_num; ++i) shape_buf[i] = (int)shape_in[i];
    return mlx_array_new_data(data, shape_buf, shape_num, dtype);
}
*/
import "C"

import (
	"encoding/binary"
	"iter"
	"reflect"
	"runtime"
	"sync"
	"unsafe"

	"dappco.re/go"
)

// Array wraps an mlx_array handle.
// Memory management relies on Go GC finalizers to call mlx_array_free,
// which decrements MLX-C's internal reference count. MLX-C handles all
// cross-array references internally — the Go wrapper does not track them.
type Array struct {
	ctx  C.mlx_array
	name string // debug label
}

// arrayPool recycles *Array wrappers across newArray / Free cycles.  The
// pool dominates the alloc surface for every MLX op on the hot path: the
// PagedKVCache single-token Prealloc bench (525 allocs/op baseline) profiles
// newArray at 92.27% of all object allocations, so amortising the heap cell
// across reuses is the single largest leverage point on the substrate's
// bedrock floor.
//
// Pool contract — load-bearing, do not weaken without re-reading the design
// rationale below:
//
//  1. Get path (newArray): the pool returns either a fresh &Array{} (from
//     New) or a previously-recycled struct whose finalizer was cancelled by
//     Free.  In both cases newArray re-applies SetFinalizer for the new
//     life.  runtime.SetFinalizer explicitly supports being called again on
//     the same pointer after a prior SetFinalizer(obj, nil).
//
//  2. Put path (Free): only Free puts back to the pool.  Free has already
//     released the C handle, zeroed ctx.ctx, and cancelled the finalizer
//     before the struct returns to the pool — so a pooled struct is fully
//     dormant (no live C resource, no pending finalizer) until Get re-arms
//     it.  The GC-fallback path (finalizeArray firing on an array the caller
//     never Free'd) does NOT route through the pool: that finalizer cleans
//     up the C handle and the struct is dropped by the GC normally.  This
//     keeps the GC-fallback safety net intact for forgotten arrays.
//
//  3. Safety rule for callers: once Free(arr) returns, the caller MUST NOT
//     dereference arr — same contract as sync.Pool everywhere (bytes.Buffer,
//     fmt printers, etc.).  Holding a pointer past Free is a use-after-pool
//     bug whether pooling lives here or not; in this codebase every Free()
//     call site immediately drops the reference (typically slice mutation or
//     local-var shadowing), so the contract is already satisfied today.
//
//  4. Defensive Put refusal: if a hypothetical bug ever called Free's
//     put-back path on a struct whose ctx wasn't cleared, the array would
//     be admitted to the pool with a live C handle.  arrayPoolPut guards
//     against that by refusing to recycle any Array with a non-nil ctx —
//     the struct is simply dropped (its existing finalizer-or-nil state is
//     unchanged), preserving correctness at the cost of one heap cell.
//
// Failure modes considered and rejected:
//
//   - SetFinalizer-after-cancel-after-SetFinalizer: documented as supported.
//   - Pool dropping a pooled struct between Put and Get: pooled structs
//     carry no live C resource (Free cleared ctx) and no finalizer, so the
//     GC reclaims them as plain heap memory.
//   - Pooled struct used by two callers concurrently: would require a
//     caller to retain the pointer past Free, which is the same use-after-
//     Pool bug class as sync.Pool everywhere.  The -race build catches it.
//   - GGUF/io_custom paths that build &Array{} directly (without newArray)
//     and SetFinalizer manually: these don't route through the pool either
//     on construction or on Free's put-back path (the struct didn't come
//     from arrayPool.Get) — they remain on the classic finalizer-only path.
//     This was a deliberate scoping decision: those are cold-load paths,
//     not hot-op paths, so the pool's reach is contained to the workloads
//     that dominate the alloc profile.
var arrayPool = sync.Pool{
	New: func() any {
		return &Array{}
	},
}

// newArray creates a named Array and registers a GC finalizer.
// The inputs parameter is accepted for API compatibility but not stored —
// MLX-C tracks inter-array references via its own refcounting.
//
// The *Array struct is recycled via arrayPool — see the arrayPool comment
// block for the lifecycle contract.  Returned arrays always have a fresh
// finalizer and a zero ctx; callers populate ctx via the MLX-C builder of
// their choice (mlx_array_new_*, mlx_<op>(&out.ctx, ...), etc.) before
// handing the wrapper on.
func newArray(name string, inputs ...*Array) *Array {
	t := arrayPool.Get().(*Array)
	t.name = name
	// Pool invariant: pooled structs always have ctx.ctx == nil because Free
	// clears it before put-back, and the New fn returns a zero-value Array.
	// Re-assert here as a debug-grade safety net — if this ever fires,
	// arrayPoolPut admitted a struct with a live ctx (a real correctness
	// bug, not a perf-tuning one).
	runtime.SetFinalizer(t, finalizeArray)
	return t
}

// arrayPoolPut returns a fully-released *Array to the recycle pool.  Only
// safe to call after the C handle has been freed, ctx zeroed, and the
// finalizer cancelled — Free is the canonical caller and guarantees all
// three preconditions.  Refuses to admit any struct with a non-nil ctx so
// that a future bug in the Free path can't smuggle a live handle into the
// pool's New cycle.
func arrayPoolPut(t *Array) {
	if t == nil || t.ctx.ctx != nil {
		return
	}
	t.name = ""
	arrayPool.Put(t)
}

// finalizeArray is called by Go GC to release the underlying C array handle.
// This is the fallback path for arrays whose caller never called Free; the
// struct does NOT return to arrayPool from here — the pool only recycles
// structs whose owner explicitly cleaned up via Free.
func finalizeArray(t *Array) {
	if t != nil && t.ctx.ctx != nil {
		C.mlx_array_free(t.ctx)
		t.ctx.ctx = nil
	}
}

type scalarTypes interface {
	~bool | ~int | ~float32 | ~float64 | ~complex64
}

// FromValue creates a scalar Array from a Go value.
func FromValue[T scalarTypes](t T) *Array {
	Init()
	tt := newArray("")
	switch v := any(t).(type) {
	case bool:
		tt.ctx = C.mlx_array_new_bool(C.bool(v))
	case int:
		tt.ctx = C.mlx_array_new_int(C.int(v))
	case float32:
		tt.ctx = C.mlx_array_new_float32(C.float(v))
	case float64:
		tt.ctx = C.mlx_array_new_float64(C.double(v))
	case complex64:
		tt.ctx = C.mlx_array_new_complex(C.float(real(v)), C.float(imag(v)))
	default:
		panic("mlx: unsupported scalar type")
	}
	return tt
}

type arrayTypes interface {
	~bool | ~uint8 | ~uint16 | ~uint32 | ~uint64 |
		~int8 | ~int16 | ~int32 | ~int64 |
		~float32 | ~float64 |
		~complex64
}

// FromValues creates an Array from a Go slice with the given shape.
// Routes through mlx_array_new_data_inline_ll so the per-call shape array is
// stack-allocated on the C side — relevant for tokenizer / prefill code that
// builds many small input tensors.
func FromValues[S ~[]E, E arrayTypes](s S, shape ...int) *Array {
	Init()
	if len(shape) == 0 {
		panic("mlx: shape required for non-scalar tensors")
	}
	if len(shape) > maxTensorRank {
		panic("FromValues: rank exceeds maxTensorRank")
	}

	// reflect.TypeOf is required here to map Go generic type parameters to MLX-C
	// dtype constants. Type assertions cannot recover the element type from a
	// generic ~[]E constraint at runtime. CGo tensor boundary — not business logic.
	var dtype DType
	switch reflect.TypeOf(s).Elem().Kind() {
	case reflect.Bool:
		dtype = DTypeBool
	case reflect.Uint8:
		dtype = DTypeUint8
	case reflect.Uint16:
		dtype = DTypeUint16
	case reflect.Uint32:
		dtype = DTypeUint32
	case reflect.Uint64:
		dtype = DTypeUint64
	case reflect.Int8:
		dtype = DTypeInt8
	case reflect.Int16:
		dtype = DTypeInt16
	case reflect.Int32:
		dtype = DTypeInt32
	case reflect.Int64:
		dtype = DTypeInt64
	case reflect.Float32:
		dtype = DTypeFloat32
	case reflect.Float64:
		dtype = DTypeFloat64
	case reflect.Complex64:
		dtype = DTypeComplex64
	default:
		panic("mlx: unsupported element type")
	}

	bts := make([]byte, binary.Size(s))
	if _, err := binary.Encode(bts, binary.LittleEndian, s); err != nil {
		panic(err)
	}

	tt := newArray("")
	shapePtr := (*C.longlong)(unsafe.Pointer(&shape[0]))
	tt.ctx = C.mlx_array_new_data_inline_ll(unsafe.Pointer(&bts[0]), shapePtr, C.int(len(shape)), C.mlx_dtype(dtype))
	if tt.ctx.ctx == nil {
		if err := lastError(); err != nil {
			panic(err)
		}
		panic("mlx: array data creation failed")
	}
	runtime.KeepAlive(bts)
	return tt
}

// fromSingleInt32 fast-paths the common "wrap one int32 as a [1] array"
// case used by token-ID emitters (sample, decode, generate). Skips the
// FromValues generic + reflect dispatch path and writes a single-int
// mlx array directly. Stack-allocated shape array means zero alloc
// beyond the Array wrapper + mlx_array context.
func fromSingleInt32(value int32) *Array {
	Init()
	cShape := [1]C.int{1}
	tt := newArray("")
	tt.ctx = C.mlx_array_new_data(unsafe.Pointer(&value), &cShape[0], C.int(1), C.mlx_dtype(DTypeInt32))
	if tt.ctx.ctx == nil {
		if err := lastError(); err != nil {
			panic(err)
		}
		panic("mlx: array data creation failed")
	}
	runtime.KeepAlive(value)
	return tt
}

// Zeros creates a zero-filled Array with the given shape and dtype.
// Routes through mlx_zeros_inline so the per-call C.int shape array is
// stack-allocated on the C side, eliminating the Go heap copy and the
// associated cgo escape — relevant for the per-token sample-mask path
// and the cache page-grow path.
func Zeros(shape []int32, dtype DType) *Array {
	Init()
	if len(shape) > maxTensorRank {
		panic("Zeros: rank exceeds maxTensorRank")
	}
	tt := newArray("ZEROS")
	var shapePtr *C.int32_t
	if len(shape) > 0 {
		shapePtr = (*C.int32_t)(unsafe.Pointer(&shape[0]))
	}
	C.mlx_zeros_inline(&tt.ctx, shapePtr, C.size_t(len(shape)), C.mlx_dtype(dtype), DefaultStream().ctx)
	return tt
}

// Zeros4 is the rank-4 scalar-pass form of Zeros — eliminates the
// []int32{...} literal allocation that escapes to heap on every call.
// Routes through mlx_zeros_inline_4 which materialises the shape buffer on
// the C stack directly from register-passed scalars.  Used by PagedKVCache
// page-grow path where []int32{B,H,pageSize,D} previously paid one slice
// escape per Zeros call (two per appendNewPagePrealloc — K + V).
//
//	page := metal.Zeros4(B, H, int32(pageSize), D, dtype)
func Zeros4(s0, s1, s2, s3 int32, dtype DType) *Array {
	Init()
	tt := newArray("ZEROS")
	C.mlx_zeros_inline_4(&tt.ctx,
		C.int32_t(s0), C.int32_t(s1), C.int32_t(s2), C.int32_t(s3),
		C.mlx_dtype(dtype), DefaultStream().ctx)
	return tt
}

// Set replaces this array's C handle with another's.
//
//	a.Set(b) // a now wraps the same C array as b
func (t *Array) Set(other *Array) {
	C.mlx_array_set(&t.ctx, other.ctx)
}

// Clone creates a new Go wrapper sharing the same C handle (increments C refcount).
//
//	saved := a.Clone() // independent Go handle, same Metal buffer
func (t *Array) Clone() *Array {
	tt := newArray(t.name)
	C.mlx_array_set(&tt.ctx, t.ctx)
	return tt
}

// Valid reports whether this Array has a non-nil mlx handle.
//
//	if !a.Valid() { return } // guard before any ops on uninitialised arrays
func (t *Array) Valid() bool {
	if t == nil {
		return false
	}
	return t.ctx.ctx != nil
}

// String returns a human-readable representation of the array.
//
//	fmt.Println(a.String()) // "array([1.0, 2.0, 3.0], dtype=float32)"
func (t *Array) String() string {
	str := C.mlx_string_new()
	defer C.mlx_string_free(str)
	C.mlx_array_tostring(&str, t.ctx)
	return core.Trim(C.GoString(C.mlx_string_data(str)))
}

// Shape returns the dimensions as int32 slice.
//
//	shape := logits.Shape() // e.g. []int32{1, 512, 32000} for [batch, seq, vocab]
func (t *Array) Shape() []int32 {
	dims := make([]int32, t.NumDims())
	for i := range dims {
		dims[i] = int32(t.Dim(i))
	}
	return dims
}

// ShapeInto writes the array's dimensions into dst[:NumDims()] and returns
// the populated subslice. dst must have cap >= NumDims(). Callers can hand
// in a stack-allocated buffer or a pooled scratch to avoid the per-call
// `make([]int32, ndim)` heap alloc that Shape() pays.
//
//	var scratch [maxTensorRank]int32
//	shape := arr.ShapeInto(scratch[:0])
func (t *Array) ShapeInto(dst []int32) []int32 {
	n := t.NumDims()
	dst = dst[:n]
	for i := 0; i < n; i++ {
		dst[i] = int32(t.Dim(i))
	}
	return dst
}

// Size returns the total number of elements.
//
//	n := weights.Size() // e.g. 4096*4096 = 16777216
func (t Array) Size() int { return int(C.mlx_array_size(t.ctx)) }

// NumBytes returns the total byte size.
//
//	mb := float64(a.NumBytes()) / 1e6 // memory footprint in MB
func (t Array) NumBytes() int { return int(C.mlx_array_nbytes(t.ctx)) }

// NumDims returns the number of dimensions.
//
//	if a.NumDims() == 4 { /* BHLД layout */ }
func (t Array) NumDims() int { return int(C.mlx_array_ndim(t.ctx)) }

// Dim returns the size of dimension i.
//
//	seqLen := logits.Dim(1) // middle dimension of [batch, seq, vocab]
func (t Array) Dim(i int) int { return int(C.mlx_array_dim(t.ctx, C.int(i))) }

// Dims returns all dimensions as int slice.
//
//	B, L, V := dims[0], dims[1], dims[2] // unpack [batch, seq, vocab]
func (t Array) Dims() []int {
	dims := make([]int, t.NumDims())
	for i := range dims {
		dims[i] = t.Dim(i)
	}
	return dims
}

// Dtype returns the array's data type.
//
//	if a.Dtype() == DTypeBFloat16 { /* mixed precision path */ }
func (t Array) Dtype() DType { return DType(C.mlx_array_dtype(t.ctx)) }

// Int extracts a scalar integer value.
//
//	id := int32(next.Int()) // read sampled token ID from argmax output
func (t Array) Int() int {
	switch t.Dtype() {
	case DTypeUint8:
		var item C.uint8_t
		C.mlx_array_item_uint8(&item, t.ctx)
		return int(item)
	case DTypeUint16:
		var item C.uint16_t
		C.mlx_array_item_uint16(&item, t.ctx)
		return int(item)
	case DTypeUint32:
		var item C.uint32_t
		C.mlx_array_item_uint32(&item, t.ctx)
		return int(item)
	case DTypeUint64:
		var item C.uint64_t
		C.mlx_array_item_uint64(&item, t.ctx)
		return int(item)
	case DTypeInt8:
		var item C.int8_t
		C.mlx_array_item_int8(&item, t.ctx)
		return int(item)
	case DTypeInt16:
		var item C.int16_t
		C.mlx_array_item_int16(&item, t.ctx)
		return int(item)
	case DTypeInt32:
		var item C.int32_t
		C.mlx_array_item_int32(&item, t.ctx)
		return int(item)
	default:
		var item C.int64_t
		C.mlx_array_item_int64(&item, t.ctx)
		return int(item)
	}
}

// Float extracts a scalar float64 value.
// Handles both float32 and float64 array dtypes.
//
//	loss := lossArr.Float() // read scalar loss value after Eval
func (t Array) Float() float64 {
	switch t.Dtype() {
	case DTypeFloat32:
		var item C.float
		C.mlx_array_item_float32(&item, t.ctx)
		return float64(item)
	default:
		var item C.double
		C.mlx_array_item_float64(&item, t.ctx)
		return float64(item)
	}
}

// Bool extracts a scalar boolean value from a bool-dtype array.
//
//	if metal.Any(mask, false); result.Bool() { /* at least one true */ }
func (t Array) Bool() bool {
	var item C.bool
	C.mlx_array_item_bool(&item, t.ctx)
	return bool(item)
}

// SetFloat64 replaces this array with a float64 scalar value.
//
//	a.SetFloat64(3.14159) // overwrite array with a new scalar
func (t *Array) SetFloat64(v float64) {
	C.mlx_array_set_float64(&t.ctx, C.double(v))
}

// ShapeRaw returns a pointer to the C shape array and the number of dimensions.
// This avoids allocation when only direct dimension access is needed.
// The returned pointer is valid only while the array is alive.
//
//	ndim := a.NumDims()
//	ptr := a.ShapeRaw() // *C.int, read ptr[0..ndim-1]
func (t Array) ShapeRaw() unsafe.Pointer {
	return unsafe.Pointer(C.mlx_array_shape(t.ctx))
}

// IsRowContiguous reports whether the array's physical memory layout is
// row-major contiguous. Non-contiguous arrays (from Transpose, BroadcastTo,
// SliceAxis, etc.) must be made contiguous before reading raw data.
func (t Array) IsRowContiguous() bool {
	var res C.bool
	C._mlx_array_is_row_contiguous(&res, t.ctx)
	return bool(res)
}

// Contiguous returns a row-major contiguous copy of the array.
// If the array is already row-contiguous, this is a no-op.
//
//	c := metal.Contiguous(transposed) // required before reading raw float data
func Contiguous(a *Array) *Array {
	out := newArray("CONTIGUOUS", a)
	C.mlx_contiguous(&out.ctx, a.ctx, C._Bool(false), DefaultStream().ctx)
	return out
}

// ensureContiguous returns a row-contiguous array, making a copy if needed.
// This must be called before any mlx_array_data_* access.
func ensureContiguous(a *Array) *Array {
	if a.IsRowContiguous() {
		return a
	}
	c := Contiguous(a)
	Materialize(c)
	return c
}

// Bytes extracts all elements as a byte slice from a uint8 array.
// Automatically handles non-contiguous arrays (transpose, broadcast, slice views).
//
//	raw := frame.Bytes() // read a packed byte buffer back to Go memory
func (t *Array) Bytes() []byte {
	src := ensureContiguous(t)
	n := src.Size()
	ptr := C.mlx_array_data_uint8(src.ctx)
	data := make([]byte, n)
	for i, b := range unsafe.Slice(ptr, n) {
		data[i] = byte(b)
	}
	runtime.KeepAlive(src)
	return data
}

// RawBytes extracts the evaluated row-major byte representation of an array in
// its current dtype. This preserves float16/bfloat16 payloads without a
// float32 staging cast.
func (t *Array) RawBytes() []byte {
	src := ensureContiguous(t)
	n := src.NumBytes()
	if n <= 0 {
		runtime.KeepAlive(src)
		return nil
	}
	ptr := rawArrayDataPointer(src)
	if ptr == nil {
		runtime.KeepAlive(src)
		return nil
	}
	data := make([]byte, n)
	copy(data, unsafe.Slice((*byte)(ptr), n))
	runtime.KeepAlive(src)
	return data
}

func rawArrayDataPointer(src *Array) unsafe.Pointer {
	switch src.Dtype() {
	case DTypeBool:
		return unsafe.Pointer(C.mlx_array_data_bool(src.ctx))
	case DTypeUint8:
		return unsafe.Pointer(C.mlx_array_data_uint8(src.ctx))
	case DTypeUint16:
		return unsafe.Pointer(C.mlx_array_data_uint16(src.ctx))
	case DTypeFloat16:
		return C.go_mlx_array_data_float16(src.ctx)
	case DTypeBFloat16:
		return C.go_mlx_array_data_bfloat16(src.ctx)
	case DTypeUint32:
		return unsafe.Pointer(C.mlx_array_data_uint32(src.ctx))
	case DTypeUint64:
		return unsafe.Pointer(C.mlx_array_data_uint64(src.ctx))
	case DTypeInt8:
		return unsafe.Pointer(C.mlx_array_data_int8(src.ctx))
	case DTypeInt16:
		return unsafe.Pointer(C.mlx_array_data_int16(src.ctx))
	case DTypeInt32:
		return unsafe.Pointer(C.mlx_array_data_int32(src.ctx))
	case DTypeInt64:
		return unsafe.Pointer(C.mlx_array_data_int64(src.ctx))
	case DTypeFloat32:
		return unsafe.Pointer(C.mlx_array_data_float32(src.ctx))
	case DTypeFloat64:
		return unsafe.Pointer(C.mlx_array_data_float64(src.ctx))
	case DTypeComplex64:
		return C.go_mlx_array_data_complex64(src.ctx)
	default:
		return nil
	}
}

// FromRawBytes creates an Array from already-packed little-endian tensor bytes.
// Routes through mlx_array_new_data_inline_ll so the per-call shape array is
// stack-allocated on the C side, eliminating the Go heap copy.
func FromRawBytes(raw []byte, shape []int, dtype DType) *Array {
	Init()
	if len(shape) == 0 {
		panic("mlx: shape required for raw tensor")
	}
	if len(raw) == 0 {
		panic("mlx: raw tensor data is empty")
	}
	if byteSize := DTypeByteSize(dtype); byteSize <= 0 || len(raw)%byteSize != 0 {
		panic("mlx: raw tensor byte length does not match dtype")
	}
	if len(shape) > maxTensorRank {
		panic("FromRawBytes: rank exceeds maxTensorRank")
	}
	tt := newArray("")
	shapePtr := (*C.longlong)(unsafe.Pointer(&shape[0]))
	tt.ctx = C.mlx_array_new_data_inline_ll(unsafe.Pointer(&raw[0]), shapePtr, C.int(len(shape)), C.mlx_dtype(dtype))
	if tt.ctx.ctx == nil {
		if err := lastError(); err != nil {
			panic(err)
		}
		panic("mlx: raw array data creation failed")
	}
	runtime.KeepAlive(raw)
	return tt
}

// Ints extracts all elements as int slice (from int32 data).
// Automatically handles non-contiguous arrays (transpose, broadcast, slice views).
//
//	ids := tokenIDs.Ints() // read token ID list from a 1-D int32 array
func (t *Array) Ints() []int {
	src := ensureContiguous(t)
	n := src.Size()
	ptr := C.mlx_array_data_int32(src.ctx)
	ints := make([]int, n)
	for i, f := range unsafe.Slice(ptr, n) {
		ints[i] = int(f)
	}
	runtime.KeepAlive(src)
	return ints
}

// DataInt32 extracts all elements as int32 slice.
// Automatically handles non-contiguous arrays (transpose, broadcast, slice views).
//
//	ids := cacheKeys.DataInt32() // read int32 indices from an attention index array
func (t *Array) DataInt32() []int32 {
	src := ensureContiguous(t)
	n := src.Size()
	ptr := C.mlx_array_data_int32(src.ctx)
	data := make([]int32, n)
	for i, f := range unsafe.Slice(ptr, n) {
		data[i] = int32(f)
	}
	runtime.KeepAlive(src)
	return data
}

// Floats extracts all elements as float32 slice.
// Automatically handles non-contiguous arrays (transpose, broadcast, slice views).
//
//	flat := kSliced.Floats() // read KV cache values for attention inspection
func (t *Array) Floats() []float32 {
	src := t
	var converted *Array
	if t.Dtype() != DTypeFloat32 {
		converted = AsType(t, DTypeFloat32)
		Materialize(converted)
		src = converted
	}
	src = ensureContiguous(src)
	Materialize(src)
	n := src.Size()
	if n == 0 {
		Free(converted)
		return nil
	}
	ptr := C.mlx_array_data_float32(src.ctx)
	if ptr == nil {
		Free(converted)
		return nil
	}
	floats := make([]float32, n)
	for i, f := range unsafe.Slice(ptr, n) {
		floats[i] = float32(f)
	}
	runtime.KeepAlive(src)
	Free(converted)
	return floats
}

// Free explicitly releases C array handles. Does not cascade — MLX-C's
// internal refcounting handles dependent arrays automatically.
//
// Free is also the put-back path for the *Array wrapper pool: after the C
// handle is released and the finalizer cancelled, the Go struct is handed
// to arrayPoolPut for re-use by the next newArray.  Callers MUST NOT touch
// the *Array after Free returns — same contract as sync.Pool everywhere.
// See the arrayPool block in this file for the full lifecycle rationale.
func Free(s ...*Array) int {
	var n int
	for _, t := range s {
		if t != nil && t.Valid() {
			n += t.NumBytes()
			C.mlx_array_free(t.ctx)
			t.ctx.ctx = nil
			runtime.SetFinalizer(t, nil) // cancel finalizer
			arrayPoolPut(t)              // recycle the Go wrapper
		}
	}
	return n
}

// Iter returns an iterator over the array's float32 elements.
// The array must be materialised and contain float32 data.
// Automatically handles non-contiguous arrays (transpose, broadcast, slice views).
func (t *Array) Iter() iter.Seq[float32] {
	src := ensureContiguous(t)
	n := src.Size()
	ptr := C.mlx_array_data_float32(src.ctx)
	return func(yield func(float32) bool) {
		defer runtime.KeepAlive(src)
		for i := range n {
			if !yield(float32(unsafe.Slice(ptr, n)[i])) {
				return
			}
		}
	}
}
