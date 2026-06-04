// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

/*
#include <stdint.h>
#include <stdlib.h>
#include "mlx/c/mlx.h"

// Bridge between mlx's void*-payload dtor contract and our uintptr_t
// identifier scheme — payload is a synthetic id (not a Go pointer), so
// we keep it as uintptr_t everywhere the Go runtime can see it and only
// widen to void* inside C where it satisfies mlx's signature. This is
// the same pattern runtime/cgo.Handle uses, and it keeps the Go side
// free of `unsafe.Pointer(uintptr)` conversions that trip `go vet`'s
// unsafeptr check.
extern void goPinnedRawArrayRelease(uintptr_t payload);

static void go_pinned_raw_array_release(void* payload) {
	goPinnedRawArrayRelease((uintptr_t)payload);
}

typedef void (*go_pinned_raw_array_release_fn)(void*);
static go_pinned_raw_array_release_fn go_pinned_raw_array_release_ptr(void) {
	return &go_pinned_raw_array_release;
}

mlx_array go_mlx_array_new_pinned_strided_data(
	void* data,
	size_t byte_count,
	const int* storage_shape,
	int storage_dim,
	const int* view_shape,
	int view_dim,
	const int64_t* view_strides,
	int strides_dim,
	size_t view_offset,
	mlx_dtype dtype,
	mlx_stream stream,
	uintptr_t payload,
	void (*dtor)(void*));

mlx_array go_mlx_array_new_pinned_data(
	void* data,
	size_t byte_count,
	const int* shape,
	int dim,
	mlx_dtype dtype,
	uintptr_t payload,
	void (*dtor)(void*));
*/
import "C"

import (
	"runtime"
	"sync"
	"sync/atomic"
	"unsafe"

	core "dappco.re/go"
)

// pinnedRawArrayBuffer carries the Go-owned raw bytes plus the
// core.PinnedView that keeps them at a stable address for mlx_array's
// pinned-data slot. mlx retains the data pointer across mlx_eval, so
// the pin must live until the C-side release callback fires.
type pinnedRawArrayBuffer struct {
	raw  []byte
	view core.PinnedView
}

var (
	pinnedRawArrayBuffers sync.Map
	pinnedRawArrayNextID  atomic.Uintptr

	// pinnedRawArrayBufferPool recycles pinnedRawArrayBuffer structs across
	// register/unregister cycles. The buffer lifetime is mlx-side-driven —
	// it lives in pinnedRawArrayBuffers until mlx fires the release dtor,
	// then unregister Releases the view + clears the slice header and Puts
	// the struct back. W10-O wired the cgo-scratch pools but left this
	// per-call `&pinnedRawArrayBuffer{}` heap alloc on the hot path; pool
	// drops the steady-state floor by 1 alloc/call on the canonical
	// pinned-array build (3→2 allocs, 120→56 B/op across L1-L16384).
	pinnedRawArrayBufferPool = sync.Pool{
		New: func() any { return &pinnedRawArrayBuffer{} },
	}
)

// pinnedShapeScratchInt / pinnedShapeScratchInt64 pool the per-call cgo
// shape/stride buffers (rank-8 sized — MLX cap). fromPinnedRawBytesStrided
// fires on every KV-cache state restore (3 cgo arrays per call); the rank-4
// KV case used to pay 4 make([]C.int|C.int64_t, 4) + 1 make([]int64, 4) per
// invocation. Pool drops the floor to 0 cgo allocs on the strides path and
// 1 alloc on the shape path (the strides one comes via contiguousStrides
// which is pool-routed in its own helper below).
var (
	pinnedShapeScratchInt = sync.Pool{
		New: func() any { s := make([]C.int, MaxTensorRank); return &s },
	}
	pinnedShapeScratchInt64 = sync.Pool{
		New: func() any { s := make([]C.int64_t, MaxTensorRank); return &s },
	}
	pinnedStrideScratchInt64 = sync.Pool{
		New: func() any { s := make([]int64, MaxTensorRank); return &s },
	}
)

func registerPinnedRawArray(raw []byte) (uintptr, unsafe.Pointer, error) {
	if len(raw) == 0 {
		return 0, nil, core.NewError("mlx: pinned array data is empty")
	}
	buffer := pinnedRawArrayBufferPool.Get().(*pinnedRawArrayBuffer)
	buffer.raw = raw
	core.PinSlice(buffer.raw, &buffer.view)
	id := pinnedRawArrayNextID.Add(1)
	pinnedRawArrayBuffers.Store(id, buffer)
	return id, buffer.view.Ptr(), nil
}

func unregisterPinnedRawArray(id uintptr) {
	if id == 0 {
		return
	}
	value, ok := pinnedRawArrayBuffers.LoadAndDelete(id)
	if !ok {
		return
	}
	buffer, ok := value.(*pinnedRawArrayBuffer)
	if !ok || buffer == nil {
		return
	}
	buffer.view.Release()
	// Drop the slice reference so the underlying bytes are eligible for
	// GC the moment mlx releases the array — the pool only holds the
	// empty shell. PinnedView.Release already zeroed view; raw needs
	// explicit clear since the pool will hand this struct out for a
	// fresh raw next call.
	buffer.raw = nil
	pinnedRawArrayBufferPool.Put(buffer)
}

//export goPinnedRawArrayRelease
func goPinnedRawArrayRelease(payload C.uintptr_t) {
	unregisterPinnedRawArray(uintptr(payload))
}

func fromPinnedRawBytes(raw []byte, shape []int, dtype DType) (*Array, error) {
	Init()
	if len(shape) == 0 {
		return nil, core.NewError("mlx: pinned array requires shape")
	}
	byteSize := DTypeByteSize(dtype)
	storageElements, ok := shapeElementCount(shape)
	if byteSize <= 0 || !ok || storageElements*byteSize != len(raw) {
		return nil, core.NewError("mlx: pinned array byte length does not match shape")
	}
	shapePtr := pinnedShapeScratchInt.Get().(*[]C.int)
	defer pinnedShapeScratchInt.Put(shapePtr)
	cShape := (*shapePtr)[:len(shape):cap(*shapePtr)]
	for i, dim := range shape {
		if dim <= 0 {
			return nil, core.NewError("mlx: pinned array shape is invalid")
		}
		cShape[i] = C.int(dim)
	}

	id, ptr, err := registerPinnedRawArray(raw)
	if err != nil {
		return nil, err
	}
	array := NewArray("PINNED_RAW")
	array.ctx = C.go_mlx_array_new_pinned_data(
		ptr,
		C.size_t(len(raw)),
		unsafe.SliceData(cShape),
		C.int(len(cShape)),
		C.mlx_dtype(dtype),
		C.uintptr_t(id),
		C.go_pinned_raw_array_release_ptr(),
	)
	if array.ctx.ctx == nil {
		unregisterPinnedRawArray(id)
		if err := LastError(); err != nil {
			return nil, err
		}
		return nil, core.NewError("mlx: pinned array data creation failed")
	}
	runtime.KeepAlive(raw)
	runtime.KeepAlive(cShape)
	return array, nil
}

func fromPinnedFloat32Values(values []float32, shape []int) (*Array, error) {
	if len(values) == 0 {
		return nil, core.NewError("mlx: pinned float32 array data is empty")
	}
	raw := unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(values))), len(values)*DTypeByteSize(DTypeFloat32))
	array, err := fromPinnedRawBytes(raw, shape, DTypeFloat32)
	runtime.KeepAlive(values)
	return array, err
}

func fromPinnedRawBytesStrided(raw []byte, storageShape, viewShape []int, viewStrides []int64, viewOffset int, dtype DType) (*Array, error) {
	Init()
	if len(storageShape) == 0 || len(viewShape) == 0 || len(viewShape) != len(viewStrides) {
		return nil, core.NewError("mlx: pinned array requires storage and view shapes")
	}
	if viewOffset < 0 {
		return nil, core.NewError("mlx: pinned array offset is invalid")
	}
	byteSize := DTypeByteSize(dtype)
	storageElements, ok := shapeElementCount(storageShape)
	if byteSize <= 0 || !ok || storageElements*byteSize != len(raw) {
		return nil, core.NewError("mlx: pinned array byte length does not match shape")
	}

	// Reuse pooled rank-8 cgo scratch buffers. Validates dims inline so the
	// pool slot is returned even on the error path.
	storagePtr := pinnedShapeScratchInt.Get().(*[]C.int)
	defer pinnedShapeScratchInt.Put(storagePtr)
	cStorageShape := (*storagePtr)[:len(storageShape):cap(*storagePtr)]
	for i, dim := range storageShape {
		if dim <= 0 {
			return nil, core.NewError("mlx: pinned array storage shape is invalid")
		}
		cStorageShape[i] = C.int(dim)
	}
	viewShapePtr := pinnedShapeScratchInt.Get().(*[]C.int)
	defer pinnedShapeScratchInt.Put(viewShapePtr)
	cViewShape := (*viewShapePtr)[:len(viewShape):cap(*viewShapePtr)]
	for i, dim := range viewShape {
		if dim <= 0 {
			return nil, core.NewError("mlx: pinned array view shape is invalid")
		}
		cViewShape[i] = C.int(dim)
	}
	viewStridesPtr := pinnedShapeScratchInt64.Get().(*[]C.int64_t)
	defer pinnedShapeScratchInt64.Put(viewStridesPtr)
	cViewStrides := (*viewStridesPtr)[:len(viewStrides):cap(*viewStridesPtr)]
	for i, stride := range viewStrides {
		if stride < 0 {
			return nil, core.NewError("mlx: pinned array view stride is invalid")
		}
		cViewStrides[i] = C.int64_t(stride)
	}

	id, ptr, err := registerPinnedRawArray(raw)
	if err != nil {
		return nil, err
	}
	array := NewArray("PINNED_RAW")
	array.ctx = C.go_mlx_array_new_pinned_strided_data(
		ptr,
		C.size_t(len(raw)),
		unsafe.SliceData(cStorageShape),
		C.int(len(cStorageShape)),
		unsafe.SliceData(cViewShape),
		C.int(len(cViewShape)),
		unsafe.SliceData(cViewStrides),
		C.int(len(cViewStrides)),
		C.size_t(viewOffset),
		C.mlx_dtype(dtype),
		DefaultStream().ctx,
		C.uintptr_t(id),
		C.go_pinned_raw_array_release_ptr(),
	)
	if array.ctx.ctx == nil {
		unregisterPinnedRawArray(id)
		if err := LastError(); err != nil {
			return nil, err
		}
		return nil, core.NewError("mlx: pinned array data creation failed")
	}
	runtime.KeepAlive(raw)
	runtime.KeepAlive(cStorageShape)
	runtime.KeepAlive(cViewShape)
	runtime.KeepAlive(cViewStrides)
	return array, nil
}

func contiguousStrides(shape []int) []int64 {
	strides := make([]int64, len(shape))
	contiguousStridesInto(strides, shape)
	return strides
}

// contiguousStridesInto writes contiguous strides for shape into dst — used
// by the pooled-buffer hot path so contiguous-stride computation is
// alloc-free even for the common KV restore case.
func contiguousStridesInto(dst []int64, shape []int) {
	stride := int64(1)
	for i := len(shape) - 1; i >= 0; i-- {
		dst[i] = stride
		stride *= int64(shape[i])
	}
}

func shapeElementCount(shape []int) (int, bool) {
	total := 1
	for _, dim := range shape {
		if dim <= 0 || total > int(^uint(0)>>1)/dim {
			return 0, false
		}
		total *= dim
	}
	return total, true
}
