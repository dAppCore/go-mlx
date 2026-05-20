// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

/*
#include <stdint.h>
#include <stdlib.h>
#include "mlx/c/mlx.h"

extern void goPinnedRawArrayRelease(void* payload);

static void go_pinned_raw_array_release(void* payload) {
	goPinnedRawArrayRelease(payload);
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
	void* payload,
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

type pinnedRawArrayBuffer struct {
	raw    []byte
	pinner runtime.Pinner
}

var (
	pinnedRawArrayBuffers sync.Map
	pinnedRawArrayNextID  atomic.Uintptr
)

func registerPinnedRawArray(raw []byte) (uintptr, unsafe.Pointer, error) {
	if len(raw) == 0 {
		return 0, nil, core.NewError("mlx: pinned array data is empty")
	}
	buffer := &pinnedRawArrayBuffer{raw: raw}
	buffer.pinner.Pin(&buffer.raw[0])
	id := pinnedRawArrayNextID.Add(1)
	pinnedRawArrayBuffers.Store(id, buffer)
	return id, unsafe.Pointer(unsafe.SliceData(buffer.raw)), nil
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
	buffer.pinner.Unpin()
}

//export goPinnedRawArrayRelease
func goPinnedRawArrayRelease(payload unsafe.Pointer) {
	unregisterPinnedRawArray(uintptr(payload))
}

func fromPinnedRawBytes(raw []byte, shape []int, dtype DType) (*Array, error) {
	return fromPinnedRawBytesStrided(raw, shape, shape, contiguousStrides(shape), 0, dtype)
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

	cStorageShape := make([]C.int, len(storageShape))
	for i, dim := range storageShape {
		if dim <= 0 {
			return nil, core.NewError("mlx: pinned array storage shape is invalid")
		}
		cStorageShape[i] = C.int(dim)
	}
	cViewShape := make([]C.int, len(viewShape))
	for i, dim := range viewShape {
		if dim <= 0 {
			return nil, core.NewError("mlx: pinned array view shape is invalid")
		}
		cViewShape[i] = C.int(dim)
	}
	cViewStrides := make([]C.int64_t, len(viewStrides))
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
	array := newArray("PINNED_RAW")
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
		unsafe.Pointer(id),
		C.go_pinned_raw_array_release_ptr(),
	)
	if array.ctx.ctx == nil {
		unregisterPinnedRawArray(id)
		if err := lastError(); err != nil {
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
	stride := int64(1)
	for i := len(shape) - 1; i >= 0; i-- {
		strides[i] = stride
		stride *= int64(shape[i])
	}
	return strides
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
