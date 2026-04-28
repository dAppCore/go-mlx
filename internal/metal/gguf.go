// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

/*
#include <stdlib.h>
#include "mlx/c/mlx.h"

int mlx_load_gguf_arrays(mlx_map_string_to_array* res, const char* file, const mlx_stream s);
int mlx_save_gguf_arrays(const char* file, const mlx_map_string_to_array param);
*/
import "C"

import (
	"iter"
	"runtime"
	"unsafe"

	"dappco.re/go/core"
)

// LoadGGUF loads tensors from a .gguf file, returning an iterator over
// (name, array) pairs.
func LoadGGUF(path string) iter.Seq2[string, *Array] {
	Init()
	return func(yield func(string, *Array) bool) {
		string2array := C.mlx_map_string_to_array_new()
		defer C.mlx_map_string_to_array_free(string2array)

		cPath := C.CString(path)
		defer C.free(unsafe.Pointer(cPath))

		cpu := C.mlx_default_cpu_stream_new()
		defer C.mlx_stream_free(cpu)

		rc := C.mlx_load_gguf_arrays(&string2array, cPath, cpu)
		if rc != 0 {
			return
		}

		it := C.mlx_map_string_to_array_iterator_new(string2array)
		defer C.mlx_map_string_to_array_iterator_free(it)

		for {
			var key *C.char
			value := C.mlx_array_new()
			if C.mlx_map_string_to_array_iterator_next(&key, &value, it) != 0 {
				C.mlx_array_free(value)
				break
			}

			name := C.GoString(key)
			arr := &Array{ctx: value, name: name}
			runtime.SetFinalizer(arr, finalizeArray)
			if !yield(name, arr) {
				Free(arr)
				break
			}
		}
	}
}

// LoadAllGGUF loads all tensors from a .gguf file into a map.
func LoadAllGGUF(path string) (map[string]*Array, error) {
	tensors := make(map[string]*Array)
	for name, arr := range LoadGGUF(path) {
		tensors[name] = arr
	}
	if len(tensors) == 0 {
		if err := lastError(); err != nil {
			return nil, err
		}
		return nil, core.E("mlx.LoadAllGGUF", "no tensors loaded from "+path, nil)
	}
	return tensors, nil
}

// SaveGGUF saves a map of named arrays to a .gguf file.
func SaveGGUF(path string, weights map[string]*Array) error {
	Init()

	cMap := C.mlx_map_string_to_array_new()
	defer C.mlx_map_string_to_array_free(cMap)

	for name, arr := range weights {
		cName := C.CString(name)
		C.mlx_map_string_to_array_insert(cMap, cName, arr.ctx)
		C.free(unsafe.Pointer(cName))
	}

	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	rc := C.mlx_save_gguf_arrays(cPath, cMap)
	if rc != 0 {
		if err := lastError(); err != nil {
			return err
		}
		return core.E("mlx.SaveGGUF", "save gguf failed: "+path, nil)
	}
	return nil
}
