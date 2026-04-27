// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

/*
#include <stdlib.h>
#include "mlx/c/mlx.h"
*/
import "C"

import (
	"iter"
	"runtime"
	"unsafe"

	"dappco.re/go/core"
)

// LoadSafetensors loads tensors from a .safetensors file, returning an iterator
// over (name, array) pairs. Tensors are loaded lazily on the CPU stream.
// Use [LoadAllSafetensors] for an error-returning variant.
//
//	for name, arr := range metal.LoadSafetensors("/path/to/model.safetensors") {
//	    weights[name] = arr
//	}
func LoadSafetensors(path string) iter.Seq2[string, *Array] {
	Init()
	return func(yield func(string, *Array) bool) {
		string2array := C.mlx_map_string_to_array_new()
		defer C.mlx_map_string_to_array_free(string2array)

		string2string := C.mlx_map_string_to_string_new()
		defer C.mlx_map_string_to_string_free(string2string)

		cPath := C.CString(path)
		defer C.free(unsafe.Pointer(cPath))

		cpu := C.mlx_default_cpu_stream_new()
		defer C.mlx_stream_free(cpu)

		rc := C.mlx_load_safetensors(&string2array, &string2string, cPath, cpu)
		if rc != 0 {
			// Error will surface via lastError(); caller iterates zero tensors.
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

// LoadAllSafetensors loads all tensors from a .safetensors file into a map.
// Returns an error if the file cannot be loaded.
//
//	weights, err := metal.LoadAllSafetensors("/path/to/model-00001-of-00004.safetensors")
func LoadAllSafetensors(path string) (map[string]*Array, error) {
	tensors := make(map[string]*Array)
	for name, arr := range LoadSafetensors(path) {
		tensors[name] = arr
	}
	if len(tensors) == 0 {
		if err := lastError(); err != nil {
			return nil, err
		}
		return nil, core.E("mlx.LoadAllSafetensors", "no tensors loaded from "+path, nil)
	}
	return tensors, nil
}
