// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

/*
#include <stdbool.h>
#include "mlx/c/mlx.h"

static int mlx_go_closure_call_one(mlx_array *out, mlx_closure cls, mlx_array input, bool has_input) {
	mlx_array inputs[1] = {input};
	mlx_vector_array inputVec = has_input ? mlx_vector_array_new_data(inputs, 1) : mlx_vector_array_new();
	mlx_vector_array outVec = mlx_vector_array_new();
	int rc = mlx_closure_apply(&outVec, cls, inputVec);
	int input_free_rc = mlx_vector_array_free(inputVec);
	if (rc != 0) {
		mlx_vector_array_free(outVec);
		return rc;
	}
	if (input_free_rc != 0) {
		mlx_vector_array_free(outVec);
		return input_free_rc;
	}
	size_t count = mlx_vector_array_size(outVec);
	if (count == 1) {
		rc = mlx_vector_array_get(out, outVec, 0);
	} else {
		rc = -1001;
	}
	int output_free_rc = mlx_vector_array_free(outVec);
	return rc != 0 ? rc : output_free_rc;
}
*/
import "C"

import (
	"runtime"
	"sync"

	"dappco.re/go"
)

// CompiledFunc wraps a function for efficient repeated execution.
// The function is lowered through MLX compile and then called as a closure.
type CompiledFunc struct {
	cls C.mlx_closure
	mu  sync.Mutex
}

// CompileShapeless wraps a function for repeated execution.
// When shapeless is true MLX can reuse the compiled trace across shape changes.
//
//	geluFn := metal.CompileShapeless(func(in []*Array) []*Array {
//	    return []*Array{geluApprox(in[0])}
//	}, true)
func CompileShapeless(fn func([]*Array) []*Array, shapeless bool) *CompiledFunc {
	Init()
	source := newClosure(fn)
	defer C.mlx_closure_free(source)

	compiled := C.mlx_closure_new()
	rc := C.mlx_compile(&compiled, source, C.bool(shapeless))
	if rc != 0 {
		if err := LastError(); err != nil {
			panic(err)
		}
		panic(core.E("mlx.CompileShapeless", core.Sprintf("compile failed (rc=%d)", rc), nil))
	}

	cf := &CompiledFunc{cls: compiled}
	runtime.SetFinalizer(cf, func(c *CompiledFunc) { c.Free() })
	return cf
}

// Call executes the function with the given inputs.
//
//	result := geluFn.Call(gateProj)[0] // fused GELU on gate projection
func (cf *CompiledFunc) Call(inputs ...*Array) []*Array {
	ensureThreadStreams()
	cf.mu.Lock()
	defer cf.mu.Unlock()
	if !cf.Valid() {
		panic(core.NewError("mlx.CompiledFunc.Call: invalid compiled closure"))
	}

	inputVec := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(inputVec)
	for _, in := range inputs {
		if in != nil && in.Valid() {
			C.mlx_vector_array_append_value(inputVec, in.ctx)
		}
	}

	outVec := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(outVec)
	rc := C.mlx_closure_apply(&outVec, cf.cls, inputVec)
	if rc != 0 {
		if err := LastError(); err != nil {
			panic(err)
		}
		panic(core.E("mlx.CompiledFunc.Call", core.Sprintf("closure apply failed (rc=%d)", rc), nil))
	}
	return vectorToArrays(outVec)
}

// CallOne executes a one-input compiled function that returns one array.
// It avoids the variadic input slice and output []*Array allocation in Call,
// which matters for per-token compiled decode helpers.
func (cf *CompiledFunc) CallOne(input *Array) *Array {
	cf.mu.Lock()
	defer cf.mu.Unlock()
	if !cf.Valid() {
		panic(core.NewError("mlx.CompiledFunc.CallOne: invalid compiled closure"))
	}

	var in C.mlx_array
	hasInput := C.bool(false)
	if input != nil && input.Valid() {
		in = input.ctx
		hasInput = true
	}
	out := NewArray("VEC_OUT")
	rc := C.mlx_go_closure_call_one(&out.ctx, cf.cls, in, hasInput)
	if rc != 0 {
		Free(out)
		if err := LastError(); err != nil {
			panic(err)
		}
		panic(core.E("mlx.CompiledFunc.CallOne", core.Sprintf("closure apply failed (rc=%d)", rc), nil))
	}
	runtime.KeepAlive(input)
	return out
}

// Valid reports whether the compiled closure still owns a native handle.
func (cf *CompiledFunc) Valid() bool {
	return cf != nil && cf.cls.ctx != nil
}

// Free releases the compiled closure. It is safe to call multiple times.
func (cf *CompiledFunc) Free() {
	if cf != nil && cf.cls.ctx != nil {
		C.mlx_closure_free(cf.cls)
		cf.cls.ctx = nil
	}
}
