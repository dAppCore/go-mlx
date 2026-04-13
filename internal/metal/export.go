//go:build darwin && arm64

package metal

/*
#include <stdlib.h>
#include <stdint.h>
#include "mlx/c/mlx.h"

// Forward declarations for Go-exported callbacks.
extern int goUnaryFunc(mlx_array *res, const mlx_array input, void *payload);
extern void goUnaryDestructor(void *payload);
extern int goKwargsFunc(mlx_vector_array *res, const mlx_vector_array args, const mlx_map_string_to_array kwargs, void *payload);
extern void goKwargsDestructor(void *payload);

// Shim converts between vector_array and single array for the unary callback.
static int goUnaryShim(mlx_vector_array *res, const mlx_vector_array inputs, void *payload) {
    mlx_array input = mlx_array_new();
    mlx_vector_array_get(&input, inputs, 0);
    mlx_array output = mlx_array_new();
    int rc = goUnaryFunc(&output, input, payload);
    mlx_array_free(input);
    if (rc == 0) {
        mlx_vector_array_set_value(res, output);
    }
    mlx_array_free(output);
    return rc;
}

// Creates an mlx_closure backed by a Go unary function via payload dispatch.
// Accepts uintptr_t to avoid Go unsafe.Pointer conversion from integer.
static mlx_closure new_unary_closure(uintptr_t id) {
    return mlx_closure_new_func_payload(&goUnaryShim, (void*)id, &goUnaryDestructor);
}

// Creates an mlx_closure_kwargs backed by a Go kwargs function via payload dispatch.
// Accepts uintptr_t to avoid Go unsafe.Pointer conversion from integer.
static mlx_closure_kwargs new_kwargs_closure(uintptr_t id) {
    return mlx_closure_kwargs_new_func_payload(&goKwargsFunc, (void*)id, &goKwargsDestructor);
}
*/
import "C"

import (
	"runtime"
	"sync"
	"sync/atomic"
	"unsafe"

	"dappco.re/go/core"
)

// ---------------------------------------------------------------------------
// Closure registries — thread-safe maps from uintptr ID to Go functions.
// ---------------------------------------------------------------------------

var (
	unaryFuncs  sync.Map
	unaryNextID atomic.Uintptr

	kwargsFuncs  sync.Map
	kwargsNextID atomic.Uintptr
)

// UnaryFunc is a Go function that operates on a single input array and
// produces a single output array. Used with NewClosure.
//
//	fn := func(input *metal.Array) *metal.Array {
//	    return metal.Add(input, metal.FromValue(float32(1.0)))
//	}
type UnaryFunc func(input *Array) *Array

// KwargsFunc is a Go function that operates on positional arrays and named
// keyword arguments. Used with NewClosureKwargs.
//
//	fn := func(args []*metal.Array, kwargs map[string]*metal.Array) []*metal.Array {
//	    x := kwargs["x"]
//	    y := kwargs["y"]
//	    return []*metal.Array{metal.Mul(x, y)}
//	}
type KwargsFunc func(args []*Array, kwargs map[string]*Array) []*Array

// ---------------------------------------------------------------------------
// CGO callback exports — called from the C shims above.
// ---------------------------------------------------------------------------

//export goUnaryFunc
func goUnaryFunc(res *C.mlx_array, input C.mlx_array, payload unsafe.Pointer) C.int {
	id := uintptr(payload)
	fnI, ok := unaryFuncs.Load(id)
	if !ok {
		return 1
	}
	fn := fnI.(UnaryFunc)

	goInput := &Array{ctx: input, name: "CLOSURE_INPUT"}
	// Do not set a finalizer — the C side owns this array.

	goOutput := fn(goInput)
	if goOutput == nil || !goOutput.Valid() {
		return 1
	}
	C.mlx_array_set(res, goOutput.ctx)
	return 0
}

//export goUnaryDestructor
func goUnaryDestructor(payload unsafe.Pointer) {
	id := uintptr(payload)
	unaryFuncs.Delete(id)
}

//export goKwargsFunc
func goKwargsFunc(res *C.mlx_vector_array, args C.mlx_vector_array, kwargs C.mlx_map_string_to_array, payload unsafe.Pointer) C.int {
	id := uintptr(payload)
	fnI, ok := kwargsFuncs.Load(id)
	if !ok {
		return 1
	}
	fn := fnI.(KwargsFunc)

	// Unpack positional arguments.
	nArgs := int(C.mlx_vector_array_size(args))
	goArgs := make([]*Array, nArgs)
	for i := range nArgs {
		a := newArray("KWARGS_ARG")
		C.mlx_vector_array_get(&a.ctx, args, C.size_t(i))
		goArgs[i] = a
	}

	// Unpack keyword arguments.
	goKwargs := make(map[string]*Array)
	it := C.mlx_map_string_to_array_iterator_new(kwargs)
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
		goKwargs[name] = arr
	}

	goOutputs := fn(goArgs, goKwargs)

	tmp := C.mlx_vector_array_new()
	for _, out := range goOutputs {
		if out != nil && out.Valid() {
			C.mlx_vector_array_append_value(tmp, out.ctx)
		}
	}
	C.mlx_vector_array_set(res, tmp)
	C.mlx_vector_array_free(tmp)
	return 0
}

//export goKwargsDestructor
func goKwargsDestructor(payload unsafe.Pointer) {
	id := uintptr(payload)
	kwargsFuncs.Delete(id)
}

// ---------------------------------------------------------------------------
// Closure constructors
// ---------------------------------------------------------------------------

// Closure wraps an mlx_closure handle. Create with NewClosure.
type Closure struct {
	ctx C.mlx_closure
}

// NewClosure creates an MLX closure from a unary Go function. The function
// receives one input array and must return one output array.
//
//	cls := metal.NewClosure(func(input *metal.Array) *metal.Array {
//	    one := metal.FromValue(float32(1.0))
//	    return metal.Add(input, one)
//	})
//	defer cls.Free()
func NewClosure(fn UnaryFunc) *Closure {
	Init()
	id := unaryNextID.Add(1)
	unaryFuncs.Store(id, fn)
	cls := &Closure{ctx: C.new_unary_closure(C.uintptr_t(id))}
	runtime.SetFinalizer(cls, func(c *Closure) { c.Free() })
	return cls
}

// Free releases the underlying C closure. Safe to call multiple times.
//
//	defer cls.Free()
func (c *Closure) Free() {
	if c != nil && c.ctx.ctx != nil {
		C.mlx_closure_free(c.ctx)
		c.ctx.ctx = nil
	}
}

// ClosureKwargs wraps an mlx_closure_kwargs handle. Create with NewClosureKwargs.
type ClosureKwargs struct {
	ctx C.mlx_closure_kwargs
}

// NewClosureKwargs creates an MLX closure that accepts keyword arguments.
// The Go function receives positional args and a map of named arrays.
//
//	cls := metal.NewClosureKwargs(func(args []*metal.Array, kwargs map[string]*metal.Array) []*metal.Array {
//	    x := kwargs["x"]
//	    y := kwargs["y"]
//	    return []*metal.Array{metal.Mul(x, y)}
//	})
//	defer cls.Free()
func NewClosureKwargs(fn KwargsFunc) *ClosureKwargs {
	Init()
	id := kwargsNextID.Add(1)
	kwargsFuncs.Store(id, fn)
	cls := &ClosureKwargs{ctx: C.new_kwargs_closure(C.uintptr_t(id))}
	runtime.SetFinalizer(cls, func(c *ClosureKwargs) { c.Free() })
	return cls
}

// Free releases the underlying C closure. Safe to call multiple times.
//
//	defer cls.Free()
func (c *ClosureKwargs) Free() {
	if c != nil && c.ctx.ctx != nil {
		C.mlx_closure_kwargs_free(c.ctx)
		c.ctx.ctx = nil
	}
}

// ---------------------------------------------------------------------------
// Export functions — serialise closures to files.
// ---------------------------------------------------------------------------

// ExportFunction serialises a closure and its example arguments to a file.
// The exported function can later be loaded with ImportFunction.
// When shapeless is true, the function accepts inputs of any shape.
//
//	cls := metal.NewClosure(incFn)
//	defer cls.Free()
//	args := []*metal.Array{metal.FromValue(float32(1.0))}
//	err := metal.ExportFunction("inc.mlxfn", cls, args, false)
func ExportFunction(path string, cls *Closure, args []*Array, shapeless bool) error {
	Init()
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	argsVec := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(argsVec)
	for _, a := range args {
		if a != nil && a.Valid() {
			C.mlx_vector_array_append_value(argsVec, a.ctx)
		}
	}

	rc := C.mlx_export_function(cPath, cls.ctx, argsVec, C.bool(shapeless))
	if rc != 0 {
		if err := lastError(); err != nil {
			return err
		}
		return core.E("mlx.ExportFunction", core.Sprintf("export failed (rc=%d)", rc), nil)
	}
	return nil
}

// ExportFunctionKwargs serialises a kwargs closure with example arguments to a file.
// The exported function can later be loaded with ImportFunction.
//
//	cls := metal.NewClosureKwargs(mulFn)
//	defer cls.Free()
//	kwargs := map[string]*metal.Array{"x": x, "y": y}
//	err := metal.ExportFunctionKwargs("mul.mlxfn", cls, nil, kwargs, false)
func ExportFunctionKwargs(path string, cls *ClosureKwargs, args []*Array, kwargs map[string]*Array, shapeless bool) error {
	Init()
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	argsVec := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(argsVec)
	for _, a := range args {
		if a != nil && a.Valid() {
			C.mlx_vector_array_append_value(argsVec, a.ctx)
		}
	}

	kwargsMap := C.mlx_map_string_to_array_new()
	defer C.mlx_map_string_to_array_free(kwargsMap)
	for name, arr := range kwargs {
		cName := C.CString(name)
		C.mlx_map_string_to_array_insert(kwargsMap, cName, arr.ctx)
		C.free(unsafe.Pointer(cName))
	}

	rc := C.mlx_export_function_kwargs(cPath, cls.ctx, argsVec, kwargsMap, C.bool(shapeless))
	if rc != 0 {
		if err := lastError(); err != nil {
			return err
		}
		return core.E("mlx.ExportFunctionKwargs", core.Sprintf("export kwargs failed (rc=%d)", rc), nil)
	}
	return nil
}

// ---------------------------------------------------------------------------
// Import functions — load serialised closures from files.
// ---------------------------------------------------------------------------

// ImportedFunction wraps a function loaded from a serialised .mlxfn file.
// Create with ImportFunction, call with Apply or ApplyKwargs.
//
//	fn, err := metal.ImportFunction("inc.mlxfn")
//	if err != nil { log.Fatal(err) }
//	defer fn.Free()
//	results, err := fn.Apply(metal.FromValue(float32(1.0)))
//	// results[0] contains the output
type ImportedFunction struct {
	ctx C.mlx_imported_function
}

// ImportFunction loads a previously exported function from a file.
// The returned ImportedFunction must be freed after use.
//
//	fn, err := metal.ImportFunction("inc.mlxfn")
//	if err != nil { log.Fatal(err) }
//	defer fn.Free()
func ImportFunction(path string) (*ImportedFunction, error) {
	Init()
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	handle := C.mlx_imported_function_new(cPath)
	if handle.ctx == nil {
		if err := lastError(); err != nil {
			return nil, err
		}
		return nil, core.E("mlx.ImportFunction", "failed to load function from "+path, nil)
	}

	fn := &ImportedFunction{ctx: handle}
	runtime.SetFinalizer(fn, func(f *ImportedFunction) { f.Free() })
	return fn, nil
}

// Apply calls the imported function with positional arguments.
// Returns the output arrays.
//
//	results, err := fn.Apply(x)
//	y := results[0]
func (f *ImportedFunction) Apply(args ...*Array) ([]*Array, error) {
	argsVec := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(argsVec)
	for _, a := range args {
		if a != nil && a.Valid() {
			C.mlx_vector_array_append_value(argsVec, a.ctx)
		}
	}

	resVec := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(resVec)

	rc := C.mlx_imported_function_apply(&resVec, f.ctx, argsVec)
	if rc != 0 {
		if err := lastError(); err != nil {
			return nil, err
		}
		return nil, core.E("mlx.ImportedFunction.Apply", "apply failed", nil)
	}
	return vectorToArrays(resVec), nil
}

// ApplyKwargs calls the imported function with positional and keyword arguments.
// Returns the output arrays.
//
//	kwargs := map[string]*metal.Array{"x": x, "y": y}
//	results, err := fn.ApplyKwargs(nil, kwargs)
func (f *ImportedFunction) ApplyKwargs(args []*Array, kwargs map[string]*Array) ([]*Array, error) {
	argsVec := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(argsVec)
	for _, a := range args {
		if a != nil && a.Valid() {
			C.mlx_vector_array_append_value(argsVec, a.ctx)
		}
	}

	kwargsMap := C.mlx_map_string_to_array_new()
	defer C.mlx_map_string_to_array_free(kwargsMap)
	for name, arr := range kwargs {
		cName := C.CString(name)
		C.mlx_map_string_to_array_insert(kwargsMap, cName, arr.ctx)
		C.free(unsafe.Pointer(cName))
	}

	resVec := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(resVec)

	rc := C.mlx_imported_function_apply_kwargs(&resVec, f.ctx, argsVec, kwargsMap)
	if rc != 0 {
		if err := lastError(); err != nil {
			return nil, err
		}
		return nil, core.E("mlx.ImportedFunction.ApplyKwargs", "apply kwargs failed", nil)
	}
	return vectorToArrays(resVec), nil
}

// Free releases the underlying C handle. Safe to call multiple times.
//
//	defer fn.Free()
func (f *ImportedFunction) Free() {
	if f != nil && f.ctx.ctx != nil {
		C.mlx_imported_function_free(f.ctx)
		f.ctx.ctx = nil
	}
}
