//go:build darwin && arm64

// Package metal provides Go bindings for Apple's MLX framework via mlx-c.
package metal

/*
#cgo CXXFLAGS: -std=c++17
#cgo CFLAGS: -mmacosx-version-min=26.0
#cgo CPPFLAGS: -I${SRCDIR}/../../dist/include
#cgo LDFLAGS: -L${SRCDIR}/../../dist/lib -lmlxc -lmlx
#cgo darwin LDFLAGS: -framework Foundation -framework Metal -framework Accelerate
#cgo darwin LDFLAGS: -Wl,-rpath,${SRCDIR}/../../dist/lib

#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include "mlx/c/mlx.h"

static _Atomic(const char *) last_mlx_error = NULL;

static void mlx_go_error_handler(const char *msg, void *data) {
    atomic_store_explicit(&last_mlx_error, msg, memory_order_release);
}

static void set_error_handler() {
    mlx_set_error_handler(&mlx_go_error_handler, NULL, NULL);
}

static const char* get_and_clear_last_error() {
    return atomic_exchange_explicit(&last_mlx_error, NULL, memory_order_acquire);
}
*/
import "C"

import (
	"log/slog"
	"sync"

	"dappco.re/go/core"

	coreerr "forge.lthn.ai/core/go-log"
)

var initOnce sync.Once

// Init sets up the MLX error handler. Called automatically on first use.
func Init() {
	initOnce.Do(func() {
		C.set_error_handler()
		slog.Debug("mlx: initialised with Metal backend")
	})
}

// lastError reads and clears the most recent MLX-C error.
// Returns nil if no error occurred since the last call.
func lastError() error {
	msg := C.get_and_clear_last_error()
	if msg == nil {
		return nil
	}
	return coreerr.E("mlx.lastError", C.GoString(msg), nil)
}

// Eval synchronously evaluates arrays on the GPU and returns any error.
// This is the error-returning variant of Materialize. Use it in code paths
// that need to propagate errors (e.g. the generate loop).
func Eval(outputs ...*Array) error {
	Init()
	vector := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(vector)

	for _, output := range outputs {
		if output != nil && output.Valid() {
			C.mlx_vector_array_append_value(vector, output.ctx)
		}
	}

	rc := C.mlx_eval(vector)
	if rc != 0 {
		if err := lastError(); err != nil {
			return err
		}
		return coreerr.E("mlx.Eval", core.Sprintf("eval failed (rc=%d)", rc), nil)
	}
	return nil
}

// EvalAsync queues arrays for asynchronous GPU evaluation and returns any error.
func EvalAsync(outputs ...*Array) error {
	Init()
	vector := C.mlx_vector_array_new()
	defer C.mlx_vector_array_free(vector)

	for _, output := range outputs {
		if output != nil && output.Valid() {
			C.mlx_vector_array_append_value(vector, output.ctx)
		}
	}

	rc := C.mlx_async_eval(vector)
	if rc != 0 {
		if err := lastError(); err != nil {
			return err
		}
		return coreerr.E("mlx.EvalAsync", core.Sprintf("async eval failed (rc=%d)", rc), nil)
	}
	return nil
}

// Materialize synchronously evaluates arrays, computing their values on the GPU.
// Errors are logged but not returned — use [Eval] when error propagation is needed.
func Materialize(outputs ...*Array) {
	if err := Eval(outputs...); err != nil {
		slog.Error("mlx: materialize", "error", err)
	}
}

// MaterializeAsync queues arrays for asynchronous GPU evaluation.
// Errors are logged but not returned — use [EvalAsync] when error propagation is needed.
func MaterializeAsync(outputs ...*Array) {
	if err := EvalAsync(outputs...); err != nil {
		slog.Error("mlx: materialize async", "error", err)
	}
}

// MetalAvailable reports whether Metal GPU is available.
func MetalAvailable() bool {
	Init()
	var available C.bool
	C.mlx_metal_is_available(&available)
	return bool(available)
}
