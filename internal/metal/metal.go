// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

// Package metal provides Go bindings for Apple's MLX framework via mlx-c.
package metal

/*
#cgo CXXFLAGS: -std=gnu++17 -O2 -DNDEBUG -Wno-deprecated-declarations -include ${SRCDIR}/mlx_build_config.h
#cgo CXXFLAGS: -DACCELERATE_NEW_LAPACK -DFMT_HEADER_ONLY=1 -DMLX_USE_ACCELERATE
#cgo CFLAGS: -mmacosx-version-min=14.0
#cgo darwin CFLAGS: -x objective-c
#cgo CPPFLAGS: -I${SRCDIR}/../../lib/mlx
#cgo CPPFLAGS: -I${SRCDIR}/../../lib/mlx-c
#cgo CPPFLAGS: -I${SRCDIR}/../../lib/fmt/include
#cgo CPPFLAGS: -I${SRCDIR}/../../lib/gguflib
#cgo CPPFLAGS: -I${SRCDIR}/../../lib/json/single_include/nlohmann
#cgo CPPFLAGS: -I${SRCDIR}/../../dist/include/metal_cpp
#cgo darwin LDFLAGS: -framework Foundation -framework Metal -framework Accelerate -framework QuartzCore

#include <stdatomic.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "mlx/c/mlx.h"

static _Atomic(char *) last_mlx_error = NULL;

// mlx_go_error_handler copies the error message because MLX-C frees the
// original buffer after the handler returns (_mlx_error uses stack-local
// std::vector<char>).
static void mlx_go_error_handler(const char *msg, void *data) {
    char *copy = strdup(msg);
    char *prev = atomic_exchange_explicit(&last_mlx_error, copy, memory_order_acq_rel);
    free(prev); // free any previous uncollected error
}

static void set_error_handler() {
    mlx_set_error_handler(&mlx_go_error_handler, NULL, NULL);
}

static const char* get_and_clear_last_error() {
    return atomic_exchange_explicit(&last_mlx_error, NULL, memory_order_acquire);
}

static bool mlx_go_metal_has_usable_device(void) {
    @autoreleasepool {
        NSArray<id<MTLDevice>> *devices = MTLCopyAllDevices();
        bool ok = devices != nil && devices.count > 0;
#if !__has_feature(objc_arc)
        [devices release];
#endif
        return ok;
    }
}
*/
import "C"

import (
	"os"
	"path/filepath"
	"sync"
	"unsafe"

	"dappco.re/go/core"
)

var initOnce sync.Once

func defaultMetallibPath() string {
	const metallib = "mlx.metallib"
	var candidates []string
	if wd, err := os.Getwd(); err == nil {
		candidates = append(candidates,
			filepath.Join(wd, "dist", "lib", metallib),
			filepath.Join(wd, "..", "..", "dist", "lib", metallib),
		)
	}
	if exe, err := os.Executable(); err == nil {
		exeDir := filepath.Dir(exe)
		candidates = append(candidates,
			filepath.Join(exeDir, metallib),
			filepath.Join(exeDir, "dist", "lib", metallib),
			filepath.Join(exeDir, "..", "lib", metallib),
		)
	}
	for _, candidate := range candidates {
		if _, err := os.Stat(candidate); err == nil {
			return candidate
		}
	}
	return metallib
}

func metalAvailableNoInit() bool {
	var available C.bool
	C.mlx_metal_is_available(&available)
	return bool(available)
}

func usableMetalDeviceNoInit() bool {
	if !metalAvailableNoInit() {
		return false
	}
	return bool(C.mlx_go_metal_has_usable_device())
}

func setDefaultCPUDeviceNoInit() {
	if usableMetalDeviceNoInit() {
		return
	}

	dev := C.mlx_device_new_type(C.MLX_CPU, 0)
	defer C.mlx_device_free(dev)

	if rc := C.mlx_set_default_device(dev); rc != 0 {
		if err := lastError(); err != nil {
			core.Error("mlx: set cpu default device", "error", err)
			return
		}
		core.Error("mlx: set cpu default device", "error", core.E("metal.Init", "set default CPU device", nil))
	}
}

// Init sets up the MLX error handler and metallib path.
// Called automatically on first use. Safe to call multiple times.
//
//	metal.Init() // idempotent; safe to call multiple times
func Init() {
	initOnce.Do(func() {
		// Set the metallib path before any Metal operation triggers device
		// initialisation. Prefer runtime locations so binaries are not tied to
		// source file paths.
		// os.Setenv is required here — core has no SetEnv, and Metal device init
		// reads this env var before any CGo call. Legitimate hardware boundary.
		if core.Env("MLX_METALLIB_PATH") == "" {
			os.Setenv("MLX_METALLIB_PATH", defaultMetallibPath())
		}

		C.set_error_handler()
		// Some headless macOS environments expose the MLX runtime without a
		// usable Metal device. Defaulting to CPU keeps direct array operations
		// and explicit cpu loads functional instead of aborting on first alloc.
		setDefaultCPUDeviceNoInit()
	})
}

// lastError reads and clears the most recent MLX-C error, or nil if none.
// The returned error message is heap-allocated by strdup in the C error handler,
// so we free it after copying to a Go string.
func lastError() error {
	msg := C.get_and_clear_last_error()
	if msg == nil {
		return nil
	}
	goMsg := C.GoString(msg)
	C.free(unsafe.Pointer(msg))
	return core.E("mlx.lastError", goMsg, nil)
}

// Eval synchronously evaluates arrays on the GPU.
// Use in code paths that need to propagate errors; see also Materialize.
//
//	if err := metal.Eval(logits); err != nil { return err }
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
		return core.E("mlx.Eval", core.Sprintf("eval failed (rc=%d)", rc), nil)
	}
	return nil
}

// EvalAsync queues arrays for asynchronous GPU evaluation.
//
//	if err := metal.EvalAsync(output); err != nil { return err }
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
		return core.E("mlx.EvalAsync", core.Sprintf("async eval failed (rc=%d)", rc), nil)
	}
	return nil
}

// Materialize synchronously evaluates arrays on the GPU; errors are logged only.
// Use [Eval] when error propagation is needed.
//
//	metal.Materialize(a, b, c)
func Materialize(outputs ...*Array) {
	if err := Eval(outputs...); err != nil {
		core.Error("mlx: materialize", "error", err)
	}
}

// MaterializeAsync queues arrays for asynchronous GPU evaluation; errors are logged only.
//
//	metal.MaterializeAsync(output)
func MaterializeAsync(outputs ...*Array) {
	if err := EvalAsync(outputs...); err != nil {
		core.Error("mlx: materialize async", "error", err)
	}
}

// MetalAvailable reports whether Metal GPU is available on this device.
//
//	if metal.MetalAvailable() { /* GPU path */ }
func MetalAvailable() bool {
	Init()
	return usableMetalDeviceNoInit()
}

// Version returns the MLX framework version string (e.g. "0.24.0").
//
//	fmt.Printf("MLX version: %s\n", metal.Version())
func Version() string {
	Init()
	str := C.mlx_string_new()
	defer C.mlx_string_free(str)
	C.mlx_version(&str)
	return C.GoString(C.mlx_string_data(str))
}
