// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

// Package metal provides Go bindings for Apple's MLX framework via mlx-c.
package metal

/*
#cgo CXXFLAGS: -std=gnu++23 -mmacosx-version-min=26.0 -O2 -DNDEBUG -Wno-deprecated-declarations -include ${SRCDIR}/mlx_build_config.h
#cgo CXXFLAGS: -DACCELERATE_NEW_LAPACK -DFMT_HEADER_ONLY=1 -DFMT_CONSTEVAL= -DMLX_USE_ACCELERATE
#cgo CFLAGS: -mmacosx-version-min=26.0
#cgo darwin CFLAGS: -x objective-c
#cgo CPPFLAGS: -I${SRCDIR}/../../../external/go-cgo/go
#cgo CPPFLAGS: -I${SRCDIR}/../../../lib/mlx
#cgo CPPFLAGS: -I${SRCDIR}/../../../lib/mlx-c
#cgo CPPFLAGS: -I${SRCDIR}/../../../lib/fmt/include
#cgo CPPFLAGS: -I${SRCDIR}/../../../lib/gguflib
#cgo CPPFLAGS: -I${SRCDIR}/../../../lib/json/single_include/nlohmann
#cgo CPPFLAGS: -I${SRCDIR}/../../../dist/include
#cgo CPPFLAGS: -I${SRCDIR}/../../../dist/include/metal_cpp
#cgo CPPFLAGS: -I${SRCDIR}/../../../build/_deps/metal_cpp-src
#cgo CPPFLAGS: -I${SRCDIR}/../../../cpp/build/_deps/metal_cpp-src
#cgo CPPFLAGS: -I${SRCDIR}/../../../cpp/cmake-build-debug/_deps/metal_cpp-src
#cgo darwin LDFLAGS: -mmacosx-version-min=26.0 -framework Foundation -framework Metal -framework Accelerate -framework QuartzCore

#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/sysctl.h>
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

// mlx_go_bind_thread_stream pins the given stream as the CURRENT OS
// thread's default. MLX 0.31.2 made the default stream thread-local
// (ThreadLocalStream): eval/item resolve the stream from the calling
// thread, and Go goroutines migrate across OS threads, so any thread may
// arrive unbound ("There is no Stream(gpu, 0) in current thread"). The
// stream handle must be passed in — fetching the thread-local default on
// an unbound thread throws the same error this bind prevents. The bind
// is a TLS write — cheap enough to run on every eval entry.
static void mlx_go_bind_thread_stream(mlx_stream s) {
    mlx_set_default_stream(s);
}

static int mlx_go_eval_data(const mlx_array *data, size_t n, mlx_stream bind) {
    if (data == NULL || n == 0) {
        return 0;
    }
    mlx_go_bind_thread_stream(bind);
    mlx_vector_array vector = mlx_vector_array_new_data(data, n);
    int rc = mlx_eval(vector);
    int free_rc = mlx_vector_array_free(vector);
    return rc != 0 ? rc : free_rc;
}

static int mlx_go_async_eval_data(const mlx_array *data, size_t n, mlx_stream bind) {
    if (data == NULL || n == 0) {
        return 0;
    }
    mlx_go_bind_thread_stream(bind);
    mlx_vector_array vector = mlx_vector_array_new_data(data, n);
    int rc = mlx_async_eval(vector);
    int free_rc = mlx_vector_array_free(vector);
    return rc != 0 ? rc : free_rc;
}

static bool mlx_go_metal_has_usable_device(void) {
    @autoreleasepool {
        id<MTLDevice> defaultDevice = MTLCreateSystemDefaultDevice();
        if (defaultDevice != nil) {
#if !__has_feature(objc_arc)
            [defaultDevice release];
#endif
            return true;
        }
        NSArray<id<MTLDevice>> *devices = MTLCopyAllDevices();
        bool ok = devices != nil && devices.count > 0;
#if !__has_feature(objc_arc)
        [devices release];
#endif
        return ok;
    }
}

typedef struct {
    char name[128];
    char architecture[128];
    size_t max_buffer_length;
    size_t max_recommended_working_set_size;
    size_t memory_size;
} mlx_go_host_device_info_t;

// Returns a strdup'd path to mlx.metallib resolved via NSBundle when the
// process runs inside a .app/.framework bundle; NULL otherwise. NSBundle
// handles the Apple-canonical Contents/Resources/ layout natively, so a
// .app shipped with the metallib at the conventional location resolves
// without any path-walking heuristics on the Go side. Caller frees.
static char* mlx_go_bundle_metallib_path(void) {
    @autoreleasepool {
        NSBundle *bundle = [NSBundle mainBundle];
        if (bundle == nil) { return NULL; }
        NSString *path = [bundle pathForResource:@"mlx" ofType:@"metallib"];
        if (path == nil) { return NULL; }
        const char *raw = [path UTF8String];
        if (raw == NULL) { return NULL; }
        return strdup(raw);
    }
}

static void mlx_go_copy_nsstring(char *dst, size_t dst_len, NSString *value) {
    if (dst == NULL || dst_len == 0 || value == nil) {
        return;
    }
    const char *raw = [value UTF8String];
    if (raw == NULL) {
        return;
    }
    strncpy(dst, raw, dst_len - 1);
    dst[dst_len - 1] = '\0';
}

static void mlx_go_copy_sysctl_string(char *dst, size_t dst_len, const char *key) {
    if (dst == NULL || dst_len == 0 || key == NULL) {
        return;
    }
    size_t size = dst_len;
    if (sysctlbyname(key, dst, &size, NULL, 0) != 0) {
        return;
    }
    dst[dst_len - 1] = '\0';
}

static uint64_t mlx_go_sysctl_uint64(const char *key) {
    uint64_t value = 0;
    size_t size = sizeof(value);
    if (key == NULL || sysctlbyname(key, &value, &size, NULL, 0) != 0) {
        return 0;
    }
    return value;
}

static mlx_go_host_device_info_t mlx_go_host_device_info(void) {
    mlx_go_host_device_info_t info;
    memset(&info, 0, sizeof(info));
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        NSArray<id<MTLDevice>> *devices = nil;
        if (device == nil) {
            devices = MTLCopyAllDevices();
            if (devices != nil && devices.count > 0) {
                device = [devices objectAtIndex:0];
#if !__has_feature(objc_arc)
                [device retain];
#endif
            }
        }
        if (device != nil) {
            mlx_go_copy_nsstring(info.name, sizeof(info.name), device.name);
            mlx_go_copy_nsstring(info.architecture, sizeof(info.architecture), device.name);
            info.max_buffer_length = (size_t)device.maxBufferLength;
            if ([device respondsToSelector:@selector(recommendedMaxWorkingSetSize)]) {
                info.max_recommended_working_set_size = (size_t)device.recommendedMaxWorkingSetSize;
                info.memory_size = info.max_recommended_working_set_size;
            }
#if !__has_feature(objc_arc)
            [device release];
#endif
        }
#if !__has_feature(objc_arc)
        [devices release];
#endif
    }
    if (info.name[0] == '\0') {
        mlx_go_copy_sysctl_string(info.name, sizeof(info.name), "machdep.cpu.brand_string");
    }
    if (info.architecture[0] == '\0') {
        strncpy(info.architecture, info.name, sizeof(info.architecture) - 1);
        info.architecture[sizeof(info.architecture) - 1] = '\0';
    }
    if (info.memory_size == 0) {
        info.memory_size = (size_t)mlx_go_sysctl_uint64("hw.memsize");
    }
    if (info.max_recommended_working_set_size == 0 && info.memory_size > 0) {
        info.max_recommended_working_set_size = (size_t)((uint64_t)info.memory_size * 9 / 10);
    }
    return info;
}
*/
import "C"

import (
	"runtime"
	"sync"
	"unsafe"

	core "dappco.re/go"
	"dappco.re/go/cgo"
)

var initOnce sync.Once

// evalOutputCtxPool holds temporary mlx_array handle buffers for Eval/EvalAsync.
// The native helper copies the handles into an MLX vector synchronously, so the
// backing slice can be reused as soon as the cgo call returns.
var evalOutputCtxPool = sync.Pool{
	New: func() any {
		buf := make([]C.mlx_array, 0, 64)
		return &buf
	},
}

// metallibResolvedPath records what Init resolved MLX_METALLIB_PATH to —
// either a pre-set env (operator export or the embed_metallib extract;
// metallibFromEnv true) or this package's own resolution (NSBundle
// Resources, then the dev-tree walk). Diagnostics only: the load itself is
// mlx's (lib/mlx device.cpp load_default_library reads the env at device
// construction).
var (
	metallibResolvedPath string
	metallibFromEnv      bool
)

// MetallibResolution reports the metallib path this process hands MLX and
// whether it arrived from a pre-set MLX_METALLIB_PATH rather than this
// package's own resolution.
//
//	path, fromEnv := metal.MetallibResolution()
func MetallibResolution() (string, bool) {
	Init()
	return metallibResolvedPath, metallibFromEnv
}

func defaultMetallibPath() string {
	const metallib = "mlx.metallib"
	// Preferred: NSBundle resolution. When this binary runs inside a
	// .app bundle (or framework) with the metallib at the Apple-canonical
	// Contents/Resources/mlx.metallib location, NSBundle returns the
	// full path directly. The CWD-walk fallback below covers dev mode
	// where the binary runs from the source tree.
	if bundled := C.mlx_go_bundle_metallib_path(); bundled != nil {
		path := C.GoString(bundled)
		C.free(unsafe.Pointer(bundled))
		if path != "" {
			return path
		}
	}
	var candidates []string
	if wd := core.Getwd(); wd.OK {
		root := wd.Value.(string)
		candidates = append(candidates,
			core.PathJoin(root, "dist", "lib", metallib),
			core.PathJoin(root, "..", "dist", "lib", metallib),
			core.PathJoin(root, "..", "..", "dist", "lib", metallib),
			core.PathJoin(root, "..", "..", "..", "dist", "lib", metallib),
			core.PathJoin(root, "..", "..", "..", "..", "dist", "lib", metallib),
			core.PathJoin(root, "..", "..", "..", "..", "..", "dist", "lib", metallib),
		)
	}
	for _, candidate := range candidates {
		if core.Stat(candidate).OK {
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

func hostMetalDeviceAvailableNoInit() bool {
	return bool(C.mlx_go_metal_has_usable_device())
}

func usableMetalDeviceNoInit() bool {
	if !hostMetalDeviceAvailableNoInit() {
		return false
	}
	if metalAvailableNoInit() {
		return true
	}
	// The bundled CGo MLX source build can report the MLX-level Metal flag as
	// unavailable even when the process has a real MTLDevice. Host Metal is the
	// load-safety boundary here; later GPU stream/device creation still returns
	// an MLX error if the backend cannot execute.
	return true
}

func hostDeviceInfo() DeviceInfo {
	info := C.mlx_go_host_device_info()
	return DeviceInfo{
		Name:                         C.GoString(&info.name[0]),
		Architecture:                 C.GoString(&info.architecture[0]),
		MaxBufferLength:              uint64(info.max_buffer_length),
		MaxRecommendedWorkingSetSize: uint64(info.max_recommended_working_set_size),
		MemorySize:                   uint64(info.memory_size),
	}
}

func setDefaultCPUDeviceNoInit() {
	if usableMetalDeviceNoInit() {
		return
	}

	dev := C.mlx_device_new_type(C.MLX_CPU, 0)
	defer C.mlx_device_free(dev)

	if rc := C.mlx_set_default_device(dev); rc != 0 {
		if err := LastError(); err != nil {
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
		if env := core.Env("MLX_METALLIB_PATH"); env != "" {
			metallibResolvedPath, metallibFromEnv = env, true
		} else {
			metallibResolvedPath = defaultMetallibPath()
			if result := core.Setenv("MLX_METALLIB_PATH", metallibResolvedPath); !result.OK {
				core.Warn("mlx: set metallib path", "error", result.Value)
			}
		}
		// MLX commits a Metal command buffer every N graph ops (50 on Ultra)
		// or every 50MB of registered output bytes. A decode token's graph
		// spans hundreds of dispatches, so the op cap alone paid several
		// commits per token on the encode path; 1000 measured +2-3% decode.
		// The byte cap stays at MLX's default: raising it was measured NET
		// WORSE on weight-heavy models (31B sweep 1024MB-16GB all ~50ms/token
		// vs 47ms at 50MB) — fine-grained commits start the GPU earliest, and
		// the eval throttle that looks like a host stall is actually the GPU
		// pacing the host. Set-if-unset honours an operator's explicit MLX
		// tuning.
		if core.Env("MLX_MAX_OPS_PER_BUFFER") == "" {
			if result := core.Setenv("MLX_MAX_OPS_PER_BUFFER", "1000"); !result.OK {
				core.Warn("mlx: set max ops per buffer", "error", result.Value)
			}
		}

		C.set_error_handler()
		// Some headless macOS environments expose the MLX runtime without a
		// usable Metal device. Keep initialisation deterministic here; model
		// loading validates the device before creating MLX streams.
		setDefaultCPUDeviceNoInit()
	})
}

// LastError reads and clears the most recent MLX-C error, or nil if none.
// The returned error message is heap-allocated by strdup in the C error
// handler — cgo.AdoptCString copies it to a Go string and frees the C
// side in a single named call. The unsafe.Pointer cast is required
// because cgo types don't unify across Go packages (go-mlx's *C.char
// and go-cgo's *C.char are distinct types despite same underlying).
func LastError() error {
	goMsg := cgo.AdoptCString(unsafe.Pointer(C.get_and_clear_last_error()))
	if goMsg == "" {
		return nil
	}
	return core.E("mlx.LastError", goMsg, nil)
}

// Eval synchronously evaluates arrays on the GPU.
// Use in code paths that need to propagate errors; see also Materialize.
//
//	if err := metal.Eval(logits); err != nil { return err }
func Eval(outputs ...*Array) error {
	Init()
	rc := evalOutputs(outputs, false)
	if rc != 0 {
		if err := LastError(); err != nil {
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
	rc := evalOutputs(outputs, true)
	if rc != 0 {
		if err := LastError(); err != nil {
			return err
		}
		return core.E("mlx.EvalAsync", core.Sprintf("async eval failed (rc=%d)", rc), nil)
	}
	return nil
}

func evalOutputs(outputs []*Array, async bool) C.int {
	bufPtr := evalOutputCtxPool.Get().(*[]C.mlx_array)
	handles := (*bufPtr)[:0]
	for _, output := range outputs {
		if output != nil && output.Valid() {
			handles = append(handles, output.ctx)
		}
	}
	if len(handles) == 0 {
		*bufPtr = handles
		evalOutputCtxPool.Put(bufPtr)
		return 0
	}
	n := len(handles)
	ptr := &handles[0]
	var rc C.int
	bind := DefaultStream()
	onEvalWorker(func() {
		if async {
			rc = C.mlx_go_async_eval_data(ptr, C.size_t(n), bind.ctx)
		} else {
			rc = C.mlx_go_eval_data(ptr, C.size_t(n), bind.ctx)
		}
	})
	runtime.KeepAlive(outputs)
	runtime.KeepAlive(handles)
	handles = handles[:0]
	*bufPtr = handles
	evalOutputCtxPool.Put(bufPtr)
	return rc
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
