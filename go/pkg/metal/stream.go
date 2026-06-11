// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

/*
#include "mlx/c/mlx.h"
#include "decode_bridge.h"

static const char* go_mlx_device_info_string(mlx_device_info info, const char* key) {
	const char* value = NULL;
	if (mlx_device_info_get_string(&value, info, key) != 0) {
		return NULL;
	}
	return value;
}

static size_t go_mlx_device_info_size(mlx_device_info info, const char* key) {
	size_t value = 0;
	if (mlx_device_info_get_size(&value, info, key) != 0) {
		return 0;
	}
	return value;
}

static const char* go_mlx_device_info_name(mlx_device_info info) {
	return go_mlx_device_info_string(info, "device_name");
}

static const char* go_mlx_device_info_architecture(mlx_device_info info) {
	return go_mlx_device_info_string(info, "architecture");
}

static size_t go_mlx_device_info_max_buffer_length(mlx_device_info info) {
	return go_mlx_device_info_size(info, "max_buffer_length");
}

static size_t go_mlx_device_info_max_recommended_working_set_size(mlx_device_info info) {
	return go_mlx_device_info_size(info, "max_recommended_working_set_size");
}

static size_t go_mlx_device_info_memory_size(mlx_device_info info) {
	return go_mlx_device_info_size(info, "memory_size");
}
*/
import "C"

import (
	"runtime"
	"sync"

	core "dappco.re/go"
)

// Stream wraps an mlx_stream handle for dispatching operations.
type Stream struct {
	ctx C.mlx_stream
}

var (
	defaultStream     *Stream
	defaultStreamOnce sync.Once

	defaultGPUStream     *Stream
	defaultGPUStreamOnce sync.Once

	defaultCPUStream     *Stream
	defaultCPUStreamOnce sync.Once

	defaultStreamOverrideMu sync.RWMutex
	defaultStreamOverride   *Stream
	defaultStreamContextMu  sync.Mutex
)

// DefaultStream returns the default stream for the current default device.
//
//	C.mlx_zeros(&out.ctx, ..., metal.DefaultStream().ctx)
func DefaultStream() *Stream {
	defaultStreamOverrideMu.RLock()
	override := defaultStreamOverride
	defaultStreamOverrideMu.RUnlock()
	if override != nil && override.ctx.ctx != nil {
		return override
	}
	defaultStreamOnce.Do(func() {
		defaultStream = &Stream{}
	})
	if device, err := currentDefaultDevice(); err == nil && device == DeviceCPU {
		return DefaultCPUStream()
	}
	return DefaultGPUStream()
}

// DefaultGPUStream returns the cached default GPU stream.
//
//	s := metal.DefaultGPUStream()
func DefaultGPUStream() *Stream {
	defaultGPUStreamOnce.Do(func() {
		Init()
		defaultGPUStream = &Stream{ctx: C.mlx_default_gpu_stream_new()}
		registerEngineStream(defaultGPUStream)
	})
	return defaultGPUStream
}

// DefaultCPUStream returns the cached default CPU stream.
//
//	s := metal.DefaultCPUStream() // used for CPU-side tensor loads
func DefaultCPUStream() *Stream {
	defaultCPUStreamOnce.Do(func() {
		Init()
		defaultCPUStream = &Stream{ctx: C.mlx_default_cpu_stream_new()}
		registerEngineStream(defaultCPUStream)
	})
	return defaultCPUStream
}

func withTemporaryDefaultStream(device DeviceType, fn func()) error {
	if fn == nil {
		return nil
	}
	if device == "" {
		device = DeviceGPU
	}
	stream, err := newStreamForDevice(device)
	if err != nil {
		return err
	}
	defer C.mlx_stream_free(stream.ctx)

	previous, err := currentDefaultStreamForDevice(device)
	if err != nil {
		return err
	}
	defer C.mlx_stream_free(previous.ctx)

	defaultStreamContextMu.Lock()
	defer defaultStreamContextMu.Unlock()

	if rc := C.mlx_set_default_stream(stream.ctx); rc != 0 {
		if err := LastError(); err != nil {
			return core.E("metal.withTemporaryDefaultStream", "set default stream", err)
		}
		return core.E("metal.withTemporaryDefaultStream", "set default stream", nil)
	}
	defaultStreamOverrideMu.Lock()
	defaultStreamOverride = stream
	defaultStreamOverrideMu.Unlock()
	defer func() {
		defaultStreamOverrideMu.Lock()
		defaultStreamOverride = nil
		defaultStreamOverrideMu.Unlock()
		if rc := C.mlx_set_default_stream(previous.ctx); rc != 0 {
			if err := LastError(); err != nil {
				core.Error("mlx: restore default stream", "error", err)
			}
		}
	}()

	fn()
	return nil
}

func newStreamForDevice(device DeviceType) (*Stream, error) {
	dev, err := newCDevice(device)
	if err != nil {
		return nil, err
	}
	defer C.mlx_device_free(dev)

	stream := &Stream{ctx: C.mlx_stream_new_device(dev)}
	if stream.ctx.ctx == nil {
		if err := LastError(); err != nil {
			return nil, core.E("metal.newStreamForDevice", "new stream", err)
		}
		return nil, core.E("metal.newStreamForDevice", "new stream", nil)
	}
	registerEngineStream(stream)
	return stream, nil
}

func currentDefaultStreamForDevice(device DeviceType) (*Stream, error) {
	Init()
	switch device {
	case DeviceCPU:
		stream := &Stream{ctx: C.mlx_default_cpu_stream_new()}
		if stream.ctx.ctx == nil {
			if err := LastError(); err != nil {
				return nil, core.E("metal.currentDefaultStreamForDevice", "cpu stream", err)
			}
			return nil, core.E("metal.currentDefaultStreamForDevice", "cpu stream", nil)
		}
		return stream, nil
	case DeviceGPU, "":
		stream := &Stream{ctx: C.mlx_default_gpu_stream_new()}
		if stream.ctx.ctx == nil {
			if err := LastError(); err != nil {
				return nil, core.E("metal.currentDefaultStreamForDevice", "gpu stream", err)
			}
			return nil, core.E("metal.currentDefaultStreamForDevice", "gpu stream", nil)
		}
		return stream, nil
	default:
		return nil, core.E("metal.currentDefaultStreamForDevice", "unsupported device: "+string(device), nil)
	}
}

// Synchronize waits for all pending operations on the stream to complete.
//
//	metal.Synchronize(metal.DefaultStream())
func Synchronize(s *Stream) {
	ensureThreadStreams()
	C.mlx_synchronize(s.ctx)
}

// SetMemoryLimit sets the Metal memory limit. Returns the previous limit.
//
//	prev := metal.SetMemoryLimit(32 << 30) // 32 GB hard limit
func SetMemoryLimit(limit uint64) uint64 {
	if !MetalAvailable() {
		return 0
	}
	var prev C.size_t
	C.mlx_set_memory_limit(&prev, C.size_t(limit))
	return uint64(prev)
}

// SetCacheLimit sets the Metal cache limit. Returns the previous limit.
//
//	prev := metal.SetCacheLimit(4 << 30) // 4 GB cache limit
func SetCacheLimit(limit uint64) uint64 {
	if !MetalAvailable() {
		return 0
	}
	var prev C.size_t
	C.mlx_set_cache_limit(&prev, C.size_t(limit))
	return uint64(prev)
}

// GetActiveMemory returns the current Metal memory usage in bytes.
//
//	fmt.Printf("active: %d MB\n", metal.GetActiveMemory()/1024/1024)
func GetActiveMemory() uint64 {
	if !MetalAvailable() {
		return 0
	}
	var mem C.size_t
	C.mlx_get_active_memory(&mem)
	return uint64(mem)
}

// GetPeakMemory returns the peak Metal memory usage in bytes.
//
//	fmt.Printf("peak: %d MB\n", metal.GetPeakMemory()/1024/1024)
func GetPeakMemory() uint64 {
	if !MetalAvailable() {
		return 0
	}
	var mem C.size_t
	C.mlx_get_peak_memory(&mem)
	return uint64(mem)
}

// ClearCache releases Metal memory held in the MLX allocator cache.
//
//	metal.ClearCache() // between chat turns to reclaim prompt cache memory
func ClearCache() {
	if !MetalAvailable() {
		return
	}
	clearCacheNoCheck()
}

func clearCacheNoCheck() {
	C.mlx_clear_cache()
}

// GetCacheMemory returns the current Metal cache memory in bytes.
//
//	fmt.Printf("cache: %d MB\n", metal.GetCacheMemory()/1024/1024)
func GetCacheMemory() uint64 {
	if !MetalAvailable() {
		return 0
	}
	var mem C.size_t
	C.mlx_get_cache_memory(&mem)
	return uint64(mem)
}

// ResetPeakMemory resets the peak memory high-water mark to zero.
//
//	metal.ResetPeakMemory() // before each generate call to measure per-call peak
func ResetPeakMemory() {
	if !MetalAvailable() {
		return
	}
	C.mlx_reset_peak_memory()
}

// SetWiredLimit sets the Metal wired memory limit. Returns the previous limit.
//
//	prev := metal.SetWiredLimit(8 << 30) // 8 GB wired memory limit
func SetWiredLimit(limit uint64) uint64 {
	if !MetalAvailable() {
		return 0
	}
	var prev C.size_t
	C.mlx_set_wired_limit(&prev, C.size_t(limit))
	return uint64(prev)
}

// DeviceInfo holds Metal GPU hardware information.
type DeviceInfo struct {
	Name                         string
	Architecture                 string
	MaxBufferLength              uint64
	MaxRecommendedWorkingSetSize uint64
	MemorySize                   uint64
}

// HostDeviceInfo returns host-reported Apple GPU memory without initialising
// MLX or checking bundled metallib availability.
func HostDeviceInfo() DeviceInfo { return hostDeviceInfo() }

// GetDeviceInfo returns Metal GPU hardware information.
func GetDeviceInfo() DeviceInfo {
	host := hostDeviceInfo()
	if !MetalAvailable() {
		return host
	}
	dev, err := newCDevice(DeviceGPU)
	if err != nil {
		return host
	}
	defer C.mlx_device_free(dev)
	info := C.mlx_device_info_new()
	defer C.mlx_device_info_free(info)
	if rc := C.mlx_device_info_get(&info, dev); rc != 0 {
		return host
	}
	device := DeviceInfo{
		Name:                         C.GoString(C.go_mlx_device_info_name(info)),
		Architecture:                 C.GoString(C.go_mlx_device_info_architecture(info)),
		MaxBufferLength:              uint64(C.go_mlx_device_info_max_buffer_length(info)),
		MaxRecommendedWorkingSetSize: uint64(C.go_mlx_device_info_max_recommended_working_set_size(info)),
		MemorySize:                   uint64(C.go_mlx_device_info_memory_size(info)),
	}
	if device.Name == "" {
		device.Name = host.Name
	}
	if device.Architecture == "" {
		device.Architecture = host.Architecture
	}
	if device.MaxBufferLength == 0 {
		device.MaxBufferLength = host.MaxBufferLength
	}
	if device.MaxRecommendedWorkingSetSize == 0 {
		device.MaxRecommendedWorkingSetSize = host.MaxRecommendedWorkingSetSize
	}
	if device.MemorySize == 0 {
		device.MemorySize = host.MemorySize
	}
	return device
}

// Thread-encoder registry — MLX 0.31.2 encodes GPU graphs on the CALLING
// thread with per-thread command encoders. Every stream the engine creates
// registers here; ensureThreadStreams replays the registrations on whichever
// OS thread is about to eval (idempotent try_emplace inside the bridge), so
// goroutine migration can never land an eval on a thread without encoders.
var (
	threadStreamRegistryMu sync.Mutex
	threadStreamRegistry   []C.mlx_stream
)

func registerEngineStream(s *Stream) {
	if s == nil || s.ctx.ctx == nil {
		return
	}
	threadStreamRegistryMu.Lock()
	threadStreamRegistry = append(threadStreamRegistry, s.ctx)
	threadStreamRegistryMu.Unlock()
}

// ensureThreadStreams binds the current OS thread for GPU encoding: the
// default stream becomes the thread default and every registered stream
// gets its command encoder registered on this thread. Cheap (a few
// thread-local map probes) — called at every eval-class entry.
func ensureThreadStreams() {
	threadStreamRegistryMu.Lock()
	streams := threadStreamRegistry
	n := len(streams)
	threadStreamRegistryMu.Unlock()
	if n == 0 {
		return
	}
	rc := C.go_mlx_ensure_thread_streams(&streams[0], C.size_t(n))
	runtime.KeepAlive(streams)
	if rc != 0 {
		if err := LastError(); err != nil {
			core.Error("mlx: ensure thread streams", "error", err)
		}
	}
}
