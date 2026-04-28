// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

/*
#include "mlx/c/mlx.h"
*/
import "C"

import "sync"

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
)

// DefaultStream returns the default stream for the current default device.
//
//	C.mlx_zeros(&out.ctx, ..., metal.DefaultStream().ctx)
func DefaultStream() *Stream {
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
	})
	return defaultCPUStream
}

// Synchronize waits for all pending operations on the stream to complete.
//
//	metal.Synchronize(metal.DefaultStream())
func Synchronize(s *Stream) {
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
	Architecture                 string
	MaxBufferLength              uint64
	MaxRecommendedWorkingSetSize uint64
	MemorySize                   uint64
}

// GetDeviceInfo returns Metal GPU hardware information.
func GetDeviceInfo() DeviceInfo {
	if !MetalAvailable() {
		return DeviceInfo{}
	}
	info := C.mlx_metal_device_info()
	return DeviceInfo{
		Architecture:                 C.GoString(&info.architecture[0]),
		MaxBufferLength:              uint64(info.max_buffer_length),
		MaxRecommendedWorkingSetSize: uint64(info.max_recommended_working_set_size),
		MemorySize:                   uint64(info.memory_size),
	}
}
