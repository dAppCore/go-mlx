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
)

// DefaultStream returns the default GPU stream, creating it on first use.
func DefaultStream() *Stream {
	defaultStreamOnce.Do(func() {
		Init()
		defaultStream = &Stream{ctx: C.mlx_default_gpu_stream_new()}
	})
	return defaultStream
}

// DefaultGPUStream returns a new GPU stream.
func DefaultGPUStream() *Stream {
	Init()
	return &Stream{ctx: C.mlx_default_gpu_stream_new()}
}

// DefaultCPUStream returns a new CPU stream.
func DefaultCPUStream() *Stream {
	Init()
	return &Stream{ctx: C.mlx_default_cpu_stream_new()}
}

// Synchronize waits for all operations on the stream to complete.
func Synchronize(s *Stream) {
	C.mlx_synchronize(s.ctx)
}

// SetMemoryLimit sets the Metal memory limit. Returns the previous limit.
func SetMemoryLimit(limit uint64) uint64 {
	var prev C.size_t
	C.mlx_set_memory_limit(&prev, C.size_t(limit))
	return uint64(prev)
}

// SetCacheLimit sets the Metal cache limit. Returns the previous limit.
func SetCacheLimit(limit uint64) uint64 {
	var prev C.size_t
	C.mlx_set_cache_limit(&prev, C.size_t(limit))
	return uint64(prev)
}

// GetActiveMemory returns the current Metal memory usage in bytes.
func GetActiveMemory() uint64 {
	var mem C.size_t
	C.mlx_get_active_memory(&mem)
	return uint64(mem)
}

// GetPeakMemory returns the peak Metal memory usage in bytes.
func GetPeakMemory() uint64 {
	var mem C.size_t
	C.mlx_get_peak_memory(&mem)
	return uint64(mem)
}

// ClearCache releases Metal memory held in the MLX allocator cache.
func ClearCache() {
	C.mlx_clear_cache()
}

// GetCacheMemory returns the current Metal cache memory in bytes.
func GetCacheMemory() uint64 {
	var mem C.size_t
	C.mlx_get_cache_memory(&mem)
	return uint64(mem)
}

// ResetPeakMemory resets the peak memory high-water mark.
func ResetPeakMemory() {
	C.mlx_reset_peak_memory()
}

// SetWiredLimit sets the Metal wired memory limit. Returns the previous limit.
func SetWiredLimit(limit uint64) uint64 {
	var prev C.size_t
	C.mlx_set_wired_limit(&prev, C.size_t(limit))
	return uint64(prev)
}

// DeviceInfo holds Metal GPU hardware information.
type DeviceInfo struct {
	Architecture                string
	MaxBufferLength             uint64
	MaxRecommendedWorkingSetSize uint64
	MemorySize                  uint64
}

// GetDeviceInfo returns Metal GPU hardware information.
func GetDeviceInfo() DeviceInfo {
	info := C.mlx_metal_device_info()
	return DeviceInfo{
		Architecture:                C.GoString(&info.architecture[0]),
		MaxBufferLength:             uint64(info.max_buffer_length),
		MaxRecommendedWorkingSetSize: uint64(info.max_recommended_working_set_size),
		MemorySize:                  uint64(info.memory_size),
	}
}
