//go:build !(darwin && arm64) || nomlx

package mlx

// DeviceInfo holds Metal GPU hardware information.
type DeviceInfo struct {
	Architecture                 string
	MaxBufferLength              uint64
	MaxRecommendedWorkingSetSize uint64
	MemorySize                   uint64
}

// SetCacheLimit is a no-op on unsupported builds.
func SetCacheLimit(_ uint64) uint64 { return 0 }

// SetMemoryLimit is a no-op on unsupported builds.
func SetMemoryLimit(_ uint64) uint64 { return 0 }

// GetActiveMemory always reports zero on unsupported builds.
func GetActiveMemory() uint64 { return 0 }

// GetPeakMemory always reports zero on unsupported builds.
func GetPeakMemory() uint64 { return 0 }

// ClearCache is a no-op on unsupported builds.
func ClearCache() {}

// GetCacheMemory always reports zero on unsupported builds.
func GetCacheMemory() uint64 { return 0 }

// ResetPeakMemory is a no-op on unsupported builds.
func ResetPeakMemory() {}

// SetWiredLimit is a no-op on unsupported builds.
func SetWiredLimit(_ uint64) uint64 { return 0 }

// GetDeviceInfo returns zero values on unsupported builds.
func GetDeviceInfo() DeviceInfo { return DeviceInfo{} }
