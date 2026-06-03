// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

/*
#include "mlx/c/mlx.h"
*/
import "C"

import (
	"sync"
	"sync/atomic"

	"dappco.re/go"
)

// DeviceType is the MLX execution device used by the root-package API.
type DeviceType string

const (
	DeviceCPU DeviceType = "cpu"
	DeviceGPU DeviceType = "gpu"
)

var defaultDeviceMu sync.Mutex

// cachedDefaultDevice memoises the result of currentDefaultDevice across
// the hot MLX op path (Slice, SliceUpdate, AsType, Zeros, etc.) to avoid
// the cgo round-trip and defer record for C.mlx_device_free on every call.
//
// Lifetime contract:
//   - DefaultStream() resolves the default device on every MLX op; without
//     this cache each resolution allocates a defer record and pays two cgo
//     calls (mlx_get_default_device + mlx_device_get_type).
//   - The default device is mutated only via setDefaultDevice, which is
//     called exclusively from withDefaultDevice under defaultDeviceMu.
//     setDefaultDevice updates the cache after a successful C-side swap so
//     subsequent reads return the post-swap value.
//   - The cache stores *DeviceType so a nil pointer is the "not yet loaded"
//     sentinel; the first successful read populates it under a one-shot
//     mutex to coalesce the racing initial cgo round-trips.
var (
	cachedDefaultDevice atomic.Pointer[DeviceType]
	cachedDefaultLoadMu sync.Mutex
)

// resetDefaultDeviceCache clears the memoised currentDefaultDevice value.
// Test-only — production callers rely on setDefaultDevice keeping the
// cache in sync with the C-side state.
func resetDefaultDeviceCache() {
	cachedDefaultDevice.Store(nil)
}

func currentDefaultDevice() (DeviceType, error) {
	if cached := cachedDefaultDevice.Load(); cached != nil {
		return *cached, nil
	}
	return loadDefaultDevice()
}

// loadDefaultDevice is the slow path — it issues the cgo calls to discover
// the current MLX default device and populates the package-private cache.
// Subsequent currentDefaultDevice calls return the cached value without
// touching cgo until setDefaultDevice or resetDefaultDeviceCache invalidates.
func loadDefaultDevice() (DeviceType, error) {
	cachedDefaultLoadMu.Lock()
	defer cachedDefaultLoadMu.Unlock()
	if cached := cachedDefaultDevice.Load(); cached != nil {
		return *cached, nil
	}
	device, err := readDefaultDeviceFromC()
	if err != nil {
		return "", err
	}
	cachedDefaultDevice.Store(&device)
	return device, nil
}

// readDefaultDeviceFromC fetches the current default device type via the
// MLX C-API. Used by the cache-fill slow path and after setDefaultDevice
// to refresh the cache.
func readDefaultDeviceFromC() (DeviceType, error) {
	Init()
	var dev C.mlx_device
	defer C.mlx_device_free(dev)

	if rc := C.mlx_get_default_device(&dev); rc != 0 {
		if err := lastError(); err != nil {
			return "", core.E("metal.currentDefaultDevice", "get default device", err)
		}
		return "", core.E("metal.currentDefaultDevice", "get default device", nil)
	}

	var kind C.mlx_device_type
	if rc := C.mlx_device_get_type(&kind, dev); rc != 0 {
		if err := lastError(); err != nil {
			return "", core.E("metal.currentDefaultDevice", "get default device type", err)
		}
		return "", core.E("metal.currentDefaultDevice", "get default device type", nil)
	}

	switch kind {
	case C.MLX_CPU:
		return DeviceCPU, nil
	case C.MLX_GPU:
		return DeviceGPU, nil
	default:
		return "", core.E("metal.currentDefaultDevice", "unknown device type", nil)
	}
}

func setDefaultDevice(device DeviceType) error {
	Init()
	dev, err := newCDevice(device)
	if err != nil {
		return core.E("metal.setDefaultDevice", "device", err)
	}
	defer C.mlx_device_free(dev)

	if rc := C.mlx_set_default_device(dev); rc != 0 {
		if err := lastError(); err != nil {
			return core.E("metal.setDefaultDevice", "set default device", err)
		}
		return core.E("metal.setDefaultDevice", "set default device", nil)
	}
	// Keep the memoised default device aligned with the post-swap C-side
	// state — withDefaultDevice toggles this twice per nested call.
	stored := device
	cachedDefaultDevice.Store(&stored)
	return nil
}

func newCDevice(device DeviceType) (C.mlx_device, error) {
	Init()
	var kind C.mlx_device_type
	switch device {
	case DeviceCPU:
		kind = C.MLX_CPU
	case DeviceGPU:
		kind = C.MLX_GPU
	default:
		return C.mlx_device{}, core.E("metal.newCDevice", "unsupported device: "+string(device), nil)
	}
	dev := C.mlx_device_new_type(kind, 0)
	if dev.ctx == nil {
		if err := lastError(); err != nil {
			return C.mlx_device{}, core.E("metal.newCDevice", "create device", err)
		}
		return C.mlx_device{}, core.E("metal.newCDevice", "create device", nil)
	}
	return dev, nil
}

func withDefaultDevice(device DeviceType, fn func()) error {
	if device == "" {
		device = DeviceGPU
	}

	defaultDeviceMu.Lock()
	defer defaultDeviceMu.Unlock()

	prev, err := currentDefaultDevice()
	if err != nil {
		return err
	}
	if prev != device {
		if err := setDefaultDevice(device); err != nil {
			return err
		}
		defer func() {
			if err := setDefaultDevice(prev); err != nil {
				core.Error("mlx: restore default device", "error", err)
			}
		}()
	}

	fn()
	return nil
}

func (m *Model) modelDevice() DeviceType {
	if m == nil || m.device == "" {
		return DeviceGPU
	}
	return m.device
}

func (m *Model) withDevice(fn func()) error {
	return withDefaultDevice(m.modelDevice(), fn)
}
