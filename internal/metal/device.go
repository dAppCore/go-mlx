//go:build darwin && arm64

package metal

/*
#include "mlx/c/mlx.h"
*/
import "C"

import (
	"sync"

	"dappco.re/go/core"
)

// DeviceType is the MLX execution device used by the root-package API.
type DeviceType string

const (
	DeviceCPU DeviceType = "cpu"
	DeviceGPU DeviceType = "gpu"
)

var defaultDeviceMu sync.Mutex

func currentDefaultDevice() (DeviceType, error) {
	Init()
	dev := C.mlx_device_new()
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
	var kind C.mlx_device_type
	switch device {
	case DeviceCPU:
		kind = C.MLX_CPU
	case DeviceGPU:
		kind = C.MLX_GPU
	default:
		return core.E("metal.setDefaultDevice", "unsupported device: "+string(device), nil)
	}

	dev := C.mlx_device_new_type(kind, 0)
	defer C.mlx_device_free(dev)

	if rc := C.mlx_set_default_device(dev); rc != 0 {
		if err := lastError(); err != nil {
			return core.E("metal.setDefaultDevice", "set default device", err)
		}
		return core.E("metal.setDefaultDevice", "set default device", nil)
	}
	return nil
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
