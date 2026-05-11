// SPDX-Licence-Identifier: EUPL-1.2

//go:build !darwin || !arm64 || nomlx

package mlx

func safeRuntimeDeviceInfo() DeviceInfo {
	return DeviceInfo{}
}
