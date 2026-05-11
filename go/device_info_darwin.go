// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && !nomlx

package mlx

import core "dappco.re/go"

func safeRuntimeDeviceInfo() DeviceInfo {
	// mlx-c can abort the process when its bundled metallib is not discoverable.
	// Capability and fit-planning reports must stay safe in package tests and
	// headless agent runs, so callers opt into native device probing explicitly.
	if core.Env("GO_MLX_REPORT_DEVICE_INFO") != "1" {
		return DeviceInfo{}
	}
	return GetDeviceInfo()
}
