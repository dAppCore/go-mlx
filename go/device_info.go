// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	core "dappco.re/go"
	"dappco.re/go/mlx/internal/metal"
)

func safeRuntimeDeviceInfo() DeviceInfo {
	// mlx-c can abort the process when its bundled metallib is not discoverable.
	// Use host-reported memory for planning by default, and only opt into the
	// full native MLX device probe when the caller explicitly asks for it.
	if core.Env("GO_MLX_REPORT_DEVICE_INFO") != "1" {
		return metal.HostDeviceInfo()
	}
	return GetDeviceInfo()
}
