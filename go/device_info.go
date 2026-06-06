// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import "dappco.re/go/mlx/pkg/metal"

// reportDeviceInfoGate opts into the full native MLX device probe (which logs
// device info) instead of the host-reported memory used for planning. It is an
// in-code diagnostic — off by default, NEVER ambient env: a debug knob belongs
// in the system (set it in code / a test), not in external process env where any
// parent could flip it.
var reportDeviceInfoGate = false

func reportDeviceInfo() bool {
	return reportDeviceInfoGate
}

func safeRuntimeDeviceInfo() DeviceInfo {
	// mlx-c can abort the process when its bundled metallib is not discoverable.
	// Use host-reported memory for planning by default, and only opt into the
	// full native MLX device probe when reportDeviceInfoGate is set in code.
	if !reportDeviceInfo() {
		return metal.HostDeviceInfo()
	}
	return GetDeviceInfo()
}
