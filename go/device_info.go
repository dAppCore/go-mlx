// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"sync"

	core "dappco.re/go"
	"dappco.re/go/mlx/internal/metal"
)

// reportDeviceInfoOnce caches the GO_MLX_REPORT_DEVICE_INFO probe gate
// across the process lifetime — it's a startup-time config knob, not a
// per-call decision. safeRuntimeDeviceInfo is invoked from every Model.Load
// path (capability check + memory planner), so the env lookup was being
// re-done thousands of times for a value that never changes.
var (
	reportDeviceInfoOnce sync.Once
	reportDeviceInfoGate bool
)

func reportDeviceInfo() bool {
	reportDeviceInfoOnce.Do(func() {
		reportDeviceInfoGate = core.Env("GO_MLX_REPORT_DEVICE_INFO") == "1"
	})
	return reportDeviceInfoGate
}

func safeRuntimeDeviceInfo() DeviceInfo {
	// mlx-c can abort the process when its bundled metallib is not discoverable.
	// Use host-reported memory for planning by default, and only opt into the
	// full native MLX device probe when the caller explicitly asks for it.
	if !reportDeviceInfo() {
		return metal.HostDeviceInfo()
	}
	return GetDeviceInfo()
}
