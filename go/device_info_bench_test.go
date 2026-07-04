// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for device_info.go — safeRuntimeDeviceInfo.
// Per AX-11 — safeRuntimeDeviceInfo is invoked from
// metalCapabilityDeviceInfo (per CapabilityReport() call from the
// inference façade) and from memoryPlannerDeviceInfo
// (per applyMemoryPlanToLoadConfig() during LoadModel-with-AutoPlan).
// Both surfaces are touched on every Model.Load path, so the host-info
// fast path needs its alloc shape pinned. The bench exercises the
// default branch only (the in-code reportDeviceInfo gate unset → host
// sysctl path); the full MLX-device probe lives behind that gate because
// it can abort the process when the bundled metallib is not
// discoverable.
//
// Run:    go test -bench='BenchmarkDeviceInfo' -benchmem -run='^$' ./go

package mlx

import (
	"testing"
)

// Sinks defeat compiler DCE.
var (
	deviceInfoBenchSinkDevice DeviceInfo
)

// --- safeRuntimeDeviceInfo ---
// Default fast path — host-reported memory; no MLX/Metal init.

func BenchmarkDeviceInfo_SafeRuntimeDeviceInfo(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		deviceInfoBenchSinkDevice = safeRuntimeDeviceInfo()
	}
}
