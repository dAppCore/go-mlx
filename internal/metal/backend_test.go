// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

func TestBackend_ResolveLoadDevice_FallsBackToCPUWhenMetalUnavailable_Good(t *testing.T) {
	previous := runtimeMetalAvailable
	runtimeMetalAvailable = func() bool { return false }
	t.Cleanup(func() { runtimeMetalAvailable = previous })

	got, fellBack := resolveLoadDevice(DeviceGPU)
	if got != DeviceCPU {
		t.Fatalf("resolveLoadDevice(gpu) = %q, want cpu", got)
	}
	if !fellBack {
		t.Fatal("resolveLoadDevice(gpu) should report CPU fallback when Metal is unavailable")
	}
}

func TestBackend_ResolveLoadDevice_DefaultsToCPUWhenMetalUnavailable_Good(t *testing.T) {
	previous := runtimeMetalAvailable
	runtimeMetalAvailable = func() bool { return false }
	t.Cleanup(func() { runtimeMetalAvailable = previous })

	got, fellBack := resolveLoadDevice("")
	if got != DeviceCPU {
		t.Fatalf("resolveLoadDevice(\"\") = %q, want cpu", got)
	}
	if !fellBack {
		t.Fatal("resolveLoadDevice(\"\") should report CPU fallback when Metal is unavailable")
	}
}

func TestBackend_ResolveLoadDevice_KeepsCPUWhenRequested_Good(t *testing.T) {
	previous := runtimeMetalAvailable
	runtimeMetalAvailable = func() bool { return false }
	t.Cleanup(func() { runtimeMetalAvailable = previous })

	got, fellBack := resolveLoadDevice(DeviceCPU)
	if got != DeviceCPU {
		t.Fatalf("resolveLoadDevice(cpu) = %q, want cpu", got)
	}
	if fellBack {
		t.Fatal("resolveLoadDevice(cpu) should not report fallback")
	}
}

func TestBackend_ResolveLoadDevice_KeepsGPUWhenMetalAvailable_Good(t *testing.T) {
	previous := runtimeMetalAvailable
	runtimeMetalAvailable = func() bool { return true }
	t.Cleanup(func() { runtimeMetalAvailable = previous })

	got, fellBack := resolveLoadDevice(DeviceGPU)
	if got != DeviceGPU {
		t.Fatalf("resolveLoadDevice(gpu) = %q, want gpu", got)
	}
	if fellBack {
		t.Fatal("resolveLoadDevice(gpu) should not report fallback when Metal is available")
	}
}
