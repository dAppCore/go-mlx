// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"testing"
)

func TestStreams_DefaultStreamsAreUsable_Good(t *testing.T) {
	Init()

	_ = LastError()

	cpu := DefaultCPUStream()
	if cpu == nil || cpu.ctx.ctx == nil {
		host := HostDeviceInfo()
		if err := LastError(); err != nil {
			t.Fatalf("DefaultCPUStream() returned nil stream: %v; host=%+v", err, host)
		}
		t.Fatalf("DefaultCPUStream() returned nil stream; host=%+v", host)
	}

	gpu := DefaultGPUStream()
	if gpu == nil || gpu.ctx.ctx == nil {
		host := HostDeviceInfo()
		if err := LastError(); err != nil {
			t.Fatalf("DefaultGPUStream() returned nil stream: %v; host=%+v", err, host)
		}
		t.Fatalf("DefaultGPUStream() returned nil stream; host=%+v", host)
	}
}

// TestStream_Synchronize_Good: Synchronize drains the default stream's queue
// after a tiny synthetic op — it must return without error on a usable device.
// (SetMemoryLimit/SetWiredLimit are deliberately not covered here: they mutate
// process-global Metal allocator limits and the live window between set and
// restore could perturb the package's flaky LastError test. Their no-device
// guard is exercised below instead.)
func TestStream_Synchronize_Good(t *testing.T) {
	requireMetalRuntime(t)

	a := FromValue(float32(1.5))
	defer Free(a)
	Materialize(a)
	Synchronize(DefaultStream()) // must not panic / deadlock on a drained queue
	if err := LastError(); err != nil {
		t.Fatalf("LastError after Synchronize = %v, want nil", err)
	}
}

// TestStream_DeviceInfo_Good: HostDeviceInfo reports Apple GPU memory without
// initialising MLX, and GetDeviceInfo returns at least the host fields on a
// usable device. Both are read-only hardware queries — no model, no allocation.
func TestStream_DeviceInfo_Good(t *testing.T) {
	requireMetalRuntime(t)

	host := HostDeviceInfo()
	if host.MemorySize == 0 {
		t.Errorf("HostDeviceInfo().MemorySize = 0, want a non-zero host memory size")
	}

	dev := GetDeviceInfo()
	// GetDeviceInfo falls back to the host info when MLX is unavailable, so it is
	// never weaker than the host report on the memory fields.
	if dev.MemorySize == 0 {
		t.Errorf("GetDeviceInfo().MemorySize = 0, want a non-zero size")
	}
	if dev.Name == "" && host.Name == "" {
		t.Error("neither GetDeviceInfo nor HostDeviceInfo reported a device name")
	}
}

// TestStream_SetMemoryLimit_NoDevice_Bad documents that the limit setters are
// covered only on their no-device guard: when Metal is unavailable they return
// 0 without touching the allocator. On a real device they are skipped (see the
// rationale on TestStream_Synchronize_Good — the live global-limit window is the
// hazard, not AX-11). This keeps the setters from ever running with a live limit
// inside the shared test binary.
func TestStream_SetMemoryLimit_NoDevice_Bad(t *testing.T) {
	if MetalAvailable() {
		t.Skip("device present: limit setters mutate global Metal state; covered only via the no-device guard")
	}
	if got := SetMemoryLimit(1 << 40); got != 0 {
		t.Errorf("SetMemoryLimit with no device = %d, want 0", got)
	}
	if got := SetWiredLimit(1 << 40); got != 0 {
		t.Errorf("SetWiredLimit with no device = %d, want 0", got)
	}
}
