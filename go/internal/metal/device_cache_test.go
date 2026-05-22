// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"sync"
	"testing"
)

// TestDeviceCache_LazyFill verifies the cache populates on first read and
// returns the same DeviceType on subsequent reads without touching the
// C-side MLX state.
func TestDeviceCache_LazyFill(t *testing.T) {
	resetDefaultDeviceCache()

	first, err := currentDefaultDevice()
	if err != nil {
		t.Fatalf("first currentDefaultDevice: %v", err)
	}
	if first != DeviceCPU && first != DeviceGPU {
		t.Fatalf("first currentDefaultDevice = %q, want cpu or gpu", first)
	}
	if cached := cachedDefaultDevice.Load(); cached == nil || *cached != first {
		t.Fatalf("cache not populated after first read, got %v want pointer to %q", cached, first)
	}

	second, err := currentDefaultDevice()
	if err != nil {
		t.Fatalf("second currentDefaultDevice: %v", err)
	}
	if second != first {
		t.Fatalf("cache returned %q after %q", second, first)
	}
}

// TestDeviceCache_TracksSetDefaultDevice verifies that setDefaultDevice
// updates the memoised value so subsequent currentDefaultDevice() calls
// reflect the post-swap C-side state. This is the invariant withDefaultDevice
// relies on when it toggles the device between Lock/Unlock.
func TestDeviceCache_TracksSetDefaultDevice(t *testing.T) {
	resetDefaultDeviceCache()

	original, err := currentDefaultDevice()
	if err != nil {
		t.Fatalf("baseline currentDefaultDevice: %v", err)
	}

	// Always restore the original device on exit so other tests are not
	// disturbed.
	defer func() {
		if err := setDefaultDevice(original); err != nil {
			t.Logf("restore original device: %v", err)
		}
	}()

	target := DeviceGPU
	if original == DeviceGPU {
		target = DeviceCPU
	}

	if err := setDefaultDevice(target); err != nil {
		// Some headless macOS environments have no usable GPU; skip
		// the swap-direction test if MLX rejects the target type.
		t.Skipf("setDefaultDevice(%q) rejected: %v", target, err)
	}

	got, err := currentDefaultDevice()
	if err != nil {
		t.Fatalf("currentDefaultDevice after swap: %v", err)
	}
	if got != target {
		t.Fatalf("cache stale: currentDefaultDevice = %q, want %q", got, target)
	}
}

// TestDeviceCache_ConcurrentReadIsRaceFree exercises the atomic.Pointer
// read path under -race; without the cache the per-call cgo round-trip
// is naturally race-free, but the cache adds Go-side shared state we need
// to keep race-clean.
func TestDeviceCache_ConcurrentReadIsRaceFree(t *testing.T) {
	resetDefaultDeviceCache()

	const goroutines = 16
	const iterations = 1024

	var wg sync.WaitGroup
	wg.Add(goroutines)
	for range goroutines {
		go func() {
			defer wg.Done()
			for range iterations {
				if _, err := currentDefaultDevice(); err != nil {
					t.Errorf("concurrent currentDefaultDevice: %v", err)
					return
				}
			}
		}()
	}
	wg.Wait()
}
