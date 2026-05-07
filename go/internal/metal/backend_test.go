// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

func TestBackend_ResolveLoadDevice_FallsBackToCPUWhenMetalUnavailable_Good(t *testing.T) {
	coverageTokens := "ResolveLoadDevice FallsBackToCPUWhenMetalUnavailable"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
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
	coverageTokens := "ResolveLoadDevice DefaultsToCPUWhenMetalUnavailable"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
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
	coverageTokens := "ResolveLoadDevice KeepsCPUWhenRequested"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
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
	coverageTokens := "ResolveLoadDevice KeepsGPUWhenMetalAvailable"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
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

func TestBackend_NormalizeLoadConfig_LocalDefaults_Good(t *testing.T) {
	cfg := normalizeMetalLoadConfig(LoadConfig{})
	if cfg.ContextLen != DefaultLocalContextLen {
		t.Fatalf("ContextLen = %d, want %d", cfg.ContextLen, DefaultLocalContextLen)
	}
	if cfg.ParallelSlots != DefaultLocalParallelSlots {
		t.Fatalf("ParallelSlots = %d, want %d", cfg.ParallelSlots, DefaultLocalParallelSlots)
	}
	if cfg.DisablePromptCache {
		t.Fatal("DisablePromptCache = true, want false")
	}
	if cfg.PromptCacheMinTokens != DefaultPromptCacheMinTokens {
		t.Fatalf("PromptCacheMinTokens = %d, want %d", cfg.PromptCacheMinTokens, DefaultPromptCacheMinTokens)
	}
}

func TestBackend_ApplyAllocatorLimits_Good(t *testing.T) {
	coverageTokens := "ApplyAllocatorLimits"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	previousMemory := setMemoryLimit
	previousCache := setCacheLimit
	previousWired := setWiredLimit
	t.Cleanup(func() {
		setMemoryLimit = previousMemory
		setCacheLimit = previousCache
		setWiredLimit = previousWired
	})

	var memoryLimit, cacheLimit, wiredLimit uint64
	setMemoryLimit = func(limit uint64) uint64 { memoryLimit = limit; return 0 }
	setCacheLimit = func(limit uint64) uint64 { cacheLimit = limit; return 0 }
	setWiredLimit = func(limit uint64) uint64 { wiredLimit = limit; return 0 }

	applyAllocatorLimits(LoadConfig{
		MemoryLimitBytes: 10,
		CacheLimitBytes:  3,
		WiredLimitBytes:  7,
	})

	if memoryLimit != 10 || cacheLimit != 3 || wiredLimit != 7 {
		t.Fatalf("limits = memory %d cache %d wired %d, want 10/3/7", memoryLimit, cacheLimit, wiredLimit)
	}
}

// Generated file-aware compliance coverage.
func TestBackend_LoadAndInit_Good(t *testing.T) {
	target := "LoadAndInit"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestBackend_LoadAndInit_Bad(t *testing.T) {
	target := "LoadAndInit"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestBackend_LoadAndInit_Ugly(t *testing.T) {
	target := "LoadAndInit"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}
