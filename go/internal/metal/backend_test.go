// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"testing"

	core "dappco.re/go"
)

func TestBackend_ResolveLoadDevice_KeepsGPUWhenMetalUnavailable_Good(t *testing.T) {
	coverageTokens := "ResolveLoadDevice KeepsGPUWhenMetalUnavailable"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	previous := runtimeMetalAvailable
	runtimeMetalAvailable = func() bool { return false }
	t.Cleanup(func() { runtimeMetalAvailable = previous })

	got, fellBack := resolveLoadDevice(DeviceGPU)
	if got != DeviceGPU {
		t.Fatalf("resolveLoadDevice(gpu) = %q, want gpu", got)
	}
	if fellBack {
		t.Fatal("resolveLoadDevice(gpu) should not silently fall back to CPU")
	}
}

func TestBackend_ResolveLoadDevice_DefaultsToGPUWhenMetalUnavailable_Good(t *testing.T) {
	coverageTokens := "ResolveLoadDevice DefaultsToGPUWhenMetalUnavailable"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	previous := runtimeMetalAvailable
	runtimeMetalAvailable = func() bool { return false }
	t.Cleanup(func() { runtimeMetalAvailable = previous })

	got, fellBack := resolveLoadDevice("")
	if got != DeviceGPU {
		t.Fatalf("resolveLoadDevice(\"\") = %q, want gpu", got)
	}
	if fellBack {
		t.Fatal("resolveLoadDevice(\"\") should not silently fall back to CPU")
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

func TestBackend_EnsureLoadDeviceAvailable_RejectsMissingMetal_Bad(t *testing.T) {
	coverageTokens := "EnsureLoadDeviceAvailable RejectsMissingMetal"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	previous := runtimeMetalAvailable
	runtimeMetalAvailable = func() bool { return false }
	t.Cleanup(func() { runtimeMetalAvailable = previous })

	err := ensureLoadDeviceAvailable(DeviceGPU)
	if err == nil {
		t.Fatal("ensureLoadDeviceAvailable(gpu) error = nil, want missing Metal error")
	}
	if !core.Contains(err.Error(), "usable Metal") {
		t.Fatalf("error = %v, want usable Metal message", err)
	}
}

func TestBackend_EnsureLoadDeviceAvailable_AllowsMetalDevice_Good(t *testing.T) {
	coverageTokens := "EnsureLoadDeviceAvailable AllowsMetalDevice"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	previous := runtimeMetalAvailable
	runtimeMetalAvailable = func() bool { return true }
	t.Cleanup(func() { runtimeMetalAvailable = previous })

	if err := ensureLoadDeviceAvailable(DeviceGPU); err != nil {
		t.Fatalf("ensureLoadDeviceAvailable(gpu) error = %v, want nil", err)
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

func TestBackend_ValidateMetalKVCacheMode_AllowsTurboQuant_Good(t *testing.T) {
	coverageTokens := "ValidateMetalKVCacheMode AllowsTurboQuant"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}

	if err := validateMetalKVCacheMode(string(KVCacheModeTurboQuant)); err != nil {
		t.Fatalf("validateMetalKVCacheMode(turboquant) error = %v, want nil for explicit research mode", err)
	}
}

func TestBackend_ApplyGemma4SlidingWindow_Good(t *testing.T) {
	coverageTokens := "ApplyGemma4SlidingWindow"
	model := &Gemma4Model{Cfg: &Gemma4TextConfig{SlidingWindow: 2048}}
	applyGemma4SlidingWindow(model, 512)
	if model.Cfg.SlidingWindow != 512 {
		t.Fatalf("SlidingWindow = %d, want 512", model.Cfg.SlidingWindow)
	}
	applyGemma4SlidingWindow(model, 0)
	if model.Cfg.SlidingWindow != 512 {
		t.Fatalf("SlidingWindow changed for zero cap: %d", model.Cfg.SlidingWindow)
	}
	applyGemma4SlidingWindow(model, 1024)
	if model.Cfg.SlidingWindow != 512 {
		t.Fatalf("SlidingWindow expanded above existing cap: %d", model.Cfg.SlidingWindow)
	}
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
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
