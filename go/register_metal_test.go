// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"testing"

	"dappco.re/go/inference"
	"dappco.re/go/mlx/memory"
	"dappco.re/go/mlx/pkg/metal"
)

func TestMetalBackendLoadModel_ForwardsCPUDeviceWhenGPULayersZero_Good(t *testing.T) {
	original := loadBackendModel
	t.Cleanup(func() { loadBackendModel = original })

	var got metal.LoadConfig
	loadBackendModel = func(_ string, cfg metal.LoadConfig) (*metal.Model, error) {
		got = cfg
		return &metal.Model{}, nil
	}

	backend := &metalbackend{}
	if _, err := backend.LoadModel("/tmp/model", inference.WithGPULayers(0)); err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	if got.Device != metal.DeviceCPU {
		t.Fatalf("device = %q, want %q", got.Device, metal.DeviceCPU)
	}
}

func TestMetalBackendLoadModel_ForwardsParallelSlots_Good(t *testing.T) {
	original := loadBackendModel
	t.Cleanup(func() { loadBackendModel = original })

	var got metal.LoadConfig
	loadBackendModel = func(_ string, cfg metal.LoadConfig) (*metal.Model, error) {
		got = cfg
		return &metal.Model{}, nil
	}

	backend := &metalbackend{}
	if _, err := backend.LoadModel("/tmp/model", inference.WithParallelSlots(4)); err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	if got.ParallelSlots != 4 {
		t.Fatalf("ParallelSlots = %d, want 4", got.ParallelSlots)
	}
}

func TestMetalBackendLoadModel_ForwardsPlannerCacheMode_Good(t *testing.T) {
	originalLoad := loadBackendModel
	originalDeviceInfo := memoryPlannerDeviceInfo
	t.Cleanup(func() {
		loadBackendModel = originalLoad
		memoryPlannerDeviceInfo = originalDeviceInfo
	})

	memoryPlannerDeviceInfo = func() DeviceInfo {
		return DeviceInfo{
			Architecture:                 "apple9",
			MemorySize:                   96 << 30,
			MaxRecommendedWorkingSetSize: 90 << 30,
		}
	}
	var got metal.LoadConfig
	loadBackendModel = func(_ string, cfg metal.LoadConfig) (*metal.Model, error) {
		got = cfg
		return &metal.Model{}, nil
	}

	backend := &metalbackend{}
	if _, err := backend.LoadModel("/tmp/model"); err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	if got.CachePolicy != string(memory.KVCacheRotating) || got.KVCacheMode != string(memory.KVCacheModePaged) {
		t.Fatalf("cache = %q/%q, want planner paged cache", got.CachePolicy, got.KVCacheMode)
	}
}

func TestRegisterMetal_RuntimeWrappersSmoke_Good(t *testing.T) {
	_ = Available()
	_ = GetActiveMemory()
	_ = GetPeakMemory()
	_ = GetCacheMemory()
	_ = GetDeviceInfo()
	ClearCache()
	ResetPeakMemory()

	previousCache := SetCacheLimit(0)
	_ = SetCacheLimit(previousCache)
	previousMemory := SetMemoryLimit(0)
	_ = SetMemoryLimit(previousMemory)
	previousWired := SetWiredLimit(0)
	_ = SetWiredLimit(previousWired)
}

func TestRegisterMetalScheduler_NilAdapter_Bad(t *testing.T) {
	var adapter *metaladapter
	_, _, err := adapter.Schedule(context.Background(), inference.ScheduledRequest{Prompt: "x"})
	if err == nil {
		t.Fatal("Schedule(nil adapter) error = nil")
	}
	result, err := adapter.CancelRequest(context.Background(), "missing")
	if err != nil {
		t.Fatalf("CancelRequest(nil adapter) error = %v", err)
	}
	if result.Reason != "not_found" {
		t.Fatalf("CancelRequest(nil adapter) = %+v, want not_found", result)
	}
}

func TestRegisterMetalCache_NilAdapter_GoodBad(t *testing.T) {
	var adapter *metaladapter
	stats, err := adapter.CacheStats(context.Background())
	if err != nil {
		t.Fatalf("CacheStats(nil adapter) error = %v", err)
	}
	if stats.Labels["block_size"] != "512" || stats.CacheMode == "" {
		t.Fatalf("CacheStats = %+v, want default block-prefix labels", stats)
	}
	entries, err := adapter.CacheEntries(context.Background(), nil)
	if err != nil {
		t.Fatalf("CacheEntries(nil adapter) error = %v", err)
	}
	if len(entries) != 0 {
		t.Fatalf("CacheEntries(nil adapter) = %v, want none", entries)
	}
	warmed, err := adapter.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: []int32{1, 2, 3}})
	if err != nil {
		t.Fatalf("WarmCache(nil adapter) error = %v", err)
	}
	if len(warmed.Blocks) != 1 || warmed.Blocks[0].TokenCount != 3 {
		t.Fatalf("WarmCache(nil adapter) = %+v, want one token block", warmed)
	}
	stats, err = adapter.ClearCache(context.Background(), nil)
	if err != nil {
		t.Fatalf("ClearCache(nil adapter) error = %v", err)
	}
	if stats.Labels["cleared"] != "1" {
		t.Fatalf("ClearCache stats = %+v, want cleared count", stats)
	}

	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := adapter.CacheStats(cancelled); err != context.Canceled {
		t.Fatalf("CacheStats(cancelled) = %v, want context.Canceled", err)
	}
}

func TestRegisterMetalParser_NilAdapter_Good(t *testing.T) {
	var adapter *metaladapter
	reasoning, err := adapter.ParseReasoning(nil, "<think>scratch</think>answer")
	if err != nil {
		t.Fatalf("ParseReasoning(nil adapter) error = %v", err)
	}
	if reasoning.VisibleText == "" {
		t.Fatalf("ParseReasoning(nil adapter) = %+v, want parsed visible text", reasoning)
	}
	tools, err := adapter.ParseTools(nil, "")
	if err != nil {
		t.Fatalf("ParseTools(nil adapter) error = %v", err)
	}
	if len(tools.Calls) != 0 {
		t.Fatalf("ParseTools(nil adapter) = %+v, want no calls", tools)
	}
}
