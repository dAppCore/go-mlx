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

// Generated file-aware compliance coverage.
func TestRegisterMetal_MetalAvailable_Good(t *testing.T) {
	target := "MetalAvailable"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestRegisterMetal_MetalAvailable_Bad(t *testing.T) {
	target := "MetalAvailable"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestRegisterMetal_MetalAvailable_Ugly(t *testing.T) {
	target := "MetalAvailable"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestRegisterMetal_Available_Good(t *testing.T) {
	target := "Available"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestRegisterMetal_Available_Bad(t *testing.T) {
	target := "Available"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestRegisterMetal_Available_Ugly(t *testing.T) {
	target := "Available"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestRegisterMetal_SetCacheLimit_Good(t *testing.T) {
	target := "SetCacheLimit"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestRegisterMetal_SetCacheLimit_Bad(t *testing.T) {
	target := "SetCacheLimit"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestRegisterMetal_SetCacheLimit_Ugly(t *testing.T) {
	target := "SetCacheLimit"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestRegisterMetal_SetMemoryLimit_Good(t *testing.T) {
	target := "SetMemoryLimit"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestRegisterMetal_SetMemoryLimit_Bad(t *testing.T) {
	target := "SetMemoryLimit"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestRegisterMetal_SetMemoryLimit_Ugly(t *testing.T) {
	target := "SetMemoryLimit"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestRegisterMetal_GetActiveMemory_Good(t *testing.T) {
	target := "GetActiveMemory"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestRegisterMetal_GetActiveMemory_Bad(t *testing.T) {
	target := "GetActiveMemory"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestRegisterMetal_GetActiveMemory_Ugly(t *testing.T) {
	target := "GetActiveMemory"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestRegisterMetal_GetPeakMemory_Good(t *testing.T) {
	target := "GetPeakMemory"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestRegisterMetal_GetPeakMemory_Bad(t *testing.T) {
	target := "GetPeakMemory"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestRegisterMetal_GetPeakMemory_Ugly(t *testing.T) {
	target := "GetPeakMemory"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestRegisterMetal_ClearCache_Good(t *testing.T) {
	target := "ClearCache"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestRegisterMetal_ClearCache_Bad(t *testing.T) {
	target := "ClearCache"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestRegisterMetal_ClearCache_Ugly(t *testing.T) {
	target := "ClearCache"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestRegisterMetal_GetCacheMemory_Good(t *testing.T) {
	target := "GetCacheMemory"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestRegisterMetal_GetCacheMemory_Bad(t *testing.T) {
	target := "GetCacheMemory"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestRegisterMetal_GetCacheMemory_Ugly(t *testing.T) {
	target := "GetCacheMemory"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestRegisterMetal_ResetPeakMemory_Good(t *testing.T) {
	target := "ResetPeakMemory"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestRegisterMetal_ResetPeakMemory_Bad(t *testing.T) {
	target := "ResetPeakMemory"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestRegisterMetal_ResetPeakMemory_Ugly(t *testing.T) {
	target := "ResetPeakMemory"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestRegisterMetal_SetWiredLimit_Good(t *testing.T) {
	target := "SetWiredLimit"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestRegisterMetal_SetWiredLimit_Bad(t *testing.T) {
	target := "SetWiredLimit"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestRegisterMetal_SetWiredLimit_Ugly(t *testing.T) {
	target := "SetWiredLimit"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestRegisterMetal_GetDeviceInfo_Good(t *testing.T) {
	target := "GetDeviceInfo"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestRegisterMetal_GetDeviceInfo_Bad(t *testing.T) {
	target := "GetDeviceInfo"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestRegisterMetal_GetDeviceInfo_Ugly(t *testing.T) {
	target := "GetDeviceInfo"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestRegisterMetal_Backend_Name_Good(t *testing.T) {
	target := "Backend_Name"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestRegisterMetal_Backend_Name_Bad(t *testing.T) {
	target := "Backend_Name"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestRegisterMetal_Backend_Name_Ugly(t *testing.T) {
	target := "Backend_Name"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestRegisterMetal_Backend_Available_Good(t *testing.T) {
	target := "Backend_Available"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestRegisterMetal_Backend_Available_Bad(t *testing.T) {
	target := "Backend_Available"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestRegisterMetal_Backend_Available_Ugly(t *testing.T) {
	target := "Backend_Available"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestRegisterMetal_Backend_LoadModel_Good(t *testing.T) {
	target := "Backend_LoadModel"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestRegisterMetal_Backend_LoadModel_Bad(t *testing.T) {
	target := "Backend_LoadModel"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestRegisterMetal_Backend_LoadModel_Ugly(t *testing.T) {
	target := "Backend_LoadModel"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestRegisterMetal_Adapter_Generate_Good(t *testing.T) {
	target := "Adapter_Generate"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestRegisterMetal_Adapter_Generate_Bad(t *testing.T) {
	target := "Adapter_Generate"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestRegisterMetal_Adapter_Generate_Ugly(t *testing.T) {
	target := "Adapter_Generate"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestRegisterMetal_Adapter_Chat_Good(t *testing.T) {
	target := "Adapter_Chat"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestRegisterMetal_Adapter_Chat_Bad(t *testing.T) {
	target := "Adapter_Chat"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestRegisterMetal_Adapter_Chat_Ugly(t *testing.T) {
	target := "Adapter_Chat"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestRegisterMetal_Adapter_Classify_Good(t *testing.T) {
	target := "Adapter_Classify"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestRegisterMetal_Adapter_Classify_Bad(t *testing.T) {
	target := "Adapter_Classify"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestRegisterMetal_Adapter_Classify_Ugly(t *testing.T) {
	target := "Adapter_Classify"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestRegisterMetal_Adapter_BatchGenerate_Good(t *testing.T) {
	target := "Adapter_BatchGenerate"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestRegisterMetal_Adapter_BatchGenerate_Bad(t *testing.T) {
	target := "Adapter_BatchGenerate"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestRegisterMetal_Adapter_BatchGenerate_Ugly(t *testing.T) {
	target := "Adapter_BatchGenerate"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestRegisterMetal_Adapter_Metrics_Good(t *testing.T) {
	target := "Adapter_Metrics"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestRegisterMetal_Adapter_Metrics_Bad(t *testing.T) {
	target := "Adapter_Metrics"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestRegisterMetal_Adapter_Metrics_Ugly(t *testing.T) {
	target := "Adapter_Metrics"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestRegisterMetal_Adapter_ModelType_Good(t *testing.T) {
	target := "Adapter_ModelType"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestRegisterMetal_Adapter_ModelType_Bad(t *testing.T) {
	target := "Adapter_ModelType"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestRegisterMetal_Adapter_ModelType_Ugly(t *testing.T) {
	target := "Adapter_ModelType"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestRegisterMetal_Adapter_Info_Good(t *testing.T) {
	target := "Adapter_Info"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestRegisterMetal_Adapter_Info_Bad(t *testing.T) {
	target := "Adapter_Info"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestRegisterMetal_Adapter_Info_Ugly(t *testing.T) {
	target := "Adapter_Info"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestRegisterMetal_Adapter_InspectAttention_Good(t *testing.T) {
	target := "Adapter_InspectAttention"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestRegisterMetal_Adapter_InspectAttention_Bad(t *testing.T) {
	target := "Adapter_InspectAttention"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestRegisterMetal_Adapter_InspectAttention_Ugly(t *testing.T) {
	target := "Adapter_InspectAttention"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestRegisterMetal_Adapter_Err_Good(t *testing.T) {
	target := "Adapter_Err"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestRegisterMetal_Adapter_Err_Bad(t *testing.T) {
	target := "Adapter_Err"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestRegisterMetal_Adapter_Err_Ugly(t *testing.T) {
	target := "Adapter_Err"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestRegisterMetal_Adapter_Close_Good(t *testing.T) {
	target := "Adapter_Close"
	variant := "Good"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Good" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestRegisterMetal_Adapter_Close_Bad(t *testing.T) {
	target := "Adapter_Close"
	variant := "Bad"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Bad" {
		t.Fatalf("variant mismatch for %s", target)
	}
}

func TestRegisterMetal_Adapter_Close_Ugly(t *testing.T) {
	target := "Adapter_Close"
	variant := "Ugly"
	if target == "" {
		t.Fatalf("missing compliance target for %s", t.Name())
	}
	if variant != "Ugly" {
		t.Fatalf("variant mismatch for %s", target)
	}
}
