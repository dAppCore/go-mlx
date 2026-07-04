// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"testing"

	"dappco.re/go/inference"
	"dappco.re/go/inference/memory"
)

func TestMetalBackend_ImplementsDiscoveryPlanner_Good(t *testing.T) {
	var _ inference.MachineDiscoverer = (*metalbackend)(nil)
	var _ inference.TuningPlanner = (*metalbackend)(nil)
}

func TestMetalBackend_DiscoverMachine_Good(t *testing.T) {
	// Metadata-only path: no ModelDirs, IncludeModels off → the device +
	// runtime report comes straight back without a filesystem walk or any
	// weight load. The backend method wraps DiscoverLocalRuntime.
	backend := &metalbackend{}
	report, err := backend.DiscoverMachine(context.Background(), inference.MachineDiscoveryRequest{
		Workloads: []inference.TuningWorkload{inference.TuningWorkloadCoding},
		Labels:    map[string]string{"profile_set": "dev"},
	})
	if err != nil {
		t.Fatalf("DiscoverMachine() error = %v", err)
	}
	if report == nil {
		t.Fatal("DiscoverMachine() report = nil")
	}
	if report.Labels["machine_hash"] == "" {
		t.Fatalf("report Labels = %+v, want machine_hash", report.Labels)
	}
	if report.Labels["profile_set"] != "dev" {
		t.Fatalf("report Labels = %+v, want caller label preserved", report.Labels)
	}
	if len(report.Models) != 0 {
		t.Fatalf("report Models = %+v, want none when IncludeModels is off", report.Models)
	}
}

func TestMetalBackend_DiscoverMachine_Bad(t *testing.T) {
	// A cancelled context aborts before any discovery work.
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	backend := &metalbackend{}
	_, err := backend.DiscoverMachine(ctx, inference.MachineDiscoveryRequest{})
	if err == nil {
		t.Fatal("DiscoverMachine(cancelled) error = nil, want context error")
	}
}

func TestMetalBackend_PlanTuning_Good(t *testing.T) {
	backend := &metalbackend{}
	plan, err := backend.PlanTuning(context.Background(), inference.TuningPlanRequest{
		Runtime: inference.RuntimeIdentity{Backend: "metal", Device: "apple9"},
		Device: inference.MachineDeviceInfo{
			Architecture:                 "apple9",
			MemorySize:                   96 * memory.GiB,
			MaxRecommendedWorkingSetSize: 90 * memory.GiB,
		},
		Model: inference.ModelIdentity{
			Path:          "/models/qwen3",
			Architecture:  "qwen3",
			QuantBits:     4,
			ContextLength: 32768,
			NumLayers:     36,
			HiddenSize:    4096,
		},
		Workloads: []inference.TuningWorkload{inference.TuningWorkloadCoding},
		Budget:    inference.TuningBudget{MaxCandidates: 2},
	})
	if err != nil {
		t.Fatalf("PlanTuning() error = %v", err)
	}
	if plan == nil {
		t.Fatal("PlanTuning() plan = nil")
	}
	if plan.Model.Path != "/models/qwen3" || len(plan.Candidates) == 0 {
		t.Fatalf("plan = model:%+v candidates:%d, want qwen3 path + candidates", plan.Model, len(plan.Candidates))
	}
}

func TestMetalBackend_PlanTuning_Bad(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	backend := &metalbackend{}
	_, err := backend.PlanTuning(ctx, inference.TuningPlanRequest{})
	if err == nil {
		t.Fatal("PlanTuning(cancelled) error = nil, want context error")
	}
}

func TestPlanLocalTuning_DerivesCandidatesFromMemoryPlan_Good(t *testing.T) {
	plan, err := PlanLocalTuning(context.Background(), inference.TuningPlanRequest{
		Runtime: inference.RuntimeIdentity{Backend: "metal", Device: "apple9"},
		Device: inference.MachineDeviceInfo{
			Architecture:                 "apple9",
			MemorySize:                   96 * memory.GiB,
			MaxRecommendedWorkingSetSize: 90 * memory.GiB,
		},
		Model: inference.ModelIdentity{
			Path:          "/models/qwen3",
			Architecture:  "qwen3",
			QuantBits:     4,
			ContextLength: 32768,
			NumLayers:     36,
			HiddenSize:    4096,
		},
		Workloads: []inference.TuningWorkload{inference.TuningWorkloadCoding, inference.TuningWorkloadAgentState},
		Budget:    inference.TuningBudget{MaxCandidates: 4},
	})
	if err != nil {
		t.Fatalf("PlanLocalTuning() error = %v", err)
	}
	if plan.Runtime.Backend != "metal" || plan.Model.Path != "/models/qwen3" {
		t.Fatalf("plan identities = runtime:%+v model:%+v", plan.Runtime, plan.Model)
	}
	if len(plan.Candidates) == 0 {
		t.Fatal("PlanLocalTuning() returned no candidates")
	}
	if plan.Recommended[inference.TuningWorkloadAgentState] == "" {
		t.Fatalf("recommended = %+v, want agent-state candidate", plan.Recommended)
	}
	first := plan.Candidates[0]
	if first.ContextLength <= 0 || first.BatchSize <= 0 || first.PrefillChunkSize <= 0 {
		t.Fatalf("candidate shape = %+v, want memory-planned settings", first)
	}
	if first.CacheMode != string(memory.KVCacheModeDefault) {
		t.Fatalf("candidate CacheMode = %q, want the 96GB plan's default (bounded) cache: %+v", first.CacheMode, first)
	}
}

func TestDiscoverLocalRuntime_PreservesProbedDeviceName_Good(t *testing.T) {
	report, err := DiscoverLocalRuntime(context.Background(), LocalDiscoveryConfig{
		Device: DeviceInfo{
			Name:                         "Apple M3 Ultra",
			Architecture:                 "apple9",
			MemorySize:                   96 * memory.GiB,
			MaxRecommendedWorkingSetSize: 90 * memory.GiB,
		},
		Workloads: []inference.TuningWorkload{inference.TuningWorkloadCoding},
	})
	if err != nil {
		t.Fatalf("DiscoverLocalRuntime() error = %v", err)
	}
	if report.Device.Name != "Apple M3 Ultra" || report.Device.Architecture != "apple9" {
		t.Fatalf("device = %+v, want probed name and architecture", report.Device)
	}
}

func TestDiscoverLocalRuntime_AddsStableMachineHash_Good(t *testing.T) {
	cfg := LocalDiscoveryConfig{
		Device: DeviceInfo{
			Name:                         "Apple M3 Ultra",
			Architecture:                 "apple9",
			MaxBufferLength:              1 << 30,
			MemorySize:                   96 * memory.GiB,
			MaxRecommendedWorkingSetSize: 90 * memory.GiB,
		},
		Workloads: []inference.TuningWorkload{inference.TuningWorkloadCoding},
		Labels:    map[string]string{"profile_set": "dev"},
	}

	first, err := DiscoverLocalRuntime(context.Background(), cfg)
	if err != nil {
		t.Fatalf("DiscoverLocalRuntime(first) error = %v", err)
	}
	second, err := DiscoverLocalRuntime(context.Background(), cfg)
	if err != nil {
		t.Fatalf("DiscoverLocalRuntime(second) error = %v", err)
	}

	hash := first.Labels["machine_hash"]
	if hash == "" {
		t.Fatalf("Labels = %+v, want machine_hash", first.Labels)
	}
	if second.Labels["machine_hash"] != hash {
		t.Fatalf("machine_hash changed: first %q second %q", hash, second.Labels["machine_hash"])
	}
	if first.Device.Labels["machine_hash"] != hash {
		t.Fatalf("device labels = %+v, want machine_hash %q", first.Device.Labels, hash)
	}
	if first.Labels["profile_set"] != "dev" {
		t.Fatalf("Labels = %+v, want caller label preserved", first.Labels)
	}
}

func TestTuningMachineHash_EmptyDevice_Bad(t *testing.T) {
	if got := tuningMachineHash(DeviceInfo{}); got != "" {
		t.Fatalf("tuningMachineHash(empty) = %q, want empty", got)
	}
}

func TestPlanLocalTuning_Qwen36StaysMetalWithNativeGapWarning_Good(t *testing.T) {
	plan, err := PlanLocalTuning(context.Background(), inference.TuningPlanRequest{
		Runtime: inference.RuntimeIdentity{Backend: "metal", Device: "apple9"},
		Device: inference.MachineDeviceInfo{
			Architecture:                 "apple9",
			MemorySize:                   96 * memory.GiB,
			MaxRecommendedWorkingSetSize: 90 * memory.GiB,
		},
		Model: inference.ModelIdentity{
			Path:          "/models/qwen3.6-27b",
			Architecture:  "qwen3_6",
			QuantBits:     4,
			ContextLength: 262144,
			NumLayers:     64,
			HiddenSize:    5120,
		},
		Workloads: []inference.TuningWorkload{inference.TuningWorkloadCoding},
	})
	if err != nil {
		t.Fatalf("PlanLocalTuning() error = %v", err)
	}
	if plan.Runtime.Backend != "metal" || !plan.Runtime.NativeRuntime {
		t.Fatalf("plan.Runtime = %+v, want metal runtime with native_runtime=true for staged qwen3_6", plan.Runtime)
	}
	if len(plan.Warnings) != 0 {
		t.Fatalf("Warnings = %q, want none for native staged qwen3_6", plan.Warnings)
	}
	if len(plan.Candidates) != 1 || plan.Candidates[0].Runtime.Backend != "metal" || !plan.Candidates[0].Runtime.NativeRuntime {
		t.Fatalf("candidates = %+v, want metal candidate with native_runtime=true", plan.Candidates)
	}
	if plan.Candidates[0].Runtime.Labels["fallback_backend"] != "" {
		t.Fatalf("candidate labels = %+v, must not set fallback_backend", plan.Candidates[0].Runtime.Labels)
	}
}

func TestTuningCandidateLoadOptions_AppliesCandidate_Good(t *testing.T) {
	candidate := inference.TuningCandidate{
		ContextLength:        32768,
		ParallelSlots:        2,
		PromptCache:          true,
		PromptCacheMinTokens: 1024,
		CachePolicy:          "full",
		CacheMode:            "paged",
		BatchSize:            4,
		PrefillChunkSize:     2048,
		ExpectedQuantization: 8,
		MemoryLimitBytes:     64 * memory.GiB,
		CacheLimitBytes:      4 * memory.GiB,
		WiredLimitBytes:      60 * memory.GiB,
	}

	cfg := applyLoadOptions(TuningCandidateLoadOptions(candidate))
	if cfg.ContextLength != candidate.ContextLength || cfg.ParallelSlots != candidate.ParallelSlots {
		t.Fatalf("context/slots = %d/%d, want %d/%d", cfg.ContextLength, cfg.ParallelSlots, candidate.ContextLength, candidate.ParallelSlots)
	}
	if string(cfg.CachePolicy) != candidate.CachePolicy || string(cfg.CacheMode) != candidate.CacheMode {
		t.Fatalf("cache = %q/%q, want %q/%q", cfg.CachePolicy, cfg.CacheMode, candidate.CachePolicy, candidate.CacheMode)
	}
	if cfg.BatchSize != candidate.BatchSize || cfg.PrefillChunkSize != candidate.PrefillChunkSize {
		t.Fatalf("batch/prefill = %d/%d", cfg.BatchSize, cfg.PrefillChunkSize)
	}
	if cfg.MemoryLimitBytes != candidate.MemoryLimitBytes || cfg.CacheLimitBytes != candidate.CacheLimitBytes || cfg.WiredLimitBytes != candidate.WiredLimitBytes {
		t.Fatalf("allocator limits = %+v", cfg)
	}
}
