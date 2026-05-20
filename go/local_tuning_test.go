// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"testing"
	"time"

	"dappco.re/go/inference"
	"dappco.re/go/inference/bench"
	"dappco.re/go/mlx/memory"
)

func TestMetalBackend_ImplementsDiscoveryPlanner_Good(t *testing.T) {
	var _ inference.MachineDiscoverer = (*metalbackend)(nil)
	var _ inference.TuningPlanner = (*metalbackend)(nil)
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
	if first.CacheMode == "" {
		t.Fatalf("candidate CacheMode empty: %+v", first)
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

func TestPlanLocalTuning_Qwen36UsesFallbackBackend_Good(t *testing.T) {
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
	if plan.Runtime.Backend != "mlx_lm" {
		t.Fatalf("plan.Runtime.Backend = %q, want mlx_lm fallback for qwen3_6", plan.Runtime.Backend)
	}
	if len(plan.Warnings) == 0 {
		t.Fatalf("Warnings empty, want native-runtime fallback warning")
	}
	if len(plan.Candidates) != 1 || plan.Candidates[0].Runtime.Backend != "mlx_lm" {
		t.Fatalf("candidates = %+v, want mlx_lm runtime candidate", plan.Candidates)
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

func TestRunLocalTuning_StreamsCandidateResults_Good(t *testing.T) {
	oldLoad := loadTuningModel
	oldBench := runTuningBench
	defer func() {
		loadTuningModel = oldLoad
		runTuningBench = oldBench
	}()

	loads := 0
	loadTuningModel = func(_ string, _ ...LoadOption) (*Model, error) {
		loads++
		return &Model{cleanup: func() error { return nil }}, nil
	}
	runTuningBench = func(_ context.Context, _ *Model, cfg bench.Config) (*bench.Report, error) {
		return &bench.Report{
			Model:     cfg.Model,
			ModelPath: cfg.ModelPath,
			Config:    cfg,
			Generation: bench.GenerationSummary{
				PromptTokens:        8,
				GeneratedTokens:     16,
				FirstTokenDuration:  40 * time.Millisecond,
				PrefillTokensPerSec: 800,
				DecodeTokensPerSec:  120,
				PeakMemoryBytes:     8 * memory.GiB,
				TotalDuration:       150 * time.Millisecond,
			},
			PromptCache: bench.PromptCacheReport{Attempted: true, HitRate: 0.8},
			KVRestore:   bench.LatencyReport{Attempted: true, Duration: 3 * time.Millisecond},
			Quality:     bench.QualityReport{Checks: []bench.QualityCheck{{Name: "non_empty_output", Pass: true, Score: 1}}},
		}, nil
	}

	var events []inference.TuningEvent
	results, err := RunLocalTuning(context.Background(), LocalTuningRunConfig{
		ModelPath: "/models/qwen3",
		Workload:  inference.TuningWorkloadAgentState,
		Candidates: []inference.TuningCandidate{
			{ID: "agent-state", ContextLength: 32768, CacheMode: "paged", PromptCache: true},
		},
		Bench: bench.Config{Prompt: "smoke", MaxTokens: 8, Runs: 1},
		Emit: func(event inference.TuningEvent) bool {
			events = append(events, event)
			return true
		},
	})
	if err != nil {
		t.Fatalf("RunLocalTuning() error = %v", err)
	}
	if loads != 1 || len(results) != 1 {
		t.Fatalf("loads/results = %d/%d, want 1/1", loads, len(results))
	}
	if len(events) != 2 || events[0].Kind != inference.TuningEventCandidate || events[1].Kind != inference.TuningEventResult {
		t.Fatalf("events = %+v, want candidate/result stream", events)
	}
	if results[0].Score.Score <= 0 || results[0].Measurements.DecodeTokensPerSec != 120 {
		t.Fatalf("result = %+v, want scored measurements", results[0])
	}
	if results[0].Measurements.LoadMilliseconds <= 0 || results[0].Measurements.FirstTokenMilliseconds != 40 || results[0].Measurements.CorrectnessSmokeResult != "passed" {
		t.Fatalf("measurements = %+v, want load, first-token, and smoke result", results[0].Measurements)
	}
}
