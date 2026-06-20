// SPDX-Licence-Identifier: EUPL-1.2

// Coverage for the pure-Go planning + discovery surface (no model load, no
// forward): PlanLocalTuning's workload sweep across every TuningWorkload arm,
// DiscoverLocalRuntime's guard + no-model paths, and the nativeTextModel
// empty-prompt fast paths + LoadNativeTextModel error prologue.

package mlx

import (
	"context"
	"testing"

	"dappco.re/go/inference"
	"dappco.re/go/mlx/pkg/metal"
)

func TestPlanLocalTuning_AllWorkloadArms(t *testing.T) {
	device := metal.DeviceInfo{Architecture: "apple-m3", MemorySize: 64 << 30, MaxRecommendedWorkingSetSize: 48 << 30}
	req := inference.TuningPlanRequest{
		Device: inference.MachineDeviceInfo{
			Architecture: device.Architecture,
			MemorySize:   device.MemorySize,
		},
		Model: inference.ModelIdentity{Architecture: "gemma4_text", NumLayers: 34, HiddenSize: 2560, VocabSize: 256000},
		Workloads: []inference.TuningWorkload{
			inference.TuningWorkloadLowLatency,
			inference.TuningWorkloadThroughput,
			inference.TuningWorkloadLongContext,
			inference.TuningWorkloadAgentState,
			inference.TuningWorkloadChat, // default arm (no special reason)
		},
	}
	plan, err := PlanLocalTuning(context.Background(), req)
	if err != nil {
		t.Fatalf("PlanLocalTuning: %v", err)
	}
	if len(plan.Candidates) != 5 {
		t.Fatalf("expected one candidate per workload, got %d", len(plan.Candidates))
	}
	// AgentState candidate carries the state_restore label.
	var sawAgent bool
	for _, c := range plan.Candidates {
		if c.Workload == inference.TuningWorkloadAgentState {
			sawAgent = true
			if c.Labels["state_restore"] != "candidate" {
				t.Fatalf("agent-state candidate missing state_restore label: %+v", c.Labels)
			}
		}
	}
	if !sawAgent {
		t.Fatalf("agent-state workload candidate not produced")
	}
}

func TestPlanLocalTuning_MaxCandidatesClampAndCancel(t *testing.T) {
	req := inference.TuningPlanRequest{
		Device:    inference.MachineDeviceInfo{Architecture: "apple-m3", MemorySize: 32 << 30},
		Workloads: []inference.TuningWorkload{inference.TuningWorkloadLowLatency, inference.TuningWorkloadThroughput, inference.TuningWorkloadLongContext},
		Budget:    inference.TuningBudget{MaxCandidates: 1},
	}
	plan, err := PlanLocalTuning(context.Background(), req)
	if err != nil {
		t.Fatalf("PlanLocalTuning: %v", err)
	}
	// MaxCandidates clamps the sweep to a single candidate.
	if len(plan.Candidates) != 1 {
		t.Fatalf("MaxCandidates=1 should produce 1 candidate, got %d", len(plan.Candidates))
	}

	// Cancelled context errors before planning.
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := PlanLocalTuning(cancelled, req); err == nil {
		t.Fatalf("cancelled context should error")
	}
}

func TestDiscoverLocalRuntime_GuardsAndNoModels(t *testing.T) {
	// Cancelled context.
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := DiscoverLocalRuntime(cancelled, LocalDiscoveryConfig{}); err == nil {
		t.Fatalf("cancelled context should error")
	}

	// Default config (no models requested) → returns the runtime report,
	// exercising the zero-device → safeRuntimeDeviceInfo fallback and the
	// no-models early return.
	report, err := DiscoverLocalRuntime(context.Background(), LocalDiscoveryConfig{})
	if err != nil {
		t.Fatalf("DiscoverLocalRuntime: %v", err)
	}
	if report.Models != nil {
		t.Fatalf("default discovery should list no models, got %v", report.Models)
	}

	// IncludeModels with an empty directory → the discovery loop runs but
	// finds nothing.
	empty := t.TempDir()
	report2, err := DiscoverLocalRuntime(context.Background(), LocalDiscoveryConfig{
		IncludeModels: true,
		ModelDirs:     []string{empty},
	})
	if err != nil {
		t.Fatalf("DiscoverLocalRuntime(empty dir): %v", err)
	}
	if len(report2.Models) != 0 {
		t.Fatalf("empty dir should discover no models, got %v", report2.Models)
	}
}

func TestNativeTextModel_EmptyPromptFastPaths(t *testing.T) {
	// A zero-value nativeTextModel never touches its tm/tok when given no
	// prompts — the empty-prompt result paths are reachable without a model.
	m := &nativeTextModel{modelType: "gemma4", info: inference.ModelInfo{Architecture: "gemma4"}}

	results, err := m.Classify(context.Background(), nil)
	if err != nil {
		t.Fatalf("Classify(empty): %v", err)
	}
	if len(results) != 0 {
		t.Fatalf("Classify(empty) = %v, want empty", results)
	}

	batch, err := m.BatchGenerate(context.Background(), nil)
	if err != nil {
		t.Fatalf("BatchGenerate(empty): %v", err)
	}
	if len(batch) != 0 {
		t.Fatalf("BatchGenerate(empty) = %v, want empty", batch)
	}

	// Trivial accessors on the zero value.
	if m.ModelType() != "gemma4" {
		t.Fatalf("ModelType = %q", m.ModelType())
	}
	if m.Info().Architecture != "gemma4" {
		t.Fatalf("Info architecture = %q", m.Info().Architecture)
	}
	if m.Err() != nil {
		t.Fatalf("Err on fresh model = %v", m.Err())
	}
	if err := m.Close(); err != nil {
		t.Fatalf("Close = %v", err)
	}
	_ = m.Metrics()
}

func TestLoadNativeTextModel_BadPathErrors(t *testing.T) {
	// A non-existent model directory fails inside LoadGemma4TokenModelDir,
	// covering the prologue (maxLen default) + the first error return.
	if _, err := LoadNativeTextModel("/nonexistent/model/dir"); err == nil {
		t.Fatalf("LoadNativeTextModel on a missing dir should error")
	}
}
