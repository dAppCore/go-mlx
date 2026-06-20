// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"strings"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	mlx "dappco.re/go/mlx"
)

// appendSplitFFNTuningCandidates builds one split-CPU-FFN candidate per
// (cache, workload) pairing from the estimator's memory reports, sorted by
// footprint. The estimator seam is mocked so no real model is loaded.
func TestAppendSplitFFNTuningCandidates_BuildsCandidates_Good(t *testing.T) {
	orig := runCPUFFNMemoryEstimate
	t.Cleanup(func() { runCPUFFNMemoryEstimate = orig })
	runCPUFFNMemoryEstimate = func(_ context.Context, _ string, cache int) (*mlx.CPUSplitFFNMemoryReport, error) {
		// Higher cache → lower peak resident, so the sort reorders them.
		return &mlx.CPUSplitFFNMemoryReport{
			TotalLayers:       4,
			LoadedLayers:      4,
			PeakResidentBytes: int64(1000 - cache*10),
			ResidentBytes:     int64(500 - cache),
		}, nil
	}

	plan := inference.TuningPlan{
		Model:     inference.ModelIdentity{Path: "/m"},
		Runtime:   inference.RuntimeIdentity{Backend: "metal"},
		Workloads: []inference.TuningWorkload{inference.TuningWorkloadChat},
	}
	out := appendSplitFFNTuningCandidates(context.Background(), plan, "/m", []int{0, 2})
	if len(out.Candidates) != 2 {
		t.Fatalf("candidates = %d, want 2 (one per cache for the single workload)", len(out.Candidates))
	}
	// The lowest-footprint estimate (cache 2) ranks first → rank label "1".
	if out.Candidates[0].Labels["cpu_ffn_cache_layers"] != "2" {
		t.Fatalf("first candidate cache = %q, want the lowest-footprint cache 2 first", out.Candidates[0].Labels["cpu_ffn_cache_layers"])
	}
	if !strings.Contains(out.Candidates[0].ID, "split_cpu_ffn") {
		t.Fatalf("candidate ID = %q, want a split_cpu_ffn id", out.Candidates[0].ID)
	}
}

// appendSplitFFNTuningCandidates records a warning (not a candidate) when the
// estimator errors, and when it returns a nil report, and defaults to the chat
// workload when the plan declares none.
func TestAppendSplitFFNTuningCandidates_EstimatorFailures_Bad(t *testing.T) {
	orig := runCPUFFNMemoryEstimate
	t.Cleanup(func() { runCPUFFNMemoryEstimate = orig })
	calls := 0
	runCPUFFNMemoryEstimate = func(_ context.Context, _ string, cache int) (*mlx.CPUSplitFFNMemoryReport, error) {
		calls++
		if cache == 1 {
			return nil, core.NewError("estimate boom")
		}
		if cache == 2 {
			return nil, nil // estimator returned no report
		}
		return &mlx.CPUSplitFFNMemoryReport{TotalLayers: 2, PeakResidentBytes: 100}, nil
	}
	// No workloads on the plan → defaults to chat.
	plan := inference.TuningPlan{Model: inference.ModelIdentity{Path: "/m"}}
	out := appendSplitFFNTuningCandidates(context.Background(), plan, "/m", []int{1, 2, 3})
	if calls != 3 {
		t.Fatalf("estimator called %d times, want 3", calls)
	}
	if len(out.Warnings) != 2 {
		t.Fatalf("warnings = %d (%v), want 2 (error + nil-report)", len(out.Warnings), out.Warnings)
	}
	if len(out.Candidates) != 1 {
		t.Fatalf("candidates = %d, want 1 (only cache 3 succeeded)", len(out.Candidates))
	}
	if out.Candidates[0].Workload != inference.TuningWorkloadChat {
		t.Fatalf("workload = %q, want chat default", out.Candidates[0].Workload)
	}
}
