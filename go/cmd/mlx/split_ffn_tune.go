// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"maps"

	core "dappco.re/go"
	"dappco.re/go/inference"
	mlx "dappco.re/go/mlx"
)

type cliSplitFFNEstimate struct {
	cache  int
	report mlx.CPUSplitFFNMemoryReport
}

func cliSplitFFNCacheLayers(value string) ([]int, error) {
	value = core.Trim(value)
	if value == "" {
		return nil, nil
	}
	parts := core.Split(value, ",")
	caches := make([]int, 0, len(parts))
	for _, part := range parts {
		part = core.Trim(part)
		if part == "" {
			continue
		}
		parsed := core.ParseInt(part, 10, 64)
		if !parsed.OK {
			return nil, core.Errorf("invalid split FFN cache layer count %q", part)
		}
		caches = append(caches, int(parsed.Value.(int64)))
	}
	return caches, nil
}

func appendSplitFFNTuningCandidates(ctx context.Context, plan inference.TuningPlan, sourcePath string, caches []int) inference.TuningPlan {
	estimates := make([]cliSplitFFNEstimate, 0, len(caches))
	for _, cache := range caches {
		report, err := runCPUFFNMemoryEstimate(ctx, sourcePath, cache)
		if err != nil {
			plan.Warnings = append(plan.Warnings, core.Sprintf("split CPU FFN cache %d: %v", cache, err))
			continue
		}
		if report == nil {
			plan.Warnings = append(plan.Warnings, core.Sprintf("split CPU FFN cache %d: estimator returned no report", cache))
			continue
		}
		estimates = append(estimates, cliSplitFFNEstimate{cache: cache, report: *report})
	}
	cliSortSplitFFNEstimates(estimates)
	workloads := plan.Workloads
	if len(workloads) == 0 {
		workloads = []inference.TuningWorkload{inference.TuningWorkloadChat}
	}
	for rank, estimate := range estimates {
		for _, workload := range workloads {
			base := cliBaseCandidateForWorkload(plan, workload)
			candidate := base
			candidate.ID = core.Sprintf("%s:split_cpu_ffn:cache%d", workload, estimate.cache)
			candidate.Workload = workload
			candidate.Model = plan.Model
			if candidate.Model.Path == "" {
				candidate.Model.Path = sourcePath
			}
			candidate.Runtime = plan.Runtime
			candidate.Labels = cliSplitFFNLabels(base.Labels, estimate, rank+1)
			candidate.Reasons = append(append([]string(nil), base.Reasons...), cliSplitFFNReason(estimate)...)
			plan.Candidates = append(plan.Candidates, candidate)
		}
	}
	return plan
}

func cliSortSplitFFNEstimates(estimates []cliSplitFFNEstimate) {
	for i := 1; i < len(estimates); i++ {
		for j := i; j > 0 && cliSplitFFNEstimateLess(estimates[j], estimates[j-1]); j-- {
			estimates[j], estimates[j-1] = estimates[j-1], estimates[j]
		}
	}
}

func cliSplitFFNEstimateLess(a, b cliSplitFFNEstimate) bool {
	if a.report.PeakResidentBytes != b.report.PeakResidentBytes {
		return a.report.PeakResidentBytes < b.report.PeakResidentBytes
	}
	if a.report.ResidentBytes != b.report.ResidentBytes {
		return a.report.ResidentBytes < b.report.ResidentBytes
	}
	if a.report.LayerLoads != b.report.LayerLoads {
		return a.report.LayerLoads < b.report.LayerLoads
	}
	return a.cache < b.cache
}

func cliBaseCandidateForWorkload(plan inference.TuningPlan, workload inference.TuningWorkload) inference.TuningCandidate {
	for _, candidate := range plan.Candidates {
		if candidate.Workload == workload {
			return candidate
		}
	}
	return inference.TuningCandidate{
		Workload: workload,
		Model:    plan.Model,
		Runtime:  plan.Runtime,
	}
}

func cliSplitFFNLabels(base map[string]string, estimate cliSplitFFNEstimate, rank int) map[string]string {
	labels := cliCloneStringLabels(base)
	labels["split"] = "cpu_ffn"
	labels["rank"] = core.Itoa(rank)
	labels["estimated"] = "true"
	labels["cpu_ffn_cache_layers"] = core.Itoa(estimate.cache)
	labels["cpu_ffn_total_layers"] = core.Itoa(estimate.report.TotalLayers)
	labels["cpu_ffn_loaded_layers"] = core.Itoa(estimate.report.LoadedLayers)
	labels["cpu_ffn_layer_loads"] = core.Itoa(estimate.report.LayerLoads)
	labels["cpu_ffn_evictions"] = core.Itoa(estimate.report.EvictedLayers)
	labels["cpu_ffn_resident_bytes"] = core.FormatInt(estimate.report.ResidentBytes, 10)
	labels["cpu_ffn_peak_resident_bytes"] = core.FormatInt(estimate.report.PeakResidentBytes, 10)
	labels["cpu_ffn_dense_equivalent_bytes"] = core.FormatInt(estimate.report.DenseEquivalentBytes, 10)
	labels["cpu_ffn_saved_bytes"] = core.FormatInt(estimate.report.SavedBytes, 10)
	labels["cpu_ffn_resident_ratio"] = core.Sprintf("%.6f", estimate.report.ResidentRatio)
	return labels
}

func cliSplitFFNReason(estimate cliSplitFFNEstimate) []string {
	reason := "split CPU FFN caches all layers after first load"
	if estimate.cache < 0 {
		reason = "split CPU FFN streams layer weights without retaining a resident cache"
	}
	if estimate.cache > 0 {
		reason = core.Sprintf("split CPU FFN keeps up to %d layers resident", estimate.cache)
	}
	return []string{
		reason,
		core.Sprintf("estimated CPU FFN peak resident %d bytes", estimate.report.PeakResidentBytes),
	}
}

func cliCloneStringLabels(labels map[string]string) map[string]string {
	out := map[string]string{}
	maps.Copy(out, labels)
	return out
}
