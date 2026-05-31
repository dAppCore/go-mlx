// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"flag"
	"io"
	"time"

	core "dappco.re/go"
	mlx "dappco.re/go/mlx"
	"dappco.re/go/mlx/memory"
)

type productionTurboQuantCompareReport struct {
	Version             int                                       `json:"version"`
	Command             string                                    `json:"command,omitempty"`
	BaselineReportPath  string                                    `json:"baseline_report_path,omitempty"`
	CandidateReportPath string                                    `json:"candidate_report_path,omitempty"`
	ComparedReportPaths []string                                  `json:"compared_report_paths,omitempty"`
	Policy              mlx.ProductionTurboQuantPolicy            `json:"policy"`
	SameModelPath       bool                                      `json:"same_model_path"`
	SamePromptShape     bool                                      `json:"same_prompt_shape"`
	SameLoadPolicy      bool                                      `json:"same_load_policy"`
	BaselineSummary     productionTurboQuantCompareSummary        `json:"baseline_summary"`
	CandidateSummary    productionTurboQuantCompareSummary        `json:"candidate_summary"`
	ComparedSummaries   []productionTurboQuantCompareSummary      `json:"compared_summaries,omitempty"`
	Evidence            mlx.ProductionTurboQuantPromotionEvidence `json:"evidence"`
	Decision            mlx.ProductionTurboQuantPromotionDecision `json:"decision"`
}

type productionTurboQuantCompareSummary struct {
	Path                       string        `json:"path,omitempty"`
	ModelPath                  string        `json:"model_path,omitempty"`
	CacheMode                  string        `json:"cache_mode,omitempty"`
	ContextLength              int           `json:"context_length,omitempty"`
	PromptCache                bool          `json:"prompt_cache,omitempty"`
	PromptCacheMinTokens       int           `json:"prompt_cache_min_tokens,omitempty"`
	CachePolicy                string        `json:"cache_policy,omitempty"`
	BatchSize                  int           `json:"batch_size,omitempty"`
	PrefillChunkSize           int           `json:"prefill_chunk_size,omitempty"`
	PromptBytes                int           `json:"prompt_bytes,omitempty"`
	PromptSuffixBytes          int           `json:"prompt_suffix_bytes,omitempty"`
	PromptChunkBytes           int           `json:"prompt_chunk_bytes,omitempty"`
	PromptRepeat               int           `json:"prompt_repeat,omitempty"`
	MaxTokens                  int           `json:"max_tokens,omitempty"`
	RequestedRuns              int           `json:"requested_runs,omitempty"`
	Chat                       bool          `json:"chat,omitempty"`
	SuccessfulRuns             int           `json:"successful_runs,omitempty"`
	VisibleTokens              int           `json:"visible_tokens,omitempty"`
	GeneratedTokens            int           `json:"generated_tokens,omitempty"`
	TotalDuration              time.Duration `json:"total_duration,omitempty"`
	RestoreAvgDuration         time.Duration `json:"restore_duration_average,omitempty"`
	DecodeTokensPerSecAverage  float64       `json:"decode_tokens_per_sec_average,omitempty"`
	PeakMemoryBytes            uint64        `json:"peak_memory_bytes,omitempty"`
	ActiveMemoryBytes          uint64        `json:"active_memory_bytes,omitempty"`
	CacheMemoryBytes           uint64        `json:"cache_memory_bytes,omitempty"`
	ActivePlusCacheMemoryBytes uint64        `json:"active_plus_cache_memory_bytes,omitempty"`
	EnergyJoules               float64       `json:"energy_joules,omitempty"`
	PowerWatts                 float64       `json:"power_watts,omitempty"`
}

type productionTurboQuantCompareDriverReport struct {
	Path   string
	Mode   memory.KVCacheMode
	Report driverProfileReport
}

func runProductionTurboQuantCompareCommand(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("production-turboquant-compare"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOut := fs.Bool("json", false, "print JSON TurboQuant promotion comparison report")
	turns := fs.Int("turns", mlx.ProductionMTPPromotionMinRetainedTurns, "retained workflow turns represented by the compared reports")
	qualityMatch := fs.Bool("quality-match", false, "mark candidate output quality as matching the baseline")
	qualityFlags := fs.String("quality-flags", "", "comma-separated quality flags from manual output review; any flag blocks promotion")
	normalContext := fs.Bool("normal-context", false, "mark the normal 30k-40k retained-context validation as present")
	stressContext := fs.Bool("stress-context", false, "mark the 100k stress-context validation as present")
	powerWatts := fs.Float64("power-watts", 0, "fallback estimated average active watts when reports do not already include energy")
	fs.Usage = func() {
		name := cliName()
		core.WriteString(stderr, core.Sprintf("Usage: %s production-turboquant-compare [flags] BASELINE.json TURBOQUANT.json [COMPARE.json...]\n", name))
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Compare driver-profile JSON reports for an explicit TurboQuant KV-cache\n")
		core.WriteString(stderr, "candidate against production anchors. The command applies the promotion\n")
		core.WriteString(stderr, "policy and returns an auditable report; rejection is not a command failure.\n")
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Flags:\n")
		fs.VisitAll(func(f *flag.Flag) {
			if f.DefValue == "" {
				core.WriteString(stderr, core.Sprintf("  -%s\n\t%s\n", f.Name, f.Usage))
				return
			}
			core.WriteString(stderr, core.Sprintf("  -%s\n\t%s (default %q)\n", f.Name, f.Usage, f.DefValue))
		})
	}
	if err := fs.Parse(args); err != nil {
		if core.Is(err, flag.ErrHelp) {
			return 0
		}
		return 2
	}
	if fs.NArg() < 2 {
		core.WriteString(stderr, core.Sprintf("%s production-turboquant-compare: expected baseline, TurboQuant candidate, and optional comparison driver-profile JSON paths\n", cliName()))
		fs.Usage()
		return 2
	}
	if *turns < 0 {
		core.WriteString(stderr, core.Sprintf("%s production-turboquant-compare: turns must be >= 0\n", cliName()))
		return 2
	}
	if *powerWatts < 0 {
		core.WriteString(stderr, core.Sprintf("%s production-turboquant-compare: power-watts must be >= 0\n", cliName()))
		return 2
	}

	entries, err := readProductionTurboQuantCompareDriverReports(fs.Args())
	if err != nil {
		core.Print(stderr, "%s production-turboquant-compare: %v", cliName(), err)
		return 1
	}
	report := newProductionTurboQuantCompareReport(entries, *turns, *qualityMatch, *qualityFlags, *normalContext, *stressContext, *powerWatts)
	if *jsonOut {
		data := core.JSONMarshalIndent(report, "", "  ")
		if !data.OK {
			core.Print(stderr, "%s production-turboquant-compare: marshal report failed", cliName())
			return 1
		}
		core.WriteString(stdout, string(data.Value.([]byte)))
		core.WriteString(stdout, "\n")
		return 0
	}
	printProductionTurboQuantCompareReport(stdout, report)
	return 0
}

func readProductionTurboQuantCompareDriverReports(paths []string) ([]productionTurboQuantCompareDriverReport, error) {
	reports := make([]productionTurboQuantCompareDriverReport, 0, len(paths))
	for _, path := range paths {
		report, err := readProductionDriverProfileReport(path)
		if err != nil {
			return nil, core.Errorf("read report %s: %v", path, err)
		}
		reports = append(reports, productionTurboQuantCompareDriverReport{
			Path:   path,
			Mode:   productionTurboQuantCompareCacheMode(report),
			Report: report,
		})
	}
	return reports, nil
}

func newProductionTurboQuantCompareReport(entries []productionTurboQuantCompareDriverReport, turns int, qualityMatch bool, qualityFlags string, normalContext bool, stressContext bool, powerWatts float64) productionTurboQuantCompareReport {
	policy := mlx.DefaultProductionTurboQuantPolicy()
	baselineIndex, candidateIndex := productionTurboQuantCompareIndexes(entries, policy.CacheMode)
	baseline := entries[baselineIndex]
	candidate := entries[candidateIndex]
	paths := productionTurboQuantComparePaths(entries)
	sameModel := productionTurboQuantCompareSameModelPath(entries)
	sameShape := productionTurboQuantCompareSamePromptShape(entries)
	sameLoad := productionTurboQuantCompareSameLoadPolicy(entries)
	flags := productionTurboQuantCompareQualityFlags(qualityFlags, sameModel, sameShape, sameLoad, baseline, candidate, powerWatts)
	comparedModes := productionTurboQuantCompareModes(entries)
	evidence := mlx.ProductionTurboQuantPromotionEvidence{
		RetainedWorkflow:             sameModel && sameShape && sameLoad,
		Turns:                        turns,
		QualityMatches:               qualityMatch,
		QualityFlags:                 flags,
		BaselineCacheMode:            baseline.Mode,
		CandidateCacheMode:           candidate.Mode,
		ComparedCacheModes:           comparedModes,
		NormalContextValidated:       normalContext,
		StressContextValidated:       stressContext,
		BaselineVisibleTokensPerSec:  baseline.Report.Summary.DecodeTokensPerSecAverage,
		CandidateVisibleTokensPerSec: candidate.Report.Summary.DecodeTokensPerSecAverage,
		BaselineWallDuration:         baseline.Report.Summary.TotalDuration,
		CandidateWallDuration:        candidate.Report.Summary.TotalDuration,
		BaselineRestoreDuration:      baseline.Report.Summary.RestoreAvgDuration,
		CandidateRestoreDuration:     candidate.Report.Summary.RestoreAvgDuration,
		BaselinePeakMemoryBytes:      baseline.Report.Summary.PeakMemoryBytes,
		CandidatePeakMemoryBytes:     candidate.Report.Summary.PeakMemoryBytes,
		BaselineEnergyJoules:         productionTurboQuantCompareEnergyJoules(baseline.Report, powerWatts),
		CandidateEnergyJoules:        productionTurboQuantCompareEnergyJoules(candidate.Report, powerWatts),
		EstimatedPowerWatts:          productionTurboQuantComparePowerWatts(baseline.Report, candidate.Report, powerWatts),
	}
	return productionTurboQuantCompareReport{
		Version:             1,
		Command:             "production-turboquant-compare",
		BaselineReportPath:  baseline.Path,
		CandidateReportPath: candidate.Path,
		ComparedReportPaths: paths,
		Policy:              policy,
		SameModelPath:       sameModel,
		SamePromptShape:     sameShape,
		SameLoadPolicy:      sameLoad,
		BaselineSummary:     productionTurboQuantCompareSummaryFromDriver(baseline),
		CandidateSummary:    productionTurboQuantCompareSummaryFromDriver(candidate),
		ComparedSummaries:   productionTurboQuantCompareSummaries(entries),
		Evidence:            evidence,
		Decision:            mlx.EvaluateProductionTurboQuantPromotion(policy, evidence),
	}
}

func productionTurboQuantCompareIndexes(entries []productionTurboQuantCompareDriverReport, candidateMode memory.KVCacheMode) (int, int) {
	baselineIndex := 0
	candidateIndex := 1
	if len(entries) < 2 {
		return baselineIndex, baselineIndex
	}
	for i, entry := range entries {
		if entry.Mode == candidateMode {
			candidateIndex = i
			break
		}
	}
	if baselineIndex == candidateIndex {
		for i, entry := range entries {
			if entry.Mode != candidateMode {
				baselineIndex = i
				break
			}
		}
	}
	return baselineIndex, candidateIndex
}

func productionTurboQuantCompareCacheMode(report driverProfileReport) memory.KVCacheMode {
	if report.Load == nil {
		return memory.KVCacheModeDefault
	}
	mode := memory.KVCacheMode(core.Trim(report.Load.CacheMode))
	if mode == "" {
		return memory.KVCacheModeDefault
	}
	return mode
}

func productionTurboQuantCompareSameModelPath(entries []productionTurboQuantCompareDriverReport) bool {
	if len(entries) == 0 || entries[0].Report.ModelPath == "" {
		return false
	}
	modelPath := entries[0].Report.ModelPath
	for _, entry := range entries[1:] {
		if entry.Report.ModelPath != modelPath {
			return false
		}
	}
	return true
}

func productionTurboQuantCompareSamePromptShape(entries []productionTurboQuantCompareDriverReport) bool {
	if len(entries) == 0 {
		return false
	}
	first := entries[0].Report
	for _, entry := range entries[1:] {
		report := entry.Report
		if first.PromptBytes != report.PromptBytes ||
			first.PromptSuffixBytes != report.PromptSuffixBytes ||
			first.PromptChunkBytes != report.PromptChunkBytes ||
			first.PromptRepeat != report.PromptRepeat ||
			first.MaxTokens != report.MaxTokens ||
			first.RequestedRuns != report.RequestedRuns ||
			first.Chat != report.Chat {
			return false
		}
	}
	return true
}

func productionTurboQuantCompareSameLoadPolicy(entries []productionTurboQuantCompareDriverReport) bool {
	if len(entries) == 0 || entries[0].Report.Load == nil {
		return false
	}
	first := entries[0].Report.Load
	for _, entry := range entries[1:] {
		load := entry.Report.Load
		if load == nil ||
			first.ContextLength != load.ContextLength ||
			first.PromptCache != load.PromptCache ||
			first.PromptCacheMinTokens != load.PromptCacheMinTokens ||
			first.CachePolicy != load.CachePolicy ||
			first.BatchSize != load.BatchSize ||
			first.PrefillChunkSize != load.PrefillChunkSize {
			return false
		}
	}
	return true
}

func productionTurboQuantCompareQualityFlags(raw string, sameModel, sameShape, sameLoad bool, baseline, candidate productionTurboQuantCompareDriverReport, powerWatts float64) []string {
	flags := make([]string, 0, 4)
	if trimmed := core.Trim(raw); trimmed != "" {
		for _, part := range core.Split(trimmed, ",") {
			flag := core.Trim(part)
			if flag != "" {
				flags = append(flags, flag)
			}
		}
	}
	if !sameModel {
		flags = append(flags, "model_path_mismatch")
	}
	if !sameShape {
		flags = append(flags, "prompt_shape_mismatch")
	}
	if !sameLoad {
		flags = append(flags, "load_policy_mismatch")
	}
	if baseline.Mode == memory.KVCacheModeDefault {
		flags = append(flags, "baseline_cache_mode_missing")
	}
	if candidate.Mode == memory.KVCacheModeDefault {
		flags = append(flags, "candidate_cache_mode_missing")
	}
	flags = productionTurboQuantCompareMetricFlags(flags, "baseline", baseline.Report, powerWatts)
	flags = productionTurboQuantCompareMetricFlags(flags, "candidate", candidate.Report, powerWatts)
	if productionTurboQuantComparePowerWatts(baseline.Report, candidate.Report, powerWatts) <= 0 {
		flags = append(flags, "estimated_power_watts_missing")
	}
	return flags
}

func productionTurboQuantCompareMetricFlags(flags []string, prefix string, report driverProfileReport, powerWatts float64) []string {
	if report.Summary.DecodeTokensPerSecAverage <= 0 {
		flags = append(flags, prefix+"_visible_throughput_missing")
	}
	if report.Summary.TotalDuration <= 0 {
		flags = append(flags, prefix+"_wall_duration_missing")
	}
	if report.Summary.RestoreAvgDuration <= 0 {
		flags = append(flags, prefix+"_restore_duration_missing")
	}
	if report.Summary.PeakMemoryBytes == 0 {
		flags = append(flags, prefix+"_peak_memory_missing")
	}
	if productionTurboQuantCompareEnergyJoules(report, powerWatts) <= 0 {
		flags = append(flags, prefix+"_energy_missing")
	}
	return flags
}

func productionTurboQuantCompareModes(entries []productionTurboQuantCompareDriverReport) []memory.KVCacheMode {
	modes := make([]memory.KVCacheMode, 0, len(entries))
	seen := make(map[memory.KVCacheMode]bool, len(entries))
	for _, entry := range entries {
		if entry.Mode == memory.KVCacheModeDefault || seen[entry.Mode] {
			continue
		}
		seen[entry.Mode] = true
		modes = append(modes, entry.Mode)
	}
	return modes
}

func productionTurboQuantComparePaths(entries []productionTurboQuantCompareDriverReport) []string {
	paths := make([]string, 0, len(entries))
	for _, entry := range entries {
		paths = append(paths, entry.Path)
	}
	return paths
}

func productionTurboQuantCompareSummaries(entries []productionTurboQuantCompareDriverReport) []productionTurboQuantCompareSummary {
	summaries := make([]productionTurboQuantCompareSummary, 0, len(entries))
	for _, entry := range entries {
		summaries = append(summaries, productionTurboQuantCompareSummaryFromDriver(entry))
	}
	return summaries
}

func productionTurboQuantCompareSummaryFromDriver(entry productionTurboQuantCompareDriverReport) productionTurboQuantCompareSummary {
	report := entry.Report
	summary := productionTurboQuantCompareSummary{
		Path:                       entry.Path,
		ModelPath:                  report.ModelPath,
		CacheMode:                  string(entry.Mode),
		PromptBytes:                report.PromptBytes,
		PromptSuffixBytes:          report.PromptSuffixBytes,
		PromptChunkBytes:           report.PromptChunkBytes,
		PromptRepeat:               report.PromptRepeat,
		MaxTokens:                  report.MaxTokens,
		RequestedRuns:              report.RequestedRuns,
		Chat:                       report.Chat,
		SuccessfulRuns:             report.Summary.SuccessfulRuns,
		VisibleTokens:              report.Summary.VisibleTokens,
		GeneratedTokens:            report.Summary.GeneratedTokens,
		TotalDuration:              report.Summary.TotalDuration,
		RestoreAvgDuration:         report.Summary.RestoreAvgDuration,
		DecodeTokensPerSecAverage:  report.Summary.DecodeTokensPerSecAverage,
		PeakMemoryBytes:            report.Summary.PeakMemoryBytes,
		ActiveMemoryBytes:          report.Summary.ActiveMemoryBytes,
		CacheMemoryBytes:           report.Summary.CacheMemoryBytes,
		ActivePlusCacheMemoryBytes: report.Summary.ActivePlusCacheMemoryBytes,
		EnergyJoules:               productionTurboQuantCompareEnergyJoules(report, 0),
		PowerWatts:                 productionTurboQuantComparePowerWatts(report, driverProfileReport{}, 0),
	}
	if report.Load != nil {
		summary.ContextLength = report.Load.ContextLength
		summary.PromptCache = report.Load.PromptCache
		summary.PromptCacheMinTokens = report.Load.PromptCacheMinTokens
		summary.CachePolicy = report.Load.CachePolicy
		summary.BatchSize = report.Load.BatchSize
		summary.PrefillChunkSize = report.Load.PrefillChunkSize
	}
	return summary
}

func productionTurboQuantCompareEnergyJoules(report driverProfileReport, fallbackPowerWatts float64) float64 {
	if report.EstimatedEnergy != nil && report.EstimatedEnergy.TotalJoules > 0 {
		return report.EstimatedEnergy.TotalJoules
	}
	if fallbackPowerWatts > 0 && report.Summary.TotalDuration > 0 {
		return durationJoules(report.Summary.TotalDuration, fallbackPowerWatts)
	}
	return 0
}

func productionTurboQuantComparePowerWatts(first, second driverProfileReport, fallbackPowerWatts float64) float64 {
	if first.EstimatedEnergy != nil && first.EstimatedEnergy.PowerWatts > 0 {
		return first.EstimatedEnergy.PowerWatts
	}
	if second.EstimatedEnergy != nil && second.EstimatedEnergy.PowerWatts > 0 {
		return second.EstimatedEnergy.PowerWatts
	}
	return fallbackPowerWatts
}

func printProductionTurboQuantCompareReport(stdout io.Writer, report productionTurboQuantCompareReport) {
	core.WriteString(stdout, core.Sprintf("production TurboQuant comparison: candidate=%t (%s)\n", report.Decision.ProductionCandidate, report.Decision.Reason))
	core.WriteString(stdout, core.Sprintf("baseline %s: %.1f tok/s, wall %s, restore %s, peak memory %d bytes, energy %.1f J\n",
		report.Evidence.BaselineCacheMode,
		report.Evidence.BaselineVisibleTokensPerSec,
		report.Evidence.BaselineWallDuration,
		report.Evidence.BaselineRestoreDuration,
		report.Evidence.BaselinePeakMemoryBytes,
		report.Evidence.BaselineEnergyJoules,
	))
	core.WriteString(stdout, core.Sprintf("candidate %s: %.1f tok/s, wall %s, restore %s, peak memory %d bytes, energy %.1f J\n",
		report.Evidence.CandidateCacheMode,
		report.Evidence.CandidateVisibleTokensPerSec,
		report.Evidence.CandidateWallDuration,
		report.Evidence.CandidateRestoreDuration,
		report.Evidence.CandidatePeakMemoryBytes,
		report.Evidence.CandidateEnergyJoules,
	))
	core.WriteString(stdout, core.Sprintf("compared modes: %s\n", core.Join(", ", productionTurboQuantModeStrings(report.Evidence.ComparedCacheModes)...)))
	if len(report.Evidence.QualityFlags) > 0 {
		core.WriteString(stdout, core.Sprintf("quality flags: %s\n", core.Join(", ", report.Evidence.QualityFlags...)))
	}
}
