// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"flag"
	"io"
	"time"

	core "dappco.re/go"
	mlx "dappco.re/go/mlx"
)

type productionMTPCompareReport struct {
	Version              int                                `json:"version"`
	Command              string                             `json:"command,omitempty"`
	TargetOnlyReportPath string                             `json:"target_only_report_path"`
	MTPReportPath        string                             `json:"mtp_report_path"`
	Policy               mlx.ProductionMTPPolicy            `json:"policy"`
	SameModelPath        bool                               `json:"same_model_path"`
	SamePromptShape      bool                               `json:"same_prompt_shape"`
	TargetOnlySummary    productionMTPCompareSummary        `json:"target_only_summary"`
	MTPSummary           productionMTPCompareSummary        `json:"mtp_summary"`
	Evidence             mlx.ProductionMTPPromotionEvidence `json:"evidence"`
	Decision             mlx.ProductionMTPPromotionDecision `json:"decision"`
}

type productionMTPCompareSummary struct {
	ModelPath                     string        `json:"model_path,omitempty"`
	SpeculativeDraftModelPath     string        `json:"speculative_draft_model_path,omitempty"`
	SpeculativeDraftTokens        int           `json:"speculative_draft_tokens,omitempty"`
	PromptBytes                   int           `json:"prompt_bytes,omitempty"`
	PromptSuffixBytes             int           `json:"prompt_suffix_bytes,omitempty"`
	PromptChunkBytes              int           `json:"prompt_chunk_bytes,omitempty"`
	PromptRepeat                  int           `json:"prompt_repeat,omitempty"`
	MaxTokens                     int           `json:"max_tokens,omitempty"`
	RequestedRuns                 int           `json:"requested_runs,omitempty"`
	Chat                          bool          `json:"chat,omitempty"`
	SuccessfulRuns                int           `json:"successful_runs,omitempty"`
	VisibleTokens                 int           `json:"visible_tokens,omitempty"`
	GeneratedTokens               int           `json:"generated_tokens,omitempty"`
	TotalDuration                 time.Duration `json:"total_duration,omitempty"`
	DecodeTokensPerSecAverage     float64       `json:"decode_tokens_per_sec_average,omitempty"`
	PeakMemoryBytes               uint64        `json:"peak_memory_bytes,omitempty"`
	ActiveMemoryBytes             uint64        `json:"active_memory_bytes,omitempty"`
	CacheMemoryBytes              uint64        `json:"cache_memory_bytes,omitempty"`
	ActivePlusCacheMemoryBytes    uint64        `json:"active_plus_cache_memory_bytes,omitempty"`
	MTPVisibleTokensPerSecAverage float64       `json:"mtp_visible_tokens_per_sec_average,omitempty"`
	MTPTargetTokensPerSecAverage  float64       `json:"mtp_target_tokens_per_sec_average,omitempty"`
	MTPWarmDecodeTokensPerSec     float64       `json:"mtp_warm_decode_tokens_per_sec_average,omitempty"`
	MTPAcceptanceRateAverage      float64       `json:"mtp_acceptance_rate_average,omitempty"`
	MTPDraftTokenSchedule         []int         `json:"mtp_draft_token_schedule,omitempty"`
	MTPProposedTokens             int           `json:"mtp_proposed_tokens,omitempty"`
	MTPAcceptedTokens             int           `json:"mtp_accepted_tokens,omitempty"`
	MTPRejectedTokens             int           `json:"mtp_rejected_tokens,omitempty"`
	MTPTargetVerifyCalls          int           `json:"mtp_target_verify_calls,omitempty"`
	MTPDraftCalls                 int           `json:"mtp_draft_calls,omitempty"`
}

func runProductionMTPCompareCommand(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("production-mtp-compare"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOut := fs.Bool("json", false, "print JSON target-only versus MTP promotion report")
	turns := fs.Int("turns", mlx.ProductionMTPPromotionMinRetainedTurns, "retained workflow turns represented by the compared reports")
	greedyMatch := fs.Bool("greedy-match", false, "mark target-only and MTP greedy visible outputs as matching")
	qualityFlags := fs.String("quality-flags", "", "comma-separated quality flags from manual output review; any flag blocks promotion")
	fs.Usage = func() {
		name := cliName()
		core.WriteString(stderr, core.Sprintf("Usage: %s production-mtp-compare [flags] TARGET_ONLY.json MTP.json\n", name))
		core.WriteString(stderr, "\n")
		core.WriteString(stderr, "Compare two driver-profile JSON reports for the same retained workflow:\n")
		core.WriteString(stderr, "one target-only run and one official Gemma 4 assistant MTP run. The\n")
		core.WriteString(stderr, "result applies the production MTP promotion policy; rejection is an\n")
		core.WriteString(stderr, "auditable report, not a command failure.\n")
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
	if fs.NArg() != 2 {
		core.WriteString(stderr, core.Sprintf("%s production-mtp-compare: expected target-only and MTP driver-profile JSON paths\n", cliName()))
		fs.Usage()
		return 2
	}
	if *turns < 0 {
		core.WriteString(stderr, core.Sprintf("%s production-mtp-compare: turns must be >= 0\n", cliName()))
		return 2
	}

	targetPath, mtpPath := fs.Arg(0), fs.Arg(1)
	target, err := readProductionMTPCompareDriverReport(targetPath)
	if err != nil {
		core.Print(stderr, "%s production-mtp-compare: read target-only report: %v", cliName(), err)
		return 1
	}
	mtp, err := readProductionMTPCompareDriverReport(mtpPath)
	if err != nil {
		core.Print(stderr, "%s production-mtp-compare: read MTP report: %v", cliName(), err)
		return 1
	}

	report := newProductionMTPCompareReport(targetPath, target, mtpPath, mtp, *turns, *greedyMatch, *qualityFlags)
	if *jsonOut {
		data := core.JSONMarshalIndent(report, "", "  ")
		if !data.OK {
			core.Print(stderr, "%s production-mtp-compare: marshal report failed", cliName())
			return 1
		}
		core.WriteString(stdout, string(data.Value.([]byte)))
		core.WriteString(stdout, "\n")
		return 0
	}
	printProductionMTPCompareReport(stdout, report)
	return 0
}

func readProductionMTPCompareDriverReport(path string) (driverProfileReport, error) {
	read := core.ReadFile(path)
	if !read.OK {
		return driverProfileReport{}, core.Errorf("read %s: %v", path, read.Value)
	}
	var report driverProfileReport
	if result := core.JSONUnmarshal(read.Value.([]byte), &report); !result.OK {
		return driverProfileReport{}, core.Errorf("decode %s: %v", path, result.Value)
	}
	return report, nil
}

func newProductionMTPCompareReport(targetPath string, target driverProfileReport, mtpPath string, mtp driverProfileReport, turns int, greedyMatch bool, qualityFlags string) productionMTPCompareReport {
	sameModel := productionMTPCompareSameModelPath(target, mtp)
	sameShape := productionMTPCompareSamePromptShape(target, mtp)
	flags := productionMTPCompareQualityFlags(qualityFlags, sameModel, sameShape)
	evidence := mlx.ProductionMTPPromotionEvidence{
		RetainedWorkflow:              sameModel && sameShape,
		Turns:                         turns,
		GreedyOutputMatches:           greedyMatch,
		QualityFlags:                  flags,
		TargetOnlyVisibleTokensPerSec: target.Summary.DecodeTokensPerSecAverage,
		MTPVisibleTokensPerSec:        productionMTPCompareMTPVisibleTokensPerSec(mtp.Summary),
		TargetOnlyWallDuration:        target.Summary.TotalDuration,
		MTPWallDuration:               mtp.Summary.TotalDuration,
		MTPProposedTokens:             mtp.Summary.MTPProposedTokens,
		MTPAcceptedTokens:             mtp.Summary.MTPAcceptedTokens,
		MTPRejectedTokens:             mtp.Summary.MTPRejectedTokens,
		MTPTargetVerifyCalls:          mtp.Summary.MTPTargetVerifyCalls,
	}
	policy := mlx.DefaultProductionMTPPolicy()
	return productionMTPCompareReport{
		Version:              1,
		Command:              "production-mtp-compare",
		TargetOnlyReportPath: targetPath,
		MTPReportPath:        mtpPath,
		Policy:               policy,
		SameModelPath:        sameModel,
		SamePromptShape:      sameShape,
		TargetOnlySummary:    productionMTPCompareSummaryFromDriver(target),
		MTPSummary:           productionMTPCompareSummaryFromDriver(mtp),
		Evidence:             evidence,
		Decision:             mlx.EvaluateProductionMTPPromotion(policy, evidence),
	}
}

func productionMTPCompareSameModelPath(target, mtp driverProfileReport) bool {
	return target.ModelPath != "" && target.ModelPath == mtp.ModelPath
}

func productionMTPCompareSamePromptShape(target, mtp driverProfileReport) bool {
	return target.PromptBytes == mtp.PromptBytes &&
		target.PromptSuffixBytes == mtp.PromptSuffixBytes &&
		target.PromptChunkBytes == mtp.PromptChunkBytes &&
		target.PromptRepeat == mtp.PromptRepeat &&
		target.MaxTokens == mtp.MaxTokens &&
		target.RequestedRuns == mtp.RequestedRuns &&
		target.Chat == mtp.Chat
}

func productionMTPCompareQualityFlags(raw string, sameModel, sameShape bool) []string {
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
	return flags
}

func productionMTPCompareMTPVisibleTokensPerSec(summary driverProfileSummary) float64 {
	if summary.MTPVisibleTokensPerSecAverage > 0 {
		return summary.MTPVisibleTokensPerSecAverage
	}
	return summary.DecodeTokensPerSecAverage
}

func productionMTPCompareSummaryFromDriver(report driverProfileReport) productionMTPCompareSummary {
	return productionMTPCompareSummary{
		ModelPath:                     report.ModelPath,
		SpeculativeDraftModelPath:     report.SpeculativeDraftModelPath,
		SpeculativeDraftTokens:        report.SpeculativeDraftTokens,
		PromptBytes:                   report.PromptBytes,
		PromptSuffixBytes:             report.PromptSuffixBytes,
		PromptChunkBytes:              report.PromptChunkBytes,
		PromptRepeat:                  report.PromptRepeat,
		MaxTokens:                     report.MaxTokens,
		RequestedRuns:                 report.RequestedRuns,
		Chat:                          report.Chat,
		SuccessfulRuns:                report.Summary.SuccessfulRuns,
		VisibleTokens:                 report.Summary.VisibleTokens,
		GeneratedTokens:               report.Summary.GeneratedTokens,
		TotalDuration:                 report.Summary.TotalDuration,
		DecodeTokensPerSecAverage:     report.Summary.DecodeTokensPerSecAverage,
		PeakMemoryBytes:               report.Summary.PeakMemoryBytes,
		ActiveMemoryBytes:             report.Summary.ActiveMemoryBytes,
		CacheMemoryBytes:              report.Summary.CacheMemoryBytes,
		ActivePlusCacheMemoryBytes:    report.Summary.ActivePlusCacheMemoryBytes,
		MTPVisibleTokensPerSecAverage: report.Summary.MTPVisibleTokensPerSecAverage,
		MTPTargetTokensPerSecAverage:  report.Summary.MTPTargetTokensPerSecAverage,
		MTPWarmDecodeTokensPerSec:     report.Summary.MTPWarmDecodeTokensPerSecAverage,
		MTPAcceptanceRateAverage:      report.Summary.MTPAcceptanceRateAverage,
		MTPDraftTokenSchedule:         productionMTPCompareDraftTokenSchedule(report),
		MTPProposedTokens:             report.Summary.MTPProposedTokens,
		MTPAcceptedTokens:             report.Summary.MTPAcceptedTokens,
		MTPRejectedTokens:             report.Summary.MTPRejectedTokens,
		MTPTargetVerifyCalls:          report.Summary.MTPTargetVerifyCalls,
		MTPDraftCalls:                 report.Summary.MTPDraftCalls,
	}
}

func productionMTPCompareDraftTokenSchedule(report driverProfileReport) []int {
	for _, run := range report.Runs {
		if run.Metrics.MTP == nil || len(run.Metrics.MTP.DraftTokenSchedule) == 0 {
			continue
		}
		return append([]int(nil), run.Metrics.MTP.DraftTokenSchedule...)
	}
	return nil
}

func printProductionMTPCompareReport(stdout io.Writer, report productionMTPCompareReport) {
	core.WriteString(stdout, core.Sprintf("production MTP comparison: promote=%t (%s)\n", report.Decision.EnableByDefault, report.Decision.Reason))
	core.WriteString(stdout, core.Sprintf("target-only: %.1f visible tok/s, wall %s, peak memory %d bytes\n",
		report.Evidence.TargetOnlyVisibleTokensPerSec,
		report.Evidence.TargetOnlyWallDuration,
		report.TargetOnlySummary.PeakMemoryBytes,
	))
	core.WriteString(stdout, core.Sprintf("mtp: %.1f visible tok/s, wall %s, draft_tokens %d, target %.1f tok/s, proposed/accepted/rejected %d/%d/%d, target verifies %d, peak memory %d bytes\n",
		report.Evidence.MTPVisibleTokensPerSec,
		report.Evidence.MTPWallDuration,
		report.MTPSummary.SpeculativeDraftTokens,
		report.MTPSummary.MTPTargetTokensPerSecAverage,
		report.Evidence.MTPProposedTokens,
		report.Evidence.MTPAcceptedTokens,
		report.Evidence.MTPRejectedTokens,
		report.Evidence.MTPTargetVerifyCalls,
		report.MTPSummary.PeakMemoryBytes,
	))
	if len(report.Evidence.QualityFlags) > 0 {
		core.WriteString(stdout, core.Sprintf("quality flags: %s\n", core.Join(", ", report.Evidence.QualityFlags...)))
	}
}
