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
	SameLoadPolicy       bool                               `json:"same_load_policy"`
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
	ContextLength                 int           `json:"context_length,omitempty"`
	PromptCache                   bool          `json:"prompt_cache,omitempty"`
	PromptCacheMinTokens          int           `json:"prompt_cache_min_tokens,omitempty"`
	CachePolicy                   string        `json:"cache_policy,omitempty"`
	CacheMode                     string        `json:"cache_mode,omitempty"`
	BatchSize                     int           `json:"batch_size,omitempty"`
	PrefillChunkSize              int           `json:"prefill_chunk_size,omitempty"`
	SuccessfulRuns                int           `json:"successful_runs,omitempty"`
	VisibleTokens                 int           `json:"visible_tokens,omitempty"`
	GeneratedTokens               int           `json:"generated_tokens,omitempty"`
	TotalDuration                 time.Duration `json:"total_duration,omitempty"`
	RestoreAvgDuration            time.Duration `json:"restore_duration_average,omitempty"`
	DecodeTokensPerSecAverage     float64       `json:"decode_tokens_per_sec_average,omitempty"`
	PeakMemoryBytes               uint64        `json:"peak_memory_bytes,omitempty"`
	ActiveMemoryBytes             uint64        `json:"active_memory_bytes,omitempty"`
	CacheMemoryBytes              uint64        `json:"cache_memory_bytes,omitempty"`
	ActivePlusCacheMemoryBytes    uint64        `json:"active_plus_cache_memory_bytes,omitempty"`
	EnergyJoules                  float64       `json:"energy_joules,omitempty"`
	PowerWatts                    float64       `json:"power_watts,omitempty"`
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

type productionMTPAssistantEvidenceInput struct {
	Architecture             string   `json:"assistant_architecture,omitempty"`
	OrderedEmbeddings        bool     `json:"assistant_ordered_embeddings"`
	Centroids                int      `json:"assistant_centroids,omitempty"`
	CentroidIntermediateTopK int      `json:"assistant_centroid_intermediate_top_k,omitempty"`
	FourLayerDrafter         bool     `json:"assistant_four_layer_drafter"`
	QualityFlags             []string `json:"-"`
}

func runProductionMTPCompareCommand(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet(cliCommandName("production-mtp-compare"), flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOut := fs.Bool("json", false, "print JSON target-only versus MTP promotion report")
	turns := fs.Int("turns", mlx.ProductionMTPPromotionMinRetainedTurns, "retained workflow turns represented by the compared reports")
	greedyMatch := fs.Bool("greedy-match", false, "mark target-only and MTP greedy visible outputs as matching")
	qualityFlags := fs.String("quality-flags", "", "comma-separated quality flags from manual output review; any flag blocks promotion")
	draftTokenSweeps := fs.String("draft-token-sweeps", "", "comma-separated MTP draft token counts covered by the benchmark matrix; required for promotion")
	powerWatts := fs.Float64("power-watts", 0, "fallback estimated average active watts when reports do not already include energy")
	assistantArchitecture := fs.String("assistant-architecture", "", "official assistant model_type/architecture evidence; production expects gemma4_assistant")
	assistantOrderedEmbeddings := fs.Bool("assistant-ordered-embeddings", false, "mark the assistant report as using ordered embedding centroid/token-ordering logits")
	assistantCentroids := fs.Int("assistant-centroids", 0, "assistant ordered-embedding centroid count from the verified config")
	assistantCentroidTopK := fs.Int("assistant-centroid-top-k", 0, "assistant ordered-embedding intermediate top-k from the verified config")
	assistantFourLayerDrafter := fs.Bool("assistant-four-layer-drafter", false, "mark the assistant report as the official four-layer drafter layout")
	officialPairReport := fs.String("official-pair-report", "", "JSON report from official-gemma4-pair-verify used to fill assistant layout evidence")
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
	if *powerWatts < 0 {
		core.WriteString(stderr, core.Sprintf("%s production-mtp-compare: power-watts must be >= 0\n", cliName()))
		return 2
	}
	if *assistantCentroids < 0 || *assistantCentroidTopK < 0 {
		core.WriteString(stderr, core.Sprintf("%s production-mtp-compare: assistant centroid counts must be >= 0\n", cliName()))
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
	observedDraftSweeps, err := productionMTPCompareObservedDraftTokenSweeps(*draftTokenSweeps, mtp)
	if err != nil {
		core.Print(stderr, "%s production-mtp-compare: %v", cliName(), err)
		return 2
	}

	assistantEvidence := productionMTPAssistantEvidenceInput{
		Architecture:             core.Trim(*assistantArchitecture),
		OrderedEmbeddings:        *assistantOrderedEmbeddings,
		Centroids:                *assistantCentroids,
		CentroidIntermediateTopK: *assistantCentroidTopK,
		FourLayerDrafter:         *assistantFourLayerDrafter,
	}
	if pairPath := core.Trim(*officialPairReport); pairPath != "" {
		assistantEvidence, err = readProductionMTPAssistantEvidenceFromPairReport(pairPath)
		if err != nil {
			core.Print(stderr, "%s production-mtp-compare: read official pair report: %v", cliName(), err)
			return 1
		}
	} else {
		assistantEvidence = mergeProductionMTPAssistantEvidence(assistantEvidence, productionMTPAssistantEvidenceFromDriverReport(mtp))
	}
	report := newProductionMTPCompareReport(targetPath, target, mtpPath, mtp, *turns, *greedyMatch, *qualityFlags, observedDraftSweeps, assistantEvidence, *powerWatts)
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
	return readProductionDriverProfileReport(path)
}

func readProductionDriverProfileReport(path string) (driverProfileReport, error) {
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

func readProductionMTPAssistantEvidenceFromPairReport(path string) (productionMTPAssistantEvidenceInput, error) {
	read := core.ReadFile(path)
	if !read.OK {
		return productionMTPAssistantEvidenceInput{}, core.Errorf("read %s: %v", path, read.Value)
	}
	var report mlx.OfficialGemma4E2BPairReport
	if result := core.JSONUnmarshal(read.Value.([]byte), &report); !result.OK {
		return productionMTPAssistantEvidenceInput{}, core.Errorf("decode %s: %v", path, result.Value)
	}
	return productionMTPAssistantEvidenceFromPairReport(report), nil
}

func productionMTPAssistantEvidenceFromPairReport(report mlx.OfficialGemma4E2BPairReport) productionMTPAssistantEvidenceInput {
	architecture := core.Trim(report.Assistant.Pack.Architecture)
	if architecture == "" {
		architecture = core.Trim(report.Assistant.Lock.ModelType)
	}
	evidence := productionMTPAssistantEvidenceInput{
		Architecture:             architecture,
		OrderedEmbeddings:        report.AssistantOrderedEmbeddings && report.AssistantOrderedEmbeddingTensorsOK,
		Centroids:                report.AssistantNumCentroids,
		CentroidIntermediateTopK: report.AssistantCentroidIntermediateTopK,
		FourLayerDrafter:         report.AssistantFourLayerDrafter,
	}
	if !report.PairOK {
		evidence.QualityFlags = append(evidence.QualityFlags, "assistant_pair_not_verified")
	}
	if !report.AssistantAttachable {
		evidence.QualityFlags = append(evidence.QualityFlags, "assistant_not_attachable")
	}
	if !report.AssistantProjectionTensorsOK {
		evidence.QualityFlags = append(evidence.QualityFlags, "assistant_projection_tensors_invalid")
	}
	if !report.AssistantOrderedEmbeddingTensorsOK {
		evidence.QualityFlags = append(evidence.QualityFlags, "assistant_ordered_embedding_tensors_invalid")
	}
	if len(report.AssistantMissingTensorNames) > 0 {
		evidence.QualityFlags = append(evidence.QualityFlags, "assistant_tensors_missing")
	}
	if len(report.AssistantInvalidTensorShapes) > 0 {
		evidence.QualityFlags = append(evidence.QualityFlags, "assistant_tensor_shapes_invalid")
	}
	return evidence
}

func productionMTPAssistantEvidenceFromDriverReport(report driverProfileReport) productionMTPAssistantEvidenceInput {
	if report.SpeculativeAssistantLayout == nil {
		return productionMTPAssistantEvidenceInput{}
	}
	layout := report.SpeculativeAssistantLayout
	return productionMTPAssistantEvidenceInput{
		Architecture:             core.Trim(layout.Architecture),
		OrderedEmbeddings:        layout.OrderedEmbeddings,
		Centroids:                layout.Centroids,
		CentroidIntermediateTopK: layout.CentroidIntermediateTopK,
		FourLayerDrafter:         layout.FourLayerDrafter,
	}
}

func mergeProductionMTPAssistantEvidence(primary, fallback productionMTPAssistantEvidenceInput) productionMTPAssistantEvidenceInput {
	if primary.Architecture == "" {
		primary.Architecture = fallback.Architecture
	}
	if !primary.OrderedEmbeddings {
		primary.OrderedEmbeddings = fallback.OrderedEmbeddings
	}
	if primary.Centroids == 0 {
		primary.Centroids = fallback.Centroids
	}
	if primary.CentroidIntermediateTopK == 0 {
		primary.CentroidIntermediateTopK = fallback.CentroidIntermediateTopK
	}
	if !primary.FourLayerDrafter {
		primary.FourLayerDrafter = fallback.FourLayerDrafter
	}
	primary.QualityFlags = append(primary.QualityFlags, fallback.QualityFlags...)
	return primary
}

func newProductionMTPCompareReport(targetPath string, target driverProfileReport, mtpPath string, mtp driverProfileReport, turns int, greedyMatch bool, qualityFlags string, observedDraftSweeps []int, assistantEvidence productionMTPAssistantEvidenceInput, powerWatts float64) productionMTPCompareReport {
	sameModel := productionMTPCompareSameModelPath(target, mtp)
	sameShape := productionMTPCompareSamePromptShape(target, mtp)
	sameLoad := productionMTPCompareSameLoadPolicy(target, mtp)
	mtpDraftSchedule := productionMTPCompareDraftTokenSchedule(mtp)
	policy := mlx.DefaultProductionMTPPolicy()
	flags := productionMTPCompareQualityFlags(qualityFlags, sameModel, sameShape, sameLoad, target, mtp, mtpDraftSchedule, observedDraftSweeps, policy.RequiredDraftTokenSweeps, powerWatts)
	flags = append(flags, assistantEvidence.QualityFlags...)
	evidencePowerWatts := productionMTPComparePowerWatts(target, mtp, powerWatts)
	evidence := mlx.ProductionMTPPromotionEvidence{
		RetainedWorkflow:                     sameModel && sameShape && sameLoad,
		Turns:                                turns,
		GreedyOutputMatches:                  greedyMatch,
		QualityFlags:                         flags,
		TargetOnlyVisibleTokensPerSec:        target.Summary.DecodeTokensPerSecAverage,
		MTPVisibleTokensPerSec:               productionMTPCompareMTPVisibleTokensPerSec(mtp.Summary),
		TargetOnlyInputOutputTokensPerSec:    productionCompareInputOutputTokensPerSec(target),
		MTPInputOutputTokensPerSec:           productionCompareInputOutputTokensPerSec(mtp),
		MTPTargetTokensPerSec:                mtp.Summary.MTPTargetTokensPerSecAverage,
		MTPWarmDecodeTokensPerSec:            mtp.Summary.MTPWarmDecodeTokensPerSecAverage,
		TargetOnlyWallDuration:               target.Summary.TotalDuration,
		MTPWallDuration:                      mtp.Summary.TotalDuration,
		TargetOnlyRestoreDuration:            target.Summary.RestoreAvgDuration,
		MTPRestoreDuration:                   mtp.Summary.RestoreAvgDuration,
		TargetOnlyPeakMemoryBytes:            target.Summary.PeakMemoryBytes,
		MTPPeakMemoryBytes:                   mtp.Summary.PeakMemoryBytes,
		TargetOnlyActivePlusCacheMemoryBytes: target.Summary.ActivePlusCacheMemoryBytes,
		MTPActivePlusCacheMemoryBytes:        mtp.Summary.ActivePlusCacheMemoryBytes,
		TargetOnlyEnergyJoules:               productionMTPCompareEnergyJoules(target, powerWatts),
		MTPEnergyJoules:                      productionMTPCompareEnergyJoules(mtp, powerWatts),
		EstimatedPowerWatts:                  evidencePowerWatts,
		SameLoadPolicy:                       sameLoad,
		TargetOnlyCachePolicy:                productionMTPCompareLoadCachePolicy(target),
		MTPCachePolicy:                       productionMTPCompareLoadCachePolicy(mtp),
		TargetOnlyCacheMode:                  productionMTPCompareLoadCacheMode(target),
		MTPCacheMode:                         productionMTPCompareLoadCacheMode(mtp),
		TargetOnlyContextLength:              productionMTPCompareLoadContextLength(target),
		MTPContextLength:                     productionMTPCompareLoadContextLength(mtp),
		SpeculativeDraftModelPath:            mtp.SpeculativeDraftModelPath,
		SpeculativeDraftTokens:               mtp.SpeculativeDraftTokens,
		AssistantArchitecture:                assistantEvidence.Architecture,
		AssistantOrderedEmbeddings:           assistantEvidence.OrderedEmbeddings,
		AssistantCentroids:                   assistantEvidence.Centroids,
		AssistantCentroidIntermediateTopK:    assistantEvidence.CentroidIntermediateTopK,
		AssistantFourLayerDrafter:            assistantEvidence.FourLayerDrafter,
		MTPDraftTokenSchedule:                mtpDraftSchedule,
		MTPObservedDraftTokenSweeps:          observedDraftSweeps,
		MTPProposedTokens:                    mtp.Summary.MTPProposedTokens,
		MTPAcceptedTokens:                    mtp.Summary.MTPAcceptedTokens,
		MTPRejectedTokens:                    mtp.Summary.MTPRejectedTokens,
		MTPTargetVerifyCalls:                 mtp.Summary.MTPTargetVerifyCalls,
		MTPDraftCalls:                        mtp.Summary.MTPDraftCalls,
	}
	return productionMTPCompareReport{
		Version:              1,
		Command:              "production-mtp-compare",
		TargetOnlyReportPath: targetPath,
		MTPReportPath:        mtpPath,
		Policy:               policy,
		SameModelPath:        sameModel,
		SamePromptShape:      sameShape,
		SameLoadPolicy:       sameLoad,
		TargetOnlySummary:    productionMTPCompareSummaryFromDriver(target, powerWatts),
		MTPSummary:           productionMTPCompareSummaryFromDriver(mtp, powerWatts),
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

func productionMTPCompareSameLoadPolicy(target, mtp driverProfileReport) bool {
	if target.Load == nil || mtp.Load == nil {
		return false
	}
	return target.Load.ContextLength == mtp.Load.ContextLength &&
		target.Load.PromptCache == mtp.Load.PromptCache &&
		target.Load.PromptCacheMinTokens == mtp.Load.PromptCacheMinTokens &&
		target.Load.CachePolicy == mtp.Load.CachePolicy &&
		target.Load.CacheMode == mtp.Load.CacheMode &&
		target.Load.BatchSize == mtp.Load.BatchSize &&
		target.Load.PrefillChunkSize == mtp.Load.PrefillChunkSize
}

func productionMTPCompareLoadCachePolicy(report driverProfileReport) string {
	if report.Load == nil {
		return ""
	}
	return report.Load.CachePolicy
}

func productionMTPCompareLoadCacheMode(report driverProfileReport) string {
	if report.Load == nil {
		return ""
	}
	return report.Load.CacheMode
}

func productionMTPCompareLoadContextLength(report driverProfileReport) int {
	if report.Load == nil {
		return 0
	}
	return report.Load.ContextLength
}

func productionMTPCompareQualityFlags(raw string, sameModel, sameShape, sameLoad bool, target, mtp driverProfileReport, mtpDraftSchedule, observedDraftSweeps, requiredDraftSweeps []int, powerWatts float64) []string {
	flags := make([]string, 0, 24)
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
	if core.Trim(target.SpeculativeDraftModelPath) != "" || target.SpeculativeDraftTokens > 0 || len(productionMTPCompareDraftTokenSchedule(target)) > 0 {
		flags = append(flags, "target_only_has_speculative_draft")
	}
	if productionMTPCompareTargetOnlyHasMTPMetrics(target) {
		flags = append(flags, "target_only_has_mtp_metrics")
	}
	if core.Trim(mtp.SpeculativeDraftModelPath) == "" {
		flags = append(flags, "mtp_draft_model_missing")
	}
	if mtp.SpeculativeDraftTokens <= 0 {
		flags = append(flags, "mtp_draft_tokens_missing")
	}
	if len(mtpDraftSchedule) == 0 {
		flags = append(flags, "mtp_draft_schedule_missing")
	}
	for _, missing := range productionMTPCompareMissingDraftTokenSweeps(requiredDraftSweeps, observedDraftSweeps) {
		flags = append(flags, core.Sprintf("mtp_draft_token_sweep_missing_%d", missing))
	}
	if target.Summary.DecodeTokensPerSecAverage <= 0 {
		flags = append(flags, "target_only_visible_throughput_missing")
	}
	if productionMTPCompareMTPVisibleTokensPerSec(mtp.Summary) <= 0 {
		flags = append(flags, "mtp_visible_throughput_missing")
	}
	if mtp.Summary.MTPTargetTokensPerSecAverage <= 0 {
		flags = append(flags, "mtp_target_throughput_missing")
	}
	if mtp.Summary.MTPWarmDecodeTokensPerSecAverage <= 0 {
		flags = append(flags, "mtp_warm_decode_missing")
	}
	if mtp.Summary.MTPProposedTokens <= 0 {
		flags = append(flags, "mtp_proposed_tokens_missing")
	}
	if mtp.Summary.MTPTargetVerifyCalls <= 0 {
		flags = append(flags, "mtp_target_verify_calls_missing")
	}
	if mtp.Summary.MTPDraftCalls <= 0 {
		flags = append(flags, "mtp_draft_calls_missing")
	}
	if mtp.Summary.MTPProposedTokens > 0 && mtp.Summary.MTPAcceptedTokens+mtp.Summary.MTPRejectedTokens != mtp.Summary.MTPProposedTokens {
		flags = append(flags, "mtp_draft_accounting_mismatch")
	}
	if mtp.Summary.MTPProposedTokens > 0 && mtp.Summary.MTPAcceptedTokens == 0 {
		flags = append(flags, "mtp_accepted_tokens_missing")
	}
	flags = productionMTPCompareMetricEvidenceFlags(flags, "target_only", target, powerWatts)
	flags = productionMTPCompareMetricEvidenceFlags(flags, "mtp", mtp, powerWatts)
	if productionMTPComparePowerWatts(target, mtp, powerWatts) <= 0 {
		flags = append(flags, "estimated_power_watts_missing")
	}
	return flags
}

func productionMTPCompareTargetOnlyHasMTPMetrics(report driverProfileReport) bool {
	if report.SpeculativeAssistantLayout != nil {
		return true
	}
	summary := report.Summary
	if summary.MTPVisibleTokensPerSecAverage > 0 ||
		summary.MTPTargetTokensPerSecAverage > 0 ||
		summary.MTPWarmDecodeTokensPerSecAverage > 0 ||
		summary.MTPAcceptanceRateAverage > 0 ||
		summary.MTPProposedTokens > 0 ||
		summary.MTPAcceptedTokens > 0 ||
		summary.MTPRejectedTokens > 0 ||
		summary.MTPTargetVerifyCalls > 0 ||
		summary.MTPDraftCalls > 0 {
		return true
	}
	for _, run := range report.Runs {
		if run.Metrics.MTP != nil {
			return true
		}
	}
	return false
}

func productionMTPCompareObservedDraftTokenSweeps(raw string, report driverProfileReport) ([]int, error) {
	raw = core.Trim(raw)
	if raw != "" {
		return productionMTPCompareParseDraftTokenSweeps(raw)
	}
	return productionMTPCompareAppendUniqueInt(nil, report.SpeculativeDraftTokens), nil
}

func productionMTPCompareParseDraftTokenSweeps(raw string) ([]int, error) {
	parts := core.Split(raw, ",")
	values := make([]int, 0, len(parts))
	for _, part := range parts {
		part = core.Trim(part)
		if part == "" {
			continue
		}
		parsed := core.ParseInt(part, 10, 64)
		if !parsed.OK {
			return nil, core.Errorf("invalid draft-token-sweeps value %q", part)
		}
		value := int(parsed.Value.(int64))
		if value <= 0 {
			return nil, core.Errorf("draft-token-sweeps values must be positive")
		}
		values = productionMTPCompareAppendUniqueInt(values, value)
	}
	return values, nil
}

func productionMTPCompareMissingDraftTokenSweeps(required, observed []int) []int {
	seen := make(map[int]bool, len(observed))
	for _, value := range observed {
		if value > 0 {
			seen[value] = true
		}
	}
	missing := make([]int, 0, len(required))
	for _, value := range required {
		if value > 0 && !seen[value] {
			missing = append(missing, value)
		}
	}
	return missing
}

func productionMTPCompareAppendUniqueInt(values []int, value int) []int {
	if value <= 0 {
		return values
	}
	for _, existing := range values {
		if existing == value {
			return values
		}
	}
	return append(values, value)
}

func productionMTPCompareMetricEvidenceFlags(flags []string, prefix string, report driverProfileReport, powerWatts float64) []string {
	if report.Summary.TotalDuration <= 0 {
		flags = append(flags, prefix+"_wall_duration_missing")
	}
	if report.Summary.RestoreAvgDuration <= 0 {
		flags = append(flags, prefix+"_restore_duration_missing")
	}
	if report.Summary.PeakMemoryBytes == 0 {
		flags = append(flags, prefix+"_peak_memory_missing")
	}
	if report.Summary.ActivePlusCacheMemoryBytes == 0 {
		flags = append(flags, prefix+"_active_plus_cache_memory_missing")
	}
	if productionMTPCompareEnergyJoules(report, powerWatts) <= 0 {
		flags = append(flags, prefix+"_energy_missing")
	}
	return flags
}

func productionMTPCompareMTPVisibleTokensPerSec(summary driverProfileSummary) float64 {
	if summary.MTPVisibleTokensPerSecAverage > 0 {
		return summary.MTPVisibleTokensPerSecAverage
	}
	return summary.DecodeTokensPerSecAverage
}

func productionCompareInputOutputTokensPerSec(report driverProfileReport) float64 {
	if report.Summary.TotalDuration <= 0 || report.Summary.SuccessfulRuns <= 0 {
		return 0
	}
	inputTokens := report.Summary.PromptTokensAverage * float64(report.Summary.SuccessfulRuns)
	totalTokens := inputTokens + float64(report.Summary.GeneratedTokens)
	if totalTokens <= 0 {
		return 0
	}
	return totalTokens / report.Summary.TotalDuration.Seconds()
}

func productionMTPCompareEnergyJoules(report driverProfileReport, fallbackPowerWatts float64) float64 {
	if report.EstimatedEnergy != nil && report.EstimatedEnergy.TotalJoules > 0 {
		return report.EstimatedEnergy.TotalJoules
	}
	if fallbackPowerWatts > 0 && report.Summary.TotalDuration > 0 {
		return durationJoules(report.Summary.TotalDuration, fallbackPowerWatts)
	}
	return 0
}

func productionMTPComparePowerWatts(first, second driverProfileReport, fallbackPowerWatts float64) float64 {
	if first.EstimatedEnergy != nil && first.EstimatedEnergy.PowerWatts > 0 {
		return first.EstimatedEnergy.PowerWatts
	}
	if second.EstimatedEnergy != nil && second.EstimatedEnergy.PowerWatts > 0 {
		return second.EstimatedEnergy.PowerWatts
	}
	return fallbackPowerWatts
}

func productionMTPCompareSummaryFromDriver(report driverProfileReport, powerWatts float64) productionMTPCompareSummary {
	summary := productionMTPCompareSummary{
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
		RestoreAvgDuration:            report.Summary.RestoreAvgDuration,
		DecodeTokensPerSecAverage:     report.Summary.DecodeTokensPerSecAverage,
		PeakMemoryBytes:               report.Summary.PeakMemoryBytes,
		ActiveMemoryBytes:             report.Summary.ActiveMemoryBytes,
		CacheMemoryBytes:              report.Summary.CacheMemoryBytes,
		ActivePlusCacheMemoryBytes:    report.Summary.ActivePlusCacheMemoryBytes,
		EnergyJoules:                  productionMTPCompareEnergyJoules(report, powerWatts),
		PowerWatts:                    productionMTPComparePowerWatts(report, driverProfileReport{}, powerWatts),
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
	if report.Load != nil {
		summary.ContextLength = report.Load.ContextLength
		summary.PromptCache = report.Load.PromptCache
		summary.PromptCacheMinTokens = report.Load.PromptCacheMinTokens
		summary.CachePolicy = report.Load.CachePolicy
		summary.CacheMode = report.Load.CacheMode
		summary.BatchSize = report.Load.BatchSize
		summary.PrefillChunkSize = report.Load.PrefillChunkSize
	}
	return summary
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
	core.WriteString(stdout, core.Sprintf("target-only: %.1f visible tok/s, wall %s, restore %s, peak memory %d bytes, energy %.1f J\n",
		report.Evidence.TargetOnlyVisibleTokensPerSec,
		report.Evidence.TargetOnlyWallDuration,
		report.Evidence.TargetOnlyRestoreDuration,
		report.TargetOnlySummary.PeakMemoryBytes,
		report.Evidence.TargetOnlyEnergyJoules,
	))
	core.WriteString(stdout, core.Sprintf("mtp: %.1f visible tok/s, wall %s, restore %s, draft_tokens %d, target %.1f tok/s, proposed/accepted/rejected %d/%d/%d, target verifies %d, draft calls %d, peak memory %d bytes, energy %.1f J\n",
		report.Evidence.MTPVisibleTokensPerSec,
		report.Evidence.MTPWallDuration,
		report.Evidence.MTPRestoreDuration,
		report.MTPSummary.SpeculativeDraftTokens,
		report.MTPSummary.MTPTargetTokensPerSecAverage,
		report.Evidence.MTPProposedTokens,
		report.Evidence.MTPAcceptedTokens,
		report.Evidence.MTPRejectedTokens,
		report.Evidence.MTPTargetVerifyCalls,
		report.Evidence.MTPDraftCalls,
		report.MTPSummary.PeakMemoryBytes,
		report.Evidence.MTPEnergyJoules,
	))
	core.WriteString(stdout, core.Sprintf("assistant: architecture %s, ordered_embeddings %t, centroids %d, centroid_top_k %d, four_layer_drafter %t\n",
		report.Evidence.AssistantArchitecture,
		report.Evidence.AssistantOrderedEmbeddings,
		report.Evidence.AssistantCentroids,
		report.Evidence.AssistantCentroidIntermediateTopK,
		report.Evidence.AssistantFourLayerDrafter,
	))
	if len(report.Evidence.QualityFlags) > 0 {
		core.WriteString(stdout, core.Sprintf("quality flags: %s\n", core.Join(", ", report.Evidence.QualityFlags...)))
	}
}
