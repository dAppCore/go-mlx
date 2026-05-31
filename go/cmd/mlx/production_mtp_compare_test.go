// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"testing"
	"time"

	core "dappco.re/go"
	mlx "dappco.re/go/mlx"
)

func TestRunCommand_ProductionMTPCompareJSON_Good(t *testing.T) {
	dir := t.TempDir()
	targetPath := core.PathJoin(dir, "target.json")
	mtpPath := core.PathJoin(dir, "mtp.json")
	writeProductionMTPCompareReport(t, targetPath, productionMTPCompareTestReport(false))
	writeProductionMTPCompareReport(t, mtpPath, productionMTPCompareTestReport(true))
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"production-mtp-compare", "-json", "-turns", "10", "-greedy-match", targetPath, mtpPath}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	for _, want := range []string{
		`"same_model_path": true`,
		`"same_prompt_shape": true`,
		`"target_only_visible_tokens_per_sec": 100`,
		`"mtp_visible_tokens_per_sec": 125`,
		`"speculative_draft_tokens": 2`,
		`"mtp_draft_token_schedule": [`,
		`"mtp_target_tokens_per_sec_average": 110`,
		`"mtp_acceptance_rate_average": 0.75`,
		`"mtp_proposed_tokens": 40`,
		`"mtp_target_verify_calls": 20`,
		`"peak_memory_bytes": 4096`,
		`"active_memory_bytes": 2048`,
		`"enable_by_default": true`,
		`"reason": "MTP retained workflow is faster than target-only with greedy parity"`,
	} {
		if !core.Contains(stdout.String(), want) {
			t.Fatalf("stdout = %q, want %s", stdout.String(), want)
		}
	}
	if stderr.Len() != 0 {
		t.Fatalf("stderr = %q, want empty", stderr.String())
	}
}

func TestRunCommand_ProductionMTPCompareRejectsMissingGreedyParity_Bad(t *testing.T) {
	dir := t.TempDir()
	targetPath := core.PathJoin(dir, "target.json")
	mtpPath := core.PathJoin(dir, "mtp.json")
	writeProductionMTPCompareReport(t, targetPath, productionMTPCompareTestReport(false))
	writeProductionMTPCompareReport(t, mtpPath, productionMTPCompareTestReport(true))
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"production-mtp-compare", "-json", "-turns", "10", targetPath, mtpPath}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0 for an auditable rejection report; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	if !core.Contains(stdout.String(), `"enable_by_default": false`) || !core.Contains(stdout.String(), "greedy output parity") {
		t.Fatalf("stdout = %q, want rejected greedy-parity decision", stdout.String())
	}
}

func productionMTPCompareTestReport(mtp bool) driverProfileReport {
	report := driverProfileReport{
		Version:       1,
		ModelPath:     "/models/gemma4-e2b",
		PromptBytes:   4096,
		MaxTokens:     500,
		RequestedRuns: 10,
		Chat:          true,
		Summary: driverProfileSummary{
			SuccessfulRuns:             10,
			PromptTokensAverage:        32768,
			PromptTokensMin:            32768,
			PromptTokensMax:            32768,
			VisibleTokens:              5000,
			GeneratedTokens:            5000,
			DecodeTokensPerSecAverage:  100,
			TotalDuration:              10 * time.Second,
			PrefillTokensPerSecAverage: 2000,
			PeakMemoryBytes:            4096,
			ActiveMemoryBytes:          2048,
			CacheMemoryBytes:           512,
			ActivePlusCacheMemoryBytes: 2560,
		},
	}
	if mtp {
		report.SpeculativeDraftModelPath = "/models/gemma4-e2b-assistant"
		report.SpeculativeDraftTokens = 2
		report.Summary.TotalDuration = 8 * time.Second
		report.Summary.DecodeTokensPerSecAverage = 120
		report.Summary.MTPVisibleTokensPerSecAverage = 125
		report.Summary.MTPTargetTokensPerSecAverage = 110
		report.Summary.MTPWarmDecodeTokensPerSecAverage = 123
		report.Summary.MTPProposedTokens = 40
		report.Summary.MTPAcceptedTokens = 30
		report.Summary.MTPRejectedTokens = 10
		report.Summary.MTPTargetVerifyCalls = 20
		report.Summary.MTPDraftCalls = 20
		report.Summary.MTPAcceptanceRateAverage = 0.75
		report.Runs = []driverProfileRun{
			{
				Metrics: mlx.Metrics{
					MTP: &mlx.MTPMetrics{
						DraftTokenSchedule: []int{2, 2},
					},
				},
			},
		}
	}
	return report
}

func writeProductionMTPCompareReport(t *testing.T, path string, report driverProfileReport) {
	t.Helper()
	data := core.JSONMarshalIndent(report, "", "  ")
	if !data.OK {
		t.Fatalf("JSONMarshalIndent(%s): %v", path, data.Value)
	}
	if result := core.WriteFile(path, data.Value.([]byte), 0o644); !result.OK {
		t.Fatalf("WriteFile(%s): %v", path, result.Value)
	}
}
