// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"testing"
	"time"

	core "dappco.re/go"
	mlx "dappco.re/go/mlx"
	"dappco.re/go/mlx/memory"
)

func TestRunCommand_ProductionMTPCompareJSON_Good(t *testing.T) {
	dir := t.TempDir()
	targetPath := core.PathJoin(dir, "target.json")
	mtpPath := core.PathJoin(dir, "mtp.json")
	pairPath := core.PathJoin(dir, "pair.json")
	writeProductionMTPCompareReport(t, targetPath, productionMTPCompareTestReport(false))
	writeProductionMTPCompareReport(t, mtpPath, productionMTPCompareTestReport(true))
	writeProductionMTPPairReport(t, pairPath, productionMTPCompareTestPairReport(true))
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	args := []string{"production-mtp-compare", "-json", "-turns", "10", "-greedy-match", "-draft-token-sweeps", "1,2,4", "-official-pair-report", pairPath, targetPath, mtpPath}
	code := runCommand(context.Background(), args, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	for _, want := range []string{
		`"same_model_path": true`,
		`"same_prompt_shape": true`,
		`"same_load_policy": true`,
		`"cache_mode": "paged"`,
		`"target_only_visible_tokens_per_sec": 100`,
		`"mtp_visible_tokens_per_sec": 125`,
		`"target_only_input_output_tokens_per_sec": 33268`,
		`"mtp_input_output_tokens_per_sec": 41585`,
		`"target_only_restore_duration": 100000000`,
		`"mtp_restore_duration": 80000000`,
		`"target_only_peak_memory_bytes": 4096`,
		`"mtp_peak_memory_bytes": 3584`,
		`"target_only_active_plus_cache_memory_bytes": 2560`,
		`"mtp_active_plus_cache_memory_bytes": 2304`,
		`"target_only_energy_joules": 1000`,
		`"mtp_energy_joules": 760`,
		`"estimated_power_watts": 100`,
		`"target_only_cache_policy": "full"`,
		`"mtp_cache_policy": "full"`,
		`"target_only_cache_mode": "paged"`,
		`"mtp_cache_mode": "paged"`,
		`"target_only_context_length": 32768`,
		`"mtp_context_length": 32768`,
		`"speculative_draft_tokens": 2`,
		`"assistant_architecture": "gemma4_assistant"`,
		`"assistant_ordered_embeddings": true`,
		`"assistant_centroids": 2048`,
		`"assistant_centroid_intermediate_top_k": 32`,
		`"assistant_four_layer_drafter": true`,
		`"assistant_token_ordering_dtype": "I64"`,
		`"assistant_token_ordering_shape": [`,
		`"official_pair_verified": true`,
		`"official_target_model_id": "google/gemma-4-E2B-it"`,
		`"official_target_revision": "905e84b50c4d2a365ebde34e685027578e6728db"`,
		`"official_assistant_model_id": "google/gemma-4-E2B-it-assistant"`,
		`"official_assistant_revision": "5810c41a67974da9c7bd6f3e6c69d5d13854d9f0"`,
		`"required_draft_token_sweeps": [`,
		`"mtp_observed_draft_token_sweeps": [`,
		`"mtp_draft_token_schedule": [`,
		`"mtp_target_tokens_per_sec_average": 110`,
		`"mtp_acceptance_rate_average": 0.75`,
		`"mtp_proposed_tokens": 40`,
		`"mtp_target_verify_calls": 20`,
		`"peak_memory_bytes": 4096`,
		`"restore_duration_average": 100000000`,
		`"energy_joules": 1000`,
		`"energy_joules": 760`,
		`"power_watts": 100`,
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

func TestRunCommand_ProductionMTPCompareAllowsTargetOnlyDefaultDraftTokens_Good(t *testing.T) {
	dir := t.TempDir()
	targetPath := core.PathJoin(dir, "target.json")
	mtpPath := core.PathJoin(dir, "mtp.json")
	pairPath := core.PathJoin(dir, "pair.json")
	targetReport := productionMTPCompareTestReport(false)
	targetReport.SpeculativeDraftTokens = mlx.ProductionMTPDefaultDraftTokens
	writeProductionMTPCompareReport(t, targetPath, targetReport)
	writeProductionMTPCompareReport(t, mtpPath, productionMTPCompareTestReport(true))
	writeProductionMTPPairReport(t, pairPath, productionMTPCompareTestPairReport(true))
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	args := []string{"production-mtp-compare", "-json", "-turns", "10", "-greedy-match", "-draft-token-sweeps", "1,2,4", "-official-pair-report", pairPath, targetPath, mtpPath}
	code := runCommand(context.Background(), args, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	if core.Contains(stdout.String(), "target_only_has_speculative_draft") {
		t.Fatalf("stdout = %q, want target-only default draft token field ignored", stdout.String())
	}
	if !core.Contains(stdout.String(), `"enable_by_default": true`) {
		t.Fatalf("stdout = %q, want default draft token field not to block promotion", stdout.String())
	}
}

func TestRunCommand_ProductionMTPCompareAggregatesDraftSweepsFromMTPReports_Good(t *testing.T) {
	dir := t.TempDir()
	targetPath := core.PathJoin(dir, "target.json")
	mtp2Path := core.PathJoin(dir, "mtp-2.json")
	mtp1Path := core.PathJoin(dir, "mtp-1.json")
	mtp4Path := core.PathJoin(dir, "mtp-4.json")
	pairPath := core.PathJoin(dir, "pair.json")
	mtp2 := productionMTPCompareTestReport(true)
	mtp2.Runs[0].Metrics.MTP.DraftTokenSchedule = []int{2, 2}
	mtp1 := productionMTPCompareTestReport(true)
	mtp1.SpeculativeDraftTokens = 1
	mtp1.Runs[0].Metrics.MTP.DraftTokenSchedule = []int{1, 1}
	mtp4 := productionMTPCompareTestReport(true)
	mtp4.SpeculativeDraftTokens = 4
	mtp4.Runs[0].Metrics.MTP.DraftTokenSchedule = []int{4, 4}
	writeProductionMTPCompareReport(t, targetPath, productionMTPCompareTestReport(false))
	writeProductionMTPCompareReport(t, mtp2Path, mtp2)
	writeProductionMTPCompareReport(t, mtp1Path, mtp1)
	writeProductionMTPCompareReport(t, mtp4Path, mtp4)
	writeProductionMTPPairReport(t, pairPath, productionMTPCompareTestPairReport(true))
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	args := []string{"production-mtp-compare", "-json", "-turns", "10", "-greedy-match", "-official-pair-report", pairPath, targetPath, mtp2Path, mtp1Path, mtp4Path}
	code := runCommand(context.Background(), args, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	for _, want := range []string{
		`"mtp_report_paths": [`,
		`"mtp_observed_draft_token_sweeps": [`,
		`"enable_by_default": true`,
		`"reason": "MTP retained workflow is faster than target-only with greedy parity"`,
	} {
		if !core.Contains(stdout.String(), want) {
			t.Fatalf("stdout = %q, want %s", stdout.String(), want)
		}
	}
	for _, bad := range []string{
		`mtp_draft_token_sweep_missing_1`,
		`mtp_draft_token_sweep_missing_4`,
		`mtp_declared_draft_token_sweep_unobserved`,
	} {
		if core.Contains(stdout.String(), bad) {
			t.Fatalf("stdout = %q, want no %s", stdout.String(), bad)
		}
	}
	if stderr.Len() != 0 {
		t.Fatalf("stderr = %q, want empty", stderr.String())
	}
}

func TestRunCommand_ProductionMTPCompareUsesDriverAssistantLayout_Good(t *testing.T) {
	dir := t.TempDir()
	targetPath := core.PathJoin(dir, "target.json")
	mtpPath := core.PathJoin(dir, "mtp.json")
	target := productionMTPCompareTestReport(false)
	mtp := productionMTPCompareTestReport(true)
	mtp.SpeculativeAssistantLayout = &mlx.SpeculativeAssistantLayout{
		Architecture:             "gemma4_assistant",
		OrderedEmbeddings:        true,
		Centroids:                2048,
		CentroidIntermediateTopK: 32,
		FourLayerDrafter:         true,
		TokenOrderingDType:       "int64",
		TokenOrderingShape:       []int{2048, 128},
	}
	writeProductionMTPCompareReport(t, targetPath, target)
	writeProductionMTPCompareReport(t, mtpPath, mtp)
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{
		"production-mtp-compare",
		"-json",
		"-turns", "10",
		"-greedy-match",
		"-draft-token-sweeps", "1,2,4",
		targetPath,
		mtpPath,
	}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	for _, want := range []string{
		`"assistant_architecture": "gemma4_assistant"`,
		`"assistant_ordered_embeddings": true`,
		`"assistant_centroids": 2048`,
		`"assistant_centroid_intermediate_top_k": 32`,
		`"assistant_four_layer_drafter": true`,
		`"assistant_token_ordering_dtype": "int64"`,
		`"assistant_token_ordering_shape": [`,
		`"official_pair_verified": false`,
		`"enable_by_default": false`,
		`"reason": "verified official Gemma 4 target+assistant pair evidence is required"`,
	} {
		if !core.Contains(stdout.String(), want) {
			t.Fatalf("stdout = %q, want %s", stdout.String(), want)
		}
	}
	if stderr.Len() != 0 {
		t.Fatalf("stderr = %q, want empty", stderr.String())
	}
}

func TestRunCommand_ProductionMTPCompareRejectsMissingAssistantLayoutEvidence_Bad(t *testing.T) {
	dir := t.TempDir()
	targetPath := core.PathJoin(dir, "target.json")
	mtpPath := core.PathJoin(dir, "mtp.json")
	writeProductionMTPCompareReport(t, targetPath, productionMTPCompareTestReport(false))
	writeProductionMTPCompareReport(t, mtpPath, productionMTPCompareTestReport(true))
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"production-mtp-compare", "-json", "-turns", "10", "-greedy-match", "-draft-token-sweeps", "1,2,4", targetPath, mtpPath}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0 for an auditable rejection report; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	for _, want := range []string{
		`"enable_by_default": false`,
		`"assistant_ordered_embeddings": false`,
		`"reason": "official Gemma 4 assistant ordered-embedding evidence is required"`,
	} {
		if !core.Contains(stdout.String(), want) {
			t.Fatalf("stdout = %q, want %s", stdout.String(), want)
		}
	}
}

func TestRunCommand_ProductionMTPCompareRejectsFailedOfficialPairReport_Bad(t *testing.T) {
	dir := t.TempDir()
	targetPath := core.PathJoin(dir, "target.json")
	mtpPath := core.PathJoin(dir, "mtp.json")
	pairPath := core.PathJoin(dir, "pair.json")
	writeProductionMTPCompareReport(t, targetPath, productionMTPCompareTestReport(false))
	writeProductionMTPCompareReport(t, mtpPath, productionMTPCompareTestReport(true))
	writeProductionMTPPairReport(t, pairPath, productionMTPCompareTestPairReport(false))
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"production-mtp-compare", "-json", "-turns", "10", "-greedy-match", "-draft-token-sweeps", "1,2,4", "-official-pair-report", pairPath, targetPath, mtpPath}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0 for an auditable rejection report; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	for _, want := range []string{
		`"enable_by_default": false`,
		`"assistant_pair_not_verified"`,
		`"assistant_not_attachable"`,
		`"assistant_ordered_embedding_tensors_invalid"`,
		`"quality flags must be empty before MTP promotion"`,
	} {
		if !core.Contains(stdout.String(), want) {
			t.Fatalf("stdout = %q, want %s", stdout.String(), want)
		}
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

func TestRunCommand_ProductionMTPCompareUsesOutputTokenHashForGreedyParity_Good(t *testing.T) {
	dir := t.TempDir()
	targetPath := core.PathJoin(dir, "target.json")
	mtpPath := core.PathJoin(dir, "mtp.json")
	pairPath := core.PathJoin(dir, "pair.json")
	targetReport := productionMTPCompareTestReport(false)
	mtpReport := productionMTPCompareTestReport(true)
	targetReport.Summary.OutputTokenIDSHA256 = "same-visible-token-sequence"
	targetReport.Summary.OutputTokenIDSHA256Consistent = true
	mtpReport.Summary.OutputTokenIDSHA256 = "same-visible-token-sequence"
	mtpReport.Summary.OutputTokenIDSHA256Consistent = true
	writeProductionMTPCompareReport(t, targetPath, targetReport)
	writeProductionMTPCompareReport(t, mtpPath, mtpReport)
	writeProductionMTPPairReport(t, pairPath, productionMTPCompareTestPairReport(true))
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"production-mtp-compare", "-json", "-turns", "10", "-draft-token-sweeps", "1,2,4", "-official-pair-report", pairPath, targetPath, mtpPath}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	for _, want := range []string{
		`"greedy_output_matches": true`,
		`"output_token_ids_sha256": "same-visible-token-sequence"`,
		`"output_token_ids_sha256_consistent": true`,
		`"official_pair_verified": true`,
		`"enable_by_default": true`,
	} {
		if !core.Contains(stdout.String(), want) {
			t.Fatalf("stdout = %q, want %s", stdout.String(), want)
		}
	}
	if stderr.Len() != 0 {
		t.Fatalf("stderr = %q, want empty", stderr.String())
	}
}

func TestRunCommand_ProductionMTPCompareRejectsOutputTokenHashMismatch_Bad(t *testing.T) {
	dir := t.TempDir()
	targetPath := core.PathJoin(dir, "target.json")
	mtpPath := core.PathJoin(dir, "mtp.json")
	targetReport := productionMTPCompareTestReport(false)
	mtpReport := productionMTPCompareTestReport(true)
	targetReport.Summary.OutputTokenIDSHA256 = "target-visible-token-sequence"
	targetReport.Summary.OutputTokenIDSHA256Consistent = true
	mtpReport.Summary.OutputTokenIDSHA256 = "mtp-visible-token-sequence"
	mtpReport.Summary.OutputTokenIDSHA256Consistent = true
	writeProductionMTPCompareReport(t, targetPath, targetReport)
	writeProductionMTPCompareReport(t, mtpPath, mtpReport)
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"production-mtp-compare", "-json", "-turns", "10", "-greedy-match", "-draft-token-sweeps", "1,2,4", "-assistant-architecture", "gemma4_assistant", "-assistant-ordered-embeddings", "-assistant-centroids", "2048", "-assistant-centroid-top-k", "32", "-assistant-four-layer-drafter", "-assistant-token-ordering-dtype", "int64", "-assistant-token-ordering-shape", "2048,128", targetPath, mtpPath}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0 for an auditable rejection report; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	for _, want := range []string{
		`"greedy_output_matches": false`,
		`"greedy_output_hash_mismatch"`,
		`"enable_by_default": false`,
		`"greedy output parity is required before MTP promotion"`,
	} {
		if !core.Contains(stdout.String(), want) {
			t.Fatalf("stdout = %q, want %s", stdout.String(), want)
		}
	}
}

func TestRunCommand_ProductionMTPCompareRejectsMissingDraftEvidence_Bad(t *testing.T) {
	dir := t.TempDir()
	targetPath := core.PathJoin(dir, "target.json")
	mtpPath := core.PathJoin(dir, "mtp.json")
	mtpReport := productionMTPCompareTestReport(true)
	mtpReport.SpeculativeDraftModelPath = ""
	mtpReport.SpeculativeDraftTokens = 0
	mtpReport.Runs = nil
	writeProductionMTPCompareReport(t, targetPath, productionMTPCompareTestReport(false))
	writeProductionMTPCompareReport(t, mtpPath, mtpReport)
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"production-mtp-compare", "-json", "-turns", "10", "-greedy-match", targetPath, mtpPath}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0 for an auditable rejection report; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	for _, want := range []string{
		`"enable_by_default": false`,
		`"mtp_draft_model_missing"`,
		`"mtp_draft_tokens_missing"`,
		`"mtp_draft_schedule_missing"`,
	} {
		if !core.Contains(stdout.String(), want) {
			t.Fatalf("stdout = %q, want %s", stdout.String(), want)
		}
	}
}

func TestRunCommand_ProductionMTPCompareRejectsTargetOnlyMTPMetrics_Bad(t *testing.T) {
	dir := t.TempDir()
	targetPath := core.PathJoin(dir, "target.json")
	mtpPath := core.PathJoin(dir, "mtp.json")
	targetReport := productionMTPCompareTestReport(false)
	targetReport.Summary.MTPVisibleTokensPerSecAverage = 111
	targetReport.Summary.MTPTargetTokensPerSecAverage = 109
	targetReport.Summary.MTPWarmDecodeTokensPerSecAverage = 110
	targetReport.Summary.MTPAcceptanceRateAverage = 0.75
	targetReport.Summary.MTPProposedTokens = 4
	targetReport.Summary.MTPAcceptedTokens = 3
	targetReport.Summary.MTPRejectedTokens = 1
	targetReport.Summary.MTPTargetVerifyCalls = 2
	targetReport.Summary.MTPDraftCalls = 2
	targetReport.SpeculativeAssistantLayout = &mlx.SpeculativeAssistantLayout{
		Architecture:      "gemma4_assistant",
		OrderedEmbeddings: true,
	}
	writeProductionMTPCompareReport(t, targetPath, targetReport)
	writeProductionMTPCompareReport(t, mtpPath, productionMTPCompareTestReport(true))
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"production-mtp-compare", "-json", "-turns", "10", "-greedy-match", "-draft-token-sweeps", "1,2,4", targetPath, mtpPath}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0 for an auditable rejection report; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	for _, want := range []string{
		`"enable_by_default": false`,
		`"target_only_has_mtp_metrics"`,
		`"quality flags must be empty before MTP promotion"`,
	} {
		if !core.Contains(stdout.String(), want) {
			t.Fatalf("stdout = %q, want %s", stdout.String(), want)
		}
	}
}

func TestRunCommand_ProductionMTPCompareRejectsLoadPolicyMismatch_Bad(t *testing.T) {
	dir := t.TempDir()
	targetPath := core.PathJoin(dir, "target.json")
	mtpPath := core.PathJoin(dir, "mtp.json")
	targetReport := productionMTPCompareTestReport(false)
	mtpReport := productionMTPCompareTestReport(true)
	targetReport.Load = productionMTPCompareTestLoadPolicy(memory.KVCacheModePaged)
	mtpReport.Load = productionMTPCompareTestLoadPolicy(memory.KVCacheModePaged)
	mtpReport.Load.CacheMode = string(memory.KVCacheModeTurboQuant)
	writeProductionMTPCompareReport(t, targetPath, targetReport)
	writeProductionMTPCompareReport(t, mtpPath, mtpReport)
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"production-mtp-compare", "-json", "-turns", "10", "-greedy-match", "-draft-token-sweeps", "1,2,4", targetPath, mtpPath}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0 for an auditable rejection report; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	for _, want := range []string{
		`"same_load_policy": false`,
		`"load_policy_mismatch"`,
		`"cache_mode": "paged"`,
		`"cache_mode": "turboquant"`,
		`"enable_by_default": false`,
	} {
		if !core.Contains(stdout.String(), want) {
			t.Fatalf("stdout = %q, want %s", stdout.String(), want)
		}
	}
}

func TestRunCommand_ProductionMTPCompareRejectsMissingThroughputAndCounters_Bad(t *testing.T) {
	dir := t.TempDir()
	targetPath := core.PathJoin(dir, "target.json")
	mtpPath := core.PathJoin(dir, "mtp.json")
	mtpReport := productionMTPCompareTestReport(true)
	mtpReport.Summary.MTPTargetTokensPerSecAverage = 0
	mtpReport.Summary.MTPWarmDecodeTokensPerSecAverage = 0
	mtpReport.Summary.MTPProposedTokens = 0
	mtpReport.Summary.MTPAcceptedTokens = 0
	mtpReport.Summary.MTPRejectedTokens = 0
	mtpReport.Summary.MTPTargetVerifyCalls = 0
	mtpReport.Summary.MTPDraftCalls = 0
	writeProductionMTPCompareReport(t, targetPath, productionMTPCompareTestReport(false))
	writeProductionMTPCompareReport(t, mtpPath, mtpReport)
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"production-mtp-compare", "-json", "-turns", "10", "-greedy-match", targetPath, mtpPath}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0 for an auditable rejection report; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	for _, want := range []string{
		`"enable_by_default": false`,
		`"mtp_target_throughput_missing"`,
		`"mtp_warm_decode_missing"`,
		`"mtp_proposed_tokens_missing"`,
		`"mtp_target_verify_calls_missing"`,
		`"mtp_draft_calls_missing"`,
	} {
		if !core.Contains(stdout.String(), want) {
			t.Fatalf("stdout = %q, want %s", stdout.String(), want)
		}
	}
}

func TestRunCommand_ProductionMTPCompareRejectsMissingVisibleThroughput_Bad(t *testing.T) {
	dir := t.TempDir()
	targetPath := core.PathJoin(dir, "target.json")
	mtpPath := core.PathJoin(dir, "mtp.json")
	targetReport := productionMTPCompareTestReport(false)
	targetReport.Summary.DecodeTokensPerSecAverage = 0
	mtpReport := productionMTPCompareTestReport(true)
	mtpReport.Summary.DecodeTokensPerSecAverage = 0
	mtpReport.Summary.MTPVisibleTokensPerSecAverage = 0
	writeProductionMTPCompareReport(t, targetPath, targetReport)
	writeProductionMTPCompareReport(t, mtpPath, mtpReport)
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"production-mtp-compare", "-json", "-turns", "10", "-greedy-match", targetPath, mtpPath}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0 for an auditable rejection report; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	for _, want := range []string{
		`"enable_by_default": false`,
		`"target_only_visible_throughput_missing"`,
		`"mtp_visible_throughput_missing"`,
	} {
		if !core.Contains(stdout.String(), want) {
			t.Fatalf("stdout = %q, want %s", stdout.String(), want)
		}
	}
}

func TestRunCommand_ProductionMTPCompareRejectsMissingMetricEvidence_Bad(t *testing.T) {
	dir := t.TempDir()
	targetPath := core.PathJoin(dir, "target.json")
	mtpPath := core.PathJoin(dir, "mtp.json")
	targetReport := productionMTPCompareTestReport(false)
	targetReport.Summary.TotalDuration = 0
	targetReport.Summary.RestoreAvgDuration = 0
	targetReport.Summary.PeakMemoryBytes = 0
	targetReport.Summary.ActivePlusCacheMemoryBytes = 0
	targetReport.EstimatedEnergy = nil
	mtpReport := productionMTPCompareTestReport(true)
	mtpReport.Summary.TotalDuration = 0
	mtpReport.Summary.RestoreAvgDuration = 0
	mtpReport.Summary.PeakMemoryBytes = 0
	mtpReport.Summary.ActivePlusCacheMemoryBytes = 0
	mtpReport.EstimatedEnergy = nil
	writeProductionMTPCompareReport(t, targetPath, targetReport)
	writeProductionMTPCompareReport(t, mtpPath, mtpReport)
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"production-mtp-compare", "-json", "-turns", "10", "-greedy-match", targetPath, mtpPath}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0 for an auditable rejection report; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	for _, want := range []string{
		`"enable_by_default": false`,
		`"target_only_wall_duration_missing"`,
		`"mtp_wall_duration_missing"`,
		`"target_only_restore_duration_missing"`,
		`"mtp_restore_duration_missing"`,
		`"target_only_peak_memory_missing"`,
		`"mtp_peak_memory_missing"`,
		`"target_only_active_plus_cache_memory_missing"`,
		`"mtp_active_plus_cache_memory_missing"`,
		`"target_only_energy_missing"`,
		`"mtp_energy_missing"`,
		`"estimated_power_watts_missing"`,
	} {
		if !core.Contains(stdout.String(), want) {
			t.Fatalf("stdout = %q, want %s", stdout.String(), want)
		}
	}
}

func TestRunCommand_ProductionMTPCompareRejectsMissingDraftTokenSweepEvidence_Bad(t *testing.T) {
	dir := t.TempDir()
	targetPath := core.PathJoin(dir, "target.json")
	mtpPath := core.PathJoin(dir, "mtp.json")
	mtpReport := productionMTPCompareTestReport(true)
	mtpReport.Runs[0].Metrics.MTP.DraftTokenSchedule = []int{2, 2}
	writeProductionMTPCompareReport(t, targetPath, productionMTPCompareTestReport(false))
	writeProductionMTPCompareReport(t, mtpPath, mtpReport)
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"production-mtp-compare", "-json", "-turns", "10", "-greedy-match", targetPath, mtpPath}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0 for an auditable rejection report; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	for _, want := range []string{
		`"enable_by_default": false`,
		`"mtp_observed_draft_token_sweeps": [`,
		`"mtp_draft_token_sweep_missing_1"`,
		`"mtp_draft_token_sweep_missing_4"`,
	} {
		if !core.Contains(stdout.String(), want) {
			t.Fatalf("stdout = %q, want %s", stdout.String(), want)
		}
	}
}

func TestRunCommand_ProductionMTPCompareRejectsDeclaredUnobservedDraftSweeps_Bad(t *testing.T) {
	dir := t.TempDir()
	targetPath := core.PathJoin(dir, "target.json")
	mtpPath := core.PathJoin(dir, "mtp.json")
	pairPath := core.PathJoin(dir, "pair.json")
	mtpReport := productionMTPCompareTestReport(true)
	mtpReport.Runs[0].Metrics.MTP.DraftTokenSchedule = []int{2, 2}
	writeProductionMTPCompareReport(t, targetPath, productionMTPCompareTestReport(false))
	writeProductionMTPCompareReport(t, mtpPath, mtpReport)
	writeProductionMTPPairReport(t, pairPath, productionMTPCompareTestPairReport(true))
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"production-mtp-compare", "-json", "-turns", "10", "-greedy-match", "-draft-token-sweeps", "1,2,4", "-official-pair-report", pairPath, targetPath, mtpPath}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0 for an auditable rejection report; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	for _, want := range []string{
		`"enable_by_default": false`,
		`"mtp_declared_draft_token_sweep_unobserved_1"`,
		`"mtp_declared_draft_token_sweep_unobserved_4"`,
		`"reason": "quality flags must be empty before MTP promotion"`,
	} {
		if !core.Contains(stdout.String(), want) {
			t.Fatalf("stdout = %q, want %s", stdout.String(), want)
		}
	}
}

func TestRunCommand_ProductionMTPCompareRejectsDraftAccountingMismatch_Bad(t *testing.T) {
	dir := t.TempDir()
	targetPath := core.PathJoin(dir, "target.json")
	mtpPath := core.PathJoin(dir, "mtp.json")
	mtpReport := productionMTPCompareTestReport(true)
	mtpReport.Summary.MTPProposedTokens = 40
	mtpReport.Summary.MTPAcceptedTokens = 0
	mtpReport.Summary.MTPRejectedTokens = 39
	writeProductionMTPCompareReport(t, targetPath, productionMTPCompareTestReport(false))
	writeProductionMTPCompareReport(t, mtpPath, mtpReport)
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"production-mtp-compare", "-json", "-turns", "10", "-greedy-match", targetPath, mtpPath}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0 for an auditable rejection report; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	for _, want := range []string{
		`"enable_by_default": false`,
		`"mtp_draft_accounting_mismatch"`,
		`"mtp_accepted_tokens_missing"`,
	} {
		if !core.Contains(stdout.String(), want) {
			t.Fatalf("stdout = %q, want %s", stdout.String(), want)
		}
	}
}

func productionMTPCompareAssistantEvidenceArgs() []string {
	return []string{
		"-assistant-architecture", mlx.OfficialGemma4E2BAssistantLock().ModelType,
		"-assistant-ordered-embeddings",
		"-assistant-centroids", "2048",
		"-assistant-centroid-top-k", "32",
		"-assistant-four-layer-drafter",
		"-assistant-token-ordering-dtype", "int64",
		"-assistant-token-ordering-shape", "2048,128",
	}
}

func productionMTPCompareAssistantEvidenceInput() productionMTPAssistantEvidenceInput {
	return productionMTPAssistantEvidenceInput{
		Architecture:              mlx.OfficialGemma4E2BAssistantLock().ModelType,
		OrderedEmbeddings:         true,
		Centroids:                 2048,
		CentroidIntermediateTopK:  32,
		FourLayerDrafter:          true,
		TokenOrderingDType:        "int64",
		TokenOrderingShape:        []int{2048, 128},
		OfficialPairVerified:      true,
		OfficialTargetModelID:     mlx.OfficialGemma4E2BTargetLock().ModelID,
		OfficialTargetRevision:    mlx.OfficialGemma4E2BTargetLock().Revision,
		OfficialAssistantModelID:  mlx.OfficialGemma4E2BAssistantLock().ModelID,
		OfficialAssistantRevision: mlx.OfficialGemma4E2BAssistantLock().Revision,
	}
}

func productionMTPCompareTestPairReport(ok bool) mlx.OfficialGemma4E2BPairReport {
	report := mlx.OfficialGemma4E2BPairReport{
		PairOK:                             ok,
		AssistantAttachable:                ok,
		AssistantOrderedEmbeddings:         true,
		AssistantNumCentroids:              2048,
		AssistantCentroidIntermediateTopK:  32,
		AssistantProjectionTensorsOK:       ok,
		AssistantOrderedEmbeddingTensorsOK: ok,
		AssistantTokenOrderingDType:        "I64",
		AssistantTokenOrderingShape:        []int{262144},
		AssistantLayerTypesCoveredByTarget: ok,
		AssistantFourLayerDrafter:          true,
		SameVocabSize:                      true,
		SameContextLength:                  true,
		AssistantBackboneMatchesTarget:     true,
		AssistantMissingTensorNames:        nil,
		AssistantInvalidTensorShapes:       nil,
		Target: mlx.OfficialGemma4E2BSnapshotReport{
			Role:     mlx.OfficialGemma4E2BRoleTarget,
			ModelID:  mlx.OfficialGemma4E2BTargetLock().ModelID,
			Revision: mlx.OfficialGemma4E2BTargetLock().Revision,
			Verified: ok,
			Lock:     mlx.OfficialGemma4E2BTargetLock(),
		},
		Assistant: mlx.OfficialGemma4E2BSnapshotReport{
			Role:     mlx.OfficialGemma4E2BRoleAssistant,
			ModelID:  mlx.OfficialGemma4E2BAssistantLock().ModelID,
			Revision: mlx.OfficialGemma4E2BAssistantLock().Revision,
			Verified: ok,
			Lock:     mlx.OfficialGemma4E2BAssistantLock(),
		},
	}
	if !ok {
		report.AssistantMissingTensorNames = []string{"masked_embedding.token_ordering"}
	}
	return report
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
			RestoreAvgDuration:         100 * time.Millisecond,
			PrefillTokensPerSecAverage: 2000,
			PeakMemoryBytes:            4096,
			ActiveMemoryBytes:          2048,
			CacheMemoryBytes:           512,
			ActivePlusCacheMemoryBytes: 2560,
		},
		EstimatedEnergy: &driverProfileEnergy{
			Method:      "test",
			PowerWatts:  100,
			TotalJoules: 1000,
		},
		Load: productionMTPCompareTestLoadPolicy(memory.KVCacheModePaged),
	}
	if mtp {
		report.SpeculativeDraftModelPath = "/models/gemma4-e2b-assistant"
		report.SpeculativeDraftTokens = 2
		report.Summary.TotalDuration = 8 * time.Second
		report.Summary.RestoreAvgDuration = 80 * time.Millisecond
		report.Summary.DecodeTokensPerSecAverage = 120
		report.Summary.PeakMemoryBytes = 3584
		report.Summary.ActiveMemoryBytes = 1792
		report.Summary.CacheMemoryBytes = 512
		report.Summary.ActivePlusCacheMemoryBytes = 2304
		report.Summary.MTPVisibleTokensPerSecAverage = 125
		report.Summary.MTPTargetTokensPerSecAverage = 110
		report.Summary.MTPWarmDecodeTokensPerSecAverage = 123
		report.Summary.MTPProposedTokens = 40
		report.Summary.MTPAcceptedTokens = 30
		report.Summary.MTPRejectedTokens = 10
		report.Summary.MTPTargetVerifyCalls = 20
		report.Summary.MTPDraftCalls = 20
		report.Summary.MTPAcceptanceRateAverage = 0.75
		report.EstimatedEnergy.TotalJoules = 760
		report.Runs = []driverProfileRun{
			{
				Metrics: mlx.Metrics{
					MTP: &mlx.MTPMetrics{
						DraftTokenSchedule: []int{1, 2, 4},
					},
				},
			},
		}
	}
	return report
}

func productionMTPCompareTestLoadPolicy(mode memory.KVCacheMode) *tuneProfileLoadSettings {
	return &tuneProfileLoadSettings{
		ContextLength:        mlx.ProductionLaneLongContextLength,
		PromptCache:          true,
		PromptCacheMinTokens: 512,
		CachePolicy:          string(memory.KVCacheFull),
		CacheMode:            string(mode),
		BatchSize:            1,
		PrefillChunkSize:     mlx.ProductionLaneLongContextPrefillChunkSize,
	}
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

func writeProductionMTPPairReport(t *testing.T, path string, report mlx.OfficialGemma4E2BPairReport) {
	t.Helper()
	data := core.JSONMarshalIndent(report, "", "  ")
	if !data.OK {
		t.Fatalf("JSONMarshalIndent(%s): %v", path, data.Value)
	}
	if result := core.WriteFile(path, data.Value.([]byte), 0o644); !result.OK {
		t.Fatalf("WriteFile(%s): %v", path, result.Value)
	}
}
