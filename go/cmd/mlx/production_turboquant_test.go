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

func TestRunCommand_ProductionTurboQuantPolicyJSON_Good(t *testing.T) {
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"production-turboquant", "-json"}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	for _, want := range []string{
		`"kind": "production-turboquant-policy"`,
		`"cache_mode": "turboquant"`,
		`"target_effective_bits_milli": 3500`,
		`"required_layout_version": "turboquant-kv-v1"`,
		`"required_key_algorithm": "turboquantprod"`,
		`"required_value_algorithm": "turboquantmse"`,
		`"required_outlier_policy": "high-half-head-dim-v1"`,
		`"requires_qjl_residual": true`,
		`"requires_metadata_accounting": true`,
		`"enabled_by_default": false`,
		`"requires_explicit_opt_in": true`,
		`"requires_normal_context_validation": true`,
		`"requires_stress_context_validation": true`,
		`"compare_against_cache_modes": [`,
		`"fp16"`,
		`"paged"`,
		`"q8"`,
		`"k-q8-v-q4"`,
		`"candidate_active_plus_cache_memory_bytes"`,
		`"baseline_active_plus_cache_memory_bytes"`,
		`"candidate_layout_version"`,
		`"candidate_qjl_residual"`,
		`"candidate_metadata_bytes"`,
		`"estimated_power_watts"`,
		`"quality_flags"`,
	} {
		if !core.Contains(stdout.String(), want) {
			t.Fatalf("stdout = %q, want %s", stdout.String(), want)
		}
	}
	if stderr.Len() != 0 {
		t.Fatalf("stderr = %q, want empty", stderr.String())
	}
}

func TestRunCommand_ProductionTurboQuantCompareJSON_Good(t *testing.T) {
	dir := t.TempDir()
	baselinePath := core.PathJoin(dir, "paged.json")
	candidatePath := core.PathJoin(dir, "turboquant.json")
	fp16Path := core.PathJoin(dir, "fp16.json")
	q8Path := core.PathJoin(dir, "q8.json")
	kq8vq4Path := core.PathJoin(dir, "k-q8-v-q4.json")
	writeProductionMTPCompareReport(t, baselinePath, productionTurboQuantCompareTestReport(memory.KVCacheModePaged))
	writeProductionMTPCompareReport(t, candidatePath, productionTurboQuantCompareTestReport(memory.KVCacheModeTurboQuant))
	writeProductionMTPCompareReport(t, fp16Path, productionTurboQuantCompareTestReport(memory.KVCacheModeFP16))
	writeProductionMTPCompareReport(t, q8Path, productionTurboQuantCompareTestReport(memory.KVCacheModeQ8))
	writeProductionMTPCompareReport(t, kq8vq4Path, productionTurboQuantCompareTestReport(memory.KVCacheModeKQ8VQ4))
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{
		"production-turboquant-compare",
		"-json",
		"-turns", "10",
		"-quality-match",
		"-normal-context",
		"-stress-context",
		"-candidate-layout-version", mlx.ProductionTurboQuantKVLayoutVersion,
		"-candidate-key-algorithm", mlx.ProductionTurboQuantKeyAlgorithm,
		"-candidate-value-algorithm", mlx.ProductionTurboQuantValueAlgorithm,
		"-candidate-outlier-policy", mlx.ProductionTurboQuantOutlierPolicy,
		"-candidate-effective-bits-milli", "3500",
		"-candidate-qjl-residual",
		"-candidate-metadata-bytes", "65536",
		baselinePath,
		candidatePath,
		fp16Path,
		q8Path,
		kq8vq4Path,
	}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	for _, want := range []string{
		`"command": "production-turboquant-compare"`,
		`"same_load_policy": true`,
		`"baseline_cache_mode": "paged"`,
		`"candidate_cache_mode": "turboquant"`,
		`"candidate_layout_version": "turboquant-kv-v1"`,
		`"candidate_key_algorithm": "turboquantprod"`,
		`"candidate_value_algorithm": "turboquantmse"`,
		`"candidate_outlier_policy": "high-half-head-dim-v1"`,
		`"candidate_effective_bits_milli": 3500`,
		`"candidate_qjl_residual": true`,
		`"candidate_metadata_bytes": 65536`,
		`"cache_policy": "full"`,
		`"compared_cache_modes": [`,
		`"fp16"`,
		`"paged"`,
		`"q8"`,
		`"k-q8-v-q4"`,
		`"normal_context_validated": true`,
		`"stress_context_validated": true`,
		`"baseline_input_output_tokens_per_sec": 33268`,
		`"candidate_input_output_tokens_per_sec": 47525`,
		`"baseline_energy_joules": 1000`,
		`"candidate_energy_joules": 700`,
		`"baseline_active_plus_cache_memory_bytes": 7500`,
		`"candidate_active_plus_cache_memory_bytes": 4500`,
		`"estimated_power_watts": 100`,
		`"baseline_cache_policy": "full"`,
		`"candidate_cache_policy": "full"`,
		`"baseline_context_length": 32768`,
		`"candidate_context_length": 32768`,
		`"production_candidate": true`,
		`"reason": "TurboQuant retained workflow saves memory/energy with quality parity"`,
	} {
		if !core.Contains(stdout.String(), want) {
			t.Fatalf("stdout = %q, want %s", stdout.String(), want)
		}
	}
	if stderr.Len() != 0 {
		t.Fatalf("stderr = %q, want empty", stderr.String())
	}
}

func TestRunCommand_ProductionTurboQuantCompareUsesCandidateReportPayloadBytes_Good(t *testing.T) {
	dir := t.TempDir()
	baselinePath := core.PathJoin(dir, "paged.json")
	candidatePath := core.PathJoin(dir, "turboquant.json")
	fp16Path := core.PathJoin(dir, "fp16.json")
	q8Path := core.PathJoin(dir, "q8.json")
	kq8vq4Path := core.PathJoin(dir, "k-q8-v-q4.json")
	candidateReport := productionTurboQuantCompareTestReport(memory.KVCacheModeTurboQuant)
	candidateReport.Summary.TurboQuantKVPayload = &mlx.TurboQuantKVPayloadEstimate{
		Pages:                     8,
		PayloadBytes:              48000,
		PaddedPayloadBytes:        49152,
		FP16BaselineBytes:         131072,
		PaddedPayloadSavingsRatio: 0.625,
	}
	writeProductionMTPCompareReport(t, baselinePath, productionTurboQuantCompareTestReport(memory.KVCacheModePaged))
	writeProductionMTPCompareReport(t, candidatePath, candidateReport)
	writeProductionMTPCompareReport(t, fp16Path, productionTurboQuantCompareTestReport(memory.KVCacheModeFP16))
	writeProductionMTPCompareReport(t, q8Path, productionTurboQuantCompareTestReport(memory.KVCacheModeQ8))
	writeProductionMTPCompareReport(t, kq8vq4Path, productionTurboQuantCompareTestReport(memory.KVCacheModeKQ8VQ4))
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{
		"production-turboquant-compare",
		"-json",
		"-turns", "10",
		"-quality-match",
		"-normal-context",
		"-stress-context",
		"-candidate-layout-version", mlx.ProductionTurboQuantKVLayoutVersion,
		"-candidate-key-algorithm", mlx.ProductionTurboQuantKeyAlgorithm,
		"-candidate-value-algorithm", mlx.ProductionTurboQuantValueAlgorithm,
		"-candidate-outlier-policy", mlx.ProductionTurboQuantOutlierPolicy,
		"-candidate-effective-bits-milli", "3500",
		"-candidate-qjl-residual",
		baselinePath,
		candidatePath,
		fp16Path,
		q8Path,
		kq8vq4Path,
	}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	for _, want := range []string{
		`"candidate_metadata_bytes": 49152`,
		`"turboquant_kv_payload"`,
		`"padded_payload_bytes": 49152`,
		`"production_candidate": true`,
	} {
		if !core.Contains(stdout.String(), want) {
			t.Fatalf("stdout = %q, want %s", stdout.String(), want)
		}
	}
	if stderr.Len() != 0 {
		t.Fatalf("stderr = %q, want empty", stderr.String())
	}
}

func TestRunCommand_ProductionTurboQuantCompareRejectsMissingLayoutEvidence_Bad(t *testing.T) {
	dir := t.TempDir()
	baselinePath := core.PathJoin(dir, "paged.json")
	candidatePath := core.PathJoin(dir, "turboquant.json")
	fp16Path := core.PathJoin(dir, "fp16.json")
	q8Path := core.PathJoin(dir, "q8.json")
	kq8vq4Path := core.PathJoin(dir, "k-q8-v-q4.json")
	writeProductionMTPCompareReport(t, baselinePath, productionTurboQuantCompareTestReport(memory.KVCacheModePaged))
	writeProductionMTPCompareReport(t, candidatePath, productionTurboQuantCompareTestReport(memory.KVCacheModeTurboQuant))
	writeProductionMTPCompareReport(t, fp16Path, productionTurboQuantCompareTestReport(memory.KVCacheModeFP16))
	writeProductionMTPCompareReport(t, q8Path, productionTurboQuantCompareTestReport(memory.KVCacheModeQ8))
	writeProductionMTPCompareReport(t, kq8vq4Path, productionTurboQuantCompareTestReport(memory.KVCacheModeKQ8VQ4))
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{
		"production-turboquant-compare",
		"-json",
		"-turns", "10",
		"-quality-match",
		"-normal-context",
		"-stress-context",
		baselinePath,
		candidatePath,
		fp16Path,
		q8Path,
		kq8vq4Path,
	}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0 for an auditable rejection report; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	for _, want := range []string{
		`"production_candidate": false`,
		`"TurboQuant layout version evidence must match turboquant-kv-v1"`,
	} {
		if !core.Contains(stdout.String(), want) {
			t.Fatalf("stdout = %q, want %s", stdout.String(), want)
		}
	}
	if stderr.Len() != 0 {
		t.Fatalf("stderr = %q, want empty", stderr.String())
	}
}

func TestRunCommand_ProductionTurboQuantCompareRejectsMissingSideBySideModes_Bad(t *testing.T) {
	dir := t.TempDir()
	baselinePath := core.PathJoin(dir, "paged.json")
	candidatePath := core.PathJoin(dir, "turboquant.json")
	writeProductionMTPCompareReport(t, baselinePath, productionTurboQuantCompareTestReport(memory.KVCacheModePaged))
	writeProductionMTPCompareReport(t, candidatePath, productionTurboQuantCompareTestReport(memory.KVCacheModeTurboQuant))
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{
		"production-turboquant-compare",
		"-json",
		"-turns", "10",
		"-quality-match",
		"-normal-context",
		"-stress-context",
		baselinePath,
		candidatePath,
	}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0 for an auditable rejection report; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	for _, want := range []string{
		`"production_candidate": false`,
		`"TurboQuant must be compared side by side against fp16, paged, q8, and k-q8-v-q4 cache modes"`,
	} {
		if !core.Contains(stdout.String(), want) {
			t.Fatalf("stdout = %q, want %s", stdout.String(), want)
		}
	}
}

func TestRunCommand_ProductionTurboQuantCompareRejectsActiveCacheRegression_Bad(t *testing.T) {
	dir := t.TempDir()
	baselinePath := core.PathJoin(dir, "paged.json")
	candidatePath := core.PathJoin(dir, "turboquant.json")
	fp16Path := core.PathJoin(dir, "fp16.json")
	q8Path := core.PathJoin(dir, "q8.json")
	kq8vq4Path := core.PathJoin(dir, "k-q8-v-q4.json")
	baselineReport := productionTurboQuantCompareTestReport(memory.KVCacheModePaged)
	candidateReport := productionTurboQuantCompareTestReport(memory.KVCacheModeTurboQuant)
	candidateReport.Summary.ActivePlusCacheMemoryBytes = baselineReport.Summary.ActivePlusCacheMemoryBytes
	writeProductionMTPCompareReport(t, baselinePath, baselineReport)
	writeProductionMTPCompareReport(t, candidatePath, candidateReport)
	writeProductionMTPCompareReport(t, fp16Path, productionTurboQuantCompareTestReport(memory.KVCacheModeFP16))
	writeProductionMTPCompareReport(t, q8Path, productionTurboQuantCompareTestReport(memory.KVCacheModeQ8))
	writeProductionMTPCompareReport(t, kq8vq4Path, productionTurboQuantCompareTestReport(memory.KVCacheModeKQ8VQ4))
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{
		"production-turboquant-compare",
		"-json",
		"-turns", "10",
		"-quality-match",
		"-normal-context",
		"-stress-context",
		baselinePath,
		candidatePath,
		fp16Path,
		q8Path,
		kq8vq4Path,
	}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0 for an auditable rejection report; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	for _, want := range []string{
		`"production_candidate": false`,
		`active+cache memory savings`,
		`"baseline_active_plus_cache_memory_bytes": 7500`,
		`"candidate_active_plus_cache_memory_bytes": 7500`,
	} {
		if !core.Contains(stdout.String(), want) {
			t.Fatalf("stdout = %q, want %s", stdout.String(), want)
		}
	}
}

func TestRunCommand_ProductionTurboQuantCompareRejectsLoadPolicyMismatch_Bad(t *testing.T) {
	dir := t.TempDir()
	baselinePath := core.PathJoin(dir, "paged.json")
	candidatePath := core.PathJoin(dir, "turboquant.json")
	fp16Path := core.PathJoin(dir, "fp16.json")
	q8Path := core.PathJoin(dir, "q8.json")
	kq8vq4Path := core.PathJoin(dir, "k-q8-v-q4.json")
	baselineReport := productionTurboQuantCompareTestReport(memory.KVCacheModePaged)
	candidateReport := productionTurboQuantCompareTestReport(memory.KVCacheModeTurboQuant)
	candidateReport.Load.ContextLength = 65536
	writeProductionMTPCompareReport(t, baselinePath, baselineReport)
	writeProductionMTPCompareReport(t, candidatePath, candidateReport)
	writeProductionMTPCompareReport(t, fp16Path, productionTurboQuantCompareTestReport(memory.KVCacheModeFP16))
	writeProductionMTPCompareReport(t, q8Path, productionTurboQuantCompareTestReport(memory.KVCacheModeQ8))
	writeProductionMTPCompareReport(t, kq8vq4Path, productionTurboQuantCompareTestReport(memory.KVCacheModeKQ8VQ4))
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{
		"production-turboquant-compare",
		"-json",
		"-turns", "10",
		"-quality-match",
		"-normal-context",
		"-stress-context",
		baselinePath,
		candidatePath,
		fp16Path,
		q8Path,
		kq8vq4Path,
	}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0 for an auditable rejection report; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	for _, want := range []string{
		`"same_load_policy": false`,
		`"load_policy_mismatch"`,
		`"context_length": 32768`,
		`"context_length": 65536`,
		`"production_candidate": false`,
	} {
		if !core.Contains(stdout.String(), want) {
			t.Fatalf("stdout = %q, want %s", stdout.String(), want)
		}
	}
}

func TestRunCommand_ProductionTurboQuantCompareRejectsMissingMetricEvidence_Bad(t *testing.T) {
	dir := t.TempDir()
	baselinePath := core.PathJoin(dir, "paged.json")
	candidatePath := core.PathJoin(dir, "turboquant.json")
	fp16Path := core.PathJoin(dir, "fp16.json")
	q8Path := core.PathJoin(dir, "q8.json")
	kq8vq4Path := core.PathJoin(dir, "k-q8-v-q4.json")
	baselineReport := productionTurboQuantCompareTestReport(memory.KVCacheModePaged)
	baselineReport.Summary.DecodeTokensPerSecAverage = 0
	baselineReport.Summary.TotalDuration = 0
	baselineReport.Summary.RestoreAvgDuration = 0
	baselineReport.Summary.PeakMemoryBytes = 0
	baselineReport.Summary.ActivePlusCacheMemoryBytes = 0
	baselineReport.EstimatedEnergy = nil
	candidateReport := productionTurboQuantCompareTestReport(memory.KVCacheModeTurboQuant)
	candidateReport.Summary.DecodeTokensPerSecAverage = 0
	candidateReport.Summary.TotalDuration = 0
	candidateReport.Summary.RestoreAvgDuration = 0
	candidateReport.Summary.PeakMemoryBytes = 0
	candidateReport.Summary.ActivePlusCacheMemoryBytes = 0
	candidateReport.EstimatedEnergy = nil
	writeProductionMTPCompareReport(t, baselinePath, baselineReport)
	writeProductionMTPCompareReport(t, candidatePath, candidateReport)
	writeProductionMTPCompareReport(t, fp16Path, productionTurboQuantCompareTestReport(memory.KVCacheModeFP16))
	writeProductionMTPCompareReport(t, q8Path, productionTurboQuantCompareTestReport(memory.KVCacheModeQ8))
	writeProductionMTPCompareReport(t, kq8vq4Path, productionTurboQuantCompareTestReport(memory.KVCacheModeKQ8VQ4))
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{
		"production-turboquant-compare",
		"-json",
		"-turns", "10",
		"-quality-match",
		"-normal-context",
		"-stress-context",
		baselinePath,
		candidatePath,
		fp16Path,
		q8Path,
		kq8vq4Path,
	}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0 for an auditable rejection report; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	for _, want := range []string{
		`"production_candidate": false`,
		`"baseline_visible_throughput_missing"`,
		`"candidate_visible_throughput_missing"`,
		`"baseline_wall_duration_missing"`,
		`"candidate_wall_duration_missing"`,
		`"baseline_restore_duration_missing"`,
		`"candidate_restore_duration_missing"`,
		`"baseline_peak_memory_missing"`,
		`"candidate_peak_memory_missing"`,
		`"baseline_active_plus_cache_memory_missing"`,
		`"candidate_active_plus_cache_memory_missing"`,
		`"baseline_energy_missing"`,
		`"candidate_energy_missing"`,
		`"estimated_power_watts_missing"`,
	} {
		if !core.Contains(stdout.String(), want) {
			t.Fatalf("stdout = %q, want %s", stdout.String(), want)
		}
	}
}

func productionTurboQuantCompareTestReport(mode memory.KVCacheMode) driverProfileReport {
	totalDuration := 10 * time.Second
	peakMemoryBytes := uint64(10_000)
	totalJoules := 1000.0
	decodeTokensPerSec := 100.0
	restoreDuration := 100 * time.Millisecond
	if mode == memory.KVCacheModeTurboQuant {
		totalDuration = 7 * time.Second
		peakMemoryBytes = 6_000
		totalJoules = 700
		decodeTokensPerSec = 120
		restoreDuration = 70 * time.Millisecond
	}
	return driverProfileReport{
		Version:       1,
		ModelPath:     "/models/gemma4-e2b",
		PromptBytes:   32768,
		MaxTokens:     500,
		RequestedRuns: 10,
		Chat:          true,
		Load: &tuneProfileLoadSettings{
			ContextLength:        32768,
			PromptCache:          true,
			PromptCacheMinTokens: 512,
			CachePolicy:          string(memory.KVCacheFull),
			CacheMode:            string(mode),
			BatchSize:            1,
			PrefillChunkSize:     512,
		},
		Summary: driverProfileSummary{
			SuccessfulRuns:             10,
			PromptTokensAverage:        32768,
			PromptTokensMin:            32768,
			PromptTokensMax:            32768,
			VisibleTokens:              5000,
			GeneratedTokens:            5000,
			DecodeTokensPerSecAverage:  decodeTokensPerSec,
			TotalDuration:              totalDuration,
			RestoreAvgDuration:         restoreDuration,
			PrefillTokensPerSecAverage: 2000,
			PeakMemoryBytes:            peakMemoryBytes,
			ActiveMemoryBytes:          peakMemoryBytes / 2,
			CacheMemoryBytes:           peakMemoryBytes / 4,
			ActivePlusCacheMemoryBytes: peakMemoryBytes * 3 / 4,
		},
		EstimatedEnergy: &driverProfileEnergy{
			Method:                "estimated_wall_clock_seconds_times_average_active_watts",
			PowerWatts:            100,
			TotalJoules:           totalJoules,
			JoulesPerVisibleToken: totalJoules / 5000,
		},
	}
}

func productionTurboQuantCompareTestLayoutEvidence() productionTurboQuantLayoutEvidenceInput {
	policy := mlx.DefaultProductionTurboQuantPolicy()
	return productionTurboQuantLayoutEvidenceInput{
		LayoutVersion:      policy.RequiredLayoutVersion,
		KeyAlgorithm:       policy.RequiredKeyAlgorithm,
		ValueAlgorithm:     policy.RequiredValueAlgorithm,
		OutlierPolicy:      policy.RequiredOutlierPolicy,
		EffectiveBitsMilli: policy.TargetEffectiveBitsMilli,
		QJLResidual:        true,
		MetadataBytes:      64 * 1024,
	}
}
