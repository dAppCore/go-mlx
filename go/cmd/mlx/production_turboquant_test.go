// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"testing"
	"time"

	core "dappco.re/go"
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
		`"enabled_by_default": false`,
		`"requires_explicit_opt_in": true`,
		`"requires_normal_context_validation": true`,
		`"requires_stress_context_validation": true`,
		`"compare_against_cache_modes": [`,
		`"fp16"`,
		`"paged"`,
		`"q8"`,
		`"k-q8-v-q4"`,
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
		`"baseline_cache_mode": "paged"`,
		`"candidate_cache_mode": "turboquant"`,
		`"compared_cache_modes": [`,
		`"fp16"`,
		`"paged"`,
		`"q8"`,
		`"k-q8-v-q4"`,
		`"normal_context_validated": true`,
		`"stress_context_validated": true`,
		`"baseline_energy_joules": 1000`,
		`"candidate_energy_joules": 700`,
		`"estimated_power_watts": 100`,
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
			ContextLength: 32768,
			PromptCache:   true,
			CacheMode:     string(mode),
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
