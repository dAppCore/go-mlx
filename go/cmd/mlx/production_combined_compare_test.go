// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/memory"
)

func TestRunCommand_ProductionCombinedMTPAndTurboQuantCompareJSON_Good(t *testing.T) {
	dir := t.TempDir()
	mtpPath := core.PathJoin(dir, "mtp-compare.json")
	turboPath := core.PathJoin(dir, "turboquant-compare.json")
	writeProductionCombinedReport(t, mtpPath, productionCombinedMTPCompareTestReport(memory.KVCacheModeTurboQuant))
	writeProductionCombinedReport(t, turboPath, productionCombinedTurboQuantCompareTestReport())
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"production-mtp-turboquant-compare", "-json", mtpPath, turboPath}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	for _, want := range []string{
		`"command": "production-mtp-turboquant-compare"`,
		`"mode": "mtp+turboquant-kv"`,
		`"cache_mode": "turboquant"`,
		`"mtp_eligible": true`,
		`"turboquant_eligible": true`,
		`"production_candidate": true`,
		`"enable_by_default": false`,
		`"reason": "combined MTP+TurboQuant retained workflow passes both lanes and remains explicit opt-in"`,
		`"mtp_acceptance_rate": 0.75`,
		`"turboquant_memory_savings_ratio": 0.4`,
	} {
		if !core.Contains(stdout.String(), want) {
			t.Fatalf("stdout = %q, want %s", stdout.String(), want)
		}
	}
	if stderr.Len() != 0 {
		t.Fatalf("stderr = %q, want empty", stderr.String())
	}
}

func TestRunCommand_ProductionCombinedMTPAndTurboQuantCompareRejectsPagedMTP_Bad(t *testing.T) {
	dir := t.TempDir()
	mtpPath := core.PathJoin(dir, "mtp-compare.json")
	turboPath := core.PathJoin(dir, "turboquant-compare.json")
	writeProductionCombinedReport(t, mtpPath, productionCombinedMTPCompareTestReport(memory.KVCacheModePaged))
	writeProductionCombinedReport(t, turboPath, productionCombinedTurboQuantCompareTestReport())
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"production-mtp-turboquant-compare", "-json", mtpPath, turboPath}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0 for an auditable rejection report; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	for _, want := range []string{
		`"production_candidate": false`,
		`"combined MTP benchmark must run target-only and MTP with TurboQuant cache mode"`,
		`"target_only_cache_mode": "paged"`,
		`"mtp_cache_mode": "paged"`,
	} {
		if !core.Contains(stdout.String(), want) {
			t.Fatalf("stdout = %q, want %s", stdout.String(), want)
		}
	}
	if stderr.Len() != 0 {
		t.Fatalf("stderr = %q, want empty", stderr.String())
	}
}

func TestRunCommand_ProductionCombinedMTPAndTurboQuantCompareRejectsMissingArgs_Bad(t *testing.T) {
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"production-mtp-turboquant-compare"}, stdout, stderr)

	if code != 2 {
		t.Fatalf("exit code = %d, want 2; stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	if !core.Contains(stderr.String(), "expected MTP and TurboQuant comparison JSON paths") {
		t.Fatalf("stderr = %q, want positional argument error", stderr.String())
	}
}

func productionCombinedMTPCompareTestReport(cacheMode memory.KVCacheMode) productionMTPCompareReport {
	target := productionMTPCompareTestReport(false)
	mtp := productionMTPCompareTestReport(true)
	target.Load = productionMTPCompareTestLoadPolicy(cacheMode)
	mtp.Load = productionMTPCompareTestLoadPolicy(cacheMode)
	return newProductionMTPCompareReport("target.json", target, "mtp.json", mtp, 10, true, "", []int{1, 2, 4}, productionMTPCompareAssistantEvidenceInput(), 0)
}

func productionCombinedTurboQuantCompareTestReport() productionTurboQuantCompareReport {
	return newProductionTurboQuantCompareReport([]productionTurboQuantCompareDriverReport{
		{
			Path:   "paged.json",
			Mode:   memory.KVCacheModePaged,
			Report: productionTurboQuantCompareTestReport(memory.KVCacheModePaged),
		},
		{
			Path:   "turboquant.json",
			Mode:   memory.KVCacheModeTurboQuant,
			Report: productionTurboQuantCompareTestReport(memory.KVCacheModeTurboQuant),
		},
		{
			Path:   "fp16.json",
			Mode:   memory.KVCacheModeFP16,
			Report: productionTurboQuantCompareTestReport(memory.KVCacheModeFP16),
		},
		{
			Path:   "q8.json",
			Mode:   memory.KVCacheModeQ8,
			Report: productionTurboQuantCompareTestReport(memory.KVCacheModeQ8),
		},
		{
			Path:   "k-q8-v-q4.json",
			Mode:   memory.KVCacheModeKQ8VQ4,
			Report: productionTurboQuantCompareTestReport(memory.KVCacheModeKQ8VQ4),
		},
	}, 10, true, "", true, true, 0)
}

func writeProductionCombinedReport(t *testing.T, path string, report any) {
	t.Helper()
	data := core.JSONMarshalIndent(report, "", "  ")
	if !data.OK {
		t.Fatalf("JSONMarshalIndent(%s): %v", path, data.Value)
	}
	if result := core.WriteFile(path, data.Value.([]byte), 0o644); !result.OK {
		t.Fatalf("WriteFile(%s): %v", path, result.Value)
	}
}
