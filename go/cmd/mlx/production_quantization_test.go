// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"testing"

	core "dappco.re/go"
	mlx "dappco.re/go/mlx"
	"dappco.re/go/mlx/memory"
)

func TestRunCommand_ProductionQuantizationDefaultJSON_Good(t *testing.T) {
	originalDeviceInfo := runGetDeviceInfo
	t.Cleanup(func() { runGetDeviceInfo = originalDeviceInfo })
	runGetDeviceInfo = func() mlx.DeviceInfo {
		return mlx.DeviceInfo{
			Architecture:                 "apple9",
			MemorySize:                   96 * memory.GiB,
			MaxRecommendedWorkingSetSize: 90 * memory.GiB,
		}
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"production-quantization", "-json", "-context", "32768"}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	for _, want := range []string{
		`"default_bits": 6`,
		`"quality_bits": 8`,
		`"constrained_bits": 4`,
		`"quantized_target_locks": [`,
		`"revision": "48ef0737faea4e72556670e49da0ba421027a545"`,
		`"revision": "40d43b05f94ee798c0e40fe19fcd9ef49928486b"`,
		`"revision": "99d9a53ff828d365a8ecae538e45f80a08d612cd"`,
		`"mtp_policy": {`,
		`"mode": "mtp"`,
		`"default_draft_tokens": 2`,
		`"enabled_by_default": false`,
		`"requires_side_by_side_benchmark": true`,
		`"bits": 6`,
		`"model_id": "mlx-community/gemma-4-e2b-it-6bit"`,
		`"reason": "default q6 tier selected"`,
	} {
		if !core.Contains(stdout.String(), want) {
			t.Fatalf("stdout = %q, want %s", stdout.String(), want)
		}
	}
	if stderr.Len() != 0 {
		t.Fatalf("stderr = %q, want empty", stderr.String())
	}
}

func TestRunCommand_ProductionQuantizationDefaultContext_Good(t *testing.T) {
	originalDeviceInfo := runGetDeviceInfo
	t.Cleanup(func() { runGetDeviceInfo = originalDeviceInfo })
	runGetDeviceInfo = func() mlx.DeviceInfo {
		return mlx.DeviceInfo{
			Architecture:                 "apple9",
			MemorySize:                   96 * memory.GiB,
			MaxRecommendedWorkingSetSize: 90 * memory.GiB,
		}
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"production-quantization", "-json"}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	for _, want := range []string{
		`"context_length": 32768`,
		`"long_context_selection": true`,
		`"bits": 6`,
		`"model_id": "mlx-community/gemma-4-e2b-it-6bit"`,
	} {
		if !core.Contains(stdout.String(), want) {
			t.Fatalf("stdout = %q, want %s", stdout.String(), want)
		}
	}
	if stderr.Len() != 0 {
		t.Fatalf("stderr = %q, want empty", stderr.String())
	}
}

func TestRunCommand_ProductionQuantizationQualityJSON_Good(t *testing.T) {
	originalDeviceInfo := runGetDeviceInfo
	t.Cleanup(func() { runGetDeviceInfo = originalDeviceInfo })
	runGetDeviceInfo = func() mlx.DeviceInfo {
		return mlx.DeviceInfo{
			Architecture:                 "apple9",
			MemorySize:                   96 * memory.GiB,
			MaxRecommendedWorkingSetSize: 90 * memory.GiB,
		}
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"production-quantization", "-json", "-quality", "-context", "32768"}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	for _, want := range []string{
		`"bits": 8`,
		`"model_id": "mlx-community/gemma-4-e2b-it-8bit"`,
		`"reason": "quality tier selected with sufficient headroom"`,
	} {
		if !core.Contains(stdout.String(), want) {
			t.Fatalf("stdout = %q, want %s", stdout.String(), want)
		}
	}
}

func TestRunCommand_ProductionQuantizationConstrainedFallback_Good(t *testing.T) {
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"production-quantization", "-json", "-memory-gib", "16", "-working-set-gib", "13", "-context", "32768"}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	for _, want := range []string{
		`"bits": 4`,
		`"model_id": "mlx-community/gemma-4-e2b-it-4bit"`,
		`"reason": "q6 does not fit requested memory/context; using q4 fallback"`,
		`"working_set_bytes": 13958643712`,
	} {
		if !core.Contains(stdout.String(), want) {
			t.Fatalf("stdout = %q, want %s", stdout.String(), want)
		}
	}
}
