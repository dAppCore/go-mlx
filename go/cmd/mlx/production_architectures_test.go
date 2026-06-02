// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"testing"

	core "dappco.re/go"
)

func TestRunCommand_ProductionArchitecturesJSON_Good(t *testing.T) {
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"production-architectures", "-json"}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	for _, want := range []string{
		`"total_architectures": 25`,
		`"native_architectures": 20`,
		`"metadata_only_architectures": 5`,
		`"remove_python_fallback_ready": false`,
		`"id": "deepseek"`,
		`"missing_native": "MoE router plus MLA attention variants"`,
		`"command": "production-architectures"`,
	} {
		if !core.Contains(stdout.String(), want) {
			t.Fatalf("stdout = %q, want %s", stdout.String(), want)
		}
	}
	if stderr.Len() != 0 {
		t.Fatalf("stderr = %q, want empty", stderr.String())
	}
}

func TestRunCommand_ProductionArchitecturesGapsOnly_Good(t *testing.T) {
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"production-architectures", "-gaps-only"}, stdout, stderr)

	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	for _, want := range []string{
		"native gaps:",
		"gpt_oss: MoE router plus channel parser validation [moe]",
		"next: channel_parser_validation, sparse_expert_router, native_load_generate_smoke",
	} {
		if !core.Contains(stdout.String(), want) {
			t.Fatalf("stdout = %q, want %q", stdout.String(), want)
		}
	}
	if core.Contains(stdout.String(), "production architectures:") {
		t.Fatalf("stdout = %q, gaps-only should omit summary header", stdout.String())
	}
	if core.Contains(stdout.String(), "bert:") || core.Contains(stdout.String(), "bert_rerank:") || core.Contains(stdout.String(), "qwen3_6:") || core.Contains(stdout.String(), "qwen3_moe:") {
		t.Fatalf("stdout = %q, staged native loaders should not remain metadata-only gaps", stdout.String())
	}
	if stderr.Len() != 0 {
		t.Fatalf("stderr = %q, want empty", stderr.String())
	}
}

func TestRunCommand_ProductionArchitecturesBadArgs(t *testing.T) {
	stdout, stderr := core.NewBuffer(), core.NewBuffer()

	code := runCommand(context.Background(), []string{"production-architectures", "extra"}, stdout, stderr)

	if code != 2 {
		t.Fatalf("exit code = %d, want 2; stderr=%q stdout=%q", code, stderr.String(), stdout.String())
	}
	if !core.Contains(stderr.String(), "expected no positional arguments") {
		t.Fatalf("stderr = %q, want argument error", stderr.String())
	}
	if stdout.Len() != 0 {
		t.Fatalf("stdout = %q, want empty", stdout.String())
	}
}
