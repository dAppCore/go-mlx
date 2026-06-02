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
		`"native_architectures": 16`,
		`"metadata_only_architectures": 9`,
		`"remove_python_fallback_ready": false`,
		`"id": "qwen3_6"`,
		`"missing_native": "hybrid linear attention"`,
		`"id": "deepseek"`,
		`"missing_native": "MoE router plus MLA attention variants"`,
		`"id": "bert_rerank"`,
		`"missing_native": "rerank scorer"`,
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
		"qwen3_6: hybrid linear attention",
		"qwen3_moe: sparse expert router [moe]",
		"bert: embedding encoder [embeddings]",
		"bert_rerank: rerank scorer [rerank]",
		"next: cross_encoder_loader, score_head_output, no_generation_kv_smoke",
	} {
		if !core.Contains(stdout.String(), want) {
			t.Fatalf("stdout = %q, want %q", stdout.String(), want)
		}
	}
	if core.Contains(stdout.String(), "production architectures:") {
		t.Fatalf("stdout = %q, gaps-only should omit summary header", stdout.String())
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
