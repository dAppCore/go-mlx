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
		`"native_architectures": 25`,
		`"metadata_only_architectures": 0`,
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
		"native gaps: none",
	} {
		if !core.Contains(stdout.String(), want) {
			t.Fatalf("stdout = %q, want %q", stdout.String(), want)
		}
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
