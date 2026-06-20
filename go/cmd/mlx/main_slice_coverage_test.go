// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"strings"
	"testing"

	core "dappco.re/go"
)

// runCommand with no args and NOT inside an .app bundle prints the top-level
// usage and exits 0 (the CLI-flat default branch).
func TestRunCommand_NoArgs_PrintsUsage_Good(t *testing.T) {
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), nil, stdout, stderr)
	if code != 0 {
		t.Fatalf("exit = %d, want 0 (no-args usage)", code)
	}
	if !strings.Contains(stdout.String(), "Usage:") || !strings.Contains(stdout.String(), "Run inference") {
		t.Fatalf("stdout = %q, want the top-level usage", stdout.String())
	}
}

// slice (human-readable, no -json) prints the slice-plan summary and writes the
// materialised slice.
func TestRunSlice_HumanSummary_Good(t *testing.T) {
	source := writeCLISlicePack(t)
	output := core.PathJoin(t.TempDir(), "client-slice")
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{"slice", "-preset", "client", "-output", output, source}, stdout, stderr)
	if code != 0 {
		t.Fatalf("exit = %d, want 0; stderr=%q", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "model slice:") || !strings.Contains(stdout.String(), "components:") {
		t.Fatalf("stdout = %q, want the slice summary", stdout.String())
	}
}

// slice with no -output is a usage error (exit 2).
func TestRunSlice_MissingOutput_Bad(t *testing.T) {
	source := writeCLISlicePack(t)
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{"slice", source}, stdout, stderr)
	if code != 2 {
		t.Fatalf("exit = %d, want 2 (missing -output)", code)
	}
	if !strings.Contains(stderr.String(), "-output is required") {
		t.Fatalf("stderr = %q, want the output-required notice", stderr.String())
	}
}

// slice with no model path is a usage error (exit 2).
func TestRunSlice_NoModelArg_Bad(t *testing.T) {
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{"slice", "-output", "/tmp/out"}, stdout, stderr)
	if code != 2 {
		t.Fatalf("exit = %d, want 2 (no model path)", code)
	}
	if !strings.Contains(stderr.String(), "exactly one model path") {
		t.Fatalf("stderr = %q, want the one-path notice", stderr.String())
	}
}

// slice with a bad flag fails to parse (exit 2).
func TestRunSlice_BadFlag_Bad(t *testing.T) {
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{"slice", "-not-a-flag"}, stdout, stderr)
	if code != 2 {
		t.Fatalf("exit = %d, want 2 (bad flag)", code)
	}
}

// slice against a nonexistent model path reaches the SliceModel error path
// (exit 1).
func TestRunSlice_SliceError_Bad(t *testing.T) {
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{
		"slice", "-output", core.PathJoin(t.TempDir(), "out"), core.PathJoin(t.TempDir(), "absent-model"),
	}, stdout, stderr)
	if code != 1 {
		t.Fatalf("exit = %d, want 1 (slice error); stderr=%q", code, stderr.String())
	}
	if !strings.Contains(stderr.String(), "slice:") {
		t.Fatalf("stderr = %q, want the slice error", stderr.String())
	}
}

// runCPUFFNMemoryEstimate (the default estimator var) surfaces an error for a
// nonexistent model path — the error-return branch of the package-level seam.
func TestRunCPUFFNMemoryEstimate_Error_Bad(t *testing.T) {
	_, err := runCPUFFNMemoryEstimate(context.Background(), core.PathJoin(t.TempDir(), "absent"), 0)
	if err == nil {
		t.Fatal("runCPUFFNMemoryEstimate(absent) error = nil, want an estimate error")
	}
}
