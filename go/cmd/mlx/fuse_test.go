// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"bytes"
	"context"
	"testing"
)

// TestRunFuseCommand_MissingFlags_Bad: the three dir flags are required.
func TestRunFuseCommand_MissingFlags_Bad(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runFuseCommand(context.Background(), nil, &stdout, &stderr); code != 2 {
		t.Errorf("missing-flags exit = %d, want 2", code)
	}
	if code := runFuseCommand(context.Background(), []string{"-base", "x"}, &stdout, &stderr); code != 2 {
		t.Errorf("partial-flags exit = %d, want 2", code)
	}
}

// TestRunFuseCommand_NoModel_Bad: an empty base dir (no safetensors) is a clean
// non-zero exit, not a panic — the verb surfaces FuseModelDir's error.
func TestRunFuseCommand_NoModel_Bad(t *testing.T) {
	var stdout, stderr bytes.Buffer
	args := []string{"-base", t.TempDir(), "-adapter", t.TempDir(), "-out", t.TempDir()}
	if code := runFuseCommand(context.Background(), args, &stdout, &stderr); code != 1 {
		t.Errorf("no-model exit = %d, want 1", code)
	}
}
