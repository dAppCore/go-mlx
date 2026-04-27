// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"bytes"
	"context"
	"strings"
	"testing"
)

func TestMain_RunCommand_Good_Help(t *testing.T) {
	var stdout, stderr bytes.Buffer

	code := runCommand(context.Background(), []string{"--help"}, &stdout, &stderr)
	if code != 0 {
		t.Fatalf("exit code = %d, want 0; stderr=%q", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "Usage: violet [flags]") {
		t.Fatalf("stdout = %q, want usage", stdout.String())
	}
}

func TestMain_RunCommand_Bad_UnknownFlag(t *testing.T) {
	var stdout, stderr bytes.Buffer

	code := runCommand(context.Background(), []string{"--unknown"}, &stdout, &stderr)
	if code == 0 {
		t.Fatalf("exit code = %d, want non-zero", code)
	}
	if !strings.Contains(stderr.String(), "flag provided but not defined") {
		t.Fatalf("stderr = %q, want unknown flag error", stderr.String())
	}
}

func TestMain_RunCommand_Bad_UnexpectedArg(t *testing.T) {
	var stdout, stderr bytes.Buffer

	code := runCommand(context.Background(), []string{"serve"}, &stdout, &stderr)
	if code == 0 {
		t.Fatalf("exit code = %d, want non-zero", code)
	}
	if !strings.Contains(stderr.String(), "unexpected argument") {
		t.Fatalf("stderr = %q, want unexpected argument error", stderr.String())
	}
}
