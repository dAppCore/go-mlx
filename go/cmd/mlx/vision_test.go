// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package main

import (
	"context"
	"testing"

	core "dappco.re/go"
)

// splitPathList splits a comma list, trimming blanks; empty input → nil.
func TestSplitPathList_Good(t *testing.T) {
	if got := splitPathList(""); got != nil {
		t.Fatalf("splitPathList(\"\") = %v, want nil", got)
	}
	if got := splitPathList("   "); got != nil {
		t.Fatalf("splitPathList(blank) = %v, want nil", got)
	}
	got := splitPathList(" a.png , ,b.jpg,, c.png ")
	want := []string{"a.png", "b.jpg", "c.png"}
	if len(got) != len(want) {
		t.Fatalf("splitPathList = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("splitPathList[%d] = %q, want %q", i, got[i], want[i])
		}
	}
}

// vision with neither -images nor -video-frames (and no model) is a usage
// error (exit 2).
func TestRunVision_NoInputs_Bad(t *testing.T) {
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{"vision", t.TempDir()}, stdout, stderr)
	if code != 2 {
		t.Fatalf("exit = %d, want 2 (no images/frames)", code)
	}
	if !core.Contains(stderr.String(), "Usage:") {
		t.Fatalf("stderr = %q, want usage", stderr.String())
	}
}

// vision with an image but no model path is a usage error (exit 2).
func TestRunVision_NoModelArg_Bad(t *testing.T) {
	dir := t.TempDir()
	img := core.JoinPath(dir, "a.png")
	if r := core.WriteFile(img, []byte("\x89PNG"), 0o644); !r.OK {
		t.Fatal(r.Value)
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{"vision", "-images", img}, stdout, stderr)
	if code != 2 {
		t.Fatalf("exit = %d, want 2 (no model path)", code)
	}
}

// vision with an image + a bad model path reaches the gemma4 load-error path
// (exit 1).
func TestRunVision_BadModelPath_Bad(t *testing.T) {
	dir := t.TempDir()
	img := core.JoinPath(dir, "a.png")
	if r := core.WriteFile(img, []byte("\x89PNG"), 0o644); !r.OK {
		t.Fatal(r.Value)
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{"vision", "-images", img, core.JoinPath(dir, "absent-model")}, stdout, stderr)
	if code != 1 {
		t.Fatalf("exit = %d, want 1 (gemma4 load failure)", code)
	}
	if !core.Contains(stderr.String(), "vision: load") {
		t.Fatalf("stderr = %q, want the vision load error", stderr.String())
	}
}

func TestRunNativeVisionCommand_BadModelPath_Bad(t *testing.T) {
	dir := t.TempDir()
	img := core.JoinPath(dir, "a.png")
	if r := core.WriteFile(img, []byte("\x89PNG"), 0o644); !r.OK {
		t.Fatal(r.Value)
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runNativeVisionCommand(context.Background(), core.JoinPath(dir, "absent-model"), []string{img}, nil, 1, "Describe.", 8, true, stdout, stderr)
	if code != 1 {
		t.Fatalf("exit = %d, want 1 (native load failure)", code)
	}
	if !core.Contains(stderr.String(), "vision: load") {
		t.Fatalf("stderr = %q, want the vision load error", stderr.String())
	}
}
