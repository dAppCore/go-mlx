// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"archive/zip"
	"context"
	"testing"

	core "dappco.re/go"
)

// ebook with no --model is a usage error (exit 2). Pure arg-parse path.
func TestRunEbook_NoModel_Bad(t *testing.T) {
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{"ebook"}, stdout, stderr)
	if code != 2 {
		t.Fatalf("exit = %d, want 2 (missing --model)", code)
	}
	if !core.Contains(stderr.String(), "Usage:") {
		t.Fatalf("stderr = %q, want usage", stderr.String())
	}
}

// ebook -h prints the flag usage. The ebook runner treats any fs.Parse error
// (including flag.ErrHelp) as exit 2 — it does not special-case -h the way
// serve/generate do — so -h surfaces the flag defaults and exits 2.
func TestRunEbook_Help_Good(t *testing.T) {
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{"ebook", "-h"}, stdout, stderr)
	if code != 2 {
		t.Fatalf("exit = %d, want 2 (ebook routes -h through the parse-error path)", code)
	}
	if !core.Contains(stderr.String(), "model directory to render") {
		t.Fatalf("stderr = %q, want the flag usage", stderr.String())
	}
}

// ebook against a missing model dir fails to build the book (exit 1) — the
// BuildModelBook error path, no model load.
func TestRunEbook_MissingDir_Bad(t *testing.T) {
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{"ebook", "--model", core.JoinPath(t.TempDir(), "absent")}, stdout, stderr)
	if code != 1 {
		t.Fatalf("exit = %d, want 1 (build failure on missing dir)", code)
	}
	if !core.Contains(stderr.String(), "ebook:") {
		t.Fatalf("stderr = %q, want an ebook error", stderr.String())
	}
}

// ebook over a tiny model dir, weights excluded, writes a valid EPUB (ZIP)
// with chapters — pure file I/O, no model loaded. Default out path is
// <model-name>.epub in the working dir, so an explicit --out keeps the test
// hermetic inside t.TempDir().
func TestRunEbook_ManifestOnly_Good(t *testing.T) {
	model := t.TempDir()
	// Write a couple of small files so the book has content (foreword + a
	// stand-in weights file the --weights=false run skips).
	writeEbookFile(t, core.JoinPath(model, "config.json"), `{"model_type":"gemma3","hidden_size":8}`)
	writeEbookFile(t, core.JoinPath(model, "README.md"), "# Tiny\nForeword anchor.\n")
	writeEbookFile(t, core.JoinPath(model, "model.safetensors"), "not-real-weights")

	out := core.JoinPath(t.TempDir(), "tiny.epub")
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{
		"ebook", "--model", model, "--out", out, "--weights=false", "--title", "Tiny", "--author", "Lethean",
	}, stdout, stderr)
	if code != 0 {
		t.Fatalf("exit = %d, want 0; stderr=%q", code, stderr.String())
	}
	if !core.Contains(stdout.String(), "wrote "+out) {
		t.Fatalf("stdout = %q, want the wrote-<path> line", stdout.String())
	}
	// The output must be a readable ZIP (EPUB is a ZIP container).
	zr, err := zip.OpenReader(out)
	if err != nil {
		t.Fatalf("output is not a valid EPUB zip: %v", err)
	}
	defer zr.Close()
	if len(zr.File) == 0 {
		t.Fatal("EPUB zip has no entries")
	}
}

func writeEbookFile(t *testing.T, path, data string) {
	t.Helper()
	if r := core.WriteFile(path, []byte(data), 0o644); !r.OK {
		t.Fatalf("write %s: %v", path, r.Value)
	}
}
