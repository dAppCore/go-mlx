// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package main

import (
	"context"
	"testing"

	core "dappco.re/go"
)

// diffuse with no model path is a usage error (exit 2) before any load.
func TestRunDiffuse_NoModel_Bad(t *testing.T) {
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{"diffuse"}, stdout, stderr)
	if code != 2 {
		t.Fatalf("exit = %d, want 2 (missing model path)", code)
	}
	if !core.Contains(stderr.String(), "Usage:") {
		t.Fatalf("stderr = %q, want usage", stderr.String())
	}
}

// diffuse against a missing model dir fails to load (exit 1) — the
// LoadDiffusionGemma error path.
func TestRunDiffuse_BadModelPath_Bad(t *testing.T) {
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{"diffuse", core.JoinPath(t.TempDir(), "absent")}, stdout, stderr)
	if code != 1 {
		t.Fatalf("exit = %d, want 1 (load failure)", code)
	}
	if !core.Contains(stderr.String(), "diffuse: load") {
		t.Fatalf("stderr = %q, want the diffuse load error", stderr.String())
	}
}

// diffuse over a TINY synthetic diffusion_gemma runs the real block-diffusion
// sampler end-to-end (load → denoise canvases → commit → decode) and prints the
// per-run diffusion summary. Bounded to a single tiny canvas + few steps.
func TestRunDiffuse_SyntheticModel_Good(t *testing.T) {
	requireSyntheticRuntime(t)
	model := writeSyntheticDiffusionModel(t, t.TempDir())
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{
		"diffuse", "-prompt", "hi", "-max-canvases", "1", "-steps", "2", "-canvas", "4", "-seed", "1", model,
	}, stdout, stderr)
	if code != 0 {
		t.Fatalf("exit = %d, want 0; stderr=%q", code, stderr.String())
	}
	if !core.Contains(stdout.String(), "diffusion") {
		t.Fatalf("stdout = %q, want the diffusion summary line", stdout.String())
	}
}

// diffuse -trace adds the per-step trace lines to stderr while still printing
// the summary — the trace OnStep/OnCanvas hooks fire.
func TestRunDiffuse_SyntheticTrace_Good(t *testing.T) {
	requireSyntheticRuntime(t)
	model := writeSyntheticDiffusionModel(t, t.TempDir())
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{
		"diffuse", "-trace", "-prompt", "hi", "-max-canvases", "1", "-steps", "2", "-canvas", "4", "-seed", "1", model,
	}, stdout, stderr)
	if code != 0 {
		t.Fatalf("exit = %d, want 0; stderr=%q", code, stderr.String())
	}
	if !core.Contains(stderr.String(), "canvas") {
		t.Fatalf("trace stderr = %q, want per-canvas trace lines", stderr.String())
	}
}
