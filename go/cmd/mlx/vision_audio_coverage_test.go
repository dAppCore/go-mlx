// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package main

import (
	"context"
	"testing"

	core "dappco.re/go"
)

// vision over a real gemma4-text checkpoint that has NO vision tower hits the
// "this checkpoint has no vision tower" guard (exit 1) — the runner loaded the
// model for real and advanced past the load into the tower check.
func TestRunVision_NoVisionTower_Bad(t *testing.T) {
	requireSyntheticRuntime(t)
	model := writeSyntheticGemma4TextModel(t)
	img := core.JoinPath(t.TempDir(), "a.png")
	if r := core.WriteFile(img, []byte("\x89PNG\r\n"), 0o644); !r.OK {
		t.Fatalf("write image: %v", r.Value)
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{"vision", "-images", img, model}, stdout, stderr)
	if code != 1 {
		t.Fatalf("exit = %d, want 1 (no vision tower); stderr=%q", code, stderr.String())
	}
	if !core.Contains(stderr.String(), "no vision tower") {
		t.Fatalf("stderr = %q, want the no-vision-tower notice", stderr.String())
	}
}

// vision with a bad flag fails to parse and exits 2 (the fs.Parse error path).
func TestRunVision_BadFlag_Bad(t *testing.T) {
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{"vision", "-not-a-flag", "x"}, stdout, stderr)
	if code != 2 {
		t.Fatalf("exit = %d, want 2 (bad flag)", code)
	}
}

// audio over a real gemma4-text checkpoint that has NO audio tower hits the
// "this checkpoint has no audio tower" guard (exit 1).
func TestRunAudio_NoAudioTower_Bad(t *testing.T) {
	requireSyntheticRuntime(t)
	model := writeSyntheticGemma4TextModel(t)
	wav := core.JoinPath(t.TempDir(), "clip.wav")
	if r := core.WriteFile(wav, []byte("RIFF\x00\x00\x00\x00WAVE"), 0o644); !r.OK {
		t.Fatalf("write wav: %v", r.Value)
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{"audio", "-audio", wav, model}, stdout, stderr)
	if code != 1 {
		t.Fatalf("exit = %d, want 1 (no audio tower); stderr=%q", code, stderr.String())
	}
	if !core.Contains(stderr.String(), "no audio tower") {
		t.Fatalf("stderr = %q, want the no-audio-tower notice", stderr.String())
	}
}

// audio with a bad flag fails to parse and exits 2.
func TestRunAudio_BadFlag_Bad(t *testing.T) {
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{"audio", "-not-a-flag", "x"}, stdout, stderr)
	if code != 2 {
		t.Fatalf("exit = %d, want 2 (bad flag)", code)
	}
}
