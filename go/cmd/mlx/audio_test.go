// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package main

import (
	"context"
	"testing"

	core "dappco.re/go"
)

// audio with no -audio clip (and no model) is a usage error (exit 2).
func TestRunAudio_MissingClip_Bad(t *testing.T) {
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{"audio", t.TempDir()}, stdout, stderr)
	if code != 2 {
		t.Fatalf("exit = %d, want 2 (missing -audio clip)", code)
	}
	if !core.Contains(stderr.String(), "Usage:") {
		t.Fatalf("stderr = %q, want usage", stderr.String())
	}
}

// audio with a clip flag but no model path is a usage error (exit 2): the
// command requires exactly one positional model path.
func TestRunAudio_NoModelArg_Bad(t *testing.T) {
	dir := t.TempDir()
	wav := core.JoinPath(dir, "clip.wav")
	if r := core.WriteFile(wav, []byte("RIFF"), 0o644); !r.OK {
		t.Fatal(r.Value)
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{"audio", "-audio", wav}, stdout, stderr)
	if code != 2 {
		t.Fatalf("exit = %d, want 2 (no model path)", code)
	}
}

// audio with a clip + a bad model path reaches the native load-error path
// (exit 1) — confirms the runner advances past arg parsing into the real load.
func TestRunAudio_BadModelPath_Bad(t *testing.T) {
	dir := t.TempDir()
	wav := core.JoinPath(dir, "clip.wav")
	if r := core.WriteFile(wav, []byte("RIFF"), 0o644); !r.OK {
		t.Fatal(r.Value)
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runCommand(context.Background(), []string{"audio", "-audio", wav, core.JoinPath(dir, "absent-model")}, stdout, stderr)
	if code != 1 {
		t.Fatalf("exit = %d, want 1 (gemma4 load failure)", code)
	}
	if !core.Contains(stderr.String(), "audio: load") {
		t.Fatalf("stderr = %q, want the audio load error", stderr.String())
	}
}

func TestRunNativeAudioCommand_BadModelPath_Bad(t *testing.T) {
	dir := t.TempDir()
	wav := core.JoinPath(dir, "clip.wav")
	if r := core.WriteFile(wav, []byte("RIFF"), 0o644); !r.OK {
		t.Fatal(r.Value)
	}
	stdout, stderr := core.NewBuffer(), core.NewBuffer()
	code := runNativeAudioCommand(context.Background(), core.JoinPath(dir, "absent-model"), wav, "Transcribe.", 8, true, stdout, stderr)
	if code != 1 {
		t.Fatalf("exit = %d, want 1 (native load failure)", code)
	}
	if !core.Contains(stderr.String(), "audio: load") {
		t.Fatalf("stderr = %q, want the audio load error", stderr.String())
	}
}

// countTokenID counts occurrences of a token id in a slice — the shared
// multimodal placeholder counter.
func TestCountTokenID_Good(t *testing.T) {
	ids := []int32{5, 1, 5, 2, 5, 5}
	if got := countTokenID(ids, 5); got != 4 {
		t.Fatalf("countTokenID(…, 5) = %d, want 4", got)
	}
	if got := countTokenID(ids, 9); got != 0 {
		t.Fatalf("countTokenID(…, 9) = %d, want 0 (absent id)", got)
	}
	if got := countTokenID(nil, 1); got != 0 {
		t.Fatalf("countTokenID(nil, 1) = %d, want 0", got)
	}
}
