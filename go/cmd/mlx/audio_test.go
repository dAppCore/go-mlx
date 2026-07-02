// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package main

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/model"
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

type audioPromptEmbeddingCommandModel struct {
	rows           map[int32][]byte
	placeholderID  int32
	embeddingBytes int
	embedIntoCalls int
}

func (m *audioPromptEmbeddingCommandModel) Embed(id int32) ([]byte, error) {
	if row, ok := m.rows[id]; ok {
		return row, nil
	}
	return []byte{byte(id)}, nil
}

func (m *audioPromptEmbeddingCommandModel) EmbeddingBytes() int { return m.embeddingBytes }

func (m *audioPromptEmbeddingCommandModel) EmbedInto(dst []byte, id int32) ([]byte, error) {
	m.embedIntoCalls++
	row, err := m.Embed(id)
	if err != nil {
		return nil, err
	}
	if len(dst) != len(row) {
		return nil, core.NewError("test audioPromptEmbeddingCommandModel: dst size mismatch")
	}
	copy(dst, row)
	return dst, nil
}

func (*audioPromptEmbeddingCommandModel) DecodeForward([][]byte) ([][]byte, error) { return nil, nil }
func (*audioPromptEmbeddingCommandModel) Head([]byte) ([]byte, error)              { return nil, nil }
func (*audioPromptEmbeddingCommandModel) Vocab() int                               { return 0 }
func (*audioPromptEmbeddingCommandModel) OpenSession() (model.DecodeStepper, error) {
	return nil, nil
}
func (*audioPromptEmbeddingCommandModel) AcceptsAudioInput() bool { return true }
func (m *audioPromptEmbeddingCommandModel) AudioPlaceholderTokenID() int32 {
	return m.placeholderID
}
func (*audioPromptEmbeddingCommandModel) AudioPlaceholderBlock(int) string { return "" }
func (*audioPromptEmbeddingCommandModel) AudioSoftTokens(int) int          { return 0 }
func (*audioPromptEmbeddingCommandModel) ProjectAudioFeatures([]byte, int, int) ([]byte, error) {
	return nil, nil
}

func TestNativeAudioPromptEmbeddingsBorrowsProjectedRows_Good(t *testing.T) {
	model := audioPromptEmbeddingCommandModel{
		placeholderID: 77,
		rows: map[int32][]byte{
			10: {0x10},
			11: {0x11},
			12: {0x12},
			77: {0x00},
		},
	}
	ids := []int32{10, 77, 11, 77, 12}
	features := []byte{0xa1, 0xa2}

	got, err := nativeAudioPromptEmbeddings(&model, ids, features)
	if err != nil {
		t.Fatalf("nativeAudioPromptEmbeddings: %v", err)
	}
	if &got[1][0] != &features[0] || &got[3][0] != &features[1] {
		t.Fatal("audio command feature rows were copied; want borrowed projected feature row views")
	}
	if &got[0][0] != &model.rows[10][0] || &got[2][0] != &model.rows[11][0] || &got[4][0] != &model.rows[12][0] {
		t.Fatal("audio command text rows were copied; want borrowed token embedding row views")
	}
}

func TestNativeAudioPromptEmbeddingsUsesEmbedInto_Good(t *testing.T) {
	model := audioPromptEmbeddingCommandModel{
		placeholderID:  77,
		embeddingBytes: 1,
		rows: map[int32][]byte{
			10: {0x10},
			11: {0x11},
			12: {0x12},
			77: {0x00},
		},
	}
	ids := []int32{10, 77, 11, 77, 12}
	features := []byte{0xa1, 0xa2}

	got, err := nativeAudioPromptEmbeddings(&model, ids, features)
	if err != nil {
		t.Fatalf("nativeAudioPromptEmbeddings: %v", err)
	}
	if model.embedIntoCalls != len(ids) {
		t.Fatalf("EmbedInto calls = %d, want %d", model.embedIntoCalls, len(ids))
	}
	if &got[1][0] != &features[0] || &got[3][0] != &features[1] {
		t.Fatal("audio command feature rows were copied; want borrowed projected feature row views")
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
