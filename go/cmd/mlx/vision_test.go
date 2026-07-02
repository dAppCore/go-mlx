// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package main

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/model"
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

type visionPromptEmbeddingCommandModel struct {
	rows           map[int32][]byte
	imageID        int32
	videoID        int32
	embeddingBytes int
	embedIntoCalls int
}

func (m *visionPromptEmbeddingCommandModel) Embed(id int32) ([]byte, error) {
	if row, ok := m.rows[id]; ok {
		return row, nil
	}
	return []byte{byte(id)}, nil
}

func (m *visionPromptEmbeddingCommandModel) EmbeddingBytes() int { return m.embeddingBytes }

func (m *visionPromptEmbeddingCommandModel) EmbedInto(dst []byte, id int32) ([]byte, error) {
	m.embedIntoCalls++
	row, err := m.Embed(id)
	if err != nil {
		return nil, err
	}
	if len(dst) != len(row) {
		return nil, core.NewError("test visionPromptEmbeddingCommandModel: dst size mismatch")
	}
	copy(dst, row)
	return dst, nil
}

func (*visionPromptEmbeddingCommandModel) DecodeForward([][]byte) ([][]byte, error) { return nil, nil }
func (*visionPromptEmbeddingCommandModel) Head([]byte) ([]byte, error)              { return nil, nil }
func (*visionPromptEmbeddingCommandModel) Vocab() int                               { return 0 }
func (*visionPromptEmbeddingCommandModel) OpenSession() (model.DecodeStepper, error) {
	return nil, nil
}
func (*visionPromptEmbeddingCommandModel) AcceptsImageInput() bool { return true }
func (m *visionPromptEmbeddingCommandModel) ImagePlaceholderTokenID() int32 {
	return m.imageID
}
func (*visionPromptEmbeddingCommandModel) ImagePlaceholderBlock(int) string { return "" }
func (m *visionPromptEmbeddingCommandModel) VideoPlaceholderTokenID() int32 {
	return m.videoID
}
func (*visionPromptEmbeddingCommandModel) VideoPlaceholderBlock(int) string { return "" }
func (*visionPromptEmbeddingCommandModel) ProjectImageFeatures([]byte) ([]byte, error) {
	return nil, nil
}

func TestNativeVisionPromptEmbeddingsBorrowsProjectedRows_Good(t *testing.T) {
	model := visionPromptEmbeddingCommandModel{
		imageID: 77,
		videoID: 88,
		rows: map[int32][]byte{
			10: {0x10},
			11: {0x11},
			12: {0x12},
			77: {0x00},
			88: {0x00},
		},
	}
	ids := []int32{10, 77, 11, 88, 12}
	imageFeatures := []byte{0xa1}
	videoFeatures := []byte{0xb1}

	got, err := nativeVisionPromptEmbeddings(&model, ids, imageFeatures, videoFeatures)
	if err != nil {
		t.Fatalf("nativeVisionPromptEmbeddings: %v", err)
	}
	if &got[1][0] != &imageFeatures[0] {
		t.Fatal("image command feature row was copied; want borrowed projected feature row view")
	}
	if &got[3][0] != &videoFeatures[0] {
		t.Fatal("video command feature row was copied; want borrowed projected feature row view")
	}
	if &got[0][0] != &model.rows[10][0] || &got[2][0] != &model.rows[11][0] || &got[4][0] != &model.rows[12][0] {
		t.Fatal("vision command text rows were copied; want borrowed token embedding row views")
	}
}

func TestNativeVisionPromptEmbeddingsUsesEmbedInto_Good(t *testing.T) {
	model := visionPromptEmbeddingCommandModel{
		imageID:        77,
		videoID:        88,
		embeddingBytes: 1,
		rows: map[int32][]byte{
			10: {0x10},
			11: {0x11},
			12: {0x12},
			77: {0x00},
			88: {0x00},
		},
	}
	ids := []int32{10, 77, 11, 88, 12}
	imageFeatures := []byte{0xa1}
	videoFeatures := []byte{0xb1}

	got, err := nativeVisionPromptEmbeddings(&model, ids, imageFeatures, videoFeatures)
	if err != nil {
		t.Fatalf("nativeVisionPromptEmbeddings: %v", err)
	}
	if model.embedIntoCalls != len(ids) {
		t.Fatalf("EmbedInto calls = %d, want %d", model.embedIntoCalls, len(ids))
	}
	if &got[1][0] != &imageFeatures[0] {
		t.Fatal("image command feature row was copied; want borrowed projected feature row view")
	}
	if &got[3][0] != &videoFeatures[0] {
		t.Fatal("video command feature row was copied; want borrowed projected feature row view")
	}
}
