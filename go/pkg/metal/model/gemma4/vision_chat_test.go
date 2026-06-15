// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"strings"
	"testing"

	"dappco.re/go/mlx/pkg/metal"
)

// TestVisionChat_ImagePlaceholderBlock_Good renders the HF processor convention
// for one image — BOI + ImageToken×softTokens + EOI — and asserts the block
// frames exactly softTokens image tokens between the boundary markers.
func TestVisionChat_ImagePlaceholderBlock_Good(t *testing.T) {
	m := &Gemma4Model{}
	const soft = 4
	block := m.ImagePlaceholderBlock(soft)

	if !strings.HasPrefix(block, Gemma4BOIToken) {
		t.Fatalf("block %q does not start with BOI %q", block, Gemma4BOIToken)
	}
	if !strings.HasSuffix(block, Gemma4EOIToken) {
		t.Fatalf("block %q does not end with EOI %q", block, Gemma4EOIToken)
	}
	if got := strings.Count(block, Gemma4ImageToken); got != soft {
		t.Fatalf("image-token count = %d, want softTokens %d", got, soft)
	}
}

// TestVisionChat_ImagePlaceholderBlock_Bad pins the non-positive guard: zero or
// negative soft tokens render an empty block (no image occupies the prompt).
func TestVisionChat_ImagePlaceholderBlock_Bad(t *testing.T) {
	m := &Gemma4Model{}
	if got := m.ImagePlaceholderBlock(0); got != "" {
		t.Fatalf("ImagePlaceholderBlock(0) = %q, want empty", got)
	}
	if got := m.ImagePlaceholderBlock(-3); got != "" {
		t.Fatalf("ImagePlaceholderBlock(-3) = %q, want empty", got)
	}
}

// TestVisionChat_ImagePlaceholderTokenID_Good reads the placeholder id straight
// from config, and the nil-config guard returns 0.
func TestVisionChat_ImagePlaceholderTokenID_Good(t *testing.T) {
	m := &Gemma4Model{Cfg: &Gemma4TextConfig{ImageTokenID: 262145}}
	if got := m.ImagePlaceholderTokenID(); got != 262145 {
		t.Fatalf("ImagePlaceholderTokenID = %d, want 262145", got)
	}

	var nilModel *Gemma4Model
	if got := nilModel.ImagePlaceholderTokenID(); got != 0 {
		t.Fatalf("ImagePlaceholderTokenID(nil) = %d, want 0", got)
	}
	noCfg := &Gemma4Model{}
	if got := noCfg.ImagePlaceholderTokenID(); got != 0 {
		t.Fatalf("ImagePlaceholderTokenID(no cfg) = %d, want 0", got)
	}
}

// TestVisionChat_AcceptsImageInput_Good distinguishes a text-only checkpoint
// (no vision tower, no projector) from one that shipped vision weights.
func TestVisionChat_AcceptsImageInput_Good(t *testing.T) {
	textOnly := &Gemma4Model{}
	if textOnly.AcceptsImageInput() {
		t.Fatal("AcceptsImageInput = true for a text-only model, want false")
	}

	var nilModel *Gemma4Model
	if nilModel.AcceptsImageInput() {
		t.Fatal("AcceptsImageInput(nil) = true, want false")
	}

	withProjector := &Gemma4Model{MultiModalProjector: &Gemma4MultiModalProjector{}}
	if !withProjector.AcceptsImageInput() {
		t.Fatal("AcceptsImageInput = false with a projector present, want true")
	}
}

// TestVisionChat_ProjectImageFeatures_Bad pins the two early guards: a model with
// no vision lane, and empty/invalid pixels, both return an error without
// touching the tower.
func TestVisionChat_ProjectImageFeatures_Bad(t *testing.T) {
	requireMetalRuntime(t)

	textOnly := &Gemma4Model{}
	pixels := metal.Zeros([]int32{1, 4, 4, 3}, metal.DTypeFloat32)
	defer metal.Free(pixels)
	if _, err := textOnly.ProjectImageFeatures(pixels); err == nil {
		t.Fatal("ProjectImageFeatures with no vision tower returned nil error")
	}

	withProjector := &Gemma4Model{MultiModalProjector: &Gemma4MultiModalProjector{}}
	if _, err := withProjector.ProjectImageFeatures(nil); err == nil {
		t.Fatal("ProjectImageFeatures(nil pixels) returned nil error")
	}
}

// TestVisionChat_EncodeImagePixels_Bad pins that a text-only model (no image
// feature config) refuses to decode image bytes rather than panicking on a nil
// processor config.
func TestVisionChat_EncodeImagePixels_Bad(t *testing.T) {
	requireMetalRuntime(t)

	textOnly := &Gemma4Model{}
	if _, _, err := textOnly.EncodeImagePixels([]byte("not an image")); err == nil {
		t.Fatal("EncodeImagePixels on a text-only model returned nil error")
	}
}

// TestVisionChat_ForwardImageFeatures_NoValidFeatures_Good asserts the
// no-valid-features fast path: when every feature array is nil/invalid, the call
// falls back to a plain text Forward and produces well-shaped logits.
func TestVisionChat_ForwardImageFeatures_NoValidFeatures_Good(t *testing.T) {
	model := loadGemma4DenseTestModel(t)

	tokens := metal.FromValues([]int32{2, 3, 4}, 1, 3)
	caches := model.NewCache()
	// All features invalid → fall back to Forward.
	logits := model.ForwardImageFeatures(tokens, []*metal.Array{nil}, caches)
	if err := metal.Eval(logits); err != nil {
		t.Fatalf("Eval logits: %v", err)
	}
	defer func() {
		metal.Free(tokens, logits)
		metal.FreeCaches(caches)
	}()

	shape := logits.Shape()
	if len(shape) != 3 || shape[0] != 1 || shape[1] != 3 || shape[2] != 10 {
		t.Fatalf("fallback logits shape = %v, want [1 3 10]", shape)
	}
}
