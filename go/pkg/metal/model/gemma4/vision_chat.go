// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

// The engine-facing vision-chat seam (#98): thin adapters over the existing
// vision lane (Gemma4ImagePixels → VisionTower → MultiModalProjector →
// ForwardMultiModal) implementing metal.VisionLanguageModel, so the neutral
// chat machinery can serve image turns without naming this family.

package gemma4

import (
	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/metal"
)

// EncodeImagePixels decodes PNG/JPEG bytes into vision-tower pixels using the
// model's loaded processor config, returning the pixels and the soft-token
// count the image occupies in the prompt.
func (m *Gemma4Model) EncodeImagePixels(data []byte) (*metal.Array, int, error) {
	return m.Gemma4ImagePixels(data, m.ImageFeatures)
}

// ImagePlaceholderBlock renders the HF processor convention for one image:
// BOI + ImageToken×softTokens + EOI. The block sits ahead of the user text
// inside the turn; the tokenizer expands it to exactly softTokens placeholder
// IDs that ForwardMultiModal swaps for projected vision features.
func (m *Gemma4Model) ImagePlaceholderBlock(softTokens int) string {
	if softTokens <= 0 {
		return ""
	}
	var b core.Builder
	b.Grow(len(Gemma4BOIToken) + len(Gemma4EOIToken) + softTokens*len(Gemma4ImageToken))
	b.WriteString(Gemma4BOIToken)
	for i := 0; i < softTokens; i++ {
		b.WriteString(Gemma4ImageToken)
	}
	b.WriteString(Gemma4EOIToken)
	return b.String()
}

// ImagePlaceholderTokenID is the token ID the placeholder block expands to.
func (m *Gemma4Model) ImagePlaceholderTokenID() int32 {
	if m == nil || m.Cfg == nil {
		return 0
	}
	return m.Cfg.ImageTokenID
}

// ProjectImageFeatures runs ONE image's pixels through the vision tower and
// multimodal projector — the expensive half of image encoding, separated
// from pixel decoding so the engine can cache projected features by image
// content (#99 vision-feature LRU). The caller owns the returned array.
func (m *Gemma4Model) ProjectImageFeatures(pixels *metal.Array) (*metal.Array, error) {
	if m == nil || (m.VisionTower == nil && m.MultiModalProjector == nil) {
		return nil, core.NewError("gemma4: model has no vision tower")
	}
	if pixels == nil || !pixels.Valid() {
		return nil, core.NewError("gemma4: image pixels are empty")
	}
	encoded := pixels
	if m.VisionTower != nil {
		encoded = m.VisionTower.Forward(pixels)
		if encoded == nil || !encoded.Valid() {
			metal.Free(encoded)
			return nil, core.NewError("gemma4: vision tower forward failed")
		}
	}
	projected := encoded
	if m.MultiModalProjector != nil {
		projected = m.MultiModalProjector.Forward(encoded)
		if encoded != pixels {
			metal.Free(encoded)
		}
	}
	if projected == pixels {
		projected = pixels.Clone()
	}
	if projected == nil || !projected.Valid() {
		metal.Free(projected)
		return nil, core.NewError("gemma4: multimodal projection failed")
	}
	return projected, nil
}

// ForwardImageFeatures is the image-bearing prefill over PRE-PROJECTED
// features: token embeddings with the features injected at placeholder
// positions, feature arrays BORROWED from the caller in placeholder order —
// ForwardUnifiedVideoMultiModal's image branch with the encode lifted out.
func (m *Gemma4Model) ForwardImageFeatures(tokens *metal.Array, features []*metal.Array, caches []metal.Cache) *metal.Array {
	valid := make([]*metal.Array, 0, len(features))
	for _, f := range features {
		if f != nil && f.Valid() {
			valid = append(valid, f)
		}
	}
	if len(valid) == 0 {
		return m.Forward(tokens, caches)
	}
	var shapeBuf [metal.MaxTensorRank]int32
	shape := tokens.ShapeInto(shapeBuf[:0])
	if len(shape) != 2 {
		return m.Forward(tokens, caches)
	}
	tokenIDs := tokens.DataInt32()

	h := m.EmbedTokens.Forward(tokens)
	scaledH := metal.MulScalar(h, m.Cfg.EmbeddingScale)
	metal.Free(h)
	h = scaledH

	injected := valid[0]
	var combined *metal.Array
	if len(valid) > 1 {
		combined = metal.Concatenate(valid, 0)
		injected = combined
	}
	h = m.injectGemma4ImageFeatures(h, tokenIDs, shape, injected)
	metal.Free(combined)
	return m.forwardGemma4EmbeddingsMasked(tokens, h, nil, caches)
}

// AcceptsImageInput reports whether this checkpoint shipped a vision tower.
func (m *Gemma4Model) AcceptsImageInput() bool {
	return m != nil && (m.VisionTower != nil || m.MultiModalProjector != nil)
}
