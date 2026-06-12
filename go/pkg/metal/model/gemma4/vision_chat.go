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

// ForwardImageMultiModal is the image-bearing prefill: token embeddings with
// projected vision features injected at the placeholder positions, image
// arrays consumed in placeholder order.
func (m *Gemma4Model) ForwardImageMultiModal(tokens *metal.Array, imagePixels []*metal.Array, caches []metal.Cache) *metal.Array {
	return m.ForwardMultiModal(tokens, imagePixels, caches)
}

// AcceptsImageInput reports whether this checkpoint shipped a vision tower.
func (m *Gemma4Model) AcceptsImageInput() bool {
	return m != nil && (m.VisionTower != nil || m.MultiModalProjector != nil)
}
