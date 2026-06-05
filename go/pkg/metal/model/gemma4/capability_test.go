// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4_test

import (
	"testing"

	"dappco.re/go/mlx/pkg/metal"
	"dappco.re/go/mlx/pkg/metal/model/gemma4"
)

// TestGemma4Capabilities_Good proves the cache + prompt capabilities the engine
// dispatches on instead of a Gemma-4 family-name check: Gemma 4 always uses the
// fixed sliding-window cache, and only the large variants (num_attention_heads
// >= 16) need the thought-channel suppressor.
func TestGemma4Capabilities_Good(t *testing.T) {
	if !(&gemma4.Gemma4Model{}).UsesFixedSlidingCache() {
		t.Fatal("UsesFixedSlidingCache() = false, want true")
	}

	large := &gemma4.Gemma4Model{Cfg: &gemma4.Gemma4TextConfig{TransformerConfig: metal.TransformerConfig{NumAttentionHeads: 16}}}
	if !large.NeedsThoughtChannelSuppressor() {
		t.Fatal("NeedsThoughtChannelSuppressor(heads=16) = false, want true for the large variant")
	}

	small := &gemma4.Gemma4Model{Cfg: &gemma4.Gemma4TextConfig{TransformerConfig: metal.TransformerConfig{NumAttentionHeads: 8}}}
	if small.NeedsThoughtChannelSuppressor() {
		t.Fatal("NeedsThoughtChannelSuppressor(heads=8) = true, want false for the small variant")
	}

	if (&gemma4.Gemma4Model{}).NeedsThoughtChannelSuppressor() {
		t.Fatal("NeedsThoughtChannelSuppressor(nil cfg) = true, want false")
	}
}
