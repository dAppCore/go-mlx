// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4_test

import (
	"testing"

	"dappco.re/go/mlx/pkg/metal"
	"dappco.re/go/mlx/pkg/metal/model/gemma4"
)

// TestGemma4Capabilities_UsesFixedSlidingCache_Good proves the fixed-sliding
// cache decision is derived from config (a declared sliding window), not a
// Gemma-4 family-name check.
func TestGemma4Capabilities_UsesFixedSlidingCache_Good(t *testing.T) {
	hybrid := &gemma4.Gemma4Model{Cfg: &gemma4.Gemma4TextConfig{SlidingWindow: 1024}}
	if !hybrid.UsesFixedSlidingCache() {
		t.Fatal("UsesFixedSlidingCache() = false, want true for a sliding-window Gemma-4 build")
	}
	if (&gemma4.Gemma4Model{Cfg: &gemma4.Gemma4TextConfig{}}).UsesFixedSlidingCache() {
		t.Fatal("UsesFixedSlidingCache() = true, want false for a build with no sliding window")
	}
}

// TestGemma4Capabilities_NeedsThoughtChannelSuppressor_Good proves only the
// large variants (num_attention_heads >= 16) need the thought-channel
// suppressor, derived from config, and that a nil Cfg answers false rather
// than panicking.
func TestGemma4Capabilities_NeedsThoughtChannelSuppressor_Good(t *testing.T) {
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

// TestGemma4EngineFeatures_CacheFromConfig_Good proves the fixed-sliding KV
// cache family is selected by the model's config, not an engine gate: a build
// that declares a sliding window declares the bounded fixed-sliding cache (so
// the engine reacts to the model and a 256K sliding model does not page its
// full context), while a dense build declares neither. The accepted always-on
// fast-paths (greedy token, fused matvecs, streaming) remain on regardless.
func TestGemma4EngineFeatures_CacheFromConfig_Good(t *testing.T) {
	hybrid := (&gemma4.Gemma4Model{Cfg: &gemma4.Gemma4TextConfig{SlidingWindow: 1024}}).EngineFeatures()
	if !hybrid.FixedSlidingCache {
		t.Fatal("FixedSlidingCache = false, want true for a sliding-window Gemma-4 build (config selects it)")
	}
	if !hybrid.FixedSlidingCacheBound {
		t.Fatal("FixedSlidingCacheBound = false, want true alongside the fixed-sliding cache")
	}
	if !hybrid.DirectGreedyToken || !hybrid.NativeMLPMatVec {
		t.Fatal("accepted always-on fast-paths must stay on for a hybrid build")
	}

	dense := (&gemma4.Gemma4Model{Cfg: &gemma4.Gemma4TextConfig{}}).EngineFeatures()
	if dense.FixedSlidingCache || dense.FixedSlidingCacheBound {
		t.Fatal("dense Gemma-4 build (no sliding window) must not declare the fixed-sliding cache")
	}
}
