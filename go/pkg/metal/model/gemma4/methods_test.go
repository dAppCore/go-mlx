// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"testing"

	"dappco.re/go/mlx/pkg/metal"
)

// TestClampSlidingWindow_Good pins the Gemma 4 load-time sliding-window clamp
// behaviour (SlidingWindowClamper): clamp down to a smaller runtime maximum,
// fill an unset (<=0) window, and never expand an already-smaller window. The
// metal-side dispatch + non-positive-window guard are pinned by package metal's
// model_dispatch_test.go; this test owns the architecture-specific rules
// (relocated from metal's backend_test.go by the gemma4 extraction).
func TestClampSlidingWindow_Good(t *testing.T) {
	model := &Gemma4Model{Cfg: &Gemma4TextConfig{SlidingWindow: 2048}}
	model.ClampSlidingWindow(512)
	if model.Cfg.SlidingWindow != 512 {
		t.Fatalf("SlidingWindow = %d, want clamped down to 512", model.Cfg.SlidingWindow)
	}
	model.ClampSlidingWindow(1024)
	if model.Cfg.SlidingWindow != 512 {
		t.Fatalf("SlidingWindow = %d, want unchanged (no expansion above existing cap)", model.Cfg.SlidingWindow)
	}

	unset := &Gemma4Model{Cfg: &Gemma4TextConfig{SlidingWindow: 0}}
	unset.ClampSlidingWindow(512)
	if unset.Cfg.SlidingWindow != 512 {
		t.Fatalf("SlidingWindow = %d, want unset window filled to 512", unset.Cfg.SlidingWindow)
	}
}

// TestClampSlidingWindow_Bad pins the nil guards: a nil model or nil config must
// not panic.
func TestClampSlidingWindow_Bad(t *testing.T) {
	var nilModel *Gemma4Model
	nilModel.ClampSlidingWindow(512)

	noCfg := &Gemma4Model{}
	noCfg.ClampSlidingWindow(512)
}

// TestFixedSlidingPrefillChunkLimit_Good pins the Gemma 4 fixed-sliding prefill
// chunk limit (FixedSlidingPrefillLimiter): the sliding window caps the chunk,
// a smaller fixed cache caps it further, and a non-fixed cache is ignored. This
// is the architecture-specific half that metal's effectivePrefillChunkSize
// consumes (its cap/dispatch glue stays pinned in package metal).
func TestFixedSlidingPrefillChunkLimit_Good(t *testing.T) {
	model := &Gemma4Model{Cfg: &Gemma4TextConfig{SlidingWindow: 512}}

	// Fixed cache at the window size + a non-fixed cache: limit stays the window.
	caches := []metal.Cache{metal.NewFixedKVCache(512), metal.NewKVCache()}
	if got := model.FixedSlidingPrefillChunkLimit(caches); got != 512 {
		t.Fatalf("FixedSlidingPrefillChunkLimit = %d, want sliding window 512", got)
	}

	// A smaller fixed cache caps the limit below the window.
	smaller := []metal.Cache{metal.NewFixedKVCache(256)}
	if got := model.FixedSlidingPrefillChunkLimit(smaller); got != 256 {
		t.Fatalf("FixedSlidingPrefillChunkLimit = %d, want smaller fixed cache 256", got)
	}
}

// TestFixedSlidingPrefillChunkLimit_Bad pins the no-sliding-window and nil
// guards: a zero window or nil model/config yields 0 (no fixed-sliding limit).
func TestFixedSlidingPrefillChunkLimit_Bad(t *testing.T) {
	noWindow := &Gemma4Model{Cfg: &Gemma4TextConfig{SlidingWindow: 0}}
	if got := noWindow.FixedSlidingPrefillChunkLimit([]metal.Cache{metal.NewFixedKVCache(256)}); got != 0 {
		t.Fatalf("FixedSlidingPrefillChunkLimit(no window) = %d, want 0", got)
	}

	var nilModel *Gemma4Model
	if got := nilModel.FixedSlidingPrefillChunkLimit(nil); got != 0 {
		t.Fatalf("FixedSlidingPrefillChunkLimit(nil model) = %d, want 0", got)
	}
}
