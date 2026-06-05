// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"testing"

	"dappco.re/go/mlx/pkg/metal"
)

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
