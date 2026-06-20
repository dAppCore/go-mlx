// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"testing"
)

func TestFirstPositive_Good(t *testing.T) {
	if got := firstPositive(0, 0, 7, 3); got != 7 {
		t.Fatalf("firstPositive(0,0,7,3) = %d, want 7", got)
	}
	if got := firstPositive(-1, 0, 5); got != 5 {
		t.Fatalf("firstPositive(-1,0,5) = %d, want 5", got)
	}
	if got := firstPositive(2); got != 2 {
		t.Fatalf("firstPositive(2) = %d, want 2", got)
	}
}

func TestFirstPositive_AllNonPositive_Ugly(t *testing.T) {
	if got := firstPositive(0, -1, -9); got != 0 {
		t.Fatalf("firstPositive(0,-1,-9) = %d, want 0", got)
	}
	if got := firstPositive(); got != 0 {
		t.Fatalf("firstPositive() = %d, want 0", got)
	}
}

func TestSampleFromGenerateConfig_Good(t *testing.T) {
	cfg := GenerateConfig{
		MaxTokens:     42,
		Temperature:   0.7,
		TopK:          40,
		TopP:          0.95,
		MinP:          0.05,
		StopTokens:    []int32{1, 2, 3},
		RepeatPenalty: 1.1,
	}
	s := sampleFromGenerateConfig(cfg)
	if s.MaxTokens != 42 || s.Temperature != 0.7 || s.TopK != 40 {
		t.Fatalf("scalar fields not forwarded: %+v", s)
	}
	if s.TopP != 0.95 || s.MinP != 0.05 || s.RepeatPenalty != 1.1 {
		t.Fatalf("float fields not forwarded: %+v", s)
	}
	if len(s.StopTokens) != 3 || s.StopTokens[0] != 1 || s.StopTokens[2] != 3 {
		t.Fatalf("StopTokens not cloned correctly: %v", s.StopTokens)
	}
	// Mutating the clone must not touch the source — defensive clone.
	s.StopTokens[0] = 99
	if cfg.StopTokens[0] != 1 {
		t.Fatalf("StopTokens clone aliased source: src=%v", cfg.StopTokens)
	}
}

func TestRenderTokensText_Good(t *testing.T) {
	tokens := []Token{
		{Text: "Hello"},
		{Text: ", "},
		{Value: "world"}, // falls back to Value when Text empty
	}
	if got := renderTokensText(tokens); got != "Hello, world" {
		t.Fatalf("renderTokensText = %q, want %q", got, "Hello, world")
	}
}

func TestRenderTokensText_Empty_Ugly(t *testing.T) {
	if got := renderTokensText(nil); got != "" {
		t.Fatalf("renderTokensText(nil) = %q, want empty", got)
	}
	if got := renderTokensText([]Token{{}, {}}); got != "" {
		t.Fatalf("renderTokensText(blank tokens) = %q, want empty", got)
	}
}

func TestIndexString_Good(t *testing.T) {
	if got := indexString("alpha/beta", "/"); got != 5 {
		t.Fatalf("indexString found at %d, want 5", got)
	}
	if got := indexString("nohit", "zzz"); got != -1 {
		t.Fatalf("indexString missing = %d, want -1", got)
	}
}

func TestFirstNonEmpty_Good(t *testing.T) {
	if got := firstNonEmpty("", "  ", "kept"); got != "kept" {
		t.Fatalf("firstNonEmpty = %q, want %q", got, "kept")
	}
	// Fast-path: leading ASCII non-whitespace returns immediately.
	if got := firstNonEmpty("url", "fallback"); got != "url" {
		t.Fatalf("firstNonEmpty fast path = %q, want %q", got, "url")
	}
	// Leading whitespace falls through to the unicode-correct branch but is
	// still non-empty after trimming, so it is returned verbatim.
	if got := firstNonEmpty("  spaced", "fallback"); got != "  spaced" {
		t.Fatalf("firstNonEmpty whitespace-lead = %q, want %q", got, "  spaced")
	}
}

func TestFirstNonEmpty_AllBlank_Ugly(t *testing.T) {
	if got := firstNonEmpty("", "   ", "\t\n"); got != "" {
		t.Fatalf("firstNonEmpty(all blank) = %q, want empty", got)
	}
	if got := firstNonEmpty(); got != "" {
		t.Fatalf("firstNonEmpty() = %q, want empty", got)
	}
}

func TestCloneStringMap_Good(t *testing.T) {
	src := map[string]string{"a": "1", "b": "2"}
	out := cloneStringMap(src)
	if len(out) != 2 || out["a"] != "1" || out["b"] != "2" {
		t.Fatalf("cloneStringMap = %v, want copy of %v", out, src)
	}
	out["a"] = "mutated"
	if src["a"] != "1" {
		t.Fatalf("cloneStringMap aliased source: %v", src)
	}
}

func TestCloneStringMap_Empty_Ugly(t *testing.T) {
	if got := cloneStringMap(nil); got != nil {
		t.Fatalf("cloneStringMap(nil) = %v, want nil", got)
	}
	if got := cloneStringMap(map[string]string{}); got != nil {
		t.Fatalf("cloneStringMap(empty) = %v, want nil", got)
	}
}
