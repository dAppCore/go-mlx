// SPDX-Licence-Identifier: EUPL-1.2

package spine

import (
	"testing"

	"dappco.re/go/inference/parser"
)

// Tests for the root GenerateConfig defaults + option folding. These are
// no-invalid-input constructors, so the menu is: Good asserts the
// documented contract; Ugly covers the empty/nil option stack (the
// boundary the fold must tolerate). No manufactured Bad — there is no
// invalid input to a defaults builder.

func TestSpine_DefaultGenerateConfig_Good(t *testing.T) {
	cfg := DefaultGenerateConfig()
	// Documented contract: 0 = generate to the context window (never a
	// fixed cap), greedy default temperature, thinking shown.
	if cfg.MaxTokens != 0 {
		t.Fatalf("MaxTokens = %d, want 0 (resolve-to-context default)", cfg.MaxTokens)
	}
	if cfg.Temperature != 0.0 {
		t.Fatalf("Temperature = %v, want 0.0 (greedy)", cfg.Temperature)
	}
	if cfg.Thinking.Mode != parser.Show {
		t.Fatalf("Thinking.Mode = %q, want %q", cfg.Thinking.Mode, parser.Show)
	}
}

func TestSpine_ApplyGenerateOptions_Good(t *testing.T) {
	// A real option that mutates two fields must fold over the defaults,
	// leaving untouched fields at their default.
	opts := []GenerateOption{
		func(c *GenerateConfig) { c.MaxTokens = 256 },
		func(c *GenerateConfig) { c.Temperature = 0.7 },
	}
	cfg := ApplyGenerateOptions(opts)
	if cfg.MaxTokens != 256 {
		t.Fatalf("MaxTokens = %d, want 256 (option applied)", cfg.MaxTokens)
	}
	if cfg.Temperature != 0.7 {
		t.Fatalf("Temperature = %v, want 0.7 (option applied)", cfg.Temperature)
	}
	// Untouched field keeps the default.
	if cfg.Thinking.Mode != parser.Show {
		t.Fatalf("Thinking.Mode = %q, want default %q", cfg.Thinking.Mode, parser.Show)
	}
}

func TestSpine_ApplyGenerateOptions_Ugly(t *testing.T) {
	// Empty and nil option stacks must yield the bare defaults. GenerateConfig
	// carries slice fields so it is not == comparable; assert the documented
	// default fields instead.
	want := DefaultGenerateConfig()
	for _, cfg := range []GenerateConfig{
		ApplyGenerateOptions(nil),
		ApplyGenerateOptions([]GenerateOption{}),
	} {
		if cfg.MaxTokens != want.MaxTokens || cfg.Temperature != want.Temperature || cfg.Thinking.Mode != want.Thinking.Mode {
			t.Fatalf("empty options = %+v, want defaults %+v", cfg, want)
		}
	}
	// Later option overrides an earlier one — fold order is preserved.
	cfg := ApplyGenerateOptions([]GenerateOption{
		func(c *GenerateConfig) { c.MaxTokens = 1 },
		func(c *GenerateConfig) { c.MaxTokens = 99 },
	})
	if cfg.MaxTokens != 99 {
		t.Fatalf("MaxTokens = %d, want 99 (last option wins)", cfg.MaxTokens)
	}
}
