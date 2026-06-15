// SPDX-Licence-Identifier: EUPL-1.2

// Tests for generate_options.go — the WithX GenerateOption functional
// options. Each option is pure: it mutates a GenerateConfig in place, so
// the whole observable behaviour is "apply to a default config, read the
// field back". The nil-sink paths are the load-bearing ones — they must
// return the shared no-op option, never wire a sink.

package mlx

import (
	"testing"

	"dappco.re/go/mlx/probe"
)

func applyGenerateOption(opt GenerateOption) GenerateConfig {
	cfg := DefaultGenerateConfig()
	opt(&cfg)
	return cfg
}

func TestGenerateOptions_WithThinkingBudget_Good(t *testing.T) {
	cfg := applyGenerateOption(WithThinkingBudget(128))
	if cfg.ThinkingBudget != 128 {
		t.Fatalf("WithThinkingBudget(128) → ThinkingBudget = %d, want 128", cfg.ThinkingBudget)
	}
}

func TestGenerateOptions_WithThinkingBudget_Ugly(t *testing.T) {
	// Zero means unlimited — the option still writes it (overriding any prior).
	cfg := DefaultGenerateConfig()
	cfg.ThinkingBudget = 99
	WithThinkingBudget(0)(&cfg)
	if cfg.ThinkingBudget != 0 {
		t.Fatalf("WithThinkingBudget(0) → ThinkingBudget = %d, want 0 (unlimited)", cfg.ThinkingBudget)
	}
}

func TestGenerateOptions_WithSuppressTokens_Good(t *testing.T) {
	cfg := applyGenerateOption(WithSuppressTokens(5, 9, 13))
	if len(cfg.SuppressTokens) != 3 || cfg.SuppressTokens[0] != 5 || cfg.SuppressTokens[2] != 13 {
		t.Fatalf("WithSuppressTokens → %v, want [5 9 13]", cfg.SuppressTokens)
	}
}

func TestGenerateOptions_WithSuppressTokens_Ugly(t *testing.T) {
	// No ids: the field is set to an empty (zero-length) slice, no panic.
	cfg := applyGenerateOption(WithSuppressTokens())
	if len(cfg.SuppressTokens) != 0 {
		t.Fatalf("WithSuppressTokens() → %v, want empty", cfg.SuppressTokens)
	}
}

func TestGenerateOptions_WithProbeCallback_Good(t *testing.T) {
	var seen int
	cfg := applyGenerateOption(WithProbeCallback(func(probe.Event) { seen++ }))
	if cfg.ProbeSink == nil {
		t.Fatal("WithProbeCallback(fn) → ProbeSink is nil, want wired")
	}
	cfg.ProbeSink.EmitProbe(probe.Event{})
	if seen != 1 {
		t.Fatalf("wired callback fired %d times, want 1", seen)
	}
}

func TestGenerateOptions_WithProbeCallback_Bad(t *testing.T) {
	// A nil callback must NOT install a sink (the shared no-op option).
	cfg := applyGenerateOption(WithProbeCallback(nil))
	if cfg.ProbeSink != nil {
		t.Fatalf("WithProbeCallback(nil) → ProbeSink = %v, want nil", cfg.ProbeSink)
	}
}

func TestGenerateOptions_WithProbeSink_Good(t *testing.T) {
	var seen int
	sink := probe.SinkFunc(func(probe.Event) { seen++ })
	cfg := applyGenerateOption(WithProbeSink(sink))
	if cfg.ProbeSink == nil {
		t.Fatal("WithProbeSink(sink) → ProbeSink is nil, want wired")
	}
	cfg.ProbeSink.EmitProbe(probe.Event{})
	if seen != 1 {
		t.Fatalf("wired sink fired %d times, want 1", seen)
	}
}

func TestGenerateOptions_WithProbeSink_Bad(t *testing.T) {
	// A nil sink must NOT install one (the shared no-op option).
	cfg := applyGenerateOption(WithProbeSink(nil))
	if cfg.ProbeSink != nil {
		t.Fatalf("WithProbeSink(nil) → ProbeSink = %v, want nil", cfg.ProbeSink)
	}
}
