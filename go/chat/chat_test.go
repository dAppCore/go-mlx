// SPDX-Licence-Identifier: EUPL-1.2

package chat

import "testing"

// --- ConfigForArchitecture ---

func TestChat_ConfigForArchitecture_Good(t *testing.T) {
	// Happy path: a known Gemma 4 large variant derives thinking-on +
	// large-variant-on from the architecture + head count.
	cfg := ConfigForArchitecture("gemma4_text", 16)
	if cfg.Architecture != "gemma4_text" {
		t.Fatalf("ConfigForArchitecture Architecture = %q, want gemma4_text", cfg.Architecture)
	}
	if !cfg.EnableThinking {
		t.Fatalf("ConfigForArchitecture(gemma4_text) EnableThinking = false, want true")
	}
	if !cfg.LargeVariant {
		t.Fatalf("ConfigForArchitecture(gemma4_text, 16) LargeVariant = false, want true")
	}
}

func TestChat_ConfigForArchitecture_Bad(t *testing.T) {
	// Error-shaped input: an unknown architecture has no registry profile,
	// so thinking defaults off and it is never a large variant regardless
	// of head count. The function must not panic or leak a stale default.
	cfg := ConfigForArchitecture("MiniMaxM2ForCausalLM", 64)
	if cfg.Architecture != "MiniMaxM2ForCausalLM" {
		t.Fatalf("ConfigForArchitecture Architecture = %q, want passthrough", cfg.Architecture)
	}
	if cfg.EnableThinking {
		t.Fatalf("ConfigForArchitecture(unknown) EnableThinking = true, want false")
	}
	if cfg.LargeVariant {
		t.Fatalf("ConfigForArchitecture(unknown, 64) LargeVariant = true, want false")
	}
}

func TestChat_ConfigForArchitecture_Ugly(t *testing.T) {
	// Edge cases: a Gemma 4 architecture below the 16-head large-variant
	// threshold is thinking-on but NOT large; and the empty architecture
	// yields the all-false zero-ish config without panicking.
	small := ConfigForArchitecture("gemma4_text", 8)
	if !small.EnableThinking {
		t.Fatalf("ConfigForArchitecture(gemma4_text, 8) EnableThinking = false, want true")
	}
	if small.LargeVariant {
		t.Fatalf("ConfigForArchitecture(gemma4_text, 8) LargeVariant = true, want false (below 16-head gate)")
	}
	empty := ConfigForArchitecture("", 0)
	if empty.Architecture != "" || empty.EnableThinking || empty.LargeVariant {
		t.Fatalf("ConfigForArchitecture(\"\", 0) = %+v, want zero config", empty)
	}
}

// --- Format ---

func TestChat_Format_Good(t *testing.T) {
	// Happy path (moved from TestFormat_PlainTemplate_Good): the plain
	// template emits only content + newline per message, skipping the
	// empty system message and emitting no role.
	got := Format([]Message{
		{Role: "system"},
		{Role: "user", Content: "plain"},
	}, Config{Template: "plain", NoGenerationPrompt: true})
	if got != "plain\n" {
		t.Fatalf("Format plain = %q, want plain only", got)
	}
}

func TestChat_Format_Bad(t *testing.T) {
	// Error-shaped input: an architecture with no registered family
	// formatter falls back to the plain renderer rather than failing or
	// returning empty — the neutral package carries no family list.
	got := Format([]Message{
		{Role: "user", Content: "fallback"},
	}, Config{Architecture: "DeepseekV3ForCausalLM"})
	if got != "fallback\n" {
		t.Fatalf("Format(unregistered arch) = %q, want plain fallback %q", got, "fallback\n")
	}
}

func TestChat_Format_Ugly(t *testing.T) {
	// Edge cases: an empty message list renders to the empty string, and
	// messages whose Content is empty are skipped by the plain renderer
	// (only the non-empty tail survives) — no stray newlines, no panic.
	if got := Format(nil, Config{Template: "plain"}); got != "" {
		t.Fatalf("Format(nil) = %q, want empty string", got)
	}
	got := Format([]Message{
		{Role: "user", Content: ""},
		{Role: "assistant", Content: ""},
		{Role: "user", Content: "only"},
	}, Config{Template: "plain"})
	if got != "only\n" {
		t.Fatalf("Format(empty-content msgs) = %q, want %q", got, "only\n")
	}
}

// --- FormatCapacity ---

func TestChat_FormatCapacity_Good(t *testing.T) {
	// Happy path: with writesRole=false the capacity is exactly the sum of
	// content lengths plus per-message overhead plus the generation-prompt
	// term, with no role reservation.
	messages := []Message{
		{Role: "user", Content: "abcde"},   // 5
		{Role: "assistant", Content: "xy"}, // 2
	}
	got := FormatCapacity(messages, 3, 7, false)
	want := 7 + (5 + 3) + (2 + 3) // gen + (len+overhead) per message
	if got != want {
		t.Fatalf("FormatCapacity(no role) = %d, want %d", got, want)
	}
}

func TestChat_FormatCapacity_Bad(t *testing.T) {
	// Error-shaped input: an aliased role ("gpt") is only maxNormalisedRoleLen
	// wide after expansion. FormatCapacity must reserve maxNormalisedRoleLen
	// (not len("gpt")) when writesRole is true, so the sizing can never
	// under-allocate and trigger a silent Builder realloc.
	messages := []Message{{Role: "gpt", Content: "hi"}} // content len 2
	got := FormatCapacity(messages, 4, 0, true)
	want := 0 + (2 + 4 + maxNormalisedRoleLen)
	if got != want {
		t.Fatalf("FormatCapacity(role) = %d, want %d (must reserve maxNormalisedRoleLen=%d)", got, want, maxNormalisedRoleLen)
	}
	if got < len("hi")+len("assistant") {
		t.Fatalf("FormatCapacity under-allocated for expanded role: got %d", got)
	}
}

func TestChat_FormatCapacity_Ugly(t *testing.T) {
	// Edge cases: an empty message list yields exactly the generation-prompt
	// overhead; the all-zero call yields zero. Neither path should reserve
	// role width for messages that do not exist.
	if got := FormatCapacity(nil, 9, 13, true); got != 13 {
		t.Fatalf("FormatCapacity(nil, _, 13, role) = %d, want 13 (gen overhead only)", got)
	}
	if got := FormatCapacity(nil, 0, 0, false); got != 0 {
		t.Fatalf("FormatCapacity(nil, 0, 0, false) = %d, want 0", got)
	}
}

// --- TemplateName ---

func TestChat_TemplateName_Good(t *testing.T) {
	// Happy path (moved from TestTemplateName_ArchitectureFamilies_Good):
	// architecture names map to their advertised family template; unknown
	// or attached-only families resolve to the empty name (→ plain).
	cases := map[string]string{
		"gemma4_text":                           "gemma4",
		"gemma4_unified":                        "gemma4",
		"Gemma4ForConditionalGeneration":        "gemma4",
		"Gemma4UnifiedForConditionalGeneration": "gemma4",
		"Gemma4ForCausalLM":                     "gemma4",
		"Gemma4TextForCausalLM":                 "gemma4",
		"gemma3":                                "gemma",
		"gemma3_text":                           "gemma",
		"Gemma3ForCausalLM":                     "gemma",
		"qwen3_moe":                             "qwen",
		"qwen3_next":                            "qwen",
		"qwen3_6":                               "qwen",
		"qwen3_6_moe":                           "qwen",
		"Qwen3ForCausalLM":                      "qwen",
		"llama3":                                "llama",
		"LlamaForCausalLM":                      "llama",
		"Gemma4AssistantForCausalLM":            "",
		"MiniMaxM2ForCausalLM":                  "",
		"DeepseekV3ForCausalLM":                 "",
		"unknown":                               "",
		"":                                      "",
	}
	for arch, want := range cases {
		if got := TemplateName(Config{Architecture: arch}); got != want {
			t.Fatalf("TemplateName(%q) = %q, want %q", arch, got, want)
		}
	}
}

func TestChat_TemplateName_Bad(t *testing.T) {
	// Error-shaped input: an architecture with no family and no explicit
	// Template resolves to the empty template name — the signal that Format
	// must use the plain fallback. A purely whitespace Template is trimmed
	// away and does NOT override, so the architecture is consulted.
	if got := TemplateName(Config{Architecture: "NotARealArch"}); got != "" {
		t.Fatalf("TemplateName(unknown arch) = %q, want empty", got)
	}
	if got := TemplateName(Config{Architecture: "gemma3", Template: "   "}); got != "gemma" {
		t.Fatalf("TemplateName(blank Template) = %q, want architecture-derived gemma", got)
	}
}

func TestChat_TemplateName_Ugly(t *testing.T) {
	// Edge cases: an explicit Template overrides the architecture entirely
	// (moved from TestTemplateName_ExplicitOverridesArchitecture_Ugly), and
	// an explicit Template is lower-cased + trimmed before use, so padded
	// mixed-case input still resolves to the canonical lowercase name.
	if got := TemplateName(Config{Architecture: "gemma3", Template: "qwen"}); got != "qwen" {
		t.Fatalf("Template did not override Architecture: got %q", got)
	}
	if got := TemplateName(Config{Architecture: "gemma3", Template: "  QWEN  "}); got != "qwen" {
		t.Fatalf("TemplateName did not normalise padded mixed-case Template: got %q", got)
	}
	if got := TemplateName(Config{}); got != "" {
		t.Fatalf("TemplateName(zero Config) = %q, want empty", got)
	}
}

// --- NormaliseRole ---

func TestChat_NormaliseRole_Good(t *testing.T) {
	// Happy path (moved from TestNormaliseRole_Aliases_Good): every known
	// alias across the HF / ShareGPT / Llama / Gemma variations canonicalises
	// to user / assistant / system; an unknown role and the empty role pass
	// through (empty → empty).
	cases := map[string]string{
		"human":     "user",
		"User":      "user",
		"gpt":       "assistant",
		"bot":       "assistant",
		"Assistant": "assistant",
		"model":     "assistant",
		"developer": "system",
		"system":    "system",
		"unknown":   "unknown",
		"":          "",
	}
	for in, want := range cases {
		if got := NormaliseRole(in); got != want {
			t.Fatalf("NormaliseRole(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestChat_NormaliseRole_Bad(t *testing.T) {
	// Error-shaped input: an unrecognised custom role is not forced into one
	// of the canonical buckets — it is lower-cased and returned verbatim, so
	// "TOOL" becomes "tool" rather than silently collapsing to assistant.
	if got := NormaliseRole("TOOL"); got != "tool" {
		t.Fatalf("NormaliseRole(TOOL) = %q, want lowered passthrough tool", got)
	}
	if got := NormaliseRole("function"); got != "function" {
		t.Fatalf("NormaliseRole(function) = %q, want passthrough function", got)
	}
}

func TestChat_NormaliseRole_Ugly(t *testing.T) {
	// Edge cases: whitespace-padded and mixed-case aliases still resolve via
	// the slow path (Trim + EqualFold) to their canonical literal; an empty
	// and a whitespace-only role both yield the empty string.
	if got := NormaliseRole("  Human  "); got != "user" {
		t.Fatalf("NormaliseRole(padded Human) = %q, want user", got)
	}
	if got := NormaliseRole("DEVELOPER"); got != "system" {
		t.Fatalf("NormaliseRole(DEVELOPER) = %q, want system", got)
	}
	if got := NormaliseRole("   "); got != "" {
		t.Fatalf("NormaliseRole(whitespace) = %q, want empty", got)
	}
}
