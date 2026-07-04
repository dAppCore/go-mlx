// SPDX-Licence-Identifier: EUPL-1.2

// Coverage-closing tests for adapter.go — the three statement blocks the
// Good/Bad/Ugly triplets in adapter_test.go leave unexecuted:
//
//   - GenerateStream / ChatStream nil-context defaulting. Every streaming
//     triplet passes context.Background(), so the `if ctx == nil` guard that
//     substitutes context.Background never fires through those tests. The
//     buffered Generate/Chat Ugly cases already drive a nil context; the
//     streaming pair did not, so they get a dedicated _NilContext case each.
//   - genOptsToInference's temperature-only branch. The buffered Good cases
//     pass MaxTokens (and sometimes Temp), exercising the two-option and
//     max-only branches; nothing passed Temp > 0 with MaxTokens == 0, so the
//     `case hasTemp:` arm was never selected. A Generate/Chat call with
//     GenOpts{Temp: ...} and no MaxTokens drives it behaviourally.
//
// Distinct descriptive names rather than another Good/Bad/Ugly suffix: the
// triplet slots for these symbols are already occupied in adapter_test.go, and
// this file stays a separate compilation unit so it can be committed in
// isolation. It reuses stubTextModel from adapter_test.go (same package).
//
// External test package — exercises the exported surface exactly as a LEM
// consumer driving adapter.New(model, "mlx") does.

package adapter_test

import (
	"context"
	"testing"

	"dappco.re/go/inference"
	"dappco.re/go/mlx/adapter"
)

// --- GenerateStream nil context ----------------------------------------

func TestAdapter_Adapter_GenerateStream_NilContext(t *testing.T) {
	// A nil context must be tolerated: the guard substitutes
	// context.Background and the stream still drains every token. This is the
	// one streaming path the Good/Bad/Ugly cases never take — they all pass an
	// explicit context.Background().
	model := &stubTextModel{tokens: []inference.Token{{Text: "a"}, {Text: "b"}, {Text: "c"}}}
	a := adapter.New(model, "mlx")
	var got string
	//nolint:staticcheck // SA1012: passing a nil context is the exact path under test.
	err := a.GenerateStream(nil, "ignored", adapter.GenOpts{MaxTokens: 8}, func(token string) error {
		got += token
		return nil
	})
	if err != nil {
		t.Fatalf("GenerateStream(nil ctx) error = %v, want nil", err)
	}
	if got != "abc" {
		t.Fatalf("GenerateStream(nil ctx) streamed %q, want abc", got)
	}
}

// --- ChatStream nil context --------------------------------------------

func TestAdapter_Adapter_ChatStream_NilContext(t *testing.T) {
	// The chat-side mirror of the GenerateStream nil-context case: a nil
	// context is replaced by context.Background and the chat stream drains in
	// full.
	model := &stubTextModel{chatTokens: []inference.Token{{Text: "x"}, {Text: "y"}, {Text: "z"}}}
	a := adapter.New(model, "mlx")
	var got string
	//nolint:staticcheck // SA1012: passing a nil context is the exact path under test.
	err := a.ChatStream(nil, []inference.Message{{Role: "user", Content: "hi"}}, adapter.GenOpts{}, func(token string) error {
		got += token
		return nil
	})
	if err != nil {
		t.Fatalf("ChatStream(nil ctx) error = %v, want nil", err)
	}
	if got != "xyz" {
		t.Fatalf("ChatStream(nil ctx) streamed %q, want xyz", got)
	}
}

// --- genOptsToInference temperature-only branch ------------------------

func TestAdapter_genOptsToInference_TempOnly(t *testing.T) {
	// GenOpts with Temp > 0 and MaxTokens == 0 selects the `case hasTemp:`
	// arm of genOptsToInference's 2x2 switch — the only branch the buffered
	// Good cases (which always carry MaxTokens) never reach. genOptsToInference
	// is unexported, so it is driven behaviourally: a successful Generate that
	// yields its stream proves the temperature-only option slice was built and
	// forwarded without error.
	model := &stubTextModel{tokens: []inference.Token{{Text: "warm"}}}
	a := adapter.New(model, "mlx")
	result, err := a.Generate(context.Background(), "ignored", adapter.GenOpts{Temp: 0.7})
	if err != nil {
		t.Fatalf("Generate(Temp-only) error = %v, want nil", err)
	}
	if result.Text != "warm" {
		t.Fatalf("Generate(Temp-only).Text = %q, want warm", result.Text)
	}
}

func TestAdapter_genOptsToInference_TempOnly_Chat(t *testing.T) {
	// The chat path drives the same temperature-only branch, guarding against a
	// future refactor that might give Chat its own option construction. A
	// successful drain confirms the branch is forwarded cleanly on both sides.
	model := &stubTextModel{chatTokens: []inference.Token{{Text: "warm chat"}}}
	a := adapter.New(model, "mlx")
	result, err := a.Chat(context.Background(), []inference.Message{{Role: "user", Content: "hi"}}, adapter.GenOpts{Temp: 0.7})
	if err != nil {
		t.Fatalf("Chat(Temp-only) error = %v, want nil", err)
	}
	if result.Text != "warm chat" {
		t.Fatalf("Chat(Temp-only).Text = %q, want warm chat", result.Text)
	}
}
