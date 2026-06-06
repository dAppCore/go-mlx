// SPDX-Licence-Identifier: EUPL-1.2

package chat

import "testing"

// The registry is how a model family's chat package contributes its formatter
// without the neutral chat package naming the family. Format dispatches on the
// resolved template name; an unregistered name falls back to the plain renderer.

func TestRegisterFormatter_DispatchesByTemplateName_Good(t *testing.T) {
	RegisterFormatter("testfmt", func(messages []Message, _ Config) string {
		return "FMT:" + messages[0].Content
	})
	got := Format([]Message{{Role: "user", Content: "x"}}, Config{Template: "testfmt"})
	if got != "FMT:x" {
		t.Fatalf("registry dispatch = %q, want %q", got, "FMT:x")
	}
}

func TestRegisterFormatter_UnregisteredFallsBackToPlain_Good(t *testing.T) {
	got := Format([]Message{{Role: "user", Content: "hi"}}, Config{Template: "nope-unregistered", NoGenerationPrompt: true})
	if got != "hi\n" {
		t.Fatalf("unregistered template = %q, want plain %q", got, "hi\n")
	}
}
