// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

// TestGenerationTokenBudget_DerivesFromContext_Good pins the generation-length
// contract: an explicit MaxTokens (>0) is the caller's word and is honoured
// as-is; MaxTokens <= 0 means "generate to the model's context" and resolves to
// the room left in the context window (contextLength - promptLen), so the loop
// runs until EOS/stop or the context is full — never a hardcoded cap. When the
// prompt already fills the context, or no context is known, the budget is 0
// (nothing to generate / cannot bound) rather than a guessed number.
func TestGenerationTokenBudget_DerivesFromContext_Good(t *testing.T) {
	cases := []struct {
		name                                string
		maxTokens, contextLength, promptLen int
		want                                int
	}{
		{"explicit request honoured", 128, 4096, 10, 128},
		{"unset derives remaining context", 0, 4096, 100, 3996},
		{"negative derives remaining context", -1, 4096, 100, 3996},
		{"prompt fills context leaves no room", 0, 4096, 4096, 0},
		{"prompt exceeds context leaves no room", 0, 4096, 5000, 0},
		{"unset with unknown context cannot bound", 0, 0, 10, 0},
		{"explicit honoured even past context", 9000, 4096, 10, 9000},
	}
	for _, c := range cases {
		if got := generationTokenBudget(c.maxTokens, c.contextLength, c.promptLen); got != c.want {
			t.Fatalf("%s: generationTokenBudget(%d, %d, %d) = %d, want %d", c.name, c.maxTokens, c.contextLength, c.promptLen, got, c.want)
		}
	}
}
