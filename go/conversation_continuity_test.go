// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package mlx

import (
	"testing"

	"dappco.re/go/inference"
)

func TestConversationTurnSplit_Good(t *testing.T) {
	cases := []struct {
		name     string
		messages []inference.Message
		want     int
	}{
		{"first turn", []inference.Message{{Role: "user", Content: "hi"}}, 0},
		{"second turn", []inference.Message{
			{Role: "user", Content: "hi"},
			{Role: "assistant", Content: "hello"},
			{Role: "user", Content: "and?"},
		}, 2},
		{"trailing tool result rides the new turn", []inference.Message{
			{Role: "user", Content: "hi"},
			{Role: "assistant", Content: "hello"},
			{Role: "user", Content: "run it"},
			{Role: "tool", Content: "ok"},
		}, 2},
		{"system plus first user", []inference.Message{
			{Role: "system", Content: "be brief"},
			{Role: "user", Content: "hi"},
		}, 1},
	}
	for _, tc := range cases {
		if got := conversationTurnSplit(tc.messages); got != tc.want {
			t.Errorf("%s: split = %d, want %d", tc.name, got, tc.want)
		}
	}
}

func TestConversationTurnSplit_Bad(t *testing.T) {
	// A request with no trailing user turn is not turn-shaped: split equals
	// the full length and the manager declines it.
	messages := []inference.Message{
		{Role: "user", Content: "hi"},
		{Role: "assistant", Content: "hello"},
	}
	if got := conversationTurnSplit(messages); got != len(messages) {
		t.Fatalf("split = %d, want %d (decline)", got, len(messages))
	}
}

func TestConversationKey_ChainInvariant_Good(t *testing.T) {
	// The key a finished turn stores under (conversation + its reply) must be
	// the key the NEXT request's prefix hashes to — the lookup chain.
	turn1 := []inference.Message{{Role: "user", Content: "tell me about the keeper"}}
	reply := " His name was Snider."
	stored := conversationKey(append(append([]inference.Message{}, turn1...), inference.Message{Role: "assistant", Content: reply}))

	turn2 := []inference.Message{
		{Role: "user", Content: "tell me about the keeper"},
		{Role: "assistant", Content: reply},
		{Role: "user", Content: "and his lamp?"},
	}
	lookup := conversationKey(turn2[:conversationTurnSplit(turn2)])
	if lookup != stored {
		t.Fatalf("lookup key %q != stored key %q", lookup, stored)
	}
}

func TestConversationKey_RoleAliases_Good(t *testing.T) {
	// Role aliases normalise before hashing, so a client that says "model"
	// where another says "assistant" still finds the same conversation.
	a := conversationKey([]inference.Message{{Role: "assistant", Content: "x"}})
	b := conversationKey([]inference.Message{{Role: "model", Content: "x"}})
	if a != b {
		t.Fatalf("role-alias keys differ: %q vs %q", a, b)
	}
}

func TestConversationKey_ContentSensitivity_Ugly(t *testing.T) {
	// Different content or role/content boundary placement must never
	// collide: the separators keep ("ab","c") distinct from ("a","bc").
	a := conversationKey([]inference.Message{{Role: "user", Content: "ab"}, {Role: "user", Content: "c"}})
	b := conversationKey([]inference.Message{{Role: "user", Content: "a"}, {Role: "user", Content: "bc"}})
	if a == b {
		t.Fatalf("boundary collision: %q", a)
	}
}
