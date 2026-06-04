// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"testing"

	core "dappco.re/go"
	state "dappco.re/go/inference/state"
)

type stateRampProfileSeedFakeTokenizer struct{}

func (stateRampProfileSeedFakeTokenizer) Encode(text string) ([]int32, error) {
	tokens := make([]int32, 0, len(text))
	for _, r := range text {
		tokens = append(tokens, int32(r))
	}
	return tokens, nil
}

func (stateRampProfileSeedFakeTokenizer) Decode(tokens []int32) (string, error) {
	runes := make([]rune, len(tokens))
	for i, token := range tokens {
		runes[i] = rune(token)
	}
	return string(runes), nil
}

func TestStateRampProfileOpenFoldStore_AppendsExisting_Good(t *testing.T) {
	ctx := context.Background()
	path := core.PathJoin(t.TempDir(), "state.mvlog")
	first, action, err := stateRampProfileOpenFoldStore(ctx, path)
	if err != nil {
		t.Fatalf("stateRampProfileOpenFoldStore(create): %v", err)
	}
	if action != "create" {
		t.Fatalf("first action = %q, want create", action)
	}
	if _, err := first.Put(ctx, "checkpoint marker", state.PutOptions{URI: "mlx://state/checkpoint"}); err != nil {
		t.Fatalf("first.Put: %v", err)
	}
	if err := first.Close(); err != nil {
		t.Fatalf("first.Close: %v", err)
	}

	second, action, err := stateRampProfileOpenFoldStore(ctx, path)
	if err != nil {
		t.Fatalf("stateRampProfileOpenFoldStore(append): %v", err)
	}
	defer second.Close()
	if action != "append" {
		t.Fatalf("second action = %q, want append", action)
	}
	chunk, err := state.ResolveURI(ctx, second, "mlx://state/checkpoint")
	if err != nil {
		t.Fatalf("ResolveURI(checkpoint): %v", err)
	}
	if chunk.Text != "checkpoint marker" {
		t.Fatalf("checkpoint text = %q, want preserved marker", chunk.Text)
	}
	ref, err := second.Put(ctx, "folded marker", state.PutOptions{URI: "mlx://state/folded"})
	if err != nil {
		t.Fatalf("second.Put: %v", err)
	}
	if ref.ChunkID != 2 {
		t.Fatalf("appended chunk id = %d, want next id 2", ref.ChunkID)
	}
}

func TestStateRampProfileSeedTokens_RepeatsSourceForWrappedTemplate_Good(t *testing.T) {
	got, err := stateRampProfileSeedTokens(stateRampProfileSeedFakeTokenizer{}, []int32{'a', 'b', 'c'}, stateRampProfileOptions{
		ChatTemplate: "custom-wrapper",
		StartTokens:  7,
	})
	if err != nil {
		t.Fatalf("stateRampProfileSeedTokens: %v", err)
	}
	want := []int32{'a', 'b', 'c', 'a', 'b', 'c', 'a'}
	if len(got) != len(want) {
		t.Fatalf("seed len = %d, want %d (%v)", len(got), len(want), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("seed[%d] = %d, want %d", i, got[i], want[i])
		}
	}
}

func TestStateRampProfileInitialPrompt_RetainedSystemPrompt_Good(t *testing.T) {
	for _, template := range []string{"gemma4", "gemma", "qwen", "llama"} {
		prompt := stateRampProfileInitialPrompt(template, "context body", false)
		if !core.Contains(prompt, defaultStateRampRetainedSystemPrompt) {
			t.Fatalf("template %q prompt = %q, want retained system prompt", template, prompt)
		}
		if core.Contains(prompt, "opencode-style engineering session") || core.Contains(prompt, "later engineering turns") {
			t.Fatalf("template %q prompt = %q, want Lemma retained context language", template, prompt)
		}
	}
}

func TestStateRampProfileGeneratedSummaryError_BadOutputIssues(t *testing.T) {
	err := stateRampProfileGeneratedSummaryError(stateRampProfileTurn{
		OutputIssues: []string{"visible_prompt_analysis"},
	}, "- summary")
	if err == nil || !core.Contains(err.Error(), "generated folded summary has output issues") {
		t.Fatalf("stateRampProfileGeneratedSummaryError() = %v, want output issue error", err)
	}
}
