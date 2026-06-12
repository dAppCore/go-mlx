// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"strings"
	"testing"
)

// TestSpeculativeServeNotice_NoDraftIsSilent_Good — without --draft there is
// no speculative lane to explain, so the notice is empty and serve prints
// nothing extra.
func TestSpeculativeServeNotice_NoDraftIsSilent_Good(t *testing.T) {
	if got := speculativeServeNotice(""); got != "" {
		t.Fatalf("speculativeServeNotice(\"\") = %q, want empty (no draft → no notice)", got)
	}
	if got := speculativeServeNotice("   "); got != "" {
		t.Fatalf("speculativeServeNotice(blank) = %q, want empty (blank draft → no notice)", got)
	}
}

// TestSpeculativeServeNotice_DraftWarnsGreedyOnlyFallback_Good — with --draft
// set the operator MUST be told the MTP lane is greedy-only and that ordinary
// (sampled) requests fall back to plain decode, so they do not assume the
// loaded drafter is accelerating their traffic. The native MTP path engages
// only for temperature/top_p/top_k all zero, which no default OpenAI client
// sends.
func TestSpeculativeServeNotice_DraftWarnsGreedyOnlyFallback_Good(t *testing.T) {
	got := speculativeServeNotice("/models/gemma-4-E2B-it-assistant-bf16")
	if got == "" {
		t.Fatalf("speculativeServeNotice(draft) = empty, want an advisory notice")
	}
	lower := strings.ToLower(got)
	for _, want := range []string{"greedy", "sampled", "plain"} {
		if !strings.Contains(lower, want) {
			t.Fatalf("notice %q missing %q — operator must learn MTP is inactive for sampled requests", got, want)
		}
	}
}
