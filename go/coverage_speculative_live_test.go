// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"runtime"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/mlx/internal/metaltest"
)

// speculativeE2BAssistantRepo is the cached native MTP assistant drafter that
// pairs with the e2b-4bit target.
const speculativeE2BAssistantRepo = "mlx-community/gemma-4-E2B-it-assistant-bf16"

// requireLiveSpeculativeTextModel loads the e2b target beside its assistant
// drafter as the speculative inference.TextModel — a TWO-model resident load, so
// this is sequenced on its own and never stacked with the shared model. Skips
// (never fails) when Metal is off, either pack is uncached, or the load hits the
// reshape flake.
func requireLiveSpeculativeTextModel(t *testing.T) (*speculativeTextModel, func()) {
	t.Helper()
	if !metaltest.RunMetalTests {
		t.Skip("build with -tags metal_runtime to drive the live speculative pair")
	}
	if !MetalAvailable() {
		t.Skip("Metal runtime unavailable")
	}
	targetDir := metaltest.HFModelPath(t, liveE2BModelRepo)
	draftDir := metaltest.HFModelPath(t, speculativeE2BAssistantRepo)

	// Drop any cached residency before a two-model load to keep the peak down.
	ClearCache()
	runtime.GC()

	tm, err := LoadSpeculativePairAsTextModel(targetDir, draftDir, WithContextLength(2048))
	if err != nil {
		if core.Contains(err.Error(), reshapeFlakeNeedle) {
			t.Skipf("speculative pair load did not survive the reshape flake: %v", err)
		}
		t.Fatalf("LoadSpeculativePairAsTextModel: %v", err)
	}
	spec, ok := tm.(*speculativeTextModel)
	if !ok {
		_ = tm.Close()
		t.Fatalf("LoadSpeculativePairAsTextModel returned %T, want *speculativeTextModel", tm)
	}
	return spec, func() {
		_ = spec.Close()
		ClearCache()
		runtime.GC()
	}
}

// TestSpeculativeTextModel_LivePair drives the native MTP speculative lane on a
// real target+drafter pair: Generate and Chat (the MTP-eligible greedy path),
// the speculativeStream loop, mtpEligible, MTPMetrics, IsSpeculativeTextModel,
// and Close — the speculative_textmodel block that is 0% without the pair.
func TestSpeculativeTextModel_LivePair(t *testing.T) {
	spec, done := requireLiveSpeculativeTextModel(t)
	defer done()
	ctx := context.Background()
	off := false

	if !IsSpeculativeTextModel(spec) {
		t.Fatal("IsSpeculativeTextModel false for the loaded pair")
	}

	// Generate — raw prompt through the MTP lane (greedy => eligible).
	gen := core.NewBuilder()
	for token := range spec.Generate(ctx, "Once upon a time,",
		inference.WithMaxTokens(24), inference.WithEnableThinking(&off)) {
		gen.WriteString(token.Text)
	}
	if err := spec.Err(); err != nil {
		t.Fatalf("Generate Err: %v", err)
	}
	t.Logf("speculative Generate -> %q", gen.String())

	// MTPMetrics — the acceptance counters off the greedy run (may be nil if the
	// lane took a fallback, so only the call is asserted reachable).
	if mtp := spec.MTPMetrics(); mtp != nil {
		t.Logf("MTP metrics: proposed=%d accepted=%d", mtp.ProposedTokens, mtp.AcceptedTokens)
	}

	// The underlying SpeculativePair's Metrics/Err facade (the target's latest
	// counters / generation error for a target+draft pair).
	pairMetrics := spec.pair.Metrics()
	t.Logf("pair metrics: prompt=%d gen=%d", pairMetrics.PromptTokens, pairMetrics.GeneratedTokens)
	if err := spec.pair.Err(); err != nil {
		t.Fatalf("pair Err after a successful generate: %v", err)
	}

	// Chat — formats with the native template then streams via the MTP lane.
	reply := core.NewBuilder()
	for token := range spec.Chat(ctx,
		[]inference.Message{{Role: "user", Content: "Reply with one short sentence about the sea."}},
		inference.WithMaxTokens(32), inference.WithEnableThinking(&off)) {
		reply.WriteString(token.Text)
	}
	if err := spec.Err(); err != nil {
		t.Fatalf("Chat Err: %v", err)
	}
	if reply.Len() == 0 {
		t.Fatal("speculative Chat produced nothing")
	}
	t.Logf("speculative Chat -> %q", reply.String())
}

// TestSpeculativeTextModel_FallbackLane drives the mtpEligible=false fallback:
// a repetition-penalty request takes plain target decode, not the MTP lane, so
// the served model never errors on a request the verify rows can't carry.
func TestSpeculativeTextModel_FallbackLane(t *testing.T) {
	spec, done := requireLiveSpeculativeTextModel(t)
	defer done()
	ctx := context.Background()
	off := false

	// Repetition penalty > 1 => mtpEligible returns false => plain decode lane.
	gen := core.NewBuilder()
	for token := range spec.Generate(ctx, "List three fruits:",
		inference.WithMaxTokens(16),
		inference.WithRepeatPenalty(1.3),
		inference.WithEnableThinking(&off)) {
		gen.WriteString(token.Text)
	}
	if err := spec.Err(); err != nil {
		t.Fatalf("fallback Generate Err: %v", err)
	}
	t.Logf("speculative fallback Generate -> %q", gen.String())
}
