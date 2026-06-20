// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/metal"
)

// Functional-arm coverage for the assistant decode paths the existing
// speculative-decode tests reach only on the default (no-softcap, dense,
// all-accept / full-reject) shape: the final-logit softcap arms, the sampled
// drafting block (dense + ordered), the ordered greedy block step, the
// suppressed-draft-token replacement, and the partial-accept verify branch.
// All drive the tiny synthetic pair (no real checkpoint), per the AX-11 rule.

// TestGemma4AssistantDecode_DraftStep_Softcap drives the FinalLogitSoftcapping
// arm of DraftStep: a non-zero softcap on the assistant config routes the
// drafter logits through logitSoftcap before the argmax. The softcap is set on
// the loaded config in place (the loader leaves it 0 for the tiny fixture), so
// this isolates the softcap branch without a bespoke on-disk config.
func TestGemma4AssistantDecode_DraftStep_Softcap(t *testing.T) {
	requireMetalRuntime(t)

	pair := loadTinyGemma4AssistantPair(t, false)
	defer pair.Close()
	pair.Assistant.Cfg.FinalLogitSoftcapping = 30.0

	caches := pair.Target.NewCache()
	defer metal.FreeCaches(caches)
	prefillLogits, previousHidden := prefillTinyGemma4AssistantTarget(t, pair, caches, []int32{1, 2, 3})
	defer metal.Free(prefillLogits, previousHidden)

	result, err := pair.DraftStep(3, previousHidden, caches)
	if err != nil {
		t.Fatalf("DraftStep(softcap): %v", err)
	}
	defer result.Close()
	if err := metal.Eval(result.Logits, result.Token, result.Hidden); err != nil {
		t.Fatalf("Eval softcap DraftStep result: %v", err)
	}
	assertShape(t, "softcap logits", result.Logits, []int32{1, 1, 10})
	// A softcap of 30 bounds every logit into (-30, 30) via the tanh squash.
	for _, v := range result.Logits.Floats() {
		if v <= -30 || v >= 30 {
			t.Fatalf("softcapped logit %g escaped the (-30, 30) bound", v)
		}
	}
}

// TestGemma4AssistantDecode_DraftBlock_Softcap drives the softcap arm reached
// through the greedy block step (draftStepGreedy): chaining a softcapped block
// still yields the requested number of bounded-logit draft tokens.
func TestGemma4AssistantDecode_DraftBlock_Softcap(t *testing.T) {
	requireMetalRuntime(t)

	pair := loadTinyGemma4AssistantPair(t, false)
	defer pair.Close()
	pair.Assistant.Cfg.FinalLogitSoftcapping = 30.0

	caches := pair.Target.NewCache()
	defer metal.FreeCaches(caches)
	prefillLogits, previousHidden := prefillTinyGemma4AssistantTarget(t, pair, caches, []int32{1, 2, 3})
	defer metal.Free(prefillLogits, previousHidden)

	block, err := pair.DraftBlock(3, previousHidden, caches, 2)
	if err != nil {
		t.Fatalf("DraftBlock(softcap): %v", err)
	}
	defer block.Close()
	if len(block.Tokens) != 2 {
		t.Fatalf("DraftBlock(softcap) tokens = %v, want 2", block.Tokens)
	}
	for _, id := range block.Tokens {
		if id < 0 || id >= 10 {
			t.Fatalf("draft token %d out of vocab range [0,10)", id)
		}
	}
}

// TestGemma4AssistantDecode_DraftBlockSampled_Dense drives draftBlockSampled
// (and thereby draftStepSampled) on the dense drafter: each draft token is
// SAMPLED from the per-step logits rather than argmaxed. A deterministic uniform
// stub keeps the draws reproducible. The result carries the requested token
// count and the final projected hidden.
func TestGemma4AssistantDecode_DraftBlockSampled_Dense(t *testing.T) {
	requireMetalRuntime(t)

	pair := loadTinyGemma4AssistantPair(t, false)
	defer pair.Close()

	caches := pair.Target.NewCache()
	defer metal.FreeCaches(caches)
	prefillLogits, previousHidden := prefillTinyGemma4AssistantTarget(t, pair, caches, []int32{1, 2, 3})
	defer metal.Free(prefillLogits, previousHidden)

	cfg := metal.GenerateConfig{Temperature: 0.8, TopP: 0.95}
	uniform := newDeterministicUniform()
	block, err := pair.draftBlockSampled(3, previousHidden, caches, 2, nil, cfg, uniform)
	if err != nil {
		t.Fatalf("draftBlockSampled(dense): %v", err)
	}
	defer block.Close()
	if len(block.Tokens) != 2 {
		t.Fatalf("draftBlockSampled tokens = %v, want 2 sampled tokens", block.Tokens)
	}
	for _, id := range block.Tokens {
		if id < 0 || id >= 10 {
			t.Fatalf("sampled draft token %d out of vocab range [0,10)", id)
		}
	}
	assertShape(t, "sampled block hidden", block.Hidden, []int32{1, 1, 8})
}

// TestGemma4AssistantDecode_DraftBlockSampled_Softcap drives the softcap arm of
// draftStepSampled: with a softcap set, the dense sampled logits are squashed
// before sampling, and the block still proposes the requested count.
func TestGemma4AssistantDecode_DraftBlockSampled_Softcap(t *testing.T) {
	requireMetalRuntime(t)

	pair := loadTinyGemma4AssistantPair(t, false)
	defer pair.Close()
	pair.Assistant.Cfg.FinalLogitSoftcapping = 30.0

	caches := pair.Target.NewCache()
	defer metal.FreeCaches(caches)
	prefillLogits, previousHidden := prefillTinyGemma4AssistantTarget(t, pair, caches, []int32{1, 2, 3})
	defer metal.Free(prefillLogits, previousHidden)

	cfg := metal.GenerateConfig{Temperature: 0.7, TopP: 0.9}
	block, err := pair.draftBlockSampled(3, previousHidden, caches, 1, nil, cfg, newDeterministicUniform())
	if err != nil {
		t.Fatalf("draftBlockSampled(softcap): %v", err)
	}
	defer block.Close()
	if len(block.Tokens) != 1 {
		t.Fatalf("draftBlockSampled(softcap) tokens = %v, want 1", block.Tokens)
	}
}

// TestGemma4AssistantDecode_DraftBlockSampled_Ordered drives the ordered-
// embedding arm of draftStepSampled: an ordered drafter's sparse candidate
// logits are scattered into a dense vocab vector (orderedEmbeddingDenseLogits)
// so the sampling path consumes a full distribution. The ordered pair has no
// other in-package sampled driver.
func TestGemma4AssistantDecode_DraftBlockSampled_Ordered(t *testing.T) {
	requireMetalRuntime(t)

	pair := loadTinyGemma4AssistantPair(t, true)
	defer pair.Close()
	if !pair.Assistant.UseOrderedEmbeddings {
		t.Fatal("ordered pair did not load ordered embeddings")
	}

	caches := pair.Target.NewCache()
	defer metal.FreeCaches(caches)
	prefillLogits, previousHidden := prefillTinyGemma4AssistantTarget(t, pair, caches, []int32{1, 2, 3})
	defer metal.Free(prefillLogits, previousHidden)

	cfg := metal.GenerateConfig{Temperature: 0.8, TopP: 0.95}
	block, err := pair.draftBlockSampled(3, previousHidden, caches, 2, nil, cfg, newDeterministicUniform())
	if err != nil {
		t.Fatalf("draftBlockSampled(ordered): %v", err)
	}
	defer block.Close()
	if len(block.Tokens) != 2 {
		t.Fatalf("draftBlockSampled(ordered) tokens = %v, want 2", block.Tokens)
	}
	for _, id := range block.Tokens {
		if id < 0 || id >= 10 {
			t.Fatalf("ordered sampled draft token %d out of vocab range [0,10)", id)
		}
	}
}

// TestGemma4AssistantDecode_DraftBlock_Ordered drives the ordered greedy block
// step (draftStepGreedy's ordered arm): a greedy DraftBlock on an ordered pair
// routes each step through orderedEmbeddingGreedyToken rather than the dense
// argmax. This is the greedy counterpart to the sampled-ordered test above.
func TestGemma4AssistantDecode_DraftBlock_Ordered(t *testing.T) {
	requireMetalRuntime(t)

	pair := loadTinyGemma4AssistantPair(t, true)
	defer pair.Close()

	caches := pair.Target.NewCache()
	defer metal.FreeCaches(caches)
	prefillLogits, previousHidden := prefillTinyGemma4AssistantTarget(t, pair, caches, []int32{1, 2, 3})
	defer metal.Free(prefillLogits, previousHidden)

	block, err := pair.DraftBlock(3, previousHidden, caches, 2)
	if err != nil {
		t.Fatalf("DraftBlock(ordered): %v", err)
	}
	defer block.Close()
	if len(block.Tokens) != 2 {
		t.Fatalf("DraftBlock(ordered) tokens = %v, want 2", block.Tokens)
	}
	assertShape(t, "ordered block hidden", block.Hidden, []int32{1, 1, 8})
}

// TestGemma4AssistantDecode_DraftStepToken_NoToken covers the nil-token guard of
// gemma4AssistantDraftStepToken: a step with no Token array yields the
// no-token error.
func TestGemma4AssistantDecode_DraftStepToken_NoToken(t *testing.T) {
	if _, err := gemma4AssistantDraftStepToken(&Gemma4AssistantDraftStepResult{}, nil); err == nil {
		t.Fatal("draftStepToken(no token) error = nil, want no-token error")
	}
	if _, err := gemma4AssistantDraftStepToken(nil, nil); err == nil {
		t.Fatal("draftStepToken(nil) error = nil, want no-token error")
	}
}

// TestGemma4AssistantDecode_DraftStepToken_Replacement covers the suppressed-id
// replacement arm: when the drafter's greedy token is in the suppress list and
// the step carries logits, the function resamples a non-suppressed greedy pick
// and swaps step.Token to the replacement. Token 2 is the raw pick; suppressing
// it forces the next-highest unsuppressed id.
func TestGemma4AssistantDecode_DraftStepToken_Replacement(t *testing.T) {
	requireMetalRuntime(t)

	// Logits peak at column 2; column 1 is the runner-up. Token says 2.
	logits := metal.FromValues([]float32{0.1, 5.0, 9.0, 0.2}, 1, 1, 4)
	token := metal.FromSingleInt32Matrix(2)
	if err := metal.Eval(logits, token); err != nil {
		metal.Free(logits, token)
		t.Fatalf("eval logits/token: %v", err)
	}
	step := &Gemma4AssistantDraftStepResult{Logits: logits, Token: token}
	defer step.Close()

	got, err := gemma4AssistantDraftStepToken(step, []int32{2})
	if err != nil {
		t.Fatalf("draftStepToken(replacement): %v", err)
	}
	if got == 2 {
		t.Fatal("draftStepToken returned the suppressed id 2, want a replacement")
	}
	if got != 1 {
		t.Fatalf("replacement = %d, want runner-up unsuppressed id 1", got)
	}
	// The step's Token must have been swapped to the replacement.
	if vals := step.Token.DataInt32(); len(vals) == 0 || vals[0] != got {
		t.Fatalf("step.Token = %v, want swapped to replacement %d", step.Token.DataInt32(), got)
	}
}

// TestGemma4AssistantDecode_DraftStepToken_AllMuted covers the muted arm: the
// drafter's greedy token is suppressed but the step carries no logits to
// resample from, so the function reports every candidate muted.
func TestGemma4AssistantDecode_DraftStepToken_AllMuted(t *testing.T) {
	requireMetalRuntime(t)

	token := metal.FromSingleInt32Matrix(2)
	if err := metal.Eval(token); err != nil {
		metal.Free(token)
		t.Fatalf("eval token: %v", err)
	}
	step := &Gemma4AssistantDraftStepResult{Token: token} // no Logits
	defer step.Close()

	if _, err := gemma4AssistantDraftStepToken(step, []int32{2}); err == nil {
		t.Fatal("draftStepToken(all muted) error = nil, want all-candidates-muted error")
	}
}

// TestGemma4AssistantDecode_VerifyDraftBlock_PartialAccept drives the partial-
// accept branch of VerifyDraftBlockWithSuppression: a two-token draft whose
// first token matches the target's greedy pick and whose second does not yields
// one accepted token, one rejected token, the replacement at the rejection
// position, and the carried hidden after the accepted prefix (k>0). This branch
// is reached by neither the all-accepted (single matching token) nor the
// full-reject (single bad token) existing tests.
func TestGemma4AssistantDecode_VerifyDraftBlock_PartialAccept(t *testing.T) {
	requireMetalRuntime(t)

	pair := loadTinyGemma4AssistantPair(t, false)
	defer pair.Close()
	caches := pair.Target.NewCache()
	defer metal.FreeCaches(caches)
	prefillLogits, previousHidden := prefillTinyGemma4AssistantTarget(t, pair, caches, []int32{1, 2, 3})
	defer metal.Free(prefillLogits, previousHidden)

	targetToken, err := gemma4AssistantGreedyToken(prefillLogits)
	if err != nil {
		t.Fatalf("greedy target token: %v", err)
	}
	badSecond := (targetToken + 1) % 10

	result, err := pair.VerifyDraftBlock(prefillLogits, []int32{targetToken, badSecond}, caches)
	if err != nil {
		t.Fatalf("VerifyDraftBlock(partial): %v", err)
	}
	defer result.Close()
	if result.AllAccepted {
		t.Fatal("partial verify reported AllAccepted, want a rejection")
	}
	if result.AcceptedCount != 1 || result.RejectedCount != 1 {
		t.Fatalf("accepted/rejected = %d/%d, want 1/1", result.AcceptedCount, result.RejectedCount)
	}
	if len(result.AcceptedTokens) != 1 || result.AcceptedTokens[0] != targetToken {
		t.Fatalf("accepted = %v, want [%d]", result.AcceptedTokens, targetToken)
	}
	if len(result.RejectedTokens) != 1 || result.RejectedTokens[0] != badSecond {
		t.Fatalf("rejected = %v, want [%d]", result.RejectedTokens, badSecond)
	}
	// k>0, so the hidden after the last accepted token is carried forward.
	assertShape(t, "partial verify hidden", result.Hidden, []int32{1, 1, 8})
	assertShape(t, "partial verify logits", result.Logits, []int32{1, 1, 10})
}

// TestGemma4AssistantDecode_VerifyDraftBlock_PartialAcceptSuppressed drives the
// same partial-accept branch through the suppression variant: the suppress list
// is applied to both the incoming and verify-row greedy picks before acceptance.
// Suppressing an unrelated id leaves the accept/reject decision intact while
// exercising the in-graph suppression arm of the batched greedy graph on the
// partial-accept path.
func TestGemma4AssistantDecode_VerifyDraftBlock_PartialAcceptSuppressed(t *testing.T) {
	requireMetalRuntime(t)

	pair := loadTinyGemma4AssistantPair(t, false)
	defer pair.Close()
	caches := pair.Target.NewCache()
	defer metal.FreeCaches(caches)
	prefillLogits, previousHidden := prefillTinyGemma4AssistantTarget(t, pair, caches, []int32{1, 2, 3})
	defer metal.Free(prefillLogits, previousHidden)

	// Find the suppressed greedy pick so the draft tracks it: with id 0 masked,
	// the target's greedy pick over the suppressed logits is the accepted token.
	suppress := []int32{0}
	targetToken, err := gemma4AssistantGreedyToken(prefillLogits, suppress)
	if err != nil {
		t.Fatalf("suppressed greedy target token: %v", err)
	}
	badSecond := targetToken + 1
	if badSecond == 0 || metal.TokenIDSuppressed(badSecond, suppress) {
		badSecond = (targetToken + 2) % 10
	}

	result, err := pair.VerifyDraftBlockWithSuppression(prefillLogits, []int32{targetToken, badSecond}, caches, suppress)
	if err != nil {
		t.Fatalf("VerifyDraftBlockWithSuppression(partial): %v", err)
	}
	defer result.Close()
	if result.AcceptedCount != 1 || result.RejectedCount != 1 {
		t.Fatalf("suppressed accepted/rejected = %d/%d, want 1/1", result.AcceptedCount, result.RejectedCount)
	}
	if result.ReplacementToken == 0 {
		t.Fatal("replacement token = 0, want a non-suppressed replacement")
	}
}

// TestGemma4AssistantDecode_VerifyDraftBlock_NeedDraftTokens covers the
// empty-draft guard of VerifyDraftBlockWithSuppression: a valid target + caches
// but an empty draft slice is rejected before the clone.
func TestGemma4AssistantDecode_VerifyDraftBlock_NeedDraftTokens(t *testing.T) {
	requireMetalRuntime(t)

	pair := loadTinyGemma4AssistantPair(t, false)
	defer pair.Close()
	caches := pair.Target.NewCache()
	defer metal.FreeCaches(caches)
	prefillLogits, previousHidden := prefillTinyGemma4AssistantTarget(t, pair, caches, []int32{1, 2, 3})
	defer metal.Free(prefillLogits, previousHidden)

	if _, err := pair.VerifyDraftBlock(prefillLogits, nil, caches); err == nil {
		t.Fatal("VerifyDraftBlock(no drafts) error = nil, want need-draft-tokens error")
	} else if !core.Contains(err.Error(), "draft tokens") {
		t.Fatalf("VerifyDraftBlock(no drafts) error = %v, want draft tokens", err)
	}
}

// newDeterministicUniform returns a uniform() stub cycling a fixed low-variance
// sequence — enough to drive the sampler's CDF walk reproducibly without RNG.
func newDeterministicUniform() func() float32 {
	seq := []float32{0.05, 0.5, 0.95, 0.25, 0.75}
	i := 0
	return func() float32 {
		v := seq[i%len(seq)]
		i++
		return v
	}
}
