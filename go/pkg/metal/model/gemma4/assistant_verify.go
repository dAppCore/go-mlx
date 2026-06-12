// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

// assistant_verify.go is the realised Gemma 4 MTP verification: TARGET-SAMPLED
// prefix acceptance (greedy and temperature alike — the target's own sampler
// decides, the assistant only proposes) plus the live-cache verify that rolls
// rejected tokens back through the speculative-trim journal instead of cloning
// every cache each round.
//
// Acceptance scheme (the Gemma 4 pair scheme, after MTPLX's
// gemma4_exact_speculative_round): ONE batched target forward over the block;
// the target samples one token per verifier row; drafted tokens are accepted
// while they match the target's sample; the first mismatch takes the target's
// token and discards the rest. No p/q residual correction. A seeded sampled
// request therefore commits byte-identically to plain AR decode — the verify
// consumes the same keyed sampler chain (one draw per committed token, in
// commit order) that metal's plain decode loop uses.
package gemma4

import (
	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/metal"
)

var (
	errAsstVerifyLiveArmRefused = core.NewError("gemma4.assistant live verify could not arm the speculative-trim journal")
	errAsstVerifyLiveTrimFailed = core.NewError("gemma4.assistant live verify journal trim failed after a partial accept")
)

// gemma4VerifyDecider selects the acceptance decision for one verify round.
// A nil Sampler is the greedy decision (batched argmax folded into the verify
// forward's Eval); a non-nil Sampler is target-sampled acceptance using the
// generation's keyed sampler. Carry marks block[0] as the committed lead from
// the previous round — it was already target-chosen, so it is accepted
// without consuming a sampler draw (greedy re-derives it for free).
type gemma4VerifyDecider struct {
	Sampler  metal.Sampler
	Suppress []int32
	Carry    bool
}

// gemma4VerifyDecision is the outcome of one acceptance walk.
type gemma4VerifyDecision struct {
	Accepted    []int32
	TargetFirst int32 // the target's pick for the first judged position (-1 = none judged)
	Replacement int32
	AllAccepted bool
}

// gemma4GreedyAcceptWalk is the pure-CPU longest-matching-prefix walk over the
// pre-read batched greedy picks: first judges position 0, rows[i-1] judges
// position i.
func gemma4GreedyAcceptWalk(first int32, rows []int32, block []int32) (gemma4VerifyDecision, error) {
	decision := gemma4VerifyDecision{TargetFirst: first}
	for i := range block {
		targetToken := first
		if i > 0 {
			if i-1 >= len(rows) {
				return gemma4VerifyDecision{}, errAsstVerifyNoTargetToken
			}
			targetToken = rows[i-1]
		}
		if targetToken != block[i] {
			decision.Replacement = targetToken
			return decision, nil
		}
		decision.Accepted = append(decision.Accepted, block[i])
	}
	decision.AllAccepted = true
	return decision, nil
}

// gemma4TargetSampledDecision is the temperature>0 acceptance walk. Each
// judged position consumes exactly one draw from the generation's keyed
// sampler — the same NewSamplerWithSuppressionKeyed chain plain decode
// samples with — so the committed sequence of a seeded request equals plain
// AR decode token-for-token. The walk stops at the first mismatch: positions
// after it consume no draw, keeping the key stream aligned with the commit
// order. The carry lead consumes no draw (its draw happened the round it was
// chosen).
func gemma4TargetSampledDecision(targetLogits, allLogits *metal.Array, block []int32, d gemma4VerifyDecider) (gemma4VerifyDecision, error) {
	decision := gemma4VerifyDecision{TargetFirst: -1}
	for i := range block {
		if i == 0 && d.Carry {
			// Committed last round — re-judging it would burn a draw and
			// could even contradict the committed token. Accept and move on.
			decision.Accepted = append(decision.Accepted, block[0])
			continue
		}
		rowLogits := targetLogits
		rowOwned := false
		if i > 0 {
			rowLogits = metal.SliceAxis(allLogits, 1, int32(i-1), int32(i))
			rowOwned = true
		}
		tokenArr, id, _, err := metal.SampleTokenIDWithSuppressionGuard(rowLogits, d.Sampler, d.Suppress, false)
		metal.Free(tokenArr)
		if rowOwned {
			metal.Free(rowLogits)
		}
		if err != nil {
			return gemma4VerifyDecision{}, core.E("gemma4.assistant verify (target-sampled)", "sample verifier row", err)
		}
		if decision.TargetFirst < 0 {
			decision.TargetFirst = id
		}
		if id != block[i] {
			decision.Replacement = id
			return decision, nil
		}
		decision.Accepted = append(decision.Accepted, block[i])
	}
	decision.AllAccepted = true
	return decision, nil
}

// fillVerifyResult populates the caller-owned result fields shared by the
// live and clone verify paths: accepted/rejected bookkeeping plus the carried
// logits (the prediction after the accepted prefix) and, when any position
// was accepted, the hidden state that seeds the next draft block.
func fillVerifyResult(result *Gemma4AssistantVerifyResult, decision gemma4VerifyDecision, block []int32, allLogits, allHidden, targetLogits *metal.Array) error {
	k := len(decision.Accepted)
	result.AcceptedTokens = decision.Accepted
	result.AcceptedCount = k
	result.AllAccepted = decision.AllAccepted
	if decision.TargetFirst >= 0 {
		result.TargetTokens = append(result.TargetTokens, decision.TargetFirst)
	}
	if !decision.AllAccepted {
		result.ReplacementToken = decision.Replacement
		result.RejectedCount = len(block) - k
		result.RejectedTokens = append([]int32(nil), block[k:]...)
	}

	bonusLogits := targetLogits
	bonusOwned := false
	if k > 0 {
		bonusLogits = metal.SliceAxis(allLogits, 1, int32(k-1), int32(k))
		bonusOwned = true
	}
	var err error
	result.Logits, err = cloneGemma4AssistantArray(bonusLogits)
	if bonusOwned {
		metal.Free(bonusLogits)
	}
	if err != nil {
		return err
	}
	if k > 0 {
		hiddenSlice := metal.SliceAxis(allHidden, 1, int32(k-1), int32(k))
		result.Hidden, err = cloneGemma4AssistantArray(hiddenSlice)
		metal.Free(hiddenSlice)
		if err != nil {
			return err
		}
	}
	return nil
}

// verifyDraftBlockLive verifies a draft block ON THE LIVE CACHES: the
// speculative-trim journal (metal.CachesArmSpeculativeTrim) is armed around
// the one batched forward, and rejected tokens are trimmed back exactly —
// across the sliding-window rotation boundary included. No per-round cache
// clone: this is where the speculative speedup stops leaking into KV copies.
// The returned result carries NO caches (result.Caches == nil); the caller's
// cache set has been advanced by the committed prefix in place.
//
//	verify, err := pair.verifyDraftBlockLive(logits, block, caches, decider)
func (pair *Gemma4AssistantPair) verifyDraftBlockLive(targetLogits *metal.Array, block []int32, caches []metal.Cache, d gemma4VerifyDecider) (*Gemma4AssistantVerifyResult, error) {
	if pair == nil || pair.Target == nil {
		return nil, errAsstVerifyNeedTargetModel
	}
	if targetLogits == nil || !targetLogits.Valid() {
		return nil, errAsstVerifyNeedTargetLogits
	}
	if len(block) == 0 {
		return nil, errAsstVerifyNeedDraftTokens
	}
	if len(caches) == 0 {
		return nil, errAsstVerifyNeedTargetCaches
	}
	if !metal.CachesArmSpeculativeTrim(caches) {
		return nil, errAsstVerifyLiveArmRefused
	}
	defer metal.CachesDisarmSpeculativeTrim(caches)

	result := &Gemma4AssistantVerifyResult{
		DraftedTokens: append([]int32(nil), block...),
	}
	vDraft := metal.FromValues(block, len(block))
	draftInput := metal.Reshape2(vDraft, 1, int32(len(block)))
	metal.Free(vDraft)
	allLogits, allHidden := pair.Target.ForwardAllTokenLogitsAndHidden(draftInput, caches)
	metal.Free(draftInput)
	if allLogits == nil || allHidden == nil || !allLogits.Valid() || !allHidden.Valid() {
		metal.Free(allLogits, allHidden)
		gemma4TruncateVerifyCaches(caches, len(block)) // roll the whole block back
		return nil, errAsstVerifyNoTargetToken
	}

	var decision gemma4VerifyDecision
	if d.Sampler == nil {
		// Greedy: fold the batched argmax picks into the SAME Eval as the
		// forward — zero additional GPU round trips.
		firstTok, rowToks, suppressOwned := gemma4AssistantBatchedGreedyGraph(targetLogits, allLogits, d.Suppress)
		if err := metal.Eval(allLogits, allHidden, firstTok, rowToks); err != nil {
			metal.Free(firstTok, rowToks)
			metal.Free(suppressOwned...)
			metal.Free(allLogits, allHidden)
			gemma4TruncateVerifyCaches(caches, len(block))
			return nil, core.E("gemma4.assistant verify (live)", "batched target forward", err)
		}
		metal.DetachCaches(caches)
		first, rows, readErr := gemma4AssistantReadGreedyTokens(firstTok, rowToks)
		metal.Free(firstTok, rowToks)
		metal.Free(suppressOwned...)
		if readErr != nil {
			metal.Free(allLogits, allHidden)
			gemma4TruncateVerifyCaches(caches, len(block))
			return nil, readErr
		}
		var decideErr error
		decision, decideErr = gemma4GreedyAcceptWalk(first, rows, block)
		if decideErr != nil {
			metal.Free(allLogits, allHidden)
			gemma4TruncateVerifyCaches(caches, len(block))
			return nil, decideErr
		}
	} else {
		if err := metal.Eval(allLogits, allHidden); err != nil {
			metal.Free(allLogits, allHidden)
			gemma4TruncateVerifyCaches(caches, len(block))
			return nil, core.E("gemma4.assistant verify (live)", "batched target forward", err)
		}
		metal.DetachCaches(caches)
		var decideErr error
		decision, decideErr = gemma4TargetSampledDecision(targetLogits, allLogits, block, d)
		if decideErr != nil {
			metal.Free(allLogits, allHidden)
			gemma4TruncateVerifyCaches(caches, len(block))
			return nil, decideErr
		}
	}
	defer metal.Free(allLogits, allHidden)

	if err := fillVerifyResult(result, decision, block, allLogits, allHidden, targetLogits); err != nil {
		gemma4TruncateVerifyCaches(caches, len(block))
		result.Close()
		return nil, err
	}
	if dropped := len(block) - len(decision.Accepted); dropped > 0 {
		for i, c := range caches {
			if !metal.CacheTruncateTo(c, c.Len()-dropped) {
				// The journal was armed — a failed trim is an engine fault,
				// not a recoverable fallback. Refuse loudly (naming the cache)
				// rather than serve from a cache polluted with rejected tokens.
				result.Close()
				return nil, core.E("gemma4.assistant verify (live)",
					core.Sprintf("cache %d (%T len=%d offset=%d) refused trim of %d after accepting %d of %d",
						i, c, c.Len(), c.Offset(), dropped, len(decision.Accepted), len(block)),
					errAsstVerifyLiveTrimFailed)
			}
		}
	}
	return result, nil
}

// verifyDraftBlockSampledClone is the clone-regime counterpart of the live
// verify for cache modes without speculative-trim journals (quantized, paged,
// turboquant): clone the caches, forward, decide via target-sampled
// acceptance, then truncate the clone (or rebuild from the live caches when
// the clone cannot trim). The caller adopts result.Caches.
func (pair *Gemma4AssistantPair) verifyDraftBlockSampledClone(targetLogits *metal.Array, block []int32, targetCaches []metal.Cache, d gemma4VerifyDecider) (*Gemma4AssistantVerifyResult, error) {
	if pair == nil || pair.Target == nil {
		return nil, errAsstVerifyNeedTargetModel
	}
	if targetLogits == nil || !targetLogits.Valid() {
		return nil, errAsstVerifyNeedTargetLogits
	}
	if len(block) == 0 {
		return nil, errAsstVerifyNeedDraftTokens
	}
	if len(targetCaches) == 0 {
		return nil, errAsstVerifyNeedTargetCaches
	}
	verifyCaches, err := metal.CloneCachePrefixes(targetCaches)
	if err != nil {
		return nil, err
	}
	result := &Gemma4AssistantVerifyResult{
		DraftedTokens: append([]int32(nil), block...),
		Caches:        verifyCaches,
	}
	vDraft := metal.FromValues(block, len(block))
	draftInput := metal.Reshape2(vDraft, 1, int32(len(block)))
	metal.Free(vDraft)
	allLogits, allHidden := pair.Target.ForwardAllTokenLogitsAndHidden(draftInput, verifyCaches)
	metal.Free(draftInput)
	if allLogits == nil || allHidden == nil || !allLogits.Valid() || !allHidden.Valid() {
		metal.Free(allLogits, allHidden)
		result.Close()
		return nil, errAsstVerifyNoTargetToken
	}
	if err := metal.Eval(allLogits, allHidden); err != nil {
		metal.Free(allLogits, allHidden)
		result.Close()
		return nil, core.E("gemma4.assistant verify (sampled)", "batched target forward", err)
	}
	metal.DetachCaches(verifyCaches)
	defer metal.Free(allLogits, allHidden)

	decision, err := gemma4TargetSampledDecision(targetLogits, allLogits, block, d)
	if err != nil {
		result.Close()
		return nil, err
	}
	if err := fillVerifyResult(result, decision, block, allLogits, allHidden, targetLogits); err != nil {
		result.Close()
		return nil, err
	}
	k := len(decision.Accepted)
	if !gemma4TruncateVerifyCaches(verifyCaches, len(block)-k) {
		rebuilt, berr := pair.rebuildAcceptedPrefixCaches(targetCaches, block[:k])
		if berr != nil {
			result.Close()
			return nil, berr
		}
		metal.FreeCaches(verifyCaches)
		result.Caches = rebuilt
	}
	return result, nil
}
