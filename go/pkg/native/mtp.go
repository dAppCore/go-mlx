// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	core "dappco.re/go"
)

// mtp.go — speculative (multi-token-prediction) decode over two ArchSessions, a fast DRAFT
// proposing K tokens and the TARGET verifying them in one pass over its OWN resident cache.
// The result is TOKEN-IDENTICAL to plain greedy Generate on the target: a draft is only ever
// an ACCELERATOR — every position is decided by the target's greedy argmax, so a wrong draft
// token merely falls back to the token the target would have emitted anyway. Correctness does
// NOT depend on draft quality (a divergent draft just accepts nothing and runs at greedy speed).
//
// Why this is the native multi-token forward against the session cache: the target's verify runs
// the K draft tokens through StepWithID, which writes each token's K/V into the session's growing
// cache at the live position and advances — exactly what Generate does internally, so the verified
// hiddens are byte-identical to stepping the same tokens one at a time. There is no separate fused
// multi-token kernel to drive (DecodeForward/DecodeForwardArch allocate their OWN caches and step
// internally, so they cannot target a session's resident cache); K sequential steps over the
// session cache IS the cache-faithful batched forward, and the win is replacing K target head+step
// rounds with one draft+verify round whenever the draft guesses right.
//
// Accept rule (standard MTP, matched to plain greedy): with h the hidden of the last committed
// token, the target's next token is T0 = greedy(head(h)). The draft proposes d0..d_{K-1}; each
// d_i is stepped on the TARGET to get hidden_i, and the target's greedy there is T_{i+1} =
// greedy(head(hidden_i)). Accept the longest prefix where d_i == T_i (i.e. the draft matched what
// the target would have emitted), then emit ONE bonus correction token T_j (the target's greedy at
// the first mismatch, or after a full-length accept) and step it. The cursor hidden h for the next
// round is the bonus token's hidden. Every emitted id is a target greedy id, so the stream equals
// target.Generate(promptIDs, maxNew, eos) token for token.
//
// Cache rollback on reject: the target steps all K draft tokens (advancing pos by K), then pos is
// reset to the accepted length + the committed bonus token. The rejected suffix's K/V rows are
// simply overwritten by the next write at that position (stepToken writes at pos and SDPA attends a
// pos+1 window — see decode_forward_arch.go), so resetting pos is a complete rollback. This is exact
// for owner caches; sliding-ring caches are not rolled back row-for-row, so a speculative window
// must not straddle a ring wrap (the dense/all-global path used by the gate has no ring).

// MTPResult reports a speculative decode: the generated ids (target-greedy, identical to plain
// Generate) plus the acceptance accounting — how many draft tokens were proposed vs accepted, and
// how many draft+verify rounds ran. Drafted/Accepted give the realised acceptance rate; with a
// perfect draft Accepted≈Drafted and Rounds is small, with a useless draft Accepted is ~0 and the
// stream is unchanged (still correct, just no speedup).
type MTPResult struct {
	Tokens   []int32
	Drafted  int // total draft tokens proposed across all rounds
	Accepted int // draft tokens that matched the target's greedy (the realised speculative win)
	Rounds   int // draft→verify rounds executed
}

// MTPDecode speculatively decodes up to maxNew tokens on target, using draft to propose K tokens
// per round, returning the target-greedy token stream (token-identical to target.Generate) plus the
// acceptance stats. eosID < 0 disables early stop. Both sessions are advanced as a side effect: the
// target ends positioned exactly after the committed sequence (prompt + emitted tokens), the draft
// after its last proposal — drive each from a single goroutine (the ArchSession contract).
//
// The two sessions are independent caches: typically draft is a small/cheap model and target the
// real one, but for correctness they may share weights (the draft then accepts everything and the
// speedup is maximal) or diverge wildly (nothing accepts, greedy speed) — the output is the same.
func MTPDecode(target, draft *ArchSession, promptIDs []int32, maxNew, eosID, k int) (*MTPResult, error) {
	if target == nil || draft == nil {
		return nil, core.NewError("native.MTPDecode: nil target/draft session")
	}
	if len(promptIDs) == 0 {
		return nil, core.NewError("native.MTPDecode: empty prompt")
	}
	if maxNew <= 0 {
		return nil, core.NewError("native.MTPDecode: maxNew must be > 0")
	}
	if k <= 0 {
		return nil, core.NewError("native.MTPDecode: k must be > 0")
	}
	// the target must have cache headroom for the speculative overshoot: a round steps up to K draft
	// tokens past the committed length before rolling back, so guard the worst case once up front.
	if target.pos+len(promptIDs)+maxNew+k > target.maxLen {
		return nil, core.NewError("native.MTPDecode: target sequence (+ speculation window) would exceed maxLen cache rows")
	}
	if draft.pos+len(promptIDs)+maxNew+k > draft.maxLen {
		return nil, core.NewError("native.MTPDecode: draft sequence (+ speculation window) would exceed maxLen cache rows")
	}

	res := &MTPResult{Tokens: make([]int32, 0, maxNew)}
	var verifyStack [16]int32
	verifyIDs := verifyStack[:1]
	if k+1 > len(verifyStack) {
		verifyIDs = make([]int32, 1, k+1)
	}
	var greedyStack [16]int32
	greedyIDs := greedyStack[:0]
	if k+1 > len(greedyStack) {
		greedyIDs = make([]int32, 0, k+1)
	}

	// prefill the prompt into BOTH sessions; keep the target's last hidden as the cursor h. The
	// draft is advanced in lockstep so its cache holds the same committed history before it proposes.
	hidden, ok, err := target.prefillMTPPrompt(promptIDs, true)
	if err != nil {
		return nil, err
	}
	if !ok {
		for i, id := range promptIDs {
			h, err := target.stepID(id)
			if err != nil {
				return nil, err
			}
			if i == len(promptIDs)-1 {
				hidden = h
			}
		}
	}
	if _, ok, err = draft.prefillMTPPrompt(promptIDs, false); err != nil {
		return nil, err
	} else if !ok {
		for _, id := range promptIDs {
			if _, err := draft.stepID(id); err != nil {
				return nil, err
			}
		}
	}
	if hidden == nil {
		return nil, core.NewError("native.MTPDecode: prompt prefill produced no cursor hidden")
	}

	// each round: read the target's greedy at the cursor (T0, always committed), let the draft
	// propose K continuations, verify them against the target's cache, commit the accepted run plus
	// one bonus correction, and carry the bonus's hidden as the next cursor — until maxNew/eos.
	for len(res.Tokens) < maxNew {
		res.Rounds++
		draftPos0 := draft.pos // draft cache position at round start; the committed run is replayed from here to keep the draft aligned with the target

		// the token the target emits at the cursor (round's first committed token); this is T0.
		t0, err := target.greedyOf(hidden)
		if err != nil {
			return nil, err
		}

		// DRAFT: propose K tokens. The draft seeds from t0 (the token actually being committed),
		// stepping its own cache; quality is irrelevant to correctness. We stop drafting early if the
		// committed sequence would already reach maxNew — no point proposing tokens we can't emit.
		room := maxNew - len(res.Tokens) // tokens still emittable INCLUDING t0
		nDraft := k
		if nDraft > room-1 { // -1: t0 itself occupies one emit slot
			nDraft = room - 1
		}
		if nDraft < 0 {
			nDraft = 0
		}
		verifyIDs = verifyIDs[:1]
		verifyIDs[0] = t0
		drafts := verifyIDs[1:1]
		seed := t0
		for d := 0; d < nDraft; d++ {
			dh, err := draft.stepID(seed)
			if err != nil {
				return nil, err
			}
			nd, err := draft.greedyOf(dh)
			if err != nil {
				return nil, err
			}
			drafts = append(drafts, nd)
			seed = nd
		}
		verifyIDs = verifyIDs[:1+len(drafts)]
		res.Drafted += len(drafts)

		// VERIFY: run [t0, drafts...] through the TARGET's cache from the current pos in one pass of
		// sequential steps (the multi-token forward against the resident cache). After stepping token
		// x at a position, the target's greedy of that hidden is what it would emit AFTER x — i.e. the
		// expected value of the NEXT proposed token. So:
		//   step t0           → expect drafts[0]
		//   step drafts[0]    → expect drafts[1]
		//   ...
		// accept the longest prefix of drafts that matches, then the first mismatch's expected token
		// is the bonus correction. posBefore lets us roll the target cache back to the committed length.
		posBefore := target.pos
		commitLen := 1             // t0 is always committed (it's the target's own greedy)
		bonusHidden := []byte(nil) // filled when we step the committed bonus token below
		accepted := 0
		var bonus int32
		// compute the target's greedy after each of [t0, drafts...]. The BATCHED path runs all of them
		// through the resident stack in ONE pass over the cache (the speculative-decode speedup — one
		// submit, weights resident, vs K stepGreedy rounds); it declines (batched=false) for models
		// outside the dense path (PLE/MoE/recorded-ICB/shared-KV), where we step sequentially. Both
		// produce the identical greedys, so the accept/reject and the emitted stream are unchanged.
		greedys, batched, verr := target.verifyBatchedInto(verifyIDs, greedyIDs[:len(verifyIDs)])
		if verr != nil {
			return nil, verr
		}
		if batched {
			bonus = greedys[0] // greedys[i] = target's greedy AFTER the i-th verified token
			for d := 0; d < len(drafts); d++ {
				if drafts[d] != greedys[d] { // mismatch: target diverges here, drafts[d] rejected
					bonus = greedys[d]
					break
				}
				commitLen++
				accepted++
				bonus = greedys[d+1]
			}
		} else {
			expected, err := target.stepGreedy(t0)
			if err != nil {
				return nil, err
			}
			bonus = expected // if drafts is empty, the bonus IS the target's next greedy after t0
			for d := 0; d < len(drafts); d++ {
				if drafts[d] != expected { // mismatch: target diverges here, drafts[d] rejected
					bonus = expected
					break
				}
				// accepted: drafts[d] is exactly the target's greedy — commit it and step the target to
				// get the NEXT expected token (and a fresh bonus in case this was the last draft).
				commitLen++
				accepted++
				expected, err = target.stepGreedy(drafts[d])
				if err != nil {
					return nil, err
				}
				bonus = expected
			}
		}
		res.Accepted += accepted

		// roll the target cache back to just the committed run (t0 + accepted drafts); the rejected
		// suffix's K/V is overwritten by the bonus step below / the next round.
		target.pos = posBefore + commitLen

		// keep the DRAFT cache aligned with the committed run too — otherwise it drifts a slot every
		// round (it proposes nDraft tokens but only `accepted` commit, and on a FULL accept it proposed
		// the last token without ever stepping it into its cache), the next round's proposals continue
		// from the wrong context, and acceptance collapses. The committed tokens [t0, accepted drafts]
		// are already resident in the draft cache from proposing at rows [draftPos0 .. draftPos0+accepted]
		// — except the full-accept case, where the final committed draft was proposed but not stepped, so
		// step it now to fill that row. Then roll the draft to the committed length so the bonus below
		// lands at the same position the target wrote it.
		if accepted == len(drafts) && accepted > 0 {
			if _, err = draft.stepID(drafts[accepted-1]); err != nil {
				return nil, err
			}
		}
		draft.pos = draftPos0 + commitLen

		// commit the accepted run, honouring maxNew/eos as plain Generate would.
		stop := false
		for _, id := range verifyIDs[:commitLen] {
			res.Tokens = append(res.Tokens, id)
			if (eosID >= 0 && int(id) == eosID) || len(res.Tokens) >= maxNew {
				stop = true
				break
			}
		}
		if stop {
			break
		}

		// commit the bonus correction token (the target's greedy after the accepted run) and step it
		// on BOTH sessions, so each cache holds it and the next round's cursor is its hidden.
		res.Tokens = append(res.Tokens, bonus)
		if bonusHidden, err = target.stepID(bonus); err != nil {
			return nil, err
		}
		if _, err = draft.stepID(bonus); err != nil {
			return nil, err
		}
		hidden = bonusHidden
		if (eosID >= 0 && int(bonus) == eosID) || len(res.Tokens) >= maxNew {
			break
		}
	}

	return res, nil
}

func (s *ArchSession) prefillMTPPrompt(ids []int32, readLast bool) ([]byte, bool, error) {
	if len(ids) == 0 {
		return nil, false, core.NewError("native.MTPDecode: empty prompt")
	}
	if s.perLayerInput != nil || s.pos+len(ids) > s.maxLen {
		return nil, false, nil
	}
	if readLast && len(ids) == 1 {
		hidden, err := s.stepID(ids[0])
		return hidden, true, err
	}
	batchIDs := ids
	if readLast {
		batchIDs = ids[:len(ids)-1]
	}
	if len(batchIDs) == 0 {
		return nil, false, nil
	}
	var embStack [16][]byte
	var embs [][]byte
	if len(batchIDs) <= len(embStack) {
		embs = embStack[:len(batchIDs)]
	} else {
		embs = make([][]byte, len(batchIDs))
	}
	for i, id := range batchIDs {
		emb, err := s.embed(id)
		if err != nil {
			return nil, false, err
		}
		embs[i] = emb
	}
	var (
		hiddens [][]byte
		ok      bool
		err     error
	)
	withAutoreleasePool(func() {
		hiddens, ok, err = s.state.stepTokensBatchedDenseResult(embs, s.pos, false)
	})
	if err != nil || !ok {
		return nil, ok, err
	}
	s.pos += len(batchIDs)
	if !readLast {
		return nil, true, nil
	}
	if len(hiddens) != 0 {
		return nil, false, core.NewError("native.MTPDecode: dense prompt prefill returned incomplete hiddens")
	}
	hidden, err := s.stepID(ids[len(ids)-1])
	return hidden, true, err
}

// stepID embeds token id and steps it through the session's resident cache at the current position,
// advancing pos — the same primitive Generate uses internally (StepWithID), so the resulting hidden
// is byte-identical to a plain greedy step on this token. PLE models thread the id correctly.
func (s *ArchSession) stepID(id int32) ([]byte, error) {
	emb, err := s.embed(id)
	if err != nil {
		return nil, err
	}
	return s.StepWithID(id, emb)
}

// greedyOf returns the greedy argmax id plain Generate would emit at this hidden.
func (s *ArchSession) greedyOf(hidden []byte) (int32, error) {
	return s.greedyFromHiddenInPool(hidden, nil)
}

// stepGreedy steps token id on the session cache and returns the greedy argmax of the resulting
// hidden — the target's expected NEXT token after id. It is stepID followed by greedyOf, the verify
// inner loop's unit of work.
func (s *ArchSession) stepGreedy(id int32) (int32, error) {
	h, err := s.stepID(id)
	if err != nil {
		return 0, err
	}
	return s.greedyOf(h)
}
