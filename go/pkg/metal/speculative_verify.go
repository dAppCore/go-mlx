// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

// sampledVerifyDecision walks a draft block under speculative SAMPLING. For each
// position it forms the target distribution p and the drafter distribution q
// (same temperature / top-k / top-p the serve samples at), accepts the drafted
// token with probability min(1, p(x)/q(x)), and on the FIRST rejection draws the
// replacement from the normalised residual (p-q)+.
//
// targetLogits[i] and draftLogits[i] are the [..., vocab] logits at position i.
// uniform() yields the accept coin and the residual draw in [0,1). Returns the
// accepted prefix; on a rejection allAccepted is false and replacement holds the
// residual token. When the whole block is accepted, allAccepted is true and the
// caller samples the bonus from the carried target logits (replacement is 0).
//
// This is the temperature>0 counterpart of the greedy longest-matching-prefix
// accept; the caller keeps greedy on its own fast path so greedy-exactness is
// untouched.
func sampledVerifyDecision(targetLogits, draftLogits []*Array, draftTokens []int32, cfg GenerateConfig, uniform func() float32, suppress []int32) (accepted []int32, replacement int32, allAccepted bool, err error) {
	for i := range draftTokens {
		// A nil draft logit marks a committed lead token (the prepended carryLead
		// that was already emitted last round): it carries no drafter distribution
		// q to weigh against, so it is accepted unconditionally — the verify just
		// forwards it to advance the cache and produce the logits for the rest of
		// the block. This is what lets the speculative SAMPLING path keep carryLead
		// (and thus the speedup) instead of paying a per-round replacement forward.
		if draftLogits[i] == nil {
			accepted = append(accepted, draftTokens[i])
			continue
		}
		p := samplingDistribution(targetLogits[i], cfg.Temperature, cfg.TopP, cfg.MinP, int(cfg.TopK))
		q := samplingDistribution(draftLogits[i], cfg.Temperature, cfg.TopP, cfg.MinP, int(cfg.TopK))
		Materialize(p, q)
		pf := append([]float32(nil), p.Floats()...)
		qf := append([]float32(nil), q.Floats()...)
		Free(p, q)
		zeroSuppressedProbs(pf, suppress)
		zeroSuppressedProbs(qf, suppress)

		if speculativeAcceptToken(pf, qf, draftTokens[i], uniform()) {
			accepted = append(accepted, draftTokens[i])
			continue
		}
		return accepted, speculativeResidualSample(pf, qf, uniform()), false, nil
	}
	return accepted, 0, true, nil
}

// zeroSuppressedProbs removes mass from suppressed tokens so neither the accept
// test nor the residual draw can land on one.
func zeroSuppressedProbs(dist []float32, suppress []int32) {
	for _, s := range suppress {
		if s >= 0 && int(s) < len(dist) {
			dist[s] = 0
		}
	}
}

// SampleTokenFromLogits draws one token from the distribution these logits would
// be sampled at (temperature / top-k / top-p / min-p, then categorical with the
// supplied uniform). It is the drafter's proposal step on the speculative
// SAMPLING path — the gemma4 assistant calls it instead of argmax when
// temperature > 0. uniform() yields a draw in [0,1).
func SampleTokenFromLogits(logits *Array, temp, topP, minP float32, topK int, suppress []int32, uniform func() float32) int32 {
	dist := samplingDistribution(logits, temp, topP, minP, topK)
	Materialize(dist)
	probs := append([]float32(nil), dist.Floats()...)
	Free(dist)
	zeroSuppressedProbs(probs, suppress)
	return sampleTokenFromProbs(probs, uniform())
}

// SpeculativeVerifyDecision is the exported entry the gemma4 assistant verifier
// calls on the temperature > 0 path: it walks the draft block, accepting each
// token with probability min(1, p/q) and drawing a reject's replacement from the
// (p-q)+ residual. See sampledVerifyDecision for the contract.
func SpeculativeVerifyDecision(targetLogits, draftLogits []*Array, draftTokens []int32, cfg GenerateConfig, uniform func() float32, suppress []int32) (accepted []int32, replacement int32, allAccepted bool, err error) {
	return sampledVerifyDecision(targetLogits, draftLogits, draftTokens, cfg, uniform, suppress)
}
