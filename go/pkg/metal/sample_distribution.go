// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

// samplingDistribution returns the probability distribution a sampler with these
// settings would draw from for the given logits: temperature scaling, then
// top-p / top-k / min-p truncation, then softmax.
//
// Speculative SAMPLING (temperature > 0) needs the full per-token distribution,
// not just the argmax: the verifier weighs each draft token's accept coin
// against p(x)/q(x) and, on rejection, resamples from the normalised (p-q)+
// residual — both require the whole distribution. The transform order mirrors
// newSampler, so this is identical to what plain sampling at the same settings
// would draw (the served distribution is preserved exactly).
//
// The returned [..., vocab] probability Array is the caller's to Free. Greedy
// (temperature == 0) is handled by the argmax path, not here.
func samplingDistribution(logits *Array, temp, topP, minP float32, topK int) *Array {
	work := logits
	owned := false
	apply := func(s Sampler) {
		next := s.Sample(work)
		if owned {
			Free(work)
		}
		work, owned = next, true
		CloseSampler(s)
	}
	if temp > 0 && temp != 1 {
		apply(Temperature(temp))
	}
	if topP > 0 && topP < 1 {
		apply(TopP(topP))
	}
	if topK > 0 {
		apply(TopKSampler(topK))
	}
	if minP > 0 {
		apply(MinPSampler(minP))
	}
	probs := Softmax(work)
	if owned {
		Free(work)
	}
	return probs
}
