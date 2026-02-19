//go:build darwin && arm64

package metal

import (
	"math"
)

// Sampler transforms logits into a sampled token index.
type Sampler interface {
	Sample(logits *Array) *Array
}

// newSampler creates a composable sampler chain from the given parameters.
// Order: TopP -> MinP -> TopK -> Temperature -> categorical sample.
func newSampler(temp, topP, minP float32, topK int) Sampler {
	if temp == 0 {
		return greedy{}
	}

	var samplers []Sampler
	if topP > 0 && topP < 1 {
		samplers = append(samplers, TopP(topP))
	}
	if minP > 0 {
		samplers = append(samplers, MinPSampler(minP))
	}
	if topK > 0 {
		samplers = append(samplers, TopKSampler(topK))
	}
	samplers = append(samplers, Temperature(temp))
	return chain(samplers)
}

// chain applies a sequence of samplers, then samples from the result.
type chain []Sampler

func (c chain) Sample(logits *Array) *Array {
	for _, s := range c {
		logits = s.Sample(logits)
	}
	// Final categorical sample from log-probabilities
	return RandomCategorical(logits)
}

// greedy returns the argmax token.
type greedy struct{}

func (greedy) Sample(logits *Array) *Array {
	return Argmax(logits, -1, false)
}

// Temperature scales logits by 1/temp.
type Temperature float32

func (t Temperature) Sample(logits *Array) *Array {
	return MulScalar(logits, 1.0/float32(t))
}

// TopKSampler masks all but the top-k logits.
type TopKSampler int

func (k TopKSampler) Sample(logits *Array) *Array {
	neg := Negative(logits)
	mask := Argpartition(neg, int(k)-1, -1)
	// Slice the indices beyond top-k
	mask = SliceAxis(mask, -1, int32(k), int32(logits.Dim(-1)))
	return PutAlongAxis(logits, mask, FromValue(float32(math.Inf(-1))), -1)
}

// TopP implements nucleus (top-p) sampling.
// Keeps the smallest set of tokens whose cumulative probability exceeds p.
type TopP float32

func (p TopP) Sample(logits *Array) *Array {
	// Convert logits to probabilities
	probs := Softmax(logits)

	// Sort descending via argsort of negated probs
	neg := Negative(probs)
	sortIdx := Argsort(neg, -1)
	sortedProbs := TakeAlongAxis(probs, sortIdx, -1)

	// Cumulative sum of sorted probabilities
	cumProbs := CumSum(sortedProbs, -1, false, true)

	// Mask in sorted space: keep tokens where cumprob (excluding current) <= threshold
	shiftedCum := Subtract(cumProbs, sortedProbs)
	threshold := FromValue(float32(p))
	sortedMask := Where(
		Greater(shiftedCum, threshold),
		FromValue(float32(math.Inf(-1))),
		FromValue(float32(0)),
	)

	// Scatter mask back to original positions
	mask := PutAlongAxis(
		Zeros(logits.Shape(), DTypeFloat32),
		sortIdx,
		sortedMask,
		-1,
	)

	// Apply mask: -inf where excluded, original logit where kept
	return Where(
		Greater(FromValue(float32(0)), mask),
		FromValue(float32(math.Inf(-1))),
		logits,
	)
}

// MinPSampler masks tokens whose probability is below min_p * max_prob.
type MinPSampler float32

func (p MinPSampler) Sample(logits *Array) *Array {
	// Convert logits to probabilities
	probs := Softmax(logits)

	// Find the maximum probability
	maxProb := MaxAxis(probs, -1, true)

	// Threshold = min_p * max_prob
	threshold := MulScalar(maxProb, float32(p))

	// Mask tokens below threshold
	mask := Where(
		Greater(threshold, probs),
		FromValue(float32(math.Inf(-1))),
		logits,
	)
	return mask
}
