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
	curr := logits
	for _, s := range c {
		next := s.Sample(curr)
		if curr != logits {
			Free(curr)
		}
		curr = next
	}
	// Final categorical sample from log-probabilities
	res := RandomCategorical(curr)
	if curr != logits {
		Free(curr)
	}
	return res
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
	maskIdx := Argpartition(neg, int(k)-1, -1)
	Free(neg)
	// Slice the indices beyond top-k
	mask := SliceAxis(maskIdx, -1, int32(k), int32(logits.Dim(-1)))
	Free(maskIdx)
	inf := FromValue(float32(math.Inf(-1)))
	res := PutAlongAxis(logits, mask, inf, -1)
	Free(mask, inf)
	return res
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
	Free(neg)
	sortedProbs := TakeAlongAxis(probs, sortIdx, -1)

	// Cumulative sum of sorted probabilities
	cumProbs := CumSum(sortedProbs, -1, false, true)

	// Mask in sorted space: keep tokens where cumprob (excluding current) <= threshold
	shiftedCum := Subtract(cumProbs, sortedProbs)
	threshold := FromValue(float32(p))
	inf := FromValue(float32(math.Inf(-1)))
	zero := FromValue(float32(0))

	gt := Greater(shiftedCum, threshold)
	sortedMask := Where(gt, inf, zero)
	Free(gt, inf, zero, threshold, shiftedCum, cumProbs, sortedProbs)

	// Scatter mask back to original positions
	emptyMask := Zeros(logits.Shape(), DTypeFloat32)
	mask := PutAlongAxis(emptyMask, sortIdx, sortedMask, -1)
	Free(emptyMask, sortIdx, sortedMask)

	// Apply mask: -inf where excluded, original logit where kept
	zeroArr := FromValue(float32(0))
	gt0 := Greater(zeroArr, mask)
	inf2 := FromValue(float32(math.Inf(-1)))
	res := Where(gt0, inf2, logits)
	Free(zeroArr, gt0, inf2, mask, probs)

	return res
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
	Free(maxProb)

	// Mask tokens below threshold
	inf := FromValue(float32(math.Inf(-1)))
	gt := Greater(threshold, probs)
	mask := Where(gt, inf, logits)
	Free(probs, threshold, inf, gt)
	return mask
}
