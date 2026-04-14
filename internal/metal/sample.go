//go:build darwin && arm64 && !nomlx

package metal

import (
	"math"
)

// Sampler transforms logits into a sampled token index.
//
//	s := newSampler(0.7, 0.9, 0, 40) // temp=0.7, topP=0.9, minP=0, topK=40
//	tokenID := s.Sample(logits)
type Sampler interface {
	Sample(logits *Array) *Array
}

// newSampler creates a composable sampler chain from the given parameters.
// Order: Temperature -> TopP -> TopK -> MinP -> categorical sample.
//
//	s := newSampler(0, 0, 0, 0)        // greedy (temp=0)
//	s := newSampler(0.7, 0.9, 0, 40)   // top-p + top-k + temperature
//	s := newSampler(1.0, 0, 0.05, 0)   // min-p sampling
func newSampler(temp, topP, minP float32, topK int) Sampler {
	samplers := make([]Sampler, 0, 4)
	if temp > 0 {
		samplers = append(samplers, Temperature(temp))
	}
	if topP > 0 && topP < 1 {
		samplers = append(samplers, TopP(topP))
	}
	if topK > 0 {
		samplers = append(samplers, TopKSampler(topK))
	}
	if minP > 0 {
		samplers = append(samplers, MinPSampler(minP))
	}
	if len(samplers) == 0 {
		return greedy{}
	}
	return chain(samplers)
}

// chain applies a sequence of samplers in order, then draws a categorical sample.
//
//	chain{TopP(0.9), TopKSampler(40), Temperature(0.7)}.Sample(logits)
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

// greedy returns the argmax token (deterministic, no sampling).
//
//	greedy{}.Sample(logits) // picks the single most likely token
type greedy struct{}

func (greedy) Sample(logits *Array) *Array {
	return Argmax(logits, -1, false)
}

// Temperature scales logits by 1/temp before categorical sampling.
// Higher values produce more random output; lower values approach greedy.
//
//	Temperature(0.7).Sample(logits) // moderate creativity
//	Temperature(0.1).Sample(logits) // near-greedy, focused output
type Temperature float32

func (t Temperature) Sample(logits *Array) *Array {
	return MulScalar(logits, 1.0/float32(t))
}

// TopKSampler masks all but the top-k logits, setting the rest to -inf.
//
//	TopKSampler(40).Sample(logits) // keep only top 40 candidates
//	TopKSampler(10).Sample(logits) // very focused — top 10 only
type TopKSampler int

func (k TopKSampler) Sample(logits *Array) *Array {
	lastDim := logits.Dim(logits.NumDims() - 1)
	if lastDim <= 0 || int(k) <= 0 || int(k) >= lastDim {
		return logits.Clone()
	}
	neg := Negative(logits)
	maskIdx := Argpartition(neg, int(k)-1, -1)
	Free(neg)
	// Slice the indices beyond top-k
	mask := SliceAxis(maskIdx, -1, int32(k), int32(lastDim))
	Free(maskIdx)
	inf := FromValue(float32(math.Inf(-1)))
	res := PutAlongAxis(logits, mask, inf, -1)
	Free(mask, inf)
	return res
}

// TopP implements nucleus (top-p) sampling.
// Keeps the smallest set of tokens whose cumulative probability exceeds p.
//
//	TopP(0.9).Sample(logits) // include tokens covering 90% of probability mass
//	TopP(0.5).Sample(logits) // conservative — only highest-probability half
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

// MinPSampler masks tokens whose probability falls below min_p * max_prob.
// Adapts the threshold relative to the best token, so the cut-off scales with confidence.
//
//	MinPSampler(0.05).Sample(logits) // drop tokens less than 5% of top-token probability
//	MinPSampler(0.1).Sample(logits)  // stricter — drop tokens below 10% of max
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
