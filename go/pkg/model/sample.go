// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	"math"
	"sort"

	core "dappco.re/go"
)

// Token sampling — backend-agnostic, the step after the LM head: it turns a vocab of
// logits into the next token id. It operates on bf16 []byte logits (the seam's lingua
// franca — whatever backend produced them), so it lives here in pkg/model, pure-Go and
// all-platforms. Greedy closes a native decode loop deterministically (the right choice
// for a tok/s bench); the temperature/top-k/top-p Sampler is for stochastic generation.
// (The served path's sampler is go-inference's until the reactive engine is pointed at
// model.Backend; this is the contract-native sampler.)

const bf16Size = 2

func bf16ToF32(lo, hi byte) float32 {
	return math.Float32frombits(uint32(uint16(lo)|uint16(hi)<<8) << 16)
}

// Greedy returns the argmax of vocab bf16 logits; ties resolve to the lowest index.
// Deterministic, no RNG — the natural choice for closing a decode loop in a bench.
func Greedy(logits []byte, vocab int) (int32, error) {
	if len(logits) != vocab*bf16Size {
		return 0, core.NewError("model.Greedy: logits must be vocab bf16 bytes")
	}
	best, bestV := 0, float32(math.Inf(-1))
	for i := 0; i < vocab; i++ {
		if v := bf16ToF32(logits[i*bf16Size], logits[i*bf16Size+1]); v > bestV {
			best, bestV = i, v
		}
	}
	return int32(best), nil
}

// SampleParams configures stochastic sampling. Temperature <= 0 makes Sample greedy.
// TopK <= 0 disables the top-k cut; TopP <= 0 or >= 1 disables the nucleus cut. The two
// cuts compose (top-k first, then top-p over the kept set), matching the usual order.
type SampleParams struct {
	Temperature float32
	TopK        int
	TopP        float32
}

// Sampler draws tokens with a reproducible RNG that ADVANCES per Sample call, so a
// generation loop gets a varied sequence from a single seed (vs re-seeding per token).
// Construct with NewSampler; Greedy draws are RNG-free so they don't perturb the state.
type Sampler struct {
	state uint64
}

// NewSampler returns a sampler seeded for reproducible draws.
func NewSampler(seed uint64) *Sampler { return &Sampler{state: seed} }

// next is splitmix64 → a uniform float32 in [0,1); advances the RNG state.
func (s *Sampler) next() float32 {
	s.state += 0x9e3779b97f4a7c15
	z := s.state
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	z ^= z >> 31
	return float32(z>>40) / float32(1<<24)
}

// Sample picks a token from vocab bf16 logits per p: greedy when Temperature <= 0, else
// temperature-scaled softmax with optional top-k then top-p (nucleus) restriction, drawn
// from the categorical with this sampler's RNG.
func (s *Sampler) Sample(logits []byte, vocab int, p SampleParams) (int32, error) {
	if len(logits) != vocab*bf16Size {
		return 0, core.NewError("model.Sample: logits must be vocab bf16 bytes")
	}
	if p.Temperature <= 0 {
		return Greedy(logits, vocab)
	}

	// temperature-scaled logits + their max (for a stable softmax).
	scaled := make([]float32, vocab)
	maxL := float32(math.Inf(-1))
	for i := 0; i < vocab; i++ {
		v := bf16ToF32(logits[i*bf16Size], logits[i*bf16Size+1]) / p.Temperature
		scaled[i] = v
		if v > maxL {
			maxL = v
		}
	}
	probs := make([]float32, vocab)
	var sum float32
	for i, v := range scaled {
		e := float32(math.Exp(float64(v - maxL)))
		probs[i] = e
		sum += e
	}
	for i := range probs {
		probs[i] /= sum
	}

	// rank by probability, descending (top-k + top-p both work over this order).
	order := make([]int, vocab)
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool { return probs[order[a]] > probs[order[b]] })

	keep := vocab
	if p.TopK > 0 && p.TopK < keep {
		keep = p.TopK
	}
	if p.TopP > 0 && p.TopP < 1 {
		var cum float32
		n := 0
		for n < keep {
			cum += probs[order[n]]
			n++
			if cum >= p.TopP {
				break
			}
		}
		keep = n
	}

	// renormalise over the kept set and draw.
	var ksum float32
	for i := 0; i < keep; i++ {
		ksum += probs[order[i]]
	}
	target := s.next() * ksum
	var acc float32
	for i := 0; i < keep; i++ {
		acc += probs[order[i]]
		if acc >= target {
			return int32(order[i]), nil
		}
	}
	return int32(order[keep-1]), nil // floating-point fall-through
}
