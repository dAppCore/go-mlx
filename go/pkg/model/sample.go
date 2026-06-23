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
	return greedySuppressed(logits, vocab, nil)
}

func greedySuppressed(logits []byte, vocab int, suppress []int32) (int32, error) {
	if len(logits) != vocab*bf16Size {
		return 0, core.NewError("model.Greedy: logits must be vocab bf16 bytes")
	}
	best, bestV := -1, float32(math.Inf(-1))
	for i := 0; i < vocab; i++ {
		if tokenSuppressed(int32(i), suppress) {
			continue
		}
		if v := bf16ToF32(logits[i*bf16Size], logits[i*bf16Size+1]); v > bestV {
			best, bestV = i, v
		}
	}
	if best < 0 {
		return 0, core.NewError("model.Greedy: all tokens are suppressed")
	}
	return int32(best), nil
}

// SampleParams configures stochastic sampling. Temperature <= 0 makes Sample greedy.
// TopK <= 0 disables the top-k cut; TopP <= 0 or >= 1 disables the nucleus cut. The two
// cuts compose (top-k first, then top-p over the kept set), matching the usual order.
type SampleParams struct {
	Temperature         float32
	TopK                int
	TopP                float32
	MinP                float32
	SuppressTokens      []int32
	MinTokensBeforeStop int
	RepeatPenalty       float32
}

// Sampler draws tokens with a reproducible RNG that ADVANCES per Sample call, so a
// generation loop gets a varied sequence from a single seed (vs re-seeding per token).
// Construct with NewSampler; Greedy draws are RNG-free so they don't perturb the state.
//
// A Sampler is NOT safe for concurrent use: its RNG state is mutable, and Sample reuses
// per-call scratch buffers held on it (the softmax/rank workspace, grown once to the vocab
// and reused — so a decode loop pays the vocab-sized allocation once, not per token). The
// served path constructs one Sampler per request (register_native.go), matching this.
type Sampler struct {
	state uint64

	// reusable softmax/rank scratch, grown to the vocab on first Sample and reused. The
	// per-token allocation of these three vocab-sized buffers (≈ the GenerateSampled path's
	// dominant heap bytes) is the AX-11 win: a 256k-vocab decode allocated ~4 MB/token here
	// before reuse. Sliced to [:vocab] and fully overwritten each call, so reuse is
	// arithmetically identical to a fresh make (the same values, the RNG drawn once as before).
	scaled, probs []float32
	order         []int
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
	if p.Temperature <= 0 && p.MinP <= 0 {
		return greedySuppressed(logits, vocab, p.SuppressTokens)
	}
	temp := p.Temperature
	if temp <= 0 {
		temp = 1
	}

	// grow-once, reuse-thereafter scratch (below the greedy guard so a zero-temp request stays
	// zero-alloc): each buffer is grown to the vocab on first need and reused on every later
	// Sample, then sliced to [:vocab] and FULLY overwritten below — so the result is identical
	// to allocating fresh, with the per-token vocab-sized allocations paid once per Sampler.
	if cap(s.scaled) < vocab {
		s.scaled = make([]float32, vocab)
		s.probs = make([]float32, vocab)
		s.order = make([]int, vocab)
	}
	scaled := s.scaled[:vocab]
	probs := s.probs[:vocab]
	order := s.order[:vocab]

	// temperature-scaled logits + their max (for a stable softmax).
	maxL := float32(math.Inf(-1))
	allowed := 0
	for i := 0; i < vocab; i++ {
		if tokenSuppressed(int32(i), p.SuppressTokens) {
			scaled[i] = float32(math.Inf(-1))
			continue
		}
		v := bf16ToF32(logits[i*bf16Size], logits[i*bf16Size+1]) / temp
		scaled[i] = v
		allowed++
		if v > maxL {
			maxL = v
		}
	}
	if allowed == 0 {
		return 0, core.NewError("model.Sample: all tokens are suppressed")
	}
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
	if p.MinP > 0 && keep > 0 {
		threshold := probs[order[0]] * p.MinP
		n := 0
		for n < keep && probs[order[n]] >= threshold {
			n++
		}
		if n > 0 {
			keep = n
		}
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

func tokenSuppressed(id int32, suppress []int32) bool {
	for _, token := range suppress {
		if id == token {
			return true
		}
	}
	return false
}

func applyRepeatPenaltyBF16(logits []byte, vocab int, history []int32, penalty float32) ([]byte, error) {
	if len(logits) != vocab*bf16Size {
		return nil, core.NewError("model.applyRepeatPenalty: logits must be vocab bf16 bytes")
	}
	if penalty <= 1 || len(history) == 0 {
		return logits, nil
	}
	ids := make([]int32, 0, len(history))
	for _, id := range history {
		if id >= 0 && int(id) < vocab {
			ids = append(ids, id)
		}
	}
	if len(ids) == 0 {
		return logits, nil
	}
	sort.Slice(ids, func(i, j int) bool { return ids[i] < ids[j] })
	out := make([]byte, len(logits))
	copy(out, logits)
	var prev int32
	for i, id := range ids {
		if i > 0 && id == prev {
			continue
		}
		prev = id
		off := int(id) * bf16Size
		v := bf16ToF32(out[off], out[off+1])
		if v > 0 {
			v /= penalty
		} else {
			v *= penalty
		}
		h := f32ToBF16(v)
		out[off] = byte(h)
		out[off+1] = byte(h >> 8)
	}
	return out, nil
}

func f32ToBF16(v float32) uint16 {
	bits := math.Float32bits(v)
	return uint16((bits + 0x7fff + ((bits >> 16) & 1)) >> 16)
}
