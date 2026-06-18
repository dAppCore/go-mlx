// SPDX-Licence-Identifier: EUPL-1.2

package model

import core "dappco.re/go"

// The token-loop contract — the rung above Backend that turns a backend's
// hidden-state decode into real tokens. Backend.DecodeForward runs the
// transformer stack (hidden → hidden); the two bookends here close the loop:
// Embedder maps a token id to its input hidden vector, LMHead maps a final
// hidden state to vocab logits. A backend that provides all three is a
// TokenModel, and Generate drives the full token-in → token-out loop over it —
// once, in pure Go, for every backend (native, metal, the future rocm). No
// backend re-hand-rolls the generation loop: it supplies the three byte-level
// pieces and inherits generation + sampling from here.
//
// Everything crosses the seam as bf16 []byte — the lingua franca QuantMatVec and
// Backend already use. An embedding is dModel bf16 bytes; logits are vocab bf16
// bytes. The model binds its arch (vocab, hidden size, embed scale, eps,
// soft-cap) at construction, so these methods carry only the per-call data.

// Embedder maps a token id to its input embedding: dModel bf16 bytes, already
// scaled (gemma4 scales the table row by sqrt(hidden)). The input bookend.
type Embedder interface {
	Embed(id int32) ([]byte, error)
}

// LMHead maps a final hidden state (dModel bf16 bytes) to vocab logits (vocab
// bf16 bytes): final norm + output projection + the optional monotonic
// soft-cap. The output bookend.
type LMHead interface {
	Head(hidden []byte) ([]byte, error)
}

// TokenModel is a backend that provides the whole token → token path: the two
// bookends plus the hidden-state decode (Backend), and the vocab size that sizes
// the logits Greedy/Sample read. native/metal/rocm each construct one;
// Generate/GenerateSampled drive it.
type TokenModel interface {
	Embedder
	Backend
	LMHead
	Vocab() int
}

// generate is the shared token loop: embed the running sequence, run the
// backend's DecodeForward over it, take the last hidden state, head it to
// logits, pick a token (pick), append it, re-embed it for the next step, and
// repeat until maxNew tokens or eos. pick is the only difference between greedy
// and sampled generation.
//
// This is whole-sequence today: DecodeForward rebuilds the KV cache per call, so
// the loop is O(n²) in sequence length. That is the correct-tokens milestone,
// not the tok/s one — incremental single-token decode with a persistent cache on
// the contract is the perf refinement (see Backend.DecodeForward).
func generate(m TokenModel, promptIDs []int32, maxNew, eos int, pick func(logits []byte, vocab int) (int32, error)) ([]int32, error) {
	if m == nil {
		return nil, core.NewError("model.Generate: nil model")
	}
	if len(promptIDs) == 0 {
		return nil, core.NewError("model.Generate: empty prompt")
	}
	if maxNew <= 0 {
		return nil, core.NewError("model.Generate: maxNew must be > 0")
	}
	vocab := m.Vocab()

	// embed the prompt into the running sequence of hidden vectors.
	seq := make([][]byte, 0, len(promptIDs)+maxNew)
	for _, id := range promptIDs {
		emb, err := m.Embed(id)
		if err != nil {
			return nil, err
		}
		seq = append(seq, emb)
	}

	gen := make([]int32, 0, maxNew)
	for len(gen) < maxNew {
		hidden, err := m.DecodeForward(seq)
		if err != nil {
			return nil, err
		}
		if len(hidden) == 0 {
			return nil, core.NewError("model.Generate: backend returned no hidden states")
		}
		logits, err := m.Head(hidden[len(hidden)-1]) // the last token's state drives the next id
		if err != nil {
			return nil, err
		}
		next, err := pick(logits, vocab)
		if err != nil {
			return nil, err
		}
		gen = append(gen, next)
		if eos >= 0 && int(next) == eos {
			break
		}
		if len(gen) >= maxNew {
			break
		}
		emb, err := m.Embed(next) // re-embed the generated token for the next step
		if err != nil {
			return nil, err
		}
		seq = append(seq, emb)
	}
	return gen, nil
}

// Generate greedily decodes up to maxNew tokens from a TokenModel, starting from
// promptIDs; eos < 0 disables early stop. Deterministic (no RNG) — the natural
// closer for a correctness gate or a greedy bench. The contract-level token
// loop: backend-agnostic, pure Go, shared by every backend.
func Generate(m TokenModel, promptIDs []int32, maxNew, eos int) ([]int32, error) {
	return generate(m, promptIDs, maxNew, eos, Greedy)
}

// GenerateSampled is Generate with stochastic sampling: the same loop, drawing
// each token from the logits via the Sampler + SampleParams (temperature, then
// optional top-k and top-p) instead of greedy. p.Temperature <= 0 falls back to
// greedy per token (so a zero-temp request is deterministic).
func GenerateSampled(m TokenModel, s *Sampler, p SampleParams, promptIDs []int32, maxNew, eos int) ([]int32, error) {
	if s == nil {
		return nil, core.NewError("model.GenerateSampled: nil sampler")
	}
	return generate(m, promptIDs, maxNew, eos, func(logits []byte, vocab int) (int32, error) {
		return s.Sample(logits, vocab, p)
	})
}
