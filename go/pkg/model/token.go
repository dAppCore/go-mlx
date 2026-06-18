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
// backend re-hand-rolls the generation loop: it supplies the byte-level pieces
// and inherits generation + sampling from here.
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
// Generate/GenerateSampled drive it. This is the minimal surface a backend MUST
// provide — the whole-sequence path. A backend that can ALSO decode
// incrementally over a persistent cache additionally implements SessionModel.
type TokenModel interface {
	Embedder
	Backend
	LMHead
	Vocab() int
}

// DecodeStepper is stateful single-token decode over a persistent KV cache: the
// cache is built when the stepper is opened and carries across Step calls, so a
// decode costs O(1) per token vs Backend.DecodeForward's whole-sequence O(n²)
// rebuild. Returned by SessionModel.OpenSession.
type DecodeStepper interface {
	// Step decodes one token embedding (dModel bf16 bytes) at the next cache
	// position, appends its K/V to the persistent cache, and returns the output
	// hidden state (dModel bf16 bytes).
	Step(emb []byte) ([]byte, error)
}

// SessionModel is a TokenModel whose backend can decode incrementally over a
// persistent cache — the OPTIONAL fast path. Generate prefers OpenSession's
// incremental stepper (O(1)/token) over the whole-sequence DecodeForward when a
// model provides it; a backend that can't maintain a persistent cache simply
// doesn't implement SessionModel and gets the whole-sequence loop, no contract
// change. Additive by design: the Backend/TokenModel surface a backend must
// implement is unchanged, so this rung never disrupts a backend mid-port.
type SessionModel interface {
	TokenModel
	// OpenSession opens a fresh stateful decode stepper with an empty cache (a
	// new generation starts at position 0).
	OpenSession() (DecodeStepper, error)
}

// generate is the shared validation + dispatch: it picks the incremental
// persistent-cache path (SessionModel) when the model offers it, else the
// whole-sequence fallback. pick is the only difference between greedy and
// sampled generation.
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
	if sm, ok := m.(SessionModel); ok {
		return generateStepwise(sm, promptIDs, maxNew, eos, pick)
	}
	return generateWholeSeq(m, promptIDs, maxNew, eos, pick)
}

// generateStepwise is the incremental path: open a persistent-cache session and
// step one token at a time (embed → Step), the cache carrying across steps so
// each token costs O(1). The decode tail (head → pick → append, eos/maxNew stop)
// is shared with the whole-sequence path in shape.
func generateStepwise(m SessionModel, promptIDs []int32, maxNew, eos int, pick func(logits []byte, vocab int) (int32, error)) ([]int32, error) {
	vocab := m.Vocab()
	sess, err := m.OpenSession()
	if err != nil {
		return nil, err
	}
	step := func(id int32) ([]byte, error) {
		emb, err := m.Embed(id)
		if err != nil {
			return nil, err
		}
		return sess.Step(emb)
	}

	var hidden []byte
	for _, id := range promptIDs { // prefill the prompt over the growing cache
		if hidden, err = step(id); err != nil {
			return nil, err
		}
	}
	gen := make([]int32, 0, maxNew)
	for len(gen) < maxNew {
		logits, err := m.Head(hidden) // the last token's state drives the next id
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
		if hidden, err = step(next); err != nil { // cache the generated token too
			return nil, err
		}
	}
	return gen, nil
}

// generateWholeSeq is the fallback for a backend without a persistent-cache
// session: embed the running sequence, run DecodeForward over it (rebuilding the
// KV cache each call → O(n²)), take the last hidden state, head → pick → append,
// re-embed the generated token, repeat. Correct for any backend; the incremental
// path supersedes it whenever a model implements SessionModel.
func generateWholeSeq(m TokenModel, promptIDs []int32, maxNew, eos int, pick func(logits []byte, vocab int) (int32, error)) ([]int32, error) {
	vocab := m.Vocab()
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
		logits, err := m.Head(hidden[len(hidden)-1])
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
// loop: backend-agnostic, pure Go, shared by every backend; incremental over a
// persistent cache when the model provides one (SessionModel), else
// whole-sequence.
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
