// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"

	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/model"
	g4 "dappco.re/go/mlx/pkg/model/gemma4"
	"dappco.re/go/mlx/pkg/tokenizer"
)

// Gemma4Session is a PERSISTENT decode session: it holds the KV caches across calls, so a
// multi-turn conversation continues without re-prefilling the whole history — each Generate
// only prefills its new prompt and decodes, attending the cache built by previous turns.
//
// The resident buffers (caches + scratch, built once in NewGemma4Session over the
// archDecodeState) survive across the per-call autorelease pools because device.NewBuffer*
// returns a retained buffer (objc "new" = +1, not autoreleased); the Go session holds the
// reference, so they live until the session is dropped. Single-goroutine: the buffers and
// position are mutable session state with no synchronisation — drive one session from one
// goroutine (one session per conversation).
// Gemma4Session decodes against resident weights+caches; embed/head are the only
// representation-specific pieces (bf16 or 4-bit), so the prefill+decode loop is shared — set
// by NewGemma4Session (bf16) or NewGemma4QuantSession (4-bit).
type Gemma4Session struct {
	arch   g4.Arch
	embed  func(id int32) ([]byte, error)      // token id → its embedded bf16 vector (dModel bytes)
	head   func(hidden []byte) ([]byte, error) // hidden bf16 → vocab bf16 logits
	state  archDecodeState
	pos    int // tokens already in the cache (the next token decodes at this position)
	maxLen int
}

// NewGemma4Session builds a session over assembled bf16 weights: it allocates the resident
// per-layer buffers + caches once (empty), ready for Generate to fill incrementally.
func NewGemma4Session(g *Gemma4BF16, arch g4.Arch, maxLen int) (*Gemma4Session, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if g == nil || len(g.Layers) != len(arch.Layer) {
		return nil, core.NewError("native.NewGemma4Session: weights/arch layer count mismatch")
	}
	if maxLen <= 0 {
		return nil, core.NewError("native.NewGemma4Session: maxLen must be > 0")
	}
	attnScale := float32(1.0 / math.Sqrt(float64(arch.HeadDim)))
	embedScale := float32(math.Sqrt(float64(arch.Hidden)))
	var sess *Gemma4Session
	withAutoreleasePool(func() {
		lb, moeWeights := buildBF16ArchLayerBufs(g.Layers, arch.Layer, arch.Hidden, arch.Heads, arch.KVHeads, arch.HeadDim, arch.FF, maxLen)
		state := newArchDecodeState(arch.Layer, lb, moeWeights, arch.Hidden, arch.Heads, arch.KVHeads, arch.HeadDim, arch.FF, arch.SlidingWindow, arch.RotaryDim, arch.RotaryDimLocal, arch.RopeBase, arch.RopeLocalBase, attnScale, arch.Eps)
		sess = &Gemma4Session{
			arch: arch, state: state, maxLen: maxLen,
			embed: func(id int32) ([]byte, error) {
				embs, err := EmbedTokensBF16(g.Embed, []int32{id}, arch.Vocab, arch.Hidden, embedScale)
				if err != nil {
					return nil, err
				}
				return embs[0], nil
			},
			head: func(hidden []byte) ([]byte, error) {
				return LMHeadBF16(hidden, g.FinalNorm, g.LMHead, arch.Hidden, arch.Vocab, arch.Eps, arch.SoftCap)
			},
		}
	})
	return sess, nil
}

// NewGemma4QuantSession builds a persistent session over assembled 4-bit weights — the quant
// sibling of NewGemma4Session. Same resident caches + shared prefill/decode loop; only the
// embed/head closures differ (EmbedTokensQuant / LMHeadQuant over the packed embedding) and
// the layer buffers carry qmv projectors (buildQuantArchLayerBufs). Per-attention-type RoPE
// applies here too (the state is built with both bases).
func NewGemma4QuantSession(g *Gemma4Quant, arch g4.Arch, maxLen int) (*Gemma4Session, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if g == nil || len(g.Layers) != len(arch.Layer) {
		return nil, core.NewError("native.NewGemma4QuantSession: weights/arch layer count mismatch")
	}
	if maxLen <= 0 {
		return nil, core.NewError("native.NewGemma4QuantSession: maxLen must be > 0")
	}
	attnScale := float32(1.0 / math.Sqrt(float64(arch.HeadDim)))
	embedScale := float32(math.Sqrt(float64(arch.Hidden)))
	gs, bits := g.GroupSize, g.Bits
	var sess *Gemma4Session
	withAutoreleasePool(func() {
		lb := buildQuantArchLayerBufs(g.Layers, arch.Layer, arch.Hidden, arch.Heads, arch.KVHeads, arch.HeadDim, arch.FF, maxLen)
		moeWeights := make([]*MoELayerWeights, len(arch.Layer)) // quant path is non-MoE for now
		state := newArchDecodeState(arch.Layer, lb, moeWeights, arch.Hidden, arch.Heads, arch.KVHeads, arch.HeadDim, arch.FF, arch.SlidingWindow, arch.RotaryDim, arch.RotaryDimLocal, arch.RopeBase, arch.RopeLocalBase, attnScale, arch.Eps)
		sess = &Gemma4Session{
			arch: arch, state: state, maxLen: maxLen,
			embed: func(id int32) ([]byte, error) {
				embs, err := EmbedTokensQuant(g.Embed, g.EmbedScales, g.EmbedBiases, []int32{id}, arch.Vocab, arch.Hidden, gs, bits, embedScale)
				if err != nil {
					return nil, err
				}
				return embs[0], nil
			},
			head: func(hidden []byte) ([]byte, error) {
				return LMHeadQuant(hidden, g.FinalNorm, g.LMHead, g.LMHeadScales, g.LMHeadBiases, arch.Hidden, arch.Vocab, gs, bits, arch.Eps, arch.SoftCap)
			},
		}
	})
	return sess, nil
}

// Pos reports the number of tokens currently in the cache (the running sequence length).
func (s *Gemma4Session) Pos() int { return s.pos }

// GenerateText is the text-in/text-out wrapper over Generate, now that the tokenizer is a
// shared no-cgo package: it encodes prompt with tok, generates up to maxNew tokens (stopping
// at the tokenizer's EOS when it has one), and decodes the result back to a string. The
// session's cache carries over across calls, so successive GenerateText turns continue the
// conversation. The whole text → tokens → decode → text path runs with no cgo and no Python.
func (s *Gemma4Session) GenerateText(tok *tokenizer.Tokenizer, prompt string, maxNew int) (string, error) {
	if tok == nil {
		return "", core.NewError("native.Gemma4Session.GenerateText: nil tokenizer")
	}
	ids := tok.Encode(prompt)
	if len(ids) == 0 {
		return "", core.NewError("native.Gemma4Session.GenerateText: prompt encoded to no tokens")
	}
	eos := -1
	if tok.HasEOSToken() {
		eos = int(tok.EOSToken())
	}
	gen, err := s.Generate(ids, maxNew, eos)
	if err != nil {
		return "", err
	}
	return tok.Decode(gen), nil
}

// Generate appends promptIDs to the running sequence and greedily decodes up to maxNew
// tokens (or until eosID; eosID < 0 disables early stop), returning the generated ids.
// EVERY token — prompt and generated — is written to the persistent cache (the generated
// tokens too, so the sequence is complete), so a following Generate continues this exact
// sequence. The cache carries over until the session is dropped.
func (s *Gemma4Session) Generate(promptIDs []int32, maxNew, eosID int) ([]int32, error) {
	if len(promptIDs) == 0 {
		return nil, core.NewError("native.Gemma4Session.Generate: empty prompt")
	}
	if maxNew <= 0 {
		return nil, core.NewError("native.Gemma4Session.Generate: maxNew must be > 0")
	}
	if s.pos+len(promptIDs)+maxNew > s.maxLen {
		return nil, core.NewError("native.Gemma4Session.Generate: sequence would exceed maxLen cache rows")
	}
	gen := make([]int32, 0, maxNew)
	var genErr error
	withAutoreleasePool(func() {
		// step one token id at the current position, write its K/V to the cache, advance.
		step := func(id int32) ([]byte, error) {
			emb, err := s.embed(id)
			if err != nil {
				return nil, err
			}
			h, err := s.state.stepToken(emb, s.pos)
			if err != nil {
				return nil, err
			}
			s.pos++
			return h, nil
		}
		// prefill the new prompt over the carried-over cache; keep the last hidden state.
		var hidden []byte
		for _, id := range promptIDs {
			if hidden, genErr = step(id); genErr != nil {
				return
			}
		}
		// decode: head → greedy → append → step the new token (caching it for the next turn).
		for len(gen) < maxNew {
			logits, err := s.head(hidden)
			if err != nil {
				genErr = err
				return
			}
			next, err := model.Greedy(logits, s.arch.Vocab)
			if err != nil {
				genErr = err
				return
			}
			gen = append(gen, next)
			if hidden, genErr = step(next); genErr != nil { // cache the generated token too
				return
			}
			if eosID >= 0 && int(next) == eosID {
				break
			}
		}
	})
	return gen, genErr
}
