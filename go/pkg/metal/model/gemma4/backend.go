// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/metal"
	"dappco.re/go/mlx/pkg/model"
)

// Gemma4Backend adapts the cgo (mlx-c) gemma4 model to pkg/model's backend-agnostic
// token-loop contract — the SECOND reference implementation alongside pkg/native's
// no-cgo NativeTokenModel, proving the contract is genuinely backend-agnostic across
// two real backends. The byte seam (not a tensor handle) is what makes that
// possible: activations cross as the model's own dtype bytes (intra-backend, so no
// conversion), and only the LM-head LOGITS are cast to bf16 — the one dtype the
// shared model.Greedy/Sample read.
//
// It is a REFERENCE adapter, not metal's production path: metal serves through its
// own fused engine (session, prompt cache, MTP). This wraps the model's pieces
// behind the contract so model.Generate drives the cgo backend the same way it
// drives native — DecodeForward injects precomputed embeddings into the proven
// stack via the embedHook (reusing forwardHiddenOverride, no change to the fused
// Forward), Head runs the same final-norm + projection + soft-cap tail. Whole-
// sequence only: an incremental DecodeStepper needs a session-lifecycle hook on the
// contract (metal caches are freed manually, unlike native's retained buffers) — a
// noted follow-up, not wired here.
type Gemma4Backend struct {
	m        *Gemma4Model
	dModel   int
	vocab    int
	actDType metal.DType // the model's activation dtype (bf16 real, f32 synthetic) — the seam transports it
	embBytes int         // dModel × element size of actDType
}

var (
	_ model.Backend      = (*Gemma4Backend)(nil)
	_ model.TokenModel   = (*Gemma4Backend)(nil)
	_ model.SessionModel = (*Gemma4Backend)(nil)
)

// NewBackend adapts a loaded gemma4 model to the token-loop contract. It rejects a
// per-layer-input model (E2B/E4B): the contract's embeddings-in DecodeForward has no
// token ids to derive the per-layer inputs from (the same limit pkg/native's
// NativeTokenModel has).
func NewBackend(m *Gemma4Model) (*Gemma4Backend, error) {
	if m == nil || m.Cfg == nil {
		return nil, core.NewError("gemma4.NewBackend: nil model")
	}
	if m.EmbedTokensPerLayer != nil {
		return nil, core.NewError("gemma4.NewBackend: per-layer-input models not supported via the token-loop contract")
	}
	dModel := int(m.Cfg.HiddenSize)
	// probe the activation dtype once (EmbedTokens output) — the seam transports it.
	probe := metal.FromValues([]int32{0}, 1, 1)
	pe := m.EmbedTokens.Forward(probe)
	actDType := pe.Dtype()
	metal.Free(probe, pe)
	elem := 4
	if actDType == metal.DTypeBFloat16 || actDType == metal.DTypeFloat16 {
		elem = 2
	}
	return &Gemma4Backend{m: m, dModel: dModel, vocab: int(m.Cfg.VocabSize), actDType: actDType, embBytes: dModel * elem}, nil
}

// Vocab is the logit width Greedy/Sample read.
func (b *Gemma4Backend) Vocab() int { return b.vocab }

// Embed gathers a token id's scaled input embedding (the model's activation dtype
// bytes) — the same EmbedTokens row × EmbeddingScale the fused Forward computes.
func (b *Gemma4Backend) Embed(id int32) ([]byte, error) {
	tok := metal.FromValues([]int32{id}, 1, 1)
	h := b.m.EmbedTokens.Forward(tok)
	scaled := metal.MulScalar(h, b.m.Cfg.EmbeddingScale)
	metal.Materialize(scaled)
	out := append([]byte(nil), scaled.RawBytes()...)
	metal.Free(tok, h, scaled)
	if len(out) != b.embBytes {
		return nil, core.NewError("gemma4.Embed: unexpected embedding byte length")
	}
	return out, nil
}

// DecodeForward runs the decoder stack on T precomputed embeddings (each the model's
// activation dtype, already scaled), returning the T pre-final-norm hidden states. It
// injects the embeddings into the proven stack via the embedHook (skipping the
// token-embed lookup) over a FRESH cache — the whole-sequence contract decode.
func (b *Gemma4Backend) DecodeForward(inputs [][]byte) ([][]byte, error) {
	T := len(inputs)
	if T == 0 {
		return nil, core.NewError("gemma4.DecodeForward: no inputs")
	}
	flat := make([]byte, 0, T*b.embBytes)
	for _, e := range inputs {
		if len(e) != b.embBytes {
			return nil, core.NewError("gemma4.DecodeForward: each input must be one embedding's bytes")
		}
		flat = append(flat, e...)
	}
	emb := metal.FromRawBytes(flat, []int{1, T, b.dModel}, b.actDType)
	tokens := metal.FromValues(make([]int32, T), 1, T) // ids drive only shape + (absent) PLE here

	// hand the precomputed embeddings to the stack in place of the (discarded) dummy
	// embed — the stack owns `emb` from here exactly as it owns the normal scaled embed.
	ov := &gemma4ForwardOverrides{embedHook: func(h *metal.Array) *metal.Array {
		metal.Free(h)
		return emb
	}}
	caches := b.m.NewCache()
	hidden, _, _ := b.m.forwardHiddenOverride(tokens, nil, caches, ov)
	metal.Materialize(hidden)
	raw := hidden.RawBytes()
	out := make([][]byte, T)
	ok := len(raw) == T*b.embBytes
	if ok {
		for t := 0; t < T; t++ {
			out[t] = append([]byte(nil), raw[t*b.embBytes:(t+1)*b.embBytes]...)
		}
	}
	metal.Free(tokens, hidden)
	metal.FreeCaches(caches)
	if !ok {
		return nil, core.NewError("gemma4.DecodeForward: hidden byte length mismatch")
	}
	return out, nil
}

// Head maps a final hidden state (the model's activation dtype bytes) to vocab logits:
// final RMSNorm + the output projection + the optional monotonic logit soft-cap —
// exactly the tail of the fused Forward — then casts to bf16, the dtype the shared
// model.Greedy/Sample read.
func (b *Gemma4Backend) Head(hidden []byte) ([]byte, error) {
	if len(hidden) != b.embBytes {
		return nil, core.NewError("gemma4.Head: hidden must be one hidden state's bytes")
	}
	h := metal.FromRawBytes(hidden, []int{1, 1, b.dModel}, b.actDType)
	normed := metal.RMSNorm(h, b.m.NormScaled, b.m.Cfg.RMSNormEps)
	out := b.m.Output.Forward(normed)
	metal.Free(h, normed)
	if b.m.Cfg.FinalLogitSoftcapping > 0 {
		softcapped := logitSoftcap(out, b.m.Cfg.FinalLogitSoftcapping)
		metal.Free(out)
		out = softcapped
	}
	logits := metal.AsType(out, metal.DTypeBFloat16) // the shared sampler reads bf16
	metal.Free(out)
	metal.Materialize(logits)
	res := append([]byte(nil), logits.RawBytes()...)
	metal.Free(logits)
	if len(res) != b.vocab*2 {
		return nil, core.NewError("gemma4.Head: unexpected logits byte length")
	}
	return res, nil
}

// OpenSession opens an incremental decode session with a fresh KV cache — the
// O(1)/token path model.Generate prefers over the whole-sequence DecodeForward.
// The returned stepper holds metal caches that its Close releases (Generate closes
// the stepper it opens — the contract's manual-memory lifecycle hook).
func (b *Gemma4Backend) OpenSession() (model.DecodeStepper, error) {
	return &gemma4Stepper{b: b, caches: b.m.NewCache()}, nil
}

// gemma4Stepper is the cgo backend's incremental DecodeStepper: it steps one token
// at a time over a PERSISTENT cache — the same embedHook injection DecodeForward
// uses, but T=1 and the cache carries across Step calls (so the N-token decode is N
// single-token steps, not N whole-sequence re-decodes).
type gemma4Stepper struct {
	b      *Gemma4Backend
	caches []metal.Cache
}

func (s *gemma4Stepper) Step(emb []byte) ([]byte, error) {
	b := s.b
	if len(emb) != b.embBytes {
		return nil, core.NewError("gemma4.Step: emb must be one embedding's bytes")
	}
	embArr := metal.FromRawBytes(emb, []int{1, 1, b.dModel}, b.actDType)
	tokens := metal.FromValues([]int32{0}, 1, 1)
	ov := &gemma4ForwardOverrides{embedHook: func(h *metal.Array) *metal.Array {
		metal.Free(h)
		return embArr
	}}
	hidden, _, _ := b.m.forwardHiddenOverride(tokens, nil, s.caches, ov)
	metal.Materialize(hidden)
	out := append([]byte(nil), hidden.RawBytes()...)
	ok := len(out) == b.embBytes
	metal.Free(tokens, hidden)
	if !ok {
		return nil, core.NewError("gemma4.Step: unexpected hidden byte length")
	}
	return out, nil
}

// Close releases the session's KV caches — the contract's manual-memory lifecycle
// hook (metal caches are freed explicitly, unlike native's retained buffers). Safe
// to call once; the stepper is spent afterwards.
func (s *gemma4Stepper) Close() error {
	if s.caches != nil {
		metal.FreeCaches(s.caches)
		s.caches = nil
	}
	return nil
}
