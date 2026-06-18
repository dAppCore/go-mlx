// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"

	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/model"
	g4 "dappco.re/go/mlx/pkg/model/gemma4"
)

// NativeTokenModel binds the no-cgo decode backend + the embed/head bookend
// weights behind model.TokenModel, so model.Generate drives the whole token loop
// (embed → decode → head → sample) over the native path with no per-backend loop
// code. The decode runs whole-sequence through NativeBackend (model.Backend);
// the embed/head closures wrap the proven bookends — bf16 (EmbedTokensBF16 /
// LMHeadBF16) or 4-bit (EmbedTokensQuant / LMHeadQuant), set by the constructor,
// exactly as Gemma4Session/NewGemma4QuantSession carry their embed/head funcs.
// This is the native side of "the surface pkg/rocm drops into yields real
// tokens"; the E2B/E4B per-layer-input tower gates here once NativeBackend
// carries it (NewQuantTokenModel rejects a PLE model until then).
type NativeTokenModel struct {
	*NativeBackend
	embed func(id int32) ([]byte, error)
	head  func(hidden []byte) ([]byte, error)
	vocab int
	// openSession builds a fresh persistent-cache decode session (Gemma4Session /
	// Gemma4QuantSession) — the incremental O(1)/token path model.Generate prefers
	// over the whole-sequence NativeBackend.DecodeForward.
	openSession func() (model.DecodeStepper, error)
}

var _ model.SessionModel = (*NativeTokenModel)(nil)

// OpenSession opens a fresh incremental decode session (empty KV cache). This
// makes model.Generate run the native path O(1)/token (stepToken over a
// persistent cache) instead of re-decoding the whole sequence each token.
func (m *NativeTokenModel) OpenSession() (model.DecodeStepper, error) { return m.openSession() }

// NewBF16TokenModel binds an assembled bf16 gemma4 (weights + arch) as a
// model.TokenModel — the contract-native generation path. Decode runs
// whole-sequence through NativeBackend (opts forwarded, e.g. WithICB); the LM
// head reads the arch's eps + soft-cap, the embed scale is √hidden. The arch
// must be PLE-free (12B/31B dense, 26B-A4B MoE, Ministral).
func NewBF16TokenModel(g *Gemma4BF16, arch g4.Arch, maxLen int, opts ...BackendOption) (*NativeTokenModel, error) {
	if g == nil || len(g.Layers) != len(arch.Layer) {
		return nil, core.NewError("native.NewBF16TokenModel: weights/arch layer count mismatch")
	}
	b, err := NewBF16Backend(arch, g.Layers, maxLen, opts...)
	if err != nil {
		return nil, err
	}
	scale := float32(math.Sqrt(float64(arch.Hidden)))
	vocab, dModel, eps, softCap := arch.Vocab, arch.Hidden, arch.Eps, arch.SoftCap
	return &NativeTokenModel{
		NativeBackend: b,
		vocab:         vocab,
		embed: func(id int32) ([]byte, error) {
			embs, err := EmbedTokensBF16(g.Embed, []int32{id}, vocab, dModel, scale)
			if err != nil {
				return nil, err
			}
			return embs[0], nil
		},
		head: func(hidden []byte) ([]byte, error) {
			return LMHeadBF16(hidden, g.FinalNorm, g.LMHead, dModel, vocab, eps, softCap)
		},
		openSession: func() (model.DecodeStepper, error) { return NewGemma4Session(g, arch, maxLen) },
	}, nil
}

// NewQuantTokenModel binds an assembled 4-bit gemma4 (weights + arch) as a
// model.TokenModel — the quant sibling of NewBF16TokenModel. Decode runs
// whole-sequence through the quant NativeBackend; the embed/head wrap the 4-bit
// bookends (EmbedTokensQuant / LMHeadQuant) over the packed embedding + tied or
// separate head. PLE models (E2B/E4B) are rejected until NativeBackend carries
// the per-layer-input tower.
func NewQuantTokenModel(g *Gemma4Quant, arch g4.Arch, maxLen int, opts ...BackendOption) (*NativeTokenModel, error) {
	if g == nil || len(g.Layers) != len(arch.Layer) {
		return nil, core.NewError("native.NewQuantTokenModel: weights/arch layer count mismatch")
	}
	if g.HasPLE() {
		return nil, core.NewError("native.NewQuantTokenModel: per-layer-input models (E2B/E4B) not yet supported via the token-loop contract")
	}
	b, err := NewQuantBackend(arch, g.Layers, maxLen, opts...)
	if err != nil {
		return nil, err
	}
	scale := float32(math.Sqrt(float64(arch.Hidden)))
	vocab, dModel, eps, softCap := arch.Vocab, arch.Hidden, arch.Eps, arch.SoftCap
	gs, bits := g.GroupSize, g.Bits
	return &NativeTokenModel{
		NativeBackend: b,
		vocab:         vocab,
		embed: func(id int32) ([]byte, error) {
			embs, err := EmbedTokensQuant(g.Embed, g.EmbedScales, g.EmbedBiases, []int32{id}, vocab, dModel, gs, bits, scale)
			if err != nil {
				return nil, err
			}
			return embs[0], nil
		},
		head: func(hidden []byte) ([]byte, error) {
			return LMHeadQuant(hidden, g.FinalNorm, g.LMHead, g.LMHeadScales, g.LMHeadBiases, dModel, vocab, gs, bits, eps, softCap)
		},
		openSession: func() (model.DecodeStepper, error) { return NewGemma4QuantSession(g, arch, maxLen) },
	}, nil
}

// Vocab is the logit width Greedy/Sample read — the LM head's output dimension.
func (m *NativeTokenModel) Vocab() int { return m.vocab }

// Embed gathers a token id's scaled input embedding (dModel bf16 bytes).
func (m *NativeTokenModel) Embed(id int32) ([]byte, error) { return m.embed(id) }

// Head maps a final hidden state to vocab logits (final norm + projection +
// optional soft-cap), bf16 bytes throughout.
func (m *NativeTokenModel) Head(hidden []byte) ([]byte, error) { return m.head(hidden) }
