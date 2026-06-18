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
// Embed/Head wrap the proven bf16 bookends (EmbedTokensBF16 / LMHeadBF16). This
// is the native side of "the surface pkg/rocm drops into yields real tokens" — a
// quant sibling layers EmbedTokensQuant / LMHeadQuant the same way, and the PLE
// tower (E2B/E4B per-layer inputs) gates here once NativeBackend carries it.
type NativeTokenModel struct {
	*NativeBackend
	embedTable, finalNorm, lmHead []byte
	vocab, dModel                 int
	embedScale, eps, softCap      float32
}

var _ model.TokenModel = (*NativeTokenModel)(nil)

// NewBF16TokenModel binds an assembled bf16 gemma4 (weights + arch) as a
// model.TokenModel — the contract-native generation path. Decode runs
// whole-sequence through NativeBackend (opts forwarded, e.g. WithICB); the LM
// head reads the arch's eps + soft-cap, the embed scale is √hidden. The arch
// must be PLE-free (12B/31B dense, 26B-A4B MoE, Ministral); E2B/E4B need the
// per-layer-input tower wired into NativeBackend before they generate here.
func NewBF16TokenModel(g *Gemma4BF16, arch g4.Arch, maxLen int, opts ...BackendOption) (*NativeTokenModel, error) {
	if g == nil || len(g.Layers) != len(arch.Layer) {
		return nil, core.NewError("native.NewBF16TokenModel: weights/arch layer count mismatch")
	}
	b, err := NewBF16Backend(arch, g.Layers, maxLen, opts...)
	if err != nil {
		return nil, err
	}
	return &NativeTokenModel{
		NativeBackend: b,
		embedTable:    g.Embed,
		finalNorm:     g.FinalNorm,
		lmHead:        g.LMHead,
		vocab:         arch.Vocab,
		dModel:        arch.Hidden,
		embedScale:    float32(math.Sqrt(float64(arch.Hidden))),
		eps:           arch.Eps,
		softCap:       arch.SoftCap,
	}, nil
}

// Vocab is the logit width Greedy/Sample read — the LM head's output dimension.
func (m *NativeTokenModel) Vocab() int { return m.vocab }

// Embed gathers a token id's scaled input embedding (dModel bf16 bytes).
func (m *NativeTokenModel) Embed(id int32) ([]byte, error) {
	embs, err := EmbedTokensBF16(m.embedTable, []int32{id}, m.vocab, m.dModel, m.embedScale)
	if err != nil {
		return nil, err
	}
	return embs[0], nil
}

// Head maps a final hidden state to vocab logits (final norm + projection +
// optional soft-cap), bf16 bytes throughout.
func (m *NativeTokenModel) Head(hidden []byte) ([]byte, error) {
	return LMHeadBF16(hidden, m.finalNorm, m.lmHead, m.dModel, m.vocab, m.eps, m.softCap)
}
