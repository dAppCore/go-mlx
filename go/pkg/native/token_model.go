// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"

	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/model"
)

// NativeTokenModel binds the no-cgo decode backend + the embed/head bookend
// weights behind model.TokenModel, so model.Generate drives the whole token loop
// (embed → decode → head → sample) over the native path with no per-backend loop
// code. The decode runs whole-sequence through NativeBackend (model.Backend);
// the embed/head closures wrap the proven bookends — bf16 (EmbedTokensBF16 /
// LMHeadBF16) or 4-bit (EmbedTokensQuant / LMHeadQuant), set by the constructor,
// exactly as ArchSession/NewArchQuantSession carry their embed/head funcs.
// This is the native side of "the surface pkg/rocm drops into yields real
// tokens". E2B/E4B per-layer-input models work via the incremental session path
// (OpenSession + StepWithID); the whole-sequence DecodeForward does not do PLE.
type NativeTokenModel struct {
	*NativeBackend
	embed func(id int32) ([]byte, error)
	head  func(hidden []byte) ([]byte, error)
	vocab int
	// openSession builds a fresh persistent-cache decode session (ArchSession /
	// ArchQuantSession) — the incremental O(1)/token path model.Generate prefers
	// over the whole-sequence NativeBackend.DecodeForward. It takes the model's shardBuffers so the
	// session binds its weights as no-copy shard views (the directory-loaded model) rather than
	// uploading copies; a nil sb (in-memory model) uses the upload path.
	openSession func(*shardBuffers, *headEncoder) (model.DecodeStepper, error)
	// shards holds the memory-mapped checkpoint + per-shard no-copy Metal buffers when the model
	// was loaded zero-copy from a directory (LoadGemma4TokenModelDir). The embed/head closures and
	// the decode buffers reference VIEWS into these mmaps, so shards lives for the model's life
	// (and outlives any OpenSession session, which re-references the same weights). nil for a model
	// built from in-memory weight bytes. Close unmaps.
	shards *shardBuffers
	// headEnc is the zero-copy LM head (the per-token serve path: model.Generate's generateStepwise
	// calls m.Head every token). It binds the [vocab×dModel] head weight no-copy from the shard mmap,
	// resolved once — killing the per-token re-upload balloon. nil for an in-memory model (Head then
	// uses the upload closure). Concurrency-safe (no shared mutable state), so the shared model can
	// serve many request goroutines. Set by LoadGemma4TokenModelDir.
	headEnc *headEncoder
}

// Close releases a directory-loaded model's memory-mapped checkpoint (no-op when the weights are
// in-memory bytes). The resident decode/serve weights live for the process in the serve shape, so
// this is for explicit teardown (tests, a model hot-swap that drains first); do not Close while a
// Generate is in flight.
func (m *NativeTokenModel) Close() error {
	if m == nil {
		return nil
	}
	return m.shards.Close()
}

var _ model.SessionModel = (*NativeTokenModel)(nil)

// OpenSession opens a fresh incremental decode session (empty KV cache). This
// makes model.Generate run the native path O(1)/token (stepToken over a
// persistent cache) instead of re-decoding the whole sequence each token.
func (m *NativeTokenModel) OpenSession() (model.DecodeStepper, error) {
	return m.openSession(m.shards, m.headEnc)
}

// NewBF16TokenModel binds an assembled bf16 gemma4 (weights + arch) as a
// model.TokenModel — the contract-native generation path. Decode runs
// whole-sequence through NativeBackend (opts forwarded, e.g. WithICB); the LM
// head reads the arch's eps + soft-cap, the embed scale is √hidden. The arch
// must be PLE-free (12B/31B dense, 26B-A4B MoE, Ministral).
func NewBF16TokenModel(g *BF16Model, arch model.Arch, maxLen int, opts ...BackendOption) (*NativeTokenModel, error) {
	if g == nil || len(g.Layers) != len(arch.Layer) {
		return nil, core.NewError("native.NewBF16TokenModel: weights/arch layer count mismatch")
	}
	b, err := NewBF16Backend(arch, g.Layers, maxLen, opts...)
	if err != nil {
		return nil, err
	}
	scale := float32(math.Sqrt(float64(arch.Hidden)))
	vocab, dModel, eps, softCap := arch.Vocab, arch.Hidden, arch.Eps, arch.SoftCap
	tm := &NativeTokenModel{
		NativeBackend: b,
		vocab:         vocab,
		embed:         func(id int32) ([]byte, error) { return embedTokenBF16(g.Embed, id, vocab, dModel, scale) },
		head: func(hidden []byte) ([]byte, error) {
			return LMHeadBF16(hidden, g.FinalNorm, g.LMHead, dModel, vocab, eps, softCap)
		},
		openSession: func(sb *shardBuffers, head *headEncoder) (model.DecodeStepper, error) {
			return newArchSessionShardsWithHead(g, arch, maxLen, sb, head)
		},
	}
	he, herr := buildHeadEncoder(nil, g.FinalNorm, g.LMHead, nil, nil, dModel, vocab, 0, 0, eps, softCap, false)
	if herr != nil {
		return nil, herr
	}
	tm.headEnc = he
	return tm, nil
}

// NewQuantTokenModel binds an assembled 4-bit gemma4 (weights + arch) as a
// model.TokenModel — the quant sibling of NewBF16TokenModel. The embed/head wrap
// the 4-bit bookends (EmbedTokensQuant / LMHeadQuant) over the packed embedding +
// tied or separate head. E2B/E4B per-layer-input models are supported via the
// INCREMENTAL session path (OpenSession's ArchQuantSession threads the per-layer
// inputs through StepWithID); the whole-sequence DecodeForward fallback does not do
// PLE, so model.Generate (which prefers the session) is the path for those.
func NewQuantTokenModel(g *QuantModel, arch model.Arch, maxLen int, opts ...BackendOption) (*NativeTokenModel, error) {
	if g == nil || len(g.Layers) != len(arch.Layer) {
		return nil, core.NewError("native.NewQuantTokenModel: weights/arch layer count mismatch")
	}
	b, err := NewQuantBackend(arch, g.Layers, maxLen, opts...)
	if err != nil {
		return nil, err
	}
	scale := float32(math.Sqrt(float64(arch.Hidden)))
	vocab, dModel, eps, softCap := arch.Vocab, arch.Hidden, arch.Eps, arch.SoftCap
	gs, bits := g.GroupSize, g.Bits
	tm := &NativeTokenModel{
		NativeBackend: b,
		vocab:         vocab,
		embed: func(id int32) ([]byte, error) {
			return embedTokenQuant(g.Embed, g.EmbedScales, g.EmbedBiases, id, vocab, dModel, gs, bits, scale)
		},
		head: func(hidden []byte) ([]byte, error) {
			return LMHeadQuant(hidden, g.FinalNorm, g.LMHead, g.LMHeadScales, g.LMHeadBiases, dModel, vocab, gs, bits, eps, softCap)
		},
		openSession: func(sb *shardBuffers, head *headEncoder) (model.DecodeStepper, error) {
			return newArchQuantSessionShardsWithHead(g, arch, maxLen, sb, head)
		},
	}
	he, herr := buildHeadEncoder(nil, g.FinalNorm, g.LMHead, g.LMHeadScales, g.LMHeadBiases, dModel, vocab, gs, bits, eps, softCap, true)
	if herr != nil {
		return nil, herr
	}
	tm.headEnc = he
	return tm, nil
}

// Vocab is the logit width Greedy/Sample read — the LM head's output dimension.
func (m *NativeTokenModel) Vocab() int { return m.vocab }

// Embed gathers a token id's scaled input embedding (dModel bf16 bytes).
func (m *NativeTokenModel) Embed(id int32) ([]byte, error) { return m.embed(id) }

// Head maps a final hidden state to vocab logits (final norm + projection +
// optional soft-cap), bf16 bytes throughout. It prefers the zero-copy head (the head weight bound
// no-copy from the shard mmap, resolved once) when the model was loaded from a directory — the
// per-token serve path runs through here, so this is where the LM-head re-upload balloon is killed.
// Falls back to the upload closure for an in-memory model.
func (m *NativeTokenModel) Head(hidden []byte) ([]byte, error) {
	if m.headEnc != nil {
		return m.headEnc.encode(hidden, false) // Head returns logits to the caller (may sample) → apply the softcap
	}
	return m.head(hidden)
}
