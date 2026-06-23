// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"unsafe"

	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/model"
	"dappco.re/go/mlx/pkg/tokenizer"
	"github.com/tmc/apple/metal"
)

// ArchSession is a PERSISTENT decode session: it holds the KV caches across calls, so a
// multi-turn conversation continues without re-prefilling the whole history — each Generate
// only prefills its new prompt and decodes, attending the cache built by previous turns.
//
// The resident buffers (caches + scratch, built once in NewArchSession over the
// archDecodeState) survive across the per-call autorelease pools because device.NewBuffer*
// returns a retained buffer (objc "new" = +1, not autoreleased); the Go session holds the
// reference, so they live until the session is dropped. Single-goroutine: the buffers and
// position are mutable session state with no synchronisation — drive one session from one
// goroutine (one session per conversation).
// ArchSession decodes against resident weights+caches; embed/head are the only
// representation-specific pieces (bf16 or 4-bit), so the prefill+decode loop is shared — set
// by NewArchSession (bf16) or NewArchQuantSession (4-bit).
type ArchSession struct {
	arch    model.Arch
	embed   func(id int32) ([]byte, error)                             // token id → its embedded bf16 vector (dModel bytes)
	head    func(hidden []byte, skipSoftcap bool) ([]byte, error)      // hidden bf16 → vocab bf16 logits; skipSoftcap for argmax callers
	greedy  func(hidden []byte, suppress []int32) (int32, bool, error) // optional direct greedy token path; ok=false falls back to head+Greedy
	headEnc *headEncoder
	// perLayerInput, when set (gemma4 E2B/E4B), computes the per-token PerLayerInputs tensor
	// from the token id + its embedding; Generate sets it on the state before stepToken. nil
	// for models without the PLE tower.
	perLayerInput func(id int32, emb []byte) ([]byte, error)
	// encNextInputsGPU, when set (e2b: 4-bit main+PLE embedding, bf16 PLE projection), encodes the GPU
	// embed-gather (token → embOut, dModel) + the GPU PLE (token, embOut → sc.out, numLayers·pliDim) for
	// one token read from tokenBuf into a shared encoder — the NEXT decode step's emb+pli produced on-GPU
	// with no host round-trip (the submit-ahead pipeline seam). nil → the host embed/PLE path stays.
	encNextInputsGPU func(enc metal.MTLComputeCommandEncoder, tokenBuf, embOut metal.MTLBuffer, sc *plGPUScratch) error
	plScratchNew     func() *plGPUScratch
	state            archDecodeState
	pos           int // tokens already in the cache (the next token decodes at this position)
	maxLen        int
	// cachedIDs are the token ids currently resident in the KV cache (prompt + generated), tracked so
	// GenerateCached can reuse the longest shared prefix of a new prompt and re-prefill only the suffix.
	cachedIDs []int32
	// cachedPromptIDs/cachedPromptHidden/cachedPromptLogits capture the exact prompt boundary. This
	// mirrors metal's prompt-cache entry hidden/logits replay: an exact prompt hit can decode
	// immediately from saved state instead of re-prefilling the last prompt token or re-running the
	// first head projection just to recreate it.
	cachedPromptIDs    []int32
	cachedPromptHidden []byte
	cachedPromptLogits []byte
	// retainedHidden is the hidden state at the current session boundary. It is
	// the native equivalent of metal's retained logits boundary for token-only
	// session operation: PrefillTokens/AppendTokens populate it, and
	// GenerateFromCache can continue without requiring a new prompt token.
	retainedHidden []byte
	// shards holds the memory-mapped checkpoint + its per-shard no-copy Metal buffers when the
	// session was loaded from a directory zero-copy (LoadGemma4*Dir). The weight []byte fields the
	// embed/head closures and the decode buffers reference are VIEWS into these mmaps, so shards
	// MUST stay alive for the session's life; Close unmaps them. nil for a session built from
	// in-memory weight bytes (NewArchSession over an already-parsed BF16Model) — those weights
	// are heap-owned, nothing to unmap.
	shards *shardBuffers
}

// Close releases a directory-loaded session's memory-mapped checkpoint. It is safe on a session
// built from in-memory bytes (shards nil ⇒ no-op) and idempotent. Call it once decoding is done;
// the no-copy weight buffers reference the mmap, so do not Close while a Generate/Step is in
// flight (single-goroutine sessions make that the caller's natural discipline).
func (s *ArchSession) Close() error {
	if s == nil {
		return nil
	}
	return s.shards.Close()
}

// NewArchSession builds a session over assembled bf16 weights: it allocates the resident
// per-layer buffers + caches once (empty), ready for Generate to fill incrementally. The weights
// are uploaded into owned Metal buffers (the in-memory path). The directory loader uses
// newArchSessionShards to bind them zero-copy from the shard mmaps instead.
func NewArchSession(g *BF16Model, arch model.Arch, maxLen int) (*ArchSession, error) {
	return newArchSessionShards(g, arch, maxLen, nil)
}

// newArchSessionShards is NewArchSession with an optional zero-copy weight source: when sb is
// non-nil, every per-layer + bookend weight is bound as a no-copy view into the shard mmaps (no
// upload, no second resident copy); when nil, the weights are uploaded into owned buffers (the
// in-memory path). The decode is byte-identical either way — only the weight binding differs.
func newArchSessionShards(g *BF16Model, arch model.Arch, maxLen int, sb *shardBuffers) (*ArchSession, error) {
	return newArchSessionShardsWithHead(g, arch, maxLen, sb, nil)
}

func newArchSessionShardsWithHead(g *BF16Model, arch model.Arch, maxLen int, sb *shardBuffers, sharedHead *headEncoder) (*ArchSession, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if g == nil || len(g.Layers) != len(arch.Layer) {
		return nil, core.NewError("native.NewArchSession: weights/arch layer count mismatch")
	}
	if maxLen <= 0 {
		return nil, core.NewError("native.NewArchSession: maxLen must be > 0")
	}
	attnScale := attnScaleOf(arch)
	embedScale := float32(math.Sqrt(float64(arch.Hidden)))
	var sess *ArchSession
	var buildErr error
	withAutoreleasePool(func() {
		lb, moeWeights, berr := buildBF16ArchLayerBufs(g.Layers, arch.Layer, arch.Hidden, arch.Heads, arch.KVHeads, arch.HeadDim, arch.FF, maxLen, arch.SlidingWindow, sb)
		if berr != nil {
			buildErr = berr
			return
		}
		state := newArchDecodeState(arch.Layer, lb, moeWeights, arch.Hidden, arch.Heads, arch.KVHeads, arch.HeadDim, arch.FF, arch.SlidingWindow, arch.RotaryDim, arch.RotaryDimLocal, arch.RopeBase, arch.RopeLocalBase, attnScale, arch.Eps, arch.ValueNorm)
		state.ropeFreqs = uploadRopePeriods(arch.RopeFreqs) // YaRN long-context spectrum (nil ⇒ base rope)
		// gemma4 per-layer-input tower (E2B/E4B), bf16 sibling of the quant session: the per-layer
		// gates carry bf16 bytes (bits 0 ⇒ the decode applies PerLayerInputGateBF16, not the qmv).
		if g.HasPLE() {
			state.pliDim = arch.PerLayerInputHidden
			state.ple = make([]pleLayer, len(g.Layers))
			for i := range g.Layers {
				if len(g.Layers[i].PostPerLayerInputNormW) > 0 {
					state.ple[i] = pleLayer{
						gate:     QuantWeight{Packed: g.Layers[i].PerLayerGate},
						proj:     QuantWeight{Packed: g.Layers[i].PerLayerProjection},
						postNorm: g.Layers[i].PostPerLayerInputNormW,
					}
				}
			}
		}
		// zero-copy head: bind the [vocab×dModel] head weight no-copy, resolved once, reused every
		// token (kills the per-token re-upload balloon). nil ⇒ no shards / unresolved ⇒ upload head.
		head := sharedHead
		if head == nil {
			var herr error
			head, herr = newHeadEncoder(sb, g.FinalNorm, g.LMHead, nil, nil, arch.Hidden, arch.Vocab, 0, 0, arch.Eps, arch.SoftCap, false)
			if herr != nil {
				buildErr = herr
				return
			}
		}
		sess = &ArchSession{
			arch: arch, state: state, maxLen: maxLen, headEnc: head,
			embed: func(id int32) ([]byte, error) {
				return embedTokenBF16(g.Embed, id, arch.Vocab, arch.Hidden, embedScale)
			},
			head: func(hidden []byte, skipSoftcap bool) ([]byte, error) {
				if head != nil {
					return head.encode(hidden, skipSoftcap)
				}
				sc := arch.SoftCap
				if skipSoftcap {
					sc = 0 // LMHeadBF16 skips the softcap when softCap<=0
				}
				return LMHeadBF16(hidden, g.FinalNorm, g.LMHead, arch.Hidden, arch.Vocab, arch.Eps, sc)
			},
			greedy: func(hidden []byte, suppress []int32) (int32, bool, error) {
				if head == nil {
					return 0, false, nil
				}
				return head.greedy(hidden, suppress)
			},
		}
		if g.HasPLE() {
			var pleProjView bufView // resident no-copy bf16 PLE projection — bound once at its shard offset, not re-uploaded per token
			if sb != nil {
				pleProjView, _ = sb.bufFor(g.PerLayerModelProjW)
			}
			sess.perLayerInput = func(id int32, emb []byte) ([]byte, error) {
				pv := pleProjView
				if pleResidentDisabled { // call-time host-path toggle (byte-identity test hook; always false in production)
					pv = bufView{}
				}
				return PerLayerInputs(g.EmbedPerLayer, nil, nil, g.PerLayerModelProjW, nil, nil, g.PerLayerProjNormW, id, emb, arch.PerLayerInputVocab, len(arch.Layer), arch.PerLayerInputHidden, arch.Hidden, 0, 0, 0, 0, arch.Eps, pv)
			}
		}
	})
	if buildErr != nil {
		return nil, buildErr
	}
	return sess, nil
}

// NewArchQuantSession builds a persistent session over assembled 4-bit weights — the quant
// sibling of NewArchSession. Same resident caches + shared prefill/decode loop; only the
// embed/head closures differ (EmbedTokensQuant / LMHeadQuant over the packed embedding) and
// the layer buffers carry qmv projectors (buildQuantArchLayerBufs). Per-attention-type RoPE
// applies here too (the state is built with both bases).
func NewArchQuantSession(g *QuantModel, arch model.Arch, maxLen int) (*ArchSession, error) {
	return newArchQuantSessionShards(g, arch, maxLen, nil)
}

// newArchQuantSessionShards is NewArchQuantSession with an optional zero-copy weight source.
// sb is kept alive on the session (the host-side embed/head read mmap views of g.Embed / g.LMHead),
// BUT the per-layer 4-bit weights are deliberately built via the COPY path (buildQuantArchLayerBufs
// is passed nil): binding the per-layer quant weights as no-copy views into the shared shard buffer
// produces NaN once a SECOND decode layer reads the first layer's output — a cross-layer hazard
// specific to the 4-bit affine_qmv reading the aliased shard buffer in a multi-layer command buffer
// (the bf16 gemv path and a single quant layer are byte-identical no-copy; isolated/repeated quant
// qmv over the shard buffer is byte-identical too — it is purely the cross-layer multi-bind case).
// Until that is understood the quant layer weights stay copies (no balloon — they are built ONCE),
// while the bf16 path and the per-token head (a single dispatch, split (d)) take the zero-copy win.
func newArchQuantSessionShards(g *QuantModel, arch model.Arch, maxLen int, sb *shardBuffers) (*ArchSession, error) {
	return newArchQuantSessionShardsWithHead(g, arch, maxLen, sb, nil)
}

func newArchQuantSessionShardsWithHead(g *QuantModel, arch model.Arch, maxLen int, sb *shardBuffers, sharedHead *headEncoder) (*ArchSession, error) {
	if err := ensureInit(); err != nil {
		return nil, err
	}
	if g == nil || len(g.Layers) != len(arch.Layer) {
		return nil, core.NewError("native.NewArchQuantSession: weights/arch layer count mismatch")
	}
	if maxLen <= 0 {
		return nil, core.NewError("native.NewArchQuantSession: maxLen must be > 0")
	}
	attnScale := attnScaleOf(arch)
	embedScale := float32(math.Sqrt(float64(arch.Hidden)))
	gs, bits := g.GroupSize, g.Bits
	var sess *ArchSession
	var buildErr error
	withAutoreleasePool(func() {
		// sb (no-copy) for the per-layer quant weights. The documented "cross-layer multi-bind NaN"
		// hypothesis = the packed uint32 weights bound at non-4-aligned offsets (Metal can't do a
		// misaligned uint32 read); bufFor now copies only those (mustBufFor4), aligned stay zero-copy.
		// If the smoke is coherent this reclaims the 4-bit 2× resident; if not, revert to nil.
		lb, moeQuant, berr := buildQuantArchLayerBufs(g.Layers, arch.Layer, arch.Hidden, arch.Heads, arch.KVHeads, arch.HeadDim, arch.FF, maxLen, arch.SlidingWindow, sb)
		if berr != nil {
			buildErr = berr
			return
		}
		moeWeights := make([]*MoELayerWeights, len(arch.Layer)) // bf16 MoE unused on the quant path
		state := newArchDecodeState(arch.Layer, lb, moeWeights, arch.Hidden, arch.Heads, arch.KVHeads, arch.HeadDim, arch.FF, arch.SlidingWindow, arch.RotaryDim, arch.RotaryDimLocal, arch.RopeBase, arch.RopeLocalBase, attnScale, arch.Eps, arch.ValueNorm)
		state.moeQuant = moeQuant
		// gemma4 per-layer-input tower (E2B/E4B): the per-layer gates + the per-token tensor.
		if g.HasPLE() {
			state.pliDim = arch.PerLayerInputHidden
			state.ple = make([]pleLayer, len(g.Layers))
			for i := range g.Layers {
				if len(g.Layers[i].PostPerLayerInputNormW) > 0 {
					state.ple[i] = pleLayer{
						gate: g.Layers[i].PerLayerGate, proj: g.Layers[i].PerLayerProjection,
						postNorm: g.Layers[i].PostPerLayerInputNormW, groupSize: gs, bits: bits,
					}
				}
			}
		}
		// zero-copy 4-bit head: bind the tied [vocab×dModel] packed embedding + scales/biases no-copy,
		// resolved once, reused every token — this is the projection the per-token balloon lived on
		// (the ~503 MB tied embedding re-uploaded per token at 12B). A single qmv dispatch over the
		// shard buffer is byte-identical (the cross-layer hazard that gates the quant LAYER weights
		// does not apply to a one-shot head). nil ⇒ no shards / unresolved ⇒ the upload head.
		head := sharedHead
		if head == nil {
			var herr error
			head, herr = newHeadEncoder(sb, g.FinalNorm, g.LMHead, g.LMHeadScales, g.LMHeadBiases, arch.Hidden, arch.Vocab, gs, bits, arch.Eps, arch.SoftCap, true)
			if herr != nil {
				buildErr = herr
				return
			}
		}
		sess = &ArchSession{
			arch: arch, state: state, maxLen: maxLen, headEnc: head,
			embed: func(id int32) ([]byte, error) {
				return embedTokenQuant(g.Embed, g.EmbedScales, g.EmbedBiases, id, arch.Vocab, arch.Hidden, gs, bits, embedScale)
			},
			head: func(hidden []byte, skipSoftcap bool) ([]byte, error) {
				if head != nil {
					return head.encode(hidden, skipSoftcap)
				}
				sc := arch.SoftCap
				if skipSoftcap {
					sc = 0 // LMHeadQuant skips the softcap when softCap<=0
				}
				return LMHeadQuant(hidden, g.FinalNorm, g.LMHead, g.LMHeadScales, g.LMHeadBiases, arch.Hidden, arch.Vocab, gs, bits, arch.Eps, sc)
			},
			greedy: func(hidden []byte, suppress []int32) (int32, bool, error) {
				if head == nil {
					return 0, false, nil
				}
				return head.greedy(hidden, suppress)
			},
		}
		if g.HasPLE() {
			var pleProjView bufView // resident no-copy PLE projection when it's bf16 (e2b: no proj scales) — bound once, not re-uploaded per token
			if sb != nil && len(g.PerLayerModelProjScales) == 0 {
				pleProjView, _ = sb.bufFor(g.PerLayerModelProjW)
			}
			sess.perLayerInput = func(id int32, emb []byte) ([]byte, error) {
				pv := pleProjView
				if pleResidentDisabled { // call-time host-path toggle (byte-identity test hook; always false in production)
					pv = bufView{}
				}
				return PerLayerInputs(g.EmbedPerLayer, g.EmbedPerLayerScales, g.EmbedPerLayerBiases, g.PerLayerModelProjW, g.PerLayerModelProjScales, g.PerLayerModelProjBiases, g.PerLayerProjNormW, id, emb, arch.PerLayerInputVocab, len(arch.Layer), arch.PerLayerInputHidden, arch.Hidden, gs, bits, g.PerLayerModelProjGS, g.PerLayerModelProjBits, arch.Eps, pv)
			}
			// GPU next-inputs seam: produce the next step's emb+pli on-GPU from a token-id buffer (no host
			// round-trip), the submit-ahead pipeline's gate. Handles e2b's shape only — 4-bit main + PLE
			// embedding, bf16 PLE projection; other shapes leave it nil and keep the host path.
			if bits == 4 && len(g.EmbedPerLayerScales) > 0 && len(g.PerLayerModelProjScales) == 0 {
				numLayers, pliDim, dModel := len(arch.Layer), arch.PerLayerInputHidden, arch.Hidden
				plDim := numLayers * pliDim
				embScalePLE := float32(math.Sqrt(float64(pliDim)))
				projScale := float32(1.0 / math.Sqrt(float64(dModel)))
				projWBuf, projWOff := pleProjView.buf, pleProjView.off
				sess.plScratchNew = func() *plGPUScratch { return newPLGPUScratch(plDim, projScale) }
				sess.encNextInputsGPU = func(enc metal.MTLComputeCommandEncoder, tokenBuf, embOut metal.MTLBuffer, sc *plGPUScratch) error {
					gpso, gerr := embedGatherPipeline()
					if gerr != nil {
						return gerr
					}
					encEmbedGatherQuant(enc, gpso, tokenBuf, residentBytes(g.Embed), residentBytes(g.EmbedScales), residentBytes(g.EmbedBiases), embOut, 0, 0, 0, dModel, gs, bits, embedScale)
					pw, pwOff := projWBuf, projWOff
					if pw == nil {
						pw, pwOff = residentBytes(g.PerLayerModelProjW), 0
					}
					return encPerLayerInputsGPU(enc, gpso, tokenBuf, embOut, residentBytes(g.EmbedPerLayer), residentBytes(g.EmbedPerLayerScales), residentBytes(g.EmbedPerLayerBiases), 0, 0, 0, pw, pwOff, residentBytes(g.PerLayerProjNormW), sc, numLayers, pliDim, dModel, gs, bits, embScalePLE, arch.Eps)
				}
			}
		}
		// gemma4 incremental ICB encode-bypass (E2B/E4B dense): record the decode stack once + replay
		// it per Step/StepWithID instead of re-encoding every layer. The replay holds its OWN linear
		// maxLen caches (the session's lb sliding caches are RING-sized + unused on this path); the PLE
		// runtime wraps the session's own perLayerInput closure (the per-token tensor stays host-side).
		if sess.icbEligible() {
			var pleRuntime *archDecodePLEInputs
			if g.HasPLE() {
				pleRuntime = &archDecodePLEInputs{compute: sess.perLayerInput}
			}
			kCaches := make([]metal.MTLBuffer, len(arch.Layer))
			vCaches := make([]metal.MTLBuffer, len(arch.Layer))
			for li := range arch.Layer {
				if arch.Layer[li].OwnsCache() { // per-layer linear maxLen cache — global layers' rows are wider
					cacheBytes := uint(maxLen * arch.KVHeads * headDimOf(arch.Layer[li], arch.HeadDim) * bf16Size)
					kCaches[li] = device.NewBufferWithLengthOptions(cacheBytes, metal.MTLResourceStorageModeShared)
					vCaches[li] = device.NewBufferWithLengthOptions(cacheBytes, metal.MTLResourceStorageModeShared)
				}
			}
			rope := icbRope{
				base: arch.RopeBase, localBase: arch.RopeLocalBase,
				rotaryDim: arch.RotaryDim, rotaryDimLocal: arch.RotaryDimLocal,
				globalHeadDim: state.globalHeadDim,
				globalFreqs:   state.globalRopeFreqs, freqs: state.ropeFreqs,
			}
			rep, rerr := recordArchICBQuant(g.Layers, arch.Layer, kCaches, vCaches, pleRuntime, arch.PerLayerInputHidden, gs, bits, arch.Hidden, arch.Heads, arch.KVHeads, arch.HeadDim, maxLen, arch.FF, arch.SlidingWindow, rope, attnScale, arch.Eps, arch.ValueNorm)
			if rerr != nil {
				buildErr = rerr
				return
			}
			sess.state.icb = rep
		}
	})
	if buildErr != nil {
		return nil, buildErr
	}
	return sess, nil
}

// icbEligible reports whether this session can replay a recorded arch ICB instead of re-encoding
// per token. The ICB core (decodeForwardArchICBCore) assumes the SIMPLE uniform decode: no MoE
// (host router), no trace (per-layer host reads), uniform head geometry, and simple uniform rope
// (single base, no YaRN spectrum, no proportional-global). A model that varies any of those falls
// back to stepToken — byte-identical, just not encode-bypassed.
func (s *ArchSession) icbEligible() bool {
	if s.state.trace {
		return false
	}
	for li := range s.state.specs {
		sp := s.state.specs[li]
		// uniform head geometry only — the ICB cache rowBytes + SDPA PSO are per-headDim, and the
		// proportional-global rope dispatches over globalHeadDim (==headDim when uniform). Per-layer
		// base / sliding theta / proportional + YaRN spectra ARE now recorded per layer (icbRope).
		// per-layer head dim is now recorded (the ICB sizes scratch/cache + picks SDPA PSO + dim
		// buffers per hd); kvHeads must stay uniform (the GQA buffer + SDPA strides assume it).
		if sp.MoE || kvHeadsOf(sp, s.state.nKVHeads) != s.state.nKVHeads {
			return false
		}
	}
	return true
}

// Pos reports the number of tokens currently in the cache (the running sequence length).
func (s *ArchSession) Pos() int { return s.pos }

var _ model.DecodeStepper = (*ArchSession)(nil)

// TokenTransform observes the selected token ID and returns the ID that should
// actually be committed into the resident decode cache. It is used for engine
// features such as thinking-budget close forcing, where changing only the
// streamed text would leave the cache conditioned on the wrong token.
type TokenTransform func(int32) int32

// PrefillTokens resets the retained decode state and prefills already-tokenised
// prompt ids into the resident KV cache. It is the token-native sibling of
// pkg/metal's ModelSession.PrefillTokens.
func (s *ArchSession) PrefillTokens(ids []int32) error {
	if len(ids) == 0 {
		return core.NewError("native.ArchSession.PrefillTokens: empty prompt tokens")
	}
	if len(ids) > s.maxLen {
		return core.NewError("native.ArchSession.PrefillTokens: sequence would exceed maxLen cache rows")
	}
	s.pos = 0
	s.resetCachedPromptEntry()
	s.resetRetainedHidden()
	resident := s.cachedIDs[:0]
	s.cachedIDs = resident
	hidden, err := s.prefillRetainedTokens(ids, "native.ArchSession.PrefillTokens")
	if err != nil {
		s.pos = 0
		s.cachedIDs = resident[:0]
		s.resetRetainedHidden()
		return err
	}
	s.cachedIDs = append(resident, ids...)
	s.rememberRetainedHidden(hidden)
	return nil
}

// AppendTokens appends already-tokenised prompt ids to the retained session
// state without replaying the existing prefix.
func (s *ArchSession) AppendTokens(ids []int32) error {
	if len(ids) == 0 {
		return core.NewError("native.ArchSession.AppendTokens: empty prompt tokens")
	}
	if s.pos == 0 || len(s.retainedHidden) != s.arch.Hidden*bf16Size {
		return core.NewError("native.ArchSession.AppendTokens: no retained prefill state")
	}
	if s.pos+len(ids) > s.maxLen {
		return core.NewError("native.ArchSession.AppendTokens: sequence would exceed maxLen cache rows")
	}
	hidden, err := s.prefillRetainedTokens(ids, "native.ArchSession.AppendTokens")
	if err != nil {
		s.cachedIDs = nil
		s.resetRetainedHidden()
		return err
	}
	s.cachedIDs = append(s.cachedIDs, ids...)
	s.clearCachedPromptHidden()
	s.rememberRetainedHidden(hidden)
	return nil
}

// GenerateFromCache greedily generates from the retained session boundary
// populated by PrefillTokens, AppendTokens, WarmPromptCache, Generate, or
// GenerateCached. No new prompt token is required.
func (s *ArchSession) GenerateFromCache(maxNew, eosID int) ([]int32, error) {
	return s.GenerateFromCacheEach(maxNew, eosID, nil)
}

// GenerateFromCacheEach is GenerateFromCache with per-token streaming.
func (s *ArchSession) GenerateFromCacheEach(maxNew, eosID int, yield func(int32) bool) ([]int32, error) {
	return s.GenerateFromCacheEachTransformed(maxNew, eosID, nil, yield)
}

// GenerateFromCacheEachTransformed is GenerateFromCacheEach with a committed-token
// transform applied before each generated token is written to the cache.
func (s *ArchSession) GenerateFromCacheEachTransformed(maxNew, eosID int, transform TokenTransform, yield func(int32) bool) ([]int32, error) {
	if maxNew <= 0 {
		return nil, core.NewError("native.ArchSession.GenerateFromCache: maxNew must be > 0")
	}
	if len(s.retainedHidden) != s.arch.Hidden*bf16Size {
		return nil, core.NewError("native.ArchSession.GenerateFromCache: no retained prefill state")
	}
	if s.pos+maxNew > s.maxLen {
		return nil, core.NewError("native.ArchSession.GenerateFromCache: sequence would exceed maxLen cache rows")
	}
	hidden := s.retainedHidden
	var gen []int32
	var err error
	withAutoreleasePool(func() {
		gen, err = s.generateFromHiddenInPool(hidden, maxNew, eosID, nil, nil, nil, transform, yield)
	})
	if err != nil {
		s.cachedIDs = nil
		s.resetRetainedHidden()
		return nil, err
	}
	s.cachedIDs = append(s.cachedIDs, gen...)
	return gen, nil
}

func (s *ArchSession) prefillRetainedTokens(ids []int32, scope string) ([]byte, error) {
	if len(ids) == 0 {
		return nil, nil
	}
	if s.pos+len(ids) > s.maxLen {
		return nil, core.NewError(scope + ": sequence would exceed maxLen cache rows")
	}
	if len(ids) > 1 {
		if err := s.prefillCachedIDs(ids[:len(ids)-1]); err != nil {
			return nil, err
		}
	}
	var hidden []byte
	var err error
	withAutoreleasePool(func() {
		hidden, err = s.stepIDInPool(ids[len(ids)-1])
	})
	return hidden, err
}

func (s *ArchSession) rememberRetainedHidden(hidden []byte) {
	if s == nil || len(hidden) != s.arch.Hidden*bf16Size {
		s.resetRetainedHidden()
		return
	}
	retained := s.retainedHidden[:0]
	s.retainedHidden = append(retained, hidden...)
}

func (s *ArchSession) resetRetainedHidden() {
	if s == nil {
		return
	}
	s.retainedHidden = s.retainedHidden[:0]
}

// Step decodes one token's embedding at the current cache position over the
// persistent KV cache, returning its output hidden state (dModel bf16 bytes) and
// advancing the position — the contract-native incremental decode
// (model.DecodeStepper), so model.Generate drives this session O(1)/token. The
// returned hidden is a fresh Go copy (stepToken copies out of the device
// buffer), so it survives the per-step autorelease pool. PLE models (E2B/E4B)
// derive a per-layer-input tensor from each token id, which Step (embedding
// only) can't supply — they must generate via Generate, so Step rejects a PLE
// session.
func (s *ArchSession) Step(emb []byte) ([]byte, error) {
	if s.perLayerInput != nil {
		return nil, core.NewError("native.ArchSession.Step: per-layer-input models must use Generate, not Step")
	}
	if len(emb) != s.arch.Hidden*bf16Size {
		return nil, core.NewError("native.ArchSession.Step: emb must be hidden bf16 bytes")
	}
	if s.pos >= s.maxLen {
		return nil, core.NewError("native.ArchSession.Step: sequence would exceed maxLen cache rows")
	}
	var res []byte
	var err error
	withAutoreleasePool(func() {
		if s.state.icb != nil { // recorded encode-bypass: replay one token over the ICB's caches
			res = s.state.icb.stepBody(emb, s.pos, nil)
		} else {
			res, err = s.state.stepToken(emb, s.pos)
		}
	})
	if err != nil {
		return nil, err
	}
	s.pos++
	return res, nil
}

// StepWithID is Step with the token id available — the contract's id-aware
// incremental step (model.Generate calls it in preference to Step when present).
// gemma4 E2B/E4B per-layer-input models need the id: the per-layer input is gathered
// from embed_tokens_per_layer[id] (not derivable from the token embedding), so
// StepWithID computes the per-layer-input tensor from (id, emb) and threads it into
// the step, exactly as Generate does. For a model without the PLE tower it is just
// Step (perLayerInput is nil), so it carries no PLE guard.
func (s *ArchSession) StepWithID(id int32, emb []byte) ([]byte, error) {
	if len(emb) != s.arch.Hidden*bf16Size {
		return nil, core.NewError("native.ArchSession.StepWithID: emb must be hidden bf16 bytes")
	}
	if s.pos >= s.maxLen {
		return nil, core.NewError("native.ArchSession.StepWithID: sequence would exceed maxLen cache rows")
	}
	var res []byte
	var err error
	withAutoreleasePool(func() {
		var pli []byte
		if s.perLayerInput != nil { // PLE: per-layer inputs from this token's id + embedding
			if pli, err = s.perLayerInput(id, emb); err != nil {
				return
			}
			s.state.perLayerInput = pli
		}
		if s.state.icb != nil { // recorded encode-bypass: replay one token over the ICB's caches
			res = s.state.icb.stepBody(emb, s.pos, pli)
		} else {
			res, err = s.state.stepToken(emb, s.pos)
		}
	})
	if err != nil {
		return nil, err
	}
	s.pos++
	return res, nil
}

func (s *ArchSession) stepIDInPool(id int32) ([]byte, error) {
	emb, err := s.embed(id)
	if err != nil {
		return nil, err
	}
	var pli []byte
	if s.perLayerInput != nil { // gemma4 PLE: per-token per-layer-input tensor, from this token's embedding
		_ptPLE := ptStart()
		pli, err = s.perLayerInput(id, emb)
		ptEnd(0, _ptPLE)
		if err != nil {
			return nil, err
		}
		s.state.perLayerInput = pli
	}
	var h []byte
	_ptICB := ptStart()
	if s.state.icb != nil && !icbDisabledForTest { // recorded encode-bypass: replay one token over the ICB (as Step/StepWithID do)
		h = s.state.icb.stepBody(emb, s.pos, pli)
	} else if h, err = s.state.stepToken(emb, s.pos); err != nil {
		return nil, err
	}
	ptEnd(1, _ptICB)
	s.pos++
	return h, nil
}

func (s *ArchSession) generateFromHidden(hidden []byte, maxNew, eosID int, firstLogits []byte) ([]int32, error) {
	return s.generateFromHiddenSuppressed(hidden, maxNew, eosID, firstLogits, nil)
}

func (s *ArchSession) generateFromHiddenSuppressed(hidden []byte, maxNew, eosID int, firstLogits []byte, suppress []int32) ([]int32, error) {
	return s.generateFromHiddenSuppressedEach(hidden, maxNew, eosID, firstLogits, suppress, nil, nil)
}

func (s *ArchSession) generateFromHiddenSuppressedEach(hidden []byte, maxNew, eosID int, firstLogits []byte, suppress []int32, transform TokenTransform, yield func(int32) bool) ([]int32, error) {
	if maxNew <= 0 {
		return nil, core.NewError("native.ArchSession.generateFromHidden: maxNew must be > 0")
	}
	if len(hidden) != s.arch.Hidden*bf16Size {
		return nil, core.NewError("native.ArchSession.generateFromHidden: hidden must be hidden bf16 bytes")
	}
	if firstLogits != nil && len(firstLogits) != s.arch.Vocab*bf16Size {
		return nil, core.NewError("native.ArchSession.generateFromHidden: logits must be vocab bf16 bytes")
	}
	if s.pos+maxNew > s.maxLen {
		return nil, core.NewError("native.ArchSession.generateFromHidden: sequence would exceed maxLen cache rows")
	}
	var gen []int32
	var err error
	withAutoreleasePool(func() {
		gen, err = s.generateFromHiddenInPool(hidden, maxNew, eosID, firstLogits, nil, suppress, transform, yield)
	})
	return gen, err
}

// stepGreedyInPool decodes one token (the prior token `id` whose embedding is `emb`) at the current cache
// position AND argmaxes the next token in ONE command buffer: the ICB replay's final hidden flows straight
// into the LM head + argmax on the same buffer, so the host syncs once per token instead of twice (replay
// then head). Returns the next token + this step's hidden (for retained-state caching). ok=false ⇒ no ICB
// or no GPU-argmax head ⇒ the caller uses the two-buffer greedy+stepID path. Must run inside a pool.
func (s *ArchSession) stepGreedyInPool(id int32, emb []byte, suppress []int32) (token int32, hidden []byte, ok bool, err error) {
	if s.state.icb == nil || icbDisabledForTest || s.headEnc == nil {
		return 0, nil, false, nil
	}
	icb := s.state.icb
	var pli []byte
	if s.perLayerInput != nil { // gemma4 PLE: per-token per-layer-input from this token's id+embedding
		pli, err = s.perLayerInput(id, emb)
		if err != nil {
			return 0, nil, false, err
		}
		s.state.perLayerInput = pli
	}
	token = -1
	withAutoreleasePool(func() {
		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		lastOut := icb.encodeStepBody(enc, emb, s.pos, pli)
		outToken, scratch, gok, gerr := s.headEnc.encodeGreedy(enc, lastOut, suppress)
		if !gok || gerr != nil {
			enc.EndEncoding()
			if scratch != nil {
				s.headEnc.putGreedyScratch(scratch)
			}
			ok, err = gok, gerr
			return
		}
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()
		token = *(*int32)(outToken.Contents())
		hidden = make([]byte, s.arch.Hidden*bf16Size)
		copy(hidden, unsafe.Slice((*byte)(lastOut.Contents()), s.arch.Hidden*bf16Size))
		s.headEnc.putGreedyScratch(scratch)
		ok = true
	})
	if err != nil || !ok {
		return 0, nil, ok, err
	}
	s.pos++
	if token < 0 || int(token) >= s.arch.Vocab {
		return 0, nil, true, core.NewError("native.ArchSession.stepGreedyInPool: invalid token")
	}
	return token, hidden, true, nil
}

// headGreedyOrLogits argmaxes the next token from `hidden`: the GPU direct-argmax head when available,
// else the logits path (with the first-token firstLogits/cacheFirstLogits boundary honoured when isFirst).
func (s *ArchSession) headGreedyOrLogits(hidden []byte, suppress []int32, firstLogits []byte, cacheFirstLogits func([]byte), isFirst bool) (int32, error) {
	if !(isFirst && (firstLogits != nil || cacheFirstLogits != nil)) && s.greedy != nil {
		_ptHead := ptStart()
		next, ok, err := s.greedy(hidden, suppress)
		ptEnd(2, _ptHead)
		if err != nil {
			return 0, err
		}
		if ok {
			return next, nil
		}
	}
	var logits []byte
	var err error
	if isFirst && firstLogits != nil {
		logits = firstLogits
	} else {
		_ptHead := ptStart()
		logits, err = s.head(hidden, true) // greedy: argmax — skip the monotonic softcap (token-identical)
		ptEnd(2, _ptHead)
		if err != nil {
			return 0, err
		}
	}
	if isFirst && cacheFirstLogits != nil {
		cacheFirstLogits(logits)
	}
	return greedyBF16Suppressed(logits, s.arch.Vocab, suppress)
}

func (s *ArchSession) generateFromHiddenInPool(hidden []byte, maxNew, eosID int, firstLogits []byte, cacheFirstLogits func([]byte), suppress []int32, transform TokenTransform, yield func(int32) bool) ([]int32, error) {
	gen := make([]int32, 0, maxNew)
	// First token: head+argmax on the prefill/retained hidden (no step yet — the chain caches each token
	// via the NEXT step, and a final step caches the last one).
	next, err := s.headGreedyOrLogits(hidden, suppress, firstLogits, cacheFirstLogits, true)
	if err != nil {
		return nil, err
	}
	if transform != nil {
		next = transform(next)
	}
	gen = append(gen, next)
	stop := (yield != nil && !yield(next)) || (eosID >= 0 && int(next) == eosID)

	// Chained-GPU decode (e2b): the prior step produces the next step's emb+pli on-GPU (encNextInputsGPU
	// appended to the step's command buffer), so each token is ONE command buffer with no host embed/PLE.
	// transform would change the token after the GPU already embedded it, so only when transform == nil.
	if s.encNextInputsGPU != nil && s.plScratchNew != nil && s.state.icb != nil && s.headEnc != nil && s.greedy != nil &&
		!stepGreedyChainDisabled && !chainedGPUInputsDisabled && !icbDisabledForTest && transform == nil {
		return s.generateChainedGPUTail(gen, maxNew, eosID, suppress, yield, stop)
	}

	for !stop && len(gen) < maxNew {
		prev := gen[len(gen)-1]
		emb, eerr := s.embed(prev)
		if eerr != nil {
			return nil, eerr
		}
		var n2 int32
		// Chain prev's stepBody with this token's head+argmax in ONE command buffer (one sync/token).
		if !stepGreedyChainDisabled {
			_ptH := ptStart()
			tok, h, ok, serr := s.stepGreedyInPool(prev, emb, suppress)
			ptEnd(2, _ptH)
			if serr != nil {
				return nil, serr
			}
			if ok {
				n2, hidden = tok, h
				goto produced
			}
		}
		// Serial fallback: step prev (cache it), then head on the new hidden.
		if hidden, err = s.stepIDInPool(prev); err != nil {
			return nil, err
		}
		if n2, err = s.headGreedyOrLogits(hidden, suppress, nil, nil, false); err != nil {
			return nil, err
		}
	produced:
		if transform != nil {
			n2 = transform(n2)
		}
		gen = append(gen, n2)
		s.rememberRetainedHidden(hidden)
		stop = (yield != nil && !yield(n2)) || (eosID >= 0 && int(n2) == eosID)
	}
	// Cache the last produced token (the chain steps prev, not the freshly produced token), so the session
	// state matches the serial loop (every generated token cached) for reuse / a second turn.
	if hidden, err = s.stepIDInPool(gen[len(gen)-1]); err != nil {
		return nil, err
	}
	s.rememberRetainedHidden(hidden)
	return gen, nil
}

// generateChainedGPUTail decodes from the first token `gen[0]` with the GPU next-inputs seam: each token's
// command buffer replays the layer stack (reading the prior step's GPU-produced emb+pli from the ICB's
// ping0/pleInput), argmaxes the head, then runs encNextInputsGPU on the GPU head output to seed THIS step's
// emb+pli for the next — no host embed/PLE round-trip. Cache/pos bookkeeping matches the serial loop: each
// step caches the token whose emb is in ping0; a final no-input step caches the last produced token (so
// session reuse / second turn is byte-identical). `stop` is the first token's stop verdict from the caller.
func (s *ArchSession) generateChainedGPUTail(gen []int32, maxNew, eosID int, suppress []int32, yield func(int32) bool, stop bool) ([]int32, error) {
	icb := s.state.icb
	sc := s.plScratchNew()
	sc.out = icb.pleInput // the PLE result lands directly in the ICB's pli input for the next step
	dModel := s.arch.Hidden
	var rerr error
	withAutoreleasePool(func() {
		tokBuf := device.NewBufferWithLengthOptions(4, metal.MTLResourceStorageModeShared)
		// Seed: produce emb(gen[last])/pli(gen[last]) into ping0/pleInput from the first token.
		*(*int32)(tokBuf.Contents()) = gen[len(gen)-1]
		seedCB := queue.CommandBuffer()
		seedEnc := seedCB.ComputeCommandEncoder()
		if e := s.encNextInputsGPU(seedEnc, tokBuf, icb.ping0, sc); e != nil {
			seedEnc.EndEncoding()
			rerr = e
			return
		}
		seedEnc.EndEncoding()
		seedCB.Commit()
		seedCB.WaitUntilCompleted()

		for !stop && len(gen) < maxNew {
			cb := queue.CommandBuffer()
			enc := cb.ComputeCommandEncoder()
			lastOut := icb.encodeStepBodyNoInput(enc, s.pos) // caches the token in ping0 (gen[last]) at s.pos
			outToken, scratch, gok, gerr := s.headEnc.encodeGreedy(enc, lastOut, suppress)
			if !gok || gerr != nil {
				enc.EndEncoding()
				if scratch != nil {
					s.headEnc.putGreedyScratch(scratch)
				}
				if rerr = gerr; rerr == nil {
					rerr = core.NewError("native.ArchSession.generateChainedGPUTail: GPU head argmax unavailable mid-chain")
				}
				return
			}
			// Produce THIS token's emb+pli on-GPU (into ping0/pleInput) for the NEXT step. Within the
			// encoder the stepBody read of ping0/pleInput is ordered before this write (serial dispatch).
			s.encNextInputsGPU(enc, outToken, icb.ping0, sc)
			enc.EndEncoding()
			cb.Commit()
			cb.WaitUntilCompleted()
			tk := *(*int32)(outToken.Contents())
			s.headEnc.putGreedyScratch(scratch)
			s.pos++
			if tk < 0 || int(tk) >= s.arch.Vocab {
				rerr = core.NewError("native.ArchSession.generateChainedGPUTail: invalid token")
				return
			}
			gen = append(gen, tk)
			stop = (yield != nil && !yield(tk)) || (eosID >= 0 && int(tk) == eosID)
		}

		// Cache the last produced token (its emb is in ping0 but stepBody hasn't run), matching the serial
		// loop's final stepID, and retain that hidden as the session boundary.
		cb := queue.CommandBuffer()
		enc := cb.ComputeCommandEncoder()
		lastOut := icb.encodeStepBodyNoInput(enc, s.pos)
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()
		s.pos++
		h := make([]byte, dModel*bf16Size)
		copy(h, unsafe.Slice((*byte)(lastOut.Contents()), dModel*bf16Size))
		s.rememberRetainedHidden(h)
	})
	if rerr != nil {
		return nil, rerr
	}
	return gen, nil
}

func (s *ArchSession) greedyFromHiddenInPool(hidden []byte, suppress []int32) (int32, error) {
	if s.greedy != nil {
		_ptHead := ptStart()
		next, ok, err := s.greedy(hidden, suppress)
		ptEnd(2, _ptHead)
		if err != nil {
			return 0, err
		}
		if ok {
			return next, nil
		}
	}
	_ptHead := ptStart()
	logits, err := s.head(hidden, true)
	ptEnd(2, _ptHead)
	if err != nil {
		return 0, err
	}
	return greedyBF16Suppressed(logits, s.arch.Vocab, suppress)
}

// GenerateText is the text-in/text-out wrapper over Generate, now that the tokenizer is a
// shared no-cgo package: it encodes prompt with tok, generates up to maxNew tokens (stopping
// at the tokenizer's EOS when it has one), and decodes the result back to a string. The
// session's cache carries over across calls, so successive GenerateText turns continue the
// conversation. The whole text → tokens → decode → text path runs with no cgo and no Python.
func (s *ArchSession) GenerateText(tok *tokenizer.Tokenizer, prompt string, maxNew int) (string, error) {
	if tok == nil {
		return "", core.NewError("native.ArchSession.GenerateText: nil tokenizer")
	}
	ids := tok.Encode(prompt)
	if len(ids) == 0 {
		return "", core.NewError("native.ArchSession.GenerateText: prompt encoded to no tokens")
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
func (s *ArchSession) Generate(promptIDs []int32, maxNew, eosID int) ([]int32, error) {
	return s.generate(promptIDs, maxNew, eosID, nil, nil)
}

// GenerateEach is Generate with per-token streaming: each token is yielded after it is
// selected and written into the session cache. If yield returns false, decoding stops
// without treating consumer stop as an error; the returned slice contains the tokens
// emitted before the stop.
func (s *ArchSession) GenerateEach(promptIDs []int32, maxNew, eosID int, yield func(int32) bool) ([]int32, error) {
	return s.GenerateEachWithSuppressionAndTransform(promptIDs, maxNew, eosID, nil, nil, yield)
}

// GenerateEachTransformed is GenerateEach with a committed-token transform
// applied before each generated token is written to the session cache.
func (s *ArchSession) GenerateEachTransformed(promptIDs []int32, maxNew, eosID int, transform TokenTransform, yield func(int32) bool) ([]int32, error) {
	return s.GenerateEachWithSuppressionAndTransform(promptIDs, maxNew, eosID, nil, transform, yield)
}

// GenerateEachWithSuppression is GenerateEach with suppressed token ids masked
// before greedy argmax.
func (s *ArchSession) GenerateEachWithSuppression(promptIDs []int32, maxNew, eosID int, suppress []int32, yield func(int32) bool) ([]int32, error) {
	return s.GenerateEachWithSuppressionAndTransform(promptIDs, maxNew, eosID, suppress, nil, yield)
}

// GenerateEachWithSuppressionAndTransform combines greedy token suppression
// with a committed-token transform.
func (s *ArchSession) GenerateEachWithSuppressionAndTransform(promptIDs []int32, maxNew, eosID int, suppress []int32, transform TokenTransform, yield func(int32) bool) ([]int32, error) {
	return s.generateWithYield(promptIDs, maxNew, eosID, nil, suppress, transform, yield)
}

// GenerateWithSuppression is the native sibling of pkg/metal's suppressed
// direct-greedy path: suppressed token ids are masked before argmax, including
// when the resident head can return the token directly without materialising
// full vocab logits.
func (s *ArchSession) GenerateWithSuppression(promptIDs []int32, maxNew, eosID int, suppress []int32) ([]int32, error) {
	return s.generate(promptIDs, maxNew, eosID, nil, suppress)
}

// GenerateOneShot is the contract-level greedy path used by model.Generate
// when it opens and closes a fresh session for one request. It uses the same
// direct greedy engine as retained Generate, but does not step the final
// generated token because no caller can reuse that closed session's final cache
// row. Retained callers should use Generate / GenerateEach instead.
func (s *ArchSession) GenerateOneShot(promptIDs []int32, maxNew, eosID int) ([]int32, error) {
	if len(promptIDs) == 0 {
		return nil, core.NewError("native.ArchSession.GenerateOneShot: empty prompt")
	}
	if maxNew <= 0 {
		return nil, core.NewError("native.ArchSession.GenerateOneShot: maxNew must be > 0")
	}
	if s.pos+len(promptIDs)+maxNew > s.maxLen {
		return nil, core.NewError("native.ArchSession.GenerateOneShot: sequence would exceed maxLen cache rows")
	}
	var gen []int32
	var genErr error
	withAutoreleasePool(func() {
		var hidden []byte
		for _, id := range promptIDs {
			if hidden, genErr = s.stepIDInPool(id); genErr != nil {
				return
			}
		}
		gen, genErr = s.generateOneShotFromHiddenInPool(hidden, maxNew, eosID)
	})
	return gen, genErr
}

func (s *ArchSession) generateOneShotFromHiddenInPool(hidden []byte, maxNew, eosID int) ([]int32, error) {
	gen := make([]int32, 0, maxNew)
	for len(gen) < maxNew {
		next, err := s.greedyFromHiddenInPool(hidden, nil)
		if err != nil {
			return nil, err
		}
		gen = append(gen, next)
		if eosID >= 0 && int(next) == eosID {
			break
		}
		if len(gen) >= maxNew {
			break
		}
		if hidden, err = s.stepIDInPool(next); err != nil {
			return nil, err
		}
	}
	return gen, nil
}

func (s *ArchSession) generate(promptIDs []int32, maxNew, eosID int, rememberPromptIDs []int32, suppress []int32) ([]int32, error) {
	return s.generateWithYield(promptIDs, maxNew, eosID, rememberPromptIDs, suppress, nil, nil)
}

func (s *ArchSession) generateWithYield(promptIDs []int32, maxNew, eosID int, rememberPromptIDs []int32, suppress []int32, transform TokenTransform, yield func(int32) bool) ([]int32, error) {
	if len(promptIDs) == 0 {
		return nil, core.NewError("native.ArchSession.Generate: empty prompt")
	}
	if maxNew <= 0 {
		return nil, core.NewError("native.ArchSession.Generate: maxNew must be > 0")
	}
	if s.pos+len(promptIDs)+maxNew > s.maxLen {
		return nil, core.NewError("native.ArchSession.Generate: sequence would exceed maxLen cache rows")
	}
	var gen []int32
	var genErr error
	withAutoreleasePool(func() {
		// prefill the new prompt over the carried-over cache; keep the last hidden state.
		var hidden []byte
		for _, id := range promptIDs {
			if hidden, genErr = s.stepIDInPool(id); genErr != nil {
				return
			}
		}
		if len(rememberPromptIDs) > 0 {
			cacheFirstLogits := func(logits []byte) {
				s.rememberCachedPromptEntry(rememberPromptIDs, hidden, logits)
			}
			gen, genErr = s.generateFromHiddenInPool(hidden, maxNew, eosID, nil, cacheFirstLogits, suppress, transform, yield)
			return
		}
		// decode: head → greedy → append → step the new token (caching it for the next turn).
		gen, genErr = s.generateFromHiddenInPool(hidden, maxNew, eosID, nil, nil, suppress, transform, yield)
	})
	return gen, genErr
}
