// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"

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
	arch  model.Arch
	embed func(id int32) ([]byte, error)      // token id → its embedded bf16 vector (dModel bytes)
	head  func(hidden []byte) ([]byte, error) // hidden bf16 → vocab bf16 logits
	// perLayerInput, when set (gemma4 E2B/E4B), computes the per-token PerLayerInputs tensor
	// from the token id + its embedding; Generate sets it on the state before stepToken. nil
	// for models without the PLE tower.
	perLayerInput func(id int32, emb []byte) ([]byte, error)
	state         archDecodeState
	pos           int // tokens already in the cache (the next token decodes at this position)
	maxLen        int
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
		head, herr := newHeadEncoder(sb, g.FinalNorm, g.LMHead, nil, nil, arch.Hidden, arch.Vocab, 0, 0, arch.Eps, arch.SoftCap, false)
		if herr != nil {
			buildErr = herr
			return
		}
		sess = &ArchSession{
			arch: arch, state: state, maxLen: maxLen,
			embed: func(id int32) ([]byte, error) {
				embs, err := EmbedTokensBF16(g.Embed, []int32{id}, arch.Vocab, arch.Hidden, embedScale)
				if err != nil {
					return nil, err
				}
				return embs[0], nil
			},
			head: func(hidden []byte) ([]byte, error) {
				if head != nil {
					return head.encode(hidden)
				}
				return LMHeadBF16(hidden, g.FinalNorm, g.LMHead, arch.Hidden, arch.Vocab, arch.Eps, arch.SoftCap)
			},
		}
		if g.HasPLE() {
			sess.perLayerInput = func(id int32, emb []byte) ([]byte, error) {
				return PerLayerInputs(g.EmbedPerLayer, nil, nil, g.PerLayerModelProjW, nil, nil, g.PerLayerProjNormW, id, emb, arch.PerLayerInputVocab, len(arch.Layer), arch.PerLayerInputHidden, arch.Hidden, 0, 0, 0, 0, arch.Eps)
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
		head, herr := newHeadEncoder(sb, g.FinalNorm, g.LMHead, g.LMHeadScales, g.LMHeadBiases, arch.Hidden, arch.Vocab, gs, bits, arch.Eps, arch.SoftCap, true)
		if herr != nil {
			buildErr = herr
			return
		}
		sess = &ArchSession{
			arch: arch, state: state, maxLen: maxLen,
			embed: func(id int32) ([]byte, error) {
				embs, err := EmbedTokensQuant(g.Embed, g.EmbedScales, g.EmbedBiases, []int32{id}, arch.Vocab, arch.Hidden, gs, bits, embedScale)
				if err != nil {
					return nil, err
				}
				return embs[0], nil
			},
			head: func(hidden []byte) ([]byte, error) {
				if head != nil {
					return head.encode(hidden)
				}
				return LMHeadQuant(hidden, g.FinalNorm, g.LMHead, g.LMHeadScales, g.LMHeadBiases, arch.Hidden, arch.Vocab, gs, bits, arch.Eps, arch.SoftCap)
			},
		}
		if g.HasPLE() {
			sess.perLayerInput = func(id int32, emb []byte) ([]byte, error) {
				return PerLayerInputs(g.EmbedPerLayer, g.EmbedPerLayerScales, g.EmbedPerLayerBiases, g.PerLayerModelProjW, g.PerLayerModelProjScales, g.PerLayerModelProjBiases, g.PerLayerProjNormW, id, emb, arch.PerLayerInputVocab, len(arch.Layer), arch.PerLayerInputHidden, arch.Hidden, gs, bits, g.PerLayerModelProjGS, g.PerLayerModelProjBits, arch.Eps)
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
	if len(promptIDs) == 0 {
		return nil, core.NewError("native.ArchSession.Generate: empty prompt")
	}
	if maxNew <= 0 {
		return nil, core.NewError("native.ArchSession.Generate: maxNew must be > 0")
	}
	if s.pos+len(promptIDs)+maxNew > s.maxLen {
		return nil, core.NewError("native.ArchSession.Generate: sequence would exceed maxLen cache rows")
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
			if s.perLayerInput != nil { // gemma4 PLE: per-token per-layer-input tensor, from this token's embedding
				pli, err := s.perLayerInput(id, emb)
				if err != nil {
					return nil, err
				}
				s.state.perLayerInput = pli
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
