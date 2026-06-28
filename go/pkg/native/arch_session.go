// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"reflect"
	"slices"
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
	arch          model.Arch
	embed         func(id int32) ([]byte, error)             // token id → its embedded bf16 vector (dModel bytes)
	embedInto     func(dst []byte, id int32) ([]byte, error) // token id → caller-owned embedded bf16 vector
	embedFuncPtr  uintptr
	head          func(hidden []byte, skipSoftcap bool) ([]byte, error)      // hidden bf16 → vocab bf16 logits; skipSoftcap for argmax callers
	greedy        func(hidden []byte, suppress []int32) (int32, bool, error) // optional direct greedy token path; ok=false falls back to head+Greedy
	headEnc       *headEncoder
	headFuncPtr   uintptr
	greedyFuncPtr uintptr
	// perLayerInput, when set (gemma4 E2B/E4B), computes the per-token PerLayerInputs tensor
	// from the token id + its embedding; Generate sets it on the state before stepToken. nil
	// for models without the PLE tower.
	perLayerInput func(id int32, emb []byte) ([]byte, error)
	// pleHostScratch reuses pinned host staging and intermediate Metal buffers for the host-side
	// resident BF16 PLE projection path. nil when the model has no PLE tower or uses quant PLE projection.
	pleHostScratch *plHostScratch
	// encNextInputsGPU, when set (e2b: 4-bit main+PLE embedding, bf16 PLE projection), encodes the GPU
	// embed-gather (token → embOut, dModel) + the GPU PLE (token, embOut → sc.out, numLayers·pliDim) for
	// one token read from tokenBuf into a shared encoder — the NEXT decode step's emb+pli produced on-GPU
	// with no host round-trip (the submit-ahead pipeline seam). nil → the host embed/PLE path stays.
	encNextInputsGPU func(enc metal.MTLComputeCommandEncoder, tokenBuf, embOut metal.MTLBuffer, sc *plGPUScratch) error
	plScratchNew     func() *plGPUScratch
	// recordPeerICB records a SECOND ICB sharing this session's KV caches (its own ping0/pleInput) — the
	// submit-ahead decode keeps two ICBs in flight over the same KV so the host can submit token t+1
	// before reading t. Recorded lazily via peerICB() (most sessions never pipeline). nil when not ICB.
	recordPeerICB      func() (*archICBReplay, error)
	icbPeer            *archICBReplay
	state              archDecodeState
	stateBlockViews    []sessionStateLayerView
	stateBlockViewsICB bool
	stateBlockLayers   []SessionStateLayerBlock
	stateBlockBounds   []int
	pos                int // tokens already in the cache (the next token decodes at this position)
	maxLen             int
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
	retainedHidden       []byte
	retainedLogits       []byte
	retainedHiddenPinned *pinnedNoCopyBytes
	retainedLogitsPinned *pinnedNoCopyBytes
	// shards holds the memory-mapped checkpoint + its per-shard no-copy Metal buffers when the
	// session was loaded from a directory zero-copy (LoadGemma4*Dir). The weight []byte fields the
	// embed/head closures and the decode buffers reference are VIEWS into these mmaps, so shards
	// MUST stay alive for the session's life; Close unmaps them. nil for a session built from
	// in-memory weight bytes (NewArchSession over an already-parsed BF16Model) — those weights
	// are heap-owned, nothing to unmap.
	shards *shardBuffers
	// sampled candidate readback scratch. Generation is single-goroutine per
	// session, so the TopK path can reuse these K-sized host buffers instead of
	// allocating logits/ids every sampled token.
	sampleCandidateLogits []byte
	sampleCandidateIDs    []int32
	sampleHeadLogits      []byte
	sampleHidden          []byte
	sampleHistory         []int32
	samplePenaltyIDs      []int32
	samplePenaltyLogits   []byte
	sampleScaled          []float32
	sampleProbs           []float32
	sampleOrder           []int32
	sampleSuppressTokens  []int32
	embedScratch          []byte
	nextInputToken        metal.MTLBuffer
	nextInputTokenPtr     *int32
	nextInputTokenPinned  *pinnedNoCopyBytes
	nextInputEmb          metal.MTLBuffer
	nextInputEmbPtr       *byte
	nextInputEmbPinned    *pinnedNoCopyBytes
	nextInputEmbHost      []byte
	nextInputPLEHost      []byte
	nextInputPLScratch    *plGPUScratch
	gpuTailPLScratch      [2]*plGPUScratch
}

// Close releases a directory-loaded session's memory-mapped checkpoint. It is safe on a session
// built from in-memory bytes (shards nil ⇒ no-op) and idempotent. Call it once decoding is done;
// the no-copy weight buffers reference the mmap, so do not Close while a Generate/Step is in
// flight (single-goroutine sessions make that the caller's natural discipline).
func (s *ArchSession) Close() error {
	if s == nil {
		return nil
	}
	if s.pleHostScratch != nil {
		s.pleHostScratch.Close()
		s.pleHostScratch = nil
	}
	s.closeSessionOwnedScratch()
	s.closeModelAndDecodeStateReferences()
	if s.shards == nil {
		return nil
	}
	err := s.shards.Close()
	s.shards = nil
	return err
}

func (s *ArchSession) closeSessionOwnedScratch() {
	s.sampleCandidateLogits = nil
	s.sampleCandidateIDs = nil
	s.sampleHeadLogits = nil
	s.sampleHidden = nil
	s.sampleHistory = nil
	s.samplePenaltyIDs = nil
	s.samplePenaltyLogits = nil
	s.sampleScaled = nil
	s.sampleProbs = nil
	s.sampleOrder = nil
	s.sampleSuppressTokens = nil
	s.embedScratch = nil

	s.nextInputToken = nil
	s.nextInputTokenPtr = nil
	if s.nextInputTokenPinned != nil {
		s.nextInputTokenPinned.Close()
		s.nextInputTokenPinned = nil
	}
	s.nextInputEmb = nil
	s.nextInputEmbPtr = nil
	if s.nextInputEmbPinned != nil {
		s.nextInputEmbPinned.Close()
		s.nextInputEmbPinned = nil
	}
	s.nextInputEmbHost = nil
	s.nextInputPLEHost = nil

	if s.nextInputPLScratch != nil {
		s.nextInputPLScratch.Close()
		s.nextInputPLScratch = nil
	}
	for i := range s.gpuTailPLScratch {
		if s.gpuTailPLScratch[i] != nil {
			s.gpuTailPLScratch[i].Close()
			s.gpuTailPLScratch[i] = nil
		}
	}
}

func (s *ArchSession) closeModelAndDecodeStateReferences() {
	s.embed = nil
	s.embedInto = nil
	s.embedFuncPtr = 0
	s.head = nil
	s.greedy = nil
	s.headEnc = nil
	s.headFuncPtr = 0
	s.greedyFuncPtr = 0
	s.perLayerInput = nil
	s.encNextInputsGPU = nil
	s.plScratchNew = nil
	s.recordPeerICB = nil
	s.icbPeer = nil

	s.state.Close()
	s.state = archDecodeState{}
	s.stateBlockViews = nil
	s.stateBlockViewsICB = false
	s.stateBlockLayers = nil
	s.stateBlockBounds = nil
	s.cachedIDs = nil
	s.cachedPromptIDs = nil
	s.cachedPromptHidden = nil
	s.cachedPromptLogits = nil
	if s.retainedHiddenPinned != nil {
		s.retainedHiddenPinned.Close()
		s.retainedHiddenPinned = nil
	}
	if s.retainedLogitsPinned != nil {
		s.retainedLogitsPinned.Close()
		s.retainedLogitsPinned = nil
	}
	s.retainedHidden = nil
	s.retainedLogits = nil

	s.arch = model.Arch{}
	s.pos = 0
	s.maxLen = 0
}

func (s *ArchSession) embedID(id int32) ([]byte, error) {
	if !s.canUseEmbedScratch() {
		return s.embed(id)
	}
	n := s.arch.Hidden * bf16Size
	if cap(s.embedScratch) < n {
		s.embedScratch = make([]byte, n)
	}
	return s.embedInto(s.embedScratch[:n], id)
}

func (s *ArchSession) markDefaultEmbedFunc() {
	if s == nil || s.embed == nil {
		return
	}
	s.embedFuncPtr = reflect.ValueOf(s.embed).Pointer()
}

func (s *ArchSession) canUseEmbedScratch() bool {
	if s == nil || s.embedInto == nil {
		return false
	}
	if s.embed == nil || s.embedFuncPtr == 0 {
		return true
	}
	return reflect.ValueOf(s.embed).Pointer() == s.embedFuncPtr
}

func (s *ArchSession) copyHiddenReadback(buf metal.MTLBuffer) []byte {
	if buf == nil {
		return nil
	}
	return s.copyHiddenReadbackFrom((*byte)(buf.Contents()))
}

func (s *ArchSession) copyHiddenReadbackFrom(ptr *byte) []byte {
	n := s.arch.Hidden * bf16Size
	if n <= 0 || ptr == nil {
		return nil
	}
	if cap(s.sampleHidden) < n {
		s.sampleHidden = make([]byte, n)
	} else {
		s.sampleHidden = s.sampleHidden[:n]
	}
	copy(s.sampleHidden, unsafe.Slice(ptr, n))
	return s.sampleHidden
}

func (s *ArchSession) retainHiddenReadbackFrom(ptr *byte) []byte {
	s.rememberRetainedHiddenFrom(ptr)
	return s.retainedHidden
}

func (s *ArchSession) headLogitsScratch(hidden []byte, skipSoftcap bool) ([]byte, error) {
	if s.headEnc == nil {
		return s.head(hidden, skipSoftcap)
	}
	var logits []byte
	var err error
	if hiddenBuf := s.retainedHiddenBufferFor(hidden); hiddenBuf != nil {
		if cap(s.sampleHeadLogits) < s.arch.Vocab*bf16Size {
			s.sampleHeadLogits = make([]byte, s.arch.Vocab*bf16Size)
		} else {
			s.sampleHeadLogits = s.sampleHeadLogits[:s.arch.Vocab*bf16Size]
		}
		err = s.headEnc.encodeBufferIntoPool(hiddenBuf, skipSoftcap, s.sampleHeadLogits)
		logits = s.sampleHeadLogits
	} else {
		logits, err = s.headEnc.encodeInto(hidden, skipSoftcap, s.sampleHeadLogits)
	}
	if err != nil {
		return nil, err
	}
	s.sampleHeadLogits = logits
	return logits, nil
}

func (s *ArchSession) markDefaultHeadFunc() {
	if s == nil || s.head == nil {
		return
	}
	s.headFuncPtr = reflect.ValueOf(s.head).Pointer()
}

func (s *ArchSession) markDefaultGreedyFunc() {
	if s == nil || s.greedy == nil {
		return
	}
	s.greedyFuncPtr = reflect.ValueOf(s.greedy).Pointer()
}

func (s *ArchSession) canUseHeadLogitsScratch() bool {
	return s != nil && s.headEnc != nil && s.head != nil && s.headFuncPtr != 0 && reflect.ValueOf(s.head).Pointer() == s.headFuncPtr
}

func (s *ArchSession) canUseDirectHeadGreedy() bool {
	return s != nil && s.canUseHeadLogitsScratch() && s.greedy != nil && s.greedyFuncPtr != 0 &&
		reflect.ValueOf(s.greedy).Pointer() == s.greedyFuncPtr && s.headEnc.directGreedyUsable()
}

func (s *ArchSession) directGreedyFromHiddenInPool(hidden []byte, suppress []int32) (int32, bool, error) {
	if s.canUseDirectHeadGreedy() {
		if hiddenBuf := s.retainedHiddenBufferFor(hidden); hiddenBuf != nil {
			return s.headEnc.greedyBufferInPool(hiddenBuf, suppress)
		}
	}
	return s.greedy(hidden, suppress)
}

func (s *ArchSession) sampleHistoryScratch(maxNew int) []int32 {
	if maxNew <= 0 {
		s.sampleHistory = s.sampleHistory[:0]
		return s.sampleHistory
	}
	if cap(s.sampleHistory) < maxNew {
		s.sampleHistory = make([]int32, 0, maxNew)
	} else {
		s.sampleHistory = s.sampleHistory[:0]
	}
	return s.sampleHistory
}

func (s *ArchSession) sampleHistoryScratchFor(params model.SampleParams, maxNew int) []int32 {
	if params.RepeatPenalty <= 1 {
		return s.sampleHistory[:0]
	}
	return s.sampleHistoryScratch(maxNew)
}

func (s *ArchSession) repeatPenaltyLogitsScratch(logits []byte, vocab int, history []int32, penalty float32) ([]byte, error) {
	if len(logits) != vocab*bf16Size {
		return nil, core.NewError("native.applyRepeatPenalty: logits must be vocab bf16 bytes")
	}
	if penalty <= 1 || len(history) == 0 {
		return logits, nil
	}
	if cap(s.samplePenaltyIDs) < len(history) {
		s.samplePenaltyIDs = make([]int32, 0, len(history))
	} else {
		s.samplePenaltyIDs = s.samplePenaltyIDs[:0]
	}
	for _, id := range history {
		if id >= 0 && int(id) < vocab {
			s.samplePenaltyIDs = append(s.samplePenaltyIDs, id)
		}
	}
	if len(s.samplePenaltyIDs) == 0 {
		return logits, nil
	}
	slices.Sort(s.samplePenaltyIDs)
	s.samplePenaltyIDs = slices.Compact(s.samplePenaltyIDs)
	if cap(s.samplePenaltyLogits) < len(logits) {
		s.samplePenaltyLogits = make([]byte, len(logits))
	} else {
		s.samplePenaltyLogits = s.samplePenaltyLogits[:len(logits)]
	}
	copy(s.samplePenaltyLogits, logits)
	applyRepeatPenaltySortedIDsBF16(s.samplePenaltyLogits, s.samplePenaltyIDs, penalty)
	return s.samplePenaltyLogits, nil
}

func (s *ArchSession) suppressionTokensScratch(base, extra []int32) []int32 {
	if len(extra) == 0 {
		return base
	}
	if len(base) == 0 {
		return extra
	}
	allExtraSuppressed := true
	for _, token := range extra {
		if !nativeTokenInSet(token, base) {
			allExtraSuppressed = false
			break
		}
	}
	if allExtraSuppressed {
		return base
	}
	wantCap := len(base) + len(extra)
	if cap(s.sampleSuppressTokens) < wantCap {
		s.sampleSuppressTokens = make([]int32, 0, wantCap)
	} else {
		s.sampleSuppressTokens = s.sampleSuppressTokens[:0]
	}
	s.sampleSuppressTokens = append(s.sampleSuppressTokens, base...)
	for _, token := range extra {
		if nativeTokenInSet(token, s.sampleSuppressTokens) {
			continue
		}
		s.sampleSuppressTokens = append(s.sampleSuppressTokens, token)
	}
	return s.sampleSuppressTokens
}

func (s *ArchSession) nextInputTokenBuffer(id int32) metal.MTLBuffer {
	if s.nextInputToken == nil {
		if pinned, err := newPinnedNoCopyBytes(4); err == nil {
			s.nextInputTokenPinned = pinned
			s.nextInputToken = pinned.buf
			s.nextInputTokenPtr = (*int32)(unsafe.Pointer(&pinned.bytes[0]))
		} else {
			s.nextInputToken = device.NewBufferWithLengthOptions(4, metal.MTLResourceStorageModeShared)
			s.nextInputTokenPtr = (*int32)(s.nextInputToken.Contents())
		}
	}
	*s.nextInputTokenPtr = id
	return s.nextInputToken
}

func (s *ArchSession) nextInputEmbBuffer(dModel int) metal.MTLBuffer {
	n := dModel * bf16Size
	if n <= 0 {
		return nil
	}
	if s.nextInputEmb == nil || int(s.nextInputEmb.Length()) != n {
		if s.nextInputEmbPinned != nil {
			s.nextInputEmbPinned.Close()
			s.nextInputEmbPinned = nil
		}
		if pinned, err := newPinnedNoCopyBytes(n); err == nil {
			s.nextInputEmbPinned = pinned
			s.nextInputEmb = pinned.buf
			s.nextInputEmbPtr = (*byte)(unsafe.Pointer(&pinned.bytes[0]))
		} else {
			s.nextInputEmb = device.NewBufferWithLengthOptions(uint(n), metal.MTLResourceStorageModeShared)
			s.nextInputEmbPtr = (*byte)(s.nextInputEmb.Contents())
		}
	}
	return s.nextInputEmb
}

func (s *ArchSession) nextInputEmbReadback(dModel int) []byte {
	n := dModel * bf16Size
	if n <= 0 {
		return nil
	}
	if s.nextInputEmbPinned != nil && len(s.nextInputEmbPinned.bytes) == n {
		return s.nextInputEmbPinned.bytes[:n]
	}
	if cap(s.nextInputEmbHost) < n {
		s.nextInputEmbHost = make([]byte, n)
	}
	return s.nextInputEmbHost[:n]
}

func (s *ArchSession) nextInputPLEReadback(plDim int) []byte {
	n := plDim * bf16Size
	if n <= 0 {
		return nil
	}
	if s.nextInputPLScratch != nil && s.nextInputPLScratch.outPinned != nil && len(s.nextInputPLScratch.outPinned.bytes) == n {
		return s.nextInputPLScratch.outPinned.bytes[:n]
	}
	if cap(s.nextInputPLEHost) < n {
		s.nextInputPLEHost = make([]byte, n)
	}
	return s.nextInputPLEHost[:n]
}

func (s *ArchSession) nextInputPLScratchBuffer() *plGPUScratch {
	if s.nextInputPLScratch == nil {
		s.nextInputPLScratch = s.plScratchNew()
	}
	return s.nextInputPLScratch
}

func (s *ArchSession) gpuTailPLScratchBuffer(slot int) *plGPUScratch {
	if s.gpuTailPLScratch[slot] == nil {
		s.gpuTailPLScratch[slot] = s.plScratchNew()
	}
	return s.gpuTailPLScratch[slot]
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
		state := newArchDecodeState(arch.Layer, lb, moeWeights, arch.Hidden, arch.Heads, arch.KVHeads, arch.HeadDim, arch.FF, arch.SlidingWindow, arch.RotaryDim, arch.RotaryDimLocal, arch.RopeBase, arch.RopeLocalBase, attnScale, arch.Eps, arch.ValueNorm, maxLen)
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
			embedInto: func(dst []byte, id int32) ([]byte, error) {
				return embedTokenBF16Into(dst, g.Embed, id, arch.Vocab, arch.Hidden, embedScale)
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
				return head.greedyInPool(hidden, suppress)
			},
		}
		sess.markDefaultEmbedFunc()
		sess.markDefaultHeadFunc()
		sess.markDefaultGreedyFunc()
		if g.HasPLE() {
			var pleProjView bufView // resident no-copy bf16 PLE projection — bound once at its shard offset, not re-uploaded per token
			if sb != nil {
				pleProjView, _ = sb.bufFor(g.PerLayerModelProjW)
			}
			var pleScratch *plHostScratch
			if pleProjView.buf != nil {
				plDim := len(arch.Layer) * arch.PerLayerInputHidden
				projScale := float32(1.0 / math.Sqrt(float64(arch.Hidden)))
				pleScratch, buildErr = newPLHostScratch(plDim, arch.Hidden, projScale)
				if buildErr != nil {
					return
				}
				sess.pleHostScratch = pleScratch
			}
			sess.perLayerInput = func(id int32, emb []byte) ([]byte, error) {
				pv := pleProjView
				scratch := pleScratch
				if pleResidentDisabled { // call-time host-path toggle (byte-identity test hook; always false in production)
					pv = bufView{}
					scratch = nil
				}
				return PerLayerInputs(g.EmbedPerLayer, nil, nil, g.PerLayerModelProjW, nil, nil, g.PerLayerProjNormW, id, emb, arch.PerLayerInputVocab, len(arch.Layer), arch.PerLayerInputHidden, arch.Hidden, 0, 0, 0, 0, arch.Eps, pv, scratch)
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
		state := newArchDecodeState(arch.Layer, lb, moeWeights, arch.Hidden, arch.Heads, arch.KVHeads, arch.HeadDim, arch.FF, arch.SlidingWindow, arch.RotaryDim, arch.RotaryDimLocal, arch.RopeBase, arch.RopeLocalBase, attnScale, arch.Eps, arch.ValueNorm, maxLen)
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
			embedInto: func(dst []byte, id int32) ([]byte, error) {
				return embedTokenQuantInto(dst, g.Embed, g.EmbedScales, g.EmbedBiases, id, arch.Vocab, arch.Hidden, gs, bits, embedScale)
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
				return head.greedyInPool(hidden, suppress)
			},
		}
		sess.markDefaultEmbedFunc()
		sess.markDefaultHeadFunc()
		sess.markDefaultGreedyFunc()
		if g.HasPLE() {
			var pleProjView bufView // resident no-copy PLE projection when it's bf16 (e2b: no proj scales) — bound once, not re-uploaded per token
			if sb != nil && len(g.PerLayerModelProjScales) == 0 {
				pleProjView, _ = sb.bufFor(g.PerLayerModelProjW)
			}
			var pleScratch *plHostScratch
			if pleProjView.buf != nil {
				plDim := len(arch.Layer) * arch.PerLayerInputHidden
				projScale := float32(1.0 / math.Sqrt(float64(arch.Hidden)))
				pleScratch, buildErr = newPLHostScratch(plDim, arch.Hidden, projScale)
				if buildErr != nil {
					return
				}
				sess.pleHostScratch = pleScratch
			}
			sess.perLayerInput = func(id int32, emb []byte) ([]byte, error) {
				pv := pleProjView
				scratch := pleScratch
				if pleResidentDisabled { // call-time host-path toggle (byte-identity test hook; always false in production)
					pv = bufView{}
					scratch = nil
				}
				return PerLayerInputs(g.EmbedPerLayer, g.EmbedPerLayerScales, g.EmbedPerLayerBiases, g.PerLayerModelProjW, g.PerLayerModelProjScales, g.PerLayerModelProjBiases, g.PerLayerProjNormW, id, emb, arch.PerLayerInputVocab, len(arch.Layer), arch.PerLayerInputHidden, arch.Hidden, gs, bits, g.PerLayerModelProjGS, g.PerLayerModelProjBits, arch.Eps, pv, scratch)
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
					cacheBytes := uint(maxLen * kvHeadsOf(arch.Layer[li], arch.KVHeads) * headDimOf(arch.Layer[li], arch.HeadDim) * bf16Size)
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
			// Recorder for a PEER ICB sharing these KV caches (own ping0/pleInput) — the submit-ahead
			// decode keeps two in flight over the same KV. Lazily invoked; most sessions never pipeline.
			sess.recordPeerICB = func() (*archICBReplay, error) {
				return recordArchICBQuant(g.Layers, arch.Layer, kCaches, vCaches, pleRuntime, arch.PerLayerInputHidden, gs, bits, arch.Hidden, arch.Heads, arch.KVHeads, arch.HeadDim, maxLen, arch.FF, arch.SlidingWindow, rope, attnScale, arch.Eps, arch.ValueNorm)
			}
			if pipelinedGPUDecodeEnabled {
				peer, perr := sess.recordPeerICB()
				if perr != nil {
					buildErr = perr
					return
				}
				sess.icbPeer = peer
			}
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
		// Per-layer head dim AND per-layer kvHeads are both recorded byte-identically: the forward-level
		// gate TestDecodeForwardArchICBQuantPerLayerKVHeads (DecodeForwardArchICBQuant ≡ DecodeForwardArchQuant
		// on a sliding-GQA/global-MQA mix) and the session-level TestArchQuantSessionICBParity_PerLayerKVHeads
		// (per-layer hidden cosine ≥ 0.9999) both pass. The old "14/24 divergence" came from a CONFOUNDED
		// session-level real-model test (PLE/head/chained paths differ from host re-encode even when the
		// recorder is byte-identical — it fails on uniform e2b too), not a recorder bug. So the 12B/31B
		// MQA-global mix now takes the fast ICB path. Only MoE (host router) and trace stay re-encode.
		if sp.MoE {
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

// GenerateSampledFromCacheEach samples from the retained session boundary
// without replaying prompt tokens or requiring captured boundary logits.
func (s *ArchSession) GenerateSampledFromCacheEach(maxNew int, stopTokens []int32, sampler *model.Sampler, params model.SampleParams, transform model.TokenTransform, yield func(int32) bool) ([]int32, error) {
	if sampler == nil {
		return nil, core.NewError("native.ArchSession.GenerateSampledFromCache: nil sampler")
	}
	if maxNew <= 0 {
		return nil, core.NewError("native.ArchSession.GenerateSampledFromCache: maxNew must be > 0")
	}
	if len(s.retainedLogits) == s.arch.Vocab*bf16Size {
		return s.GenerateSampledFromCacheLogitsEach(s.retainedLogits, maxNew, stopTokens, sampler, params, transform, yield)
	}
	if len(s.retainedHidden) != s.arch.Hidden*bf16Size {
		return nil, core.NewError("native.ArchSession.GenerateSampledFromCache: no retained prefill state")
	}
	if s.pos+maxNew > s.maxLen {
		return nil, core.NewError("native.ArchSession.GenerateSampledFromCache: sequence would exceed maxLen cache rows")
	}
	hidden := s.retainedHidden
	var gen []int32
	var err error
	withAutoreleasePool(func() {
		gen, err = s.generateSampledFromHiddenInPool(hidden, maxNew, stopTokens, sampler, params, transform, yield, true)
	})
	if err != nil {
		s.cachedIDs = nil
		s.resetRetainedHidden()
		return nil, err
	}
	s.cachedIDs = append(s.cachedIDs, gen...)
	return gen, nil
}

// BoundaryLogits returns the bf16 logits at the retained session boundary.
// Restore paths can use these logits to select the first continuation token
// without recomputing the restored prompt prefix.
func (s *ArchSession) BoundaryLogits() ([]byte, error) {
	if len(s.retainedLogits) == s.arch.Vocab*bf16Size {
		return s.retainedLogits, nil
	}
	if len(s.retainedHidden) != s.arch.Hidden*bf16Size {
		return nil, core.NewError("native.ArchSession.BoundaryLogits: no retained prefill state")
	}
	var logits []byte
	var err error
	if hiddenBuf := s.retainedHiddenBufferFor(s.retainedHidden); hiddenBuf != nil && s.headEnc != nil {
		if pinned, ok := s.ensureRetainedLogitsPinned(s.arch.Vocab * bf16Size); ok {
			logits, err = s.headEnc.encodeBufferInto(hiddenBuf, false, pinned.bytes)
			if err != nil {
				return nil, err
			}
			s.retainedLogits = logits
			s.sampleHeadLogits = nil
			return s.retainedLogits, nil
		}
		logits, err = s.headEnc.encodeBufferInto(hiddenBuf, false, s.sampleHeadLogits)
		if err == nil {
			s.sampleHeadLogits = logits
		}
	} else {
		logits, err = s.head(s.retainedHidden, false)
	}
	if err != nil {
		return nil, err
	}
	s.rememberRetainedLogits(logits)
	return s.retainedLogits, nil
}

// GenerateFromCacheLogitsEach greedily continues a restored cache from already
// captured boundary logits. The first token is selected directly from
// firstLogits; subsequent tokens use the resident K/V cache and normal native
// step path, so the prompt prefix is not replayed.
func (s *ArchSession) GenerateFromCacheLogitsEach(firstLogits []byte, maxNew, eosID int, yield func(int32) bool) ([]int32, error) {
	if maxNew <= 0 {
		return nil, core.NewError("native.ArchSession.GenerateFromCacheLogits: maxNew must be > 0")
	}
	if len(firstLogits) != s.arch.Vocab*bf16Size {
		return nil, core.NewError("native.ArchSession.GenerateFromCacheLogits: logits must be vocab bf16 bytes")
	}
	if s.pos+maxNew > s.maxLen {
		return nil, core.NewError("native.ArchSession.GenerateFromCacheLogits: sequence would exceed maxLen cache rows")
	}
	var gen []int32
	var err error
	withAutoreleasePool(func() {
		gen, err = s.generateFromLogitsInPool(firstLogits, maxNew, eosID, yield)
	})
	if err != nil {
		s.cachedIDs = nil
		s.resetRetainedHidden()
		return nil, err
	}
	s.cachedIDs = append(s.cachedIDs, gen...)
	return gen, nil
}

// GenerateSampledFromCacheLogitsEach samples a restored-cache continuation from
// already captured boundary logits. The first token is sampled from firstLogits;
// subsequent tokens reuse the resident K/V cache and sampled native step loop.
func (s *ArchSession) GenerateSampledFromCacheLogitsEach(firstLogits []byte, maxNew int, stopTokens []int32, sampler *model.Sampler, params model.SampleParams, transform model.TokenTransform, yield func(int32) bool) ([]int32, error) {
	if sampler == nil {
		return nil, core.NewError("native.ArchSession.GenerateSampledFromCacheLogits: nil sampler")
	}
	if maxNew <= 0 {
		return nil, core.NewError("native.ArchSession.GenerateSampledFromCacheLogits: maxNew must be > 0")
	}
	if len(firstLogits) != s.arch.Vocab*bf16Size {
		return nil, core.NewError("native.ArchSession.GenerateSampledFromCacheLogits: logits must be vocab bf16 bytes")
	}
	if s.pos+maxNew > s.maxLen {
		return nil, core.NewError("native.ArchSession.GenerateSampledFromCacheLogits: sequence would exceed maxLen cache rows")
	}
	var gen []int32
	var err error
	withAutoreleasePool(func() {
		gen, err = s.generateSampledFromLogitsInPool(firstLogits, maxNew, stopTokens, sampler, params, transform, yield, true)
	})
	if err != nil {
		s.cachedIDs = nil
		s.resetRetainedHidden()
		return nil, err
	}
	s.cachedIDs = append(s.cachedIDs, gen...)
	return gen, nil
}

// GenerateFromCacheEachTransformed is GenerateFromCacheEach with a committed-token
// transform applied before each generated token is written to the cache.
func (s *ArchSession) GenerateFromCacheEachTransformed(maxNew, eosID int, transform TokenTransform, yield func(int32) bool) ([]int32, error) {
	if maxNew <= 0 {
		return nil, core.NewError("native.ArchSession.GenerateFromCache: maxNew must be > 0")
	}
	if transform == nil && len(s.retainedLogits) == s.arch.Vocab*bf16Size {
		return s.GenerateFromCacheLogitsEach(s.retainedLogits, maxNew, eosID, yield)
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
	if hidden, ok, err := s.prefillRetainedTokensBatchedDense(ids, scope); ok || err != nil {
		return hidden, err
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

func (s *ArchSession) prefillPromptRetainedInPool(ids []int32) ([]byte, error) {
	if len(ids) == 0 {
		return nil, nil
	}
	if hidden, ok, err := s.prefillPromptRetainedGPUInputsInPool(ids); ok || err != nil {
		return hidden, err
	}
	var err error
	for _, id := range ids[:len(ids)-1] {
		if _, err = s.stepIDInPool(id); err != nil {
			return nil, err
		}
	}
	return s.stepIDRetainedInPool(ids[len(ids)-1])
}

func (s *ArchSession) prefillPromptRetainedGPUInputsInPool(ids []int32) ([]byte, bool, error) {
	if s.state.icb == nil || icbDisabledForTest || s.encNextInputsGPU == nil || s.plScratchNew == nil || chainedGPUInputsDisabled {
		return nil, false, nil
	}
	if len(ids) > 1 {
		if err := s.prefillCachedIDsGPUInputs(ids[:len(ids)-1]); err != nil {
			return nil, true, err
		}
	}
	return s.stepIDRetainedGPUInputsInPool(ids[len(ids)-1])
}

func (s *ArchSession) prefillRetainedTokensBatchedDense(ids []int32, scope string) ([]byte, bool, error) {
	if len(ids) == 0 {
		return nil, false, nil
	}
	if s.pos+len(ids) > s.maxLen {
		return nil, false, core.NewError(scope + ": sequence would exceed maxLen cache rows")
	}
	if s.perLayerInput != nil || s.state.icb != nil {
		return nil, false, nil
	}
	var embStack [16][]byte
	var embs [][]byte
	if len(ids) <= len(embStack) {
		embs = embStack[:len(ids)]
	} else {
		embs = make([][]byte, len(ids))
	}
	for i, id := range ids {
		emb, err := s.embed(id)
		if err != nil {
			return nil, false, err
		}
		embs[i] = emb
	}
	var (
		hidden []byte
		ok     bool
		err    error
	)
	dst := s.sampleHidden
	retained := false
	if pinned, pinnedOK := s.ensureRetainedHiddenPinned(s.arch.Hidden * bf16Size); pinnedOK {
		s.resetRetainedLogits()
		dst = pinned.bytes[:s.arch.Hidden*bf16Size]
		retained = true
	}
	withAutoreleasePool(func() {
		hidden, ok, err = s.state.stepTokensBatchedDenseLastInto(embs, s.pos, dst)
	})
	if err != nil || !ok {
		return nil, ok, err
	}
	if retained {
		s.sampleHidden = nil
		s.retainedHidden = hidden
	} else {
		s.sampleHidden = hidden
	}
	s.pos += len(ids)
	return hidden, true, nil
}

func (s *ArchSession) rememberRetainedHidden(hidden []byte) {
	if s == nil || len(hidden) != s.arch.Hidden*bf16Size {
		s.resetRetainedHidden()
		return
	}
	s.resetRetainedLogits()
	if len(s.retainedHidden) == len(hidden) && len(hidden) != 0 && unsafe.Pointer(&hidden[0]) == unsafe.Pointer(&s.retainedHidden[0]) {
		return
	}
	if pinned, ok := s.ensureRetainedHiddenPinned(len(hidden)); ok {
		copy(pinned.bytes, hidden)
		s.retainedHidden = pinned.bytes[:len(hidden)]
		return
	}
	retained := s.retainedHidden[:0]
	s.retainedHidden = append(retained, hidden...)
}

func (s *ArchSession) rememberRetainedHiddenFrom(ptr *byte) {
	if s == nil || ptr == nil || s.arch.Hidden <= 0 {
		s.resetRetainedHidden()
		return
	}
	s.resetRetainedLogits()
	n := s.arch.Hidden * bf16Size
	if pinned, ok := s.ensureRetainedHiddenPinned(n); ok {
		s.retainedHidden = pinned.bytes[:n]
		copy(s.retainedHidden, unsafe.Slice(ptr, n))
		return
	}
	if cap(s.retainedHidden) < n {
		s.closeRetainedHiddenPinned()
		s.retainedHidden = make([]byte, n)
	} else {
		s.retainedHidden = s.retainedHidden[:n]
	}
	copy(s.retainedHidden, unsafe.Slice(ptr, n))
}

func (s *ArchSession) resetRetainedHidden() {
	if s == nil {
		return
	}
	s.resetRetainedLogits()
	if s.retainedHiddenPinned != nil && s.retainedHiddenPinned.bytes != nil {
		s.retainedHidden = s.retainedHiddenPinned.bytes[:0]
		return
	}
	s.retainedHidden = s.retainedHidden[:0]
}

func (s *ArchSession) rememberRetainedLogits(logits []byte) {
	if s == nil || len(logits) != s.arch.Vocab*bf16Size {
		s.resetRetainedLogits()
		return
	}
	if len(s.retainedLogits) == len(logits) && len(logits) != 0 && unsafe.Pointer(&logits[0]) == unsafe.Pointer(&s.retainedLogits[0]) {
		return
	}
	if pinned, ok := s.ensureRetainedLogitsPinned(len(logits)); ok {
		copy(pinned.bytes, logits)
		s.retainedLogits = pinned.bytes
		return
	}
	retained := s.retainedLogits[:0]
	s.retainedLogits = append(retained, logits...)
}

func (s *ArchSession) resetRetainedLogits() {
	if s == nil {
		return
	}
	if s.retainedLogitsPinned != nil && s.retainedLogitsPinned.bytes != nil {
		s.retainedLogits = s.retainedLogitsPinned.bytes[:0]
		return
	}
	s.retainedLogits = s.retainedLogits[:0]
}

func (s *ArchSession) ensureRetainedHiddenPinned(n int) (*pinnedNoCopyBytes, bool) {
	if s == nil || n <= 0 {
		return nil, false
	}
	if s.retainedHiddenPinned != nil {
		if len(s.retainedHiddenPinned.bytes) == n && s.retainedHiddenPinned.buf != nil {
			return s.retainedHiddenPinned, true
		}
		s.closeRetainedHiddenPinned()
	}
	pinned, err := newPinnedNoCopyBytes(n)
	if err != nil {
		return nil, false
	}
	s.retainedHiddenPinned = pinned
	return pinned, true
}

func (s *ArchSession) closeRetainedHiddenPinned() {
	if s == nil || s.retainedHiddenPinned == nil {
		return
	}
	s.retainedHiddenPinned.Close()
	s.retainedHiddenPinned = nil
	s.retainedHidden = nil
}

func (s *ArchSession) ensureRetainedLogitsPinned(n int) (*pinnedNoCopyBytes, bool) {
	if s == nil || n <= 0 {
		return nil, false
	}
	if s.retainedLogitsPinned != nil {
		if len(s.retainedLogitsPinned.bytes) == n && s.retainedLogitsPinned.buf != nil {
			return s.retainedLogitsPinned, true
		}
		s.closeRetainedLogitsPinned()
	}
	pinned, err := newPinnedNoCopyBytes(n)
	if err != nil {
		return nil, false
	}
	s.retainedLogitsPinned = pinned
	return pinned, true
}

func (s *ArchSession) closeRetainedLogitsPinned() {
	if s == nil || s.retainedLogitsPinned == nil {
		return
	}
	s.retainedLogitsPinned.Close()
	s.retainedLogitsPinned = nil
	s.retainedLogits = nil
}

func (s *ArchSession) retainedHiddenBuffer() metal.MTLBuffer {
	if s == nil || len(s.retainedHidden) == 0 || s.retainedHiddenPinned == nil || s.retainedHiddenPinned.buf == nil || len(s.retainedHiddenPinned.bytes) != len(s.retainedHidden) {
		return nil
	}
	if unsafe.Pointer(&s.retainedHidden[0]) != unsafe.Pointer(&s.retainedHiddenPinned.bytes[0]) {
		return nil
	}
	return s.retainedHiddenPinned.buf
}

func (s *ArchSession) retainedHiddenBufferFor(hidden []byte) metal.MTLBuffer {
	if s == nil || len(hidden) == 0 || len(hidden) != len(s.retainedHidden) || len(s.retainedHidden) == 0 {
		return nil
	}
	if unsafe.Pointer(&hidden[0]) != unsafe.Pointer(&s.retainedHidden[0]) {
		return nil
	}
	return s.retainedHiddenBuffer()
}

func (s *ArchSession) retainedLogitsBuffer() metal.MTLBuffer {
	if s == nil || len(s.retainedLogits) == 0 || s.retainedLogitsPinned == nil || s.retainedLogitsPinned.buf == nil || len(s.retainedLogitsPinned.bytes) != len(s.retainedLogits) {
		return nil
	}
	if unsafe.Pointer(&s.retainedLogits[0]) != unsafe.Pointer(&s.retainedLogitsPinned.bytes[0]) {
		return nil
	}
	return s.retainedLogitsPinned.buf
}

func (s *ArchSession) retainedLogitsBufferFor(logits []byte) metal.MTLBuffer {
	if s == nil || len(logits) == 0 || len(logits) != len(s.retainedLogits) || len(s.retainedLogits) == 0 {
		return nil
	}
	if unsafe.Pointer(&logits[0]) != unsafe.Pointer(&s.retainedLogits[0]) {
		return nil
	}
	return s.retainedLogitsBuffer()
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
	emb, err := s.embedID(id)
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
		icb := s.state.icb
		if icb.lastOutPtr == nil {
			icb.cacheLastOutContents()
		}
		icb.stepBodyNoResult(emb, s.pos, pli)
		h = s.retainHiddenReadbackFrom(icb.lastOutPtr)
		if h == nil {
			h = make([]byte, s.arch.Hidden*bf16Size)
			icb.copyLastOutInto(h)
		}
	} else if h, err = s.state.stepToken(emb, s.pos); err != nil {
		return nil, err
	}
	ptEnd(1, _ptICB)
	s.pos++
	return h, nil
}

func (s *ArchSession) stepIDRetainedInPool(id int32) ([]byte, error) {
	emb, err := s.embedID(id)
	if err != nil {
		return nil, err
	}
	var pli []byte
	if s.perLayerInput != nil {
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
	if s.state.icb != nil && !icbDisabledForTest {
		icb := s.state.icb
		if icb.lastOutPtr == nil {
			icb.cacheLastOutContents()
		}
		icb.stepBodyNoResult(emb, s.pos, pli)
		h = s.retainHiddenReadbackFrom(icb.lastOutPtr)
		if h == nil {
			h = make([]byte, s.arch.Hidden*bf16Size)
			icb.copyLastOutInto(h)
		}
	} else if pinned, ok := s.ensureRetainedHiddenPinned(s.arch.Hidden * bf16Size); ok {
		s.resetRetainedLogits()
		h, err = s.state.stepTokenInto(emb, s.pos, pinned.bytes[:s.arch.Hidden*bf16Size])
		if err != nil {
			return nil, err
		}
		s.retainedHidden = h
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

func (s *ArchSession) generateFromLogitsInPool(firstLogits []byte, maxNew, eosID int, yield func(int32) bool) ([]int32, error) {
	next, err := greedyBF16Suppressed(firstLogits, s.arch.Vocab, nil)
	if err != nil {
		return nil, err
	}
	gen := make([]int32, 0, maxNew)
	gen = append(gen, next)
	stop := (yield != nil && !yield(next)) || (eosID >= 0 && int(next) == eosID)
	if s.encNextInputsGPU != nil && s.plScratchNew != nil && s.state.icb != nil && s.headEnc != nil && s.greedy != nil &&
		!stepGreedyChainDisabled && !chainedGPUInputsDisabled && !icbDisabledForTest {
		if pipelinedGPUDecodeEnabled && s.recordPeerICB != nil {
			return s.generatePipelinedGPUTail(gen, maxNew, eosID, nil, yield, stop)
		}
		return s.generateChainedGPUTail(gen, maxNew, eosID, nil, yield, stop)
	}
	var hidden []byte
	for !stop && len(gen) < maxNew {
		prev := gen[len(gen)-1]
		if hidden, err = s.stepIDRetainedInPool(prev); err != nil {
			return nil, err
		}
		if next, err = s.headGreedyOrLogits(hidden, nil, nil, nil, false); err != nil {
			return nil, err
		}
		gen = append(gen, next)
		s.rememberRetainedHidden(hidden)
		stop = (yield != nil && !yield(next)) || (eosID >= 0 && int(next) == eosID)
	}
	if hidden, err = s.stepIDRetainedInPool(gen[len(gen)-1]); err != nil {
		return nil, err
	}
	s.rememberRetainedHidden(hidden)
	return gen, nil
}

func (s *ArchSession) generateSampledFromLogitsInPool(firstLogits []byte, maxNew int, stopTokens []int32, sampler *model.Sampler, params model.SampleParams, transform model.TokenTransform, yield func(int32) bool, cacheFinal bool) ([]int32, error) {
	gen := make([]int32, 0, maxNew)
	history := s.sampleHistoryScratchFor(params, maxNew)
	finalHistory := history
	defer func() { s.sampleHistory = finalHistory }()

	pickParams := params
	if params.MinTokensBeforeStop > 0 {
		pickParams.SuppressTokens = s.suppressionTokensScratch(params.SuppressTokens, stopTokens)
	}
	next, err := s.sampleTokenFromLogits(firstLogits, sampler, pickParams, history)
	if err != nil {
		return nil, err
	}
	if transform != nil {
		next = transform(next)
	}
	gen = append(gen, next)
	if params.RepeatPenalty > 1 {
		history = append(history, next)
		finalHistory = history
	}
	stop := (yield != nil && !yield(next)) || nativeTokenInSet(next, stopTokens)
	if !cacheFinal && (stop || len(gen) >= maxNew) {
		return gen, nil
	}
	if !stop && len(gen) < maxNew && s.sampledChainedGPUTailCanContinue(params, history, transform) {
		var tail []int32
		tail, finalHistory, err = s.generateSampledChainedGPUTail(gen, maxNew, stopTokens, sampler, params, yield, cacheFinal, 0, history)
		if err != nil {
			return nil, err
		}
		return tail, nil
	}
	hidden, err := s.stepIDRetainedInPool(next)
	if err != nil {
		return nil, err
	}
	s.rememberRetainedHidden(hidden)
	if stop || len(gen) >= maxNew {
		return gen, nil
	}
	var tail []int32
	tail, finalHistory, err = s.generateSampledFromHiddenInPoolWithHistory(hidden, maxNew-len(gen), stopTokens, sampler, params, transform, yield, cacheFinal, len(gen), history)
	if err != nil {
		return nil, err
	}
	gen = append(gen, tail...)
	return gen, nil
}

func (s *ArchSession) sampleTokenFromLogits(logits []byte, sampler *model.Sampler, params model.SampleParams, history []int32) (int32, error) {
	if sampledGreedyParamsEligible(params) {
		return greedyBF16Suppressed(logits, s.arch.Vocab, params.SuppressTokens)
	}
	if sampledTopOneGreedyParamsEligible(params, history) {
		sampler.Draw()
		return greedyBF16Suppressed(logits, s.arch.Vocab, params.SuppressTokens)
	}
	if sampleLogitsTokenCPUPreferred(params, s.arch.Vocab) {
		return sampleSmallVocabBF16(logits, s.arch.Vocab, sampler, params)
	}
	if !retainedLogitsCompactSampleEligible(params, history) {
		logitsBuf := s.retainedLogitsBufferFor(logits)
		if logitsBuf != nil && s.retainedLogitsSampleParamsEligible(params) {
			token, ok, err := s.headEnc.sampleLogitsBufferInPool(logitsBuf, params, sampler.Draw(), history)
			if err != nil {
				return 0, err
			}
			if ok {
				return token, nil
			}
		}
	}
	if params.TopK > 0 && params.TopK <= headSampleTopKMaxK && (params.RepeatPenalty <= 1 || len(history) == 0) {
		candidateLogits, candidateIDs, ok, err := s.sampleTopKCandidatesFromLogits(logits, params.TopK, params.SuppressTokens)
		if err != nil {
			return 0, err
		}
		if ok {
			return sampleSortedBF16Candidates(candidateLogits, candidateIDs, sampler, params)
		}
	}
	pickLogits := logits
	var err error
	if params.RepeatPenalty > 1 {
		pickLogits, err = s.repeatPenaltyLogitsScratch(logits, s.arch.Vocab, history, params.RepeatPenalty)
		if err != nil {
			return 0, err
		}
	}
	return s.sampleVocabBF16(pickLogits, s.arch.Vocab, sampler, params)
}

func retainedLogitsCompactSampleEligible(params model.SampleParams, history []int32) bool {
	return params.TopK > 0 && params.TopK <= headSampleTopKMaxK && (params.RepeatPenalty <= 1 || len(history) == 0)
}

func (s *ArchSession) sampleTopKCandidatesFromLogits(logits []byte, topK int, suppress []int32) ([]byte, []int32, bool, error) {
	vocab := s.arch.Vocab
	if len(logits) != vocab*bf16Size {
		return nil, nil, true, core.NewError("native.ArchSession.sampleTopKCandidatesFromLogits: logits must be vocab bf16 bytes")
	}
	if topK <= 0 || topK > headSampleTopKMaxK || topK > vocab {
		return nil, nil, false, nil
	}
	if cap(s.sampleCandidateLogits) < topK*bf16Size {
		s.sampleCandidateLogits = make([]byte, topK*bf16Size)
	} else {
		s.sampleCandidateLogits = s.sampleCandidateLogits[:topK*bf16Size]
	}
	if cap(s.sampleCandidateIDs) < topK {
		s.sampleCandidateIDs = make([]int32, topK)
	} else {
		s.sampleCandidateIDs = s.sampleCandidateIDs[:topK]
	}
	var scores [headSampleTopKMaxK]float32
	count := 0
	for id := 0; id < vocab; id++ {
		if tokenSuppressed(id, suppress) {
			continue
		}
		off := id * bf16Size
		v := bf16ToF32(logits[off], logits[off+1])
		insert := count
		for insert > 0 && (v > scores[insert-1] || (v == scores[insert-1] && int32(id) < s.sampleCandidateIDs[insert-1])) {
			insert--
		}
		if insert >= topK {
			continue
		}
		if count < topK {
			count++
		}
		for j := count - 1; j > insert; j-- {
			scores[j] = scores[j-1]
			s.sampleCandidateIDs[j] = s.sampleCandidateIDs[j-1]
			prev := (j - 1) * bf16Size
			dst := j * bf16Size
			s.sampleCandidateLogits[dst] = s.sampleCandidateLogits[prev]
			s.sampleCandidateLogits[dst+1] = s.sampleCandidateLogits[prev+1]
		}
		scores[insert] = v
		s.sampleCandidateIDs[insert] = int32(id)
		dst := insert * bf16Size
		s.sampleCandidateLogits[dst] = logits[off]
		s.sampleCandidateLogits[dst+1] = logits[off+1]
	}
	if count == 0 {
		return nil, nil, true, core.NewError("native.ArchSession.sampleTopKCandidatesFromLogits: all vocab ids are suppressed")
	}
	return s.sampleCandidateLogits[:count*bf16Size], s.sampleCandidateIDs[:count], true, nil
}

func sampleSortedBF16Candidates(logits []byte, ids []int32, sampler *model.Sampler, params model.SampleParams) (int32, error) {
	if sampler == nil {
		return 0, core.NewError("native.sampleSortedBF16Candidates: nil sampler")
	}
	if len(ids) == 0 {
		return 0, core.NewError("native.sampleSortedBF16Candidates: empty candidates")
	}
	if len(ids) > headSampleTopKMaxK {
		return 0, core.NewError("native.sampleSortedBF16Candidates: too many candidates")
	}
	if len(logits) != len(ids)*bf16Size {
		return 0, core.NewError("native.sampleSortedBF16Candidates: logits must be candidate bf16 bytes")
	}
	if sampledGreedyParamsEligible(params) {
		best := -1
		var bestV float32
		for i, id := range ids {
			if nativeTokenInSet(id, params.SuppressTokens) {
				continue
			}
			v := bf16ToF32(logits[i*bf16Size], logits[i*bf16Size+1])
			if best < 0 || v > bestV {
				best, bestV = i, v
			}
		}
		if best < 0 {
			return 0, core.NewError("native.sampleSortedBF16Candidates: all candidates are suppressed")
		}
		return ids[best], nil
	}
	if params.TopK == 1 {
		for _, id := range ids {
			if nativeTokenInSet(id, params.SuppressTokens) {
				continue
			}
			sampler.Draw()
			return id, nil
		}
		return 0, core.NewError("native.sampleSortedBF16Candidates: all candidates are suppressed")
	}
	temp := params.Temperature
	if temp <= 0 {
		temp = 1
	}
	var weights [headSampleTopKMaxK]float32
	maxL := float32(math.Inf(-1))
	allowed := 0
	for i, id := range ids {
		if nativeTokenInSet(id, params.SuppressTokens) {
			weights[i] = float32(math.Inf(-1))
			continue
		}
		v := bf16ToF32(logits[i*bf16Size], logits[i*bf16Size+1]) / temp
		weights[i] = v
		allowed++
		if v > maxL {
			maxL = v
		}
	}
	if allowed == 0 {
		return 0, core.NewError("native.sampleSortedBF16Candidates: all candidates are suppressed")
	}
	for i := range ids {
		if weights[i] == float32(math.Inf(-1)) {
			weights[i] = 0
			continue
		}
		weights[i] = float32(math.Exp(float64(weights[i] - maxL)))
	}
	keep := len(ids)
	if params.TopK > 0 && params.TopK < keep {
		keep = params.TopK
	}
	if params.TopP > 0 && params.TopP < 1 {
		var keptMass float32
		for i := 0; i < keep; i++ {
			keptMass += weights[i]
		}
		var cum float32
		n := 0
		for n < keep {
			cum += weights[n]
			n++
			if cum >= params.TopP*keptMass {
				break
			}
		}
		keep = n
	}
	if params.MinP > 0 && keep > 0 {
		threshold := weights[0] * params.MinP
		n := 0
		for n < keep && weights[n] >= threshold {
			n++
		}
		if n > 0 {
			keep = n
		}
	}
	var ksum float32
	for i := 0; i < keep; i++ {
		ksum += weights[i]
	}
	if ksum == 0 {
		return 0, core.NewError("native.sampleSortedBF16Candidates: empty sampled distribution")
	}
	target := sampler.Draw() * ksum
	var acc float32
	for i := 0; i < keep; i++ {
		acc += weights[i]
		if acc >= target {
			return ids[i], nil
		}
	}
	return ids[keep-1], nil
}

func sampleSmallVocabBF16(logits []byte, vocab int, sampler *model.Sampler, params model.SampleParams) (int32, error) {
	if sampler == nil {
		return 0, core.NewError("native.sampleSmallVocabBF16: nil sampler")
	}
	if vocab <= 0 || vocab > headSampleTopKMaxK || len(logits) != vocab*bf16Size {
		return 0, core.NewError("native.sampleSmallVocabBF16: logits must be small-vocab bf16 bytes")
	}
	if sampledGreedyParamsEligible(params) {
		return greedyBF16Suppressed(logits, vocab, params.SuppressTokens)
	}
	if params.TopK == 1 {
		next, err := greedyBF16Suppressed(logits, vocab, params.SuppressTokens)
		if err != nil {
			return 0, err
		}
		sampler.Draw()
		return next, nil
	}
	temp := params.Temperature
	if temp <= 0 {
		temp = 1
	}
	var scaled [headSampleTopKMaxK]float32
	var probs [headSampleTopKMaxK]float32
	var order [headSampleTopKMaxK]int
	maxL := float32(math.Inf(-1))
	allowed := 0
	for i := 0; i < vocab; i++ {
		if tokenSuppressed(i, params.SuppressTokens) {
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
		return 0, core.NewError("native.sampleSmallVocabBF16: all tokens are suppressed")
	}
	var sum float32
	for i := 0; i < vocab; i++ {
		e := float32(math.Exp(float64(scaled[i] - maxL)))
		probs[i] = e
		sum += e
		order[i] = i
	}
	for i := 0; i < vocab; i++ {
		probs[i] /= sum
	}
	for i := 1; i < vocab; i++ {
		key := order[i]
		j := i - 1
		for j >= 0 && probs[order[j]] < probs[key] {
			order[j+1] = order[j]
			j--
		}
		order[j+1] = key
	}
	keep := vocab
	if params.TopK > 0 && params.TopK < keep {
		keep = params.TopK
	}
	if params.TopP > 0 && params.TopP < 1 {
		var keptMass float32
		for i := 0; i < keep; i++ {
			keptMass += probs[order[i]]
		}
		var cum float32
		n := 0
		for n < keep {
			cum += probs[order[n]]
			n++
			if cum >= params.TopP*keptMass {
				break
			}
		}
		keep = n
	}
	if params.MinP > 0 && keep > 0 {
		threshold := probs[order[0]] * params.MinP
		n := 0
		for n < keep && probs[order[n]] >= threshold {
			n++
		}
		if n > 0 {
			keep = n
		}
	}
	var ksum float32
	for i := 0; i < keep; i++ {
		ksum += probs[order[i]]
	}
	if ksum == 0 {
		return 0, core.NewError("native.sampleSmallVocabBF16: empty sampled distribution")
	}
	target := sampler.Draw() * ksum
	var acc float32
	for i := 0; i < keep; i++ {
		acc += probs[order[i]]
		if acc >= target {
			return int32(order[i]), nil
		}
	}
	return int32(order[keep-1]), nil
}

func (s *ArchSession) sampleVocabBF16(logits []byte, vocab int, sampler *model.Sampler, params model.SampleParams) (int32, error) {
	if vocab <= headSampleTopKMaxK {
		return sampleSmallVocabBF16(logits, vocab, sampler, params)
	}
	if sampler == nil {
		return 0, core.NewError("native.ArchSession.sampleVocabBF16: nil sampler")
	}
	if vocab <= 0 || len(logits) != vocab*bf16Size {
		return 0, core.NewError("native.ArchSession.sampleVocabBF16: logits must be vocab bf16 bytes")
	}
	if sampledGreedyParamsEligible(params) {
		return greedyBF16Suppressed(logits, vocab, params.SuppressTokens)
	}
	if params.TopK == 1 {
		next, err := greedyBF16Suppressed(logits, vocab, params.SuppressTokens)
		if err != nil {
			return 0, err
		}
		sampler.Draw()
		return next, nil
	}
	rankFilter := sampleRankPrefixPreferred(params, vocab)
	s.sampleScaled = nil
	temp := params.Temperature
	if temp <= 0 {
		temp = 1
	}
	noSuppress := len(params.SuppressTokens) == 0
	maxL := float32(math.Inf(-1))
	allowed := 0
	if noSuppress {
		allowed = vocab
		for i := 0; i < vocab; i++ {
			v := bf16ToF32(logits[i*bf16Size], logits[i*bf16Size+1]) / temp
			if v > maxL {
				maxL = v
			}
		}
	} else {
		for i := 0; i < vocab; i++ {
			if tokenSuppressed(i, params.SuppressTokens) {
				continue
			}
			v := bf16ToF32(logits[i*bf16Size], logits[i*bf16Size+1]) / temp
			allowed++
			if v > maxL {
				maxL = v
			}
		}
	}
	if allowed == 0 {
		return 0, core.NewError("native.ArchSession.sampleVocabBF16: all tokens are suppressed")
	}
	if !rankFilter {
		s.sampleProbs = nil
		s.sampleOrder = nil
		if noSuppress {
			return sampleVocabBF16InVocabOrderStreamingNoSuppress(logits, vocab, sampler, temp, maxL)
		}
		return sampleVocabBF16InVocabOrderStreaming(logits, vocab, sampler, params, temp, maxL)
	}
	s.sampleProbs = nil
	if cap(s.sampleOrder) < vocab {
		s.sampleOrder = make([]int32, vocab)
	} else {
		s.sampleOrder = s.sampleOrder[:vocab]
	}
	for i := 0; i < vocab; i++ {
		s.sampleOrder[i] = int32(i)
	}
	if noSuppress {
		probTotal := sampleVocabBF16WeightTotalNoSuppress(logits, vocab, temp, maxL)
		keep := rankSampleOrderPrefixLogitsNoSuppress(s.sampleOrder, logits, probTotal, params, temp, maxL)
		var ksum float32
		for i := 0; i < keep; i++ {
			ksum += sampleVocabBF16IDWeightNoSuppress(logits, s.sampleOrder[i], temp, maxL)
		}
		if ksum == 0 {
			return 0, core.NewError("native.ArchSession.sampleVocabBF16: empty sampled distribution")
		}
		target := sampler.Draw() * ksum
		var acc float32
		for i := 0; i < keep; i++ {
			acc += sampleVocabBF16IDWeightNoSuppress(logits, s.sampleOrder[i], temp, maxL)
			if acc >= target {
				return s.sampleOrder[i], nil
			}
		}
		return s.sampleOrder[keep-1], nil
	}
	probTotal := sampleVocabBF16WeightTotal(logits, vocab, params, temp, maxL)
	keep := rankSampleOrderPrefixLogits(s.sampleOrder, logits, probTotal, params, temp, maxL)
	var ksum float32
	for i := 0; i < keep; i++ {
		ksum += sampleVocabBF16IDWeight(logits, s.sampleOrder[i], params, temp, maxL)
	}
	if ksum == 0 {
		return 0, core.NewError("native.ArchSession.sampleVocabBF16: empty sampled distribution")
	}
	target := sampler.Draw() * ksum
	var acc float32
	for i := 0; i < keep; i++ {
		acc += sampleVocabBF16IDWeight(logits, s.sampleOrder[i], params, temp, maxL)
		if acc >= target {
			return s.sampleOrder[i], nil
		}
	}
	return s.sampleOrder[keep-1], nil
}

func sampleVocabBF16InVocabOrderStreamingNoSuppress(logits []byte, vocab int, sampler *model.Sampler, temp, maxL float32) (int32, error) {
	var sum float32
	for i := 0; i < vocab; i++ {
		v := bf16ToF32(logits[i*bf16Size], logits[i*bf16Size+1]) / temp
		sum += float32(math.Exp(float64(v - maxL)))
	}
	if sum == 0 {
		return 0, core.NewError("native.ArchSession.sampleVocabBF16: empty sampled distribution")
	}
	target := sampler.Draw() * sum
	var acc float32
	for i := 0; i < vocab; i++ {
		v := bf16ToF32(logits[i*bf16Size], logits[i*bf16Size+1]) / temp
		acc += float32(math.Exp(float64(v - maxL)))
		if acc >= target {
			return int32(i), nil
		}
	}
	return int32(vocab - 1), nil
}

func sampleVocabBF16InVocabOrderStreaming(logits []byte, vocab int, sampler *model.Sampler, params model.SampleParams, temp, maxL float32) (int32, error) {
	var sum float32
	for i := 0; i < vocab; i++ {
		if tokenSuppressed(i, params.SuppressTokens) {
			continue
		}
		v := bf16ToF32(logits[i*bf16Size], logits[i*bf16Size+1]) / temp
		sum += float32(math.Exp(float64(v - maxL)))
	}
	if sum == 0 {
		return 0, core.NewError("native.ArchSession.sampleVocabBF16: empty sampled distribution")
	}
	target := sampler.Draw() * sum
	var acc float32
	for i := 0; i < vocab; i++ {
		e := float32(0)
		if !tokenSuppressed(i, params.SuppressTokens) {
			v := bf16ToF32(logits[i*bf16Size], logits[i*bf16Size+1]) / temp
			e = float32(math.Exp(float64(v - maxL)))
		}
		acc += e
		if acc >= target {
			return int32(i), nil
		}
	}
	return int32(vocab - 1), nil
}

func sampleVocabBF16WeightTotal(logits []byte, vocab int, params model.SampleParams, temp, maxL float32) float32 {
	var sum float32
	for i := 0; i < vocab; i++ {
		sum += sampleVocabBF16IDWeight(logits, int32(i), params, temp, maxL)
	}
	return sum
}

func sampleVocabBF16WeightTotalNoSuppress(logits []byte, vocab int, temp, maxL float32) float32 {
	var sum float32
	for i := 0; i < vocab; i++ {
		sum += sampleVocabBF16IDWeightNoSuppress(logits, int32(i), temp, maxL)
	}
	return sum
}

func sampleVocabBF16IDWeight(logits []byte, id int32, params model.SampleParams, temp, maxL float32) float32 {
	if id < 0 || int(id) >= len(logits)/bf16Size || nativeTokenInSet(id, params.SuppressTokens) {
		return 0
	}
	v := bf16ToF32(logits[int(id)*bf16Size], logits[int(id)*bf16Size+1]) / temp
	return float32(math.Exp(float64(v - maxL)))
}

func sampleVocabBF16IDWeightNoSuppress(logits []byte, id int32, temp, maxL float32) float32 {
	v := bf16ToF32(logits[int(id)*bf16Size], logits[int(id)*bf16Size+1]) / temp
	return float32(math.Exp(float64(v - maxL)))
}

func rankSampleOrderPrefixLogits(order []int32, logits []byte, probTotal float32, params model.SampleParams, temp, maxL float32) int {
	if len(order) == 0 {
		return 0
	}
	if probTotal <= 0 {
		probTotal = 1
	}
	heapifySampleOrderLogits(order, logits, params)
	heapLen := len(order)
	popped := 0
	keptMass := float32(0)
	if params.TopK > 0 && params.TopK < heapLen {
		for popped < params.TopK {
			id := popSampleOrderHeapLogits(order, logits, params, heapLen)
			heapLen--
			popped++
			keptMass += sampleVocabBF16IDWeight(logits, id, params, temp, maxL)
		}
		reverseSampleOrderTailToPrefix(order, popped)
		keep := popped
		if params.TopP > 0 && params.TopP < 1 {
			keep = sampleOrderTopPKeepLogits(order, logits, params, temp, maxL, keep, params.TopP*keptMass)
		}
		return sampleOrderMinPKeepLogits(order, logits, params, temp, maxL, keep)
	}
	if params.TopP > 0 && params.TopP < 1 {
		target := params.TopP * probTotal
		for heapLen > 0 {
			id := popSampleOrderHeapLogits(order, logits, params, heapLen)
			heapLen--
			popped++
			keptMass += sampleVocabBF16IDWeight(logits, id, params, temp, maxL)
			if keptMass >= target {
				break
			}
		}
		reverseSampleOrderTailToPrefix(order, popped)
		return sampleOrderMinPKeepLogits(order, logits, params, temp, maxL, popped)
	}
	if params.MinP > 0 {
		id := popSampleOrderHeapLogits(order, logits, params, heapLen)
		heapLen--
		popped++
		threshold := sampleVocabBF16IDWeight(logits, id, params, temp, maxL) * params.MinP
		for heapLen > 0 && sampleVocabBF16IDWeight(logits, order[0], params, temp, maxL) >= threshold {
			popSampleOrderHeapLogits(order, logits, params, heapLen)
			heapLen--
			popped++
		}
		reverseSampleOrderTailToPrefix(order, popped)
		return popped
	}
	return len(order)
}

func sampleOrderTopPKeepLogits(order []int32, logits []byte, params model.SampleParams, temp, maxL float32, keep int, targetMass float32) int {
	var cum float32
	n := 0
	for n < keep {
		cum += sampleVocabBF16IDWeight(logits, order[n], params, temp, maxL)
		n++
		if cum >= targetMass {
			break
		}
	}
	return n
}

func sampleOrderMinPKeepLogits(order []int32, logits []byte, params model.SampleParams, temp, maxL float32, keep int) int {
	if params.MinP <= 0 || keep <= 0 {
		return keep
	}
	threshold := sampleVocabBF16IDWeight(logits, order[0], params, temp, maxL) * params.MinP
	n := 0
	for n < keep && sampleVocabBF16IDWeight(logits, order[n], params, temp, maxL) >= threshold {
		n++
	}
	if n > 0 {
		return n
	}
	return keep
}

func rankSampleOrderPrefixLogitsNoSuppress(order []int32, logits []byte, probTotal float32, params model.SampleParams, temp, maxL float32) int {
	if len(order) == 0 {
		return 0
	}
	if probTotal <= 0 {
		probTotal = 1
	}
	heapifySampleOrderLogitsNoSuppress(order, logits)
	heapLen := len(order)
	popped := 0
	keptMass := float32(0)
	if params.TopK > 0 && params.TopK < heapLen {
		for popped < params.TopK {
			id := popSampleOrderHeapLogitsNoSuppress(order, logits, heapLen)
			heapLen--
			popped++
			keptMass += sampleVocabBF16IDWeightNoSuppress(logits, id, temp, maxL)
		}
		reverseSampleOrderTailToPrefix(order, popped)
		keep := popped
		if params.TopP > 0 && params.TopP < 1 {
			keep = sampleOrderTopPKeepLogitsNoSuppress(order, logits, temp, maxL, keep, params.TopP*keptMass)
		}
		return sampleOrderMinPKeepLogitsNoSuppress(order, logits, temp, maxL, keep, params.MinP)
	}
	if params.TopP > 0 && params.TopP < 1 {
		target := params.TopP * probTotal
		for heapLen > 0 {
			id := popSampleOrderHeapLogitsNoSuppress(order, logits, heapLen)
			heapLen--
			popped++
			keptMass += sampleVocabBF16IDWeightNoSuppress(logits, id, temp, maxL)
			if keptMass >= target {
				break
			}
		}
		reverseSampleOrderTailToPrefix(order, popped)
		return sampleOrderMinPKeepLogitsNoSuppress(order, logits, temp, maxL, popped, params.MinP)
	}
	if params.MinP > 0 {
		id := popSampleOrderHeapLogitsNoSuppress(order, logits, heapLen)
		heapLen--
		popped++
		threshold := sampleVocabBF16IDWeightNoSuppress(logits, id, temp, maxL) * params.MinP
		for heapLen > 0 && sampleVocabBF16IDWeightNoSuppress(logits, order[0], temp, maxL) >= threshold {
			popSampleOrderHeapLogitsNoSuppress(order, logits, heapLen)
			heapLen--
			popped++
		}
		reverseSampleOrderTailToPrefix(order, popped)
		return popped
	}
	return len(order)
}

func sampleOrderTopPKeepLogitsNoSuppress(order []int32, logits []byte, temp, maxL float32, keep int, targetMass float32) int {
	var cum float32
	n := 0
	for n < keep {
		cum += sampleVocabBF16IDWeightNoSuppress(logits, order[n], temp, maxL)
		n++
		if cum >= targetMass {
			break
		}
	}
	return n
}

func sampleOrderMinPKeepLogitsNoSuppress(order []int32, logits []byte, temp, maxL float32, keep int, minP float32) int {
	if minP <= 0 || keep <= 0 {
		return keep
	}
	threshold := sampleVocabBF16IDWeightNoSuppress(logits, order[0], temp, maxL) * minP
	n := 0
	for n < keep && sampleVocabBF16IDWeightNoSuppress(logits, order[n], temp, maxL) >= threshold {
		n++
	}
	if n > 0 {
		return n
	}
	return keep
}

func heapifySampleOrderLogits(order []int32, logits []byte, params model.SampleParams) {
	for i := len(order)/2 - 1; i >= 0; i-- {
		siftSampleOrderHeapLogits(order, logits, params, i, len(order))
	}
}

func popSampleOrderHeapLogits(order []int32, logits []byte, params model.SampleParams, heapLen int) int32 {
	top := order[0]
	last := heapLen - 1
	order[0] = order[last]
	order[last] = top
	siftSampleOrderHeapLogits(order, logits, params, 0, last)
	return top
}

func siftSampleOrderHeapLogits(order []int32, logits []byte, params model.SampleParams, root, heapLen int) {
	for {
		child := root*2 + 1
		if child >= heapLen {
			return
		}
		if right := child + 1; right < heapLen && sampleOrderLogitsLess(order[right], order[child], logits, params) {
			child = right
		}
		if !sampleOrderLogitsLess(order[child], order[root], logits, params) {
			return
		}
		order[root], order[child] = order[child], order[root]
		root = child
	}
}

func sampleOrderLogitsLess(a, b int32, logits []byte, params model.SampleParams) bool {
	aSuppressed, bSuppressed := nativeTokenInSet(a, params.SuppressTokens), nativeTokenInSet(b, params.SuppressTokens)
	if aSuppressed || bSuppressed {
		if aSuppressed != bSuppressed {
			return !aSuppressed
		}
		return a < b
	}
	ai, bi := int(a)*bf16Size, int(b)*bf16Size
	av, bv := bf16ToF32(logits[ai], logits[ai+1]), bf16ToF32(logits[bi], logits[bi+1])
	return av > bv || (av == bv && a < b)
}

func heapifySampleOrderLogitsNoSuppress(order []int32, logits []byte) {
	for i := len(order)/2 - 1; i >= 0; i-- {
		siftSampleOrderHeapLogitsNoSuppress(order, logits, i, len(order))
	}
}

func popSampleOrderHeapLogitsNoSuppress(order []int32, logits []byte, heapLen int) int32 {
	top := order[0]
	last := heapLen - 1
	order[0] = order[last]
	order[last] = top
	siftSampleOrderHeapLogitsNoSuppress(order, logits, 0, last)
	return top
}

func siftSampleOrderHeapLogitsNoSuppress(order []int32, logits []byte, root, heapLen int) {
	for {
		child := root*2 + 1
		if child >= heapLen {
			return
		}
		if right := child + 1; right < heapLen && sampleOrderLogitsLessNoSuppress(order[right], order[child], logits) {
			child = right
		}
		if !sampleOrderLogitsLessNoSuppress(order[child], order[root], logits) {
			return
		}
		order[root], order[child] = order[child], order[root]
		root = child
	}
}

func sampleOrderLogitsLessNoSuppress(a, b int32, logits []byte) bool {
	ai, bi := int(a)*bf16Size, int(b)*bf16Size
	av, bv := bf16ToF32(logits[ai], logits[ai+1]), bf16ToF32(logits[bi], logits[bi+1])
	return av > bv || (av == bv && a < b)
}

func sampleRankPrefixPreferred(params model.SampleParams, vocab int) bool {
	if params.TopK > 0 && params.TopK < vocab {
		return true
	}
	if params.TopP > 0 && params.TopP < 1 {
		return true
	}
	return params.MinP > 0
}

func rankSampleOrderPrefix(order []int32, probs []float32, probTotal float32, params model.SampleParams) int {
	if len(order) == 0 {
		return 0
	}
	if probTotal <= 0 {
		probTotal = 1
	}
	heapifySampleOrder(order, probs)
	heapLen := len(order)
	popped := 0
	keptMass := float32(0)
	if params.TopK > 0 && params.TopK < heapLen {
		for popped < params.TopK {
			id := popSampleOrderHeap(order, probs, heapLen)
			heapLen--
			popped++
			keptMass += probs[id]
		}
		reverseSampleOrderTailToPrefix(order, popped)
		keep := popped
		if params.TopP > 0 && params.TopP < 1 {
			keep = sampleOrderTopPKeep(order, probs, keep, params.TopP*keptMass)
		}
		return sampleOrderMinPKeep(order, probs, keep, params.MinP)
	}
	if params.TopP > 0 && params.TopP < 1 {
		target := params.TopP * probTotal
		for heapLen > 0 {
			id := popSampleOrderHeap(order, probs, heapLen)
			heapLen--
			popped++
			keptMass += probs[id]
			if keptMass >= target {
				break
			}
		}
		reverseSampleOrderTailToPrefix(order, popped)
		return sampleOrderMinPKeep(order, probs, popped, params.MinP)
	}
	if params.MinP > 0 {
		id := popSampleOrderHeap(order, probs, heapLen)
		heapLen--
		popped++
		threshold := probs[id] * params.MinP
		for heapLen > 0 && probs[order[0]] >= threshold {
			popSampleOrderHeap(order, probs, heapLen)
			heapLen--
			popped++
		}
		reverseSampleOrderTailToPrefix(order, popped)
		return popped
	}
	sortSampleOrderByProb(order, probs)
	return len(order)
}

func sampleOrderTopPKeep(order []int32, probs []float32, keep int, targetMass float32) int {
	var cum float32
	n := 0
	for n < keep {
		cum += probs[int(order[n])]
		n++
		if cum >= targetMass {
			break
		}
	}
	return n
}

func sampleOrderMinPKeep(order []int32, probs []float32, keep int, minP float32) int {
	if minP <= 0 || keep <= 0 {
		return keep
	}
	threshold := probs[int(order[0])] * minP
	n := 0
	for n < keep && probs[int(order[n])] >= threshold {
		n++
	}
	if n > 0 {
		return n
	}
	return keep
}

func heapifySampleOrder(order []int32, probs []float32) {
	for i := len(order)/2 - 1; i >= 0; i-- {
		siftSampleOrderHeap(order, probs, i, len(order))
	}
}

func popSampleOrderHeap(order []int32, probs []float32, heapLen int) int32 {
	top := order[0]
	last := heapLen - 1
	order[0] = order[last]
	order[last] = top
	siftSampleOrderHeap(order, probs, 0, last)
	return top
}

func siftSampleOrderHeap(order []int32, probs []float32, root, heapLen int) {
	for {
		child := root*2 + 1
		if child >= heapLen {
			return
		}
		if right := child + 1; right < heapLen && sampleOrderLess(order[right], order[child], probs) {
			child = right
		}
		if !sampleOrderLess(order[child], order[root], probs) {
			return
		}
		order[root], order[child] = order[child], order[root]
		root = child
	}
}

func reverseSampleOrderTailToPrefix(order []int32, n int) {
	start := len(order) - n
	for i, j := start, len(order)-1; i < j; i, j = i+1, j-1 {
		order[i], order[j] = order[j], order[i]
	}
	if start > 0 {
		copy(order[:n], order[start:])
	}
}

func sortSampleOrderByProb(order []int32, probs []float32) {
	if len(order) < 2 {
		return
	}
	sortSampleOrderByProbRange(order, probs, 0, len(order)-1)
}

func sortSampleOrderByProbRange(order []int32, probs []float32, lo, hi int) {
	for hi-lo > 12 {
		mid := lo + (hi-lo)/2
		if sampleOrderLess(order[mid], order[lo], probs) {
			order[mid], order[lo] = order[lo], order[mid]
		}
		if sampleOrderLess(order[hi], order[mid], probs) {
			order[hi], order[mid] = order[mid], order[hi]
			if sampleOrderLess(order[mid], order[lo], probs) {
				order[mid], order[lo] = order[lo], order[mid]
			}
		}
		pivot := order[mid]
		i, j := lo, hi
		for {
			for sampleOrderLess(order[i], pivot, probs) {
				i++
			}
			for sampleOrderLess(pivot, order[j], probs) {
				j--
			}
			if i >= j {
				break
			}
			order[i], order[j] = order[j], order[i]
			i++
			j--
		}
		if j-lo < hi-i {
			sortSampleOrderByProbRange(order, probs, lo, j)
			lo = i
		} else {
			sortSampleOrderByProbRange(order, probs, i, hi)
			hi = j
		}
	}
	for i := lo + 1; i <= hi; i++ {
		v := order[i]
		j := i - 1
		for j >= lo && sampleOrderLess(v, order[j], probs) {
			order[j+1] = order[j]
			j--
		}
		order[j+1] = v
	}
}

func sampleOrderLess(a, b int32, probs []float32) bool {
	pa, pb := probs[int(a)], probs[int(b)]
	return pa > pb || (pa == pb && a < b)
}

func (s *ArchSession) sampleTopKParamsEligible(params model.SampleParams) bool {
	if s.headEnc == nil {
		return false
	}
	if params.TopK <= 0 || params.TopK > headSampleTopKMaxK {
		return false
	}
	if params.RepeatPenalty > 1 {
		return false
	}
	return true
}

func (s *ArchSession) sampleTopKTokenParamsEligible(params model.SampleParams) bool {
	if s.headEnc == nil || params.Temperature <= 0 {
		return false
	}
	if params.TopK <= 0 || params.TopK > headSampleTopKMaxK {
		return false
	}
	return s.headEnc.topKSampleUsable(params.TopK)
}

func (s *ArchSession) sampleLogitsTokenParamsEligible(params model.SampleParams) bool {
	if s.headEnc == nil || params.Temperature <= 0 {
		return false
	}
	if params.TopK < 0 || params.TopK > headSampleTopKMaxK {
		return false
	}
	if params.TopK == 0 && params.TopP > 0 && params.TopP < 1 && !logitsSampleTopPOnlyFullVocab(params, s.arch.Vocab) {
		return false
	}
	return s.headEnc.logitsSampleUsable()
}

func (s *ArchSession) retainedLogitsSampleParamsEligible(params model.SampleParams) bool {
	if s.headEnc == nil || params.Temperature <= 0 {
		return false
	}
	if params.TopK < 0 || params.TopK > headSampleTopKMaxK {
		return false
	}
	if params.TopK == 0 && params.TopP > 0 && params.TopP < 1 && !logitsSampleTopPOnlyFullVocab(params, s.arch.Vocab) {
		return false
	}
	return s.headEnc.logitsBufferSampleUsable()
}

func sampleLogitsTokenCPUPreferred(params model.SampleParams, vocab int) bool {
	return params.TopK == 0 && params.TopP > 0 && params.TopP < 1 && params.RepeatPenalty <= 1 && vocab > 0 && vocab <= headSampleTopKMaxK
}

func logitsSampleTopPOnlyFullVocab(params model.SampleParams, vocab int) bool {
	return params.TopK == 0 && params.TopP > 0 && params.TopP < 1 && vocab > 0
}

func logitsSampleKernelTopK(params model.SampleParams, vocab int) int {
	if logitsSampleTopPOnlyFullVocab(params, vocab) {
		return vocab
	}
	return params.TopK
}

func sampledGreedyParamsEligible(params model.SampleParams) bool {
	return params.Temperature <= 0 && params.MinP <= 0 && params.RepeatPenalty <= 1
}

func sampledTopOneGreedyParamsEligible(params model.SampleParams, history []int32) bool {
	return params.TopK == 1 && !sampledGreedyParamsEligible(params) && (params.RepeatPenalty <= 1 || len(history) == 0)
}

// stepSampleTopKCandidatesInPool is the sampled sibling of stepGreedyInPool.
// For ICB sessions it decodes token id at the current cache row and runs the
// resident TopK head over the resulting hidden in the same command buffer. The
// host waits once, then reads this step's hidden plus only K candidate logits.
func (s *ArchSession) stepSampleTopKCandidatesInPool(id int32, params model.SampleParams) (hidden, logits []byte, ids []int32, ok bool, err error) {
	if s.state.icb == nil || icbDisabledForTest || !s.sampleTopKParamsEligible(params) {
		return nil, nil, nil, false, nil
	}
	if s.encNextInputsGPU != nil && s.plScratchNew != nil && !chainedGPUInputsDisabled {
		return s.stepSampleTopKCandidatesGPUInputsInPool(id, params)
	}
	emb, err := s.embedID(id)
	if err != nil {
		return nil, nil, nil, false, err
	}
	var pli []byte
	if s.perLayerInput != nil {
		pli, err = s.perLayerInput(id, emb)
		if err != nil {
			return nil, nil, nil, false, err
		}
		s.state.perLayerInput = pli
	}
	icb := s.state.icb
	var scratch *headTopKScratch
	withAutoreleasePool(func() {
		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		lastOut := icb.encodeStepBody(enc, emb, s.pos, pli)
		scratch, ok, err = s.headEnc.encodeTopKCandidates(enc, lastOut, params.TopK, params.SuppressTokens, false)
		if !ok || err != nil {
			endEncodingFast(enc)
			if scratch != nil {
				s.headEnc.putTopKScratch(scratch)
				scratch = nil
			}
			return
		}
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		hidden = s.retainHiddenReadbackFrom(icb.lastOutPtr)
		var readOK bool
		logits, ids, readOK, err = s.headEnc.readTopKCandidatesInto(scratch, params.TopK, s.sampleCandidateLogits, s.sampleCandidateIDs)
		s.sampleCandidateLogits, s.sampleCandidateIDs = logits, ids
		s.headEnc.putTopKScratch(scratch)
		scratch = nil
		ok = readOK
	})
	if err != nil || !ok {
		return nil, nil, nil, ok, err
	}
	s.pos++
	return hidden, logits, ids, true, nil
}

func (s *ArchSession) stepSampleTopKCandidatesGPUInputsInPool(id int32, params model.SampleParams) (hidden, logits []byte, ids []int32, ok bool, err error) {
	icb := s.state.icb
	if icb == nil || s.encNextInputsGPU == nil || s.plScratchNew == nil {
		return nil, nil, nil, false, nil
	}
	var scratch *headTopKScratch
	withAutoreleasePool(func() {
		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		var lastOut metal.MTLBuffer
		lastOut, err = s.encodeStepBodyFromGPUInputsInPool(enc, id)
		if err != nil {
			endEncodingFast(enc)
			return
		}
		scratch, ok, err = s.headEnc.encodeTopKCandidates(enc, lastOut, params.TopK, params.SuppressTokens, false)
		if !ok || err != nil {
			endEncodingFast(enc)
			if scratch != nil {
				s.headEnc.putTopKScratch(scratch)
				scratch = nil
			}
			return
		}
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		hidden = s.retainHiddenReadbackFrom(icb.lastOutPtr)
		var readOK bool
		logits, ids, readOK, err = s.headEnc.readTopKCandidatesInto(scratch, params.TopK, s.sampleCandidateLogits, s.sampleCandidateIDs)
		s.sampleCandidateLogits, s.sampleCandidateIDs = logits, ids
		s.headEnc.putTopKScratch(scratch)
		scratch = nil
		ok = readOK
	})
	if err != nil || !ok {
		return nil, nil, nil, ok, err
	}
	s.pos++
	return hidden, logits, ids, true, nil
}

func (s *ArchSession) stepSampleTopKTokenInPool(id int32, params model.SampleParams, draw float32, history []int32) (hidden []byte, token int32, ok bool, err error) {
	if s.state.icb == nil || icbDisabledForTest || !s.sampleTopKTokenParamsEligible(params) {
		return nil, 0, false, nil
	}
	if s.encNextInputsGPU != nil && s.plScratchNew != nil && !chainedGPUInputsDisabled {
		return s.stepSampleTopKTokenGPUInputsInPool(id, params, draw, history)
	}
	emb, err := s.embedID(id)
	if err != nil {
		return nil, 0, false, err
	}
	var pli []byte
	if s.perLayerInput != nil {
		pli, err = s.perLayerInput(id, emb)
		if err != nil {
			return nil, 0, false, err
		}
		s.state.perLayerInput = pli
	}
	icb := s.state.icb
	var scratch *headTopKScratch
	withAutoreleasePool(func() {
		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		lastOut := icb.encodeStepBody(enc, emb, s.pos, pli)
		scratch, ok, err = s.headEnc.encodeTopKSample(enc, lastOut, params, draw, history, false)
		if !ok || err != nil {
			endEncodingFast(enc)
			if scratch != nil {
				s.headEnc.putTopKScratch(scratch)
				scratch = nil
			}
			return
		}
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		hidden = s.retainHiddenReadbackFrom(icb.lastOutPtr)
		token = scratch.token()
		s.headEnc.putTopKScratch(scratch)
		scratch = nil
	})
	if err != nil || !ok {
		return nil, 0, ok, err
	}
	if token < 0 || int(token) >= s.arch.Vocab {
		return nil, 0, true, core.NewError(core.Sprintf("native.ArchSession.stepSampleTopKTokenInPool: sampled invalid token %d for vocab %d", token, s.arch.Vocab))
	}
	s.pos++
	return hidden, token, true, nil
}

func (s *ArchSession) encodeStepBodyFromGPUInputsInPool(enc metal.MTLComputeCommandEncoder, id int32) (metal.MTLBuffer, error) {
	icb := s.state.icb
	if icb == nil || s.encNextInputsGPU == nil || s.plScratchNew == nil {
		return nil, core.NewError("native.ArchSession.encodeStepBodyFromGPUInputsInPool: GPU inputs unavailable")
	}
	sc := s.gpuTailPLScratchBuffer(0)
	sc.out = icb.pleInput
	tokBuf := s.nextInputTokenBuffer(id)
	if err := s.encNextInputsGPU(enc, tokBuf, icb.ping0, sc); err != nil {
		return nil, err
	}
	enc.MemoryBarrierWithScope(metal.MTLBarrierScopeBuffers)
	return icb.encodeStepBodyNoInput(enc, s.pos), nil
}

func (s *ArchSession) stepSampleTopKTokenGPUInputsInPool(id int32, params model.SampleParams, draw float32, history []int32) (hidden []byte, token int32, ok bool, err error) {
	icb := s.state.icb
	if icb == nil || s.encNextInputsGPU == nil || s.plScratchNew == nil {
		return nil, 0, false, nil
	}
	var scratch *headTopKScratch
	token = -1
	withAutoreleasePool(func() {
		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		var lastOut metal.MTLBuffer
		lastOut, err = s.encodeStepBodyFromGPUInputsInPool(enc, id)
		if err != nil {
			endEncodingFast(enc)
			return
		}
		scratch, ok, err = s.headEnc.encodeTopKSample(enc, lastOut, params, draw, history, false)
		if !ok || err != nil {
			endEncodingFast(enc)
			if scratch != nil {
				s.headEnc.putTopKScratch(scratch)
				scratch = nil
			}
			return
		}
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		hidden = s.retainHiddenReadbackFrom(icb.lastOutPtr)
		token = scratch.token()
		s.headEnc.putTopKScratch(scratch)
		scratch = nil
	})
	if err != nil || !ok {
		return nil, 0, ok, err
	}
	if token < 0 || int(token) >= s.arch.Vocab {
		return nil, 0, true, core.NewError(core.Sprintf("native.ArchSession.stepSampleTopKTokenGPUInputsInPool: sampled invalid token %d for vocab %d", token, s.arch.Vocab))
	}
	s.pos++
	return hidden, token, true, nil
}

func (s *ArchSession) stepSampleLogitsTokenInPool(id int32, params model.SampleParams, draw float32, history []int32) (hidden []byte, token int32, ok bool, err error) {
	if s.state.icb == nil || icbDisabledForTest || !s.sampleLogitsTokenParamsEligible(params) {
		return nil, 0, false, nil
	}
	if s.encNextInputsGPU != nil && s.plScratchNew != nil && !chainedGPUInputsDisabled {
		return s.stepSampleLogitsTokenGPUInputsInPool(id, params, draw, history)
	}
	emb, err := s.embedID(id)
	if err != nil {
		return nil, 0, false, err
	}
	var pli []byte
	if s.perLayerInput != nil {
		pli, err = s.perLayerInput(id, emb)
		if err != nil {
			return nil, 0, false, err
		}
		s.state.perLayerInput = pli
	}
	icb := s.state.icb
	var scratch *headGreedyScratch
	withAutoreleasePool(func() {
		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		lastOut := icb.encodeStepBody(enc, emb, s.pos, pli)
		scratch, ok, err = s.headEnc.encodeLogitsSample(enc, lastOut, params, draw, history)
		if !ok || err != nil {
			endEncodingFast(enc)
			if scratch != nil {
				s.headEnc.putGreedyScratch(scratch)
				scratch = nil
			}
			return
		}
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		hidden = s.retainHiddenReadbackFrom(icb.lastOutPtr)
		token = scratch.token()
		s.headEnc.putGreedyScratch(scratch)
		scratch = nil
	})
	if err != nil || !ok {
		return nil, 0, ok, err
	}
	if token < 0 || int(token) >= s.arch.Vocab {
		return nil, 0, true, core.NewError(core.Sprintf("native.ArchSession.stepSampleLogitsTokenInPool: sampled invalid token %d for vocab %d", token, s.arch.Vocab))
	}
	s.pos++
	return hidden, token, true, nil
}

func (s *ArchSession) stepSampleLogitsTokenGPUInputsInPool(id int32, params model.SampleParams, draw float32, history []int32) (hidden []byte, token int32, ok bool, err error) {
	icb := s.state.icb
	if icb == nil || s.encNextInputsGPU == nil || s.plScratchNew == nil {
		return nil, 0, false, nil
	}
	var scratch *headGreedyScratch
	token = -1
	withAutoreleasePool(func() {
		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		var lastOut metal.MTLBuffer
		lastOut, err = s.encodeStepBodyFromGPUInputsInPool(enc, id)
		if err != nil {
			endEncodingFast(enc)
			return
		}
		scratch, ok, err = s.headEnc.encodeLogitsSample(enc, lastOut, params, draw, history)
		if !ok || err != nil {
			endEncodingFast(enc)
			if scratch != nil {
				s.headEnc.putGreedyScratch(scratch)
				scratch = nil
			}
			return
		}
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		hidden = s.retainHiddenReadbackFrom(icb.lastOutPtr)
		token = scratch.token()
		s.headEnc.putGreedyScratch(scratch)
		scratch = nil
	})
	if err != nil || !ok {
		return nil, 0, ok, err
	}
	if token < 0 || int(token) >= s.arch.Vocab {
		return nil, 0, true, core.NewError(core.Sprintf("native.ArchSession.stepSampleLogitsTokenGPUInputsInPool: sampled invalid token %d for vocab %d", token, s.arch.Vocab))
	}
	s.pos++
	return hidden, token, true, nil
}

func (s *ArchSession) stepGreedyInPool(id int32, emb []byte, suppress []int32) (token int32, hidden []byte, ok bool, err error) {
	if s.state.icb == nil || icbDisabledForTest || s.headEnc == nil {
		return 0, nil, false, nil
	}
	if emb == nil {
		emb, err = s.embedID(id)
		if err != nil {
			return 0, nil, false, err
		}
	}
	icb := s.state.icb
	var pli []byte
	if s.perLayerInput != nil {
		pli, err = s.perLayerInput(id, emb)
		if err != nil {
			return 0, nil, false, err
		}
		s.state.perLayerInput = pli
	}
	token = -1
	withAutoreleasePool(func() {
		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		lastOut := icb.encodeStepBody(enc, emb, s.pos, pli)
		scratch, gok, gerr := s.headEnc.encodeGreedy(enc, lastOut, suppress)
		if !gok || gerr != nil {
			endEncodingFast(enc)
			if scratch != nil {
				s.headEnc.putGreedyScratch(scratch)
			}
			ok, err = gok, gerr
			return
		}
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		token = scratch.token()
		hidden = s.retainHiddenReadbackFrom(icb.lastOutPtr)
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
		next, ok, err := s.directGreedyFromHiddenInPool(hidden, suppress)
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
		// cacheFirstLogits retains this slice for prompt replay, so keep that path on
		// the owned logits backing. Other greedy fallback calls consume logits
		// immediately and can reuse the session scratch.
		if isFirst && cacheFirstLogits != nil {
			logits, err = s.head(hidden, true) // greedy: argmax — skip the monotonic softcap (token-identical)
		} else if s.canUseHeadLogitsScratch() {
			logits, err = s.headLogitsScratch(hidden, true)
		} else {
			logits, err = s.head(hidden, true)
		}
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
		if pipelinedGPUDecodeEnabled && s.recordPeerICB != nil {
			return s.generatePipelinedGPUTail(gen, maxNew, eosID, suppress, yield, stop)
		}
		return s.generateChainedGPUTail(gen, maxNew, eosID, suppress, yield, stop)
	}

	for !stop && len(gen) < maxNew {
		prev := gen[len(gen)-1]
		emb, eerr := s.embedID(prev)
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
		if hidden, err = s.stepIDRetainedInPool(prev); err != nil {
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
	if hidden, err = s.stepIDRetainedInPool(gen[len(gen)-1]); err != nil {
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
	sc := s.gpuTailPLScratchBuffer(0)
	sc.out = icb.pleInput // the PLE result lands directly in the ICB's pli input for the next step
	var rerr error
	withAutoreleasePool(func() {
		// Seed: produce emb(gen[last])/pli(gen[last]) into ping0/pleInput from the first token.
		tokBuf := s.nextInputTokenBuffer(gen[len(gen)-1])
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
			scratch, gok, gerr := s.headEnc.encodeGreedy(enc, lastOut, suppress)
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
			s.encNextInputsGPU(enc, scratch.outToken, icb.ping0, sc)
			enc.EndEncoding()
			cb.Commit()
			cb.WaitUntilCompleted()
			if pieceTimingOn {
				chainedGPUSpanNs += int64(float64(cb.GPUEndTime()-cb.GPUStartTime()) * 1e9)
			}
			tk := scratch.token()
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
		icb.encodeStepBodyNoInput(enc, s.pos)
		enc.EndEncoding()
		cb.Commit()
		cb.WaitUntilCompleted()
		s.pos++
		s.rememberRetainedHiddenFrom(icb.lastOutPtr)
	})
	if rerr != nil {
		return nil, rerr
	}
	return gen, nil
}

// peerICB lazily records (once) the second ICB sharing this session's KV caches — its own ping0/pleInput,
// the same KV — for the submit-ahead decode's double buffer.
func (s *ArchSession) peerICB() (*archICBReplay, error) {
	if s.icbPeer != nil {
		return s.icbPeer, nil
	}
	if s.recordPeerICB == nil {
		return nil, core.NewError("native.ArchSession.peerICB: no peer recorder")
	}
	rep, err := s.recordPeerICB()
	if err != nil {
		return nil, err
	}
	s.icbPeer = rep
	return rep, nil
}

// generatePipelinedGPUTail is the submit-ahead form of generateChainedGPUTail: two ICBs (A/B) over the
// SAME KV caches, each with its own ping0/pleInput. Each step's cb writes the NEXT step's emb+pli into the
// OTHER ICB, so the host submits step t+1 before reading t's token — one command buffer always in flight
// ahead, the GPU serialising them through the shared KV. 1-ahead is discard-safe for greedy: each cb
// caches the token it reads (advancing pos by one per submit, so cached-count == pos), and the trailing
// speculative cb's produced token is dropped past eos/maxNew. Cache/pos byte-identical to the serial loop.
func (s *ArchSession) generatePipelinedGPUTail(gen []int32, maxNew, eosID int, suppress []int32, yield func(int32) bool, stop bool) ([]int32, error) {
	icbB, err := s.peerICB()
	if err != nil {
		return nil, err
	}
	icbs := [2]*archICBReplay{s.state.icb, icbB}
	sc := [2]*plGPUScratch{s.gpuTailPLScratchBuffer(0), s.gpuTailPLScratchBuffer(1)}
	type infl struct {
		cb      metal.MTLCommandBuffer
		lastOut *byte
		scratch *headGreedyScratch
	}
	var rerr error
	withAutoreleasePool(func() {
		// Seed icbA's inputs from the first token.
		tokBuf := s.nextInputTokenBuffer(gen[len(gen)-1])
		sc[0].out = icbs[0].pleInput
		seedCB := queue.CommandBuffer()
		seedEnc := seedCB.ComputeCommandEncoder()
		if e := s.encNextInputsGPU(seedEnc, tokBuf, icbs[0].ping0, sc[0]); e != nil {
			seedEnc.EndEncoding()
			rerr = e
			return
		}
		seedEnc.EndEncoding()
		seedCB.Commit()
		seedCB.WaitUntilCompleted()

		// submit encodes+commits one step on ICB i, writing the next step's emb+pli into ICB 1-i (no wait).
		submit := func(i int) (infl, bool) {
			icb, tgt := icbs[i], icbs[1-i]
			sc[i].out = tgt.pleInput
			cb := queue.CommandBuffer()
			enc := cb.ComputeCommandEncoder()
			lastOut := icb.encodeStepBodyNoInput(enc, s.pos)
			scratch, gok, gerr := s.headEnc.encodeGreedy(enc, lastOut, suppress)
			if !gok || gerr != nil {
				enc.EndEncoding()
				if scratch != nil {
					s.headEnc.putGreedyScratch(scratch)
				}
				if rerr = gerr; rerr == nil {
					rerr = core.NewError("native.ArchSession.generatePipelinedGPUTail: GPU head argmax unavailable mid-chain")
				}
				return infl{}, false
			}
			s.encNextInputsGPU(enc, scratch.outToken, tgt.ping0, sc[i])
			enc.EndEncoding()
			cb.Commit()
			s.pos++
			return infl{cb: cb, lastOut: icb.lastOutPtr, scratch: scratch}, true
		}

		read := func(p infl) (int32, bool) {
			p.cb.WaitUntilCompleted()
			if pieceTimingOn {
				chainedGPUSpanNs += int64(float64(p.cb.GPUEndTime()-p.cb.GPUStartTime()) * 1e9)
			}
			tk := p.scratch.token()
			s.headEnc.putGreedyScratch(p.scratch)
			if tk < 0 || int(tk) >= s.arch.Vocab {
				rerr = core.NewError("native.ArchSession.generatePipelinedGPUTail: invalid token")
				return 0, false
			}
			return tk, true
		}

		prev, ok := submit(0)
		if !ok {
			return
		}
		i := 1
		for len(gen) < maxNew && !stop {
			nxt, ok := submit(i)
			if !ok {
				prev.cb.WaitUntilCompleted()
				s.headEnc.putGreedyScratch(prev.scratch)
				return
			}
			i = 1 - i
			tk, valid := read(prev)
			if !valid {
				nxt.cb.WaitUntilCompleted()
				s.headEnc.putGreedyScratch(nxt.scratch)
				return
			}
			gen = append(gen, tk)
			stop = (yield != nil && !yield(tk)) || (eosID >= 0 && int(tk) == eosID)
			prev = nxt
		}
		// Drain the trailing in-flight cb. Its produced token is appended only if still within budget
		// (it was a needed token), else dropped (speculation past eos/maxNew). Either way its stepBody
		// cached the last appended token — so retain its hidden as the session boundary.
		tk, valid := read(prev)
		if valid && !stop && len(gen) < maxNew {
			gen = append(gen, tk)
		}
		s.rememberRetainedHiddenFrom(prev.lastOut)
	})
	if rerr != nil {
		return nil, rerr
	}
	return gen, nil
}

func (s *ArchSession) greedyFromHiddenInPool(hidden []byte, suppress []int32) (int32, error) {
	if s.greedy != nil {
		_ptHead := ptStart()
		next, ok, err := s.directGreedyFromHiddenInPool(hidden, suppress)
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

// GenerateSampledEach is native's sampled retained-session path: it keeps the
// transformer stack on the ArchSession replay path, materialises full vocab
// logits for the host sampler, then commits every sampled token into the
// resident cache. This is the sampled sibling of GenerateEach for serve paths
// that cannot use direct on-GPU greedy argmax.
func (s *ArchSession) GenerateSampledEach(promptIDs []int32, maxNew int, stopTokens []int32, sampler *model.Sampler, params model.SampleParams, transform model.TokenTransform, yield func(int32) bool) ([]int32, error) {
	if sampler == nil {
		return nil, core.NewError("native.ArchSession.GenerateSampledEach: nil sampler")
	}
	if len(promptIDs) == 0 {
		return nil, core.NewError("native.ArchSession.GenerateSampledEach: empty prompt")
	}
	if maxNew <= 0 {
		return nil, core.NewError("native.ArchSession.GenerateSampledEach: maxNew must be > 0")
	}
	if s.pos+len(promptIDs)+maxNew > s.maxLen {
		return nil, core.NewError("native.ArchSession.GenerateSampledEach: sequence would exceed maxLen cache rows")
	}
	var gen []int32
	var genErr error
	withAutoreleasePool(func() {
		hidden, err := s.prefillPromptRetainedInPool(promptIDs)
		if err != nil {
			genErr = err
			return
		}
		gen, genErr = s.generateSampledFromHiddenInPool(hidden, maxNew, stopTokens, sampler, params, transform, yield, true)
	})
	return gen, genErr
}

// GenerateSampledOneShotEach is the serve/request sibling of GenerateSampledEach:
// it streams sampled tokens through the native session but does not cache the
// final generated token because the fresh request session is about to be
// dropped. That mirrors GenerateOneShot's greedy final-step saving.
func (s *ArchSession) GenerateSampledOneShotEach(promptIDs []int32, maxNew int, stopTokens []int32, sampler *model.Sampler, params model.SampleParams, transform model.TokenTransform, yield func(int32) bool) ([]int32, error) {
	if sampler == nil {
		return nil, core.NewError("native.ArchSession.GenerateSampledOneShotEach: nil sampler")
	}
	if len(promptIDs) == 0 {
		return nil, core.NewError("native.ArchSession.GenerateSampledOneShotEach: empty prompt")
	}
	if maxNew <= 0 {
		return nil, core.NewError("native.ArchSession.GenerateSampledOneShotEach: maxNew must be > 0")
	}
	if s.pos+len(promptIDs)+maxNew > s.maxLen {
		return nil, core.NewError("native.ArchSession.GenerateSampledOneShotEach: sequence would exceed maxLen cache rows")
	}
	var gen []int32
	var genErr error
	withAutoreleasePool(func() {
		hidden, err := s.prefillPromptRetainedInPool(promptIDs)
		if err != nil {
			genErr = err
			return
		}
		gen, genErr = s.generateSampledFromHiddenInPool(hidden, maxNew, stopTokens, sampler, params, transform, yield, false)
	})
	return gen, genErr
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
		hidden, err := s.prefillPromptRetainedInPool(promptIDs)
		if err != nil {
			genErr = err
			return
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

func (s *ArchSession) generateSampledFromHiddenInPool(hidden []byte, maxNew int, stopTokens []int32, sampler *model.Sampler, params model.SampleParams, transform model.TokenTransform, yield func(int32) bool, cacheFinal bool) ([]int32, error) {
	history := s.sampleHistoryScratchFor(params, maxNew)
	finalHistory := history
	defer func() { s.sampleHistory = finalHistory }()
	gen, finalHistory, err := s.generateSampledFromHiddenInPoolWithHistory(hidden, maxNew, stopTokens, sampler, params, transform, yield, cacheFinal, 0, history)
	return gen, err
}

func (s *ArchSession) generateSampledFromHiddenInPoolWithHistory(hidden []byte, maxNew int, stopTokens []int32, sampler *model.Sampler, params model.SampleParams, transform model.TokenTransform, yield func(int32) bool, cacheFinal bool, initialGenerated int, history []int32) ([]int32, []int32, error) {
	gen := make([]int32, 0, maxNew)
	var readyLogits []byte
	var readyIDs []int32
	var readyToken int32
	readyTokenOK := false
	for len(gen) < maxNew {
		pickParams := params
		if params.MinTokensBeforeStop > 0 && initialGenerated+len(gen) < params.MinTokensBeforeStop {
			pickParams.SuppressTokens = s.suppressionTokensScratch(params.SuppressTokens, stopTokens)
		}
		var next int32
		var err error
		if sampledGreedyParamsEligible(pickParams) {
			next, err = s.headGreedyOrLogits(hidden, pickParams.SuppressTokens, nil, nil, false)
			readyLogits, readyIDs = nil, nil
			readyTokenOK = false
		} else if readyTokenOK {
			next = readyToken
			readyTokenOK = false
		} else if readyIDs != nil {
			next, err = sampler.SampleCandidates(readyLogits, readyIDs, pickParams)
			readyLogits, readyIDs = nil, nil
		} else if sampledTopOneGreedyParamsEligible(pickParams, history) {
			sampler.Draw()
			next, err = s.headGreedyOrLogits(hidden, pickParams.SuppressTokens, nil, nil, false)
			readyLogits, readyIDs = nil, nil
			readyTokenOK = false
		} else if s.sampleTopKTokenParamsEligible(pickParams) {
			draw := sampler.Draw()
			var ok bool
			next, ok, err = s.sampleTopKTokenFromHiddenInPool(hidden, pickParams, draw, history)
			if !ok && err == nil {
				err = core.NewError("native.ArchSession.generateSampledFromHiddenInPool: TopK token path declined after eligibility check")
			}
		} else if s.sampleLogitsTokenParamsEligible(pickParams) && !sampleLogitsTokenCPUPreferred(pickParams, s.arch.Vocab) {
			draw := sampler.Draw()
			var ok bool
			next, ok, err = s.sampleLogitsTokenFromHiddenInPool(hidden, pickParams, draw, history)
			if !ok && err == nil {
				err = core.NewError("native.ArchSession.generateSampledFromHiddenInPool: logits token path declined after eligibility check")
			}
		} else if candidateLogits, candidateIDs, ok, topKErr := s.sampleTopKCandidatesFromHiddenInPool(hidden, pickParams); topKErr != nil {
			return nil, history, topKErr
		} else if ok {
			next, err = sampler.SampleCandidates(candidateLogits, candidateIDs, pickParams)
		} else {
			logits, headErr := s.headLogitsScratch(hidden, false)
			if headErr != nil {
				return nil, history, headErr
			}
			pickLogits := logits
			if params.RepeatPenalty > 1 {
				pickLogits, err = s.repeatPenaltyLogitsScratch(logits, s.arch.Vocab, history, params.RepeatPenalty)
				if err != nil {
					return nil, history, err
				}
			}
			if sampleLogitsTokenCPUPreferred(pickParams, s.arch.Vocab) {
				next, err = sampleSmallVocabBF16(pickLogits, s.arch.Vocab, sampler, pickParams)
			} else {
				next, err = s.sampleVocabBF16(pickLogits, s.arch.Vocab, sampler, pickParams)
			}
		}
		if err != nil {
			return nil, history, err
		}
		if transform != nil {
			next = transform(next)
		}
		gen = append(gen, next)
		if params.RepeatPenalty > 1 {
			history = append(history, next)
		}
		stop := (yield != nil && !yield(next)) || nativeTokenInSet(next, stopTokens)
		if !cacheFinal && (stop || len(gen) >= maxNew) {
			break
		}
		nextPickParams := params
		if params.MinTokensBeforeStop > 0 && initialGenerated+len(gen) < params.MinTokensBeforeStop {
			nextPickParams.SuppressTokens = s.suppressionTokensScratch(params.SuppressTokens, stopTokens)
		}
		if !stop && len(gen) < maxNew && s.sampledChainedGPUTailCanContinue(nextPickParams, history, transform) {
			return s.generateSampledChainedGPUTail(gen, maxNew, stopTokens, sampler, params, yield, cacheFinal, initialGenerated, history)
		}
		stepped := false
		if !sampledGreedyParamsEligible(nextPickParams) {
			if sampledTopOneGreedyParamsEligible(nextPickParams, history) && s.state.icb != nil && !icbDisabledForTest && s.headEnc != nil && s.greedy != nil {
				sampler.Draw()
				if chainedToken, chainedHidden, ok, chainErr := s.stepGreedyInPool(next, nil, nextPickParams.SuppressTokens); chainErr != nil {
					return nil, history, chainErr
				} else if ok {
					hidden, readyToken, readyTokenOK = chainedHidden, chainedToken, true
					readyLogits, readyIDs = nil, nil
					stepped = true
				}
			} else if s.sampleTopKTokenParamsEligible(nextPickParams) {
				draw := sampler.Draw()
				if chainedHidden, chainedToken, ok, chainErr := s.stepSampleTopKTokenInPool(next, nextPickParams, draw, history); chainErr != nil {
					return nil, history, chainErr
				} else if ok {
					hidden, readyToken, readyTokenOK = chainedHidden, chainedToken, true
					readyLogits, readyIDs = nil, nil
					stepped = true
				}
			} else if s.sampleLogitsTokenParamsEligible(nextPickParams) {
				draw := sampler.Draw()
				if chainedHidden, chainedToken, ok, chainErr := s.stepSampleLogitsTokenInPool(next, nextPickParams, draw, history); chainErr != nil {
					return nil, history, chainErr
				} else if ok {
					hidden, readyToken, readyTokenOK = chainedHidden, chainedToken, true
					readyLogits, readyIDs = nil, nil
					stepped = true
				}
			}
		}
		if !stepped && !sampledGreedyParamsEligible(nextPickParams) {
			if chainedHidden, chainedLogits, chainedIDs, ok, chainErr := s.stepSampleTopKCandidatesInPool(next, nextPickParams); chainErr != nil {
				return nil, history, chainErr
			} else if ok {
				hidden, readyLogits, readyIDs = chainedHidden, chainedLogits, chainedIDs
				readyTokenOK = false
				stepped = true
			}
		}
		if !stepped {
			hidden, err = s.stepIDRetainedInPool(next)
			if err != nil {
				return nil, history, err
			}
		}
		s.rememberRetainedHidden(hidden)
		if stop {
			break
		}
	}
	return gen, history, nil
}

func (s *ArchSession) sampledChainedGPUTailCanContinue(params model.SampleParams, history []int32, transform model.TokenTransform) bool {
	if transform != nil || chainedGPUInputsDisabled || icbDisabledForTest {
		return false
	}
	if s == nil || s.state.icb == nil || s.encNextInputsGPU == nil || s.plScratchNew == nil || s.headEnc == nil {
		return false
	}
	if sampledGreedyParamsEligible(params) || sampledTopOneGreedyParamsEligible(params, history) {
		return false
	}
	if s.sampleTopKTokenParamsEligible(params) {
		return true
	}
	return s.sampleLogitsTokenParamsEligible(params) && !sampleLogitsTokenCPUPreferred(params, s.arch.Vocab)
}

func (s *ArchSession) sampledPipelinedGPUTailCanContinue(params model.SampleParams, history []int32, transform model.TokenTransform) bool {
	return pipelinedGPUDecodeEnabled &&
		params.RepeatPenalty <= 1 &&
		s != nil &&
		s.recordPeerICB != nil &&
		s.sampledChainedGPUTailCanContinue(params, history, transform)
}

func (s *ArchSession) generateSampledChainedGPUTail(gen []int32, maxNew int, stopTokens []int32, sampler *model.Sampler, params model.SampleParams, yield func(int32) bool, cacheFinal bool, initialGenerated int, history []int32) ([]int32, []int32, error) {
	if cacheFinal && s.sampledPipelinedGPUTailCanContinue(params, history, nil) {
		return s.generateSampledPipelinedGPUTail(gen, maxNew, stopTokens, sampler, params, yield, initialGenerated, history)
	}
	icb := s.state.icb
	sc := s.gpuTailPLScratchBuffer(0)
	sc.out = icb.pleInput
	if len(gen) == 0 {
		return gen, history, core.NewError("native.ArchSession.generateSampledChainedGPUTail: empty generation seed")
	}
	tokBuf := s.nextInputTokenBuffer(gen[len(gen)-1])
	seedCB := commandBufferFast(queue)
	seedEnc := computeCommandEncoderFast(seedCB)
	if err := s.encNextInputsGPU(seedEnc, tokBuf, icb.ping0, sc); err != nil {
		endEncodingFast(seedEnc)
		return gen, history, err
	}
	endEncodingFast(seedEnc)
	commitCommandBufferFast(seedCB)
	waitUntilCompletedFast(seedCB)

	for len(gen) < maxNew {
		pickParams := params
		if params.MinTokensBeforeStop > 0 && initialGenerated+len(gen) < params.MinTokensBeforeStop {
			pickParams.SuppressTokens = s.suppressionTokensScratch(params.SuppressTokens, stopTokens)
		}
		if !s.sampledChainedGPUTailCanContinue(pickParams, history, nil) {
			break
		}
		draw := sampler.Draw()
		var token int32
		var ok bool
		var stepErr error
		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		lastOut := icb.encodeStepBodyNoInput(enc, s.pos)
		if s.sampleTopKTokenParamsEligible(pickParams) {
			var scratch *headTopKScratch
			scratch, ok, stepErr = s.headEnc.encodeTopKSample(enc, lastOut, pickParams, draw, history, false)
			if !ok || stepErr != nil {
				endEncodingFast(enc)
				if scratch != nil {
					s.headEnc.putTopKScratch(scratch)
				}
				if stepErr == nil {
					stepErr = core.NewError("native.ArchSession.generateSampledChainedGPUTail: TopK token path declined mid-chain")
				}
				return gen, history, stepErr
			}
			stepErr = s.encNextInputsGPU(enc, scratch.outToken, icb.ping0, sc)
			endEncodingFast(enc)
			if stepErr != nil {
				s.headEnc.putTopKScratch(scratch)
				return gen, history, stepErr
			}
			commitCommandBufferFast(cb)
			waitUntilCompletedFast(cb)
			token = scratch.token()
			s.headEnc.putTopKScratch(scratch)
		} else {
			var scratch *headGreedyScratch
			scratch, ok, stepErr = s.headEnc.encodeLogitsSample(enc, lastOut, pickParams, draw, history)
			if !ok || stepErr != nil {
				endEncodingFast(enc)
				if scratch != nil {
					s.headEnc.putGreedyScratch(scratch)
				}
				if stepErr == nil {
					stepErr = core.NewError("native.ArchSession.generateSampledChainedGPUTail: logits token path declined mid-chain")
				}
				return gen, history, stepErr
			}
			stepErr = s.encNextInputsGPU(enc, scratch.outToken, icb.ping0, sc)
			endEncodingFast(enc)
			if stepErr != nil {
				s.headEnc.putGreedyScratch(scratch)
				return gen, history, stepErr
			}
			commitCommandBufferFast(cb)
			waitUntilCompletedFast(cb)
			token = scratch.token()
			s.headEnc.putGreedyScratch(scratch)
		}
		s.pos++
		if token < 0 || int(token) >= s.arch.Vocab {
			return gen, history, core.NewError("native.ArchSession.generateSampledChainedGPUTail: sampled invalid token")
		}
		s.rememberRetainedHiddenFrom(icb.lastOutPtr)
		gen = append(gen, token)
		if params.RepeatPenalty > 1 {
			history = append(history, token)
		}
		stop := (yield != nil && !yield(token)) || nativeTokenInSet(token, stopTokens)
		if !cacheFinal && (stop || len(gen) >= maxNew) {
			return gen, history, nil
		}
		if stop {
			break
		}
	}
	if cacheFinal && len(gen) > 0 {
		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		icb.encodeStepBodyNoInput(enc, s.pos)
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		waitUntilCompletedFast(cb)
		s.pos++
		s.rememberRetainedHiddenFrom(icb.lastOutPtr)
	}
	return gen, history, nil
}

func (s *ArchSession) generateSampledPipelinedGPUTail(gen []int32, maxNew int, stopTokens []int32, sampler *model.Sampler, params model.SampleParams, yield func(int32) bool, initialGenerated int, history []int32) ([]int32, []int32, error) {
	if len(gen) == 0 {
		return gen, history, core.NewError("native.ArchSession.generateSampledPipelinedGPUTail: empty generation seed")
	}
	icbB, err := s.peerICB()
	if err != nil {
		return gen, history, err
	}
	icbs := [2]*archICBReplay{s.state.icb, icbB}
	sc := [2]*plGPUScratch{s.gpuTailPLScratchBuffer(0), s.gpuTailPLScratchBuffer(1)}

	type inflightSampledStep struct {
		cb      metal.MTLCommandBuffer
		lastOut *byte
		topK    *headTopKScratch
		logits  *headGreedyScratch
	}
	var rerr error

	release := func(p inflightSampledStep) {
		if p.topK != nil {
			s.headEnc.putTopKScratch(p.topK)
		}
		if p.logits != nil {
			s.headEnc.putGreedyScratch(p.logits)
		}
	}

	read := func(p inflightSampledStep) (int32, bool) {
		p.cb.WaitUntilCompleted()
		if pieceTimingOn {
			chainedGPUSpanNs += int64(float64(p.cb.GPUEndTime()-p.cb.GPUStartTime()) * 1e9)
		}
		var token int32
		switch {
		case p.topK != nil:
			token = p.topK.token()
		case p.logits != nil:
			token = p.logits.token()
		default:
			rerr = core.NewError("native.ArchSession.generateSampledPipelinedGPUTail: missing sampled scratch")
			return 0, false
		}
		release(p)
		if token < 0 || int(token) >= s.arch.Vocab {
			rerr = core.NewError("native.ArchSession.generateSampledPipelinedGPUTail: sampled invalid token")
			return 0, false
		}
		return token, true
	}

	submit := func(i, generatedBefore int) (inflightSampledStep, bool) {
		pickParams := params
		if params.MinTokensBeforeStop > 0 && initialGenerated+generatedBefore < params.MinTokensBeforeStop {
			pickParams.SuppressTokens = s.suppressionTokensScratch(params.SuppressTokens, stopTokens)
		}
		if !s.sampledPipelinedGPUTailCanContinue(pickParams, history, nil) {
			rerr = core.NewError("native.ArchSession.generateSampledPipelinedGPUTail: sampled parameters changed to a non-pipeline shape")
			return inflightSampledStep{}, false
		}
		draw := sampler.Draw()
		icb, tgt := icbs[i], icbs[1-i]
		sc[i].out = tgt.pleInput
		cb := commandBufferFast(queue)
		enc := computeCommandEncoderFast(cb)
		lastOut := icb.encodeStepBodyNoInput(enc, s.pos)
		if s.sampleTopKTokenParamsEligible(pickParams) {
			scratch, ok, stepErr := s.headEnc.encodeTopKSample(enc, lastOut, pickParams, draw, history, false)
			if !ok || stepErr != nil {
				endEncodingFast(enc)
				if scratch != nil {
					s.headEnc.putTopKScratch(scratch)
				}
				if stepErr == nil {
					stepErr = core.NewError("native.ArchSession.generateSampledPipelinedGPUTail: TopK token path declined mid-pipeline")
				}
				rerr = stepErr
				return inflightSampledStep{}, false
			}
			if stepErr = s.encNextInputsGPU(enc, scratch.outToken, tgt.ping0, sc[i]); stepErr != nil {
				endEncodingFast(enc)
				s.headEnc.putTopKScratch(scratch)
				rerr = stepErr
				return inflightSampledStep{}, false
			}
			endEncodingFast(enc)
			commitCommandBufferFast(cb)
			s.pos++
			return inflightSampledStep{cb: cb, lastOut: icb.lastOutPtr, topK: scratch}, true
		}
		scratch, ok, stepErr := s.headEnc.encodeLogitsSample(enc, lastOut, pickParams, draw, history)
		if !ok || stepErr != nil {
			endEncodingFast(enc)
			if scratch != nil {
				s.headEnc.putGreedyScratch(scratch)
			}
			if stepErr == nil {
				stepErr = core.NewError("native.ArchSession.generateSampledPipelinedGPUTail: logits token path declined mid-pipeline")
			}
			rerr = stepErr
			return inflightSampledStep{}, false
		}
		if stepErr = s.encNextInputsGPU(enc, scratch.outToken, tgt.ping0, sc[i]); stepErr != nil {
			endEncodingFast(enc)
			s.headEnc.putGreedyScratch(scratch)
			rerr = stepErr
			return inflightSampledStep{}, false
		}
		endEncodingFast(enc)
		commitCommandBufferFast(cb)
		s.pos++
		return inflightSampledStep{cb: cb, lastOut: icb.lastOutPtr, logits: scratch}, true
	}

	tokBuf := s.nextInputTokenBuffer(gen[len(gen)-1])
	sc[0].out = icbs[0].pleInput
	seedCB := commandBufferFast(queue)
	seedEnc := computeCommandEncoderFast(seedCB)
	if err := s.encNextInputsGPU(seedEnc, tokBuf, icbs[0].ping0, sc[0]); err != nil {
		endEncodingFast(seedEnc)
		return gen, history, err
	}
	endEncodingFast(seedEnc)
	commitCommandBufferFast(seedCB)
	waitUntilCompletedFast(seedCB)

	prev, ok := submit(0, len(gen))
	if !ok {
		return gen, history, rerr
	}
	i := 1
	stop := false
	for len(gen) < maxNew && !stop {
		nxt, ok := submit(i, len(gen)+1)
		if !ok {
			prev.cb.WaitUntilCompleted()
			release(prev)
			return gen, history, rerr
		}
		i = 1 - i
		token, valid := read(prev)
		if !valid {
			nxt.cb.WaitUntilCompleted()
			release(nxt)
			return gen, history, rerr
		}
		gen = append(gen, token)
		stop = (yield != nil && !yield(token)) || nativeTokenInSet(token, stopTokens)
		prev = nxt
	}
	token, valid := read(prev)
	if valid && !stop && len(gen) < maxNew {
		gen = append(gen, token)
	}
	if rerr != nil {
		return gen, history, rerr
	}
	s.rememberRetainedHiddenFrom(prev.lastOut)
	return gen, history, nil
}

func (s *ArchSession) sampleTopKCandidatesFromHiddenInPool(hidden []byte, params model.SampleParams) ([]byte, []int32, bool, error) {
	if !s.sampleTopKParamsEligible(params) {
		return nil, nil, false, nil
	}
	var logits []byte
	var ids []int32
	var ok bool
	var err error
	if hiddenBuf := s.retainedHiddenBufferFor(hidden); hiddenBuf != nil {
		logits, ids, ok, err = s.headEnc.sampleTopKCandidatesBufferInto(hiddenBuf, params.TopK, params.SuppressTokens, s.sampleCandidateLogits, s.sampleCandidateIDs, false)
	} else {
		logits, ids, ok, err = s.headEnc.sampleTopKCandidatesInto(hidden, params.TopK, params.SuppressTokens, s.sampleCandidateLogits, s.sampleCandidateIDs, false)
	}
	if ok {
		s.sampleCandidateLogits, s.sampleCandidateIDs = logits, ids
	}
	return logits, ids, ok, err
}

func (s *ArchSession) sampleTopKTokenFromHiddenInPool(hidden []byte, params model.SampleParams, draw float32, history []int32) (int32, bool, error) {
	if !s.sampleTopKTokenParamsEligible(params) {
		return 0, false, nil
	}
	if hiddenBuf := s.retainedHiddenBufferFor(hidden); hiddenBuf != nil {
		return s.headEnc.sampleTopKTokenBufferInPool(hiddenBuf, params, draw, history)
	}
	return s.headEnc.sampleTopKTokenInPool(hidden, params, draw, history)
}

func (s *ArchSession) sampleLogitsTokenFromHiddenInPool(hidden []byte, params model.SampleParams, draw float32, history []int32) (int32, bool, error) {
	if !s.sampleLogitsTokenParamsEligible(params) {
		return 0, false, nil
	}
	if hiddenBuf := s.retainedHiddenBufferFor(hidden); hiddenBuf != nil {
		return s.headEnc.sampleLogitsTokenBufferInPool(hiddenBuf, params, draw, history)
	}
	return s.headEnc.sampleLogitsTokenInPool(hidden, params, draw, history)
}

func (s *ArchSession) sampleTokenFromRetainedLogitsInPool(params model.SampleParams, draw float32, history []int32) (int32, bool, error) {
	logitsBuf := s.retainedLogitsBuffer()
	if logitsBuf == nil || !s.retainedLogitsSampleParamsEligible(params) {
		return 0, false, nil
	}
	return s.headEnc.sampleLogitsBufferInPool(logitsBuf, params, draw, history)
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
		hidden, err := s.prefillPromptRetainedInPool(promptIDs)
		if err != nil {
			genErr = err
			return
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

func nativeTokenInSet(id int32, tokens []int32) bool {
	for _, token := range tokens {
		if id == token {
			return true
		}
	}
	return false
}

func nativeAppendSuppressionTokens(base, extra []int32) []int32 {
	if len(extra) == 0 {
		return base
	}
	out := make([]int32, 0, len(base)+len(extra))
	out = append(out, base...)
	for _, token := range extra {
		if nativeTokenInSet(token, out) {
			continue
		}
		out = append(out, token)
	}
	return out
}

func nativeApplyRepeatPenaltyBF16(logits []byte, vocab int, history []int32, penalty float32) ([]byte, error) {
	if len(logits) != vocab*bf16Size {
		return nil, core.NewError("native.applyRepeatPenalty: logits must be vocab bf16 bytes")
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
	slices.Sort(ids)
	out := make([]byte, len(logits))
	copy(out, logits)
	applyRepeatPenaltySortedIDsBF16(out, ids, penalty)
	return out, nil
}

func applyRepeatPenaltySortedIDsBF16(out []byte, ids []int32, penalty float32) {
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
}
