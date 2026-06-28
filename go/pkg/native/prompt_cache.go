// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"unsafe"

	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/model"
)

// prompt_cache.go is native automatic prompt caching (12-14): the metal serve path reuses a warm KV
// cache when a new request shares a prefix with the last one (generate.go PromptCache); the no-cgo path
// had no equivalent. GenerateCached tracks the token ids resident in the cache and, on a new prompt,
// rolls back to the longest shared prefix so only the divergent suffix is re-prefilled — the prefix's
// K/V is reused intact. Because the resident cache is byte-faithful to a fresh forward (proven by
// TestSessionKVCacheByteIdentical), the result is TOKEN-IDENTICAL to a cold Generate; the win is
// skipping the recompute of the shared prefix (the dominant cost in multi-turn chat / shared system
// prompts). Single-goroutine.

// GenerateCached is Generate with automatic prompt-cache prefix reuse: it finds the longest prefix of
// promptIDs already resident from a prior call, re-prefills only the suffix, and decodes maxNew tokens.
// Exact prompt hits replay the cached prompt-boundary hidden/logits state, mirroring metal prompt-cache
// entries and avoiding the old native fallback that re-prefilled the last prompt token and re-ran the
// first head projection just to recreate them.
// eosID < 0 disables early stop. The returned token stream is identical to Generate(promptIDs, ...) on a
// cold session; only prefix recompute is skipped. The cache's resident ids are updated to promptIDs +
// the generated run.
func (s *ArchSession) GenerateCached(promptIDs []int32, maxNew, eosID int) ([]int32, error) {
	return s.generateCached(promptIDs, maxNew, eosID, nil, nil, nil)
}

// GenerateCachedEach is GenerateCached with per-token streaming. Tokens are yielded after
// they are selected and written into the resident cache; returning false from yield stops
// generation and leaves the cache at the emitted token boundary.
func (s *ArchSession) GenerateCachedEach(promptIDs []int32, maxNew, eosID int, yield func(int32) bool) ([]int32, error) {
	return s.GenerateCachedEachWithSuppressionAndTransform(promptIDs, maxNew, eosID, nil, nil, yield)
}

// GenerateCachedEachTransformed is GenerateCachedEach with a committed-token
// transform applied before each generated token is written to the cache.
func (s *ArchSession) GenerateCachedEachTransformed(promptIDs []int32, maxNew, eosID int, transform TokenTransform, yield func(int32) bool) ([]int32, error) {
	return s.GenerateCachedEachWithSuppressionAndTransform(promptIDs, maxNew, eosID, nil, transform, yield)
}

// GenerateCachedEachWithSuppression is GenerateCachedEach with suppressed token
// ids masked before greedy argmax.
func (s *ArchSession) GenerateCachedEachWithSuppression(promptIDs []int32, maxNew, eosID int, suppress []int32, yield func(int32) bool) ([]int32, error) {
	return s.GenerateCachedEachWithSuppressionAndTransform(promptIDs, maxNew, eosID, suppress, nil, yield)
}

// GenerateCachedEachWithSuppressionAndTransform combines cached greedy token
// suppression with a committed-token transform.
func (s *ArchSession) GenerateCachedEachWithSuppressionAndTransform(promptIDs []int32, maxNew, eosID int, suppress []int32, transform TokenTransform, yield func(int32) bool) ([]int32, error) {
	return s.generateCached(promptIDs, maxNew, eosID, suppress, transform, yield)
}

// GenerateCachedSampledEach is GenerateSampledEach with automatic prompt-cache
// prefix reuse. Exact prompt hits replay the cached prompt-boundary hidden
// state and enter the normal sampled decoder, so sampling semantics stay
// identical to a cold GenerateSampledEach while prompt prefill is skipped.
func (s *ArchSession) GenerateCachedSampledEach(promptIDs []int32, maxNew int, stopTokens []int32, sampler *model.Sampler, params model.SampleParams, transform model.TokenTransform, yield func(int32) bool) ([]int32, error) {
	return s.generateCachedSampled(promptIDs, maxNew, stopTokens, sampler, params, transform, yield, true)
}

func (s *ArchSession) generateCached(promptIDs []int32, maxNew, eosID int, suppress []int32, transform TokenTransform, yield func(int32) bool) ([]int32, error) {
	if len(promptIDs) == 0 {
		return nil, core.NewError("native.GenerateCached: empty prompt")
	}
	if maxNew <= 0 {
		return nil, core.NewError("native.GenerateCached: maxNew must be > 0")
	}
	// longest common prefix of the new prompt and the resident ids — capped at len-1 so at least one
	// token is re-prefilled (Generate needs a token to produce the decode cursor hidden).
	lcp := 0
	for lcp < len(promptIDs) && lcp < len(s.cachedIDs) && promptIDs[lcp] == s.cachedIDs[lcp] {
		lcp++
	}
	if lcp == len(promptIDs) {
		if hidden := s.cachedPromptHiddenFor(promptIDs); hidden != nil {
			logits := s.cachedPromptLogitsFor(promptIDs)
			s.rememberRetainedHidden(hidden)
			hidden = s.retainedHidden
			s.pos = lcp // roll back over any generated tail; prompt K/V rows stay resident
			gen, err := s.generateFromHiddenSuppressedEach(hidden, maxNew, eosID, logits, suppress, transform, yield)
			if err != nil {
				s.cachedIDs = nil
				s.clearCachedPromptHidden()
				return nil, err
			}
			resident := s.cachedIDs[:0]
			resident = append(resident, promptIDs...)
			resident = append(resident, gen...)
			s.cachedIDs = resident
			return gen, nil
		}
		lcp = len(promptIDs) - 1
	}
	s.pos = lcp // roll the resident cache back to the shared prefix; its K/V rows are reused as-is
	gen, err := s.generateWithYield(promptIDs[lcp:], maxNew, eosID, promptIDs, suppress, transform, yield)
	if err != nil {
		s.cachedIDs = nil // a failed run leaves the cache in an unknown state; force a cold next call
		s.clearCachedPromptHidden()
		return nil, err
	}
	resident := s.cachedIDs[:0]
	resident = append(resident, promptIDs...)
	resident = append(resident, gen...)
	s.cachedIDs = resident
	return gen, nil
}

func (s *ArchSession) generateCachedSampled(promptIDs []int32, maxNew int, stopTokens []int32, sampler *model.Sampler, params model.SampleParams, transform model.TokenTransform, yield func(int32) bool, cacheFinal bool) ([]int32, error) {
	if sampler == nil {
		return nil, core.NewError("native.GenerateCachedSampledEach: nil sampler")
	}
	if len(promptIDs) == 0 {
		return nil, core.NewError("native.GenerateCachedSampledEach: empty prompt")
	}
	if maxNew <= 0 {
		return nil, core.NewError("native.GenerateCachedSampledEach: maxNew must be > 0")
	}
	if len(promptIDs)+maxNew > s.maxLen {
		return nil, core.NewError("native.GenerateCachedSampledEach: sequence would exceed maxLen cache rows")
	}
	lcp := 0
	for lcp < len(promptIDs) && lcp < len(s.cachedIDs) && promptIDs[lcp] == s.cachedIDs[lcp] {
		lcp++
	}
	if lcp == len(promptIDs) {
		if hidden := s.cachedPromptHiddenFor(promptIDs); hidden != nil {
			s.rememberRetainedHidden(hidden)
			hidden = s.retainedHidden
			s.pos = lcp
			var gen []int32
			var err error
			withAutoreleasePool(func() {
				gen, err = s.generateSampledFromHiddenInPool(hidden, maxNew, stopTokens, sampler, params, transform, yield, cacheFinal)
			})
			if err != nil {
				s.cachedIDs = nil
				s.clearCachedPromptHidden()
				return nil, err
			}
			resident := s.cachedIDs[:0]
			resident = append(resident, promptIDs...)
			resident = append(resident, gen...)
			s.cachedIDs = resident
			return gen, nil
		}
		lcp = len(promptIDs) - 1
	}
	s.pos = lcp
	var gen []int32
	var genErr error
	withAutoreleasePool(func() {
		hidden, err := s.prefillPromptRetainedInPool(promptIDs[lcp:])
		if err != nil {
			genErr = err
			return
		}
		s.rememberCachedPromptEntry(promptIDs, hidden, nil)
		gen, genErr = s.generateSampledFromHiddenInPool(hidden, maxNew, stopTokens, sampler, params, transform, yield, cacheFinal)
	})
	if genErr != nil {
		s.cachedIDs = nil
		s.clearCachedPromptHidden()
		return nil, genErr
	}
	resident := s.cachedIDs[:0]
	resident = append(resident, promptIDs...)
	resident = append(resident, gen...)
	s.cachedIDs = resident
	return gen, nil
}

// ClearPromptCache drops native retained-prefix metadata and rewinds the decode
// cursor. Existing K/V rows are left in place; pos=0 makes the next Generate or
// GenerateCached overwrite them from the beginning, mirroring metal's model
// prompt-cache clear without touching loaded weights.
func (s *ArchSession) ClearPromptCache() {
	if s == nil {
		return
	}
	s.pos = 0
	s.cachedIDs = nil
	s.clearCachedPromptHidden()
	s.resetRetainedHidden()
}

// WarmPromptCache prefills ids into the resident KV cache and records them as
// the reusable native prompt prefix. It mirrors metal's WarmPromptCache at the
// session-token layer: the exact prompt boundary stores hidden/logits, so later
// GenerateCached calls can replay the prompt without a throwaway generation,
// last-token re-prefill, or first-head recompute during generation.
func (s *ArchSession) WarmPromptCache(ids []int32) error {
	if len(ids) == 0 {
		return core.NewError("native.WarmPromptCache: empty prompt")
	}
	s.pos = 0
	s.resetCachedPromptEntry()
	s.resetRetainedHidden()
	resident := s.cachedIDs[:0]
	s.cachedIDs = resident
	hidden, logits, err := s.prefillPromptCacheEntry(ids)
	if err != nil {
		s.pos = 0
		s.cachedIDs = resident[:0]
		s.resetCachedPromptEntry()
		s.resetRetainedHidden()
		return err
	}
	s.cachedIDs = append(resident, ids...)
	s.rememberCachedPromptEntry(ids, hidden, logits)
	if s.retainedHiddenBufferFor(hidden) == nil {
		s.rememberRetainedHidden(hidden)
	}
	return nil
}

// CompactCache evicts the oldest resident tokens, keeping only the most recent `keep`, so a long
// conversation can continue past maxLen (or under a context budget) without unbounded cache growth. It
// re-prefills the kept tokens from position 0 — correct by construction, because each cached K row
// carries RoPE baked in at its ABSOLUTE position, so a naive shift-down would mis-rotate them; re-prefill
// re-rotates the kept tokens at their new positions [0..keep). The trade is the recompute of `keep`
// tokens for a compacted, correctly-positioned cache. After this, decoding continues exactly as a fresh
// session prefilled with the kept tokens would (proven in prompt_cache_test.go). keep >= the resident
// length is a no-op.
func (s *ArchSession) CompactCache(keep int) error {
	if keep < 0 {
		return core.NewError("native.CompactCache: keep must be >= 0")
	}
	if keep >= len(s.cachedIDs) {
		return nil // nothing to evict
	}
	kept := s.cachedIDs[len(s.cachedIDs)-keep:]
	s.pos = 0
	s.clearCachedPromptHidden()
	s.cachedIDs = nil
	if err := s.prefillCachedIDs(kept); err != nil {
		s.pos = 0
		s.resetRetainedHidden()
		return err
	}
	s.cachedIDs = kept
	s.resetRetainedHidden()
	return nil
}

func (s *ArchSession) prefillCachedIDs(ids []int32) error {
	if len(ids) == 0 {
		return nil
	}
	if s.pos+len(ids) > s.maxLen {
		return core.NewError("native.CompactCache: sequence would exceed maxLen cache rows")
	}
	if s.perLayerInput == nil && s.state.icb == nil {
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
				return err
			}
			embs[i] = emb
		}
		var ok bool
		var err error
		withAutoreleasePool(func() {
			ok, err = s.state.stepTokensBatchedDenseNoResult(embs, s.pos)
		})
		if err != nil {
			return err
		}
		if ok {
			s.pos += len(ids)
			return nil
		}
		withAutoreleasePool(func() {
			for _, emb := range embs {
				if err = s.state.stepTokenNoResult(emb, s.pos); err != nil {
					return
				}
				s.pos++
			}
		})
		return err
	}
	var err error
	withAutoreleasePool(func() {
		for _, id := range ids {
			var emb []byte
			emb, err = s.embedID(id)
			if err != nil {
				return
			}
			var pli []byte
			if s.perLayerInput != nil {
				pli, err = s.perLayerInput(id, emb)
				if err != nil {
					return
				}
				s.state.perLayerInput = pli
			}
			if s.state.icb != nil {
				s.state.icb.stepBodyNoResult(emb, s.pos, pli)
			} else if err = s.state.stepTokenNoResult(emb, s.pos); err != nil {
				return
			}
			s.pos++
		}
	})
	return err
}

func (s *ArchSession) prefillPromptCacheEntry(ids []int32) ([]byte, []byte, error) {
	if len(ids) == 0 {
		return nil, nil, nil
	}
	if s.pos+len(ids) > s.maxLen {
		return nil, nil, core.NewError("native.WarmPromptCache: sequence would exceed maxLen cache rows")
	}
	if len(ids) > 1 {
		if err := s.prefillCachedIDs(ids[:len(ids)-1]); err != nil {
			return nil, nil, err
		}
	}
	var hidden, logits []byte
	var err error
	withAutoreleasePool(func() {
		hidden, err = s.stepIDRetainedInPool(ids[len(ids)-1])
		if err != nil {
			return
		}
		logits, err = s.promptCacheLogitsFromRetainedHidden(hidden)
	})
	if err != nil {
		return nil, nil, err
	}
	return hidden, logits, nil
}

func (s *ArchSession) promptCacheLogitsFromRetainedHidden(hidden []byte) ([]byte, error) {
	if hiddenBuf := s.retainedHiddenBufferFor(hidden); hiddenBuf != nil && s.headEnc != nil {
		if pinned, ok := s.ensureRetainedLogitsPinned(s.arch.Vocab * bf16Size); ok {
			logits, err := s.headEnc.encodeBufferInto(hiddenBuf, true, pinned.bytes)
			if err != nil {
				return nil, err
			}
			s.retainedLogits = logits
			s.sampleHeadLogits = nil
			return s.retainedLogits, nil
		}
		logits, err := s.headEnc.encodeBufferInto(hiddenBuf, true, s.sampleHeadLogits)
		if err != nil {
			return nil, err
		}
		s.sampleHeadLogits = logits
		s.rememberRetainedLogits(logits)
		return s.retainedLogits, nil
	}
	logits, err := s.head(hidden, true)
	if err != nil {
		return nil, err
	}
	s.rememberRetainedLogits(logits)
	return s.retainedLogits, nil
}

// CachedPrefixLen reports how many leading tokens of promptIDs would be served from the warm cache by
// GenerateCached (0 on a cold session) — the prompt-cache hit length, for serve-side metrics.
func (s *ArchSession) CachedPrefixLen(promptIDs []int32) int {
	lcp := 0
	for lcp < len(promptIDs) && lcp < len(s.cachedIDs) && promptIDs[lcp] == s.cachedIDs[lcp] {
		lcp++
	}
	if lcp == len(promptIDs) && s.cachedPromptHiddenFor(promptIDs) != nil {
		return lcp
	}
	if lcp > len(promptIDs)-1 && len(promptIDs) > 0 {
		lcp = len(promptIDs) - 1
	}
	return lcp
}

func (s *ArchSession) rememberCachedPromptEntry(promptIDs []int32, hidden []byte, logits []byte) {
	if len(promptIDs) == 0 || len(hidden) == 0 {
		s.clearCachedPromptHidden()
		return
	}
	ids := s.cachedPromptIDs[:0]
	ids = append(ids, promptIDs...)
	s.cachedPromptIDs = ids
	s.cachedPromptHidden = s.stableCachedPromptHidden(hidden)
	if len(logits) == 0 {
		s.cachedPromptLogits = nil
		return
	}
	s.cachedPromptLogits = s.stableCachedPromptLogits(logits)
}

func (s *ArchSession) stableCachedPromptHidden(hidden []byte) []byte {
	n := len(hidden)
	if n == 0 {
		return nil
	}
	if cap(s.cachedPromptHidden) < n || sameByteBacking(s.cachedPromptHidden, s.retainedHidden) {
		s.cachedPromptHidden = make([]byte, n)
	} else {
		s.cachedPromptHidden = s.cachedPromptHidden[:n]
	}
	copy(s.cachedPromptHidden, hidden)
	return s.cachedPromptHidden
}

func (s *ArchSession) stableCachedPromptLogits(logits []byte) []byte {
	n := len(logits)
	if n == 0 {
		return nil
	}
	if cap(s.cachedPromptLogits) < n || sameByteBacking(s.cachedPromptLogits, s.retainedLogits) {
		s.cachedPromptLogits = make([]byte, n)
	} else {
		s.cachedPromptLogits = s.cachedPromptLogits[:n]
	}
	copy(s.cachedPromptLogits, logits)
	return s.cachedPromptLogits
}

func sameByteBacking(a, b []byte) bool {
	return byteBackingPointer(a) != nil && byteBackingPointer(a) == byteDataPointer(b)
}

func byteBackingPointer(b []byte) unsafe.Pointer {
	if cap(b) == 0 {
		return nil
	}
	return unsafe.Pointer(&b[:cap(b)][0])
}

func byteDataPointer(b []byte) unsafe.Pointer {
	if len(b) == 0 {
		return nil
	}
	return unsafe.Pointer(&b[0])
}

func (s *ArchSession) clearCachedPromptHidden() {
	if s == nil {
		return
	}
	s.cachedPromptIDs = nil
	s.cachedPromptHidden = nil
	s.cachedPromptLogits = nil
}

func (s *ArchSession) resetCachedPromptEntry() {
	if s == nil {
		return
	}
	s.cachedPromptIDs = s.cachedPromptIDs[:0]
	s.cachedPromptHidden = s.cachedPromptHidden[:0]
	s.cachedPromptLogits = s.cachedPromptLogits[:0]
}

func (s *ArchSession) cachedPromptHiddenFor(promptIDs []int32) []byte {
	if len(s.cachedPromptHidden) != s.arch.Hidden*bf16Size || !s.cachedPromptIDsMatch(promptIDs) {
		return nil
	}
	return s.cachedPromptHidden
}

func (s *ArchSession) cachedPromptLogitsFor(promptIDs []int32) []byte {
	if len(s.cachedPromptLogits) != s.arch.Vocab*bf16Size || !s.cachedPromptIDsMatch(promptIDs) {
		return nil
	}
	return s.cachedPromptLogits
}

func (s *ArchSession) cachedPromptIDsMatch(promptIDs []int32) bool {
	if len(promptIDs) == 0 || len(promptIDs) != len(s.cachedPromptIDs) {
		return false
	}
	for i, id := range promptIDs {
		if s.cachedPromptIDs[i] != id {
			return false
		}
	}
	return true
}
