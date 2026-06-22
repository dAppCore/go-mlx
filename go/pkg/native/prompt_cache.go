// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import core "dappco.re/go"

// prompt_cache.go is native automatic prompt caching (12-14): the metal serve path reuses a warm KV
// cache when a new request shares a prefix with the last one (generate.go PromptCache); the no-cgo path
// had no equivalent. GenerateCached tracks the token ids resident in the cache and, on a new prompt,
// rolls back to the longest shared prefix so only the divergent suffix is re-prefilled — the prefix's
// K/V is reused intact. Because the resident cache is byte-faithful to a fresh forward (proven by
// TestSessionKVCacheByteIdentical), the result is TOKEN-IDENTICAL to a cold Generate; the win is
// skipping the recompute of the shared prefix (the dominant cost in multi-turn chat / shared system
// prompts). Single-goroutine.

// GenerateCached is Generate with automatic prompt-cache prefix reuse: it finds the longest prefix of
// promptIDs already resident from a prior call, re-prefills only the suffix (always at least the last
// prompt token, so there is a hidden to decode from), and decodes maxNew tokens. eosID < 0 disables
// early stop. The returned token stream is identical to Generate(promptIDs, ...) on a cold session; only
// the prefix recompute is skipped. The cache's resident ids are updated to promptIDs + the generated run.
func (s *ArchSession) GenerateCached(promptIDs []int32, maxNew, eosID int) ([]int32, error) {
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
	if lcp > len(promptIDs)-1 {
		lcp = len(promptIDs) - 1
	}
	s.pos = lcp // roll the resident cache back to the shared prefix; its K/V rows are reused as-is
	gen, err := s.Generate(promptIDs[lcp:], maxNew, eosID)
	if err != nil {
		s.cachedIDs = nil // a failed run leaves the cache in an unknown state; force a cold next call
		return nil, err
	}
	resident := make([]int32, 0, len(promptIDs)+len(gen))
	resident = append(resident, promptIDs...)
	resident = append(resident, gen...)
	s.cachedIDs = resident
	return gen, nil
}

// CachedPrefixLen reports how many leading tokens of promptIDs would be served from the warm cache by
// GenerateCached (0 on a cold session) — the prompt-cache hit length, for serve-side metrics.
func (s *ArchSession) CachedPrefixLen(promptIDs []int32) int {
	lcp := 0
	for lcp < len(promptIDs) && lcp < len(s.cachedIDs) && promptIDs[lcp] == s.cachedIDs[lcp] {
		lcp++
	}
	if lcp > len(promptIDs)-1 && len(promptIDs) > 0 {
		lcp = len(promptIDs) - 1
	}
	return lcp
}
