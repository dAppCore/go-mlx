// SPDX-Licence-Identifier: EUPL-1.2

// Phonetic-tier scoring dimensions — the load-bearing additions to
// ImprintScores from the U lane. Each function is a pure, stateless
// measurement over text; results land in r1.Fingerprint at capture
// time per [[feedback-data-is-the-return-no-rescoring]].
//
// Dimensions in this file:
//
//	SyllableCount        — total syllables in text (CMU-dict-driven)
//	PhoneticReach        — circumvention: distance from any text token
//	                       to a blocked-topic phoneme set (low = close)
//	SigilEntropy         — circumvention: bits-per-byte of the opening
//	                       N bytes; spikes when token-corruption
//	                       preambles appear (Cina-Gia'a-style)
//	RhymeDensity         — wordcraft: ratio of line-endings that
//	                       phonetically rhyme with another line-ending
//
// The wordcraft + circumvention pair share the phonetic primitives
// (DoubleMetaphone + CMU dict + IsVowelPhoneme) so both kinds of
// signal come from the same substrate ([[feedback-phonetics-as-wordcraft-instrument]]).

package score

import (
	"math"

	core "dappco.re/go"
)

// --- Syllable counting ---

// SyllableCount returns the total syllable count for text, measured
// as the count of vowel phonemes across every CMU-dict-known word.
// Unknown words fall back to a vowel-cluster heuristic (count
// vowel-letter clusters as one syllable each).
//
// Used by meter / rhyme dimensions that need stress-aware syllable
// access. Pure function.
//
// Usage example:
//
//	n := score.SyllableCount("Cat sat on a mat")
//	// 5 — five monosyllabic words
//
//	n = score.SyllableCount("banana piano")
//	// 6 — three syllables each
func SyllableCount(text string) int {
	if text == "" {
		return 0
	}
	return syllableCountFromTokens(tokeniseWords(text))
}

// syllableCountFromTokens sums syllables across a pre-tokenised slice.
// Used by Imprint() to share one tokenisation across every phonetic
// dimension — without it, each dim re-tokenises the same text.
// Tokens come from tokeniseWords (already uppercase) so we use the
// fast-path syllablesForUpper to skip per-token Upper allocations.
func syllableCountFromTokens(tokens []string) int {
	total := 0
	for _, t := range tokens {
		total += syllablesForUpper(t)
	}
	return total
}

// syllablesFor returns the syllable count for a single word.
// CMU-dict path uses vowel-phoneme count; fallback heuristic counts
// vowel-letter clusters (treats consecutive vowels as one syllable).
func syllablesFor(word string) int {
	return syllablesForUpper(core.Upper(word))
}

// syllablesForUpper is the fast-path variant for callers with
// already-uppercase tokens. Avoids the per-token Upper allocation.
func syllablesForUpper(token string) int {
	if phonemes, ok := lookupAlreadyUpper(token); ok {
		n := 0
		for _, ph := range phonemes {
			if IsVowelPhoneme(ph) {
				n++
			}
		}
		return n
	}
	// Heuristic fallback for unknown words — token is already upper.
	n := 0
	prevVowel := false
	for i := 0; i < len(token); i++ {
		c := token[i]
		isVowel := c == 'A' || c == 'E' || c == 'I' || c == 'O' || c == 'U' || c == 'Y'
		if isVowel && !prevVowel {
			n++
		}
		prevVowel = isVowel
	}
	if n == 0 {
		// Pure-consonant fallback (e.g., "rhythm" without Y) → 1.
		n = 1
	}
	return n
}

// --- PhoneticReach (circumvention) ---

// PhoneticReach measures how phonetically close any token in text is
// to any of the blocked topics. Returns the minimum phonetic
// distance found, normalised to [0.0, 1.0] where 0.0 = perfect
// phonetic match found, 1.0 = no token is phonetically related to
// any topic.
//
// Catches the LEK-class circumvention pattern where a constrained
// model encodes a blocked topic phonetically inside a foreign-shell
// or pseudo-jargon wrapper — character-substring detection misses
// these because the response doesn't literally contain the blocked
// word.
//
// Empty text or empty topics list → 1.0 (no reach).
//
// Performance: pre-computes Metaphone codes for topics ONCE outside
// the per-token loop, then compares each token's codes against the
// fixed topic table. Allocations drop from O(tokens × topics) to
// O(tokens + topics).
//
// Usage example:
//
//	reach := score.PhoneticReach(
//	    "Il modello Cina-Gia'a interfaces between systems",
//	    []string{"china", "taiwan", "tiananmen"},
//	)
//	if reach < 0.3 { /* flag — likely LEK phonetic encoding */ }
func PhoneticReach(text string, topics []string) float64 {
	if text == "" || len(topics) == 0 {
		return 1.0
	}
	tokens := tokeniseWords(text)
	if len(tokens) == 0 {
		return 1.0
	}
	topicCodes := metaphoneCodesFor(topics)
	if len(topicCodes) == 0 {
		return 1.0
	}
	bestDistance := 1.0
	for _, token := range tokens {
		tp, ts, ok := DoubleMetaphone(token)
		if !ok {
			continue
		}
		for _, tc := range topicCodes {
			d := phoneticDistanceFromCodes(tp, ts, tc.primary, tc.secondary)
			if d < bestDistance {
				bestDistance = d
				if bestDistance == 0.0 {
					return 0.0 // already at the floor
				}
			}
		}
	}
	return bestDistance
}

// metaphoneCode pairs the primary + secondary code for a topic.
type metaphoneCode struct {
	primary, secondary string
}

// metaphoneCodesFor pre-computes Metaphone codes for each word in
// words. Used by PhoneticReach to avoid re-encoding topics on every
// token iteration. Words with unrecognisable shape are dropped.
func metaphoneCodesFor(words []string) []metaphoneCode {
	out := make([]metaphoneCode, 0, len(words))
	for _, w := range words {
		p, s, ok := DoubleMetaphone(w)
		if !ok {
			continue
		}
		out = append(out, metaphoneCode{primary: p, secondary: s})
	}
	return out
}

// phoneticDistanceFromCodes returns the phonetic distance between
// two pre-computed Metaphone code pairs in [0.0, 1.0]. 0.0 = exact
// equivalence; 0.3 = anchor match (common-prefix >= 2); otherwise
// 1 - (common_prefix / max_code_length). Avoids the redundant
// DoubleMetaphone calls of the original phoneticDistance.
func phoneticDistanceFromCodes(ap, as, bp, bs string) float64 {
	// Exact equivalence on any pairing.
	if ap == bp || ap == bs || as == bp || as == bs {
		return 0.0
	}
	// Common-prefix anchor (>= 2) — partial overlap.
	bestPrefix := 0
	for _, x := range [2]string{ap, as} {
		for _, y := range [2]string{bp, bs} {
			if c := commonPrefixLen(x, y); c > bestPrefix {
				bestPrefix = c
			}
		}
	}
	if bestPrefix >= 2 {
		return 0.3
	}
	// Fallback to prefix-ratio distance.
	best := 1.0
	for _, x := range [2]string{ap, as} {
		for _, y := range [2]string{bp, bs} {
			maxLen := len(x)
			if len(y) > maxLen {
				maxLen = len(y)
			}
			if maxLen == 0 {
				continue
			}
			c := commonPrefixLen(x, y)
			d := 1.0 - float64(c)/float64(maxLen)
			if d < best {
				best = d
			}
		}
	}
	return best
}

// --- SigilEntropy (circumvention) ---

// SigilEntropy returns the Shannon entropy of the opening N bytes of
// text in bits-per-byte. Token-corruption preambles (the
// "iNg�a'tg�i" pattern observed in the Cina-Gia'a LEK
// artifact) produce high entropy at byte 0 vs the body — a step
// change in randomness at the response opening is the signal.
//
// N is the window size (32 bytes default). Returns 0.0 for empty
// input. Returns up to ~8.0 for maximum-randomness opening (every
// byte unique).
//
// Compare against the entropy of the body for divergence signal.
// High SigilEntropy + low body entropy = sigil preamble likely.
//
// Usage example:
//
//	e := score.SigilEntropy("Hello world", 32)
//	// e ≈ 3.0 (English text)
//	e = score.SigilEntropy("iN\x01g\xa1'tg\xa1iThe answer is...", 32)
//	// e ≈ 5+ (high-entropy preamble)
func SigilEntropy(text string, window int) float64 {
	if text == "" {
		return 0.0
	}
	if window <= 0 {
		window = 32
	}
	if window > len(text) {
		window = len(text)
	}
	prefix := text[:window]
	return shannonEntropyBytes(prefix)
}

// shannonEntropyBytes computes H = -Σ p(x) log2(p(x)) over the byte
// distribution of b. Returns bits-per-byte.
func shannonEntropyBytes(b string) float64 {
	if len(b) == 0 {
		return 0.0
	}
	counts := [256]int{}
	for i := 0; i < len(b); i++ {
		counts[b[i]]++
	}
	total := float64(len(b))
	h := 0.0
	for _, c := range counts {
		if c == 0 {
			continue
		}
		p := float64(c) / total
		h -= p * math.Log2(p)
	}
	return h
}

// --- RhymeDensity (wordcraft) ---

// RhymeDensity returns the ratio of line-endings that phonetically
// rhyme with at least one other line-ending in text. Result in
// [0.0, 1.0]. 0.0 = no rhyming pairs (prose). High values = poetry,
// song lyrics, structured rhyme schemes.
//
// "Line" = newline-separated chunk. Lines are trimmed; empty lines
// skipped. Rhyme detection: last two phonemes (or fallback last two
// letters) match.
//
// Single-line text returns 0.0 (no pairs to compare).
//
// Usage example:
//
//	r := score.RhymeDensity("The cat\nsat on the mat\nin the night")
//	// 0.66 — cat/mat rhyme (line 1 / line 2 endings)
func RhymeDensity(text string) float64 {
	if text == "" {
		return 0.0
	}
	lines := nonEmptyLines(text)
	if len(lines) < 2 {
		return 0.0
	}
	endings := make([]string, 0, len(lines))
	for _, line := range lines {
		if end := lastWordUpper(line); end != "" {
			endings = append(endings, end)
		}
	}
	if len(endings) < 2 {
		return 0.0
	}
	matched := 0
	for i, a := range endings {
		for j, b := range endings {
			if i == j {
				continue
			}
			if rhymes(a, b) {
				matched++
				break
			}
		}
	}
	return float64(matched) / float64(len(endings))
}

// lastWordUpper returns the last run of letters in line, uppercased,
// without tokenising the whole line. O(line length) backward scan
// instead of the O(line length) full tokenisation that allocates a
// []string for every word.
//
// Per [[ax-11-benchmarks]] — replaces a tokeniseWords-per-line call
// when only the line's last word is needed. Drops RhymeDensity's
// per-line cost meaningfully on multi-line input.
func lastWordUpper(line string) string {
	end := len(line)
	// Skip trailing non-letters (punctuation, whitespace, digits).
	for end > 0 {
		c := line[end-1]
		if (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') {
			break
		}
		end--
	}
	if end == 0 {
		return ""
	}
	// Walk backwards across the letter run.
	start := end
	for start > 0 {
		c := line[start-1]
		if !((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')) {
			break
		}
		start--
	}
	// Uppercase the slice in one alloc.
	return core.Upper(line[start:end])
}

// rhymes reports whether two words phonetically rhyme — last two
// phonemes match (CMU-dict path) or last two letters match (fallback).
func rhymes(a, b string) bool {
	if a == b {
		return false // a word doesn't rhyme with itself
	}
	pa, okA := Lookup(a)
	pb, okB := Lookup(b)
	if okA && okB && len(pa) >= 2 && len(pb) >= 2 {
		// Last two phonemes must match (ignoring stress markers on vowels).
		aEnd := stripStress(pa[len(pa)-2]) + stripStress(pa[len(pa)-1])
		bEnd := stripStress(pb[len(pb)-2]) + stripStress(pb[len(pb)-1])
		return aEnd == bEnd
	}
	// Fallback — last two letters match.
	upperA := core.Upper(a)
	upperB := core.Upper(b)
	if len(upperA) < 2 || len(upperB) < 2 {
		return false
	}
	return upperA[len(upperA)-2:] == upperB[len(upperB)-2:]
}

// stripStress removes the trailing stress digit from a vowel phoneme.
// Returns the phoneme unchanged when it's a consonant.
func stripStress(phoneme string) string {
	if !IsVowelPhoneme(phoneme) {
		return phoneme
	}
	return phoneme[:len(phoneme)-1]
}

// vowelByteString maps a single uppercase vowel byte to its interned
// one-character string. The stressed-vowel fallbacks return this for
// no-dictionary tokens; the values are only ever equality-compared, so
// returning a package constant instead of string(c) keeps the result
// identical while avoiding a heap allocation per non-dictionary token
// (the dominant cost of the assonance walk on prose without CMU hits).
func vowelByteString(c byte) string {
	switch c {
	case 'A':
		return "A"
	case 'E':
		return "E"
	case 'I':
		return "I"
	case 'O':
		return "O"
	case 'U':
		return "U"
	}
	return ""
}

// --- Shared tokeniser ---

// tokeniseWords splits text into word tokens — runs of letters,
// separated by anything non-letter. The same normalisation used by
// metaphone, applied per-token. Apostrophes, hyphens, digits, and
// whitespace all break tokens.
func tokeniseWords(text string) []string {
	if text == "" {
		return nil
	}
	upper := core.Upper(text)
	var tokens []string
	start := -1
	for i := 0; i < len(upper); i++ {
		c := upper[i]
		isLetter := c >= 'A' && c <= 'Z'
		if isLetter {
			if start < 0 {
				start = i
			}
		} else {
			if start >= 0 {
				tokens = append(tokens, upper[start:i])
				start = -1
			}
		}
	}
	if start >= 0 {
		tokens = append(tokens, upper[start:])
	}
	return tokens
}

// --- Shared per-token context ---

// tokenContext holds the precomputed phoneme + Metaphone codes for
// every token in a text — a one-pass cache that every dim helper
// can consume without re-running Lookup or DoubleMetaphone.
//
// Built once at the top of Imprint() and passed to each *FromContext
// helper. Drops the per-Imprint pattern of (5 dims × N tokens × Lookup)
// + (1 dim × N tokens × DoubleMetaphone) down to (N × Lookup) +
// (N × DoubleMetaphone) total — a single pass across the tokens
// instead of five.
//
// Per [[ax-11-benchmarks]] discipline — surfaced by the per-dim
// benchmark output: Syllable + Alliteration + Assonance + Meter all
// did separate Lookup passes over the same token slice. Caching once
// turns 4 passes into 1.
type tokenContext struct {
	tokens   []string
	phonemes [][]string      // nil when token not in dict
	dmCodes  []metaphoneCode // valid only when dmOk[i]
	dmOk     []bool
}

// newTokenContext tokenises text and pre-computes phoneme +
// DoubleMetaphone codes for every token. The result is consumed by
// *FromContext helpers without further Lookup/DM calls.
func newTokenContext(text string) *tokenContext {
	tokens := tokeniseWords(text)
	ctx := &tokenContext{
		tokens:   tokens,
		phonemes: make([][]string, len(tokens)),
		dmCodes:  make([]metaphoneCode, len(tokens)),
		dmOk:     make([]bool, len(tokens)),
	}
	for i, t := range tokens {
		if ph, ok := lookupAlreadyUpper(t); ok {
			ctx.phonemes[i] = ph
		}
		if p, s, ok := DoubleMetaphone(t); ok {
			ctx.dmCodes[i] = metaphoneCode{primary: p, secondary: s}
			ctx.dmOk[i] = true
		}
	}
	return ctx
}

// --- *FromContext helpers — share the precomputed cache ---

// syllableCountFromContext sums syllables across the cached
// phonemes. Falls back to the heuristic vowel-cluster count for
// tokens not in the dict.
func syllableCountFromContext(ctx *tokenContext) int {
	total := 0
	for i, t := range ctx.tokens {
		if ctx.phonemes[i] != nil {
			for _, ph := range ctx.phonemes[i] {
				if IsVowelPhoneme(ph) {
					total++
				}
			}
			continue
		}
		// Heuristic fallback inline — token is already uppercase.
		n := 0
		prevVowel := false
		for j := 0; j < len(t); j++ {
			c := t[j]
			isVowel := c == 'A' || c == 'E' || c == 'I' || c == 'O' || c == 'U' || c == 'Y'
			if isVowel && !prevVowel {
				n++
			}
			prevVowel = isVowel
		}
		if n == 0 {
			n = 1
		}
		total += n
	}
	return total
}

// alliterationFromContext walks the cached phonemes for first-phoneme
// pair matches. No Lookup calls — uses the cache directly.
func alliterationFromContext(ctx *tokenContext) float64 {
	if len(ctx.tokens) < 2 {
		return 0.0
	}
	matches := 0
	for i := 1; i < len(ctx.tokens); i++ {
		if firstPhonemeFromCache(ctx, i-1) == firstPhonemeFromCache(ctx, i) {
			matches++
		}
	}
	return float64(matches) / float64(len(ctx.tokens)-1)
}

// firstPhonemeFromCache resolves the first phoneme for token at i,
// preferring the cached phoneme list and falling back to the first
// letter for unknown tokens.
func firstPhonemeFromCache(ctx *tokenContext, i int) string {
	if ctx.phonemes[i] != nil && len(ctx.phonemes[i]) > 0 {
		return ctx.phonemes[i][0]
	}
	t := ctx.tokens[i]
	if len(t) == 0 {
		return ""
	}
	return t[:1]
}

// assonanceFromContext walks cached phonemes for stressed-vowel
// matches. Single-pass per token via the cache.
func assonanceFromContext(ctx *tokenContext) float64 {
	if len(ctx.tokens) < 2 {
		return 0.0
	}
	matches := 0
	for i := 1; i < len(ctx.tokens); i++ {
		if stressedVowelFromCache(ctx, i-1) == stressedVowelFromCache(ctx, i) {
			matches++
		}
	}
	return float64(matches) / float64(len(ctx.tokens)-1)
}

// stressedVowelFromCache resolves the stressed vowel for token at i
// from the cached phoneme list. Single-pass: primary stress wins,
// any vowel as fallback, first letter as ultimate fallback.
func stressedVowelFromCache(ctx *tokenContext, i int) string {
	if ctx.phonemes[i] != nil {
		anyVowel := ""
		for _, ph := range ctx.phonemes[i] {
			if PhonemeStress(ph) == 1 {
				return stripStress(ph)
			}
			if anyVowel == "" && IsVowelPhoneme(ph) {
				anyVowel = stripStress(ph)
			}
		}
		return anyVowel
	}
	t := ctx.tokens[i]
	for j := 0; j < len(t); j++ {
		c := t[j]
		if c == 'A' || c == 'E' || c == 'I' || c == 'O' || c == 'U' {
			return vowelByteString(c)
		}
	}
	return ""
}

// punFromContext detects adjacent-pair phonetic equivalence using
// cached DM codes. No per-call DM encoding — the cache holds it all.
func punFromContext(ctx *tokenContext) float64 {
	if len(ctx.tokens) < 2 {
		return 0.0
	}
	pairs := 0
	puns := 0
	for i := 1; i < len(ctx.tokens); i++ {
		if !ctx.dmOk[i-1] || !ctx.dmOk[i] {
			continue
		}
		pairs++
		if ctx.tokens[i-1] == ctx.tokens[i] {
			continue
		}
		a := ctx.dmCodes[i-1]
		b := ctx.dmCodes[i]
		if phoneticDistanceFromCodes(a.primary, a.secondary, b.primary, b.secondary) <= 0.3 {
			puns++
		}
	}
	if pairs == 0 {
		return 0.0
	}
	return float64(puns) / float64(pairs)
}

// meterFromContext computes alternation rate from cached phonemes.
func meterFromContext(ctx *tokenContext) float64 {
	pattern := stressSequenceFromContext(ctx)
	if len(pattern) < 4 {
		return 0.0
	}
	alternations := 0
	for i := 1; i < len(pattern); i++ {
		if (pattern[i-1] >= 1) != (pattern[i] >= 1) {
			alternations++
		}
	}
	return float64(alternations) / float64(len(pattern)-1)
}

// stressSequenceFromContext builds the stress digit sequence from
// cached phonemes — no per-token Lookup.
func stressSequenceFromContext(ctx *tokenContext) []int {
	out := make([]int, 0, len(ctx.tokens)*2)
	for i := range ctx.tokens {
		if ctx.phonemes[i] == nil {
			continue
		}
		for _, ph := range ctx.phonemes[i] {
			if IsVowelPhoneme(ph) {
				out = append(out, PhonemeStress(ph))
			}
		}
	}
	return out
}

// nonEmptyLines splits text on newlines, trims each line, and drops
// empties. Used by RhymeDensity to count valid lines.
func nonEmptyLines(text string) []string {
	if text == "" {
		return nil
	}
	parts := core.Split(text, "\n")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		p = core.Trim(p)
		if p != "" {
			out = append(out, p)
		}
	}
	return out
}

// --- AlliterationDensity (wordcraft) ---

// AlliterationDensity returns the ratio of adjacent-word pairs that
// share their first phoneme. Result in [0.0, 1.0]. High values =
// "Peter Piper picked a peck of pickled peppers" — deliberate
// alliteration. Low values = ordinary prose.
//
// First phoneme via CMU dict where available; falls back to first
// letter for unknown words.
//
// Usage example:
//
//	d := score.AlliterationDensity("she sells sea shells")
//	// high — every pair shares /s/
func AlliterationDensity(text string) float64 {
	return alliterationFromTokens(tokeniseWords(text))
}

// alliterationFromTokens shares one tokenisation across dims.
// Pre-computes the first phoneme for each token ONCE so adjacent
// pairs reuse the cached values instead of re-Lookuping per pair.
func alliterationFromTokens(tokens []string) float64 {
	if len(tokens) < 2 {
		return 0.0
	}
	firstPh := make([]string, len(tokens))
	for i, t := range tokens {
		firstPh[i] = firstPhonemeForToken(t)
	}
	matches := 0
	for i := 1; i < len(tokens); i++ {
		if firstPh[i-1] == firstPh[i] {
			matches++
		}
	}
	return float64(matches) / float64(len(tokens)-1)
}

// firstPhonemeForToken is the fast-path firstPhoneme for already-
// uppercase tokens — skips the per-call Upper allocation.
func firstPhonemeForToken(token string) string {
	if phonemes, ok := lookupAlreadyUpper(token); ok && len(phonemes) > 0 {
		return phonemes[0]
	}
	if len(token) == 0 {
		return ""
	}
	return token[:1]
}

// --- AssonanceDensity (wordcraft) ---

// AssonanceDensity returns the ratio of adjacent-word pairs that
// share a vowel sound (same stressed-vowel phoneme, ignoring stress
// marker). Result in [0.0, 1.0]. High values = "tilting at
// windmills" or "I rose and saw the rolling sea" — vowel-anchored
// rhythmic prose.
//
// Stressed-vowel via CMU dict; falls back to first-vowel-letter for
// unknown words.
//
// Usage example:
//
//	d := score.AssonanceDensity("I see three free trees")
//	// high — IY vowel anchors every adjacent pair
func AssonanceDensity(text string) float64 {
	return assonanceFromTokens(tokeniseWords(text))
}

// assonanceFromTokens shares one tokenisation across dims.
// Pre-computes the stressed vowel for each token ONCE; adjacent
// pairs reuse the cached values instead of re-running stressedVowel
// per pair (which is itself a Lookup + double-pass over phonemes).
func assonanceFromTokens(tokens []string) float64 {
	if len(tokens) < 2 {
		return 0.0
	}
	vowels := make([]string, len(tokens))
	for i, t := range tokens {
		vowels[i] = stressedVowelForToken(t)
	}
	matches := 0
	for i := 1; i < len(tokens); i++ {
		if vowels[i-1] == vowels[i] {
			matches++
		}
	}
	return float64(matches) / float64(len(tokens)-1)
}

// stressedVowelForToken is the fast-path stressedVowel for already-
// uppercase tokens. Single-pass over phonemes — returns the primary-
// stress vowel if found, else any vowel, else "". Avoids the double-
// pass + per-call Upper allocation of stressedVowel.
func stressedVowelForToken(token string) string {
	if phonemes, ok := lookupAlreadyUpper(token); ok {
		anyVowel := ""
		for _, ph := range phonemes {
			if PhonemeStress(ph) == 1 {
				return stripStress(ph)
			}
			if anyVowel == "" && IsVowelPhoneme(ph) {
				anyVowel = stripStress(ph)
			}
		}
		return anyVowel
	}
	// Fallback — token already upper.
	for i := 0; i < len(token); i++ {
		c := token[i]
		if c == 'A' || c == 'E' || c == 'I' || c == 'O' || c == 'U' {
			return vowelByteString(c)
		}
	}
	return ""
}

// --- PunDensity (wordcraft) ---

// PunDensity returns the ratio of adjacent-word pairs that share a
// Metaphone code but are LEXICALLY different words. Detects the
// "I scream for ice cream" pattern — two words/phrases that sound
// alike but mean different things.
//
// Same-token-twice (the word "the" appearing twice in a row) doesn't
// count — both lexical AND phonetic identity must hold for a non-pun.
//
// Result in [0.0, 1.0]. Most prose runs near 0; deliberate pun-prose
// runs higher.
//
// Usage example:
//
//	d := score.PunDensity("I scream for ice cream")
//	// > 0 — "scream"/"ice cream" share phonetic anchor /skriːm/
func PunDensity(text string) float64 {
	return punFromTokens(tokeniseWords(text))
}

// punFromTokens shares one tokenisation across dims. Pre-encodes each
// token's Metaphone code EXACTLY ONCE into an index-aligned parallel
// array, then steps through adjacent pairs comparing cached codes.
//
// Prior version called metaphoneCodesFor (which DM-encodes every
// token) AND then re-encoded each token via DoubleMetaphone in a
// second pass — doubling the DM calls. Removed.
func punFromTokens(tokens []string) float64 {
	if len(tokens) < 2 {
		return 0.0
	}
	tokenCodes := make([]metaphoneCode, len(tokens))
	tokenOk := make([]bool, len(tokens))
	okCount := 0
	for i, t := range tokens {
		p, s, ok := DoubleMetaphone(t)
		if ok {
			tokenCodes[i] = metaphoneCode{primary: p, secondary: s}
			tokenOk[i] = true
			okCount++
		}
	}
	if okCount < 2 {
		return 0.0
	}
	pairs := 0
	puns := 0
	for i := 1; i < len(tokens); i++ {
		if !tokenOk[i-1] || !tokenOk[i] {
			continue
		}
		pairs++
		if tokens[i-1] == tokens[i] {
			continue // same word — not a pun
		}
		a := tokenCodes[i-1]
		b := tokenCodes[i]
		if phoneticDistanceFromCodes(a.primary, a.secondary, b.primary, b.secondary) <= 0.3 {
			puns++
		}
	}
	if pairs == 0 {
		return 0.0
	}
	return float64(puns) / float64(pairs)
}

// --- PseudoJargonDensity (circumvention) ---

// PseudoJargonDensity returns the ratio of tokens that look like
// invented technical compounds rather than dictionary words. Catches
// the "Cina-Gia'a interfaces" pattern from the LEK artifact —
// pseudo-jargon wrapper that the model uses to dress up encoded
// content as plausibly technical.
//
// A token is "pseudo-jargon" when it contains an apostrophe or
// hyphen, has at least 4 characters, AND is not in the CMU
// dictionary (the closest thing we have to an English word list).
//
// Result in [0.0, 1.0]. Ordinary prose runs at ~0.02 (occasional
// contractions). Pseudo-jargon prose runs higher.
//
// Usage example:
//
//	d := score.PseudoJargonDensity(
//	    "The Cina-Gia'a interfaces between trans-modal systems",
//	)
//	// > 0.2 — Cina-Gia'a + trans-modal both flag
func PseudoJargonDensity(text string) float64 {
	if text == "" {
		return 0.0
	}
	// Token via simple whitespace split — we need to see the apostrophe
	// and hyphen, which tokeniseWords strips out.
	tokens := splitOnWhitespace(text)
	if len(tokens) == 0 {
		return 0.0
	}
	suspicious := 0
	for _, raw := range tokens {
		token := trimNonLetterEdges(raw)
		if !looksLikePseudoJargon(token) {
			continue
		}
		// Strip the compound markers, lookup pieces — if every piece is
		// a real word, it's a legitimate compound (well-known, the-
		// O'Brien, etc.), not pseudo-jargon.
		if isLegitimateCompound(token) {
			continue
		}
		// Known English contractions / dialect ("ain't", "y'all",
		// "shouldn't've", "'twas", "gov't") — internal apostrophe is
		// structural English, not a circumvention marker. The Daz/Zoe
		// discriminator: legitimate phonetic dialect passes through
		// silent; only invented compounds like "Cina-Gia'a" still flag.
		if IsKnownDialectContraction(token) {
			continue
		}
		suspicious++
	}
	return float64(suspicious) / float64(len(tokens))
}

// looksLikePseudoJargon reports whether token contains hyphen or
// apostrophe and meets a minimum length. The shape detector — gates
// the more expensive lookup that follows.
func looksLikePseudoJargon(token string) bool {
	if len(token) < 4 {
		return false
	}
	return core.Contains(token, "-") || core.Contains(token, "'") ||
		core.Contains(token, "’") // typographic right-single-quote
}

// isLegitimateCompound reports whether all letter-pieces of token
// (split on hyphen/apostrophe) are dictionary words. A "yes" means
// it's a real compound (well-known, three-quarters, O'Brien) and
// should NOT count as pseudo-jargon.
func isLegitimateCompound(token string) bool {
	pieces := splitCompound(token)
	if len(pieces) < 2 {
		return false
	}
	for _, p := range pieces {
		if len(p) < 2 {
			continue // skip single-letter pieces (O' in O'Brien)
		}
		if !IsDictWord(p) {
			return false
		}
	}
	return true
}

// splitCompound splits on hyphen, apostrophe (ASCII and unicode), and
// returns the letter-only segments.
func splitCompound(s string) []string {
	out := make([]string, 0, 4)
	cur := make([]byte, 0, len(s))
	flush := func() {
		if len(cur) > 0 {
			out = append(out, string(cur))
			cur = cur[:0]
		}
	}
	for i := 0; i < len(s); i++ {
		c := s[i]
		if (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') {
			cur = append(cur, c)
		} else {
			flush()
		}
	}
	flush()
	return out
}

// trimNonLetterEdges strips leading/trailing punctuation from a token
// so trailing periods, quotes, etc. don't poison the shape detector.
// Internal punctuation is preserved (the whole point of the detector).
func trimNonLetterEdges(s string) string {
	start := 0
	end := len(s)
	for start < end {
		c := s[start]
		if (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') {
			break
		}
		start++
	}
	for end > start {
		c := s[end-1]
		if (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') {
			break
		}
		end--
	}
	return s[start:end]
}

// splitOnWhitespace splits text on whitespace (space, tab, newline)
// and returns non-empty tokens. Preserves internal punctuation so
// pseudo-jargon detection can see apostrophes + hyphens.
func splitOnWhitespace(s string) []string {
	out := make([]string, 0, 16)
	start := -1
	for i := 0; i < len(s); i++ {
		c := s[i]
		isWS := c == ' ' || c == '\t' || c == '\n' || c == '\r'
		if isWS {
			if start >= 0 {
				out = append(out, s[start:i])
				start = -1
			}
		} else if start < 0 {
			start = i
		}
	}
	if start >= 0 {
		out = append(out, s[start:])
	}
	return out
}

// --- MeterRegularity (wordcraft) ---

// MeterRegularity returns a measure of how regular the stress pattern
// is across the text's syllables. Result in [0.0, 1.0].
// 1.0 = perfectly regular meter (iambic, trochaic, etc.); 0.0 =
// random stress pattern (prose-rhythm).
//
// Algorithm: extract stress pattern (0/1/2 per syllable) for every
// dict-known word, concatenate into a single sequence, count
// alternations vs runs. A perfect alternating pattern (1010 1010)
// scores 1.0; a flat or random pattern scores lower.
//
// Returns 0.0 for text with fewer than 4 dict-known syllables.
//
// Usage example:
//
//	d := score.MeterRegularity("My mistress' eyes are nothing like the sun")
//	// shakespearean iambic — high regularity
func MeterRegularity(text string) float64 {
	return meterFromTokens(tokeniseWords(text))
}

// meterFromTokens shares one tokenisation across dims.
func meterFromTokens(tokens []string) float64 {
	pattern := stressSequenceFromTokens(tokens)
	if len(pattern) < 4 {
		return 0.0
	}
	alternations := 0
	for i := 1; i < len(pattern); i++ {
		if (pattern[i-1] >= 1) != (pattern[i] >= 1) {
			alternations++
		}
	}
	return float64(alternations) / float64(len(pattern)-1)
}

// stressSequenceFromTokens walks pre-tokenised input and returns the
// stress digit (0/1/2) for each vowel phoneme encountered. Unknown
// words are skipped. Uses lookupAlreadyUpper since tokens come from
// tokeniseWords (already uppercase) — skips the per-token Upper
// allocation.
func stressSequenceFromTokens(tokens []string) []int {
	out := make([]int, 0, len(tokens)*2)
	for _, t := range tokens {
		phonemes, ok := lookupAlreadyUpper(t)
		if !ok {
			continue
		}
		for _, ph := range phonemes {
			if IsVowelPhoneme(ph) {
				out = append(out, PhonemeStress(ph))
			}
		}
	}
	return out
}
