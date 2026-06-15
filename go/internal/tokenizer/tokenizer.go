// SPDX-Licence-Identifier: EUPL-1.2

package tokenizer

import (
	"slices"
	"sync"
	"unicode/utf8"

	"dappco.re/go"

	coreio "dappco.re/go/io"
)

const (
	tokenizerBPECacheLimit           = 4096
	tokenizerBPECacheMaxSegmentBytes = 64 << 10
	tokenizerBPECacheMaxTokens       = 16 << 10
)

// Tokenizer handles text-to-token and token-to-text conversion.
type Tokenizer struct {
	vocab        map[string]int32
	invVocab     map[int32]string
	merges       []mergePair
	mergeRanks   map[string]int // "a b" → rank for O(1) merge lookup
	special      map[string]int32
	specialOrder []string

	bosToken int32
	eosToken int32
	hasBOS   bool
	hasEOS   bool

	// GPT-2 byte-level BPE support (used by Qwen, GPT, Llama, etc.)
	isGPT2BPE   bool
	gpt2Decoder map[rune]byte // Unicode char → original byte
	gpt2Encoder map[byte]rune // original byte → Unicode char

	bpeCacheMu    sync.RWMutex
	bpeCache      map[string][]int32
	bpeCacheOrder []string
}

type mergePair struct {
	a, b string
	rank int
}

// tokenizerJSON is the HuggingFace tokenizer.json format.
type tokenizerJSON struct {
	Model struct {
		Type         string `json:"type"`
		Vocab        any    `json:"vocab"`
		Merges       any    `json:"merges"`
		ByteFallback bool   `json:"byte_fallback"`
	} `json:"model"`
	AddedTokens []struct {
		ID      int32  `json:"id"`
		Content string `json:"content"`
		Special bool   `json:"special"`
	} `json:"added_tokens"`
}

// indexIn returns the byte position of substr in s, or -1 if not found.
// Replaces strings.Index without importing the strings package.
//
//	pos := indexIn("hello world", "world") // → 6
//	pos := indexIn("hello", "xyz")         // → -1
func indexIn(s, substr string) int {
	subLen := len(substr)
	if subLen == 0 {
		return 0
	}
	if subLen > len(s) {
		return -1
	}
	for i := range len(s) - subLen + 1 {
		if s[i:i+subLen] == substr {
			return i
		}
	}
	return -1
}

// LoadTokenizer reads a tokenizer.json file and creates a Tokenizer.
//
//	tok, err := metal.LoadTokenizer("/path/to/model/tokenizer.json")
func LoadTokenizer(path string) (*Tokenizer, error) {
	str, err := coreio.Local.Read(path)
	if err != nil {
		return nil, core.E("tokenizer.LoadTokenizer", "read "+path, err)
	}
	data := []byte(str)

	var tj tokenizerJSON
	if r := core.JSONUnmarshal(data, &tj); !r.OK {
		return nil, core.E("tokenizer.LoadTokenizer", "parse", nil)
	}

	tokenizer := &Tokenizer{
		vocab:    make(map[string]int32),
		invVocab: make(map[int32]string),
		special:  make(map[string]int32),
	}

	// Vocab arrives as any (map[string]interface{} from JSON) — convert
	// to map[string]int32 by re-marshalling through core.JSONMarshal.
	if tj.Model.Vocab != nil {
		vocabBytes := core.JSONMarshal(tj.Model.Vocab)
		if !vocabBytes.OK {
			return nil, core.E("tokenizer.LoadTokenizer", "re-encode vocab", nil)
		}
		var vocab map[string]int32
		if r := core.JSONUnmarshal(vocabBytes.Value.([]byte), &vocab); !r.OK {
			return nil, core.E("tokenizer.LoadTokenizer", "parse vocab", nil)
		}
		tokenizer.vocab = vocab
		for tokenText, tokenID := range vocab {
			tokenizer.invVocab[tokenID] = tokenText
		}
	}

	// Merges arrives as any — supports both ["a b", ...] and [["a","b"], ...]
	if tj.Model.Merges != nil {
		mergeBytes := core.JSONMarshal(tj.Model.Merges)
		if mergeBytes.OK {
			raw := mergeBytes.Value.([]byte)
			var stringMerges []string
			if r := core.JSONUnmarshal(raw, &stringMerges); r.OK {
				for rank, merge := range stringMerges {
					parts := core.SplitN(merge, " ", 2)
					if len(parts) == 2 {
						tokenizer.merges = append(tokenizer.merges, mergePair{a: parts[0], b: parts[1], rank: rank})
					}
				}
			} else {
				var arrayMerges [][]string
				if r := core.JSONUnmarshal(raw, &arrayMerges); r.OK {
					for rank, pair := range arrayMerges {
						if len(pair) == 2 {
							tokenizer.merges = append(tokenizer.merges, mergePair{a: pair[0], b: pair[1], rank: rank})
						}
					}
				}
			}
		}
	}

	tokenizer.mergeRanks = make(map[string]int, len(tokenizer.merges))
	for _, merge := range tokenizer.merges {
		tokenizer.mergeRanks[merge.a+" "+merge.b] = merge.rank
	}

	for _, added := range tj.AddedTokens {
		if added.Special {
			tokenizer.special[added.Content] = added.ID
		}
		tokenizer.vocab[added.Content] = added.ID
		tokenizer.invVocab[added.ID] = added.Content
	}
	tokenizer.specialOrder = make([]string, 0, len(tokenizer.special))
	for tokenText := range tokenizer.special {
		tokenizer.specialOrder = append(tokenizer.specialOrder, tokenText)
	}
	slices.SortFunc(tokenizer.specialOrder, func(a, b string) int {
		if len(a) != len(b) {
			return len(b) - len(a)
		}
		switch {
		case a < b:
			return -1
		case a > b:
			return 1
		default:
			return 0
		}
	})

	// Detect GPT-2 byte-level BPE (Qwen, GPT, DeepSeek use Ġ for space).
	// Check for "Ġthe" rather than bare "Ġ" — large SentencePiece vocabs
	// (Gemma3 262K) may include Ġ as an obscure character without using
	// GPT-2 byte encoding.
	if _, ok := tokenizer.vocab["Ġthe"]; ok {
		tokenizer.isGPT2BPE = true
		tokenizer.gpt2Decoder, tokenizer.gpt2Encoder = buildGPT2ByteMaps()
	}

	if id, ok := tokenizer.special["<bos>"]; ok {
		tokenizer.bosToken = id
		tokenizer.hasBOS = true
	}
	if id, ok := tokenizer.special["<eos>"]; ok {
		tokenizer.eosToken = id
		tokenizer.hasEOS = true
	}
	// Gemma: <end_of_turn> is the generation stop token
	if id, ok := tokenizer.special["<end_of_turn>"]; ok {
		tokenizer.eosToken = id
		tokenizer.hasEOS = true
	}
	// Qwen3: <|im_end|> is the generation stop token
	if id, ok := tokenizer.special["<|im_end|>"]; ok {
		tokenizer.eosToken = id
		tokenizer.hasEOS = true
	}
	// Qwen3 BOS: <|im_start|>
	if id, ok := tokenizer.special["<|im_start|>"]; ok {
		tokenizer.bosToken = id
		tokenizer.hasBOS = true
	}
	// Llama 3: <|eot_id|> is the turn-end token
	if id, ok := tokenizer.special["<|eot_id|>"]; ok {
		tokenizer.eosToken = id
		tokenizer.hasEOS = true
	}
	// Llama 3 BOS: <|begin_of_text|>
	if id, ok := tokenizer.special["<|begin_of_text|>"]; ok {
		tokenizer.bosToken = id
		tokenizer.hasBOS = true
	}

	return tokenizer, nil
}

func (t *Tokenizer) matchSpecialToken(input string) (string, int32, bool) {
	for _, tok := range t.specialOrder {
		if core.HasPrefix(input, tok) {
			return tok, t.special[tok], true
		}
	}
	return "", 0, false
}

func (t *Tokenizer) nextSpecialBoundary(input string) int {
	end := len(input)
	for _, tok := range t.specialOrder {
		if idx := indexIn(input, tok); idx > 0 && idx < end {
			end = idx
		}
	}
	return end
}

func normalizeSentencePieceSegment(segment string) string {
	if segment == "" {
		return ""
	}
	normalized := core.Replace(segment, " ", "▁")
	if !core.HasPrefix(normalized, "▁") {
		normalized = "▁" + normalized
	}
	return normalized
}

// spCacheKeyPrefix and gpt2CacheKeyPrefix namespace BPE cache entries. Kept
// byte-identical to the old tokenizerBPECacheKey(kind, …) layout ("kind"+"\x00"
// +text) so existing keys are unchanged.
const spCacheKeyPrefix = "sp\x00"
const gpt2CacheKeyPrefix = "gpt2\x00"

const sentencePieceMarker = "▁"

// keyScratchPool hands out reusable byte buffers for building a BPE cache key
// during the warm-cache lookup. The map lookup uses string(scratch), which the
// compiler resolves without allocating a string (the well-known m[string(b)]
// no-copy form), so a cache hit — the steady state once the per-segment cache
// is populated — touches the heap zero times. Pooled rather than a Tokenizer
// field because Encode is called concurrently (the cache is RWMutex-guarded).
var keyScratchPool = sync.Pool{New: func() any { b := make([]byte, 0, 256); return &b }}

// appendSentencePieceKey writes spCacheKeyPrefix + the normalised SentencePiece
// form of segment into dst[:0] and returns the grown buffer. The bytes are
// byte-identical to what sentencePieceCacheKey produces, so a key built here for
// the lookup matches a key stored on a miss. Caller must ensure segment != "".
func appendSentencePieceKey(dst []byte, segment string) []byte {
	dst = append(dst[:0], spCacheKeyPrefix...)
	// normalizeSentencePieceSegment prepends ▁ only when the post-replace text
	// does not already start with ▁. After replacement the first character is ▁
	// exactly when segment begins with a space (replaced) or with a literal ▁.
	if segment[0] != ' ' && !core.HasPrefix(segment, sentencePieceMarker) {
		dst = append(dst, sentencePieceMarker...)
	}
	for i := 0; i < len(segment); i++ {
		if segment[i] == ' ' {
			dst = append(dst, sentencePieceMarker...)
		} else {
			dst = append(dst, segment[i])
		}
	}
	return dst
}

// sentencePieceCacheKey returns the namespaced BPE cache key and the normalised
// SentencePiece text for segment in a SINGLE allocation. key is
// spCacheKeyPrefix+spText and spText is the zero-copy suffix key[len(prefix):].
// Called only on a cache MISS now — the warm-hit path builds the key into a
// pooled scratch buffer and never materialises these strings.
func sentencePieceCacheKey(segment string) (key, spText string) {
	if segment == "" {
		return "", ""
	}
	scratch := keyScratchPool.Get().(*[]byte)
	key = string(appendSentencePieceKey((*scratch)[:0], segment))
	keyScratchPool.Put(scratch)
	return key, key[len(spCacheKeyPrefix):]
}

// countByte counts occurrences of c in s without importing strings.
func countByte(s string, c byte) int {
	n := 0
	for i := 0; i < len(s); i++ {
		if s[i] == c {
			n++
		}
	}
	return n
}

// buildGPT2ByteMaps creates the GPT-2 byte-level BPE encoding/decoding maps.
// GPT-2 maps all 256 bytes to printable Unicode characters to avoid control chars
// in the vocabulary. Printable ASCII + Latin-1 Supplement map to themselves;
// everything else (0-32, 127-160, 173) maps to U+0100 onwards.
func buildGPT2ByteMaps() (decoder map[rune]byte, encoder map[byte]rune) {
	encoder = make(map[byte]rune, 256)
	decoder = make(map[rune]byte, 256)

	// Self-mapping ranges: printable ASCII + Latin-1 Supplement
	// Use int loop variable to avoid byte overflow at 255.
	selfMap := func(lo, hi int) {
		for b := lo; b <= hi; b++ {
			encoder[byte(b)] = rune(b)
			decoder[rune(b)] = byte(b)
		}
	}
	selfMap(33, 126)  // ! through ~
	selfMap(161, 172) // ¡ through ¬
	selfMap(174, 255) // ® through ÿ

	// Non-self-mapping: control chars, space, DEL, and gaps
	nonSelfMapped := 0
	for b := range 256 {
		if _, ok := encoder[byte(b)]; !ok {
			mappedRune := rune(256 + nonSelfMapped)
			encoder[byte(b)] = mappedRune
			decoder[mappedRune] = byte(b)
			nonSelfMapped++
		}
	}
	return
}

// appendGPT2Key writes gpt2CacheKeyPrefix + the GPT-2 byte-level encoding of
// segment (each byte mapped through gpt2Encoder, UTF-8 encoded) into dst[:0] and
// returns the grown buffer. Byte-identical to what gpt2CacheKey produces via the
// Builder, so a key built here for the lookup matches a key stored on a miss.
func (t *Tokenizer) appendGPT2Key(dst []byte, segment string) []byte {
	dst = append(dst[:0], gpt2CacheKeyPrefix...)
	for i := 0; i < len(segment); i++ {
		if r, ok := t.gpt2Encoder[segment[i]]; ok {
			dst = utf8.AppendRune(dst, r)
		}
	}
	return dst
}

// gpt2CacheKey returns the namespaced cache key and the byte-encoded text
// (zero-copy suffix key[len(prefix):]) in a single allocation. Called only on a
// cache MISS — the warm-hit path builds the key into a pooled scratch buffer.
func (t *Tokenizer) gpt2CacheKey(segment string) (key, encodedText string) {
	scratch := keyScratchPool.Get().(*[]byte)
	key = string(t.appendGPT2Key((*scratch)[:0], segment))
	keyScratchPool.Put(scratch)
	return key, key[len(gpt2CacheKeyPrefix):]
}

// bpeMerge applies BPE merges to a sequence of symbols until no more merges apply.
// Uses the standard algorithm: repeatedly find the lowest-rank adjacent pair and merge it.
func (t *Tokenizer) bpeMerge(symbols []string) []string {
	for len(symbols) > 1 {
		// Find the pair with the lowest merge rank.
		bestRank := -1
		bestIdx := -1
		for i := range len(symbols) - 1 {
			key := symbols[i] + " " + symbols[i+1]
			if rank, ok := t.mergeRanks[key]; ok {
				if bestRank < 0 || rank < bestRank {
					bestRank = rank
					bestIdx = i
				}
			}
		}
		if bestIdx < 0 {
			break // No more merges available.
		}
		// Merge the pair at bestIdx without allocating a replacement slice.
		symbols[bestIdx] += symbols[bestIdx+1]
		copy(symbols[bestIdx+1:], symbols[bestIdx+2:])
		symbols = symbols[:len(symbols)-1]
	}
	return symbols
}

func (t *Tokenizer) cachedBPETokens(key string) ([]int32, bool) {
	t.bpeCacheMu.RLock()
	defer t.bpeCacheMu.RUnlock()
	if len(t.bpeCache) == 0 {
		return nil, false
	}
	tokens, ok := t.bpeCache[key]
	return tokens, ok
}

// cachedBPETokensBytes is the zero-allocation lookup twin of cachedBPETokens:
// the map is indexed with string(key), which the compiler does WITHOUT copying
// key into a heap string. The warm-cache path uses this so a hit never
// allocates the namespaced key string just to discard it after the lookup.
func (t *Tokenizer) cachedBPETokensBytes(key []byte) ([]int32, bool) {
	t.bpeCacheMu.RLock()
	defer t.bpeCacheMu.RUnlock()
	if len(t.bpeCache) == 0 {
		return nil, false
	}
	tokens, ok := t.bpeCache[string(key)]
	return tokens, ok
}

func (t *Tokenizer) storeBPETokens(key string, tokens []int32) {
	if len(key) > tokenizerBPECacheMaxSegmentBytes || len(tokens) > tokenizerBPECacheMaxTokens {
		return
	}
	t.bpeCacheMu.Lock()
	defer t.bpeCacheMu.Unlock()
	if t.bpeCache == nil {
		t.bpeCache = make(map[string][]int32)
	}
	if _, ok := t.bpeCache[key]; ok {
		t.bpeCache[key] = append([]int32(nil), tokens...)
		return
	}
	for len(t.bpeCacheOrder) >= tokenizerBPECacheLimit {
		oldest := t.bpeCacheOrder[0]
		copy(t.bpeCacheOrder, t.bpeCacheOrder[1:])
		t.bpeCacheOrder = t.bpeCacheOrder[:len(t.bpeCacheOrder)-1]
		delete(t.bpeCache, oldest)
	}
	t.bpeCache[key] = append([]int32(nil), tokens...)
	t.bpeCacheOrder = append(t.bpeCacheOrder, key)
}

func (t *Tokenizer) shouldPrependBOS(text string) bool {
	if !t.hasBOS {
		return false
	}
	bosText := t.invVocab[t.bosToken]
	return bosText == "" || !core.HasPrefix(text, bosText)
}

func (t *Tokenizer) encodeSentencePieceSegment(segment string) []int32 {
	if segment == "" {
		return nil
	}
	// Warm-cache lookup with ZERO allocations: build the namespaced key into a
	// pooled scratch buffer and probe the cache via string(scratch), which the
	// compiler resolves without copying to the heap. Only on a miss do we
	// materialise the real key string (sentencePieceCacheKey) for storage. The
	// previous code allocated the key on EVERY call, including the warm hits
	// that dominate once the per-segment cache is populated.
	scratch := keyScratchPool.Get().(*[]byte)
	keyBytes := appendSentencePieceKey((*scratch)[:0], segment)
	cached, ok := t.cachedBPETokensBytes(keyBytes)
	*scratch = keyBytes
	keyScratchPool.Put(scratch)
	if ok {
		return cached
	}

	key, spText := sentencePieceCacheKey(segment)

	symbols := make([]string, 0, len(spText))
	for _, r := range spText {
		symbols = append(symbols, string(r))
	}
	symbols = t.bpeMerge(symbols)

	tokens := make([]int32, 0, len(symbols))
	for _, sym := range symbols {
		if id, ok := t.vocab[sym]; ok {
			tokens = append(tokens, id)
		}
	}
	t.storeBPETokens(key, tokens)
	return tokens
}

func (t *Tokenizer) encodeGPT2Segment(segment string) []int32 {
	if segment == "" {
		return nil
	}
	// Warm-cache lookup with ZERO allocations: build the namespaced key into a
	// pooled scratch buffer and probe via string(scratch) (the compiler's
	// no-copy m[string(b)] form). Only on a miss do we materialise the real key
	// string (gpt2CacheKey) for storage. The previous code allocated the encoded
	// key on EVERY call, including the warm hits that dominate once the cache is
	// populated.
	scratch := keyScratchPool.Get().(*[]byte)
	keyBytes := t.appendGPT2Key((*scratch)[:0], segment)
	empty := len(keyBytes) == len(gpt2CacheKeyPrefix)
	var cached []int32
	var ok bool
	if !empty {
		cached, ok = t.cachedBPETokensBytes(keyBytes)
	}
	*scratch = keyBytes
	keyScratchPool.Put(scratch)
	if empty {
		return nil
	}
	if ok {
		return cached
	}

	key, encodedText := t.gpt2CacheKey(segment)

	symbols := make([]string, 0, len(encodedText))
	for _, r := range encodedText {
		symbols = append(symbols, string(r))
	}
	symbols = t.bpeMerge(symbols)

	tokens := make([]int32, 0, len(symbols))
	for _, sym := range symbols {
		if id, ok := t.vocab[sym]; ok {
			tokens = append(tokens, id)
		}
	}
	t.storeBPETokens(key, tokens)
	return tokens
}

// Encode converts text to token IDs (prepends BOS token).
//
//	ids := tok.Encode("Hello world") // → []int32{2, 9906, 1917}
func (t *Tokenizer) Encode(text string) []int32 {
	if t.isGPT2BPE {
		return t.encodeGPT2(text)
	}

	tokens := make([]int32, 0, len(text)+1)
	if t.shouldPrependBOS(text) {
		tokens = append(tokens, t.bosToken)
	}

	// SentencePiece style: split into segments around special tokens, then BPE each segment.
	remaining := text
	for remaining != "" {
		// Check for special tokens at the current position.
		if tok, id, ok := t.matchSpecialToken(remaining); ok {
			tokens = append(tokens, id)
			remaining = remaining[len(tok):]
			continue
		}

		// Find the next special token boundary (or end of string).
		end := t.nextSpecialBoundary(remaining)
		segment := remaining[:end]
		remaining = remaining[end:]

		tokens = append(tokens, t.encodeSentencePieceSegment(segment)...)
	}

	return tokens
}

// encodeGPT2 encodes text using GPT-2 byte-level BPE.
func (t *Tokenizer) encodeGPT2(text string) []int32 {
	tokens := make([]int32, 0, len(text)+1)
	if t.shouldPrependBOS(text) {
		tokens = append(tokens, t.bosToken)
	}

	// Split text around special tokens (matched in original form, not byte-encoded).
	remaining := text
	for remaining != "" {
		// Check for special tokens at the current position.
		if tok, id, ok := t.matchSpecialToken(remaining); ok {
			tokens = append(tokens, id)
			remaining = remaining[len(tok):]
			continue
		}

		// Find the next special token boundary (or end of string).
		end := t.nextSpecialBoundary(remaining)
		segment := remaining[:end]
		remaining = remaining[end:]

		tokens = append(tokens, t.encodeGPT2Segment(segment)...)
	}

	return tokens
}

// Decode converts token IDs back to text (strips SentencePiece leading space).
//
//	text := tok.Decode([]int32{9906, 1917}) // → "Hello world"
func (t *Tokenizer) Decode(tokens []int32) string {
	if t.isGPT2BPE {
		var sb core.Builder
		for _, id := range tokens {
			if text, ok := t.invVocab[id]; ok {
				if _, isSpecial := t.special[text]; isSpecial {
					continue
				}
				sb.WriteString(text)
			}
		}
		return t.decodeGPT2Bytes(sb.String())
	}

	// SentencePiece: replace ▁→space WHILE building, then strip one leading
	// space, so the whole decode is a single allocation (Builder.String). The
	// previous code built raw with the Builder, then core.Replace rebuilt the
	// full string a second time to swap the marker — two full-length allocs of
	// the response for what one inline pass does. Value Builder (no *Builder
	// heap pointer); no precomputed Grow because that needs a second map-lookup
	// pass over tokens whose CPU cost outweighs the Builder's own growth.
	var sb core.Builder
	for _, id := range tokens {
		text, ok := t.invVocab[id]
		if !ok {
			continue
		}
		if _, isSpecial := t.special[text]; isSpecial {
			continue
		}
		writeSentencePieceReplaced(&sb, text)
	}
	result := sb.String()
	if core.HasPrefix(result, " ") {
		return result[1:]
	}
	return result
}

// writeSentencePieceReplaced writes text into sb with every SentencePiece
// marker ("▁", 3 bytes UTF-8) replaced by a single space, byte-identical to
// core.Replace(text, "▁", " ") but straight into the destination builder.
func writeSentencePieceReplaced(sb *core.Builder, text string) {
	for i := 0; i < len(text); {
		if i+len(sentencePieceMarker) <= len(text) && text[i:i+len(sentencePieceMarker)] == sentencePieceMarker {
			sb.WriteByte(' ')
			i += len(sentencePieceMarker)
			continue
		}
		sb.WriteByte(text[i])
		i++
	}
}

// DecodeToken converts a single token ID to text for streaming.
// Preserves the leading space (word boundary) for correct inter-token spacing.
//
//	text := tok.DecodeToken(1917) // → " world" (note leading space)
func (t *Tokenizer) DecodeToken(id int32) string {
	text, ok := t.invVocab[id]
	if !ok {
		return ""
	}
	if _, isSpecial := t.special[text]; isSpecial {
		return ""
	}

	if t.isGPT2BPE {
		return t.decodeGPT2Bytes(text)
	}

	// SentencePiece: replace with space but keep it (it's the word boundary)
	return core.Replace(text, "▁", " ")
}

// DecodeOne mirrors Decode([]int32{id}) semantics for a single token without
// allocating a one-element slice header at the call site. The hot path is the
// root-package Tokenizer.IDToken wrapper, which fires once per emitted
// generation token. Direct vocab lookup + leading-space strip replaces the
// allocation + Builder + final string() path that Decode([]int32{id}) would
// take.
//
//	text := tok.DecodeOne(1917) // → "world" (leading SP space stripped)
func (t *Tokenizer) DecodeOne(id int32) string {
	text, ok := t.invVocab[id]
	if !ok {
		return ""
	}
	if _, isSpecial := t.special[text]; isSpecial {
		return ""
	}

	if t.isGPT2BPE {
		return t.decodeGPT2Bytes(text)
	}

	// SentencePiece: replace ▁ with space, then strip a single leading space
	// to match Decode([]int32{id}) exactly. A solo "▁" therefore returns ""
	// — the root wrapper substitutes a bare space for that case from its
	// inverse-vocab fallback.
	//
	// Zero-allocation fast paths for the two cases that dominate per-token
	// streaming decode, both byte-identical to the Replace+strip fallback:
	//   • leading marker, none elsewhere ("▁hello") — the marker maps to the
	//     space that gets stripped, so the result is exactly the suffix after
	//     it. Return text[3:] (a slice, no heap).
	//   • no marker at all ("hello") — nothing to replace or strip, return
	//     text unchanged.
	// Any text with an interior or repeated marker (a real space → ▁ INSIDE
	// the piece) still needs the full Replace, so it falls through.
	const m = sentencePieceMarker // "▁", 3 bytes UTF-8
	if len(text) >= len(m) && text[:len(m)] == m {
		if indexIn(text[len(m):], m) < 0 {
			return text[len(m):]
		}
	} else if indexIn(text, m) < 0 {
		return text
	}
	result := core.Replace(text, "▁", " ")
	if core.HasPrefix(result, " ") {
		return result[1:]
	}
	return result
}

// decodeGPT2Bytes converts GPT-2 byte-level BPE Unicode back to real bytes.
// Fires once per emitted token on byte-level models (Qwen/GPT/Llama), so the
// buffer is sized up front: every rune maps to either one decoded byte or its
// own UTF-8 length, so the output is never longer than len(s) — one right-sized
// allocation, no append growth. utf8.AppendRune replaces []byte(string(r)) in
// the pass-through branch, which allocated an intermediate string per rune.
func (t *Tokenizer) decodeGPT2Bytes(s string) string {
	buf := make([]byte, 0, len(s))
	for _, r := range s {
		if b, ok := t.gpt2Decoder[r]; ok {
			buf = append(buf, b)
		} else {
			// Non-mapped runes pass through as UTF-8.
			buf = utf8.AppendRune(buf, r)
		}
	}
	return string(buf)
}

// BOSToken returns the beginning-of-sequence token ID.
func (t *Tokenizer) BOSToken() int32 { return t.bosToken }

// EOSToken returns the end-of-sequence (generation stop) token ID.
func (t *Tokenizer) EOSToken() int32 { return t.eosToken }

// HasBOSToken reports whether the tokenizer explicitly defines a BOS token.
func (t *Tokenizer) HasBOSToken() bool { return t != nil && t.hasBOS }

// HasEOSToken reports whether the tokenizer explicitly defines an EOS/stop token.
func (t *Tokenizer) HasEOSToken() bool { return t != nil && t.hasEOS }

// BOS returns the beginning-of-sequence token ID.
func (t *Tokenizer) BOS() int32 { return t.BOSToken() }

// EOS returns the end-of-sequence (generation stop) token ID.
func (t *Tokenizer) EOS() int32 { return t.EOSToken() }

// TokenID looks up a token string in the vocabulary.
func (t *Tokenizer) TokenID(text string) (int32, bool) {
	id, ok := t.vocab[text]
	return id, ok
}

// IDToken looks up the text for a token ID.
func (t *Tokenizer) IDToken(id int32) string {
	return t.invVocab[id]
}

// FormatGemmaPrompt applies the Gemma 3 chat template.
func FormatGemmaPrompt(prompt string) string {
	return core.Sprintf("<bos><start_of_turn>user\n%s<end_of_turn>\n<start_of_turn>model\n", prompt)
}
