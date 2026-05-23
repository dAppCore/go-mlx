// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"container/heap"
	"slices"
	"sync"

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
	mergeRanks   map[mergeKey]int
	special      map[string]int32
	specialOrder []string

	bosToken int32
	eosToken int32
	hasBOS   bool
	hasEOS   bool

	addPrefixSpace bool

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

type mergeKey struct {
	a string
	b string
}

type bpeNode struct {
	token   string
	prev    int
	next    int
	alive   bool
	version uint32
}

type bpeCandidate struct {
	rank         int
	left         int
	right        int
	leftVersion  uint32
	rightVersion uint32
}

type bpeCandidateHeap []bpeCandidate

func (h bpeCandidateHeap) Len() int {
	return len(h)
}

func (h bpeCandidateHeap) Less(i, j int) bool {
	if h[i].rank != h[j].rank {
		return h[i].rank < h[j].rank
	}
	return h[i].left < h[j].left
}

func (h bpeCandidateHeap) Swap(i, j int) {
	h[i], h[j] = h[j], h[i]
}

func (h *bpeCandidateHeap) Push(x any) {
	*h = append(*h, x.(bpeCandidate))
}

func (h *bpeCandidateHeap) Pop() any {
	old := *h
	n := len(old)
	item := old[n-1]
	*h = old[:n-1]
	return item
}

// tokenizerJSON is the HuggingFace tokenizer.json format.
type tokenizerJSON struct {
	Normalizer struct {
		Type    string `json:"type"`
		Content string `json:"content"`
	} `json:"normalizer"`
	PreTokenizer struct {
		Type     string `json:"type"`
		Behavior string `json:"behavior"`
	} `json:"pre_tokenizer"`
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
// Routes through core.Index — stdlib substring search uses Rabin-Karp /
// two-way under the hood, an order of magnitude faster than the naive
// O(n*m) byte-walk this used to do because every iteration constructed
// a fresh `s[i:i+subLen] == substr` slice header for comparison.
//
//	pos := indexIn("hello world", "world") // → 6
//	pos := indexIn("hello", "xyz")         // → -1
func indexIn(s, substr string) int {
	return core.Index(s, substr)
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
		vocab:          make(map[string]int32),
		invVocab:       make(map[int32]string),
		special:        make(map[string]int32),
		addPrefixSpace: true,
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

	tokenizer.mergeRanks = make(map[mergeKey]int, len(tokenizer.merges))
	for _, merge := range tokenizer.merges {
		tokenizer.mergeRanks[mergeKey{a: merge.a, b: merge.b}] = merge.rank
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
	if tj.Normalizer.Type == "Replace" && tj.Normalizer.Content == "▁" &&
		tj.PreTokenizer.Type == "Split" && tj.PreTokenizer.Behavior == "MergedWithPrevious" {
		tokenizer.addPrefixSpace = false
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
	// Gemma 4: <turn|> is the assistant turn stop token.
	if id, ok := tokenizer.special["<turn|>"]; ok {
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

func (t *Tokenizer) normalizeSentencePieceSegment(segment string) string {
	if segment == "" {
		return ""
	}
	// Decide upfront whether we need the leading ▁ prefix. The original
	// code called Replace first (allocating a new string), then checked
	// the result for "▁" prefix, then prefixed it (a SECOND alloc). Both
	// can be merged into a single Builder pass:
	//
	//   - Count spaces to compute exact output size (▁ is 3 bytes, ' ' is
	//     1, so each space adds 2 bytes to the output length).
	//   - Decide prefix decision up front: needs ▁ iff addPrefixSpace AND
	//     the segment's first byte is not the ▁-leader (E2). The latter
	//     test is a single byte compare instead of HasPrefix walking 3.
	//   - If no work needed (no spaces, no prefix), return segment as-is
	//     — zero allocations, the input string passes through directly.
	needPrefix := t.addPrefixSpace
	if needPrefix && segment[0] == 0xE2 && len(segment) >= 3 &&
		segment[1] == 0x96 && segment[2] == 0x81 {
		needPrefix = false
	}

	// Count spaces — also tells us if Replace work is needed.
	spaces := 0
	for i := 0; i < len(segment); i++ {
		if segment[i] == ' ' {
			spaces++
		}
	}

	if !needPrefix && spaces == 0 {
		return segment
	}

	// Output size known exactly: prefix (3) + segment + 2 per space.
	outLen := len(segment) + 2*spaces
	if needPrefix {
		outLen += 3
	}
	buf := make([]byte, 0, outLen)
	if needPrefix {
		buf = append(buf, 0xE2, 0x96, 0x81)
	}
	for i := 0; i < len(segment); i++ {
		b := segment[i]
		if b == ' ' {
			buf = append(buf, 0xE2, 0x96, 0x81)
			continue
		}
		buf = append(buf, b)
	}
	return core.AsString(buf)
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

// bpeMerge applies BPE merges to a sequence of symbols until no more merges apply.
// Uses the standard algorithm: repeatedly find the lowest-rank adjacent pair and merge it.
func (t *Tokenizer) bpeMerge(symbols []string) []string {
	if len(symbols) <= 1 || len(t.mergeRanks) == 0 {
		return symbols
	}

	nodes := make([]bpeNode, len(symbols))
	for i, sym := range symbols {
		nodes[i] = bpeNode{
			token: sym,
			prev:  i - 1,
			next:  i + 1,
			alive: true,
		}
	}
	nodes[len(nodes)-1].next = -1

	candidates := make(bpeCandidateHeap, 0, len(nodes)-1)
	pushPair := func(left int) {
		if left < 0 || left >= len(nodes) || !nodes[left].alive {
			return
		}
		right := nodes[left].next
		if right < 0 || right >= len(nodes) || !nodes[right].alive {
			return
		}
		rank, ok := t.mergeRanks[mergeKey{a: nodes[left].token, b: nodes[right].token}]
		if !ok {
			return
		}
		heap.Push(&candidates, bpeCandidate{
			rank:         rank,
			left:         left,
			right:        right,
			leftVersion:  nodes[left].version,
			rightVersion: nodes[right].version,
		})
	}
	for i := 0; i < len(nodes)-1; i++ {
		pushPair(i)
	}
	heap.Init(&candidates)

	for candidates.Len() > 0 {
		candidate := heap.Pop(&candidates).(bpeCandidate)
		left, right := candidate.left, candidate.right
		if left < 0 || right < 0 || left >= len(nodes) || right >= len(nodes) {
			continue
		}
		if !nodes[left].alive || !nodes[right].alive || nodes[left].next != right || nodes[right].prev != left {
			continue
		}
		if nodes[left].version != candidate.leftVersion || nodes[right].version != candidate.rightVersion {
			continue
		}
		if rank, ok := t.mergeRanks[mergeKey{a: nodes[left].token, b: nodes[right].token}]; !ok || rank != candidate.rank {
			continue
		}

		nodes[left].token += nodes[right].token
		nodes[left].next = nodes[right].next
		nodes[left].version++
		nodes[right].alive = false
		nodes[right].version++
		if next := nodes[right].next; next >= 0 {
			nodes[next].prev = left
		}

		pushPair(nodes[left].prev)
		pushPair(left)
	}

	merged := symbols[:0]
	for i := 0; i >= 0; i = nodes[i].next {
		merged = append(merged, nodes[i].token)
	}
	return merged
}

func tokenizerBPECacheKey(kind, segment string) string {
	return kind + "\x00" + segment
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

func (t *Tokenizer) encodeSentencePieceSegment(segment string) []int32 {
	spText := t.normalizeSentencePieceSegment(segment)
	if spText == "" {
		return nil
	}
	key := tokenizerBPECacheKey("sp", spText)
	if cached, ok := t.cachedBPETokens(key); ok {
		return cached
	}

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
	encoded := core.NewBuilder()
	for _, b := range []byte(segment) {
		if r, ok := t.gpt2Encoder[b]; ok {
			encoded.WriteRune(r)
		}
	}
	encodedText := encoded.String()
	if encodedText == "" {
		return nil
	}
	key := tokenizerBPECacheKey("gpt2", encodedText)
	if cached, ok := t.cachedBPETokens(key); ok {
		return cached
	}

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

func (t *Tokenizer) shouldPrependBOS(text string) bool {
	if !t.hasBOS {
		return false
	}
	bosText := t.invVocab[t.bosToken]
	return bosText == "" || !core.HasPrefix(text, bosText)
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
	// GPT-2 byte-level path is handled by walking the raw concatenation
	// through decodeGPT2Bytes — the byte-level decoder strips its own
	// envelope, so the SentencePiece ▁-translation must NOT run on it.
	if t.isGPT2BPE {
		sb := core.NewBuilder()
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

	// SentencePiece path — translate ▁ → space inline while assembling,
	// then strip the single leading space (the prefix-space marker on
	// the first emitted token). Replaces the prior triple walk:
	//   1) Builder.WriteString accumulation → raw
	//   2) core.Replace(raw, "▁", " ")      → result (new alloc)
	//   3) HasPrefix(" ") + slice           → leading-space strip
	// with a single Builder pass that splits on ▁ via indexBytePrefix —
	// the fast-path for tokens without ▁ falls into a single WriteString
	// (memmove), and the only translation work is per-▁-occurrence.
	//
	// A pre-sizing pass (Grow on summed-text length) was tried and
	// reverted — the second map-walk cost outweighs the saved geometric
	// reallocs at every shape from 3 to 64 tokens. Builder's default
	// growth strategy wins here.
	sb := core.NewBuilder()
	for _, id := range tokens {
		text, ok := t.invVocab[id]
		if !ok {
			continue
		}
		if _, isSpecial := t.special[text]; isSpecial {
			continue
		}
		// Bulk-write tokens without ▁ (common case — most vocab tokens
		// are leaf-bytes or non-prefixed merges).
		for {
			idx := indexBytePrefix(text)
			if idx < 0 {
				sb.WriteString(text)
				break
			}
			if idx > 0 {
				sb.WriteString(text[:idx])
			}
			sb.WriteByte(' ')
			text = text[idx+3:]
			if text == "" {
				break
			}
		}
	}
	out := sb.String()
	if len(out) > 0 && out[0] == ' ' {
		return out[1:]
	}
	return out
}

// indexBytePrefix returns the byte offset of the SentencePiece ▁
// marker (U+2581, E2 96 81) in s, or -1 if absent. Inlined so Decode's
// inner loop can branch on a simple int compare instead of the more
// general core.Index three-byte-string-needle call.
func indexBytePrefix(s string) int {
	for i := 0; i+2 < len(s); i++ {
		if s[i] == 0xE2 && s[i+1] == 0x96 && s[i+2] == 0x81 {
			return i
		}
	}
	// Trailing 2 bytes can't contain the 3-byte marker.
	return -1
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
	result := core.Replace(text, "▁", " ")
	if core.HasPrefix(result, " ") {
		return result[1:]
	}
	return result
}

// decodeGPT2Bytes converts GPT-2 byte-level BPE Unicode back to real bytes.
func (t *Tokenizer) decodeGPT2Bytes(s string) string {
	if s == "" {
		return ""
	}
	// Pre-size to the input byte length — GPT-2 maps every rune to exactly
	// one byte (the encoder covers all 256 source bytes), so output bytes
	// ≤ input bytes (every multi-byte rune collapses to 1 byte; ASCII
	// runes stay 1:1). One allocation, no geometric growth.
	//
	// AsString wraps the freshly built buffer in a zero-copy string view —
	// the prior `string(buf)` did a full copy.
	buf := make([]byte, 0, len(s))
	for _, r := range s {
		if b, ok := t.gpt2Decoder[r]; ok {
			buf = append(buf, b)
			continue
		}
		// Non-mapped runes pass through as UTF-8. Encode the rune
		// directly into buf to avoid the intermediate `[]byte(string(r))`
		// double allocation. utf8.EncodeRune writes up to 4 bytes; grow
		// buf inline rather than detouring through a per-rune string.
		var enc [4]byte
		n := utf8EncodeRune(enc[:], r)
		buf = append(buf, enc[:n]...)
	}
	return core.AsString(buf)
}

// utf8EncodeRune writes the UTF-8 encoding of r into p (which must be
// at least 4 bytes) and returns the byte count. Inlined alternative to
// importing unicode/utf8 in this file — the only caller is
// decodeGPT2Bytes's non-mapped-rune fallback, which is effectively
// unreachable for valid GPT-2 input (the encoder maps all 256 source
// bytes) but kept as a safety net.
func utf8EncodeRune(p []byte, r rune) int {
	switch {
	case r < 0x80:
		p[0] = byte(r)
		return 1
	case r < 0x800:
		p[0] = 0xC0 | byte(r>>6)
		p[1] = 0x80 | (byte(r) & 0x3F)
		return 2
	case r < 0x10000:
		p[0] = 0xE0 | byte(r>>12)
		p[1] = 0x80 | (byte(r>>6) & 0x3F)
		p[2] = 0x80 | (byte(r) & 0x3F)
		return 3
	default:
		p[0] = 0xF0 | byte(r>>18)
		p[1] = 0x80 | (byte(r>>12) & 0x3F)
		p[2] = 0x80 | (byte(r>>6) & 0x3F)
		p[3] = 0x80 | (byte(r) & 0x3F)
		return 4
	}
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
	return core.Sprintf("<start_of_turn>user\n%s<end_of_turn>\n<start_of_turn>model\n", prompt)
}
