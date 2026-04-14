//go:build darwin && arm64

package metal

import (
	"dappco.re/go/core"

	coreio "dappco.re/go/core/io"
)

// Tokenizer handles text-to-token and token-to-text conversion.
type Tokenizer struct {
	vocab      map[string]int32
	invVocab   map[int32]string
	merges     []mergePair
	mergeRanks map[string]int // "a b" → rank for O(1) merge lookup
	special    map[string]int32

	bosToken int32
	eosToken int32

	// GPT-2 byte-level BPE support (used by Qwen, GPT, Llama, etc.)
	isGPT2BPE   bool
	gpt2Decoder map[rune]byte // Unicode char → original byte
	gpt2Encoder map[byte]rune // original byte → Unicode char
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
	}
	if id, ok := tokenizer.special["<eos>"]; ok {
		tokenizer.eosToken = id
	}
	// Gemma: <end_of_turn> is the generation stop token
	if id, ok := tokenizer.special["<end_of_turn>"]; ok {
		tokenizer.eosToken = id
	}
	// Qwen3: <|im_end|> is the generation stop token
	if id, ok := tokenizer.special["<|im_end|>"]; ok {
		tokenizer.eosToken = id
	}
	// Qwen3 BOS: <|im_start|>
	if id, ok := tokenizer.special["<|im_start|>"]; ok {
		tokenizer.bosToken = id
	}
	// Llama 3: <|eot_id|> is the turn-end token
	if id, ok := tokenizer.special["<|eot_id|>"]; ok {
		tokenizer.eosToken = id
	}
	// Llama 3 BOS: <|begin_of_text|>
	if id, ok := tokenizer.special["<|begin_of_text|>"]; ok {
		tokenizer.bosToken = id
	}

	return tokenizer, nil
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
		// Merge the pair at bestIdx.
		merged := symbols[bestIdx] + symbols[bestIdx+1]
		symbols = append(symbols[:bestIdx], append([]string{merged}, symbols[bestIdx+2:]...)...)
	}
	return symbols
}

// Encode converts text to token IDs (prepends BOS token).
//
//	ids := tok.Encode("Hello world") // → []int32{2, 9906, 1917}
func (t *Tokenizer) Encode(text string) []int32 {
	if t.isGPT2BPE {
		return t.encodeGPT2(text)
	}

	tokens := []int32{t.bosToken}

	// SentencePiece style: split into segments around special tokens, then BPE each segment.
	remaining := text
	for remaining != "" {
		// Check for special tokens at the current position.
		found := false
		for tok, id := range t.special {
			if core.HasPrefix(remaining, tok) {
				tokens = append(tokens, id)
				remaining = remaining[len(tok):]
				found = true
				break
			}
		}
		if found {
			continue
		}

		// Find the next special token boundary (or end of string).
		end := len(remaining)
		for tok := range t.special {
			if idx := indexIn(remaining, tok); idx > 0 && idx < end {
				end = idx
			}
		}
		segment := remaining[:end]
		remaining = remaining[end:]

		// SentencePiece: prefix the segment with ▁ (space marker) and split into characters.
		spText := "▁" + segment
		symbols := make([]string, 0, len([]rune(spText)))
		for _, r := range spText {
			symbols = append(symbols, string(r))
		}

		// Apply BPE merges.
		symbols = t.bpeMerge(symbols)

		// Look up merged symbols in vocab.
		for _, sym := range symbols {
			if id, ok := t.vocab[sym]; ok {
				tokens = append(tokens, id)
			}
		}
	}

	return tokens
}

// encodeGPT2 encodes text using GPT-2 byte-level BPE.
func (t *Tokenizer) encodeGPT2(text string) []int32 {
	tokens := []int32{t.bosToken}

	// Split text around special tokens (matched in original form, not byte-encoded).
	remaining := text
	for remaining != "" {
		// Check for special tokens at the current position.
		found := false
		for tok, id := range t.special {
			if core.HasPrefix(remaining, tok) {
				tokens = append(tokens, id)
				remaining = remaining[len(tok):]
				found = true
				break
			}
		}
		if found {
			continue
		}

		// Find the next special token boundary (or end of string).
		end := len(remaining)
		for tok := range t.special {
			if idx := indexIn(remaining, tok); idx > 0 && idx < end {
				end = idx
			}
		}
		segment := remaining[:end]
		remaining = remaining[end:]

		// Convert segment bytes to GPT-2 Unicode representation.
		encoded := core.NewBuilder()
		for _, b := range []byte(segment) {
			if r, ok := t.gpt2Encoder[b]; ok {
				encoded.WriteRune(r)
			}
		}

		// Split into individual runes (GPT-2 BPE operates on Unicode chars).
		runes := []rune(encoded.String())
		symbols := make([]string, len(runes))
		for i, r := range runes {
			symbols[i] = string(r)
		}

		// Apply BPE merges.
		symbols = t.bpeMerge(symbols)

		// Look up merged symbols in vocab.
		for _, sym := range symbols {
			if id, ok := t.vocab[sym]; ok {
				tokens = append(tokens, id)
			}
		}
	}

	return tokens
}

// Decode converts token IDs back to text (strips SentencePiece leading space).
//
//	text := tok.Decode([]int32{9906, 1917}) // → "Hello world"
func (t *Tokenizer) Decode(tokens []int32) string {
	sb := core.NewBuilder()
	for _, id := range tokens {
		if text, ok := t.invVocab[id]; ok {
			// Skip special tokens in decode output
			if _, isSpecial := t.special[text]; isSpecial {
				continue
			}
			sb.WriteString(text)
		}
	}
	raw := sb.String()

	if t.isGPT2BPE {
		return t.decodeGPT2Bytes(raw)
	}

	// SentencePiece style
	result := core.Replace(raw, "▁", " ")
	if core.HasPrefix(result, " ") {
		result = result[1:]
	}
	return result
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

// decodeGPT2Bytes converts GPT-2 byte-level BPE Unicode back to real bytes.
func (t *Tokenizer) decodeGPT2Bytes(s string) string {
	var buf []byte
	for _, r := range s {
		if b, ok := t.gpt2Decoder[r]; ok {
			buf = append(buf, b)
		} else {
			// Non-mapped runes pass through as UTF-8
			buf = append(buf, []byte(string(r))...)
		}
	}
	return string(buf)
}

// BOSToken returns the beginning-of-sequence token ID.
func (t *Tokenizer) BOSToken() int32 { return t.bosToken }

// EOSToken returns the end-of-sequence (generation stop) token ID.
func (t *Tokenizer) EOSToken() int32 { return t.eosToken }

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
