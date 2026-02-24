//go:build darwin && arm64

package metal

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
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
		Type         string          `json:"type"`
		Vocab        json.RawMessage `json:"vocab"`
		Merges       json.RawMessage `json:"merges"`
		ByteFallback bool            `json:"byte_fallback"`
	} `json:"model"`
	AddedTokens []struct {
		ID      int32  `json:"id"`
		Content string `json:"content"`
		Special bool   `json:"special"`
	} `json:"added_tokens"`
}

// LoadTokenizer reads a tokenizer.json file and creates a Tokenizer.
func LoadTokenizer(path string) (*Tokenizer, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("tokenizer: read %s: %w", path, err)
	}

	var tj tokenizerJSON
	if err := json.Unmarshal(data, &tj); err != nil {
		return nil, fmt.Errorf("tokenizer: parse: %w", err)
	}

	t := &Tokenizer{
		vocab:    make(map[string]int32),
		invVocab: make(map[int32]string),
		special:  make(map[string]int32),
	}

	// Parse vocab
	var vocab map[string]int32
	if err := json.Unmarshal(tj.Model.Vocab, &vocab); err != nil {
		return nil, fmt.Errorf("tokenizer: parse vocab: %w", err)
	}
	t.vocab = vocab
	for k, v := range vocab {
		t.invVocab[v] = k
	}

	// Parse merges — supports both ["a b", ...] and [["a","b"], ...] formats
	if len(tj.Model.Merges) > 0 {
		var stringMerges []string
		if err := json.Unmarshal(tj.Model.Merges, &stringMerges); err == nil {
			for rank, merge := range stringMerges {
				parts := strings.SplitN(merge, " ", 2)
				if len(parts) == 2 {
					t.merges = append(t.merges, mergePair{a: parts[0], b: parts[1], rank: rank})
				}
			}
		} else {
			var arrayMerges [][]string
			if err := json.Unmarshal(tj.Model.Merges, &arrayMerges); err == nil {
				for rank, pair := range arrayMerges {
					if len(pair) == 2 {
						t.merges = append(t.merges, mergePair{a: pair[0], b: pair[1], rank: rank})
					}
				}
			}
		}
	}

	// Build merge rank lookup for BPE.
	t.mergeRanks = make(map[string]int, len(t.merges))
	for _, m := range t.merges {
		t.mergeRanks[m.a+" "+m.b] = m.rank
	}

	// Parse special tokens
	for _, tok := range tj.AddedTokens {
		if tok.Special {
			t.special[tok.Content] = tok.ID
		}
		t.vocab[tok.Content] = tok.ID
		t.invVocab[tok.ID] = tok.Content
	}

	// Detect GPT-2 byte-level BPE (Qwen, GPT, DeepSeek use Ġ for space).
	// Check for "Ġthe" rather than bare "Ġ" — large SentencePiece vocabs
	// (Gemma3 262K) may include Ġ as an obscure character without using
	// GPT-2 byte encoding.
	if _, ok := t.vocab["Ġthe"]; ok {
		t.isGPT2BPE = true
		t.gpt2Decoder, t.gpt2Encoder = buildGPT2ByteMaps()
	}

	// Set BOS/EOS — detect model family from special tokens
	if id, ok := t.special["<bos>"]; ok {
		t.bosToken = id
	}
	if id, ok := t.special["<eos>"]; ok {
		t.eosToken = id
	}
	// Gemma: <end_of_turn> is the generation stop token
	if id, ok := t.special["<end_of_turn>"]; ok {
		t.eosToken = id
	}
	// Qwen3: <|im_end|> is the generation stop token
	if id, ok := t.special["<|im_end|>"]; ok {
		t.eosToken = id
	}
	// Qwen3 BOS: <|im_start|>
	if id, ok := t.special["<|im_start|>"]; ok {
		t.bosToken = id
	}
	// Llama 3: <|eot_id|> is the turn-end token
	if id, ok := t.special["<|eot_id|>"]; ok {
		t.eosToken = id
	}
	// Llama 3 BOS: <|begin_of_text|>
	if id, ok := t.special["<|begin_of_text|>"]; ok {
		t.bosToken = id
	}

	return t, nil
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
	n := 0
	for b := range 256 {
		if _, ok := encoder[byte(b)]; !ok {
			r := rune(256 + n)
			encoder[byte(b)] = r
			decoder[r] = byte(b)
			n++
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

// Encode converts text to token IDs. Prepends BOS token.
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
			if strings.HasPrefix(remaining, tok) {
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
			if idx := strings.Index(remaining, tok); idx > 0 && idx < end {
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
			if strings.HasPrefix(remaining, tok) {
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
			if idx := strings.Index(remaining, tok); idx > 0 && idx < end {
				end = idx
			}
		}
		segment := remaining[:end]
		remaining = remaining[end:]

		// Convert segment bytes to GPT-2 Unicode representation.
		var encoded strings.Builder
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

// Decode converts token IDs back to text.
// For full-sequence decoding, the SentencePiece leading space is stripped.
func (t *Tokenizer) Decode(tokens []int32) string {
	var sb strings.Builder
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
	result := strings.ReplaceAll(raw, "▁", " ")
	if strings.HasPrefix(result, " ") {
		result = result[1:]
	}
	return result
}

// DecodeToken converts a single token ID to text for streaming.
// Unlike Decode, it preserves the leading space (word boundary) so that
// token-by-token output maintains correct spacing between words.
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
	return strings.ReplaceAll(text, "▁", " ")
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

// EOSToken returns the end-of-sequence token ID.
func (t *Tokenizer) EOSToken() int32 { return t.eosToken }

// FormatGemmaPrompt applies the Gemma 3 chat template.
func FormatGemmaPrompt(prompt string) string {
	return fmt.Sprintf("<start_of_turn>user\n%s<end_of_turn>\n<start_of_turn>model\n", prompt)
}
