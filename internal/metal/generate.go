//go:build darwin && arm64

package metal

import (
	"context"
	"fmt"
	"iter"
	"slices"
	"strings"
	"time"
)

// Token represents a single generated token.
type Token struct {
	ID   int32
	Text string
}

// ChatMessage represents a chat turn.
type ChatMessage struct {
	Role    string
	Content string
}

// GenerateConfig holds generation parameters.
type GenerateConfig struct {
	MaxTokens     int
	Temperature   float32
	TopK          int
	TopP          float32
	StopTokens    []int32
	RepeatPenalty float32
}

// Metrics holds performance metrics from the last inference operation.
type Metrics struct {
	PromptTokens        int
	GeneratedTokens     int
	PrefillDuration     time.Duration
	DecodeDuration      time.Duration
	TotalDuration       time.Duration
	PrefillTokensPerSec float64
	DecodeTokensPerSec  float64
	PeakMemoryBytes     uint64
	ActiveMemoryBytes   uint64
}

// Model wraps a loaded transformer model for text generation.
type Model struct {
	model       InternalModel
	tokenizer   *Tokenizer
	modelType   string
	contextLen  int // 0 = unbounded (model default)
	lastErr     error
	lastMetrics Metrics
}

// ModelType returns the architecture identifier (e.g. "gemma3", "qwen3").
func (m *Model) ModelType() string { return m.modelType }

// Err returns the error from the last Generate/Chat call, if any.
func (m *Model) Err() error { return m.lastErr }

// LastMetrics returns performance metrics from the last inference operation.
func (m *Model) LastMetrics() Metrics { return m.lastMetrics }

// ModelInfo holds metadata about a loaded model.
type ModelInfo struct {
	Architecture string
	VocabSize    int
	NumLayers    int
	HiddenSize   int
	QuantBits    int
	QuantGroup   int
}

// Info returns metadata about the loaded model.
func (m *Model) Info() ModelInfo {
	info := ModelInfo{
		Architecture: m.modelType,
		NumLayers:    m.model.NumLayers(),
	}
	switch v := m.model.(type) {
	case *GemmaModel:
		info.VocabSize = int(v.Cfg.VocabSize)
		info.HiddenSize = int(v.Cfg.HiddenSize)
		if v.Cfg.Quantization != nil {
			info.QuantBits = v.Cfg.Quantization.Bits
			info.QuantGroup = v.Cfg.Quantization.GroupSize
		}
	case *Qwen3Model:
		info.VocabSize = int(v.Cfg.VocabSize)
		info.HiddenSize = int(v.Cfg.HiddenSize)
		if v.Cfg.Quantization != nil {
			info.QuantBits = v.Cfg.Quantization.Bits
			info.QuantGroup = v.Cfg.Quantization.GroupSize
		}
	}
	return info
}

// Close releases all model weight arrays and associated resources.
// After Close, the Model must not be used for generation.
func (m *Model) Close() error {
	if m.model == nil {
		return nil
	}
	switch v := m.model.(type) {
	case *GemmaModel:
		closeGemma(v)
	case *Qwen3Model:
		closeQwen3(v)
	}
	m.model = nil
	m.tokenizer = nil
	return nil
}

// Chat formats messages using the model's native template, then generates.
func (m *Model) Chat(ctx context.Context, messages []ChatMessage, cfg GenerateConfig) iter.Seq[Token] {
	prompt := m.formatChat(messages)
	return m.Generate(ctx, prompt, cfg)
}

// Generate streams tokens for the given prompt.
//
// Each call allocates fresh KV caches that are released to GC when the iterator
// completes. For multi-turn chat, call [ClearCache] between turns to reclaim
// Metal memory promptly rather than waiting for GC finalisers.
func (m *Model) Generate(ctx context.Context, prompt string, cfg GenerateConfig) iter.Seq[Token] {
	m.lastErr = nil
	m.lastMetrics = Metrics{}

	return func(yield func(Token) bool) {
		totalStart := time.Now()
		ResetPeakMemory()

		tokens := m.tokenizer.Encode(prompt)
		promptLen := len(tokens)
		caches := m.newCaches()
		defer freeCaches(caches)

		sampler := newSampler(cfg.Temperature, cfg.TopP, 0, cfg.TopK)
		var genCount int
		var prefillDur time.Duration

		defer func() {
			decodeDur := time.Since(totalStart) - prefillDur
			totalDur := time.Since(totalStart)
			m.lastMetrics = Metrics{
				PromptTokens:      promptLen,
				GeneratedTokens:   genCount,
				PrefillDuration:   prefillDur,
				DecodeDuration:    decodeDur,
				TotalDuration:     totalDur,
				PeakMemoryBytes:   GetPeakMemory(),
				ActiveMemoryBytes: GetActiveMemory(),
			}
			if prefillDur > 0 {
				m.lastMetrics.PrefillTokensPerSec = float64(promptLen) / prefillDur.Seconds()
			}
			if decodeDur > 0 {
				m.lastMetrics.DecodeTokensPerSec = float64(genCount) / decodeDur.Seconds()
			}
		}()

		// Prefill: process entire prompt
		prefillStart := time.Now()
		vInput := FromValues(tokens, len(tokens))
		input := Reshape(vInput, 1, int32(len(tokens)))
		logits := m.model.Forward(input, caches)
		Free(vInput, input)

		if err := Eval(logits); err != nil {
			m.lastErr = fmt.Errorf("prefill: %w", err)
			return
		}
		prefillDur = time.Since(prefillStart)

		// Track generated token IDs for repeat penalty.
		var history []int32

		defer func() {
			Free(logits)
		}()

		for i := 0; i < cfg.MaxTokens; i++ {
			select {
			case <-ctx.Done():
				m.lastErr = ctx.Err()
				return
			default:
			}

			// Sample from last position logits
			l1 := SliceAxis(logits, 1, int32(logits.Dim(1)-1), int32(logits.Dim(1)))
			lastPos := Reshape(l1, 1, int32(l1.Dim(2)))
			Free(l1)

			// Apply repeat penalty before sampling.
			if cfg.RepeatPenalty > 1.0 && len(history) > 0 {
				oldLastPos := lastPos
				lastPos = applyRepeatPenalty(lastPos, history, cfg.RepeatPenalty)
				Free(oldLastPos)
			}

			next := sampler.Sample(lastPos)
			if err := Eval(next); err != nil {
				m.lastErr = fmt.Errorf("sample step %d: %w", i, err)
				Free(lastPos, next)
				return
			}

			id := int32(next.Int())
			history = append(history, id)
			Free(lastPos)

			// Check stop conditions
			if id == m.tokenizer.EOSToken() {
				Free(next)
				return
			}
			if slices.Contains(cfg.StopTokens, id) {
				Free(next)
				return
			}

			genCount++
			text := m.tokenizer.DecodeToken(id)
			if !yield(Token{ID: id, Text: text}) {
				Free(next)
				return
			}
			Free(next)

			// Next step input
			vNextInput := FromValues([]int32{id}, 1)
			nextInput := Reshape(vNextInput, 1, 1)
			Free(vNextInput)

			oldLogits := logits
			logits = m.model.Forward(nextInput, caches)
			Free(nextInput, oldLogits)

			if err := Eval(logits); err != nil {
				m.lastErr = fmt.Errorf("decode step %d: %w", i, err)
				return
			}
		}
	}
}

// InspectAttention runs a single prefill pass and extracts K vectors from each layer's KV cache.
// Returns the post-RoPE K tensors as CPU float32 slices indexed [layer][head][seq_len*head_dim].
func (m *Model) InspectAttention(ctx context.Context, prompt string) (*AttentionResult, error) {
	tokens := m.tokenizer.Encode(prompt)
	if len(tokens) == 0 {
		return nil, fmt.Errorf("empty prompt after tokenisation")
	}

	caches := m.newCaches()
	defer freeCaches(caches)

	// Single prefill pass — populates KV caches for all layers.
	input := FromValues(tokens, len(tokens))
	input = Reshape(input, 1, int32(len(tokens)))
	logits := m.model.Forward(input, caches)
	if err := Eval(logits); err != nil {
		return nil, fmt.Errorf("prefill: %w", err)
	}

	info := m.Info()
	seqLen := len(tokens)

	// Extract K vectors from each layer's cache.
	keys := make([][][]float32, info.NumLayers)
	var numHeads, headDim int

	for i := 0; i < info.NumLayers && i < len(caches); i++ {
		state := caches[i].State()
		if len(state) < 1 {
			continue
		}
		kArray := state[0] // K tensor from cache: [B, H, L_alloc, D]
		shape := kArray.Shape()
		if len(shape) != 4 {
			continue
		}
		numHeads = int(shape[1])
		headDim = int(shape[3])

		// Slice to valid tokens only (cache may have pre-allocated padding).
		validLen := min(caches[i].Len(), seqLen)
		kSliced := Slice(kArray, []int32{0, 0, 0, 0}, []int32{shape[0], shape[1], int32(validLen), shape[3]})
		if err := Eval(kSliced); err != nil {
			Free(kSliced)
			continue
		}

		// Extract all floats then reshape per head.
		flat := kSliced.Floats() // len = 1 * H * validLen * D
		Free(kSliced)

		keys[i] = make([][]float32, numHeads)
		stride := validLen * headDim
		for h := 0; h < numHeads; h++ {
			start := h * stride
			end := start + stride
			if end > len(flat) {
				break
			}
			head := make([]float32, stride)
			copy(head, flat[start:end])
			keys[i][h] = head
		}
	}

	return &AttentionResult{
		NumLayers:    info.NumLayers,
		NumHeads:     numHeads,
		SeqLen:       seqLen,
		HeadDim:      headDim,
		Keys:         keys,
		Architecture: info.Architecture,
	}, nil
}

// AttentionResult holds extracted K vectors from the KV cache.
type AttentionResult struct {
	NumLayers    int
	NumHeads     int
	SeqLen       int
	HeadDim      int
	Keys         [][][]float32 // [layer][head] → flat float32 of len seq_len*head_dim
	Architecture string
}

// applyRepeatPenalty modifies logits to discourage repeated tokens.
// For each unique token ID in history: positive logits are divided by penalty,
// negative logits are multiplied by penalty. Both make the token less likely.
func applyRepeatPenalty(logits *Array, history []int32, penalty float32) *Array {
	// Deduplicate history to get unique token IDs.
	seen := make(map[int32]bool, len(history))
	var indices []int32
	for _, id := range history {
		if !seen[id] {
			seen[id] = true
			indices = append(indices, id)
		}
	}

	idx := FromValues(indices, 1, len(indices))
	gathered := TakeAlongAxis(logits, idx, -1)

	zero := FromValue(float32(0))
	invPenalty := FromValue(1.0 / penalty)
	penaltyVal := FromValue(penalty)

	// Positive logits: divide by penalty. Negative logits: multiply by penalty.
	gt := Greater(gathered, zero)
	m1 := Mul(gathered, invPenalty)
	m2 := Mul(gathered, penaltyVal)
	penalised := Where(gt, m1, m2)
	Free(gt, m1, m2)

	res := PutAlongAxis(logits, idx, penalised, -1)
	Free(idx, gathered, zero, invPenalty, penaltyVal, penalised)
	return res
}

// newCaches creates per-layer KV caches. If contextLen is set, all unbounded
// caches are replaced with RotatingKVCache to cap memory usage.
func (m *Model) newCaches() []Cache {
	caches := m.model.NewCache()
	if m.contextLen <= 0 {
		return caches
	}
	for i, c := range caches {
		// Only replace unbounded caches — rotating caches already have a limit.
		if _, ok := c.(*KVCache); ok {
			caches[i] = NewRotatingKVCache(m.contextLen)
		}
	}
	return caches
}

// formatChat applies the model's native chat template.
func (m *Model) formatChat(messages []ChatMessage) string {
	switch m.modelType {
	case "gemma3":
		return formatGemmaChat(messages)
	case "qwen2", "qwen3":
		return formatQwenChat(messages)
	case "llama":
		return formatLlamaChat(messages)
	default:
		var s strings.Builder
		for _, msg := range messages {
			s.WriteString(msg.Content + "\n")
		}
		return s.String()
	}
}

func formatGemmaChat(messages []ChatMessage) string {
	var s strings.Builder
	for _, msg := range messages {
		switch msg.Role {
		case "system":
			s.WriteString("<start_of_turn>user\n" + msg.Content + "<end_of_turn>\n")
		case "user":
			s.WriteString("<start_of_turn>user\n" + msg.Content + "<end_of_turn>\n")
		case "assistant":
			s.WriteString("<start_of_turn>model\n" + msg.Content + "<end_of_turn>\n")
		}
	}
	s.WriteString("<start_of_turn>model\n")
	return s.String()
}

func formatQwenChat(messages []ChatMessage) string {
	var s strings.Builder
	for _, msg := range messages {
		s.WriteString("<|im_start|>" + msg.Role + "\n" + msg.Content + "<|im_end|>\n")
	}
	s.WriteString("<|im_start|>assistant\n")
	return s.String()
}

func formatLlamaChat(messages []ChatMessage) string {
	var s strings.Builder
	s.WriteString("<|begin_of_text|>")
	for _, msg := range messages {
		s.WriteString("<|start_header_id|>" + msg.Role + "<|end_header_id|>\n\n" + msg.Content + "<|eot_id|>")
	}
	s.WriteString("<|start_header_id|>assistant<|end_header_id|>\n\n")
	return s.String()
}
