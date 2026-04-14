//go:build darwin && arm64 && !nomlx

package metal

import (
	"context"
	"iter"
	"slices"
	"time"

	"dappco.re/go/core"
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
	MinP          float32
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
	device      DeviceType
	contextLen  int // 0 = unbounded (model default)
	lastErr     error
	lastMetrics Metrics
}

// ModelType returns the architecture identifier (e.g. "gemma3", "qwen3").
//
//	switch m.ModelType() { case "gemma3": ...; case "qwen3": ... }
func (m *Model) ModelType() string { return m.modelType }

// Err returns the error from the last Generate/Chat call, if any.
//
//	if err := m.Err(); err != nil { log.Fatal(err) }
func (m *Model) Err() error { return m.lastErr }

// LastMetrics returns performance metrics from the last inference call.
//
//	met := m.LastMetrics()
//	fmt.Printf("decode: %.0f tok/s, peak GPU: %d MB\n", met.DecodeTokensPerSec, met.PeakMemoryBytes/1024/1024)
func (m *Model) LastMetrics() Metrics { return m.lastMetrics }

// ModelInfo holds metadata about a loaded model.
type ModelInfo struct {
	Architecture  string
	VocabSize     int
	NumLayers     int
	HiddenSize    int
	QuantBits     int
	QuantGroup    int
	ContextLength int
}

// Info returns metadata about the loaded model.
//
//	info := m.Info()
//	fmt.Printf("arch=%s vocab=%d layers=%d quant=%d-bit\n", info.Architecture, info.VocabSize, info.NumLayers, info.QuantBits)
func (m *Model) Info() ModelInfo {
	info := ModelInfo{
		Architecture: m.modelType,
		NumLayers:    m.model.NumLayers(),
	}
	switch v := m.model.(type) {
	case *GemmaModel:
		info.VocabSize = int(v.Cfg.VocabSize)
		info.HiddenSize = int(v.Cfg.HiddenSize)
		info.ContextLength = int(v.Cfg.MaxPositionEmbeddings)
		if v.Cfg.Quantization != nil {
			info.QuantBits = v.Cfg.Quantization.Bits
			info.QuantGroup = v.Cfg.Quantization.GroupSize
		}
	case *Gemma4Model:
		info.VocabSize = int(v.Cfg.VocabSize)
		info.HiddenSize = int(v.Cfg.HiddenSize)
		info.ContextLength = int(v.Cfg.MaxPositionEmbeddings)
		if v.Cfg.Quantization != nil {
			info.QuantBits = v.Cfg.Quantization.Bits
			info.QuantGroup = v.Cfg.Quantization.GroupSize
		}
	case *Qwen3Model:
		info.VocabSize = int(v.Cfg.VocabSize)
		info.HiddenSize = int(v.Cfg.HiddenSize)
		info.ContextLength = int(v.Cfg.MaxPositionEmbeddings)
		if v.Cfg.Quantization != nil {
			info.QuantBits = v.Cfg.Quantization.Bits
			info.QuantGroup = v.Cfg.Quantization.GroupSize
		}
	}
	if m.contextLen > 0 {
		info.ContextLength = m.contextLen
	}
	return info
}

// Close releases all model weight arrays. After Close, the Model must not be used.
func (m *Model) Close() error {
	if m.model == nil {
		return nil
	}
	switch v := m.model.(type) {
	case *GemmaModel:
		closeGemma(v)
	case *Gemma4Model:
		closeGemma4(v)
	case *Qwen3Model:
		closeQwen3(v)
	}
	m.model = nil
	m.tokenizer = nil
	// Closing a model should release its freed weights from the global MLX
	// allocator cache as well, so callers can immediately load another model.
	ClearCache()
	return nil
}

// Chat formats messages using the model's native template and streams tokens.
//
//	for tok := range m.Chat(ctx, []metal.ChatMessage{{Role: "user", Content: "Hello"}}, cfg) {
//	    fmt.Print(tok.Text)
//	}
func (m *Model) Chat(ctx context.Context, messages []ChatMessage, cfg GenerateConfig) iter.Seq[Token] {
	prompt := m.formatChat(messages)
	return m.Generate(ctx, prompt, cfg)
}

// Generate streams tokens for the given prompt.
// Each call allocates fresh KV caches released when the iterator completes.
//
//	for tok := range m.Generate(ctx, "What is 2+2?", metal.GenerateConfig{MaxTokens: 64}) {
//	    fmt.Print(tok.Text)
//	}
func (m *Model) Generate(ctx context.Context, prompt string, cfg GenerateConfig) iter.Seq[Token] {
	m.lastErr = nil
	m.lastMetrics = Metrics{}

	inner := m.generate(ctx, prompt, cfg)
	return func(yield func(Token) bool) {
		if err := m.withDevice(func() { inner(yield) }); err != nil {
			m.lastErr = err
		}
	}
}

func (m *Model) generate(ctx context.Context, prompt string, cfg GenerateConfig) iter.Seq[Token] {
	return func(yield func(Token) bool) {
		totalStart := time.Now()
		ResetPeakMemory()

		tokens := m.tokenizer.Encode(prompt)
		promptLen := len(tokens)
		caches := m.newCaches()
		defer freeCaches(caches)

		sampler := newSampler(cfg.Temperature, cfg.TopP, cfg.MinP, cfg.TopK)
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

		prefillStart := time.Now()
		vInput := FromValues(tokens, len(tokens))
		input := Reshape(vInput, 1, int32(len(tokens)))
		logits := m.model.Forward(input, caches)
		Free(vInput, input)

		if err := Eval(logits); err != nil {
			m.lastErr = core.E("Model.Generate", "prefill", err)
			return
		}
		// Detach logits and cache arrays to release the entire prefill computation
		// graph. After Eval, data is materialised — graph connections only pin Metal
		// memory from intermediate tensors (34 layers × ~20 ops each).
		Detach(logits)
		for _, c := range caches {
			c.Detach()
		}
		prefillDur = time.Since(prefillStart)

		var history []int32 // for repeat penalty

		defer func() {
			Free(logits)
		}()

		for i := range cfg.MaxTokens {
			select {
			case <-ctx.Done():
				m.lastErr = ctx.Err()
				return
			default:
			}

			l1 := SliceAxis(logits, 1, int32(logits.Dim(1)-1), int32(logits.Dim(1)))
			lastPos := Reshape(l1, 1, int32(l1.Dim(2)))
			Free(l1)

			if cfg.RepeatPenalty > 1.0 && len(history) > 0 {
				oldLastPos := lastPos
				lastPos = applyRepeatPenalty(lastPos, history, cfg.RepeatPenalty)
				Free(oldLastPos)
			}

			next := sampler.Sample(lastPos)
			if err := Eval(next); err != nil {
				m.lastErr = core.E("Model.Generate", core.Sprintf("sample step %d", i), err)
				Free(lastPos, next)
				return
			}

			id := int32(next.Int())
			history = append(history, id)
			Free(lastPos)

			if m.tokenizer.HasEOSToken() && id == m.tokenizer.EOSToken() {
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

			vNextInput := FromValues([]int32{id}, 1)
			nextInput := Reshape(vNextInput, 1, 1)
			Free(vNextInput)

			oldLogits := logits
			logits = m.model.Forward(nextInput, caches)
			Free(nextInput, oldLogits)

			if err := Eval(logits); err != nil {
				m.lastErr = core.E("Model.Generate", core.Sprintf("decode step %d", i), err)
				return
			}

			// Detach logits and cache arrays to break the computation graph.
			// Without this, each step's logits holds shared_ptrs through the
			// entire forward pass (SDPA → Slice → cache), pinning hundreds of
			// Metal buffers per step that accumulate to tens of GB.
			Detach(logits)
			for _, c := range caches {
				c.Detach()
			}
		}
	}
}

// InspectAttention runs a single prefill pass and returns post-RoPE K tensors.
// Result.Keys is indexed [layer][head], each slice is seq_len*head_dim float32.
//
//	result, err := m.InspectAttention(ctx, "What is kindness?")
//	fmt.Printf("layers=%d heads=%d seq=%d\n", result.NumLayers, result.NumHeads, result.SeqLen)
func (m *Model) InspectAttention(ctx context.Context, prompt string) (*AttentionResult, error) {
	var (
		result *AttentionResult
		err    error
	)
	if deviceErr := m.withDevice(func() {
		result, err = m.inspectAttention(ctx, prompt)
	}); deviceErr != nil {
		return nil, deviceErr
	}
	return result, err
}

func (m *Model) inspectAttention(ctx context.Context, prompt string) (*AttentionResult, error) {
	tokens := m.tokenizer.Encode(prompt)
	if len(tokens) == 0 {
		return nil, core.E("Model.InspectAttention", "empty prompt after tokenisation", nil)
	}

	caches := m.newCaches()
	defer freeCaches(caches)

	vInput := FromValues(tokens, len(tokens))
	input := Reshape(vInput, 1, int32(len(tokens)))
	Free(vInput)
	logits := m.model.Forward(input, caches)
	Free(input)
	if err := Eval(logits); err != nil {
		return nil, core.E("Model.InspectAttention", "prefill", err)
	}

	info := m.Info()
	seqLen := len(tokens)

	keys := make([][][]float32, info.NumLayers)
	cacheIndexByLayer := attentionCacheIndexByLayer(m.model, info.NumLayers, len(caches))
	cacheSnapshots := make(map[int]attentionCacheSnapshot, len(caches))
	var numHeads, headDim int

	for layerIdx, cacheIdx := range cacheIndexByLayer {
		if cacheIdx < 0 {
			continue
		}
		snapshot, ok := cacheSnapshots[cacheIdx]
		if !ok {
			var extracted bool
			snapshot, extracted = inspectAttentionCache(caches[cacheIdx], seqLen)
			if !extracted {
				continue
			}
			cacheSnapshots[cacheIdx] = snapshot
		}
		keys[layerIdx] = cloneAttentionHeads(snapshot.Keys)
		if numHeads == 0 {
			numHeads = snapshot.NumHeads
		}
		if headDim == 0 {
			headDim = snapshot.HeadDim
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

type attentionCacheSnapshot struct {
	NumHeads int
	HeadDim  int
	Keys     [][]float32
}

func attentionCacheIndexByLayer(model InternalModel, numLayers, numCaches int) []int {
	cacheIndexByLayer := make([]int, numLayers)
	for i := range cacheIndexByLayer {
		cacheIndexByLayer[i] = -1
	}

	switch concrete := model.(type) {
	case *Gemma4Model:
		concrete.ensureCacheLayout()
		for layerIdx := 0; layerIdx < numLayers && layerIdx < len(concrete.PreviousKVs); layerIdx++ {
			ownerIdx := int(concrete.PreviousKVs[layerIdx])
			if ownerIdx < 0 || ownerIdx >= len(concrete.CacheIndexByLayer) {
				continue
			}
			cacheIdx := int(concrete.CacheIndexByLayer[ownerIdx])
			if cacheIdx < 0 || cacheIdx >= numCaches {
				continue
			}
			cacheIndexByLayer[layerIdx] = cacheIdx
		}
	default:
		limit := numLayers
		if numCaches < limit {
			limit = numCaches
		}
		for i := 0; i < limit; i++ {
			cacheIndexByLayer[i] = i
		}
	}

	return cacheIndexByLayer
}

func inspectAttentionCache(cache Cache, seqLen int) (attentionCacheSnapshot, bool) {
	if cache == nil {
		return attentionCacheSnapshot{}, false
	}
	state := cache.State()
	if len(state) < 1 {
		return attentionCacheSnapshot{}, false
	}
	kArray := state[0] // K tensor from cache: [B, H, L_alloc, D]
	shape := kArray.Shape()
	if len(shape) != 4 {
		return attentionCacheSnapshot{}, false
	}

	numHeads := int(shape[1])
	headDim := int(shape[3])
	validLen := min(cache.Len(), seqLen)
	if validLen <= 0 {
		return attentionCacheSnapshot{}, false
	}

	kSliced := Slice(kArray, []int32{0, 0, 0, 0}, []int32{shape[0], shape[1], int32(validLen), shape[3]})
	if err := Eval(kSliced); err != nil {
		Free(kSliced)
		return attentionCacheSnapshot{}, false
	}

	flat := kSliced.Floats() // len = 1 * H * validLen * D
	Free(kSliced)

	keys := make([][]float32, numHeads)
	stride := validLen * headDim
	for h := 0; h < numHeads; h++ {
		start := h * stride
		end := start + stride
		if end > len(flat) {
			break
		}
		head := make([]float32, stride)
		copy(head, flat[start:end])
		keys[h] = head
	}

	return attentionCacheSnapshot{
		NumHeads: numHeads,
		HeadDim:  headDim,
		Keys:     keys,
	}, true
}

func cloneAttentionHeads(src [][]float32) [][]float32 {
	if len(src) == 0 {
		return nil
	}
	cloned := make([][]float32, len(src))
	for i, head := range src {
		if len(head) == 0 {
			continue
		}
		buf := make([]float32, len(head))
		copy(buf, head)
		cloned[i] = buf
	}
	return cloned
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
	case "gemma3", "gemma4", "gemma4_text":
		return formatGemmaChat(messages)
	case "qwen2", "qwen3":
		return formatQwenChat(messages)
	case "llama":
		return formatLlamaChat(messages)
	default:
		builder := core.NewBuilder()
		for _, msg := range messages {
			builder.WriteString(msg.Content + "\n")
		}
		return builder.String()
	}
}

func formatGemmaChat(messages []ChatMessage) string {
	builder := core.NewBuilder()
	for _, msg := range messages {
		switch msg.Role {
		case "system":
			builder.WriteString("<start_of_turn>user\n" + msg.Content + "<end_of_turn>\n")
		case "user":
			builder.WriteString("<start_of_turn>user\n" + msg.Content + "<end_of_turn>\n")
		case "assistant":
			builder.WriteString("<start_of_turn>model\n" + msg.Content + "<end_of_turn>\n")
		}
	}
	builder.WriteString("<start_of_turn>model\n")
	return builder.String()
}

func formatQwenChat(messages []ChatMessage) string {
	builder := core.NewBuilder()
	for _, msg := range messages {
		builder.WriteString("<|im_start|>" + msg.Role + "\n" + msg.Content + "<|im_end|>\n")
	}
	builder.WriteString("<|im_start|>assistant\n")
	return builder.String()
}

func formatLlamaChat(messages []ChatMessage) string {
	builder := core.NewBuilder()
	builder.WriteString("<|begin_of_text|>")
	for _, msg := range messages {
		builder.WriteString("<|start_header_id|>" + msg.Role + "<|end_header_id|>\n\n" + msg.Content + "<|eot_id|>")
	}
	builder.WriteString("<|start_header_id|>assistant<|end_header_id|>\n\n")
	return builder.String()
}
