// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"context"
	"iter"
	"slices"
	"sync"
	"time"

	"dappco.re/go"
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

var (
	enableAsyncDecodePrefetch = core.Env("GO_MLX_ENABLE_ASYNC_DECODE_PREFETCH") == "1"
	enableGenerationStream    = core.Env("GO_MLX_ENABLE_GENERATION_STREAM") == "1"
)

const defaultGenerationClearCacheInterval = 256

// GenerateConfig holds generation parameters.
type GenerateConfig struct {
	MaxTokens        int
	Temperature      float32
	TopK             int
	TopP             float32
	MinP             float32
	StopTokens       []int32
	SuppressTokens   []int32
	RepeatPenalty    float32
	ProbeSink        ProbeSink
	TraceTokenPhases bool
}

// Metrics holds performance metrics from the last inference operation.
type Metrics struct {
	PromptTokens               int
	GeneratedTokens            int
	FirstTokenDuration         time.Duration
	PrefillDuration            time.Duration
	DecodeDuration             time.Duration
	TotalDuration              time.Duration
	PrefillTokensPerSec        float64
	DecodeTokensPerSec         float64
	PeakMemoryBytes            uint64
	ActiveMemoryBytes          uint64
	CacheMemoryBytes           uint64
	ProcessVirtualMemoryBytes  uint64
	ProcessResidentMemoryBytes uint64
	ProcessPeakResidentBytes   uint64
	PromptCacheHits            int
	PromptCacheMisses          int
	PromptCacheHitTokens       int
	PromptCacheMissTokens      int
	PromptCacheRestoreDuration time.Duration
	TokenPhases                []TokenPhaseTrace
	Adapter                    AdapterInfo
}

// TokenPhaseTrace reports coarse timing buckets for one decode-loop token.
type TokenPhaseTrace struct {
	Step                int                `json:"step"`
	FinalToken          bool               `json:"final_token,omitempty"`
	TotalDuration       time.Duration      `json:"total_duration,omitempty"`
	LogitsDuration      time.Duration      `json:"logits_duration,omitempty"`
	SampleDuration      time.Duration      `json:"sample_duration,omitempty"`
	SampleEvalDuration  time.Duration      `json:"sample_eval_duration,omitempty"`
	TokenReadDuration   time.Duration      `json:"token_read_duration,omitempty"`
	DecodeTextDuration  time.Duration      `json:"decode_text_duration,omitempty"`
	ProbeTokenDuration  time.Duration      `json:"probe_token_duration,omitempty"`
	YieldDuration       time.Duration      `json:"yield_duration,omitempty"`
	NextInputDuration   time.Duration      `json:"next_input_duration,omitempty"`
	ForwardDuration     time.Duration      `json:"forward_duration,omitempty"`
	MaterializeDuration time.Duration      `json:"materialize_duration,omitempty"`
	DetachDuration      time.Duration      `json:"detach_duration,omitempty"`
	CacheProbeDuration  time.Duration      `json:"cache_probe_duration,omitempty"`
	OtherDuration       time.Duration      `json:"other_duration,omitempty"`
	NativeEvents        []NativePhaseTrace `json:"native_events,omitempty"`
}

// NativePhaseTrace reports a gated native materialisation event inside a
// decode forward pass.
type NativePhaseTrace struct {
	Name     string        `json:"name"`
	Duration time.Duration `json:"duration"`
	Error    string        `json:"error,omitempty"`
}

// AdapterInfo identifies an active LoRA inference adapter.
type AdapterInfo struct {
	Name       string
	Path       string
	Hash       string
	Rank       int
	Alpha      float32
	Scale      float32
	TargetKeys []string
}

// Model wraps a loaded transformer model for text generation.
type Model struct {
	model                InternalModel
	tokenizer            *Tokenizer
	modelType            string
	device               DeviceType
	contextLen           int // 0 = unbounded (model default)
	cachePolicy          string
	cacheMode            string
	batchSizeLimit       int
	prefillChunkSize     int
	parallelSlots        chan struct{}
	promptCacheMu        sync.Mutex
	promptCacheEnabled   bool
	promptCacheMinTokens int
	promptCache          *promptCacheEntry
	adapter              *LoRAAdapter
	adapterInfo          AdapterInfo
	lastErr              error
	lastMetrics          Metrics
}

// ModelType returns the architecture identifier (e.g. "gemma3", "qwen3").
//
//	switch m.ModelType() { case "gemma3": ...; case "qwen3": ... }
func (m *Model) ModelType() string { return m.modelType }

// Err returns the error from the last Generate/Chat call, if any.
//
//	if err := m.Err(); err != nil { log.Fatal(err) }
func (m *Model) Err() error { return m.lastErr }

func (m *Model) requireTextRuntime(operation string) error {
	if m == nil || m.model == nil {
		return core.NewError("mlx: model is nil")
	}
	architecture := m.modelType
	if architecture == "" {
		architecture = m.model.ModelType()
	}
	switch m.model.(type) {
	case *miniMaxM2StagedModel:
		return core.NewError(operation + ": minimax_m2 staged loader has no native decode kernels yet")
	}
	if m.tokenizer == nil {
		if architecture == "" {
			architecture = "unknown"
		}
		return core.NewError(operation + ": tokenizer unavailable for " + architecture)
	}
	return nil
}

// LastMetrics returns performance metrics from the last inference call.
//
//	met := m.LastMetrics()
//	fmt.Printf("decode: %.0f tok/s, peak GPU: %d MB\n", met.DecodeTokensPerSec, met.PeakMemoryBytes/1024/1024)
func (m *Model) LastMetrics() Metrics { return m.lastMetrics }

func (m *Model) acquireSlot(ctx context.Context) (func(), error) {
	if m == nil || m.parallelSlots == nil {
		return func() {}, nil
	}
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}
	select {
	case m.parallelSlots <- struct{}{}:
		released := false
		return func() {
			if released {
				return
			}
			released = true
			<-m.parallelSlots
		}, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

// ModelInfo holds metadata about a loaded model.
type ModelInfo struct {
	Architecture        string
	VocabSize           int
	NumLayers           int
	HiddenSize          int
	QuantBits           int
	QuantGroup          int
	ContextLength       int
	Gemma4SlidingWindow int
	Adapter             AdapterInfo
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
		info.Gemma4SlidingWindow = int(v.Cfg.SlidingWindow)
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
	case *miniMaxM2StagedModel:
		info.VocabSize = v.plan.Config.VocabSize
		info.HiddenSize = v.plan.Config.HiddenSize
		info.ContextLength = v.plan.Config.MaxPositionEmbeddings
		if info.ContextLength == 0 {
			info.ContextLength = v.plan.Config.SlidingWindow
		}
		info.QuantBits = v.plan.JANG.MXTQBits.RoutedExpert
		if info.QuantBits == 0 {
			info.QuantBits = v.plan.JANG.Quantization.BitsDefault
		}
		info.QuantGroup = v.plan.JANG.Quantization.GroupSize
	}
	if m.contextLen > 0 {
		info.ContextLength = m.contextLen
	}
	info.Adapter = m.Adapter()
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
	m.adapter = nil
	m.adapterInfo = AdapterInfo{}
	m.clearPromptCache()
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
	if err := m.requireTextRuntime("Model.Chat"); err != nil {
		return func(yield func(Token) bool) {
			if m != nil {
				m.lastErr = err
			}
		}
	}
	prompt := m.formatChat(messages)
	return m.Generate(ctx, prompt, cfg)
}

// ChatChunks formats messages with the native chat template and streams tokens
// from bounded prompt chunks.
func (m *Model) ChatChunks(ctx context.Context, messages []ChatMessage, chunkBytes int, cfg GenerateConfig) iter.Seq[Token] {
	if err := m.requireTextRuntime("Model.ChatChunks"); err != nil {
		return func(yield func(Token) bool) {
			if m != nil {
				m.lastErr = err
			}
		}
	}
	return m.GenerateChunks(ctx, m.formatChatChunks(messages, chunkBytes), cfg)
}

// WarmPromptCache prefills and stores an exact token-prefix KV cache.
func (m *Model) WarmPromptCache(ctx context.Context, prompt string) error {
	if err := m.requireTextRuntime("Model.WarmPromptCache"); err != nil {
		return err
	}
	if ctx == nil {
		ctx = context.Background()
	}
	release, err := m.acquireSlot(ctx)
	if err != nil {
		return err
	}
	defer release()
	releasePromptCache := m.acquirePromptCache()
	defer releasePromptCache()

	var warmErr error
	if deviceErr := m.withDevice(func() {
		streamErr := m.withGenerationStream(func() {
			tokens := m.tokenizer.Encode(prompt)
			warmErr = m.warmPromptCacheTokens(ctx, tokens)
		})
		if streamErr != nil {
			warmErr = streamErr
		}
	}); deviceErr != nil {
		return deviceErr
	}
	return warmErr
}

// WarmPromptCacheChunks prefills and stores an exact token-prefix KV cache from
// bounded prompt chunks.
func (m *Model) WarmPromptCacheChunks(ctx context.Context, chunks iter.Seq[string]) error {
	if err := m.requireTextRuntime("Model.WarmPromptCacheChunks"); err != nil {
		return err
	}
	if ctx == nil {
		ctx = context.Background()
	}
	release, err := m.acquireSlot(ctx)
	if err != nil {
		return err
	}
	defer release()
	releasePromptCache := m.acquirePromptCache()
	defer releasePromptCache()

	var warmErr error
	if deviceErr := m.withDevice(func() {
		streamErr := m.withGenerationStream(func() {
			warmErr = m.warmPromptCacheChunks(ctx, chunks)
		})
		if streamErr != nil {
			warmErr = streamErr
		}
	}); deviceErr != nil {
		return deviceErr
	}
	return warmErr
}

func (m *Model) warmPromptCacheTokens(ctx context.Context, tokens []int32) error {
	caches := m.newPromptSnapshotCaches()
	defer freeCaches(caches)
	logits, err := m.prefillTokenBlock(ctx, tokens, caches)
	if err == nil {
		err = m.storePromptCache(tokens, caches, logits)
	}
	Free(logits)
	return err
}

func (m *Model) warmPromptCacheChunks(ctx context.Context, chunks iter.Seq[string]) error {
	caches := m.newPromptSnapshotCaches()
	defer freeCaches(caches)
	tokens, logits, err := m.prefillPromptChunks(ctx, chunks, caches)
	if err == nil {
		err = m.storePromptCache(tokens, caches, logits)
	}
	Free(logits)
	return err
}

// Generate streams tokens for the given prompt.
// Each call allocates fresh KV caches released when the iterator completes.
//
//	for tok := range m.Generate(ctx, "What is 2+2?", metal.GenerateConfig{MaxTokens: 64}) {
//	    fmt.Print(tok.Text)
//	}
func (m *Model) Generate(ctx context.Context, prompt string, cfg GenerateConfig) iter.Seq[Token] {
	return func(yield func(Token) bool) {
		if m == nil {
			return
		}
		m.lastErr = nil
		m.lastMetrics = Metrics{}
		if err := m.requireTextRuntime("Model.Generate"); err != nil {
			m.lastErr = err
			return
		}
		release, err := m.acquireSlot(ctx)
		if err != nil {
			m.lastErr = err
			return
		}
		defer release()
		releasePromptCache := m.acquirePromptCache()
		defer releasePromptCache()
		if err := m.withDevice(func() {
			if streamErr := m.withGenerationStream(func() {
				m.generate(ctx, prompt, cfg)(yield)
			}); streamErr != nil {
				m.lastErr = streamErr
			}
		}); err != nil {
			m.lastErr = err
		}
	}
}

// GenerateChunks streams tokens for a prompt supplied as bounded text chunks.
// Each chunk is tokenized independently and appended to one logical token
// stream, avoiding pathological tokenizer work on very large prompt strings.
func (m *Model) GenerateChunks(ctx context.Context, chunks iter.Seq[string], cfg GenerateConfig) iter.Seq[Token] {
	return func(yield func(Token) bool) {
		if m == nil {
			return
		}
		m.lastErr = nil
		m.lastMetrics = Metrics{}
		if err := m.requireTextRuntime("Model.GenerateChunks"); err != nil {
			m.lastErr = err
			return
		}
		release, err := m.acquireSlot(ctx)
		if err != nil {
			m.lastErr = err
			return
		}
		defer release()
		releasePromptCache := m.acquirePromptCache()
		defer releasePromptCache()
		if err := m.withDevice(func() {
			if streamErr := m.withGenerationStream(func() {
				tokens, encodeErr := m.encodePromptChunks(chunks)
				if encodeErr != nil {
					m.lastErr = encodeErr
					return
				}
				m.generateTokens(ctx, tokens, cfg)(yield)
			}); streamErr != nil {
				m.lastErr = streamErr
			}
		}); err != nil {
			m.lastErr = err
		}
	}
}

func generationStreamEnabled() bool {
	return enableGenerationStream || generationStreamRuntimeEnabled()
}

func generationClearCacheEnabled() bool {
	return generationClearCacheRuntimeEnabled()
}

func generationClearCacheInterval() int {
	if parsed := core.ParseInt(core.Trim(RuntimeGateValue("GO_MLX_GENERATION_CLEAR_CACHE_INTERVAL")), 10, 64); parsed.OK {
		if value := int(parsed.Value.(int64)); value > 0 {
			return value
		}
	}
	return defaultGenerationClearCacheInterval
}

func maybeClearGenerationCache() {
	if generationClearCacheEnabled() {
		ClearCache()
	}
}

func (m *Model) withGenerationStream(fn func()) error {
	if !generationStreamEnabled() {
		fn()
		return nil
	}
	return withTemporaryDefaultStream(m.modelDevice(), fn)
}

func (m *Model) generate(ctx context.Context, prompt string, cfg GenerateConfig) iter.Seq[Token] {
	return m.generateTokens(ctx, m.tokenizer.Encode(prompt), cfg)
}

func (m *Model) encodePromptChunks(chunks iter.Seq[string]) ([]int32, error) {
	if m == nil || m.tokenizer == nil {
		return nil, core.NewError("mlx: tokenizer is nil")
	}
	if chunks == nil {
		return nil, core.NewError("mlx: prompt chunks are nil")
	}
	tokens := []int32{}
	seenContent := false
	for chunk := range chunks {
		if chunk == "" {
			continue
		}
		ids := m.tokenizer.Encode(chunk)
		if seenContent {
			ids = stripImplicitChunkBOS(m.tokenizer, ids)
		}
		tokens = append(tokens, ids...)
		seenContent = true
	}
	if len(tokens) == 0 {
		return nil, core.NewError("Model.GenerateChunks: empty prompt after tokenisation")
	}
	return tokens, nil
}

func (m *Model) prefillPromptChunks(ctx context.Context, chunks iter.Seq[string], caches []Cache) ([]int32, *Array, error) {
	return m.prefillPromptChunksWithPrefix(ctx, chunks, caches, false, "Model.GenerateChunks")
}

func (m *Model) prefillPromptChunksWithPrefix(ctx context.Context, chunks iter.Seq[string], caches []Cache, seenContent bool, scope string) ([]int32, *Array, error) {
	if m == nil || m.tokenizer == nil {
		return nil, nil, core.NewError("mlx: tokenizer is nil")
	}
	if chunks == nil {
		return nil, nil, core.NewError("mlx: prompt chunks are nil")
	}
	tokens := []int32{}
	var logits *Array
	if scope == "" {
		scope = "Model.GenerateChunks"
	}
	for chunk := range chunks {
		if chunk == "" {
			continue
		}
		ids := m.tokenizer.Encode(chunk)
		if seenContent {
			ids = stripImplicitChunkBOS(m.tokenizer, ids)
		}
		if len(ids) == 0 {
			continue
		}
		nextLogits, err := m.prefillTokenBlock(ctx, ids, caches)
		if err != nil {
			Free(logits)
			return nil, nil, core.E(scope, core.Sprintf("prefill chunk tokens=%d", len(tokens)), err)
		}
		Free(logits)
		logits = nextLogits
		tokens = append(tokens, ids...)
		seenContent = true
	}
	if len(tokens) == 0 {
		return nil, nil, core.NewError(scope + ": empty prompt after tokenisation")
	}
	return tokens, logits, nil
}

func stripImplicitChunkBOS(tokenizer *Tokenizer, tokens []int32) []int32 {
	if tokenizer == nil || !tokenizer.HasBOSToken() || len(tokens) == 0 {
		return tokens
	}
	if tokens[0] != tokenizer.BOSToken() {
		return tokens
	}
	return tokens[1:]
}

func (m *Model) generateTokens(ctx context.Context, tokens []int32, cfg GenerateConfig) iter.Seq[Token] {
	return func(yield func(Token) bool) {
		totalStart := time.Now()
		ResetPeakMemory()

		promptLen := len(tokens)
		prepared, err := m.preparePrompt(ctx, tokens, cfg)
		if err != nil {
			m.lastErr = err
			return
		}
		caches := prepared.caches
		logits := prepared.logits
		prefillDur := prepared.duration
		defer freeCaches(caches)
		emitProbeCachePressure(cfg.ProbeSink, ProbePhasePrefill, promptLen, 0, -1, caches)
		emitProbeMemoryPressure(cfg.ProbeSink, ProbePhasePrefill, -1)

		sampler := newSamplerWithSuppression(cfg.Temperature, cfg.TopP, cfg.MinP, cfg.TopK, cfg.SuppressTokens)
		var genCount int
		var firstTokenDuration time.Duration
		var tokenPhases []TokenPhaseTrace

		defer func() {
			decodeDur := time.Since(totalStart) - prefillDur
			totalDur := time.Since(totalStart)
			processMemory := GetProcessMemory()
			m.lastMetrics = Metrics{
				PromptTokens:               promptLen,
				GeneratedTokens:            genCount,
				FirstTokenDuration:         firstTokenDuration,
				PrefillDuration:            prefillDur,
				DecodeDuration:             decodeDur,
				TotalDuration:              totalDur,
				PeakMemoryBytes:            GetPeakMemory(),
				ActiveMemoryBytes:          GetActiveMemory(),
				CacheMemoryBytes:           GetCacheMemory(),
				ProcessVirtualMemoryBytes:  processMemory.VirtualMemoryBytes,
				ProcessResidentMemoryBytes: processMemory.ResidentMemoryBytes,
				ProcessPeakResidentBytes:   processMemory.PeakResidentMemoryBytes,
				TokenPhases:                tokenPhases,
				Adapter:                    m.Adapter(),
			}
			if prefillDur > 0 {
				m.lastMetrics.PrefillTokensPerSec = float64(promptLen) / prefillDur.Seconds()
			}
			if decodeDur > 0 {
				m.lastMetrics.DecodeTokensPerSec = float64(genCount) / decodeDur.Seconds()
			}
			if prepared.cacheHit {
				m.lastMetrics.PromptCacheHits = 1
			} else {
				m.lastMetrics.PromptCacheMisses = 1
			}
			m.lastMetrics.PromptCacheHitTokens = prepared.cacheHitTokens
			m.lastMetrics.PromptCacheMissTokens = prepared.cacheMissTokens
			m.lastMetrics.PromptCacheRestoreDuration = prepared.restoreDuration
		}()

		var history []int32 // for repeat penalty
		var directNext *Array

		defer func() {
			Free(logits, directNext)
		}()

		for i := range cfg.MaxTokens {
			tracePhases := cfg.TraceTokenPhases
			var phaseStart, phaseLast time.Time
			var phase TokenPhaseTrace
			if tracePhases {
				phaseStart = time.Now()
				phaseLast = phaseStart
				phase = TokenPhaseTrace{Step: i}
			}
			select {
			case <-ctx.Done():
				m.lastErr = ctx.Err()
				return
			default:
			}

			var next *Array
			nextEvaluated := false
			if directNext != nil {
				next = directNext
				directNext = nil
				if tracePhases {
					phase.LogitsDuration = time.Since(phaseLast)
					phaseLast = time.Now()
				}
			} else if nativeGreedyDecodeAvailable(cfg, history, logits) {
				var err error
				next, err = nativeGreedyDecodeToken(logits)
				if err != nil {
					m.lastErr = core.E("Model.Generate", core.Sprintf("native greedy decode step %d", i), err)
					return
				}
				if tracePhases {
					phase.LogitsDuration = time.Since(phaseLast)
					phaseLast = time.Now()
				}
			} else {
				lastPos, err := lastTokenLogits(logits)
				if err != nil {
					m.lastErr = core.E("Model.Generate", core.Sprintf("last logits step %d", i), err)
					return
				}

				if cfg.RepeatPenalty > 1.0 && len(history) > 0 {
					oldLastPos := lastPos
					lastPos = applyRepeatPenalty(lastPos, history, cfg.RepeatPenalty)
					Free(oldLastPos)
				}
				if tracePhases {
					phase.LogitsDuration = time.Since(phaseLast)
					phaseLast = time.Now()
				}

				if err := emitProbeLogits(cfg.ProbeSink, ProbePhaseDecode, i, lastPos); err != nil {
					m.lastErr = core.E("Model.Generate", core.Sprintf("probe logits step %d", i), err)
					Free(lastPos)
					return
				}
				if tracePhases && cfg.ProbeSink != nil {
					phase.CacheProbeDuration += time.Since(phaseLast)
				}
				if tracePhases {
					phaseLast = time.Now()
				}

				var sampleErr error
				next, sampleErr = sampleTokenWithSuppressionGuard(lastPos, sampler, cfg.SuppressTokens)
				if sampleErr != nil {
					m.lastErr = core.E("Model.Generate", core.Sprintf("sample step %d", i), sampleErr)
					Free(lastPos)
					return
				}
				nextEvaluated = true
				if tracePhases {
					phase.SampleDuration = time.Since(phaseLast)
					phaseLast = time.Now()
				}
				Free(lastPos)
			}
			if !nextEvaluated {
				if err := Eval(next); err != nil {
					m.lastErr = core.E("Model.Generate", core.Sprintf("sample step %d", i), err)
					Free(next)
					return
				}
			}
			if tracePhases {
				phase.SampleEvalDuration = time.Since(phaseLast)
				phaseLast = time.Now()
			}
			// Eval(next) also materialises the lazy decode forward that produced
			// logits for this token, so detach caches at this boundary.
			detachCaches(caches)
			if generationClearCacheEnabled() {
				if interval := generationClearCacheInterval(); interval > 0 && (i+1)%interval == 0 {
					ClearCache()
				}
			}
			if tracePhases {
				phase.DetachDuration = time.Since(phaseLast)
				phaseLast = time.Now()
			}
			emitProbeCachePressure(cfg.ProbeSink, ProbePhaseDecode, promptLen, genCount, i, caches)
			emitProbeMemoryPressure(cfg.ProbeSink, ProbePhaseDecode, i)
			if tracePhases && cfg.ProbeSink != nil {
				phase.CacheProbeDuration += time.Since(phaseLast)
			}
			if tracePhases {
				phaseLast = time.Now()
			}

			id := int32(next.Int())
			if tracePhases {
				phase.TokenReadDuration = time.Since(phaseLast)
				phaseLast = time.Now()
			}
			history = append(history, id)
			text := m.tokenizer.DecodeToken(id)
			if tracePhases {
				phase.DecodeTextDuration = time.Since(phaseLast)
				phaseLast = time.Now()
			}
			emitProbeToken(cfg.ProbeSink, ProbePhaseDecode, i, id, text, promptLen, genCount+1)
			if tracePhases {
				phase.ProbeTokenDuration = time.Since(phaseLast)
				phaseLast = time.Now()
			}

			if m.tokenizer.HasEOSToken() && id == m.tokenizer.EOSToken() {
				Free(next)
				if tracePhases {
					phase.FinalToken = true
					tokenPhases = appendTokenPhaseTrace(tokenPhases, phase, phaseStart)
				}
				return
			}
			if slices.Contains(cfg.StopTokens, id) {
				Free(next)
				if tracePhases {
					phase.FinalToken = true
					tokenPhases = appendTokenPhaseTrace(tokenPhases, phase, phaseStart)
				}
				return
			}

			genCount++
			if firstTokenDuration == 0 {
				firstTokenDuration = time.Since(totalStart)
			}
			if !yield(Token{ID: id, Text: text}) {
				Free(next)
				if tracePhases {
					phase.FinalToken = true
					tokenPhases = appendTokenPhaseTrace(tokenPhases, phase, phaseStart)
				}
				return
			}
			if tracePhases {
				phase.YieldDuration = time.Since(phaseLast)
				phaseLast = time.Now()
			}
			Free(next)
			if i == cfg.MaxTokens-1 {
				if tracePhases {
					phase.FinalToken = true
					tokenPhases = appendTokenPhaseTrace(tokenPhases, phase, phaseStart)
				}
				return
			}

			vNextInput := FromValues([]int32{id}, 1)
			nextInput := Reshape(vNextInput, 1, 1)
			Free(vNextInput)
			if tracePhases {
				phase.NextInputDuration = time.Since(phaseLast)
				phaseLast = time.Now()
			}

			oldLogits := logits
			if directGreedyTokenAvailable(cfg, history, m.model) {
				if tracePhases {
					resetNativePhaseTraceEvents()
				}
				nextToken, _ := m.forwardGreedyToken(nextInput, nil, caches, cfg.SuppressTokens)
				if tracePhases {
					phase.ForwardDuration = time.Since(phaseLast)
					phase.NativeEvents = takeNativePhaseTraceEvents()
					phaseLast = time.Now()
				}
				Free(nextInput)
				if nextToken == nil || !nextToken.Valid() {
					if err := lastError(); err != nil {
						m.lastErr = core.E("Model.Generate", core.Sprintf("direct greedy decode step %d", i), err)
					} else {
						m.lastErr = core.E("Model.Generate", core.Sprintf("direct greedy decode step %d", i), core.NewError("model forward returned nil token"))
					}
					Free(oldLogits, nextToken)
					logits = nil
					return
				}
				Free(oldLogits)
				logits = nil
				directNext = nextToken
				if err := asyncDecodePrefetch(i, "direct greedy token", directNext); err != nil {
					m.lastErr = err
					return
				}
			} else {
				if tracePhases {
					resetNativePhaseTraceEvents()
				}
				nextLogits, _ := m.forwardLastTokenLogits(nextInput, nil, caches)
				if tracePhases {
					phase.ForwardDuration = time.Since(phaseLast)
					phase.NativeEvents = takeNativePhaseTraceEvents()
					phaseLast = time.Now()
				}
				Free(nextInput)
				if nextLogits == nil || !nextLogits.Valid() {
					if err := lastError(); err != nil {
						m.lastErr = core.E("Model.Generate", core.Sprintf("decode step %d", i), err)
					} else {
						m.lastErr = core.E("Model.Generate", core.Sprintf("decode step %d", i), core.NewError("model forward returned nil logits"))
					}
					Free(oldLogits, nextLogits)
					logits = nil
					return
				}
				Free(oldLogits)
				logits = nextLogits
				if err := asyncDecodePrefetch(i, "next logits", logits); err != nil {
					m.lastErr = err
					return
				}
			}
			if tracePhases {
				tokenPhases = appendTokenPhaseTrace(tokenPhases, phase, phaseStart)
			}
		}
	}
}

func directGreedyTokenAvailable(cfg GenerateConfig, history []int32, model InternalModel) bool {
	if !directGreedyTokenEnabled() {
		return false
	}
	if _, ok := model.(GreedyTokenModel); !ok {
		return false
	}
	return cfg.ProbeSink == nil &&
		cfg.Temperature == 0 &&
		cfg.TopP == 0 &&
		cfg.MinP == 0 &&
		cfg.TopK == 0 &&
		(len(cfg.SuppressTokens) == 0 || suppressedGreedyTokenAvailable(model)) &&
		(cfg.RepeatPenalty <= 1 || len(history) == 0)
}

func suppressedGreedyTokenAvailable(model InternalModel) bool {
	_, ok := model.(SuppressedGreedyTokenModel)
	return ok
}

func (m *Model) forwardGreedyToken(tokens *Array, mask *Array, caches []Cache, suppressTokens []int32) (*Array, bool) {
	if len(suppressTokens) > 0 {
		greedyModel, ok := m.model.(SuppressedGreedyTokenModel)
		if !ok {
			return nil, false
		}
		return greedyModel.ForwardGreedyTokenWithSuppression(tokens, mask, caches, suppressTokens), true
	}
	greedyModel, ok := m.model.(GreedyTokenModel)
	if !ok {
		return nil, false
	}
	return greedyModel.ForwardGreedyToken(tokens, mask, caches), true
}

func asyncDecodePrefetch(step int, label string, out *Array) error {
	if !enableAsyncDecodePrefetch || out == nil || !out.Valid() {
		return nil
	}
	if err := EvalAsync(out); err != nil {
		return core.E("Model.Generate", core.Sprintf("async prefetch %s step %d", label, step), err)
	}
	return nil
}

func appendTokenPhaseTrace(phases []TokenPhaseTrace, phase TokenPhaseTrace, start time.Time) []TokenPhaseTrace {
	phase.TotalDuration = time.Since(start)
	if accounted := tokenPhaseAccountedDuration(phase); phase.TotalDuration > accounted {
		phase.OtherDuration = phase.TotalDuration - accounted
	}
	return append(phases, phase)
}

func tokenPhaseAccountedDuration(phase TokenPhaseTrace) time.Duration {
	return phase.LogitsDuration +
		phase.SampleDuration +
		phase.SampleEvalDuration +
		phase.TokenReadDuration +
		phase.DecodeTextDuration +
		phase.ProbeTokenDuration +
		phase.YieldDuration +
		phase.NextInputDuration +
		phase.ForwardDuration +
		phase.MaterializeDuration +
		phase.DetachDuration +
		phase.CacheProbeDuration
}

// InspectAttention runs a single prefill pass and returns post-RoPE K tensors.
// Result.Keys is indexed [layer][head], each slice is seq_len*head_dim float32.
//
//	result, err := m.InspectAttention(ctx, "What is kindness?")
//	fmt.Printf("layers=%d heads=%d seq=%d\n", result.NumLayers, result.NumHeads, result.SeqLen)
func (m *Model) InspectAttention(ctx context.Context, prompt string) (*AttentionResult, error) {
	if err := m.requireTextRuntime("Model.InspectAttention"); err != nil {
		return nil, err
	}
	var (
		result *AttentionResult
		err    error
	)
	release, slotErr := m.acquireSlot(ctx)
	if slotErr != nil {
		return nil, slotErr
	}
	defer release()
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
	defer Free(logits)
	Free(input)
	if err := Eval(logits); err != nil {
		return nil, core.E("Model.InspectAttention", "prefill", err)
	}
	detachEvalState(logits, caches)

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
		NumLayers:     info.NumLayers,
		NumHeads:      numHeads,
		SeqLen:        seqLen,
		HeadDim:       headDim,
		NumQueryHeads: attentionQueryHeads(m.model),
		Keys:          keys,
		Architecture:  info.Architecture,
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
	state, ownedState := cacheReadState(cache)
	defer Free(ownedState...)
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

func detachEvalState(logits *Array, caches []Cache) {
	Detach(logits)
	detachCaches(caches)
}

func detachCaches(caches []Cache) {
	for _, cache := range caches {
		if cache != nil {
			cache.Detach()
		}
	}
}

// AttentionResult holds extracted K vectors from the KV cache.
type AttentionResult struct {
	NumLayers     int
	NumHeads      int
	SeqLen        int
	HeadDim       int
	NumQueryHeads int
	Keys          [][][]float32 // [layer][head] → flat float32 of len seq_len*head_dim
	Queries       [][][]float32 // [layer][head] → flat float32 of len seq_len*head_dim
	Architecture  string
}

func attentionQueryHeads(model InternalModel) int {
	switch concrete := model.(type) {
	case *GemmaModel:
		if concrete.Cfg != nil {
			return int(concrete.Cfg.NumAttentionHeads)
		}
	case *Gemma4Model:
		if concrete.Cfg != nil {
			return int(concrete.Cfg.NumAttentionHeads)
		}
	case *Qwen3Model:
		if concrete.Cfg != nil {
			return int(concrete.Cfg.NumAttentionHeads)
		}
	}
	return 0
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
	return m.newCachesWithRequestFixedSize(0)
}

func (m *Model) newGenerationCaches(promptTokens int, cfg GenerateConfig) []Cache {
	return m.newCachesWithRequestFixedSize(m.generationFixedGemma4CacheSize(promptTokens, cfg.MaxTokens))
}

func (m *Model) newCachesWithRequestFixedSize(requestFixedSize int) []Cache {
	caches := m.model.NewCache()
	if mode := KVCacheMode(m.cacheMode); mode == KVCacheModeQ8 || mode == KVCacheModeKQ8VQ4 || mode == KVCacheModePaged {
		maxSize := 0
		if m.cachePolicy != "full" && m.contextLen > 0 {
			maxSize = m.contextLen
		}
		storageDType, hasStorageDType := kvCacheStorageDType()
		for i := range caches {
			layerMaxSize := replacementCacheMaxSize(caches[i], maxSize)
			switch mode {
			case KVCacheModeQ8:
				caches[i] = NewQuantizedKVCache(layerMaxSize, 8, 8)
			case KVCacheModeKQ8VQ4:
				caches[i] = NewQuantizedKVCache(layerMaxSize, 8, 4)
			case KVCacheModePaged:
				if fixedGemma4CacheEnabled() && maxSize > 0 && (m.modelType == "gemma4" || m.modelType == "gemma4_text") {
					fixedSize := fixedGemma4CacheSize(maxSize, requestFixedSize)
					if fixedGemma4SlidingCacheBoundEnabled() && layerMaxSize > 0 {
						fixedSize = min(fixedSize, layerMaxSize)
					}
					if hasStorageDType {
						caches[i] = NewFixedKVCacheWithDType(fixedSize, storageDType)
					} else {
						caches[i] = NewFixedKVCache(fixedSize)
					}
				} else {
					if hasStorageDType {
						caches[i] = NewPagedKVCacheWithDType(layerMaxSize, 0, storageDType)
					} else {
						caches[i] = NewPagedKVCache(layerMaxSize, 0)
					}
				}
			}
		}
		return caches
	}
	return m.applyContextCachePolicy(caches)
}

func kvCacheStorageDType() (DType, bool) {
	value := core.Lower(core.Trim(RuntimeGateValue("GO_MLX_KV_CACHE_DTYPE")))
	switch value {
	case "", "native", "default":
		return DTypeFloat32, false
	case "fp16", "float16", "f16":
		return DTypeFloat16, true
	case "bf16", "bfloat16":
		return DTypeBFloat16, true
	default:
		return DTypeFloat32, false
	}
}

func (m *Model) generationFixedGemma4CacheSize(promptTokens, maxTokens int) int {
	if m == nil || !fixedGemma4CacheEnabled() || promptTokens <= 0 || maxTokens <= 0 {
		return 0
	}
	if KVCacheMode(m.cacheMode) != KVCacheModePaged || m.contextLen <= 0 {
		return 0
	}
	modelType := m.modelType
	if modelType == "" && m.model != nil {
		modelType = m.model.ModelType()
	}
	if modelType != "gemma4" && modelType != "gemma4_text" {
		return 0
	}
	size := promptTokens + maxTokens
	if size < promptTokens {
		return 0
	}
	return roundUpPositive(size, 32)
}

func fixedGemma4CacheSize(maxSize, requestSize int) int {
	if maxSize <= 0 {
		return maxSize
	}
	parsed := core.ParseInt(core.Trim(core.Env("GO_MLX_FIXED_GEMMA4_CACHE_SIZE")), 10, 64)
	if parsed.OK {
		size := int(parsed.Value.(int64))
		if size > 0 {
			return min(size, maxSize)
		}
	}
	if requestSize > 0 {
		return min(requestSize, maxSize)
	}
	return maxSize
}

func roundUpPositive(value, multiple int) int {
	if value <= 0 || multiple <= 0 {
		return value
	}
	remainder := value % multiple
	if remainder == 0 {
		return value
	}
	return value + multiple - remainder
}

func replacementCacheMaxSize(cache Cache, maxSize int) int {
	if maxSize <= 0 {
		return maxSize
	}
	if rotating, ok := cache.(*RotatingKVCache); ok && rotating.maxSize > 0 {
		return min(maxSize, rotating.maxSize)
	}
	return maxSize
}

func (m *Model) newPromptSnapshotCaches() []Cache {
	switch KVCacheMode(m.cacheMode) {
	case KVCacheModeKQ8VQ4:
		return m.applyContextCachePolicy(m.model.NewCache())
	default:
		return m.newCaches()
	}
}

func (m *Model) applyContextCachePolicy(caches []Cache) []Cache {
	if m.cachePolicy == "full" {
		return caches
	}
	if m.contextLen <= 0 {
		return caches
	}
	for i, c := range caches {
		switch cache := c.(type) {
		// Replace unbounded caches with rotating caches to honour the requested
		// context cap.
		case *KVCache:
			caches[i] = NewRotatingKVCache(m.contextLen)
		// Sliding-window caches are already bounded, but still need shrinking
		// when the caller requests a smaller context than the model default.
		case *RotatingKVCache:
			if cache.maxSize > m.contextLen {
				caches[i] = NewRotatingKVCache(m.contextLen)
			}
		default:
			continue
		}
	}
	return caches
}

// formatChat applies the model's native chat template.
func (m *Model) formatChat(messages []ChatMessage) string {
	switch m.modelType {
	case "gemma4", "gemma4_text":
		return formatGemma4Chat(messages)
	case "gemma2", "gemma3", "gemma3_text":
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

func (m *Model) formatChatChunks(messages []ChatMessage, chunkBytes int) iter.Seq[string] {
	return func(yield func(string) bool) {
		switch m.modelType {
		case "gemma4", "gemma4_text":
			formatGemma4ChatChunks(messages, chunkBytes, yield)
		case "gemma2", "gemma3", "gemma3_text":
			formatGemmaChatChunks(messages, chunkBytes, yield)
		case "qwen2", "qwen3":
			formatQwenChatChunks(messages, chunkBytes, yield)
		case "llama":
			formatLlamaChatChunks(messages, chunkBytes, yield)
		default:
			for _, msg := range messages {
				if !yieldChatTextChunks(yield, msg.Content+"\n", chunkBytes) {
					return
				}
			}
		}
	}
}

func yieldChatTextChunks(yield func(string) bool, text string, chunkBytes int) bool {
	if text == "" {
		return true
	}
	if chunkBytes <= 0 || len(text) <= chunkBytes {
		return yield(text)
	}
	start := 0
	for index := range text {
		if index == start || index-start < chunkBytes {
			continue
		}
		if !yield(text[start:index]) {
			return false
		}
		start = index
	}
	if start < len(text) {
		return yield(text[start:])
	}
	return true
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

func formatGemmaChatChunks(messages []ChatMessage, chunkBytes int, yield func(string) bool) {
	for _, msg := range messages {
		switch msg.Role {
		case "system", "user":
			if !yield("<start_of_turn>user\n") || !yieldChatTextChunks(yield, msg.Content, chunkBytes) || !yield("<end_of_turn>\n") {
				return
			}
		case "assistant":
			if !yield("<start_of_turn>model\n") || !yieldChatTextChunks(yield, msg.Content, chunkBytes) || !yield("<end_of_turn>\n") {
				return
			}
		}
	}
	yield("<start_of_turn>model\n")
}

func formatGemma4Chat(messages []ChatMessage) string {
	builder := core.NewBuilder()
	builder.WriteString("<bos>")
	for _, msg := range messages {
		role := core.Lower(core.Trim(msg.Role))
		content := core.Trim(msg.Content)
		switch role {
		case "assistant", "model":
			role = "model"
		case "developer", "system":
			role = "system"
		case "human", "user":
			role = "user"
		default:
			continue
		}
		builder.WriteString("<|turn>" + role + "\n" + content + "<turn|>\n")
	}
	builder.WriteString("<|turn>model\n")
	builder.WriteString("<|channel>thought\n<channel|>")
	return builder.String()
}

func formatGemma4ChatChunks(messages []ChatMessage, chunkBytes int, yield func(string) bool) {
	if !yield("<bos>") {
		return
	}
	for _, msg := range messages {
		role := core.Lower(core.Trim(msg.Role))
		content := core.Trim(msg.Content)
		switch role {
		case "assistant", "model":
			role = "model"
		case "developer", "system":
			role = "system"
		case "human", "user":
			role = "user"
		default:
			continue
		}
		if !yield("<|turn>"+role+"\n") || !yieldChatTextChunks(yield, content, chunkBytes) || !yield("<turn|>\n") {
			return
		}
	}
	if !yield("<|turn>model\n") {
		return
	}
	yield("<|channel>thought\n<channel|>")
}

func formatQwenChat(messages []ChatMessage) string {
	builder := core.NewBuilder()
	for _, msg := range messages {
		builder.WriteString("<|im_start|>" + msg.Role + "\n" + msg.Content + "<|im_end|>\n")
	}
	builder.WriteString("<|im_start|>assistant\n")
	return builder.String()
}

func formatQwenChatChunks(messages []ChatMessage, chunkBytes int, yield func(string) bool) {
	for _, msg := range messages {
		if !yield("<|im_start|>"+msg.Role+"\n") || !yieldChatTextChunks(yield, msg.Content, chunkBytes) || !yield("<|im_end|>\n") {
			return
		}
	}
	yield("<|im_start|>assistant\n")
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

func formatLlamaChatChunks(messages []ChatMessage, chunkBytes int, yield func(string) bool) {
	if !yield("<|begin_of_text|>") {
		return
	}
	for _, msg := range messages {
		if !yield("<|start_header_id|>"+msg.Role+"<|end_header_id|>\n\n") || !yieldChatTextChunks(yield, msg.Content, chunkBytes) || !yield("<|eot_id|>") {
			return
		}
	}
	yield("<|start_header_id|>assistant<|end_header_id|>\n\n")
}

func lastTokenLogits(logits *Array) (*Array, error) {
	if logits == nil || !logits.Valid() {
		return nil, core.NewError("mlx: logits are empty")
	}
	ndim := logits.NumDims()
	if ndim <= 0 {
		return nil, core.NewError("mlx: logits rank is invalid")
	}
	if ndim == 1 {
		return Reshape(logits, 1, int32(logits.Dim(0))), nil
	}
	if ndim == 2 {
		rows := logits.Dim(0)
		if rows <= 0 {
			return nil, core.NewError("mlx: logits sequence is empty")
		}
		last := SliceAxis(logits, 0, int32(rows-1), int32(rows))
		out := Reshape(last, 1, int32(last.Dim(last.NumDims()-1)))
		Free(last)
		return out, nil
	}
	seqAxis := ndim - 2
	seqLen := logits.Dim(seqAxis)
	if seqLen <= 0 {
		return nil, core.NewError("mlx: logits sequence is empty")
	}
	last := SliceAxis(logits, seqAxis, int32(seqLen-1), int32(seqLen))
	out := Reshape(last, 1, int32(last.Dim(last.NumDims()-1)))
	Free(last)
	return out, nil
}

func materializeLastTokenLogits(logits *Array) (*Array, error) {
	if logits == nil {
		return nil, core.NewError("mlx: logits are empty")
	}
	if !logits.Valid() {
		if err := lastError(); err != nil {
			return nil, core.E("mlx", "logits are empty", err)
		}
		return nil, core.NewError("mlx: logits are empty")
	}
	if err := Eval(logits); err != nil {
		Free(logits)
		return nil, err
	}
	last, err := lastTokenLogits(logits)
	if err != nil {
		Free(logits)
		return nil, err
	}
	if err := Eval(last); err != nil {
		Free(logits, last)
		return nil, err
	}
	Detach(last)
	Free(logits)
	return last, nil
}
