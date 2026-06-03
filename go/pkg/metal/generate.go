// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"context"
	"iter"
	"slices"
	"sync"
	"time"
	"unsafe"

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
	MaxTokens           int
	Temperature         float32
	TopK                int
	TopP                float32
	MinP                float32
	Seed                uint64
	SeedSet             bool
	StopTokens          []int32
	SuppressTokens      []int32
	MinTokensBeforeStop int
	RepeatPenalty       float32
	ProbeSink           ProbeSink
	TraceTokenPhases    bool
	TraceTokenText      bool
	// EnableThinking toggles Gemma 4 reasoning at prompt-build time. nil = model
	// default (on for Gemma 4); &true = on; &false = off (plain template, plus the
	// 26B/31B ghost-channel suppressor). Ignored by non-Gemma-4 architectures.
	EnableThinking *bool
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
	CacheProfile               *CacheProfile
	TurboQuantKVPayload        *TurboQuantKVCachePayloadEstimate
	TokenPhases                []TokenPhaseTrace
	MTP                        *MTPMetrics
	Adapter                    AdapterInfo
}

// MTPMetrics records counters from an attached multi-token-prediction drafter.
type MTPMetrics struct {
	DraftTokenSchedule     []int
	ProposedTokens         int
	AcceptedTokens         int
	RejectedTokens         int
	TargetVerifyCalls      int
	TargetCalls            int
	DraftCalls             int
	AcceptanceRate         float64
	VisibleTokensPerSec    float64
	TargetTokensPerSec     float64
	WarmDecodeTokensPerSec float64
	WallDuration           time.Duration
	RestoreDuration        time.Duration
	TargetVerifyDuration   time.Duration
	TargetDuration         time.Duration
	DraftDuration          time.Duration
	PeakMemoryBytes        uint64
}

// TokenPhaseTrace reports coarse timing buckets for one decode-loop token.
type TokenPhaseTrace struct {
	Step                   int                `json:"step"`
	TokenID                int32              `json:"token_id"`
	TokenText              string             `json:"token_text,omitempty"`
	FinalToken             bool               `json:"final_token,omitempty"`
	TotalDuration          time.Duration      `json:"total_duration,omitempty"`
	LogitsDuration         time.Duration      `json:"logits_duration,omitempty"`
	SampleDuration         time.Duration      `json:"sample_duration,omitempty"`
	SampleEvalDuration     time.Duration      `json:"sample_eval_duration,omitempty"`
	TokenReadDuration      time.Duration      `json:"token_read_duration,omitempty"`
	DecodeTextDuration     time.Duration      `json:"decode_text_duration,omitempty"`
	ProbeTokenDuration     time.Duration      `json:"probe_token_duration,omitempty"`
	YieldDuration          time.Duration      `json:"yield_duration,omitempty"`
	NextInputDuration      time.Duration      `json:"next_input_duration,omitempty"`
	ForwardDuration        time.Duration      `json:"forward_duration,omitempty"`
	PrefetchDuration       time.Duration      `json:"prefetch_duration,omitempty"`
	PrefetchLogitsDuration time.Duration      `json:"prefetch_logits_duration,omitempty"`
	PrefetchCacheDuration  time.Duration      `json:"prefetch_cache_duration,omitempty"`
	MaterializeDuration    time.Duration      `json:"materialize_duration,omitempty"`
	DetachDuration         time.Duration      `json:"detach_duration,omitempty"`
	CacheProbeDuration     time.Duration      `json:"cache_probe_duration,omitempty"`
	OtherDuration          time.Duration      `json:"other_duration,omitempty"`
	NativeEvents           []NativePhaseTrace `json:"native_events,omitempty"`
}

// NativePhaseTrace reports a gated native materialisation event inside a
// decode forward pass.
type NativePhaseTrace struct {
	Name     string        `json:"name"`
	Duration time.Duration `json:"duration"`
	Error    string        `json:"error,omitempty"`
	Pages    int           `json:"pages,omitempty"`
	Tokens   int           `json:"tokens,omitempty"`
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
	switch v := m.model.(type) {
	case *miniMaxM2StagedModel:
		return core.NewError(operation + ": minimax_m2 staged loader has no native decode kernels yet")
	case *Qwen3MoEModel:
		if !qwen3MoETextRuntimeAvailable(v) {
			return core.NewError(operation + ": qwen3_moe model is loaded but native sparse-expert decode kernels are not yet linked")
		}
	case *qwen36StagedModel:
		return core.NewError(operation + ": qwen3_6 staged loader has no native hybrid linear-attention decode kernels yet")
	case *MixtralModel:
		if !mixtralTextRuntimeAvailable(v) {
			return core.NewError(operation + ": mixtral model is loaded but native sparse-expert decode kernels are not yet linked")
		}
	case *KimiModel:
		if !kimiTextRuntimeAvailable(v) {
			return core.NewError(operation + ": kimi model is loaded but native sparse-expert decode kernels are not yet linked")
		}
	case *GptOssModel:
		if !gptOssTextRuntimeAvailable(v) {
			return core.NewError(operation + ": gpt_oss model is loaded but native sparse-expert decode kernels are not yet linked")
		}
	case *moeStagedModel:
		return core.NewError(operation + ": " + architecture + " staged loader has no native sparse-expert decode kernels yet")
	case *qwen36MoEStagedModel:
		return core.NewError(operation + ": qwen3_6_moe staged loader has no native hybrid linear-attention and sparse-expert decode kernels yet")
	case *bertStagedModel:
		return core.NewError(operation + ": " + architecture + " staged loader has no native text decode kernels; use the encoder/rerank API once scorer kernels land")
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
	NumHeads            int
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
	if reporter, ok := m.model.(modelInfoReporter); ok {
		reporter.fillModelInfo(&info)
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
	if closer, ok := m.model.(modelCloser); ok {
		closer.closeModel()
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
	prompt := m.formatChat(messages, cfg)
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
	return m.GenerateChunks(ctx, m.formatChatChunks(messages, chunkBytes, cfg), cfg)
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
				if seedErr := applyGenerationSeed(cfg); seedErr != nil {
					m.lastErr = seedErr
					return
				}
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
				if seedErr := applyGenerationSeed(cfg); seedErr != nil {
					m.lastErr = seedErr
					return
				}
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

func applyGenerationSeed(cfg GenerateConfig) error {
	if !cfg.SeedSet {
		return nil
	}
	return SeedRandom(cfg.Seed)
}

func generationStreamEnabled() bool {
	return enableGenerationStream || generationStreamRuntimeEnabled()
}

func asyncDecodePrefetchEnabled() bool {
	return enableAsyncDecodePrefetch || asyncDecodePrefetchRuntimeEnabled()
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
		defer closeSampler(sampler)
		earlySuppressTokens := cfg.SuppressTokens
		earlySampler := sampler
		earlySamplerDistinct := false
		if cfg.MinTokensBeforeStop > 0 {
			earlySuppressTokens = generationStopSuppressionTokens(cfg.SuppressTokens, cfg.StopTokens, m.tokenizer)
			if len(earlySuppressTokens) != len(cfg.SuppressTokens) {
				earlySampler = newSamplerWithSuppression(cfg.Temperature, cfg.TopP, cfg.MinP, cfg.TopK, earlySuppressTokens)
				earlySamplerDistinct = true
			}
		}
		if earlySamplerDistinct {
			defer closeSampler(earlySampler)
		}
		var genCount int
		var firstTokenDuration time.Duration
		tokenPhases := newTokenPhaseTraceBuffer(cfg)

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
				CacheProfile:               modelCacheProfile(m.model, caches),
				TurboQuantKVPayload:        turboQuantKVCachesPayloadEstimate(caches),
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
		var suppressTokensArray *Array
		if len(cfg.SuppressTokens) > 0 && directGreedyTokenEnabled() {
			suppressTokensArray = suppressTokenArray(cfg.SuppressTokens)
		}
		var earlySuppressTokensArray *Array
		if len(earlySuppressTokens) > 0 && len(earlySuppressTokens) != len(cfg.SuppressTokens) && directGreedyTokenEnabled() {
			earlySuppressTokensArray = suppressTokenArray(earlySuppressTokens)
		}

		defer func() {
			Free(logits, directNext, suppressTokensArray, earlySuppressTokensArray)
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
			var sampledID int32
			sampledIDSet := false
			nextEvaluated := false
			stepCfg := cfg
			stepSampler := sampler
			stepSuppressTokens := cfg.SuppressTokens
			if generationStopSuppressionActive(genCount, cfg) {
				stepCfg.SuppressTokens = earlySuppressTokens
				stepSampler = earlySampler
				stepSuppressTokens = earlySuppressTokens
			}
			if directNext != nil {
				next = directNext
				directNext = nil
				if tracePhases {
					phase.LogitsDuration = time.Since(phaseLast)
					phaseLast = time.Now()
				}
			} else if nativeGreedyDecodeAvailable(stepCfg, history, logits) {
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
				var sampleTimings sampleTokenTimings
				next, sampledID, sampleTimings, sampleErr = sampleTokenIDWithSuppressionGuard(lastPos, stepSampler, stepSuppressTokens, tracePhases)
				if sampleErr != nil {
					m.lastErr = core.E("Model.Generate", core.Sprintf("sample step %d", i), sampleErr)
					Free(lastPos)
					return
				}
				sampledIDSet = true
				nextEvaluated = true
				if tracePhases {
					phase.SampleDuration = sampleTimings.Build
					phase.SampleEvalDuration = sampleTimings.Eval
					phase.TokenReadDuration += sampleTimings.TokenRead
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
				if tracePhases {
					phase.SampleEvalDuration += time.Since(phaseLast)
					phaseLast = time.Now()
				}
			}
			// Eval(next) also materialises the lazy decode forward that produced
			// logits for this token, so detach logits and caches at this
			// boundary before building the next one-token graph.
			detachEvalState(logits, caches)
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

			id := sampledID
			if !sampledIDSet {
				id = int32(next.Int())
				if tracePhases {
					phase.TokenReadDuration += time.Since(phaseLast)
					phaseLast = time.Now()
				}
			}
			if cfg.RepeatPenalty > 1.0 {
				history = append(history, id)
			}
			text := m.tokenizer.DecodeToken(id)
			if tracePhases {
				phase.TokenID = id
				if cfg.TraceTokenText {
					phase.TokenText = text
				}
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

			nextInput := fromSingleInt32Matrix(id)
			if tracePhases {
				phase.NextInputDuration = time.Since(phaseLast)
				phaseLast = time.Now()
			}

			oldLogits := logits
			nextCfg := cfg
			nextSuppressTokens := cfg.SuppressTokens
			nextSuppressTokensArray := suppressTokensArray
			if generationStopSuppressionActive(genCount, cfg) {
				nextCfg.SuppressTokens = earlySuppressTokens
				nextSuppressTokens = earlySuppressTokens
				if earlySuppressTokensArray != nil {
					nextSuppressTokensArray = earlySuppressTokensArray
				}
			}
			if directGreedyTokenAvailable(nextCfg, history, m.model) {
				if tracePhases {
					resetNativePhaseTraceEvents()
				}
				nextToken, _ := m.forwardGreedyToken(nextInput, nil, caches, nextSuppressTokens, nextSuppressTokensArray)
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
				var prefetchTimings asyncDecodePrefetchTimings
				var prefetchErr error
				if tracePhases {
					prefetchTimings, prefetchErr = asyncDecodePrefetchWithCachesTrace("Model.Generate", i, "direct greedy token and dirty KV", directNext, caches)
				} else {
					prefetchErr = asyncDecodePrefetchWithCaches("Model.Generate", i, "direct greedy token and dirty KV", directNext, caches)
				}
				if prefetchErr != nil {
					m.lastErr = prefetchErr
					return
				}
				if tracePhases {
					phase.PrefetchDuration = time.Since(phaseLast)
					phase.PrefetchLogitsDuration = prefetchTimings.Logits
					phase.PrefetchCacheDuration = prefetchTimings.Cache
					phaseLast = time.Now()
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
				var prefetchTimings asyncDecodePrefetchTimings
				var prefetchErr error
				if tracePhases {
					prefetchTimings, prefetchErr = asyncDecodePrefetchWithCachesTrace("Model.Generate", i, "next logits and dirty KV", logits, caches)
				} else {
					prefetchErr = asyncDecodePrefetchWithCaches("Model.Generate", i, "next logits and dirty KV", logits, caches)
				}
				if prefetchErr != nil {
					m.lastErr = prefetchErr
					return
				}
				if tracePhases {
					phase.PrefetchDuration = time.Since(phaseLast)
					phase.PrefetchLogitsDuration = prefetchTimings.Logits
					phase.PrefetchCacheDuration = prefetchTimings.Cache
					phaseLast = time.Now()
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

func generationStopSuppressionActive(generated int, cfg GenerateConfig) bool {
	return cfg.MinTokensBeforeStop > 0 && generated < cfg.MinTokensBeforeStop
}

func generationStopSuppressionTokens(base, stop []int32, tokenizer *Tokenizer) []int32 {
	out := base
	if tokenizer != nil && tokenizer.HasEOSToken() {
		out = appendUniqueSuppressionToken(out, tokenizer.EOSToken(), base)
	}
	for _, id := range stop {
		out = appendUniqueSuppressionToken(out, id, base)
	}
	return out
}

func appendUniqueSuppressionToken(out []int32, id int32, base []int32) []int32 {
	if slices.Contains(out, id) {
		return out
	}
	if len(out) == len(base) {
		out = append([]int32(nil), out...)
	}
	return append(out, id)
}

func suppressedGreedyTokenAvailable(model InternalModel) bool {
	_, ok := model.(SuppressedGreedyTokenModel)
	return ok
}

type borrowedSuppressedGreedyTokenModel interface {
	forwardGreedyTokenWithSuppressionArray(tokens *Array, mask *Array, caches []Cache, suppressTokens []int32, suppress *Array) *Array
}

func (m *Model) forwardGreedyToken(tokens *Array, mask *Array, caches []Cache, suppressTokens []int32, suppress *Array) (*Array, bool) {
	if len(suppressTokens) > 0 {
		if greedyModel, ok := m.model.(borrowedSuppressedGreedyTokenModel); ok {
			return greedyModel.forwardGreedyTokenWithSuppressionArray(tokens, mask, caches, suppressTokens, suppress), true
		}
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
	return asyncDecodePrefetchFor("Model.Generate", step, label, out)
}

func asyncDecodePrefetchFor(scope string, step int, label string, out *Array) error {
	if !asyncDecodePrefetchEnabled() || out == nil || !out.Valid() {
		return nil
	}
	return asyncDecodePrefetchArraysFor(scope, step, label, out)
}

type asyncDecodePrefetchTimings struct {
	Logits time.Duration
	Cache  time.Duration
}

func asyncDecodePrefetchWithCaches(scope string, step int, label string, out *Array, caches []Cache) error {
	if !asyncDecodePrefetchEnabled() {
		return nil
	}
	var stack [64]*Array
	outputs := stack[:0]
	if out != nil && out.Valid() {
		outputs = append(outputs, out)
	}
	for _, cache := range caches {
		outputs = appendCacheDirtyState(outputs, cache)
	}
	if len(outputs) == 0 {
		return nil
	}
	return asyncDecodePrefetchArraysFor(scope, step, label, outputs...)
}

func asyncDecodePrefetchWithCachesTrace(scope string, step int, label string, out *Array, caches []Cache) (asyncDecodePrefetchTimings, error) {
	var timings asyncDecodePrefetchTimings
	if !asyncDecodePrefetchEnabled() {
		return timings, nil
	}
	var stack [64]*Array
	outputs := stack[:0]
	hasLogits := false
	if out != nil && out.Valid() {
		outputs = append(outputs, out)
		hasLogits = true
	}
	for _, cache := range caches {
		outputs = appendCacheDirtyState(outputs, cache)
	}
	if len(outputs) == 0 {
		return timings, nil
	}
	start := time.Now()
	if err := asyncDecodePrefetchArraysFor(scope, step, label, outputs...); err != nil {
		return timings, err
	}
	elapsed := nonZeroTraceDuration(time.Since(start))
	if hasLogits {
		// Keep trace mode on the same combined eval boundary as production.
		// Splitting logits and dirty K/V into separate EvalAsync calls gives
		// cleaner attribution but changes the graph shape being measured.
		timings.Logits = elapsed
	} else {
		timings.Cache = elapsed
	}
	return timings, nil
}

func asyncDecodePrefetchWithCachesTraceSplit(scope string, step int, label string, out *Array, caches []Cache) (asyncDecodePrefetchTimings, error) {
	var timings asyncDecodePrefetchTimings
	if !asyncDecodePrefetchEnabled() {
		return timings, nil
	}
	if out != nil && out.Valid() {
		start := time.Now()
		if err := asyncDecodePrefetchArraysFor(scope, step, label+" logits", out); err != nil {
			return timings, err
		}
		timings.Logits = nonZeroTraceDuration(time.Since(start))
	}
	var stack [64]*Array
	dirty := stack[:0]
	for _, cache := range caches {
		dirty = appendCacheDirtyState(dirty, cache)
	}
	if len(dirty) > 0 {
		start := time.Now()
		if err := asyncDecodePrefetchArraysFor(scope, step, label+" dirty KV", dirty...); err != nil {
			return timings, err
		}
		timings.Cache = nonZeroTraceDuration(time.Since(start))
	}
	return timings, nil
}

func asyncDecodePrefetchArraysFor(scope string, step int, label string, outputs ...*Array) error {
	if !asyncDecodePrefetchEnabled() || len(outputs) == 0 {
		return nil
	}
	if err := EvalAsync(outputs...); err != nil {
		if core.Trim(scope) == "" {
			scope = "Model.Generate"
		}
		return core.E(scope, core.Sprintf("async prefetch %s step %d", label, step), err)
	}
	return nil
}

func nonZeroTraceDuration(d time.Duration) time.Duration {
	if d <= 0 {
		return time.Nanosecond
	}
	return d
}

func appendTokenPhaseTrace(phases []TokenPhaseTrace, phase TokenPhaseTrace, start time.Time) []TokenPhaseTrace {
	phase.TotalDuration = time.Since(start)
	if accounted := tokenPhaseAccountedDuration(phase); phase.TotalDuration > accounted {
		phase.OtherDuration = phase.TotalDuration - accounted
	}
	return append(phases, phase)
}

func newTokenPhaseTraceBuffer(cfg GenerateConfig) []TokenPhaseTrace {
	if !cfg.TraceTokenPhases || cfg.MaxTokens <= 0 {
		return nil
	}
	return make([]TokenPhaseTrace, 0, cfg.MaxTokens)
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
		phase.PrefetchDuration +
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
	input := Reshape2(vInput, 1, int32(len(tokens)))
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
	if layouter, ok := model.(attentionCacheLayouter); ok {
		return layouter.attentionCacheLayout(numLayers, numCaches)
	}
	if planner, ok := model.(qwen36HybridCachePlanner); ok {
		return qwen36AttentionCacheIndexByLayer(planner, numLayers, numCaches)
	}

	// Default: identity mapping (layer i → cache i), capped by cache count.
	cacheIndexByLayer := make([]int, numLayers)
	for i := range cacheIndexByLayer {
		cacheIndexByLayer[i] = -1
	}
	limit := numLayers
	if numCaches < limit {
		limit = numCaches
	}
	for i := 0; i < limit; i++ {
		cacheIndexByLayer[i] = i
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

	// W11-X / W11-AE: borrow an MLX-memory view rather than copying the full
	// [1, H, L, D] K-tensor into a fresh Go []float32 (Floats() does
	// make + per-element copy — ~16MB on a 32-head/1024-token/128-dim
	// cache).  Per-head slices are copied into independent buffers via
	// the loop below, so the borrowed view ends at function return.
	// W11-AE: kSliced was Eval'd above, so the fast-path skips the final
	// Materialize crossing when dtype + layout already match.
	flat, flatCleanup, err := materialiseFloat32ViewFast(kSliced)
	if err != nil {
		Free(kSliced)
		return attentionCacheSnapshot{}, false
	}
	defer flatCleanup()
	if len(flat) == 0 {
		Free(kSliced)
		return attentionCacheSnapshot{}, false
	}

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
	Free(kSliced)

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
	if counter, ok := model.(queryHeadCounter); ok {
		return counter.numQueryHeads()
	}
	return 0
}

// repeatPenaltyScratch is a pooled []int32 buffer reused for history dedup
// inside applyRepeatPenalty.  Sampling fires once per emitted token, so
// recycling the dedup scratch eliminates the map+slice allocation pair on
// the per-token hot path.  Capacity grows as needed and stays in the pool.
var repeatPenaltyScratch = sync.Pool{
	New: func() any {
		buf := make([]int32, 0, 64)
		return &buf
	},
}

// applyRepeatPenalty modifies logits to discourage repeated tokens.
// For each unique token ID in history: positive logits are divided by penalty,
// negative logits are multiplied by penalty. Both make the token less likely.
func applyRepeatPenalty(logits *Array, history []int32, penalty float32) *Array {
	// Deduplicate history via pooled scratch slice — sort + compact beats
	// map[int32]bool for the typical history sizes (≤256 tokens) and avoids
	// the per-call map allocation that dominated B/op.
	scratchPtr := repeatPenaltyScratch.Get().(*[]int32)
	scratch := (*scratchPtr)[:0]
	if cap(scratch) < len(history) {
		scratch = make([]int32, 0, len(history))
	}
	scratch = append(scratch, history...)
	slices.Sort(scratch)
	indices := slices.Compact(scratch)

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

	// Return the scratch buffer to the pool — FromValues has copied the
	// indices into MLX-owned memory already.
	*scratchPtr = scratch
	repeatPenaltyScratch.Put(scratchPtr)
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
	if mode := KVCacheMode(m.cacheMode); mode == KVCacheModeQ8 || mode == KVCacheModeKQ8VQ4 || mode == KVCacheModePaged || mode == KVCacheModeTurboQuant {
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
			case KVCacheModeTurboQuant:
				cache := NewTurboQuantKVCache(layerMaxSize, 0)
				cache.SetLayerIdentity(i, i, i, "unknown")
				caches[i] = cache
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
	parsed := core.ParseInt(core.Trim(RuntimeGateValue("GO_MLX_FIXED_GEMMA4_CACHE_SIZE")), 10, 64)
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

func lastTokenLogits(logits *Array) (*Array, error) {
	if logits == nil || !logits.Valid() {
		return nil, core.NewError("mlx: logits are empty")
	}
	ndim := logits.NumDims()
	if ndim <= 0 {
		return nil, core.NewError("mlx: logits rank is invalid")
	}
	shape := logits.ShapeRaw()
	if ndim == 1 {
		return Reshape2(logits, 1, int32(shapeRawDim(shape, 0))), nil
	}
	if ndim == 2 {
		rows := shapeRawDim(shape, 0)
		if rows <= 0 {
			return nil, core.NewError("mlx: logits sequence is empty")
		}
		if rows == 1 {
			return Reshape2(logits, 1, int32(shapeRawDim(shape, 1))), nil
		}
		last := SliceAxis(logits, 0, int32(rows-1), int32(rows))
		out := Reshape2(last, 1, int32(shapeRawDim(shape, 1)))
		Free(last)
		return out, nil
	}
	seqAxis := ndim - 2
	seqLen := shapeRawDim(shape, seqAxis)
	if seqLen <= 0 {
		return nil, core.NewError("mlx: logits sequence is empty")
	}
	if seqLen == 1 && lastTokenLogitsSinglePosition(shape, ndim) {
		return Reshape2(logits, 1, int32(shapeRawDim(shape, ndim-1))), nil
	}
	last := SliceAxis(logits, seqAxis, int32(seqLen-1), int32(seqLen))
	out := Reshape2(last, 1, int32(shapeRawDim(shape, ndim-1)))
	Free(last)
	return out, nil
}

func lastTokenLogitsSinglePosition(shape unsafe.Pointer, ndim int) bool {
	for axis := 0; axis < ndim-1; axis++ {
		if shapeRawDim(shape, axis) != 1 {
			return false
		}
	}
	return true
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
