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

// ChatMessage represents a chat turn. Images carries encoded PNG/JPEG bytes
// attached to the turn (#98) — the vision-chat lane splices their projected
// features into the prompt; text-only engines reject image turns loudly.
type ChatMessage struct {
	Role    string
	Content string
	Images  [][]byte
}

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
	ClearCache          bool
	ClearCacheInterval  int
	// EnableThinking toggles Gemma 4 reasoning at prompt-build time. nil = model
	// default (on for Gemma 4); &true = on; &false = off (plain template, plus the
	// 26B/31B ghost-channel suppressor). Ignored by non-Gemma-4 architectures.
	EnableThinking *bool
}

// Metrics holds performance metrics from the last inference operation.
type Metrics struct {
	PromptTokens        int
	GeneratedTokens     int
	FirstTokenDuration  time.Duration
	PrefillDuration     time.Duration
	DecodeDuration      time.Duration
	TotalDuration       time.Duration
	PrefillTokensPerSec float64
	DecodeTokensPerSec  float64
	// WarmDecodeTokensPerSec excludes the FIRST decode step (kernel JIT
	// compiles, cache growth, allocator warmup) — the steady-state rate.
	// DecodeTokensPerSec includes that cold start, so it RISES asymptotically
	// with generation length as the fixed cost amortises; this one stays flat.
	// "Decode got faster with more tokens" is this dilution, not acceleration.
	WarmDecodeTokensPerSec     float64
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
	// DecodeLane names the loop that served the generation ("pipelined" or
	// "serial"), and DecodeLaneReason carries the first failed eligibility
	// condition when serial — rate triage starts by knowing which loop ran.
	DecodeLane       string
	DecodeLaneReason string
	// CompiledLayerHits counts whole-layer compiled decode steps during this
	// generation (all layers compiled = layers × tokens).
	CompiledLayerHits uint64
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
	model                 InternalModel
	tokenizer             *Tokenizer
	modelType             string
	device                DeviceType
	contextLen            int // 0 = unbounded (model default)
	cachePolicy           string
	cacheMode             string
	kvCacheStorageDType   string
	pagedKVPageSize       int
	pagedKVPrealloc       bool
	fixedSlidingCacheSize int
	batchSizeLimit        int
	prefillChunkSize      int
	parallelSlots         chan struct{}
	promptCacheMu         sync.Mutex
	promptCacheEnabled    bool
	promptCacheMinTokens  int
	promptCache           *PromptCacheEntry
	adapter               *LoRAAdapter
	adapterInfo           AdapterInfo
	lastErr               error
	lastMetrics           Metrics
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
	if r, ok := m.model.(MoETextRuntimeReporter); ok {
		if !r.MoETextRuntimeAvailable() {
			return core.NewError(operation + ": " + r.MoETextDecodeFamily() + " model is loaded but native sparse-expert decode kernels are not yet linked")
		}
	}
	if r, ok := m.model.(DecodeUnavailableReporter); ok {
		return r.DecodeUnavailableError(operation)
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
	Architecture          string
	VocabSize             int
	NumLayers             int
	NumHeads              int
	NumKVHeads            int
	HeadDim               int
	HiddenSize            int
	QuantBits             int
	QuantGroup            int
	ContextLength         int
	SlidingWindow         int
	KVCacheStorageDType   string
	PagedKVPageSize       int
	PagedKVPrealloc       bool
	FixedSlidingCacheSize int
	Adapter               AdapterInfo
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
	if reporter, ok := m.model.(ModelInfoReporter); ok {
		reporter.FillModelInfo(&info)
	}
	if m.contextLen > 0 {
		info.ContextLength = m.contextLen
	}
	info.KVCacheStorageDType = m.kvCacheStorageDType
	info.PagedKVPageSize = m.pagedKVPageSize
	info.PagedKVPrealloc = m.pagedKVPrealloc
	info.FixedSlidingCacheSize = m.fixedSlidingCacheSize
	info.Adapter = m.Adapter()
	return info
}

// Close releases all model weight arrays. After Close, the Model must not be used.
func (m *Model) Close() error {
	if m.model == nil {
		return nil
	}
	if closer, ok := m.model.(ModelCloser); ok {
		closer.CloseModel()
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
	if chatMessagesCarryImages(messages) {
		return m.chatVision(ctx, messages, cfg)
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
	defer FreeCaches(caches)
	logits, err := m.prefillTokenBlock(ctx, tokens, caches)
	if err == nil {
		err = m.storePromptCache(tokens, caches, logits)
	}
	Free(logits)
	return err
}

func (m *Model) warmPromptCacheChunks(ctx context.Context, chunks iter.Seq[string]) error {
	caches := m.newPromptSnapshotCaches()
	defer FreeCaches(caches)
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
		if bd, ok := m.model.(BlockDiffusionModel); ok {
			// Diffusion checkpoints decode by canvas denoising — the
			// autoregressive lanes never see them.
			m.generateViaBlockDiffusion(ctx, bd, prompt, cfg)(yield)
			return
		}
		if m.sessionRouteEligible(cfg) {
			m.generateViaSession(ctx, prompt, cfg)(yield)
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

// sessionRouteEligible reports whether a one-shot generation can ride the
// session machinery — the pipelined decode + compiled closures + prompt-cache
// restore live there, so the session route is the fast path (e2b: 180.9 tok/s
// session vs 126.5 one-shot on the same snapshot). The one-shot loop remains
// for the configs sessions do not implement.
func (m *Model) sessionRouteEligible(cfg GenerateConfig) bool {
	// Sessions do not implement the allocator clear-cache debug lever.
	return !cfg.ClearCache
}

// generateViaSession runs a one-shot generation through a throwaway session.
// The session takes its own slot/prompt-cache/device scopes per operation, so
// this wraps NOTHING — double-acquiring the slot semaphore would deadlock a
// single-slot model. Session.Generate writes m.lastMetrics in its defer;
// the session error is mirrored into m.lastErr for the Model.Err contract.
func (m *Model) generateViaSession(ctx context.Context, prompt string, cfg GenerateConfig) iter.Seq[Token] {
	return func(yield func(Token) bool) {
		sess := m.NewSession()
		defer sess.Close()
		if err := sess.Prefill(ctx, prompt); err != nil {
			m.lastErr = err
			return
		}
		if seedErr := applyGenerationSeed(cfg); seedErr != nil {
			m.lastErr = seedErr
			return
		}
		sess.Generate(ctx, cfg)(yield)
		if err := sess.Err(); err != nil {
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

// samplerKeysForConfig builds the per-generation explicit PRNG key sequence:
// seeded configs replay the same draws, unseeded get a random root. One
// sequence is shared by a generation's sampler AND earlySampler so every
// drawn token consumes a distinct key (the global mlx_random_seed state
// cannot give per-request reproducibility — concurrent requests interleave
// on it).
func samplerKeysForConfig(cfg GenerateConfig) *SamplerKeys {
	if cfg.SeedSet {
		return NewSamplerKeys(cfg.Seed)
	}
	return newRandomSamplerKeys()
}

// generationStreamEnabled reports whether the streaming decode path is active.
// The value is carried by the runtime gate, which the loaded model's
// EngineFeatures.Apply sets (and CLI / shell-env diagnostics may override) —
// there is no separate init-time package var, so a later clear is honoured
// rather than frozen at boot. (#55 slice 3b)
func generationStreamEnabled() bool {
	return generationStreamRuntimeEnabled()
}

// asyncDecodePrefetchEnabled reports whether decode overlaps the next step's
// weight prefetch. Carried by the runtime gate (set by the loaded model's
// EngineFeatures.Apply; CLI / shell-env may override) — no init-time package
// var, so a clear is honoured rather than frozen at boot. (#55 slice 3b)
func asyncDecodePrefetchEnabled() bool {
	return asyncDecodePrefetchRuntimeEnabled()
}

func generationClearCacheInterval(cfg GenerateConfig) int {
	if cfg.ClearCacheInterval > 0 {
		return cfg.ClearCacheInterval
	}
	return defaultGenerationClearCacheInterval
}

func maybeClearGenerationCache(cfg GenerateConfig) {
	if cfg.ClearCache {
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

// promptPreparer primes a generation: caches + last-token logits for a token
// prompt. preparePrompt is the text lane; the vision-chat lane supplies a
// multimodal preparer that injects image features during prefill (#98).
type promptPreparer func(context.Context, []int32, GenerateConfig) (PromptPreparation, error)

func (m *Model) generateTokens(ctx context.Context, tokens []int32, cfg GenerateConfig) iter.Seq[Token] {
	return m.generateTokensFrom(ctx, tokens, cfg, m.preparePrompt)
}

func (m *Model) generateTokensFrom(ctx context.Context, tokens []int32, cfg GenerateConfig, prepare promptPreparer) iter.Seq[Token] {
	return func(yield func(Token) bool) {
		totalStart := time.Now()
		ResetPeakMemory()

		promptLen := len(tokens)
		prepared, err := prepare(ctx, tokens, cfg)
		if err != nil {
			m.lastErr = err
			return
		}
		caches := prepared.Caches
		logits := prepared.Logits
		prefillDur := prepared.Duration
		defer FreeCaches(caches)
		emitProbeCachePressure(cfg.ProbeSink, ProbePhasePrefill, promptLen, 0, -1, caches)
		emitProbeMemoryPressure(cfg.ProbeSink, ProbePhasePrefill, -1)

		samplerKeys := samplerKeysForConfig(cfg)
		sampler := NewSamplerWithSuppressionKeyed(cfg.Temperature, cfg.TopP, cfg.MinP, cfg.TopK, cfg.SuppressTokens, samplerKeys)
		defer CloseSampler(sampler)
		earlySuppressTokens := cfg.SuppressTokens
		earlySampler := sampler
		earlySamplerDistinct := false
		if cfg.MinTokensBeforeStop > 0 {
			earlySuppressTokens = generationStopSuppressionTokens(cfg.SuppressTokens, cfg.StopTokens, m.tokenizer)
			if len(earlySuppressTokens) != len(cfg.SuppressTokens) {
				earlySampler = NewSamplerWithSuppressionKeyed(cfg.Temperature, cfg.TopP, cfg.MinP, cfg.TopK, earlySuppressTokens, samplerKeys)
				earlySamplerDistinct = true
			}
		}
		if earlySamplerDistinct {
			defer CloseSampler(earlySampler)
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
			// firstTokenDuration is measured from totalStart (includes prefill);
			// the first DECODE step's share is firstTokenDuration - prefillDur.
			if genCount > 1 && firstTokenDuration > prefillDur {
				warmDur := decodeDur - (firstTokenDuration - prefillDur)
				if warmDur > 0 {
					m.lastMetrics.WarmDecodeTokensPerSec = float64(genCount-1) / warmDur.Seconds()
				}
			}
			if prepared.CacheHit {
				m.lastMetrics.PromptCacheHits = 1
			} else {
				m.lastMetrics.PromptCacheMisses = 1
			}
			m.lastMetrics.PromptCacheHitTokens = prepared.CacheHitTokens
			m.lastMetrics.PromptCacheMissTokens = prepared.CacheMissTokens
			m.lastMetrics.PromptCacheRestoreDuration = prepared.RestoreDuration
		}()

		var history []int32 // for repeat penalty
		var directNext *Array
		var suppressTokensArray *Array
		if len(cfg.SuppressTokens) > 0 && directGreedyTokenEnabled() {
			suppressTokensArray = SuppressTokenArray(cfg.SuppressTokens)
		}
		var earlySuppressTokensArray *Array
		if len(earlySuppressTokens) > 0 && len(earlySuppressTokens) != len(cfg.SuppressTokens) && directGreedyTokenEnabled() {
			earlySuppressTokensArray = SuppressTokenArray(earlySuppressTokens)
		}

		defer func() {
			Free(logits, directNext, suppressTokensArray, earlySuppressTokensArray)
		}()

		// Resolve the generation budget from truth — an explicit MaxTokens is
		// honoured; MaxTokens <= 0 generates to the model's remaining context
		// (the EOS/stop checks below terminate the loop), never a hardcoded cap.
		budget := generationTokenBudget(cfg.MaxTokens, m.Info().ContextLength, len(tokens))
		for i := 0; i < budget; i++ {
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
					m.lastErr = core.E("Model.Generate", core.Sprintf("native Greedy decode step %d", i), err)
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
				next, sampledID, sampleTimings, sampleErr = SampleTokenIDWithSuppressionGuard(lastPos, stepSampler, stepSuppressTokens, tracePhases)
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
			if cfg.ClearCache {
				if interval := generationClearCacheInterval(cfg); interval > 0 && (i+1)%interval == 0 {
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
			if i == budget-1 {
				if tracePhases {
					phase.FinalToken = true
					tokenPhases = appendTokenPhaseTrace(tokenPhases, phase, phaseStart)
				}
				return
			}

			nextInput := FromSingleInt32Matrix(id)
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
					if err := LastError(); err != nil {
						m.lastErr = core.E("Model.Generate", core.Sprintf("direct Greedy decode step %d", i), err)
					} else {
						m.lastErr = core.E("Model.Generate", core.Sprintf("direct Greedy decode step %d", i), core.NewError("model forward returned nil token"))
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
					prefetchTimings, prefetchErr = asyncDecodePrefetchWithCachesTrace("Model.Generate", i, "direct Greedy token and dirty KV", directNext, caches)
				} else {
					prefetchErr = asyncDecodePrefetchWithCaches("Model.Generate", i, "direct Greedy token and dirty KV", directNext, caches)
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
					if err := LastError(); err != nil {
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
	defer FreeCaches(caches)

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
	if layouter, ok := model.(AttentionCacheLayouter); ok {
		return layouter.AttentionCacheLayout(numLayers, numCaches)
	}
	if planner, ok := model.(HybridAttentionCachePlanner); ok {
		return hybridAttentionCacheIndexByLayer(planner, numLayers, numCaches)
	}

	// Default: identity mapping (layer i → cache i), capped by cache count.
	cacheIndexByLayer := make([]int, numLayers)
	for i := range cacheIndexByLayer {
		cacheIndexByLayer[i] = -1
	}
	limit := min(numCaches, numLayers)
	for i := 0; i < limit; i++ {
		cacheIndexByLayer[i] = i
	}
	return cacheIndexByLayer
}

func hybridAttentionCacheIndexByLayer(model HybridAttentionCachePlanner, numLayers, numCaches int) []int {
	cacheIndexByLayer := make([]int, numLayers)
	for i := range cacheIndexByLayer {
		cacheIndexByLayer[i] = -1
	}
	plan, ok := model.HybridAttentionCachePlan()
	if !ok {
		return cacheIndexByLayer
	}
	for layerIdx := 0; layerIdx < numLayers && layerIdx < len(plan.CacheIndexByLayer); layerIdx++ {
		cacheIdx := plan.CacheIndexByLayer[layerIdx]
		if cacheIdx >= 0 && cacheIdx < numCaches {
			cacheIndexByLayer[layerIdx] = cacheIdx
		}
	}
	return cacheIndexByLayer
}

func inspectAttentionCache(cache Cache, seqLen int) (attentionCacheSnapshot, bool) {
	if cache == nil {
		return attentionCacheSnapshot{}, false
	}
	state, ownedState := CacheReadState(cache)
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
	for h := range numHeads {
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
	DetachCaches(caches)
}

func DetachCaches(caches []Cache) {
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
	if counter, ok := model.(QueryHeadCounter); ok {
		return counter.NumQueryHeads()
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
	budget := generationTokenBudget(cfg.MaxTokens, m.Info().ContextLength, promptTokens)
	return m.newCachesWithRequestFixedSize(m.generationFixedSlidingCacheSize(promptTokens, budget))
}

func (m *Model) newCachesWithRequestFixedSize(requestFixedSize int) []Cache {
	caches := m.model.NewCache()
	mode := KVCacheMode(m.cacheMode)
	// The fixed-cache regime: a model that declares the fixed-sliding cache
	// (EngineFeatures, e.g. hybrid gemma4) gets sized FixedKVCaches — the
	// compiled+pipelined decode shape — with zero flags in the default mode,
	// or under the explicit -kv-cache paged + -context pair. The serve and
	// the CLI must not need a magic flag to reach the fast lane (#72).
	if mode == KVCacheModeDefault || mode == KVCacheModePaged {
		if replaced, ok := m.fixedSlidingReplacement(caches, requestFixedSize); ok {
			return replaced
		}
	}
	if mode == KVCacheModeQ8 || mode == KVCacheModeKQ8VQ4 || mode == KVCacheModePaged || mode == KVCacheModeTurboQuant {
		maxSize := 0
		if m.cachePolicy != "full" && m.contextLen > 0 {
			maxSize = m.contextLen
		}
		storageDType, hasStorageDType := parseKVCacheStorageDType(m.kvCacheStorageDType)
		for i := range caches {
			layerMaxSize := replacementCacheMaxSize(caches[i], maxSize)
			switch mode {
			case KVCacheModeQ8:
				caches[i] = NewQuantizedKVCache(layerMaxSize, 8, 8)
			case KVCacheModeKQ8VQ4:
				caches[i] = NewQuantizedKVCache(layerMaxSize, 8, 4)
			case KVCacheModePaged:
				if hasStorageDType {
					caches[i] = NewPagedKVCacheWithDTypeAndPrealloc(layerMaxSize, m.pagedKVPageSize, storageDType, m.pagedKVPrealloc)
				} else {
					caches[i] = NewPagedKVCacheWithPrealloc(layerMaxSize, m.pagedKVPageSize, m.pagedKVPrealloc)
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

// DefaultFixedCacheBound is the zero-flag context bound for the fixed-cache
// regime: ample for agent multi-turn work (the ten-chapter book demo peaks
// under 10K tokens) while keeping the lazily-allocated fixed buffers modest,
// and free in decode speed — the rate is flat in the bound (e2b: 181 tok/s
// at 8K, 24K and 64K alike). -context overrides it in either direction.
const DefaultFixedCacheBound = 24576

// defaultFixedCacheBound resolves the zero-flag bound: the model's declared
// context clamped to DefaultFixedCacheBound — a 128K-context model must not
// allocate 128K-token fixed buffers on the first request.
func (m *Model) defaultFixedCacheBound() int {
	ctx := m.Info().ContextLength
	if ctx <= 0 {
		return DefaultFixedCacheBound
	}
	return min(ctx, DefaultFixedCacheBound)
}

// fixedSlidingReplacement swaps the model's template caches for sized
// FixedKVCaches when the fixed-cache regime applies: the model declares the
// fixed-sliding cache, the cache policy permits bounding, and a bound
// resolves (-context, or the zero-flag default in the default mode). Sliding
// layers clamp to their window (the bound gate); global layers carry the
// request size when known, else the bound.
func (m *Model) fixedSlidingReplacement(caches []Cache, requestFixedSize int) ([]Cache, bool) {
	if !fixedSlidingCacheEnabled() || !modelUsesFixedSlidingCache(m.model) {
		return nil, false
	}
	if m.cachePolicy == "full" {
		return nil, false
	}
	bound := m.contextLen
	if bound <= 0 {
		// Explicit paged mode without -context keeps its paged semantics;
		// only the default mode derives the zero-flag bound from the model.
		if KVCacheMode(m.cacheMode) == KVCacheModePaged {
			return nil, false
		}
		bound = m.defaultFixedCacheBound()
	}
	if bound <= 0 {
		return nil, false
	}
	fixedSize := fixedSlidingCacheSize(bound, requestFixedSize, m.fixedSlidingCacheSize)
	storageDType, hasStorageDType := parseKVCacheStorageDType(m.kvCacheStorageDType)
	for i := range caches {
		layerSize := fixedSize
		if layerMaxSize := replacementCacheMaxSize(caches[i], bound); fixedSlidingCacheBoundEnabled() && layerMaxSize > 0 {
			layerSize = min(layerSize, layerMaxSize)
		}
		if hasStorageDType {
			caches[i] = NewFixedKVCacheWithDType(layerSize, storageDType)
		} else {
			caches[i] = NewFixedKVCache(layerSize)
		}
	}
	return caches, true
}

func parseKVCacheStorageDType(value string) (DType, bool) {
	value = core.Lower(core.Trim(value))
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

// generationTokenBudget resolves how many tokens a request may generate. A
// caller-set MaxTokens (>0) is honoured verbatim — the caller's word, even past
// the context window (sliding-window models rotate). MaxTokens <= 0 means
// "generate to the model's context": the budget is the room left in the window
// (contextLength - promptLen), so the loop runs until EOS/stop or the context
// fills — never a hardcoded cap. Returns 0 when the prompt already fills the
// context or no context is known, so generation is bounded by truth, not a
// guessed default.
func generationTokenBudget(maxTokens, contextLength, promptLen int) int {
	if maxTokens > 0 {
		return maxTokens
	}
	if contextLength > promptLen {
		return contextLength - promptLen
	}
	return 0
}

func (m *Model) generationFixedSlidingCacheSize(promptTokens, maxTokens int) int {
	if m == nil || !fixedSlidingCacheEnabled() || promptTokens <= 0 || maxTokens <= 0 {
		return 0
	}
	if !m.fixedCacheRegimeActive() {
		return 0
	}
	size := promptTokens + maxTokens
	if size < promptTokens {
		return 0
	}
	return roundUpPositive(size, 32)
}

// fixedCacheRegimeActive reports whether generation caches run the sized
// fixed-cache shape: by model declaration in the default mode (zero-flag),
// or explicitly via -kv-cache paged with -context. Quantised and turbo cache
// modes keep their own storage strategies.
func (m *Model) fixedCacheRegimeActive() bool {
	if !modelUsesFixedSlidingCache(m.model) || m.cachePolicy == "full" {
		return false
	}
	switch KVCacheMode(m.cacheMode) {
	case KVCacheModeDefault:
		return true
	case KVCacheModePaged:
		return m.contextLen > 0
	default:
		return false
	}
}

// modelUsesFixedSlidingCache reports whether the loaded model declares the
// fixed-size sliding-window KV cache (FixedSlidingCacheModel) — the engine
// dispatches on the capability, not the model family.
func modelUsesFixedSlidingCache(model InternalModel) bool {
	cache, ok := model.(FixedSlidingCacheModel)
	return ok && cache.UsesFixedSlidingCache()
}

func fixedSlidingCacheSize(maxSize, requestSize, configuredSize int) int {
	if maxSize <= 0 {
		return maxSize
	}
	if configuredSize > 0 {
		return min(configuredSize, maxSize)
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
		if err := LastError(); err != nil {
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
