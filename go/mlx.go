// SPDX-Licence-Identifier: EUPL-1.2

// Package mlx provides Apple Metal GPU inference via mlx-c bindings.
//
// This package implements the [inference.Backend] interface from
// dappco.re/go/inference for Apple Silicon (M1-M4) GPUs.
// Import it blank to register the "metal" backend automatically:
//
//	import _ "dappco.re/go/mlx"
//
// Build mlx-c before use:
//
//	go generate ./...
//
// # Generate text
//
//	model, err := inference.LoadModel("/path/to/model/")
//	if err != nil { log.Fatal(err) }
//	defer model.Close()
//
//	ctx := context.Background()
//	for token := range model.Generate(ctx, "What is 2+2?", inference.WithMaxTokens(128)) {
//	    fmt.Print(token.Text)
//	}
//	if err := model.Err(); err != nil { log.Fatal(err) }
//
// # Multi-turn chat
//
// Chat applies the model's native template (Gemma3, Qwen3, Llama3):
//
//	for token := range model.Chat(ctx, []inference.Message{
//	    {Role: "system", Content: "You are a helpful assistant."},
//	    {Role: "user", Content: "Translate 'hello' to French."},
//	}, inference.WithMaxTokens(64)) {
//	    fmt.Print(token.Text)
//	}
//
// # Batch classification
//
// Classify runs a single forward pass per prompt (prefill only, no decoding):
//
//	results, err := model.Classify(ctx, []string{
//	    "Bonjour, comment allez-vous?",
//	    "The quarterly report shows growth.",
//	}, inference.WithTemperature(0))
//	for index, result := range results {
//	    fmt.Printf("prompt %d → %q\n", index, result.Token.Text)
//	}
//
// # Batch generation
//
//	results, err := model.BatchGenerate(ctx, []string{
//	    "The capital of France is",
//	    "Water boils at",
//	}, inference.WithMaxTokens(32))
//	for index, result := range results {
//	    for _, token := range result.Tokens {
//	        fmt.Print(token.Text)
//	    }
//	    fmt.Println()
//	}
//
// # Performance metrics
//
// After any inference call, retrieve timing and memory statistics:
//
//	for token := range model.Generate(ctx, prompt, inference.WithMaxTokens(128)) {
//	    fmt.Print(token.Text)
//	}
//	metrics := model.Metrics()
//	fmt.Printf("decode: %.0f tok/s, peak GPU: %d MB\n",
//	    metrics.DecodeTokensPerSec, metrics.PeakMemoryBytes/1024/1024)
//
// # Model info
//
//	modelInfo := model.Info()
//	fmt.Printf("%s %d-layer, %d-bit quantised\n",
//	    modelInfo.Architecture, modelInfo.NumLayers, modelInfo.QuantBits)
//
// # Model discovery
//
//	discoveredModels, err := inference.Discover("/path/to/models/")
//	for _, discoveredModel := range discoveredModels {
//	    fmt.Printf("%s (%s, %d-bit)\n", discoveredModel.Path, discoveredModel.ModelType, discoveredModel.QuantBits)
//	}
//
// # Metal memory controls
//
// These control the Metal allocator directly, not individual models:
//
//	mlx.SetCacheLimit(4 << 30)  // 4 GB cache limit
//	mlx.SetMemoryLimit(32 << 30) // 32 GB hard limit
//
//	// Between chat turns, reclaim prompt cache memory:
//	mlx.ClearCache()
//	model1.Close()
//	mlx.GC() // run Go finalizers for CGO-owned memory without importing runtime
//
//	fmt.Printf("active: %d MB, peak: %d MB\n",
//	    mlx.GetActiveMemory()/1024/1024, mlx.GetPeakMemory()/1024/1024)
package mlx

import (
	// Note: AX-6 - time.Duration is part of the public Metrics API.
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/parser"
	coreio "dappco.re/go/io"
	"dappco.re/go/mlx/lora"
	"dappco.re/go/mlx/memory"
	"dappco.re/go/mlx/pkg/metal"
	"dappco.re/go/mlx/probe"
)

//go:generate cmake -S . -B build -DCMAKE_INSTALL_PREFIX=dist -DCMAKE_BUILD_TYPE=Release
//go:generate cmake --build build --parallel
//go:generate cmake --install build

// GC runs Go garbage collection for MLX CGO lifecycle cleanup.
//
// Use this after closing large models when prompt/model memory must be
// reclaimed promptly, without importing runtime at call sites.
func GC() { metal.RuntimeGC() }

// SeedRandom resets MLX's default random sequence for subsequent sampling.
func SeedRandom(seed uint64) error { return metal.SeedRandom(seed) }

const (
	// DefaultLocalContextLength is the opt-in local cap used by production
	// lanes and explicit workstation profiles. Default loads leave context
	// length at 0 so the native model metadata can supply the full window.
	DefaultLocalContextLength = 131072
	// DefaultLocalParallelSlots keeps one foreground native request active.
	DefaultLocalParallelSlots = 1
	// DefaultPromptCacheMinTokens avoids cache overhead for short prompts.
	DefaultPromptCacheMinTokens = 2048
)

// Token is a generated token from the RFC-style root API.
type Token struct {
	ID    int32
	Value string
	Text  string
}

// Metrics reports performance counters from the last inference call.
type Metrics struct {
	PromptTokens               int                          `json:"prompt_tokens"`
	GeneratedTokens            int                          `json:"generated_tokens"`
	FirstTokenDuration         time.Duration                `json:"first_token_duration,omitempty"`
	PrefillDuration            time.Duration                `json:"prefill_duration"`
	DecodeDuration             time.Duration                `json:"decode_duration"`
	TotalDuration              time.Duration                `json:"total_duration"`
	PrefillTokensPerSec        float64                      `json:"prefill_tokens_per_sec"`
	DecodeTokensPerSec         float64                      `json:"decode_tokens_per_sec"`
	PeakMemoryBytes            uint64                       `json:"peak_memory_bytes"`
	ActiveMemoryBytes          uint64                       `json:"active_memory_bytes"`
	CacheMemoryBytes           uint64                       `json:"cache_memory_bytes"`
	ProcessVirtualMemoryBytes  uint64                       `json:"process_virtual_memory_bytes"`
	ProcessResidentMemoryBytes uint64                       `json:"process_resident_memory_bytes"`
	ProcessPeakResidentBytes   uint64                       `json:"process_peak_resident_bytes"`
	PromptCacheHits            int                          `json:"prompt_cache_hits,omitempty"`
	PromptCacheMisses          int                          `json:"prompt_cache_misses,omitempty"`
	PromptCacheHitTokens       int                          `json:"prompt_cache_hit_tokens,omitempty"`
	PromptCacheMissTokens      int                          `json:"prompt_cache_miss_tokens,omitempty"`
	PromptCacheRestoreDuration time.Duration                `json:"prompt_cache_restore_duration,omitempty"`
	CacheProfile               *CacheProfile                `json:"cache_profile,omitempty"`
	TurboQuantKVPayload        *TurboQuantKVPayloadEstimate `json:"turboquant_kv_payload,omitempty"`
	TokenPhases                []TokenPhaseTrace            `json:"token_phases,omitempty"`
	MTP                        *MTPMetrics                  `json:"mtp,omitempty"`
	Adapter                    lora.AdapterInfo             `json:"adapter"`
}

// TurboQuantKVPayloadEstimate summarises the compressed TurboQuant K/V payload
// currently retained by a generation cache. PayloadBytes is section data before
// alignment padding; PaddedPayloadBytes is the actual retained binary span.
type TurboQuantKVPayloadEstimate struct {
	Pages                     int     `json:"pages"`
	PageVectors               uint64  `json:"page_vectors,omitempty"`
	PageElements              uint64  `json:"page_elements,omitempty"`
	KeyCentroidBytes          uint64  `json:"key_centroid_bytes,omitempty"`
	KeyQJLSignBytes           uint64  `json:"key_qjl_sign_bytes,omitempty"`
	KeyNormBytes              uint64  `json:"key_norm_bytes,omitempty"`
	KeyResidualNormBytes      uint64  `json:"key_residual_norm_bytes,omitempty"`
	ValueCentroidBytes        uint64  `json:"value_centroid_bytes,omitempty"`
	ValueNormBytes            uint64  `json:"value_norm_bytes,omitempty"`
	OutlierMaskBytes          uint64  `json:"outlier_mask_bytes,omitempty"`
	PayloadBytes              uint64  `json:"payload_bytes,omitempty"`
	PaddedPayloadBytes        uint64  `json:"padded_payload_bytes,omitempty"`
	AlignmentPaddingBytes     uint64  `json:"alignment_padding_bytes,omitempty"`
	FP16BaselineBytes         uint64  `json:"fp16_baseline_bytes,omitempty"`
	PayloadToFP16Ratio        float64 `json:"payload_to_fp16_ratio,omitempty"`
	PaddedPayloadToFP16Ratio  float64 `json:"padded_payload_to_fp16_ratio,omitempty"`
	PayloadSavingsRatio       float64 `json:"payload_savings_ratio,omitempty"`
	PaddedPayloadSavingsRatio float64 `json:"padded_payload_savings_ratio,omitempty"`
}

// MTPMetrics records attached multi-token-prediction drafter counters.
type MTPMetrics struct {
	DraftTokenSchedule     []int         `json:"draft_token_schedule,omitempty"`
	ProposedTokens         int           `json:"proposed_tokens,omitempty"`
	AcceptedTokens         int           `json:"accepted_tokens,omitempty"`
	RejectedTokens         int           `json:"rejected_tokens,omitempty"`
	TargetVerifyCalls      int           `json:"target_verify_calls,omitempty"`
	TargetCalls            int           `json:"target_calls,omitempty"`
	DraftCalls             int           `json:"draft_calls,omitempty"`
	AcceptanceRate         float64       `json:"acceptance_rate,omitempty"`
	VisibleTokensPerSec    float64       `json:"visible_tokens_per_sec,omitempty"`
	TargetTokensPerSec     float64       `json:"target_tokens_per_sec,omitempty"`
	WarmDecodeTokensPerSec float64       `json:"warm_decode_tokens_per_sec,omitempty"`
	WallDuration           time.Duration `json:"wall_duration,omitempty"`
	RestoreDuration        time.Duration `json:"restore_duration,omitempty"`
	TargetVerifyDuration   time.Duration `json:"target_verify_duration,omitempty"`
	TargetDuration         time.Duration `json:"target_duration,omitempty"`
	DraftDuration          time.Duration `json:"draft_duration,omitempty"`
	PeakMemoryBytes        uint64        `json:"peak_memory_bytes,omitempty"`
}

// CacheProfile reports the model/cache topology observed after a generation
// turn. Gemma 4 uses this to prove local sliding caches stay bounded while
// global owner layers carry the retained long-context state.
type CacheProfile struct {
	Architecture       string `json:"architecture,omitempty"`
	TotalCaches        int    `json:"total_caches"`
	LocalCaches        int    `json:"local_caches"`
	GlobalCaches       int    `json:"global_caches"`
	SharedLayers       int    `json:"shared_layers"`
	CachelessLayers    int    `json:"cacheless_layers"`
	LocalWindowTokens  int    `json:"local_window_tokens"`
	MaxLocalTokens     int    `json:"max_local_tokens"`
	MaxLocalCapacity   int    `json:"max_local_capacity"`
	MaxGlobalTokens    int    `json:"max_global_tokens"`
	MaxGlobalCapacity  int    `json:"max_global_capacity"`
	MaxCacheTokens     int    `json:"max_cache_tokens"`
	MaxCacheCapacity   int    `json:"max_cache_capacity"`
	MaxProcessedTokens int    `json:"max_processed_tokens"`
	FullCaches         int    `json:"full_caches"`
	RotatingCaches     int    `json:"rotating_caches"`
	FixedCaches        int    `json:"fixed_caches"`
	PagedCaches        int    `json:"paged_caches"`
	QuantizedCaches    int    `json:"quantized_caches"`
	UnknownCaches      int    `json:"unknown_caches"`
	UnboundedCaches    int    `json:"unbounded_caches"`
	LocalWindowLeaked  bool   `json:"local_window_leaked"`
}

// TokenPhaseTrace reports the coarse decode-loop cost for one generated token.
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

// NativePhaseTrace reports an optional native materialisation event captured
// during a decode forward pass.
type NativePhaseTrace struct {
	Name     string        `json:"name"`
	Duration time.Duration `json:"duration"`
	Error    string        `json:"error,omitempty"`
	Pages    int           `json:"pages,omitempty"`
	Tokens   int           `json:"tokens,omitempty"`
}

// ClassifyResult holds the sampled token for a single prompt and optional logits.
type ClassifyResult struct {
	Token  Token
	Logits []float32
}

// BatchResult holds the streamed tokens for a single prompt in a batch call.
type BatchResult struct {
	Tokens []Token
	Err    error
}

// AttentionSnapshot contains post-RoPE key tensors extracted from KV caches.
type AttentionSnapshot struct {
	NumLayers     int
	NumHeads      int
	SeqLen        int
	HeadDim       int
	NumQueryHeads int
	Keys          [][][]float32
	Queries       [][][]float32
	Architecture  string
}

// HasQueries reports whether query tensors are present in the snapshot.
func (s *AttentionSnapshot) HasQueries() bool {
	// len(nil) == 0 — the explicit s.Queries != nil check is redundant,
	// and dropping it lets the inliner fold the single bounds load into
	// a fused nil-check + length compare instead of a three-step chain.
	return s != nil && len(s.Queries) > 0
}

// ModelInfo describes a loaded model.
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
	ParallelSlots         int
	PromptCache           bool
	PromptCacheMinTokens  int
	CachePolicy           memory.KVCachePolicy
	CacheMode             memory.KVCacheMode
	KVCacheStorageDType   string
	PagedKVPageSize       int
	PagedKVPrealloc       bool
	FixedSlidingCacheSize int
	BatchSize             int
	PrefillChunkSize      int
	ExpectedQuantization  int
	MemoryLimitBytes      uint64
	CacheLimitBytes       uint64
	WiredLimitBytes       uint64
	Adapter               lora.AdapterInfo
}

// GenerateConfig holds generation parameters for the RFC-style root API.
type GenerateConfig struct {
	MaxTokens                    int
	Temperature                  float32
	TopK                         int
	TopP                         float32
	MinP                         float32
	Seed                         uint64
	SeedSet                      bool
	ReturnLogits                 bool
	StopTokens                   []int32
	SuppressTokens               []int32
	MinTokensBeforeStop          int
	RepeatPenalty                float32
	ProbeSink                    probe.Sink
	TraceTokenPhases             bool
	TraceTokenText               bool
	GenerationClearCache         bool
	GenerationClearCacheInterval int
	Thinking                     parser.Config
}

// DefaultGenerateConfig returns sensible defaults for root-package generation.
func DefaultGenerateConfig() GenerateConfig {
	return GenerateConfig{
		MaxTokens:   256,
		Temperature: 0.0,
		Thinking:    parser.Config{Mode: parser.Show},
	}
}

// GenerateOption configures root-package text generation.
type GenerateOption func(*GenerateConfig)

// WithMaxTokens sets the maximum number of tokens to generate.
func WithMaxTokens(n int) GenerateOption {
	return func(c *GenerateConfig) { c.MaxTokens = n }
}

// WithTemperature sets the sampling temperature. 0 = greedy.
func WithTemperature(t float32) GenerateOption {
	return func(c *GenerateConfig) { c.Temperature = t }
}

// WithTopK sets top-k sampling. 0 = disabled.
func WithTopK(k int) GenerateOption {
	return func(c *GenerateConfig) { c.TopK = k }
}

// WithTopP sets nucleus sampling. 0 = disabled.
func WithTopP(p float32) GenerateOption {
	return func(c *GenerateConfig) { c.TopP = p }
}

// WithMinP sets minimum-probability sampling relative to the best token.
func WithMinP(p float32) GenerateOption {
	return func(c *GenerateConfig) { c.MinP = p }
}

// WithSeed resets MLX's default RNG before this generation call.
func WithSeed(seed uint64) GenerateOption {
	return func(c *GenerateConfig) {
		c.Seed = seed
		c.SeedSet = true
	}
}

// withLogitsOption / withTokenPhaseTraceOption are the package-init
// singleton closures returned by every WithLogits / WithReturnLogits /
// WithTokenPhaseTrace call. The no-argument option builders captured
// nothing, so the prior `return func(...){...}` form heap-allocated a
// fresh closure on every call — measurable in the option-stack bench
// because every Generate call site that asks for logits walks through
// this builder. Hoisting the closure once at package init makes the
// builder a pure pointer return, dropping the alloc to zero.
var (
	withLogitsOption          GenerateOption = func(c *GenerateConfig) { c.ReturnLogits = true }
	withTokenPhaseTraceOption GenerateOption = func(c *GenerateConfig) { c.TraceTokenPhases = true }
	withTokenPhaseTextOption  GenerateOption = func(c *GenerateConfig) {
		c.TraceTokenPhases = true
		c.TraceTokenText = true
	}
)

// WithLogits requests classification logits when the called API supports them.
func WithLogits() GenerateOption {
	return withLogitsOption
}

// WithReturnLogits is an alias for WithLogits.
func WithReturnLogits() GenerateOption {
	return withLogitsOption
}

// WithStopTokens sets token IDs that stop generation.
func WithStopTokens(ids ...int32) GenerateOption {
	return func(c *GenerateConfig) { c.StopTokens = ids }
}

// WithSuppressTokens masks token IDs out of the sampling distribution.
func WithSuppressTokens(ids ...int32) GenerateOption {
	return func(c *GenerateConfig) { c.SuppressTokens = ids }
}

// WithMinTokensBeforeStop masks stop tokens until n real tokens have been
// emitted, then restores normal stop behaviour.
func WithMinTokensBeforeStop(n int) GenerateOption {
	return func(c *GenerateConfig) { c.MinTokensBeforeStop = n }
}

// WithRepeatPenalty sets the repetition penalty.
func WithRepeatPenalty(p float32) GenerateOption {
	return func(c *GenerateConfig) { c.RepeatPenalty = p }
}

// WithGenerationClearCacheInterval sets the decode-token interval used when
// generation clear-cache mode is enabled. 0 leaves the backend default.
func WithGenerationClearCacheInterval(n int) GenerateOption {
	return func(c *GenerateConfig) { c.GenerationClearCacheInterval = n }
}

// WithGenerationClearCache clears the native allocator cache after prefill and
// periodically during decode for this request.
func WithGenerationClearCache() GenerateOption {
	return func(c *GenerateConfig) { c.GenerationClearCache = true }
}

// WithTokenPhaseTrace records per-token decode-loop timings in Metrics.
func WithTokenPhaseTrace() GenerateOption {
	return withTokenPhaseTraceOption
}

// WithTokenPhaseTraceText records decoded token text alongside phase timings.
func WithTokenPhaseTraceText() GenerateOption {
	return withTokenPhaseTextOption
}

// withNoopGenerateOption is the no-op closure returned by WithProbeSink and
// WithProbeCallback when the caller passes a nil sink/callback. Sharing one
// package-init function value eliminates the per-call empty-closure alloc
// the prior `return func(*GenerateConfig) {}` form re-emitted, matching the
// withLogitsOption / withTokenPhaseTraceOption pattern above.
var withNoopGenerateOption GenerateOption = func(*GenerateConfig) {}

// WithProbeSink streams typed probe events during generation.
//
//	model.Generate(prompt, mlx.WithProbeSink(sink))
func WithProbeSink(sink probe.Sink) GenerateOption {
	if sink == nil {
		return withNoopGenerateOption
	}
	return func(c *GenerateConfig) { c.ProbeSink = sink }
}

// WithProbeCallback streams typed probe events to a callback during generation.
//
//	model.Generate(prompt, mlx.WithProbeCallback(func(e probe.Event) { … }))
func WithProbeCallback(callback func(probe.Event)) GenerateOption {
	if callback == nil {
		return withNoopGenerateOption
	}
	return WithProbeSink(probe.SinkFunc(callback))
}

func applyGenerateOptions(opts []GenerateOption) GenerateConfig {
	cfg := DefaultGenerateConfig()
	for _, opt := range opts {
		opt(&cfg)
	}
	return cfg
}

// LoadConfig holds root-package model loading parameters.
type LoadConfig struct {
	ContextLength         int
	ParallelSlots         int
	PromptCache           bool
	PromptCacheMinTokens  int
	Quantization          int
	Device                string
	AdapterPath           string
	Medium                coreio.Medium
	AutoMemoryPlan        bool
	MemoryPlan            *memory.Plan
	CachePolicy           memory.KVCachePolicy
	CacheMode             memory.KVCacheMode
	KVCacheStorageDType   string
	PagedKVPageSize       int
	PagedKVPrealloc       bool
	FixedSlidingCacheSize int
	BatchSize             int
	PrefillChunkSize      int
	ExpectedQuantization  int
	MemoryLimitBytes      uint64
	CacheLimitBytes       uint64
	WiredLimitBytes       uint64
	SplitInference        *inference.SplitInferencePlan
	contextLengthExplicit bool
}

// DefaultLoadConfig returns sensible defaults for root-package loading.
func DefaultLoadConfig() LoadConfig {
	return LoadConfig{
		ParallelSlots:        DefaultLocalParallelSlots,
		PromptCache:          true,
		PromptCacheMinTokens: DefaultPromptCacheMinTokens,
		Device:               "gpu",
		AutoMemoryPlan:       true,
	}
}

// LoadOption configures root-package model loading.
type LoadOption func(*LoadConfig)

// WithContextLength bounds the KV cache to the given context window.
func WithContextLength(n int) LoadOption {
	return func(c *LoadConfig) {
		c.ContextLength = n
		c.contextLengthExplicit = n > 0
	}
}

// WithParallelSlots bounds concurrent native inference calls for this model.
// 0 leaves the backend default unchanged.
func WithParallelSlots(n int) LoadOption {
	return func(c *LoadConfig) { c.ParallelSlots = n }
}

// withPromptCacheEnabledOption / withPromptCacheDisabledOption are the two
// package-init singleton closures returned by WithPromptCache. The builder
// only takes a bool so the value space is exhausted by two pre-built
// closures, dropping the per-call alloc to zero and matching the Wave 5
// switch-cached static closure pattern (finite-domain builders return a
// pointer to a pre-existing closure instead of constructing a new one).
var (
	withPromptCacheEnabledOption  LoadOption = func(c *LoadConfig) { c.PromptCache = true }
	withPromptCacheDisabledOption LoadOption = func(c *LoadConfig) { c.PromptCache = false }
)

// WithPromptCache enables or disables exact token-prefix KV caching.
func WithPromptCache(enabled bool) LoadOption {
	if enabled {
		return withPromptCacheEnabledOption
	}
	return withPromptCacheDisabledOption
}

// WithPromptCacheMinTokens sets the minimum prefix length considered cacheable.
func WithPromptCacheMinTokens(n int) LoadOption {
	return func(c *LoadConfig) { c.PromptCacheMinTokens = n }
}

// WithQuantization validates the loaded quantisation width.
func WithQuantization(bits int) LoadOption {
	return func(c *LoadConfig) { c.Quantization = bits }
}

// WithExpectedQuantization tells the native loader which quantisation width the
// planner expects before post-load validation can inspect model metadata.
func WithExpectedQuantization(bits int) LoadOption {
	return func(c *LoadConfig) { c.ExpectedQuantization = bits }
}

// withDeviceGPUOption / withDeviceCPUOption short-cut the two canonical
// device values WithDevice receives in 99% of caller paths. The string
// space is theoretically open (callers can pass any string and have
// normalizeLoadConfig reject it), but the package-level singleton
// closures eliminate the per-call alloc for the two values that actually
// reach this builder — matching the Wave 5 switch-cached static closure
// pattern. The default branch preserves the original semantics for the
// fallback path.
var (
	withDeviceGPUOption LoadOption = func(c *LoadConfig) { c.Device = "gpu" }
	withDeviceCPUOption LoadOption = func(c *LoadConfig) { c.Device = "cpu" }
)

// WithDevice selects the execution device: "gpu" or "cpu".
func WithDevice(device string) LoadOption {
	switch device {
	case "gpu":
		return withDeviceGPUOption
	case "cpu":
		return withDeviceCPUOption
	}
	return func(c *LoadConfig) { c.Device = device }
}

// WithAdapterPath injects a LoRA adapter directory at model load time.
func WithAdapterPath(path string) LoadOption {
	return func(c *LoadConfig) { c.AdapterPath = path }
}

// WithMedium stages model files from the supplied io.Medium before loading.
// The model path passed to LoadModel is interpreted within that medium.
func WithMedium(medium coreio.Medium) LoadOption {
	return func(c *LoadConfig) { c.Medium = medium }
}

// withAutoMemoryPlanEnabledOption / withAutoMemoryPlanDisabledOption are the
// pre-built closures returned by WithAutoMemoryPlan — same switch-cached
// finite-domain pattern as withPromptCacheEnabledOption.
var (
	withAutoMemoryPlanEnabledOption  LoadOption = func(c *LoadConfig) { c.AutoMemoryPlan = true }
	withAutoMemoryPlanDisabledOption LoadOption = func(c *LoadConfig) { c.AutoMemoryPlan = false }
)

// WithAutoMemoryPlan enables or disables measured-device runtime planning.
func WithAutoMemoryPlan(enabled bool) LoadOption {
	if enabled {
		return withAutoMemoryPlanEnabledOption
	}
	return withAutoMemoryPlanDisabledOption
}

// WithMemoryPlan applies an explicit memory plan instead of probing the device.
func WithMemoryPlan(plan memory.Plan) LoadOption {
	return func(c *LoadConfig) {
		cloned := plan
		c.MemoryPlan = &cloned
		c.AutoMemoryPlan = false
	}
}

// withCachePolicy*Option singletons exhaust the memory.KVCachePolicy
// constant set ("", "rotating", "full"). Returning the pre-built closure
// for each known value drops the WithCachePolicy alloc to zero on the
// option-stack hot path — same pattern as withPromptCache*Option.
var (
	withCachePolicyDefaultOption  LoadOption = func(c *LoadConfig) { c.CachePolicy = memory.KVCacheDefault }
	withCachePolicyRotatingOption LoadOption = func(c *LoadConfig) { c.CachePolicy = memory.KVCacheRotating }
	withCachePolicyFullOption     LoadOption = func(c *LoadConfig) { c.CachePolicy = memory.KVCacheFull }
)

// WithCachePolicy selects the KV cache policy used by the native backend.
func WithCachePolicy(policy memory.KVCachePolicy) LoadOption {
	switch policy {
	case memory.KVCacheDefault:
		return withCachePolicyDefaultOption
	case memory.KVCacheRotating:
		return withCachePolicyRotatingOption
	case memory.KVCacheFull:
		return withCachePolicyFullOption
	}
	return func(c *LoadConfig) { c.CachePolicy = policy }
}

// withCacheMode*Option singletons exhaust the memory.KVCacheMode constant
// set ("", "fp16", "q8", "k-q8-v-q4", "paged", "turboquant"). Each known mode returns the
// pre-built closure so WithKVCacheMode allocates nothing on the canonical
// caller paths — same finite-domain pattern as withCachePolicy*Option.
var (
	withCacheModeDefaultOption    LoadOption = func(c *LoadConfig) { c.CacheMode = memory.KVCacheModeDefault }
	withCacheModeFP16Option       LoadOption = func(c *LoadConfig) { c.CacheMode = memory.KVCacheModeFP16 }
	withCacheModeQ8Option         LoadOption = func(c *LoadConfig) { c.CacheMode = memory.KVCacheModeQ8 }
	withCacheModeKQ8VQ4Option     LoadOption = func(c *LoadConfig) { c.CacheMode = memory.KVCacheModeKQ8VQ4 }
	withCacheModePagedOption      LoadOption = func(c *LoadConfig) { c.CacheMode = memory.KVCacheModePaged }
	withCacheModeTurboQuantOption LoadOption = func(c *LoadConfig) { c.CacheMode = memory.KVCacheModeTurboQuant }
)

// WithKVCacheMode selects the native KV cache storage mode.
func WithKVCacheMode(mode memory.KVCacheMode) LoadOption {
	switch mode {
	case memory.KVCacheModeDefault:
		return withCacheModeDefaultOption
	case memory.KVCacheModeFP16:
		return withCacheModeFP16Option
	case memory.KVCacheModeQ8:
		return withCacheModeQ8Option
	case memory.KVCacheModeKQ8VQ4:
		return withCacheModeKQ8VQ4Option
	case memory.KVCacheModePaged:
		return withCacheModePagedOption
	case memory.KVCacheModeTurboQuant:
		return withCacheModeTurboQuantOption
	}
	return func(c *LoadConfig) { c.CacheMode = mode }
}

// WithKVCacheStorageDType selects the native retained KV storage dtype for
// cache implementations that support typed storage. "" leaves backend-native
// storage.
func WithKVCacheStorageDType(dtype string) LoadOption {
	switch dtype {
	case "", "native", "default":
		return func(c *LoadConfig) { c.KVCacheStorageDType = "" }
	case "fp16", "bf16":
		return func(c *LoadConfig) { c.KVCacheStorageDType = dtype }
	}
	return func(c *LoadConfig) { c.KVCacheStorageDType = dtype }
}

// WithPagedKVPageSize selects the page size for native paged KV caches.
// 0 leaves the backend default.
func WithPagedKVPageSize(n int) LoadOption {
	return func(c *LoadConfig) { c.PagedKVPageSize = n }
}

// WithPagedKVPrealloc selects full-page preallocation for native paged KV
// caches. This is a memory-residency diagnostic option, not a default speed
// path; use only when the lower active+cache footprint is worth the decode cost.
func WithPagedKVPrealloc(enabled bool) LoadOption {
	return func(c *LoadConfig) { c.PagedKVPrealloc = enabled }
}

// WithFixedSlidingCacheSize selects an explicit fixed Gemma 4 KV cache size.
// 0 leaves the backend to derive the size from context or request shape.
func WithFixedSlidingCacheSize(n int) LoadOption {
	return func(c *LoadConfig) { c.FixedSlidingCacheSize = n }
}

// WithBatchSize sets the planner batch shape for native batched generation.
func WithBatchSize(n int) LoadOption {
	return func(c *LoadConfig) { c.BatchSize = n }
}

// WithPrefillChunkSize bounds long prompt prefill passes into token chunks.
func WithPrefillChunkSize(n int) LoadOption {
	return func(c *LoadConfig) { c.PrefillChunkSize = n }
}

// WithAllocatorLimits applies Metal allocator limits in bytes.
func WithAllocatorLimits(memory, cache, wired uint64) LoadOption {
	return func(c *LoadConfig) {
		c.MemoryLimitBytes = memory
		c.CacheLimitBytes = cache
		c.WiredLimitBytes = wired
	}
}

// WithSplitInference attaches a validated split-inference plan to the load
// request. Remote execution is still planned; local plans are accepted so UIs
// can persist the same shape before backend execution lands.
func WithSplitInference(plan inference.SplitInferencePlan) LoadOption {
	return func(c *LoadConfig) {
		c.SplitInference = cloneSplitInferencePlan(plan)
	}
}

func applyLoadOptions(opts []LoadOption) LoadConfig {
	cfg := DefaultLoadConfig()
	for _, opt := range opts {
		opt(&cfg)
	}
	return cfg
}

// normalizeLoadConfig validation errors hoisted to package vars — the
// failure paths are rare in callers but each core.NewError() allocates
// a fresh error value; reusing a single instance per message keeps the
// rare path alloc-free and preserves errors.Is comparability.
var (
	errMlxContextLengthNegative    = core.NewError("mlx: context length must be >= 0")
	errMlxParallelSlotsNegative    = core.NewError("mlx: parallel slots must be >= 0")
	errMlxPromptCacheMinTokensNeg  = core.NewError("mlx: prompt cache minimum tokens must be >= 0")
	errMlxQuantizationNegative     = core.NewError("mlx: quantization bits must be >= 0")
	errMlxBatchSizeNegative        = core.NewError("mlx: batch size must be >= 0")
	errMlxPrefillChunkSizeNegative = core.NewError("mlx: prefill chunk size must be >= 0")
	errMlxExpectedQuantizationNeg  = core.NewError("mlx: expected quantization bits must be >= 0")
	errMlxSplitInferenceRemotePlan = core.NewError("mlx: split inference execution is planned; remote FFN/expert execution is not wired yet")
)

func normalizeLoadConfig(cfg LoadConfig) (LoadConfig, error) {
	if cfg.ContextLength < 0 {
		return LoadConfig{}, errMlxContextLengthNegative
	}
	if cfg.ParallelSlots < 0 {
		return LoadConfig{}, errMlxParallelSlotsNegative
	}
	if cfg.PromptCacheMinTokens < 0 {
		return LoadConfig{}, errMlxPromptCacheMinTokensNeg
	}
	if cfg.PromptCache && cfg.PromptCacheMinTokens == 0 {
		cfg.PromptCacheMinTokens = DefaultPromptCacheMinTokens
	}
	if cfg.Quantization < 0 {
		return LoadConfig{}, errMlxQuantizationNegative
	}
	if cfg.BatchSize < 0 {
		return LoadConfig{}, errMlxBatchSizeNegative
	}
	if cfg.PrefillChunkSize < 0 {
		return LoadConfig{}, errMlxPrefillChunkSizeNegative
	}
	if cfg.ExpectedQuantization < 0 {
		return LoadConfig{}, errMlxExpectedQuantizationNeg
	}
	if cfg.PagedKVPageSize < 0 {
		return LoadConfig{}, core.NewError("mlx: paged KV page size must be >= 0")
	}
	if cfg.FixedSlidingCacheSize < 0 {
		return LoadConfig{}, core.NewError("mlx: fixed Gemma 4 cache size must be >= 0")
	}
	if cfg.SplitInference != nil {
		if err := inference.ValidateSplitInferencePlan(*cfg.SplitInference); err != nil {
			return LoadConfig{}, err
		}
		mode := cfg.SplitInference.Mode
		if mode == "" {
			mode = inference.SplitInferenceModeLocal
		}
		if mode != inference.SplitInferenceModeLocal {
			return LoadConfig{}, errMlxSplitInferenceRemotePlan
		}
	}
	if !memory.IsKnownKVCacheMode(cfg.CacheMode) {
		return LoadConfig{}, core.NewError("mlx: unsupported KV cache mode: " + string(cfg.CacheMode))
	}
	cfg.KVCacheStorageDType = normalizeKVCacheStorageDType(cfg.KVCacheStorageDType)
	if cfg.KVCacheStorageDType == "unsupported" {
		return LoadConfig{}, core.NewError("mlx: unsupported KV cache storage dtype")
	}

	// Fast-path the canonical "", "gpu", "cpu" values that the default
	// LoadConfig and almost every caller provide. core.Lower/Trim each
	// walk the string and Trim allocates a fresh substring for any
	// whitespace input, which dominates a 90%-clean hot path. Skip both
	// scans when the input is already canonical and only fall through
	// to the normalising slow path when the device string actually
	// needs work.
	switch cfg.Device {
	case "gpu", "cpu":
		return cfg, nil
	case "":
		cfg.Device = "gpu"
		return cfg, nil
	}
	device := core.Lower(core.Trim(cfg.Device))
	if device == "" {
		device = "gpu"
	}
	switch device {
	case "gpu", "cpu":
		cfg.Device = device
		return cfg, nil
	default:
		return LoadConfig{}, core.NewError("mlx: unsupported device: " + device)
	}
}

func normalizeKVCacheStorageDType(dtype string) string {
	switch core.Lower(core.Trim(dtype)) {
	case "", "native", "default":
		return ""
	case "fp16", "float16", "f16":
		return "fp16"
	case "bf16", "bfloat16":
		return "bf16"
	default:
		return "unsupported"
	}
}

func cloneSplitInferencePlan(plan inference.SplitInferencePlan) *inference.SplitInferencePlan {
	// plan is already a value-copy taken on parameter receive — mutating
	// its slice/map fields in place builds the cloned shape without the
	// extra `cloned := plan` struct-copy the prior form paid. Returning
	// &plan escapes the parameter to heap, replacing the two-copy
	// (parameter + cloned local) pattern with one heap-allocated value.
	//
	// core.SliceClone still short-circuits to nil for nil-input slices,
	// keeping the typical "Components present, Notes empty" plan shape
	// alloc-light for the slice/map sub-fields.
	plan.LocalSlice.Components = core.SliceClone(plan.LocalSlice.Components)
	plan.LocalSlice.Notes = core.SliceClone(plan.LocalSlice.Notes)
	plan.LocalSlice.Labels = cloneInferenceLabels(plan.LocalSlice.Labels)
	plan.Endpoints = cloneInferenceSplitEndpoints(plan.Endpoints)
	plan.Labels = cloneInferenceLabels(plan.Labels)
	return &plan
}
