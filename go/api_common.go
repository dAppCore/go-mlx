// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"dappco.re/go/mlx/memory"
	// Note: AX-6 - time.Duration is part of the public Metrics API.
	"time"

	"dappco.re/go"
	"dappco.re/go/inference/parser"
	coreio "dappco.re/go/io"
	"dappco.re/go/mlx/lora"
	"dappco.re/go/mlx/probe"
)

const (
	// DefaultLocalContextLength bounds KV growth for local workstation runs.
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
	PromptTokens               int             `json:"prompt_tokens"`
	GeneratedTokens            int             `json:"generated_tokens"`
	PrefillDuration            time.Duration   `json:"prefill_duration"`
	DecodeDuration             time.Duration   `json:"decode_duration"`
	TotalDuration              time.Duration   `json:"total_duration"`
	PrefillTokensPerSec        float64         `json:"prefill_tokens_per_sec"`
	DecodeTokensPerSec         float64         `json:"decode_tokens_per_sec"`
	PeakMemoryBytes            uint64          `json:"peak_memory_bytes"`
	ActiveMemoryBytes          uint64          `json:"active_memory_bytes"`
	PromptCacheHits            int             `json:"prompt_cache_hits,omitempty"`
	PromptCacheMisses          int             `json:"prompt_cache_misses,omitempty"`
	PromptCacheHitTokens       int             `json:"prompt_cache_hit_tokens,omitempty"`
	PromptCacheMissTokens      int             `json:"prompt_cache_miss_tokens,omitempty"`
	PromptCacheRestoreDuration time.Duration   `json:"prompt_cache_restore_duration,omitempty"`
	Adapter                    lora.AdapterInfo `json:"adapter,omitempty"`
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
	return s != nil && s.Queries != nil && len(s.Queries) > 0
}

// ModelInfo describes a loaded model.
type ModelInfo struct {
	Architecture  string
	VocabSize     int
	NumLayers     int
	HiddenSize    int
	QuantBits     int
	QuantGroup    int
	ContextLength int
	Adapter       lora.AdapterInfo
}

// GenerateConfig holds generation parameters for the RFC-style root API.
type GenerateConfig struct {
	MaxTokens     int
	Temperature   float32
	TopK          int
	TopP          float32
	MinP          float32
	ReturnLogits  bool
	StopTokens    []int32
	RepeatPenalty float32
	ProbeSink     probe.Sink
	Thinking      parser.Config
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

// WithLogits requests classification logits when the called API supports them.
func WithLogits() GenerateOption {
	return func(c *GenerateConfig) { c.ReturnLogits = true }
}

// WithReturnLogits is an alias for WithLogits.
func WithReturnLogits() GenerateOption {
	return WithLogits()
}

// WithStopTokens sets token IDs that stop generation.
func WithStopTokens(ids ...int32) GenerateOption {
	return func(c *GenerateConfig) { c.StopTokens = ids }
}

// WithRepeatPenalty sets the repetition penalty.
func WithRepeatPenalty(p float32) GenerateOption {
	return func(c *GenerateConfig) { c.RepeatPenalty = p }
}

// WithProbeSink streams typed probe events during generation.
//
//	model.Generate(prompt, mlx.WithProbeSink(sink))
func WithProbeSink(sink probe.Sink) GenerateOption {
	return func(c *GenerateConfig) { c.ProbeSink = sink }
}

// WithProbeCallback streams typed probe events to a callback during generation.
//
//	model.Generate(prompt, mlx.WithProbeCallback(func(e probe.Event) { … }))
func WithProbeCallback(callback func(probe.Event)) GenerateOption {
	if callback == nil {
		return func(*GenerateConfig) {}
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
	ContextLength        int
	ParallelSlots        int
	PromptCache          bool
	PromptCacheMinTokens int
	Quantization         int
	Device               string
	AdapterPath          string
	Medium               coreio.Medium
	AutoMemoryPlan       bool
	MemoryPlan           *memory.Plan
	CachePolicy          memory.KVCachePolicy
	CacheMode            memory.KVCacheMode
	BatchSize            int
	PrefillChunkSize     int
	ExpectedQuantization int
	MemoryLimitBytes     uint64
	CacheLimitBytes      uint64
	WiredLimitBytes      uint64
}

// DefaultLoadConfig returns sensible defaults for root-package loading.
func DefaultLoadConfig() LoadConfig {
	return LoadConfig{
		ContextLength:        DefaultLocalContextLength,
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
	return func(c *LoadConfig) { c.ContextLength = n }
}

// WithParallelSlots bounds concurrent native inference calls for this model.
// 0 leaves the backend default unchanged.
func WithParallelSlots(n int) LoadOption {
	return func(c *LoadConfig) { c.ParallelSlots = n }
}

// WithPromptCache enables or disables exact token-prefix KV caching.
func WithPromptCache(enabled bool) LoadOption {
	return func(c *LoadConfig) { c.PromptCache = enabled }
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

// WithDevice selects the execution device: "gpu" or "cpu".
func WithDevice(device string) LoadOption {
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

// WithAutoMemoryPlan enables or disables measured-device runtime planning.
func WithAutoMemoryPlan(enabled bool) LoadOption {
	return func(c *LoadConfig) { c.AutoMemoryPlan = enabled }
}

// WithMemoryPlan applies an explicit memory plan instead of probing the device.
func WithMemoryPlan(plan memory.Plan) LoadOption {
	return func(c *LoadConfig) {
		cloned := plan
		c.MemoryPlan = &cloned
		c.AutoMemoryPlan = false
	}
}

// WithCachePolicy selects the KV cache policy used by the native backend.
func WithCachePolicy(policy memory.KVCachePolicy) LoadOption {
	return func(c *LoadConfig) { c.CachePolicy = policy }
}

// WithKVCacheMode selects the native KV cache storage mode.
func WithKVCacheMode(mode memory.KVCacheMode) LoadOption {
	return func(c *LoadConfig) { c.CacheMode = mode }
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

func applyLoadOptions(opts []LoadOption) LoadConfig {
	cfg := DefaultLoadConfig()
	for _, opt := range opts {
		opt(&cfg)
	}
	return cfg
}

func normalizeLoadConfig(cfg LoadConfig) (LoadConfig, error) {
	if cfg.ContextLength < 0 {
		return LoadConfig{}, core.NewError("mlx: context length must be >= 0")
	}
	if cfg.ParallelSlots < 0 {
		return LoadConfig{}, core.NewError("mlx: parallel slots must be >= 0")
	}
	if cfg.PromptCacheMinTokens < 0 {
		return LoadConfig{}, core.NewError("mlx: prompt cache minimum tokens must be >= 0")
	}
	if cfg.PromptCache && cfg.PromptCacheMinTokens == 0 {
		cfg.PromptCacheMinTokens = DefaultPromptCacheMinTokens
	}
	if cfg.Quantization < 0 {
		return LoadConfig{}, core.NewError("mlx: quantization bits must be >= 0")
	}
	if cfg.BatchSize < 0 {
		return LoadConfig{}, core.NewError("mlx: batch size must be >= 0")
	}
	if cfg.PrefillChunkSize < 0 {
		return LoadConfig{}, core.NewError("mlx: prefill chunk size must be >= 0")
	}
	if cfg.ExpectedQuantization < 0 {
		return LoadConfig{}, core.NewError("mlx: expected quantization bits must be >= 0")
	}
	switch cfg.CacheMode {
	case memory.KVCacheModeDefault, memory.KVCacheModeFP16, memory.KVCacheModeQ8, memory.KVCacheModeKQ8VQ4, memory.KVCacheModePaged:
	default:
		return LoadConfig{}, core.NewError("mlx: unsupported KV cache mode: " + string(cfg.CacheMode))
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
