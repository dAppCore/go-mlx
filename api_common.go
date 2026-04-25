package mlx

import (
	// Note: AX-6 - time.Duration is part of the public Metrics API.
	"time"

	"dappco.re/go/core"
	coreio "dappco.re/go/core/io"
)

// Token is a generated token from the RFC-style root API.
type Token struct {
	ID    int32
	Value string
	Text  string
}

// Metrics reports performance counters from the last inference call.
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
}

// DefaultGenerateConfig returns sensible defaults for root-package generation.
func DefaultGenerateConfig() GenerateConfig {
	return GenerateConfig{
		MaxTokens:   256,
		Temperature: 0.0,
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

func applyGenerateOptions(opts []GenerateOption) GenerateConfig {
	cfg := DefaultGenerateConfig()
	for _, opt := range opts {
		opt(&cfg)
	}
	return cfg
}

// LoadConfig holds root-package model loading parameters.
type LoadConfig struct {
	ContextLength int
	Quantization  int
	Device        string
	AdapterPath   string
	Medium        coreio.Medium
}

// DefaultLoadConfig returns sensible defaults for root-package loading.
func DefaultLoadConfig() LoadConfig {
	return LoadConfig{Device: "gpu"}
}

// LoadOption configures root-package model loading.
type LoadOption func(*LoadConfig)

// WithContextLength bounds the KV cache to the given context window.
func WithContextLength(n int) LoadOption {
	return func(c *LoadConfig) { c.ContextLength = n }
}

// WithQuantization validates the loaded quantisation width.
func WithQuantization(bits int) LoadOption {
	return func(c *LoadConfig) { c.Quantization = bits }
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
	if cfg.Quantization < 0 {
		return LoadConfig{}, core.NewError("mlx: quantization bits must be >= 0")
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
