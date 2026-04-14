package mlx

import (
	"errors"
	"strings"

	coreio "dappco.re/go/core/io"
)

// Token is a generated token from the RFC-style root API.
type Token struct {
	ID    int32
	Value string
	Text  string
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
		return LoadConfig{}, errors.New("mlx: context length must be >= 0")
	}
	if cfg.Quantization < 0 {
		return LoadConfig{}, errors.New("mlx: quantization bits must be >= 0")
	}

	device := strings.ToLower(strings.TrimSpace(cfg.Device))
	if device == "" {
		device = "gpu"
	}
	switch device {
	case "gpu", "cpu":
		cfg.Device = device
		return cfg, nil
	default:
		return LoadConfig{}, errors.New("mlx: unsupported device: " + device)
	}
}
