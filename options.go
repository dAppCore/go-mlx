package mlx

// GenerateConfig holds generation parameters.
type GenerateConfig struct {
	MaxTokens   int
	Temperature float32
	TopK        int
	TopP        float32
	StopTokens  []int32
}

// DefaultGenerateConfig returns sensible defaults.
func DefaultGenerateConfig() GenerateConfig {
	return GenerateConfig{
		MaxTokens:   256,
		Temperature: 0.0,
	}
}

// GenerateOption configures text generation.
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

// WithTopP sets nucleus sampling threshold. 0 = disabled.
func WithTopP(p float32) GenerateOption {
	return func(c *GenerateConfig) { c.TopP = p }
}

// WithStopTokens sets token IDs that stop generation.
func WithStopTokens(ids ...int32) GenerateOption {
	return func(c *GenerateConfig) { c.StopTokens = ids }
}

// LoadConfig holds model loading parameters.
type LoadConfig struct {
	Backend string // "metal" (default), "mlx_lm"
}

// LoadOption configures model loading.
type LoadOption func(*LoadConfig)

// WithBackend selects a specific inference backend by name.
func WithBackend(name string) LoadOption {
	return func(c *LoadConfig) { c.Backend = name }
}

// ApplyGenerateOpts builds a GenerateConfig from options.
func ApplyGenerateOpts(opts []GenerateOption) GenerateConfig {
	cfg := DefaultGenerateConfig()
	for _, o := range opts {
		o(&cfg)
	}
	return cfg
}

// ApplyLoadOpts builds a LoadConfig from options.
func ApplyLoadOpts(opts []LoadOption) LoadConfig {
	var cfg LoadConfig
	for _, o := range opts {
		o(&cfg)
	}
	return cfg
}
