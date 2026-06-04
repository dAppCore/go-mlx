// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"math"
	"strconv"

	core "dappco.re/go"
	"dappco.re/go/mlx/dataset"
)

const (
	defaultSimpleSelfDistillationMaxTokens   = 256
	defaultSimpleSelfDistillationTemperature = 0.7
	defaultSimpleSelfDistillationTopK        = 64
	defaultSimpleSelfDistillationTopP        = 0.95
)

// SimpleSelfDistillationConfig configures native self-distillation.
type SimpleSelfDistillationConfig struct {
	SampleMaxTokens   int
	SampleTemperature float32
	SampleTopK        int
	SampleTopP        float32
	SampleMinP        float32
	DecodeTemperature float32
	SFT               SFTConfig
}

// SimpleSelfDistillationRunner supplies the native generation and SFT steps.
type SimpleSelfDistillationRunner struct {
	Generate func(context.Context, string, GenerateConfig) (string, error)
	TrainSFT func(context.Context, dataset.Dataset, SFTConfig) (*SFTResult, error)
}

// SimpleSelfDistillationSample records one raw sampled response.
type SimpleSelfDistillationSample struct {
	Prompt   string            `json:"prompt"`
	Response string            `json:"response"`
	Meta     map[string]string `json:"meta,omitempty"`
}

// SimpleSelfDistillationResult records a native SSD run.
type SimpleSelfDistillationResult struct {
	Samples           []SimpleSelfDistillationSample `json:"samples"`
	SFT               *SFTResult                     `json:"-"`
	SampleTemperature float32                        `json:"sample_temperature"`
	DecodeTemperature float32                        `json:"decode_temperature"`
	SampleMaxTokens   int                            `json:"sample_max_tokens"`
}

// RunSimpleSelfDistillation samples raw outputs from a frozen model, then
// trains those unverified outputs with the existing native SFT cross-entropy
// path. It intentionally has no verifier, teacher, or RL hook.
func RunSimpleSelfDistillation(ctx context.Context, runner SimpleSelfDistillationRunner, ds dataset.Dataset, cfg SimpleSelfDistillationConfig) (*SimpleSelfDistillationResult, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if ds == nil {
		return nil, core.NewError("mlx: SSD dataset is nil")
	}
	if runner.Generate == nil {
		return nil, core.NewError("mlx: SSD generate function is nil")
	}
	if runner.TrainSFT == nil {
		return nil, core.NewError("mlx: SSD TrainSFT function is nil")
	}
	cfg = normalizeSimpleSelfDistillationConfig(cfg)
	if err := validateSimpleSelfDistillationConfig(cfg); err != nil {
		return nil, err
	}

	generated, samples, err := buildSimpleSelfDistillationDataset(ctx, runner, ds, cfg)
	if err != nil {
		return nil, err
	}
	if len(samples) == 0 {
		return nil, core.NewError("mlx: SSD dataset produced no prompts")
	}
	sftResult, err := runner.TrainSFT(ctx, dataset.NewSliceDataset(generated), cfg.SFT)
	if err != nil {
		return &SimpleSelfDistillationResult{
			Samples:           samples,
			SFT:               sftResult,
			SampleTemperature: cfg.SampleTemperature,
			DecodeTemperature: cfg.DecodeTemperature,
			SampleMaxTokens:   cfg.SampleMaxTokens,
		}, err
	}
	return &SimpleSelfDistillationResult{
		Samples:           samples,
		SFT:               sftResult,
		SampleTemperature: cfg.SampleTemperature,
		DecodeTemperature: cfg.DecodeTemperature,
		SampleMaxTokens:   cfg.SampleMaxTokens,
	}, nil
}

// RunSimpleSelfDistillation samples from m and fine-tunes m with native SFT.
func (m *Model) RunSimpleSelfDistillation(ctx context.Context, ds dataset.Dataset, cfg SimpleSelfDistillationConfig) (*SimpleSelfDistillationResult, error) {
	if m == nil || m.model == nil {
		return nil, errMLXModelNil
	}
	return RunSimpleSelfDistillation(ctx, SimpleSelfDistillationRunner{
		Generate: m.generateForSimpleSelfDistillation,
		TrainSFT: m.TrainSFT,
	}, ds, cfg)
}

func buildSimpleSelfDistillationDataset(ctx context.Context, runner SimpleSelfDistillationRunner, ds dataset.Dataset, cfg SimpleSelfDistillationConfig) ([]dataset.Sample, []SimpleSelfDistillationSample, error) {
	generated := make([]dataset.Sample, 0, 16)
	samples := make([]SimpleSelfDistillationSample, 0, 16)
	genCfg := simpleSelfDistillationGenerateConfig(cfg)
	for index := 0; ; index++ {
		if err := ctx.Err(); err != nil {
			return generated, samples, err
		}
		sample, ok, err := ds.Next()
		if err != nil {
			return generated, samples, err
		}
		if !ok {
			break
		}
		prompt := simpleSelfDistillationPrompt(sample)
		if prompt == "" {
			continue
		}
		response, err := runner.Generate(ctx, prompt, genCfg)
		if err != nil {
			return generated, samples, err
		}
		meta := dataset.CloneSample(sample).Meta
		if meta == nil {
			meta = make(map[string]string, 4)
		}
		meta["ssd"] = "simple_self_distillation"
		meta["ssd_source_index"] = strconv.Itoa(index)
		meta["ssd_sample_temperature"] = formatSimpleSelfDistillationFloat32(cfg.SampleTemperature)
		row := dataset.Sample{Prompt: prompt, Response: response, Meta: meta}
		generated = append(generated, row)
		samples = append(samples, SimpleSelfDistillationSample{
			Prompt:   prompt,
			Response: response,
			Meta:     dataset.CloneSample(row).Meta,
		})
	}
	return generated, samples, nil
}

func simpleSelfDistillationPrompt(sample dataset.Sample) string {
	if sample.Prompt != "" {
		return sample.Prompt
	}
	return sample.Text
}

func simpleSelfDistillationGenerateConfig(cfg SimpleSelfDistillationConfig) GenerateConfig {
	return GenerateConfig{
		MaxTokens:   cfg.SampleMaxTokens,
		Temperature: cfg.SampleTemperature,
		TopK:        cfg.SampleTopK,
		TopP:        cfg.SampleTopP,
		MinP:        cfg.SampleMinP,
	}
}

func normalizeSimpleSelfDistillationConfig(cfg SimpleSelfDistillationConfig) SimpleSelfDistillationConfig {
	if cfg.SampleMaxTokens <= 0 {
		cfg.SampleMaxTokens = defaultSimpleSelfDistillationMaxTokens
	}
	if cfg.SampleTemperature == 0 {
		cfg.SampleTemperature = defaultSimpleSelfDistillationTemperature
	}
	if cfg.SampleTopK == 0 {
		cfg.SampleTopK = defaultSimpleSelfDistillationTopK
	}
	if cfg.SampleTopP == 0 {
		cfg.SampleTopP = defaultSimpleSelfDistillationTopP
	}
	if cfg.DecodeTemperature != 0 && cfg.SFT.EvalTemperature == 0 {
		cfg.SFT.EvalTemperature = cfg.DecodeTemperature
	}
	cfg.SFT = normalizeSFTConfig(cfg.SFT)
	return cfg
}

func validateSimpleSelfDistillationConfig(cfg SimpleSelfDistillationConfig) error {
	if cfg.SampleTemperature <= 0 || math.IsNaN(float64(cfg.SampleTemperature)) || math.IsInf(float64(cfg.SampleTemperature), 0) {
		return core.NewError("mlx: SSD sample temperature must be positive and finite")
	}
	if cfg.SampleTemperature == 1 {
		return core.NewError("mlx: SSD sample temperature must be non-unit")
	}
	if cfg.DecodeTemperature < 0 || math.IsNaN(float64(cfg.DecodeTemperature)) || math.IsInf(float64(cfg.DecodeTemperature), 0) {
		return core.NewError("mlx: SSD decode temperature must be finite")
	}
	if cfg.SampleMaxTokens <= 0 {
		return core.NewError("mlx: SSD sample max tokens must be positive")
	}
	return nil
}

func (m *Model) generateForSimpleSelfDistillation(ctx context.Context, prompt string, cfg GenerateConfig) (string, error) {
	builder := core.NewBuilder()
	builder.Grow(cfg.MaxTokens * 4)
	for token := range m.GenerateStream(ctx, prompt, simpleSelfDistillationOptions(cfg)...) {
		builder.WriteString(token.Text)
	}
	if err := m.model.Err(); err != nil {
		return "", err
	}
	if ctx != nil {
		if err := ctx.Err(); err != nil {
			return "", err
		}
	}
	return builder.String(), nil
}

func simpleSelfDistillationOptions(cfg GenerateConfig) []GenerateOption {
	opts := []GenerateOption{
		WithMaxTokens(cfg.MaxTokens),
		WithTemperature(cfg.Temperature),
	}
	if cfg.TopK != 0 {
		opts = append(opts, WithTopK(cfg.TopK))
	}
	if cfg.TopP != 0 {
		opts = append(opts, WithTopP(cfg.TopP))
	}
	if cfg.MinP != 0 {
		opts = append(opts, WithMinP(cfg.MinP))
	}
	return opts
}

func formatSimpleSelfDistillationFloat32(value float32) string {
	return strconv.FormatFloat(float64(value), 'g', -1, 32)
}
