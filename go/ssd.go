// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/mlx/dataset"
	"dappco.re/go/mlx/train"
)

// ssd.go: the Model-bound SSD entry point; the pipeline itself lives in
// dappco.re/go/mlx/train.

// SSDConfig configures a native SSD run.
type SSDConfig = train.SSDConfig

// SSDRecipe describes a native SSD parity recipe.
type SSDRecipe = train.SSDRecipe

// SSDRunner supplies the native generation and SFT steps.
type SSDRunner = train.SSDRunner

// SSDSample records one raw sampled response.
type SSDSample = train.SSDSample

// SSDResult records a native SSD run.
type SSDResult = train.SSDResult

// RunSSD samples raw outputs from a frozen model, then trains those
// unverified outputs with the existing native SFT cross-entropy path.
func RunSSD(ctx context.Context, runner SSDRunner, ds dataset.Dataset, cfg SSDConfig) (*SSDResult, error) {
	return train.RunSSD(ctx, runner, ds, cfg)
}

// DefaultSSDConfig returns the ml-ssd data-generation defaults.
func DefaultSSDConfig() SSDConfig { return train.DefaultSSDConfig() }

// SSDRecipes lists the built-in SSD parity recipes.
func SSDRecipes() []SSDRecipe { return train.SSDRecipes() }

// LookupSSDRecipe resolves a built-in recipe by name.
func LookupSSDRecipe(name string) (SSDRecipe, bool) { return train.LookupSSDRecipe(name) }

func (m *Model) RunSSD(ctx context.Context, ds dataset.Dataset, cfg SSDConfig) (*SSDResult, error) {
	if m == nil || m.model == nil {
		return nil, errMLXModelNil
	}
	return RunSSD(ctx, SSDRunner{
		ModelInfo: func(context.Context) ModelInfo { return m.Info() },
		Generate:  m.generateForSSD,
		TrainSFT:  m.TrainSFT,
		// The kernel-prefix lane (#97): prefill the kernel ONCE as the
		// exact token-prefix cache; every sample's generation reuses that
		// KV state. An engine without warm support degrades to plain
		// prefix concatenation — same output, just recomputed — so only
		// a real prefill failure is loud.
		WarmPrefix: func(_ context.Context, prefix string) error {
			err := m.WarmPromptCache(prefix)
			if err == errMLXPromptCacheWarmUnsupp {
				return nil
			}
			return err
		},
	}, ds, cfg)
}

// DefaultSSDConfig returns the ml-ssd data-generation
// defaults, with the SFT internals still caller-owned.
func (m *Model) generateForSSD(ctx context.Context, prompt string, cfg GenerateConfig) (string, error) {
	builder := core.NewBuilder()
	builder.Grow(cfg.MaxTokens * 4)
	for token := range m.GenerateStream(ctx, prompt, ssdOptions(cfg)...) {
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

func ssdOptions(cfg GenerateConfig) []GenerateOption {
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
	if cfg.RepeatPenalty != 0 {
		opts = append(opts, WithRepeatPenalty(cfg.RepeatPenalty))
	}
	return opts
}

// --- SSD code-benchmark surface (implementation in train/ssd_eval.go) ---

// SSDCodeBenchmarkConfig configures native code-generation benchmark runs.
type SSDCodeBenchmarkConfig = train.SSDCodeBenchmarkConfig

// SSDCodeBenchmarkRunner supplies generation and native code-execution
// test evaluation for each candidate.
type SSDCodeBenchmarkRunner = train.SSDCodeBenchmarkRunner

// SSDCodeBenchmarkSample is one code benchmark task.
type SSDCodeBenchmarkSample = train.SSDCodeBenchmarkSample

// SSDCodeCandidate is one generated solution for a benchmark task.
type SSDCodeCandidate = train.SSDCodeCandidate

// SSDCodeExecution records a candidate's test execution outcome.
type SSDCodeExecution = train.SSDCodeExecution

// SSDCodeBenchmarkCandidateResult pairs a candidate with its execution.
type SSDCodeBenchmarkCandidateResult = train.SSDCodeBenchmarkCandidateResult

// SSDCodeBenchmarkSampleResult collects all candidates for one task.
type SSDCodeBenchmarkSampleResult = train.SSDCodeBenchmarkSampleResult

// SSDCodeBenchmarkMetrics summarises a benchmark run (pass@k, difficulty).
type SSDCodeBenchmarkMetrics = train.SSDCodeBenchmarkMetrics

// SSDCodeBenchmarkReport is the full benchmark output.
type SSDCodeBenchmarkReport = train.SSDCodeBenchmarkReport

// DefaultSSDCodeBenchmarkConfig returns the LiveCodeBench-v6 defaults.
func DefaultSSDCodeBenchmarkConfig() SSDCodeBenchmarkConfig {
	return train.DefaultSSDCodeBenchmarkConfig()
}

// LoadSSDCodeBenchmarkJSONLFile reads benchmark samples from a JSONL file.
func LoadSSDCodeBenchmarkJSONLFile(path string) ([]SSDCodeBenchmarkSample, error) {
	return train.LoadSSDCodeBenchmarkJSONLFile(path)
}

// LoadSSDLiveCodeBenchV6JSONLFile reads LiveCodeBench-v6 samples from a JSONL file.
func LoadSSDLiveCodeBenchV6JSONLFile(path string) ([]SSDCodeBenchmarkSample, error) {
	return train.LoadSSDLiveCodeBenchV6JSONLFile(path)
}

// LoadSSDCodeBenchmarkJSONL parses benchmark samples from raw JSONL.
func LoadSSDCodeBenchmarkJSONL(raw string) ([]SSDCodeBenchmarkSample, error) {
	return train.LoadSSDCodeBenchmarkJSONL(raw)
}

// LoadSSDLiveCodeBenchV6JSONL parses LiveCodeBench-v6 samples from raw JSONL.
func LoadSSDLiveCodeBenchV6JSONL(raw string) ([]SSDCodeBenchmarkSample, error) {
	return train.LoadSSDLiveCodeBenchV6JSONL(raw)
}

// FilterSSDLiveCodeBenchV6Samples keeps the canonical evaluation slice.
func FilterSSDLiveCodeBenchV6Samples(samples []SSDCodeBenchmarkSample) []SSDCodeBenchmarkSample {
	return train.FilterSSDLiveCodeBenchV6Samples(samples)
}

// RunSSDCodeBenchmark runs the code benchmark over samples with runner.
func RunSSDCodeBenchmark(ctx context.Context, runner SSDCodeBenchmarkRunner, samples []SSDCodeBenchmarkSample, cfg SSDCodeBenchmarkConfig) (*SSDCodeBenchmarkReport, error) {
	return train.RunSSDCodeBenchmark(ctx, runner, samples, cfg)
}

// SSDPostProcessCode extracts the final code block from a model response.
func SSDPostProcessCode(response string) (string, bool) {
	return train.SSDPostProcessCode(response)
}

// FormatSSDLiveCodeBenchPrompt renders the benchmark prompt for a sample.
func FormatSSDLiveCodeBenchPrompt(sample SSDCodeBenchmarkSample) string {
	return train.FormatSSDLiveCodeBenchPrompt(sample)
}
