// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"math"
	"sort"
	"strconv"

	core "dappco.re/go"
	"dappco.re/go/mlx/dataset"
)

const (
	defaultSimpleSelfDistillationMaxTokens   = 256
	defaultSimpleSelfDistillationTemperature = 0.7
	defaultSimpleSelfDistillationTopK        = 64
	defaultSimpleSelfDistillationTopP        = 0.95

	SimpleSelfDistillationRecipe4BInstruct     = "SimpleSD-4B-instruct"
	SimpleSelfDistillationRecipe4BThinking     = "SimpleSD-4B-thinking"
	SimpleSelfDistillationRecipe30BA3BInstruct = "SimpleSD-30b-a3b-instruct"
)

// SimpleSelfDistillationConfig configures native self-distillation.
type SimpleSelfDistillationConfig struct {
	SampleMaxTokens       int       `json:"sample_max_tokens,omitempty"`
	SampleTemperature     float32   `json:"sample_temperature,omitempty"`
	SampleTopK            int       `json:"sample_top_k,omitempty"`
	SampleTopP            float32   `json:"sample_top_p,omitempty"`
	SampleMinP            float32   `json:"sample_min_p,omitempty"`
	RepetitionPenalty     float32   `json:"repetition_penalty,omitempty"`
	FilterShortestPercent float32   `json:"filter_shortest_percent,omitempty"`
	DecodeTemperature     float32   `json:"decode_temperature,omitempty"`
	SFT                   SFTConfig `json:"sft,omitempty"`
}

// SimpleSelfDistillationRecipe describes a native SSD parity recipe.
type SimpleSelfDistillationRecipe struct {
	Name          string                                    `json:"name"`
	Model         string                                    `json:"model"`
	Dataset       string                                    `json:"dataset,omitempty"`
	DatasetConfig string                                    `json:"dataset_config,omitempty"`
	DatasetSplit  string                                    `json:"dataset_split,omitempty"`
	Train         SimpleSelfDistillationConfig              `json:"train"`
	Eval          SimpleSelfDistillationCodeBenchmarkConfig `json:"eval"`
	Notes         []string                                  `json:"notes,omitempty"`
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
	Samples               []SimpleSelfDistillationSample `json:"samples"`
	SFT                   *SFTResult                     `json:"-"`
	SampleTemperature     float32                        `json:"sample_temperature"`
	DecodeTemperature     float32                        `json:"decode_temperature"`
	SampleMaxTokens       int                            `json:"sample_max_tokens"`
	SampleTopK            int                            `json:"sample_top_k,omitempty"`
	SampleTopP            float32                        `json:"sample_top_p,omitempty"`
	SampleMinP            float32                        `json:"sample_min_p,omitempty"`
	RepetitionPenalty     float32                        `json:"repetition_penalty,omitempty"`
	FilterShortestPercent float32                        `json:"filter_shortest_percent,omitempty"`
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
		return newSimpleSelfDistillationResult(samples, sftResult, cfg), err
	}
	return newSimpleSelfDistillationResult(samples, sftResult, cfg), nil
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

// DefaultSimpleSelfDistillationConfig returns the ml-ssd data-generation
// defaults, with the SFT internals still caller-owned.
func DefaultSimpleSelfDistillationConfig() SimpleSelfDistillationConfig {
	return SimpleSelfDistillationConfig{
		SampleMaxTokens:       65536,
		SampleTemperature:     1.5,
		SampleTopK:            20,
		SampleTopP:            0.8,
		RepetitionPenalty:     1.0,
		FilterShortestPercent: 10,
	}
}

// DefaultSimpleSelfDistillationCodeBenchmarkConfig returns the ml-ssd
// LiveCodeBench-v6 evaluation defaults.
func DefaultSimpleSelfDistillationCodeBenchmarkConfig() SimpleSelfDistillationCodeBenchmarkConfig {
	return SimpleSelfDistillationCodeBenchmarkConfig{
		Benchmark: "LiveCodeBench-v6",
		NRepeat:   20,
		Seeds:     []uint64{0, 1234, 1234, 1234},
		Generate: GenerateConfig{
			MaxTokens:   32768,
			Temperature: 0.6,
			TopP:        0.95,
			TopK:        20,
			MinP:        0,
		},
	}
}

// SimpleSelfDistillationRecipes returns the released ml-ssd model recipe
// descriptors with native data-generation and evaluation defaults.
func SimpleSelfDistillationRecipes() []SimpleSelfDistillationRecipe {
	train := DefaultSimpleSelfDistillationConfig()
	eval := DefaultSimpleSelfDistillationCodeBenchmarkConfig()
	return []SimpleSelfDistillationRecipe{
		newSimpleSelfDistillationRecipe(SimpleSelfDistillationRecipe4BInstruct, "apple/SimpleSD-4B-instruct", train, eval),
		newSimpleSelfDistillationRecipe(SimpleSelfDistillationRecipe4BThinking, "apple/SimpleSD-4B-thinking", train, eval),
		newSimpleSelfDistillationRecipe(SimpleSelfDistillationRecipe30BA3BInstruct, "apple/SimpleSD-30b-a3b-instruct", train, eval),
	}
}

// LookupSimpleSelfDistillationRecipe returns a named SSD parity recipe.
func LookupSimpleSelfDistillationRecipe(name string) (SimpleSelfDistillationRecipe, bool) {
	for _, recipe := range SimpleSelfDistillationRecipes() {
		if recipe.Name == name || recipe.Model == name {
			return recipe, true
		}
	}
	return SimpleSelfDistillationRecipe{}, false
}

// SampleGenerateConfig returns the frozen-model sampling configuration used to
// create the raw SSD training rows.
func (r *SimpleSelfDistillationResult) SampleGenerateConfig() GenerateConfig {
	if r == nil {
		return GenerateConfig{}
	}
	return GenerateConfig{
		MaxTokens:     r.SampleMaxTokens,
		Temperature:   r.SampleTemperature,
		TopK:          r.SampleTopK,
		TopP:          r.SampleTopP,
		MinP:          r.SampleMinP,
		RepeatPenalty: r.RepetitionPenalty,
	}
}

// DecodeGenerateConfig returns the post-SSD decode configuration with the
// separately tuned decode temperature. The token budget remains caller-owned.
func (r *SimpleSelfDistillationResult) DecodeGenerateConfig(maxTokens int) GenerateConfig {
	if r == nil {
		return GenerateConfig{MaxTokens: maxTokens}
	}
	return GenerateConfig{
		MaxTokens:   maxTokens,
		Temperature: r.DecodeTemperature,
	}
}

func newSimpleSelfDistillationResult(samples []SimpleSelfDistillationSample, sft *SFTResult, cfg SimpleSelfDistillationConfig) *SimpleSelfDistillationResult {
	return &SimpleSelfDistillationResult{
		Samples:               samples,
		SFT:                   sft,
		SampleTemperature:     cfg.SampleTemperature,
		DecodeTemperature:     cfg.DecodeTemperature,
		SampleMaxTokens:       cfg.SampleMaxTokens,
		SampleTopK:            cfg.SampleTopK,
		SampleTopP:            cfg.SampleTopP,
		SampleMinP:            cfg.SampleMinP,
		RepetitionPenalty:     cfg.RepetitionPenalty,
		FilterShortestPercent: cfg.FilterShortestPercent,
	}
}

func newSimpleSelfDistillationRecipe(name, model string, train SimpleSelfDistillationConfig, eval SimpleSelfDistillationCodeBenchmarkConfig) SimpleSelfDistillationRecipe {
	return SimpleSelfDistillationRecipe{
		Name:          name,
		Model:         model,
		Dataset:       "microsoft/rStar-Coder",
		DatasetConfig: "seed_sft",
		DatasetSplit:  "train",
		Train:         train,
		Eval:          eval,
		Notes: []string{
			"Use the released model card for model-specific decode sampling when it differs from the upstream eval example.",
			"Store runtime artefacts under docs/runtime/ when reproducing this recipe locally.",
		},
	}
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
	return filterSimpleSelfDistillationShortest(generated, cfg.FilterShortestPercent), samples, nil
}

func filterSimpleSelfDistillationShortest(rows []dataset.Sample, percent float32) []dataset.Sample {
	if percent <= 0 || len(rows) <= 1 {
		return rows
	}
	drop := int(math.Ceil(float64(len(rows)) * float64(percent) / 100))
	if drop <= 0 {
		return rows
	}
	if drop >= len(rows) {
		drop = len(rows) - 1
	}
	order := make([]int, len(rows))
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(i, j int) bool {
		return len(rows[order[i]].Response) < len(rows[order[j]].Response)
	})
	dropped := make(map[int]struct{}, drop)
	for _, index := range order[:drop] {
		dropped[index] = struct{}{}
	}
	filtered := make([]dataset.Sample, 0, len(rows)-drop)
	for index, row := range rows {
		if _, ok := dropped[index]; ok {
			continue
		}
		filtered = append(filtered, row)
	}
	return filtered
}

func simpleSelfDistillationPrompt(sample dataset.Sample) string {
	if sample.Prompt != "" {
		return sample.Prompt
	}
	return sample.Text
}

func simpleSelfDistillationGenerateConfig(cfg SimpleSelfDistillationConfig) GenerateConfig {
	return GenerateConfig{
		MaxTokens:     cfg.SampleMaxTokens,
		Temperature:   cfg.SampleTemperature,
		TopK:          cfg.SampleTopK,
		TopP:          cfg.SampleTopP,
		MinP:          cfg.SampleMinP,
		RepeatPenalty: cfg.RepetitionPenalty,
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
	if cfg.RepetitionPenalty < 0 || math.IsNaN(float64(cfg.RepetitionPenalty)) || math.IsInf(float64(cfg.RepetitionPenalty), 0) {
		return core.NewError("mlx: SSD repetition penalty must be finite and non-negative")
	}
	if cfg.FilterShortestPercent < 0 || cfg.FilterShortestPercent > 100 || math.IsNaN(float64(cfg.FilterShortestPercent)) || math.IsInf(float64(cfg.FilterShortestPercent), 0) {
		return core.NewError("mlx: SSD filter shortest percent must be finite between 0 and 100")
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
	if cfg.RepeatPenalty != 0 {
		opts = append(opts, WithRepeatPenalty(cfg.RepeatPenalty))
	}
	return opts
}

func formatSimpleSelfDistillationFloat32(value float32) string {
	return strconv.FormatFloat(float64(value), 'g', -1, 32)
}
