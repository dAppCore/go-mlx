// SPDX-Licence-Identifier: EUPL-1.2

package train

import (
	"context"
	"math"
	"sort"
	"strconv"

	core "dappco.re/go"
	"dappco.re/go/mlx/dataset"
	"dappco.re/go/mlx/spine"
)

// ssd.go: the native SSD pipeline (sample raw outputs from a frozen model,
// fine-tune on them with the SFT path) — hooks-based and model-free; the
// root mlx package wires Model.RunSSD into RunSSD's SSDRunner.

const (
	defaultSSDMaxTokens   = 256
	defaultSSDTemperature = 0.7
	defaultSSDTopK        = 64
	defaultSSDTopP        = 0.95

	SSDRecipe4BInstruct     = "SimpleSD-4B-instruct"
	SSDRecipe4BThinking     = "SimpleSD-4B-thinking"
	SSDRecipe30BA3BInstruct = "SimpleSD-30b-a3b-instruct"
)

// SSDConfig configures native self-distillation.
type SSDConfig struct {
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

// SSDRecipe describes a native SSD parity recipe.
type SSDRecipe struct {
	Name          string                 `json:"name"`
	Model         string                 `json:"model"`
	Dataset       string                 `json:"dataset,omitempty"`
	DatasetConfig string                 `json:"dataset_config,omitempty"`
	DatasetSplit  string                 `json:"dataset_split,omitempty"`
	Train         SSDConfig              `json:"train"`
	Eval          SSDCodeBenchmarkConfig `json:"eval"`
	Notes         []string               `json:"notes,omitempty"`
}

// SSDRunner supplies the native generation and SFT steps.
type SSDRunner struct {
	ModelInfo func(context.Context) spine.ModelInfo
	Generate  func(context.Context, string, spine.GenerateConfig) (string, error)
	TrainSFT  func(context.Context, dataset.Dataset, SFTConfig) (*SFTResult, error)
}

// SSDSample records one raw sampled response.
type SSDSample struct {
	Prompt   string            `json:"prompt"`
	Response string            `json:"response"`
	Meta     map[string]string `json:"meta,omitempty"`
}

// SSDResult records a native SSD run.
type SSDResult struct {
	Samples               []SSDSample `json:"samples"`
	SFT                   *SFTResult  `json:"-"`
	SampleTemperature     float32     `json:"sample_temperature"`
	DecodeTemperature     float32     `json:"decode_temperature"`
	SampleMaxTokens       int         `json:"sample_max_tokens"`
	SampleTopK            int         `json:"sample_top_k,omitempty"`
	SampleTopP            float32     `json:"sample_top_p,omitempty"`
	SampleMinP            float32     `json:"sample_min_p,omitempty"`
	RepetitionPenalty     float32     `json:"repetition_penalty,omitempty"`
	FilterShortestPercent float32     `json:"filter_shortest_percent,omitempty"`
}

// RunSSD samples raw outputs from a frozen model, then
// trains those unverified outputs with the existing native SFT cross-entropy
// path. It intentionally has no verifier, teacher, or RL hook.
func RunSSD(ctx context.Context, runner SSDRunner, ds dataset.Dataset, cfg SSDConfig) (*SSDResult, error) {
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
	if runner.ModelInfo != nil {
		cfg = normalizeSSDConfigForModel(cfg, runner.ModelInfo(ctx))
	} else {
		cfg = normalizeSSDConfig(cfg)
	}
	if err := validateSSDConfig(cfg); err != nil {
		return nil, err
	}

	generated, samples, err := buildSSDDataset(ctx, runner, ds, cfg)
	if err != nil {
		return nil, err
	}
	if len(samples) == 0 {
		return nil, core.NewError("mlx: SSD dataset produced no prompts")
	}
	sftResult, err := runner.TrainSFT(ctx, dataset.NewSliceDataset(generated), cfg.SFT)
	if err != nil {
		return newSSDResult(samples, sftResult, cfg), err
	}
	return newSSDResult(samples, sftResult, cfg), nil
}

// RunSSD samples from m and fine-tunes m with native SFT.
func DefaultSSDConfig() SSDConfig {
	return SSDConfig{
		SampleMaxTokens:       65536,
		SampleTemperature:     1.5,
		SampleTopK:            20,
		SampleTopP:            0.8,
		RepetitionPenalty:     1.0,
		FilterShortestPercent: 10,
	}
}

// DefaultSSDCodeBenchmarkConfig returns the ml-ssd
// LiveCodeBench-v6 evaluation defaults.
func DefaultSSDCodeBenchmarkConfig() SSDCodeBenchmarkConfig {
	return SSDCodeBenchmarkConfig{
		Benchmark: "LiveCodeBench-v6",
		NRepeat:   20,
		Seeds:     []uint64{0, 1234, 1234, 1234},
		Generate: spine.GenerateConfig{
			MaxTokens:   32768,
			Temperature: 0.6,
			TopP:        0.95,
			TopK:        20,
			MinP:        0,
		},
	}
}

// SSDRecipes returns the released ml-ssd model recipe
// descriptors with native data-generation and evaluation defaults.
func SSDRecipes() []SSDRecipe {
	train := DefaultSSDConfig()
	eval := DefaultSSDCodeBenchmarkConfig()
	return []SSDRecipe{
		newSSDRecipe(SSDRecipe4BInstruct, "apple/SimpleSD-4B-instruct", train, eval),
		newSSDRecipe(SSDRecipe4BThinking, "apple/SimpleSD-4B-thinking", train, eval),
		newSSDRecipe(SSDRecipe30BA3BInstruct, "apple/SimpleSD-30b-a3b-instruct", train, eval),
	}
}

// LookupSSDRecipe returns a named SSD parity recipe.
func LookupSSDRecipe(name string) (SSDRecipe, bool) {
	for _, recipe := range SSDRecipes() {
		if recipe.Name == name || recipe.Model == name {
			return recipe, true
		}
	}
	return SSDRecipe{}, false
}

// SampleGenerateConfig returns the frozen-model sampling configuration used to
// create the raw SSD training rows.
func (r *SSDResult) SampleGenerateConfig() spine.GenerateConfig {
	if r == nil {
		return spine.GenerateConfig{}
	}
	return spine.GenerateConfig{
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
func (r *SSDResult) DecodeGenerateConfig(maxTokens int) spine.GenerateConfig {
	if r == nil {
		return spine.GenerateConfig{MaxTokens: maxTokens}
	}
	return spine.GenerateConfig{
		MaxTokens:   maxTokens,
		Temperature: r.DecodeTemperature,
	}
}

func newSSDResult(samples []SSDSample, sft *SFTResult, cfg SSDConfig) *SSDResult {
	return &SSDResult{
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

func newSSDRecipe(name, model string, train SSDConfig, eval SSDCodeBenchmarkConfig) SSDRecipe {
	return SSDRecipe{
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

func buildSSDDataset(ctx context.Context, runner SSDRunner, ds dataset.Dataset, cfg SSDConfig) ([]dataset.Sample, []SSDSample, error) {
	generated := make([]dataset.Sample, 0, 16)
	samples := make([]SSDSample, 0, 16)
	genCfg := ssdGenerateConfig(cfg)
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
		prompt := ssdPrompt(sample)
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
		meta["ssd_sample_temperature"] = formatSSDFloat32(cfg.SampleTemperature)
		row := dataset.Sample{Prompt: prompt, Response: response, Meta: meta}
		generated = append(generated, row)
		samples = append(samples, SSDSample{
			Prompt:   prompt,
			Response: response,
			Meta:     dataset.CloneSample(row).Meta,
		})
	}
	return filterSSDShortest(generated, cfg.FilterShortestPercent), samples, nil
}

func filterSSDShortest(rows []dataset.Sample, percent float32) []dataset.Sample {
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

func ssdPrompt(sample dataset.Sample) string {
	if sample.Prompt != "" {
		return sample.Prompt
	}
	return sample.Text
}

func ssdGenerateConfig(cfg SSDConfig) spine.GenerateConfig {
	return spine.GenerateConfig{
		MaxTokens:     cfg.SampleMaxTokens,
		Temperature:   cfg.SampleTemperature,
		TopK:          cfg.SampleTopK,
		TopP:          cfg.SampleTopP,
		MinP:          cfg.SampleMinP,
		RepeatPenalty: cfg.RepetitionPenalty,
	}
}

func normalizeSSDConfig(cfg SSDConfig) SSDConfig {
	return normalizeSSDConfigWithSFT(cfg, normalizeSFTConfig)
}

func normalizeSSDConfigForModel(cfg SSDConfig, info spine.ModelInfo) SSDConfig {
	return normalizeSSDConfigWithSFT(cfg, func(sft SFTConfig) SFTConfig {
		return NormalizeSFTConfigForModel(sft, info)
	})
}

func normalizeSSDConfigWithSFT(cfg SSDConfig, normalizeSFT func(SFTConfig) SFTConfig) SSDConfig {
	if cfg.SampleMaxTokens <= 0 {
		cfg.SampleMaxTokens = defaultSSDMaxTokens
	}
	if cfg.SampleTemperature == 0 {
		cfg.SampleTemperature = defaultSSDTemperature
	}
	if cfg.SampleTopK == 0 {
		cfg.SampleTopK = defaultSSDTopK
	}
	if cfg.SampleTopP == 0 {
		cfg.SampleTopP = defaultSSDTopP
	}
	if cfg.DecodeTemperature != 0 && cfg.SFT.EvalTemperature == 0 {
		cfg.SFT.EvalTemperature = cfg.DecodeTemperature
	}
	cfg.SFT = normalizeSFT(cfg.SFT)
	return cfg
}

func validateSSDConfig(cfg SSDConfig) error {
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

func formatSSDFloat32(value float32) string {
	return strconv.FormatFloat(float64(value), 'g', -1, 32)
}
