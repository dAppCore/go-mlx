// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"flag"
	"io"

	core "dappco.re/go"
	mlx "dappco.re/go/mlx"
)

type ssdRecipesReport struct {
	Version      int                   `json:"version"`
	Kind         string                `json:"kind"`
	NoPython     bool                  `json:"no_python"`
	TrainDefault ssdRecipeTrainConfig  `json:"train_default"`
	EvalDefault  ssdRecipeEvalConfig   `json:"eval_default"`
	Recipes      []ssdRecipeDescriptor `json:"recipes"`
	Notes        []string              `json:"notes,omitempty"`
}

type ssdRecipeDescriptor struct {
	Name          string               `json:"name"`
	Model         string               `json:"model"`
	Dataset       string               `json:"dataset,omitempty"`
	DatasetConfig string               `json:"dataset_config,omitempty"`
	DatasetSplit  string               `json:"dataset_split,omitempty"`
	Train         ssdRecipeTrainConfig `json:"train"`
	Eval          ssdRecipeEvalConfig  `json:"eval"`
	Notes         []string             `json:"notes,omitempty"`
}

type ssdRecipeTrainConfig struct {
	SampleMaxTokens       int     `json:"sample_max_tokens,omitempty"`
	SampleTemperature     float32 `json:"sample_temperature,omitempty"`
	SampleTopK            int     `json:"sample_top_k,omitempty"`
	SampleTopP            float32 `json:"sample_top_p,omitempty"`
	SampleMinP            float32 `json:"sample_min_p,omitempty"`
	RepetitionPenalty     float32 `json:"repetition_penalty,omitempty"`
	FilterShortestPercent float32 `json:"filter_shortest_percent,omitempty"`
}

type ssdRecipeEvalConfig struct {
	Benchmark string                  `json:"benchmark,omitempty"`
	NRepeat   int                     `json:"n_repeat,omitempty"`
	Generate  ssdRecipeGenerateConfig `json:"generate"`
	Seeds     []uint64                `json:"seeds,omitempty"`
}

type ssdRecipeGenerateConfig struct {
	MaxTokens   int     `json:"max_tokens,omitempty"`
	Temperature float32 `json:"temperature,omitempty"`
	TopP        float32 `json:"top_p,omitempty"`
	TopK        int     `json:"top_k,omitempty"`
	MinP        float32 `json:"min_p,omitempty"`
}

func runSSDRecipesCommand(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("ssd-recipes", flag.ContinueOnError)
	fs.SetOutput(stderr)
	jsonOut := fs.Bool("json", false, "write JSON recipe report")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		core.Print(stderr, "%s ssd-recipes: expected no positional arguments", cliName())
		return 2
	}
	report := ssdRecipesReportFromDefaults()
	if *jsonOut {
		return writeSSDRecipesJSON(stdout, stderr, report)
	}
	core.WriteString(stdout, "simple self-distillation recipes\n")
	core.WriteString(stdout, core.Sprintf("  data-gen: max_tokens=%d temperature=%.1f top_p=%.1f top_k=%d repetition_penalty=%.1f filter_shortest_percent=%.0f\n",
		report.TrainDefault.SampleMaxTokens,
		report.TrainDefault.SampleTemperature,
		report.TrainDefault.SampleTopP,
		report.TrainDefault.SampleTopK,
		report.TrainDefault.RepetitionPenalty,
		report.TrainDefault.FilterShortestPercent,
	))
	core.WriteString(stdout, core.Sprintf("  eval: %s n_repeat=%d max_tokens=%d temperature=%.1f top_p=%.2f top_k=%d\n",
		report.EvalDefault.Benchmark,
		report.EvalDefault.NRepeat,
		report.EvalDefault.Generate.MaxTokens,
		report.EvalDefault.Generate.Temperature,
		report.EvalDefault.Generate.TopP,
		report.EvalDefault.Generate.TopK,
	))
	for _, recipe := range report.Recipes {
		core.WriteString(stdout, core.Sprintf("  %s: %s (%s/%s)\n", recipe.Name, recipe.Model, recipe.Dataset, recipe.DatasetConfig))
	}
	return 0
}

func ssdRecipesReportFromDefaults() ssdRecipesReport {
	train := mlx.DefaultSimpleSelfDistillationConfig()
	eval := mlx.DefaultSimpleSelfDistillationCodeBenchmarkConfig()
	return ssdRecipesReport{
		Version:      1,
		Kind:         "simple-self-distillation-recipes",
		NoPython:     true,
		TrainDefault: ssdRecipeTrainConfigFromConfig(train),
		EvalDefault:  ssdRecipeEvalConfigFromConfig(eval),
		Recipes:      ssdRecipeDescriptorsFromRecipes(mlx.SimpleSelfDistillationRecipes()),
		Notes: []string{
			"The go-mlx SSD pipeline and benchmark harness are native Go/Metal; LiveCodeBench language execution stays behind the caller-supplied RunTests callback.",
			"Use this report as the source manifest for docs/runtime SSD parity artefacts before heavyweight recipe runs are reproduced locally.",
		},
	}
}

func ssdRecipeDescriptorsFromRecipes(recipes []mlx.SimpleSelfDistillationRecipe) []ssdRecipeDescriptor {
	descriptors := make([]ssdRecipeDescriptor, 0, len(recipes))
	for _, recipe := range recipes {
		descriptors = append(descriptors, ssdRecipeDescriptor{
			Name:          recipe.Name,
			Model:         recipe.Model,
			Dataset:       recipe.Dataset,
			DatasetConfig: recipe.DatasetConfig,
			DatasetSplit:  recipe.DatasetSplit,
			Train:         ssdRecipeTrainConfigFromConfig(recipe.Train),
			Eval:          ssdRecipeEvalConfigFromConfig(recipe.Eval),
			Notes:         recipe.Notes,
		})
	}
	return descriptors
}

func ssdRecipeTrainConfigFromConfig(cfg mlx.SimpleSelfDistillationConfig) ssdRecipeTrainConfig {
	return ssdRecipeTrainConfig{
		SampleMaxTokens:       cfg.SampleMaxTokens,
		SampleTemperature:     cfg.SampleTemperature,
		SampleTopK:            cfg.SampleTopK,
		SampleTopP:            cfg.SampleTopP,
		SampleMinP:            cfg.SampleMinP,
		RepetitionPenalty:     cfg.RepetitionPenalty,
		FilterShortestPercent: cfg.FilterShortestPercent,
	}
}

func ssdRecipeEvalConfigFromConfig(cfg mlx.SimpleSelfDistillationCodeBenchmarkConfig) ssdRecipeEvalConfig {
	return ssdRecipeEvalConfig{
		Benchmark: cfg.Benchmark,
		NRepeat:   cfg.NRepeat,
		Generate: ssdRecipeGenerateConfig{
			MaxTokens:   cfg.Generate.MaxTokens,
			Temperature: cfg.Generate.Temperature,
			TopP:        cfg.Generate.TopP,
			TopK:        cfg.Generate.TopK,
			MinP:        cfg.Generate.MinP,
		},
		Seeds: core.SliceClone(cfg.Seeds),
	}
}

func writeSSDRecipesJSON(stdout, stderr io.Writer, report ssdRecipesReport) int {
	data := core.JSONMarshalIndent(report, "", "  ")
	if !data.OK {
		core.Print(stderr, "%s ssd-recipes: marshal report failed", cliName())
		return 1
	}
	core.WriteString(stdout, string(data.Value.([]byte)))
	core.WriteString(stdout, "\n")
	return 0
}
