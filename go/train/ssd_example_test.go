// SPDX-Licence-Identifier: EUPL-1.2

// Runnable usage-in-situ for the SSD sampling phase. Each carries an Output:
// comment so it executes under `go test` and doubles as the usage doc
// (AX principle 2). Generation is injected, so the whole self-distillation
// sampling loop runs with no model and no Metal.
//
// SSD is decoupled from SFT: RunSSD STOPS at the scored trace — it samples the
// frozen model, scores each sample at birth, and returns the rows. A separate
// SFT run trains on a curated subset later; RunSSD never trains.
//
// Run:    go test -tags metal_runtime -run='Example' ./go

package train

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/mlx/dataset"
	"dappco.re/go/mlx/spine"
)

// ExampleRunSSD samples one response per prompt from the injected generator and
// stops at the scored trace. With ScoreSamples on, each sample is scored at
// birth: the score rides the sample meta (ssd_lek) so the trace is explainable
// downstream, and a sidecar path is set on the result. The returned Samples
// ARE the deliverable — no training step runs here.
func ExampleRunSSD() {
	dir := core.MkdirTemp("", "ssd-example-*")
	defer core.RemoveAll(dir.Value.(string))

	replies := map[string]string{
		"prove a lemma":     "I work it through step by step and check each claim.",
		"hold a hard truth": "I feel the weight of it and I choose to look straight at it.",
	}
	result, err := RunSSD(context.Background(), SSDRunner{
		Generate: func(_ context.Context, prompt string, _ spine.GenerateConfig) (string, error) {
			return replies[prompt], nil
		},
	}, dataset.NewSliceDataset([]dataset.Sample{
		{Prompt: "prove a lemma"},
		{Prompt: "hold a hard truth"},
	}), SSDConfig{
		SampleTemperature:     1.5,
		SampleMaxTokens:       64,
		FilterShortestPercent: 0,
		ScoreSamples:          true,
		SFT:                   SFTConfig{CheckpointDir: dir.Value.(string)},
	})
	if err != nil {
		core.Println("error:", err)
		return
	}
	core.Println("samples:", len(result.Samples))
	core.Println("first prompt:", result.Samples[0].Prompt)
	core.Println("ssd marker:", result.Samples[0].Meta["ssd"])
	core.Println("score on meta:", result.Samples[0].Meta["ssd_lek"] != "")
	core.Println("sidecar set:", result.SampleScoreSidecar != "")
	// Output:
	// samples: 2
	// first prompt: prove a lemma
	// ssd marker: simple_self_distillation
	// score on meta: true
	// sidecar set: true
}

// ExampleDefaultSSDConfig shows the ml-ssd data-generation sampling defaults:
// a high non-unit temperature for diversity, a token budget, and the
// shortest-fraction filter.
func ExampleDefaultSSDConfig() {
	cfg := DefaultSSDConfig()
	core.Println("temperature:", cfg.SampleTemperature)
	core.Println("max tokens:", cfg.SampleMaxTokens)
	core.Println("filter percent:", cfg.FilterShortestPercent)
	// Output:
	// temperature: 1.5
	// max tokens: 65536
	// filter percent: 10
}

// ExampleDefaultSSDCodeBenchmarkConfig shows the LiveCodeBench-v6 evaluation
// defaults: the benchmark name, the repeat count for pass@k, and the decode
// budget.
func ExampleDefaultSSDCodeBenchmarkConfig() {
	cfg := DefaultSSDCodeBenchmarkConfig()
	core.Println("benchmark:", cfg.Benchmark)
	core.Println("n repeat:", cfg.NRepeat)
	core.Println("max tokens:", cfg.Generate.MaxTokens)
	// Output:
	// benchmark: LiveCodeBench-v6
	// n repeat: 20
	// max tokens: 32768
}

// ExampleSSDRecipes lists the released ml-ssd parity recipes, each pairing a
// model with its native data-generation and evaluation defaults.
func ExampleSSDRecipes() {
	recipes := SSDRecipes()
	core.Println("recipes:", len(recipes))
	for _, r := range recipes {
		core.Println(r.Name, "->", r.Model)
	}
	// Output:
	// recipes: 3
	// SimpleSD-4B-instruct -> apple/SimpleSD-4B-instruct
	// SimpleSD-4B-thinking -> apple/SimpleSD-4B-thinking
	// SimpleSD-30b-a3b-instruct -> apple/SimpleSD-30b-a3b-instruct
}

// ExampleLookupSSDRecipe resolves a recipe by its model string, returning the
// descriptor and true; an unknown key returns false.
func ExampleLookupSSDRecipe() {
	recipe, ok := LookupSSDRecipe("apple/SimpleSD-4B-thinking")
	core.Println("found:", ok)
	core.Println("name:", recipe.Name)
	_, miss := LookupSSDRecipe("nope")
	core.Println("miss:", miss)
	// Output:
	// found: true
	// name: SimpleSD-4B-thinking
	// miss: false
}

// ExampleSSDResult_SampleGenerateConfig rebuilds the frozen-model sampling
// config from a result's recorded sampling fields — the exact knobs that
// produced the raw trace.
func ExampleSSDResult_SampleGenerateConfig() {
	result := &SSDResult{SampleMaxTokens: 128, SampleTemperature: 0.6, SampleTopK: 48}
	cfg := result.SampleGenerateConfig()
	core.Println("max tokens:", cfg.MaxTokens)
	core.Println("temperature:", cfg.Temperature)
	core.Println("top k:", cfg.TopK)
	// Output:
	// max tokens: 128
	// temperature: 0.6
	// top k: 48
}

// ExampleSSDResult_DecodeGenerateConfig returns the post-SSD decode config with
// the separately-tuned decode temperature and a caller-owned token budget; the
// sampling-only knobs are dropped.
func ExampleSSDResult_DecodeGenerateConfig() {
	result := &SSDResult{DecodeTemperature: 0.15, SampleTopK: 48}
	cfg := result.DecodeGenerateConfig(2048)
	core.Println("max tokens:", cfg.MaxTokens)
	core.Println("temperature:", cfg.Temperature)
	core.Println("top k:", cfg.TopK)
	// Output:
	// max tokens: 2048
	// temperature: 0.15
	// top k: 0
}
