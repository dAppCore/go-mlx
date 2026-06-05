// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference/bench"
	"dappco.re/go/mlx/pkg/metal"
)

func ExampleRunFastEvalBench() {
	model, _ := exampleFastEvalModel("ok")

	report, err := RunFastEvalBench(context.Background(), model, bench.Config{
		Prompt:    "prompt",
		MaxTokens: 1,
		Runs:      1,
	})

	core.Println(err == nil, report.Generation.Runs, report.Generation.GeneratedTokens, report.ModelInfo.Adapter.Name)
	// Output: true 1 1 demo-lora
}

func ExampleRunFastEval() {
	runner := bench.Runner{
		Generate: func(context.Context, string, bench.GenerateOptions) (bench.Generation, error) {
			return bench.Generation{
				Text:    "ok",
				Metrics: bench.GenerationMetrics{GeneratedTokens: 1},
			}, nil
		},
	}

	report, err := RunFastEval(context.Background(), runner, bench.Config{Prompt: "prompt", MaxTokens: 1, Runs: 1})

	core.Println(err == nil, report.Generation.Runs, report.Generation.Samples[0].Text)
	// Output: true 1 ok
}

func ExampleNewModelFastEvalRunner() {
	model, native := exampleFastEvalModel("runner")
	runner := NewModelFastEvalRunner(model)

	info := runner.Info(context.Background())
	generation, err := runner.Generate(context.Background(), "prompt", bench.GenerateOptions{MaxTokens: 3})

	core.Println(info.Architecture, info.Adapter.Name, generation.Text, native.lastGenerateConfig.MaxTokens, err == nil)
	// Output: gemma4_text demo-lora runner 3 true
}

func exampleFastEvalModel(text string) (*Model, *fakeNativeModel) {
	native := &fakeNativeModel{
		info: metal.ModelInfo{
			Architecture:  "gemma4_text",
			ContextLength: 262144,
			Adapter: metal.AdapterInfo{
				Name:       "demo-lora",
				TargetKeys: []string{"q_proj", "v_proj", "o_proj"},
			},
		},
		modelType: "gemma4_text",
		tokens:    []metal.Token{{ID: 1, Text: text}},
		metrics:   metal.Metrics{GeneratedTokens: 1},
	}
	return &Model{model: native}, native
}
