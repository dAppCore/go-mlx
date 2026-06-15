// SPDX-Licence-Identifier: EUPL-1.2

package hf

import (
	"context"
	"fmt"

	"dappco.re/go/mlx/memory"
)

// ExamplePlanFits estimates local Apple fit for a model whose metadata comes
// from an injected source (the same ModelSource interface a network-backed
// RemoteSource satisfies — no network in this example). The report ranks
// models and records the resolved architecture and quantisation.
func ExamplePlanFits() {
	source := &fakeHFModelSource{
		byID: map[string]ModelMetadata{
			"Qwen/Qwen3-0.6B": {
				ID: "Qwen/Qwen3-0.6B",
				Config: ModelConfig{
					ModelType:             "qwen3",
					HiddenSize:            1024,
					NumHiddenLayers:       28,
					NumAttentionHeads:     16,
					NumKeyValueHeads:      8,
					MaxPositionEmbeddings: 40960,
					Quantization:          &QuantizationConfig{Bits: 4, GroupSize: 64},
				},
				Files: []ModelFile{{Name: "model.safetensors", Size: 420 * 1024 * 1024}},
			},
		},
	}

	report, err := PlanFits(context.Background(), FitConfig{
		ModelIDs: []string{"Qwen/Qwen3-0.6B"},
		Device: memory.DeviceInfo{
			MemorySize:                   96 * memory.GiB,
			MaxRecommendedWorkingSetSize: 86 * memory.GiB,
		},
		Source: source,
	})
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	plan := report.Models[0]
	fmt.Println(plan.ModelID, plan.Architecture, plan.QuantBits, plan.Source)
	// Output: Qwen/Qwen3-0.6B qwen3 4 huggingface
}

// ExamplePlanFits_unsupportedArchitecture shows the advisory side of a fit
// report. An architecture the native loaders don't recognise still gets a
// memory estimate, but the plan is flagged unsupported / non-loadable and
// carries an explanatory note so a caller can surface *why* the model won't
// run — no network (metadata comes from an injected source).
func ExamplePlanFits_unsupportedArchitecture() {
	source := &fakeHFModelSource{
		byID: map[string]ModelMetadata{
			"future/model": {
				ID: "future/model",
				Config: ModelConfig{
					ModelType:             "future_arch",
					HiddenSize:            4096,
					NumHiddenLayers:       32,
					NumAttentionHeads:     32,
					MaxPositionEmbeddings: 32768,
				},
				Files: []ModelFile{{Name: "model.safetensors", Size: 2 * 1024 * 1024 * 1024}},
			},
		},
	}

	report, err := PlanFits(context.Background(), FitConfig{
		ModelIDs: []string{"future/model"},
		Device: memory.DeviceInfo{
			MemorySize:                   96 * memory.GiB,
			MaxRecommendedWorkingSetSize: 86 * memory.GiB,
		},
		Source: source,
	})
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	plan := report.Models[0]
	fmt.Println(plan.Architecture, plan.SupportedArchitecture, plan.NativeLoadable)
	fmt.Println(plan.Notes[0])
	// Output:
	// future_arch false false
	// architecture is not currently supported by native go-mlx loaders
}
