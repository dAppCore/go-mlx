// SPDX-Licence-Identifier: EUPL-1.2

package hf

import (
	"context"
	"fmt"

	"dappco.re/go/mlx/memory"
)

// ExampleInferJANG shows JANG metadata inference from a model's id, tags and
// filenames. A "jangtq" token (here in the tag list) selects the fixed JANGTQ
// profile; the group size falls back to 64 when no quantization block declares
// one. The filename is only a needle — quant width comes from the profile, not
// the file name.
func ExampleInferJANG() {
	info := InferJANG(ModelMetadata{
		ID:   "dealignai/MiniMax-M2-JANGTQ",
		Tags: []string{"mlx", "jang", "jangtq"},
		Files: []ModelFile{
			{Name: "model-00001-of-00061.safetensors"},
			{Name: "jangtq_runtime.safetensors"},
		},
	})
	fmt.Println(info.Profile, info.WeightFormat, info.BitsDefault, info.GroupSize)
	// Output: JANGTQ mxtq 2 64
}

// ExampleNewRemoteSource constructs a Hugging Face Hub metadata source. The
// constructor trims a trailing slash from the base URL and defaults the
// user-agent when none is supplied — no network is touched here.
func ExampleNewRemoteSource() {
	source := NewRemoteSource(RemoteConfig{
		BaseURL: "https://huggingface.co/",
	})
	fmt.Println(source.baseURL, source.userAgent)
	// Output: https://huggingface.co go-mlx
}

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
