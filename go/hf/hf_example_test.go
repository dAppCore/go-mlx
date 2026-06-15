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

// ExampleInferJANG_filenameNeedle shows that the JANGTQ profile is selected
// from a weight *filename* alone — neither the id nor the tags carry a needle
// here. A "jangtq" filename is the strongest signal and pins the JANGTQ
// profile (2-bit MXTQ, group size 64) just as a tag would.
func ExampleInferJANG_filenameNeedle() {
	info := InferJANG(ModelMetadata{
		ID: "acme/MiniMax-M2",
		Files: []ModelFile{
			{Name: "model-00001-of-00061.safetensors"},
			{Name: "jangtq_runtime.safetensors"},
		},
	})
	fmt.Println(info.Profile, info.WeightFormat, info.BitsDefault, info.GroupSize)
	// Output: JANGTQ mxtq 2 64
}

// ExampleInferJANG_noNeedle shows the negative result: a model with no JANG
// needle in its id, tags or filenames is not a JANG model, so InferJANG
// returns nil. Callers treat nil as "ordinary (non-JANG) weights".
func ExampleInferJANG_noNeedle() {
	info := InferJANG(ModelMetadata{
		ID:    "Qwen/Qwen3-0.6B",
		Tags:  []string{"mlx", "text-generation"},
		Files: []ModelFile{{Name: "model.safetensors"}},
	})
	fmt.Println(info == nil)
	// Output: true
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
