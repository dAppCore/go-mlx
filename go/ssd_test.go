// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"errors"
	"testing"

	"dappco.re/go/mlx/dataset"
	"dappco.re/go/mlx/profile"
)

func TestRunSimpleSelfDistillation_GeneratesRawSFTDataset_Good(t *testing.T) {
	source := dataset.NewSliceDataset([]dataset.Sample{
		{Prompt: "prove a lemma", Meta: map[string]string{"split": "train"}},
		{Text: "free prompt text"},
		{Response: "ignored without prompt"},
	})
	var generatedPrompts []string
	var generatedCfgs []GenerateConfig
	var trainRows []dataset.Sample

	result, err := RunSimpleSelfDistillation(context.Background(), SimpleSelfDistillationRunner{
		Generate: func(_ context.Context, prompt string, cfg GenerateConfig) (string, error) {
			generatedPrompts = append(generatedPrompts, prompt)
			generatedCfgs = append(generatedCfgs, cfg)
			return "raw:" + prompt, nil
		},
		TrainSFT: func(_ context.Context, ds dataset.Dataset, cfg SFTConfig) (*SFTResult, error) {
			if cfg.BatchSize != 2 || cfg.Epochs != 1 {
				t.Fatalf("SFT config = %+v, want caller batch and normalised epoch", cfg)
			}
			if cfg.EvalTemperature != 0.2 {
				t.Fatalf("SFT eval temperature = %f, want SSD decode temperature", cfg.EvalTemperature)
			}
			for {
				sample, ok, err := ds.Next()
				if err != nil {
					t.Fatalf("generated dataset Next() error = %v", err)
				}
				if !ok {
					break
				}
				trainRows = append(trainRows, sample)
			}
			return &SFTResult{Steps: 1, Samples: len(trainRows)}, nil
		},
	}, source, SimpleSelfDistillationConfig{
		SampleMaxTokens:   42,
		SampleTemperature: 0.8,
		SampleTopK:        32,
		SampleTopP:        0.9,
		SampleMinP:        0.05,
		RepetitionPenalty: 1.1,
		DecodeTemperature: 0.2,
		SFT:               SFTConfig{BatchSize: 2},
	})
	if err != nil {
		t.Fatalf("RunSimpleSelfDistillation() error = %v", err)
	}
	if len(generatedPrompts) != 2 || generatedPrompts[0] != "prove a lemma" || generatedPrompts[1] != "free prompt text" {
		t.Fatalf("generated prompts = %#v, want prompt/text rows only", generatedPrompts)
	}
	if generatedCfgs[0].MaxTokens != 42 || generatedCfgs[0].Temperature != 0.8 || generatedCfgs[0].TopK != 32 || generatedCfgs[0].TopP != 0.9 || generatedCfgs[0].MinP != 0.05 || generatedCfgs[0].RepeatPenalty != 1.1 {
		t.Fatalf("generate config = %+v, want sampling config forwarded", generatedCfgs[0])
	}
	if len(trainRows) != 2 || trainRows[0].Prompt != "prove a lemma" || trainRows[0].Response != "raw:prove a lemma" {
		t.Fatalf("train rows = %+v, want raw generated prompt/response rows", trainRows)
	}
	if trainRows[0].Meta["split"] != "train" || trainRows[0].Meta["ssd"] != "simple_self_distillation" || trainRows[0].Meta["ssd_source_index"] != "0" {
		t.Fatalf("train row meta = %+v, want source metadata plus SSD markers", trainRows[0].Meta)
	}
	if result.SampleTemperature != 0.8 || result.DecodeTemperature != 0.2 || result.SampleMaxTokens != 42 ||
		result.SampleTopK != 32 || result.SampleTopP != 0.9 || result.SampleMinP != 0.05 || result.RepetitionPenalty != 1.1 {
		t.Fatalf("result sampling fields = %+v", result)
	}
	if result.SFT == nil || result.SFT.Samples != 2 || len(result.Samples) != 2 {
		t.Fatalf("result = %+v, want SFT result and sampled rows", result)
	}
}

func TestRunSimpleSelfDistillation_Gemma4ModelInfoUsesSharedLoRATargetPolicy_Good(t *testing.T) {
	source := dataset.NewSliceDataset([]dataset.Sample{{Prompt: "explain a retained Gemma state"}})
	var trainCfg SFTConfig

	result, err := RunSimpleSelfDistillation(context.Background(), SimpleSelfDistillationRunner{
		ModelInfo: func(context.Context) ModelInfo {
			return ModelInfo{Architecture: "Gemma4ForConditionalGeneration", NumHeads: 16}
		},
		Generate: func(_ context.Context, prompt string, cfg GenerateConfig) (string, error) {
			if prompt != "explain a retained Gemma state" {
				t.Fatalf("SSD prompt = %q", prompt)
			}
			if cfg.MaxTokens != 77 || cfg.Temperature != 0.8 || cfg.TopK != 24 || cfg.TopP != 0.9 || cfg.MinP != 0.04 || cfg.RepeatPenalty != 1.1 {
				t.Fatalf("SSD sample generate config = %+v", cfg)
			}
			return "retain prompt, cache, and adapter identity", nil
		},
		TrainSFT: func(_ context.Context, ds dataset.Dataset, cfg SFTConfig) (*SFTResult, error) {
			trainCfg = cfg
			sample, ok, err := ds.Next()
			if err != nil {
				t.Fatalf("generated dataset Next() error = %v", err)
			}
			if !ok || sample.Prompt != "explain a retained Gemma state" || sample.Response == "" {
				t.Fatalf("generated training sample = %+v ok=%v", sample, ok)
			}
			return &SFTResult{Steps: 1, Samples: 1}, nil
		},
	}, source, SimpleSelfDistillationConfig{
		SampleMaxTokens:   77,
		SampleTemperature: 0.8,
		SampleTopK:        24,
		SampleTopP:        0.9,
		SampleMinP:        0.04,
		RepetitionPenalty: 1.1,
		DecodeTemperature: 0.25,
	})
	if err != nil {
		t.Fatalf("RunSimpleSelfDistillation() error = %v", err)
	}
	wantTargets := profile.Gemma4DefaultLoRATargets()
	if !equalStringSlices(trainCfg.LoRA.TargetKeys, wantTargets) {
		t.Fatalf("SSD SFT TargetKeys = %v, want Gemma-4 shared defaults %v", trainCfg.LoRA.TargetKeys, wantTargets)
	}
	if !equalStringSlices(trainCfg.LoRA.TargetLayers, wantTargets) {
		t.Fatalf("SSD SFT TargetLayers = %v, want Gemma-4 shared defaults %v", trainCfg.LoRA.TargetLayers, wantTargets)
	}
	if trainCfg.EvalTemperature != 0.25 {
		t.Fatalf("SSD SFT EvalTemperature = %f, want decode temperature", trainCfg.EvalTemperature)
	}
	if result == nil || result.SFT == nil || result.SFT.Samples != 1 || len(result.Samples) != 1 {
		t.Fatalf("SSD result = %+v, want sampled row and SFT result", result)
	}
	sampleCfg := result.SampleGenerateConfig()
	if sampleCfg.MaxTokens != 77 || sampleCfg.Temperature != 0.8 || sampleCfg.TopK != 24 || sampleCfg.TopP != 0.9 || sampleCfg.MinP != 0.04 || sampleCfg.RepeatPenalty != 1.1 {
		t.Fatalf("SampleGenerateConfig() = %+v", sampleCfg)
	}
	decodeCfg := result.DecodeGenerateConfig(4096)
	if decodeCfg.MaxTokens != 4096 || decodeCfg.Temperature != 0.25 {
		t.Fatalf("DecodeGenerateConfig() = %+v", decodeCfg)
	}
}

func TestSimpleSelfDistillationResult_GenerateConfigs_Good(t *testing.T) {
	result := &SimpleSelfDistillationResult{
		SampleMaxTokens:   128,
		SampleTemperature: 0.6,
		SampleTopK:        48,
		SampleTopP:        0.92,
		SampleMinP:        0.03,
		RepetitionPenalty: 1.2,
		DecodeTemperature: 0.15,
	}

	sample := result.SampleGenerateConfig()
	if sample.MaxTokens != 128 || sample.Temperature != 0.6 || sample.TopK != 48 || sample.TopP != 0.92 || sample.MinP != 0.03 || sample.RepeatPenalty != 1.2 {
		t.Fatalf("SampleGenerateConfig() = %+v", sample)
	}
	decode := result.DecodeGenerateConfig(2048)
	if decode.MaxTokens != 2048 || decode.Temperature != 0.15 || decode.TopK != 0 || decode.TopP != 0 || decode.MinP != 0 {
		t.Fatalf("DecodeGenerateConfig() = %+v", decode)
	}

	var nilResult *SimpleSelfDistillationResult
	if got := nilResult.SampleGenerateConfig(); got.MaxTokens != 0 || got.Temperature != 0 || got.TopK != 0 || got.TopP != 0 || got.MinP != 0 || got.RepeatPenalty != 0 {
		t.Fatalf("nil SampleGenerateConfig() = %+v", got)
	}
	if got := nilResult.DecodeGenerateConfig(64); got.MaxTokens != 64 || got.Temperature != 0 {
		t.Fatalf("nil DecodeGenerateConfig() = %+v", got)
	}
}

func TestSimpleSelfDistillationDefaultsAndRecipes_Good(t *testing.T) {
	train := DefaultSimpleSelfDistillationConfig()
	if train.SampleMaxTokens != 65536 || train.SampleTemperature != 1.5 || train.SampleTopK != 20 || train.SampleTopP != 0.8 ||
		train.RepetitionPenalty != 1.0 || train.FilterShortestPercent != 10 {
		t.Fatalf("DefaultSimpleSelfDistillationConfig() = %+v, want ml-ssd data-generation defaults", train)
	}
	eval := DefaultSimpleSelfDistillationCodeBenchmarkConfig()
	if eval.Benchmark != "LiveCodeBench-v6" || eval.NRepeat != 20 || eval.Generate.MaxTokens != 32768 ||
		eval.Generate.Temperature != 0.6 || eval.Generate.TopP != 0.95 || eval.Generate.TopK != 20 || len(eval.Seeds) != 4 || eval.Seeds[0] != 0 {
		t.Fatalf("DefaultSimpleSelfDistillationCodeBenchmarkConfig() = %+v, want ml-ssd eval defaults", eval)
	}

	recipes := SimpleSelfDistillationRecipes()
	if len(recipes) != 3 {
		t.Fatalf("SimpleSelfDistillationRecipes() = %d, want released ml-ssd recipes", len(recipes))
	}
	recipe, ok := LookupSimpleSelfDistillationRecipe("apple/SimpleSD-4B-thinking")
	if !ok || recipe.Name != SimpleSelfDistillationRecipe4BThinking || recipe.Dataset != "microsoft/rStar-Coder" || recipe.DatasetConfig != "seed_sft" {
		t.Fatalf("LookupSimpleSelfDistillationRecipe() = %+v/%t", recipe, ok)
	}
	if _, ok := LookupSimpleSelfDistillationRecipe("missing"); ok {
		t.Fatal("LookupSimpleSelfDistillationRecipe(missing) ok = true")
	}
}

func TestRunSimpleSelfDistillation_FiltersShortestGenerations_Good(t *testing.T) {
	source := dataset.NewSliceDataset([]dataset.Sample{
		{Prompt: "p0"},
		{Prompt: "p1"},
		{Prompt: "p2"},
		{Prompt: "p3"},
		{Prompt: "p4"},
		{Prompt: "p5"},
		{Prompt: "p6"},
		{Prompt: "p7"},
		{Prompt: "p8"},
		{Prompt: "p9"},
	})
	responses := map[string]string{
		"p0": "x",
		"p1": "medium response",
		"p2": "longer response text",
		"p3": "longer response text plus detail",
		"p4": "longer response text plus detail again",
		"p5": "answer with enough body",
		"p6": "answer with enough body and evidence",
		"p7": "answer with enough body and evidence plus notes",
		"p8": "answer with enough body and evidence plus notes twice",
		"p9": "answer with enough body and evidence plus notes twice over",
	}
	var trainRows []dataset.Sample

	result, err := RunSimpleSelfDistillation(context.Background(), SimpleSelfDistillationRunner{
		Generate: func(_ context.Context, prompt string, _ GenerateConfig) (string, error) {
			return responses[prompt], nil
		},
		TrainSFT: func(_ context.Context, ds dataset.Dataset, _ SFTConfig) (*SFTResult, error) {
			for {
				sample, ok, err := ds.Next()
				if err != nil {
					t.Fatalf("generated dataset Next() error = %v", err)
				}
				if !ok {
					break
				}
				trainRows = append(trainRows, sample)
			}
			return &SFTResult{Samples: len(trainRows)}, nil
		},
	}, source, SimpleSelfDistillationConfig{FilterShortestPercent: 10})
	if err != nil {
		t.Fatalf("RunSimpleSelfDistillation() error = %v", err)
	}
	if len(result.Samples) != 10 {
		t.Fatalf("sampled rows = %d, want all raw generations recorded", len(result.Samples))
	}
	if len(trainRows) != 9 {
		t.Fatalf("train rows = %d, want shortest decile filtered before SFT", len(trainRows))
	}
	for _, row := range trainRows {
		if row.Prompt == "p0" {
			t.Fatalf("train rows include shortest response: %+v", trainRows)
		}
	}
	if result.FilterShortestPercent != 10 {
		t.Fatalf("FilterShortestPercent = %f, want 10", result.FilterShortestPercent)
	}
}

func TestRunSimpleSelfDistillation_Defaults_Good(t *testing.T) {
	var gotCfg GenerateConfig
	_, err := RunSimpleSelfDistillation(context.Background(), SimpleSelfDistillationRunner{
		Generate: func(_ context.Context, _ string, cfg GenerateConfig) (string, error) {
			gotCfg = cfg
			return "answer", nil
		},
		TrainSFT: func(context.Context, dataset.Dataset, SFTConfig) (*SFTResult, error) {
			return &SFTResult{Steps: 1}, nil
		},
	}, dataset.NewSliceDataset([]dataset.Sample{{Prompt: "p"}}), SimpleSelfDistillationConfig{})
	if err != nil {
		t.Fatalf("RunSimpleSelfDistillation() error = %v", err)
	}
	if gotCfg.MaxTokens != defaultSimpleSelfDistillationMaxTokens ||
		gotCfg.Temperature != defaultSimpleSelfDistillationTemperature ||
		gotCfg.TopK != defaultSimpleSelfDistillationTopK ||
		gotCfg.TopP != defaultSimpleSelfDistillationTopP {
		t.Fatalf("default generate config = %+v", gotCfg)
	}
}

func TestRunSimpleSelfDistillation_RejectsUnitSampleTemperature_Bad(t *testing.T) {
	_, err := RunSimpleSelfDistillation(context.Background(), SimpleSelfDistillationRunner{
		Generate: func(context.Context, string, GenerateConfig) (string, error) { return "", nil },
		TrainSFT: func(context.Context, dataset.Dataset, SFTConfig) (*SFTResult, error) {
			return &SFTResult{}, nil
		},
	}, dataset.NewSliceDataset([]dataset.Sample{{Prompt: "p"}}), SimpleSelfDistillationConfig{SampleTemperature: 1})
	if err == nil {
		t.Fatal("RunSimpleSelfDistillation() error = nil, want unit-temperature rejection")
	}
}

func TestRunSimpleSelfDistillation_ReturnsPartialResultOnSFTError_Ugly(t *testing.T) {
	wantErr := errors.New("train failed")
	result, err := RunSimpleSelfDistillation(context.Background(), SimpleSelfDistillationRunner{
		Generate: func(context.Context, string, GenerateConfig) (string, error) { return "raw", nil },
		TrainSFT: func(context.Context, dataset.Dataset, SFTConfig) (*SFTResult, error) {
			return &SFTResult{Samples: 1}, wantErr
		},
	}, dataset.NewSliceDataset([]dataset.Sample{{Prompt: "p"}}), SimpleSelfDistillationConfig{})
	if !errors.Is(err, wantErr) {
		t.Fatalf("RunSimpleSelfDistillation() error = %v, want %v", err, wantErr)
	}
	if result == nil || len(result.Samples) != 1 || result.SFT == nil || result.SFT.Samples != 1 {
		t.Fatalf("partial result = %+v, want sampled rows and partial SFT result", result)
	}
}
