// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"errors"
	"testing"

	"dappco.re/go/mlx/dataset"
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
		DecodeTemperature: 0.2,
		SFT:               SFTConfig{BatchSize: 2},
	})
	if err != nil {
		t.Fatalf("RunSimpleSelfDistillation() error = %v", err)
	}
	if len(generatedPrompts) != 2 || generatedPrompts[0] != "prove a lemma" || generatedPrompts[1] != "free prompt text" {
		t.Fatalf("generated prompts = %#v, want prompt/text rows only", generatedPrompts)
	}
	if generatedCfgs[0].MaxTokens != 42 || generatedCfgs[0].Temperature != 0.8 || generatedCfgs[0].TopK != 32 || generatedCfgs[0].TopP != 0.9 || generatedCfgs[0].MinP != 0.05 {
		t.Fatalf("generate config = %+v, want sampling config forwarded", generatedCfgs[0])
	}
	if len(trainRows) != 2 || trainRows[0].Prompt != "prove a lemma" || trainRows[0].Response != "raw:prove a lemma" {
		t.Fatalf("train rows = %+v, want raw generated prompt/response rows", trainRows)
	}
	if trainRows[0].Meta["split"] != "train" || trainRows[0].Meta["ssd"] != "simple_self_distillation" || trainRows[0].Meta["ssd_source_index"] != "0" {
		t.Fatalf("train row meta = %+v, want source metadata plus SSD markers", trainRows[0].Meta)
	}
	if result.SampleTemperature != 0.8 || result.DecodeTemperature != 0.2 || result.SampleMaxTokens != 42 ||
		result.SampleTopK != 32 || result.SampleTopP != 0.9 || result.SampleMinP != 0.05 {
		t.Fatalf("result sampling fields = %+v", result)
	}
	if result.SFT == nil || result.SFT.Samples != 2 || len(result.Samples) != 2 {
		t.Fatalf("result = %+v, want SFT result and sampled rows", result)
	}
}

func TestSimpleSelfDistillationResult_GenerateConfigs_Good(t *testing.T) {
	result := &SimpleSelfDistillationResult{
		SampleMaxTokens:   128,
		SampleTemperature: 0.6,
		SampleTopK:        48,
		SampleTopP:        0.92,
		SampleMinP:        0.03,
		DecodeTemperature: 0.15,
	}

	sample := result.SampleGenerateConfig()
	if sample.MaxTokens != 128 || sample.Temperature != 0.6 || sample.TopK != 48 || sample.TopP != 0.92 || sample.MinP != 0.03 {
		t.Fatalf("SampleGenerateConfig() = %+v", sample)
	}
	decode := result.DecodeGenerateConfig(2048)
	if decode.MaxTokens != 2048 || decode.Temperature != 0.15 || decode.TopK != 0 || decode.TopP != 0 || decode.MinP != 0 {
		t.Fatalf("DecodeGenerateConfig() = %+v", decode)
	}

	var nilResult *SimpleSelfDistillationResult
	if got := nilResult.SampleGenerateConfig(); got.MaxTokens != 0 || got.Temperature != 0 || got.TopK != 0 || got.TopP != 0 || got.MinP != 0 {
		t.Fatalf("nil SampleGenerateConfig() = %+v", got)
	}
	if got := nilResult.DecodeGenerateConfig(64); got.MaxTokens != 64 || got.Temperature != 0 {
		t.Fatalf("nil DecodeGenerateConfig() = %+v", got)
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
