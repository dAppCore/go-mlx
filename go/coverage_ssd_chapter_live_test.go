// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"testing"

	"dappco.re/go/inference"
	"dappco.re/go/mlx/chaptersmoke"
	"dappco.re/go/mlx/dataset"
)

// TestModel_RunSSDLiveModel drives Model.RunSSD on the live model — the
// self-distillation SAMPLING path. With no SFT checkpoint dir and capture
// disabled it samples the frozen model and stops at the scored trace, which is
// exactly the generateForSSD body (0% without a real model) plus the package
// RunSSD passthrough. The SFT phase is left unconfigured: this asserts the
// sampling lane runs and returns a result, not that it trains.
func TestModel_RunSSDLiveModel(t *testing.T) {
	m := requireLiveE2BModel(t)

	ds := dataset.NewSliceDataset([]dataset.Sample{
		{Prompt: "Name a primary colour."},
		{Prompt: "Name a day of the week."},
	})

	cfg := DefaultSSDConfig()
	cfg.SampleMaxTokens = 8
	cfg.SampleTemperature = 0
	cfg.DisableCapture = true

	result, err := m.RunSSD(context.Background(), ds, cfg)
	if err != nil {
		t.Fatalf("RunSSD: %v", err)
	}
	if result == nil {
		t.Fatal("RunSSD returned nil result")
	}
	t.Logf("RunSSD: %d samples captured", len(result.Samples))
	if len(result.Samples) == 0 {
		t.Errorf("RunSSD captured no samples")
	}
}

// TestModel_RunSSDNilGuards covers the package RunSSD validation guards reached
// before any generation: nil dataset and a runner with no Generate func.
func TestModel_RunSSDNilGuards(t *testing.T) {
	if _, err := RunSSD(context.Background(), SSDRunner{}, nil, DefaultSSDConfig()); err == nil {
		t.Error("RunSSD with nil dataset should error")
	}
	ds := dataset.NewSliceDataset([]dataset.Sample{{Prompt: "x"}})
	if _, err := RunSSD(context.Background(), SSDRunner{}, ds, DefaultSSDConfig()); err == nil {
		t.Error("RunSSD with nil Generate func should error")
	}
}

// TestModel_RunSSDCodeBenchmarkLiveModel drives the package RunSSDCodeBenchmark
// passthrough with a real-model-backed generate runner over one trivial task.
func TestModel_RunSSDCodeBenchmarkLiveModel(t *testing.T) {
	m := requireLiveE2BModel(t)

	runner := SSDCodeBenchmarkRunner{
		// generateForSSD has the exact (ctx, prompt, GenerateConfig) shape the
		// runner's Generate field wants.
		Generate: m.generateForSSD,
		// RunTests marks every candidate as failing — the benchmark only needs a
		// non-nil tester to score pass@k; real code execution is out of scope.
		RunTests: func(ctx context.Context, sample SSDCodeBenchmarkSample, candidate SSDCodeCandidate) (SSDCodeExecution, error) {
			return SSDCodeExecution{Passed: false}, nil
		},
	}
	samples := []SSDCodeBenchmarkSample{
		{ID: "echo", Prompt: "Write a function that returns 1."},
	}
	cfg := DefaultSSDCodeBenchmarkConfig()
	cfg.NRepeat = 1
	cfg.Generate.MaxTokens = 16
	cfg.Generate.Temperature = 0

	report, err := RunSSDCodeBenchmark(context.Background(), runner, samples, cfg)
	if err != nil {
		t.Fatalf("RunSSDCodeBenchmark: %v", err)
	}
	if report == nil {
		t.Fatal("RunSSDCodeBenchmark returned nil report")
	}
	t.Logf("RunSSDCodeBenchmark: %d sample results", len(report.Results))
}

// TestModel_StateKVChapterSmokeLiveModel drives RunModelStateKVChapterSmoke on
// the live model — the chapter-smoke runner's Capture (prefill+sleep) and
// Generate (wake+append+decode) closures plus NewModelStateKVChapterRunner,
// chapterGenerateConfig, and stateKVChapterGenerateOptions, all 0% without a
// real model.
func TestModel_StateKVChapterSmokeLiveModel(t *testing.T) {
	m := requireLiveE2BModel(t)

	cfg := chaptersmoke.Config{
		AnswerMaxTokens: 24,
		Temperature:     0,
		Chapters: []chaptersmoke.Input{
			{
				Name:          "keeper",
				Text:          "The lighthouse keeper is named Snider and his lamp burns teal.",
				Question:      " What colour does the lamp burn?",
				ExpectedTerms: []string{"teal"},
			},
		},
	}

	report, err := RunModelStateKVChapterSmoke(context.Background(), m, cfg)
	if err != nil {
		t.Fatalf("RunModelStateKVChapterSmoke: %v", err)
	}
	if report == nil {
		t.Fatal("RunModelStateKVChapterSmoke returned nil report")
	}
	t.Logf("chapter smoke: %d chapters", len(report.Chapters))

	// The nil-model guard arm.
	if _, err := RunModelStateKVChapterSmoke(context.Background(), nil, cfg); err == nil {
		t.Error("RunModelStateKVChapterSmoke(nil model) should error")
	}
}

// TestTrainingModel_LiveModel drives the TrainingModel facade — casting the live
// *metaladapter (an inference.TrainableModel) to reach its InternalModel.
func TestTrainingModel_LiveModel(t *testing.T) {
	adapter, done := requireLiveE2BAdapter(t)
	defer done()

	var trainable inference.TrainableModel = adapter
	internal := TrainingModel(trainable)
	if internal == nil {
		t.Fatal("TrainingModel returned nil InternalModel")
	}
}
