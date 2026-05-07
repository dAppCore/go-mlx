// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && !nomlx

package mlx

import (
	"context"
	"testing"

	core "dappco.re/go"
)

func requireRealEvalModel(t *testing.T) string {
	t.Helper()
	if core.Getenv("GO_MLX_RUN_MODEL_EVAL_TESTS") != "1" {
		t.Skip("set GO_MLX_RUN_MODEL_EVAL_TESTS=1 to enable real model eval tests")
	}
	modelPath := core.Getenv("GO_MLX_EVAL_MODEL")
	if modelPath == "" {
		t.Skip("set GO_MLX_EVAL_MODEL to a local model pack")
	}
	return modelPath
}

func TestRunModelEval_RealModelSkip_Good(t *testing.T) {
	modelPath := requireRealEvalModel(t)
	model, err := LoadModel(modelPath, WithContextLength(512), WithBatchSize(1))
	if err != nil {
		t.Fatalf("LoadModel() error = %v", err)
	}
	t.Cleanup(func() {
		_ = model.Close()
		ClearCache()
	})

	report, err := RunModelEval(context.Background(), model, NewSFTSliceDataset([]SFTSample{
		{Text: "Local evaluation should produce a finite loss."},
	}), EvalConfig{Batch: DatasetBatchConfig{BatchSize: 1, MaxSeqLen: 64}})
	if err != nil {
		t.Fatalf("RunModelEval() error = %v", err)
	}
	if report.Metrics.Tokens == 0 || report.Metrics.Perplexity == 0 {
		t.Fatalf("metrics = %+v, want tokens and perplexity", report.Metrics)
	}
}

func TestRunModelEval_RealModelLoRASkip_Ugly(t *testing.T) {
	modelPath := requireRealEvalModel(t)
	adapterPath := core.Getenv("GO_MLX_EVAL_ADAPTER")
	if adapterPath == "" {
		t.Skip("set GO_MLX_EVAL_ADAPTER to a local LoRA adapter package")
	}
	model, err := LoadModel(modelPath, WithContextLength(512), WithBatchSize(1))
	if err != nil {
		t.Fatalf("LoadModel() error = %v", err)
	}
	t.Cleanup(func() {
		_ = model.Close()
		ClearCache()
	})

	report, err := RunModelEval(context.Background(), model, NewSFTSliceDataset([]SFTSample{
		{Prompt: "Explain local MLX eval.", Response: "It computes masked token loss over a dataset."},
	}), EvalConfig{AdapterPath: adapterPath, Batch: DatasetBatchConfig{BatchSize: 1, MaxSeqLen: 96}})
	if err != nil {
		t.Fatalf("RunModelEval() error = %v", err)
	}
	if report.Adapter.Path == "" || report.Metrics.Tokens == 0 {
		t.Fatalf("adapter=%+v metrics=%+v, want adapter identity and tokens", report.Adapter, report.Metrics)
	}
}
