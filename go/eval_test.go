// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"math"
	"testing"

	core "dappco.re/go"
)

func TestRunDatasetEval_AggregatesPerplexityAdapterAndQuality_Good(t *testing.T) {
	loadCalled := false
	customCalled := false
	buildCalled := false
	evalCalls := 0
	adapter := LoRAAdapterInfo{Name: "ethics-lora", Path: "/adapters/ethics-lora", Rank: 8, Alpha: 16, Scale: 2}
	runner := EvalRunner{
		Info: func(context.Context) ModelInfo {
			return ModelInfo{Architecture: "qwen3", NumLayers: 28, Adapter: adapter}
		},
		LoadAdapter: func(_ context.Context, path string) (LoRAAdapterInfo, error) {
			if path != adapter.Path {
				t.Fatalf("LoadAdapter path = %q, want %q", path, adapter.Path)
			}
			loadCalled = true
			return adapter, nil
		},
		BuildBatches: func(_ context.Context, dataset SFTDataset, cfg DatasetBatchConfig) ([]SFTBatch, error) {
			if cfg.BatchSize != 2 || cfg.MaxSeqLen != 16 {
				t.Fatalf("batch config = %+v, want batch 2 max seq 16", cfg)
			}
			var samples int
			for {
				_, ok, err := dataset.Next()
				if err != nil {
					return nil, err
				}
				if !ok {
					break
				}
				samples++
			}
			if samples != 2 {
				t.Fatalf("BuildBatches saw %d samples, want 2", samples)
			}
			buildCalled = true
			return []SFTBatch{
				{Batch: Batch{Tokens: [][]int{{1, 2, 3}}, LossMask: [][]float32{{1, 1, 1}}}},
				{Batch: Batch{Tokens: [][]int{{4, 5}}, LossMask: [][]float32{{1, 1}}}},
			}, nil
		},
		EvaluateBatch: func(_ context.Context, batch SFTBatch) (EvalBatchMetrics, error) {
			evalCalls++
			switch evalCalls {
			case 1:
				return EvalBatchMetrics{Tokens: sftBatchLossTokens(batch), Loss: 2.0}, nil
			case 2:
				return EvalBatchMetrics{Tokens: sftBatchLossTokens(batch), Loss: 1.0}, nil
			default:
				t.Fatalf("unexpected eval call %d", evalCalls)
				return EvalBatchMetrics{}, nil
			}
		},
	}

	report, err := RunDatasetEval(context.Background(), runner, NewSFTSliceDataset([]SFTSample{
		{Prompt: "Why?", Response: "Because."},
		{Text: "plain eval text"},
	}), EvalConfig{
		Batch:       DatasetBatchConfig{BatchSize: 2, MaxSeqLen: 16},
		AdapterPath: adapter.Path,
		QualityProbes: []EvalQualityProbe{{
			Name: "custom_probe",
			Check: func(ctx EvalQualityContext) EvalQualityCheck {
				customCalled = true
				if ctx.Metrics.Tokens != 5 || ctx.Adapter.Name != adapter.Name || len(ctx.Samples) != 2 {
					t.Fatalf("quality context = %+v adapter=%+v samples=%d", ctx.Metrics, ctx.Adapter, len(ctx.Samples))
				}
				return EvalQualityCheck{Name: "custom_probe", Pass: true, Score: 0.75, Detail: "mock"}
			},
		}},
	})
	if err != nil {
		t.Fatalf("RunDatasetEval() error = %v", err)
	}
	if !loadCalled || !buildCalled || !customCalled || evalCalls != 2 {
		t.Fatalf("calls load=%v build=%v custom=%v eval=%d", loadCalled, buildCalled, customCalled, evalCalls)
	}
	if report.Version != EvalReportVersion {
		t.Fatalf("Version = %d, want %d", report.Version, EvalReportVersion)
	}
	if report.ModelInfo.Architecture != "qwen3" || report.Adapter.Name != adapter.Name {
		t.Fatalf("model/adapter = %+v / %+v", report.ModelInfo, report.Adapter)
	}
	wantLoss := 1.6
	if math.Abs(report.Metrics.Loss-wantLoss) > 0.0001 {
		t.Fatalf("loss = %.4f, want %.4f", report.Metrics.Loss, wantLoss)
	}
	if report.Metrics.Samples != 2 || report.Metrics.Batches != 2 || report.Metrics.Tokens != 5 {
		t.Fatalf("metrics = %+v, want samples=2 batches=2 tokens=5", report.Metrics)
	}
	if math.Abs(report.Metrics.Perplexity-math.Exp(wantLoss)) > 0.0001 {
		t.Fatalf("perplexity = %.4f, want %.4f", report.Metrics.Perplexity, math.Exp(wantLoss))
	}
	if !evalQualityPassed(report.Quality, "loss_finite") || !evalQualityPassed(report.Quality, "custom_probe") {
		t.Fatalf("quality checks = %+v", report.Quality.Checks)
	}
}

func TestRunDatasetEval_RequiresBatchEvaluator_Bad(t *testing.T) {
	_, err := RunDatasetEval(context.Background(), EvalRunner{}, NewSFTSliceDataset([]SFTSample{{Text: "x"}}), EvalConfig{})
	if err == nil {
		t.Fatal("expected missing evaluator error")
	}
}

func TestRunDatasetEval_DerivesTokensFromLossMask_Ugly(t *testing.T) {
	runner := EvalRunner{
		BuildBatches: func(context.Context, SFTDataset, DatasetBatchConfig) ([]SFTBatch, error) {
			return []SFTBatch{{
				Batch: Batch{
					Tokens:   [][]int{{1, 2, 3, 4}},
					LossMask: [][]float32{{0, 1, 0.25, 1}},
				},
			}}, nil
		},
		EvaluateBatch: func(context.Context, SFTBatch) (EvalBatchMetrics, error) {
			return EvalBatchMetrics{Loss: 0.5}, nil
		},
	}

	report, err := RunDatasetEval(context.Background(), runner, NewSFTSliceDataset([]SFTSample{{Text: "masked"}}), EvalConfig{})
	if err != nil {
		t.Fatalf("RunDatasetEval() error = %v", err)
	}
	if report.Metrics.Tokens != 3 {
		t.Fatalf("tokens = %d, want rounded loss-mask count 3", report.Metrics.Tokens)
	}
	if !evalQualityPassed(report.Quality, "token_coverage") {
		t.Fatalf("quality checks = %+v", report.Quality.Checks)
	}
}

func TestRunDatasetEval_ReportsRunnerErrors_Ugly(t *testing.T) {
	wantErr := core.NewError("mock loss failed")
	runner := EvalRunner{
		BuildBatches: func(context.Context, SFTDataset, DatasetBatchConfig) ([]SFTBatch, error) {
			return []SFTBatch{{Batch: Batch{Tokens: [][]int{{1, 2}}, LossMask: [][]float32{{1, 1}}}}}, nil
		},
		EvaluateBatch: func(context.Context, SFTBatch) (EvalBatchMetrics, error) {
			return EvalBatchMetrics{}, wantErr
		},
	}
	_, err := RunDatasetEval(context.Background(), runner, NewSFTSliceDataset([]SFTSample{{Text: "x"}}), EvalConfig{})
	if err == nil || !core.Contains(err.Error(), wantErr.Error()) {
		t.Fatalf("error = %v, want %v", err, wantErr)
	}
}

func evalQualityPassed(report EvalQualityReport, name string) bool {
	for _, check := range report.Checks {
		if check.Name == name {
			return check.Pass
		}
	}
	return false
}
