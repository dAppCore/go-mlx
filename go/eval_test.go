// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"math"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/lora"
)

func TestRunDatasetEval_AggregatesPerplexityAdapterAndQuality_Good(t *testing.T) {
	loadCalled := false
	customCalled := false
	buildCalled := false
	evalCalls := 0
	adapter := lora.AdapterInfo{Name: "ethics-lora", Path: "/adapters/ethics-lora", Rank: 8, Alpha: 16, Scale: 2}
	runner := EvalRunner{
		Info: func(context.Context) ModelInfo {
			return ModelInfo{Architecture: "qwen3", NumLayers: 28, Adapter: adapter}
		},
		LoadAdapter: func(_ context.Context, path string) (lora.AdapterInfo, error) {
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

func TestRunDatasetEval_ErrorBranches_Bad(t *testing.T) {
	if _, err := RunModelEval(context.Background(), nil, NewSFTSliceDataset([]SFTSample{{Text: "x"}}), EvalConfig{}); err == nil {
		t.Fatal("expected nil model eval error")
	}
	runner := EvalRunner{EvaluateBatch: func(context.Context, SFTBatch) (EvalBatchMetrics, error) {
		return EvalBatchMetrics{Tokens: 1, Loss: 0.1}, nil
	}}
	if _, err := RunDatasetEval(context.Background(), runner, nil, EvalConfig{}); err == nil {
		t.Fatal("expected nil dataset error")
	}
	if _, err := RunDatasetEval(context.Background(), runner, NewSFTSliceDataset(nil), EvalConfig{}); err == nil {
		t.Fatal("expected empty dataset error")
	}
	if _, err := RunDatasetEval(context.Background(), runner, NewSFTSliceDataset([]SFTSample{{Text: "x"}}), EvalConfig{AdapterPath: "adapter"}); err == nil {
		t.Fatal("expected unsupported adapter loading error")
	}
	if _, err := evalBatches(context.Background(), runner, NewSFTSliceDataset([]SFTSample{{Text: "x"}}), DatasetBatchConfig{}); err == nil {
		t.Fatal("expected missing tokenizer/build batches error")
	}

	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := collectEvalSamples(cancelled, NewSFTSliceDataset([]SFTSample{{Text: "x"}}), 0); err != context.Canceled {
		t.Fatalf("collectEvalSamples(cancelled) = %v, want context.Canceled", err)
	}
	if _, err := evaluateBatches(cancelled, runner, []SFTBatch{{Batch: Batch{Tokens: [][]int{{1}}}}}, 1); err != context.Canceled {
		t.Fatalf("evaluateBatches(cancelled) = %v, want context.Canceled", err)
	}
}

func TestEvaluateBatches_ErrorBranches_Ugly(t *testing.T) {
	nonFinite := EvalRunner{EvaluateBatch: func(context.Context, SFTBatch) (EvalBatchMetrics, error) {
		return EvalBatchMetrics{Tokens: 1, Loss: math.Inf(1)}, nil
	}}
	if _, err := evaluateBatches(context.Background(), nonFinite, []SFTBatch{{Batch: Batch{Tokens: [][]int{{1}}}}}, 1); err == nil {
		t.Fatal("expected non-finite loss error")
	}
	noTokens := EvalRunner{EvaluateBatch: func(context.Context, SFTBatch) (EvalBatchMetrics, error) {
		return EvalBatchMetrics{Loss: 0.2}, nil
	}}
	if _, err := evaluateBatches(context.Background(), noTokens, []SFTBatch{{}}, 1); err == nil {
		t.Fatal("expected no loss tokens error")
	}

	if got := sftBatchLossTokens(SFTBatch{Batch: Batch{Length: []int{2, 0, 3}}}); got != 5 {
		t.Fatalf("sftBatchLossTokens(length) = %d, want 5", got)
	}
	if got := sftBatchLossTokens(SFTBatch{Batch: Batch{Tokens: [][]int{{1, 2}, {3}}}}); got != 3 {
		t.Fatalf("sftBatchLossTokens(tokens) = %d, want 3", got)
	}
	if got := fractionScore(1, 0); got != 0 {
		t.Fatalf("fractionScore(1,0) = %f, want 0", got)
	}
}

func TestEvalQualityProbes_NilAndDefaultNames_Ugly(t *testing.T) {
	report := runEvalQualityProbes(EvalQualityContext{
		Config: EvalConfig{QualityProbes: []EvalQualityProbe{
			{Name: "nil_probe"},
			{Name: "default_name", Check: func(EvalQualityContext) EvalQualityCheck {
				return EvalQualityCheck{Pass: true, Score: 1}
			}},
		}},
		Samples: []SFTSample{{}},
		Metrics: EvalMetrics{Tokens: 0, Loss: math.NaN(), Perplexity: math.Inf(1)},
	})
	if !evalQualityPassed(report, "default_name") {
		t.Fatalf("quality checks = %+v, want default_name pass", report.Checks)
	}
	if evalQualityPassed(report, "nil_probe") {
		t.Fatalf("quality checks = %+v, nil probe should fail", report.Checks)
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
