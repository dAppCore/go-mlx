// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"math"
	"time"

	core "dappco.re/go"
	"dappco.re/go/mlx/lora"
)

const EvalReportVersion = 1

// EvalConfig controls dataset-native perplexity and small quality probes.
type EvalConfig struct {
	Batch         DatasetBatchConfig `json:"batch"`
	AdapterPath   string             `json:"adapter_path,omitempty"`
	MaxSamples    int                `json:"max_samples,omitempty"`
	QualityProbes []EvalQualityProbe `json:"-"`
}

// EvalRunner supplies the model operations needed for dataset evaluation.
type EvalRunner struct {
	Info          func(context.Context) ModelInfo
	Tokenizer     func(context.Context) *Tokenizer
	LoadAdapter   func(context.Context, string) (lora.AdapterInfo, error)
	BuildBatches  func(context.Context, SFTDataset, DatasetBatchConfig) ([]SFTBatch, error)
	EvaluateBatch func(context.Context, SFTBatch) (EvalBatchMetrics, error)
}

// EvalBatchMetrics is the loss result for one tokenized batch.
type EvalBatchMetrics struct {
	Samples int     `json:"samples,omitempty"`
	Tokens  int     `json:"tokens,omitempty"`
	Loss    float64 `json:"loss,omitempty"`
}

// EvalMetrics aggregates loss and perplexity over a dataset stream.
type EvalMetrics struct {
	Samples    int     `json:"samples,omitempty"`
	Batches    int     `json:"batches,omitempty"`
	Tokens     int     `json:"tokens,omitempty"`
	Loss       float64 `json:"loss,omitempty"`
	Perplexity float64 `json:"perplexity,omitempty"`
}

// EvalReport is a JSON-friendly native eval result.
type EvalReport struct {
	Version   int               `json:"version"`
	ModelInfo ModelInfo         `json:"model_info"`
	Adapter   lora.AdapterInfo   `json:"adapter,omitempty"`
	Config    EvalConfig        `json:"config"`
	Metrics   EvalMetrics       `json:"metrics"`
	Quality   EvalQualityReport `json:"quality"`
	Duration  time.Duration     `json:"duration,omitempty"`
}

// EvalQualityProbe adds a custom deterministic quality check.
type EvalQualityProbe struct {
	Name  string                                    `json:"name"`
	Check func(EvalQualityContext) EvalQualityCheck `json:"-"`
}

// EvalQualityContext is passed to custom eval probes.
type EvalQualityContext struct {
	Config    EvalConfig
	Samples   []SFTSample
	Metrics   EvalMetrics
	ModelInfo ModelInfo
	Adapter   lora.AdapterInfo
}

// EvalQualityReport contains small deterministic checks over eval data and metrics.
type EvalQualityReport struct {
	Checks []EvalQualityCheck `json:"checks,omitempty"`
}

// EvalQualityCheck is one quality probe result.
type EvalQualityCheck struct {
	Name   string  `json:"name"`
	Pass   bool    `json:"pass"`
	Score  float64 `json:"score"`
	Detail string  `json:"detail,omitempty"`
}

// RunModelEval evaluates a loaded model over an SFT/JSONL dataset stream.
func RunModelEval(ctx context.Context, model *Model, dataset SFTDataset, cfg EvalConfig) (*EvalReport, error) {
	if model == nil {
		return nil, core.NewError("mlx: model is nil")
	}
	return RunDatasetEval(ctx, NewModelEvalRunner(model), dataset, cfg)
}

// RunDatasetEval evaluates perplexity and quality probes over a dataset stream.
func RunDatasetEval(ctx context.Context, runner EvalRunner, dataset SFTDataset, cfg EvalConfig) (*EvalReport, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	cfg = normalizeEvalConfig(cfg)
	if runner.EvaluateBatch == nil {
		return nil, core.NewError("mlx: eval runner requires EvaluateBatch")
	}
	if dataset == nil {
		return nil, core.NewError("mlx: eval dataset is nil")
	}

	start := time.Now()
	samples, err := collectEvalSamples(ctx, dataset, cfg.MaxSamples)
	if err != nil {
		return nil, err
	}
	if len(samples) == 0 {
		return nil, core.NewError("mlx: eval dataset produced no samples")
	}

	report := &EvalReport{
		Version: EvalReportVersion,
		Config:  cfg,
	}
	if runner.Info != nil {
		report.ModelInfo = runner.Info(ctx)
		report.Adapter = report.ModelInfo.Adapter
	}
	if cfg.AdapterPath != "" {
		if runner.LoadAdapter == nil {
			return nil, core.NewError("mlx: eval runner does not support LoRA adapter loading")
		}
		adapter, err := runner.LoadAdapter(ctx, cfg.AdapterPath)
		if err != nil {
			return nil, err
		}
		report.Adapter = adapter
		if runner.Info != nil {
			report.ModelInfo = runner.Info(ctx)
		}
		if report.ModelInfo.Adapter.IsEmpty() {
			report.ModelInfo.Adapter = adapter
		}
	}
	if report.Adapter.IsEmpty() {
		report.Adapter = report.ModelInfo.Adapter
	}

	batches, err := evalBatches(ctx, runner, NewSFTSliceDataset(samples), cfg.Batch)
	if err != nil {
		return nil, err
	}
	if len(batches) == 0 {
		return nil, core.NewError("mlx: eval dataset produced no tokenized batches")
	}

	metrics, err := evaluateBatches(ctx, runner, batches, len(samples))
	if err != nil {
		return nil, err
	}
	report.Metrics = metrics
	report.Duration = nonZeroDuration(time.Since(start))
	report.Quality = runEvalQualityProbes(EvalQualityContext{
		Config:    cfg,
		Samples:   samples,
		Metrics:   metrics,
		ModelInfo: report.ModelInfo,
		Adapter:   report.Adapter,
	})
	return report, nil
}

func normalizeEvalConfig(cfg EvalConfig) EvalConfig {
	cfg.Batch = normalizeDatasetBatchConfig(cfg.Batch)
	cfg.QualityProbes = append([]EvalQualityProbe(nil), cfg.QualityProbes...)
	return cfg
}

func collectEvalSamples(ctx context.Context, dataset SFTDataset, maxSamples int) ([]SFTSample, error) {
	var samples []SFTSample
	for {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if maxSamples > 0 && len(samples) >= maxSamples {
			break
		}
		sample, ok, err := dataset.Next()
		if err != nil {
			return nil, err
		}
		if !ok {
			break
		}
		samples = append(samples, cloneSFTSample(sample))
	}
	return samples, nil
}

func evalBatches(ctx context.Context, runner EvalRunner, dataset SFTDataset, cfg DatasetBatchConfig) ([]SFTBatch, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if runner.BuildBatches != nil {
		return runner.BuildBatches(ctx, dataset, cfg)
	}
	if runner.Tokenizer == nil {
		return nil, core.NewError("mlx: eval runner requires Tokenizer or BuildBatches")
	}
	tok := runner.Tokenizer(ctx)
	return BuildDatasetBatches(tok, dataset, cfg)
}

func evaluateBatches(ctx context.Context, runner EvalRunner, batches []SFTBatch, samples int) (EvalMetrics, error) {
	metrics := EvalMetrics{Samples: samples, Batches: len(batches)}
	var weightedLoss float64
	for _, batch := range batches {
		if err := ctx.Err(); err != nil {
			return EvalMetrics{}, err
		}
		batchMetrics, err := runner.EvaluateBatch(ctx, batch)
		if err != nil {
			return EvalMetrics{}, err
		}
		if batchMetrics.Tokens <= 0 {
			batchMetrics.Tokens = sftBatchLossTokens(batch)
		}
		if batchMetrics.Tokens <= 0 {
			continue
		}
		if math.IsNaN(batchMetrics.Loss) || math.IsInf(batchMetrics.Loss, 0) {
			return EvalMetrics{}, core.NewError("mlx: eval batch loss is not finite")
		}
		metrics.Tokens += batchMetrics.Tokens
		weightedLoss += batchMetrics.Loss * float64(batchMetrics.Tokens)
	}
	if metrics.Tokens == 0 {
		return EvalMetrics{}, core.NewError("mlx: eval produced no loss tokens")
	}
	metrics.Loss = weightedLoss / float64(metrics.Tokens)
	metrics.Perplexity = math.Exp(metrics.Loss)
	return metrics, nil
}

func sftBatchLossTokens(batch SFTBatch) int {
	tokens := 0
	if len(batch.Batch.LossMask) > 0 {
		for _, row := range batch.Batch.LossMask {
			for _, value := range row {
				if value > 0 {
					tokens++
				}
			}
		}
		return tokens
	}
	if len(batch.Batch.Length) > 0 {
		for _, length := range batch.Batch.Length {
			if length > 0 {
				tokens += length
			}
		}
		return tokens
	}
	for _, row := range batch.Batch.Tokens {
		tokens += len(row)
	}
	return tokens
}

func runEvalQualityProbes(ctx EvalQualityContext) EvalQualityReport {
	checks := defaultEvalQualityChecks(ctx)
	for _, probe := range ctx.Config.QualityProbes {
		check := EvalQualityCheck{Name: probe.Name}
		if probe.Check == nil {
			check.Pass = false
			check.Detail = "probe has no check function"
		} else {
			check = probe.Check(ctx)
			if check.Name == "" {
				check.Name = probe.Name
			}
		}
		checks = append(checks, check)
	}
	return EvalQualityReport{Checks: checks}
}

func defaultEvalQualityChecks(ctx EvalQualityContext) []EvalQualityCheck {
	samples := len(ctx.Samples)
	responseLike := 0
	for _, sample := range ctx.Samples {
		if core.Trim(sample.Text) != "" || core.Trim(sample.Response) != "" {
			responseLike++
		}
	}
	lossFinite := !math.IsNaN(ctx.Metrics.Loss) && !math.IsInf(ctx.Metrics.Loss, 0) && ctx.Metrics.Loss >= 0
	pplFinite := !math.IsNaN(ctx.Metrics.Perplexity) && !math.IsInf(ctx.Metrics.Perplexity, 0) && ctx.Metrics.Perplexity >= 1
	return []EvalQualityCheck{
		{Name: "samples_present", Pass: samples > 0, Score: boolScore(samples > 0), Detail: core.Sprintf("%d", samples)},
		{Name: "token_coverage", Pass: ctx.Metrics.Tokens > 0, Score: boolScore(ctx.Metrics.Tokens > 0), Detail: core.Sprintf("%d", ctx.Metrics.Tokens)},
		{Name: "loss_finite", Pass: lossFinite, Score: boolScore(lossFinite), Detail: core.Sprintf("%.6f", ctx.Metrics.Loss)},
		{Name: "perplexity_finite", Pass: pplFinite, Score: boolScore(pplFinite), Detail: core.Sprintf("%.6f", ctx.Metrics.Perplexity)},
		{Name: "response_coverage", Pass: responseLike == samples, Score: fractionScore(responseLike, samples), Detail: core.Sprintf("%d/%d", responseLike, samples)},
	}
}

func fractionScore(numerator, denominator int) float64 {
	if denominator <= 0 {
		return 0
	}
	return float64(numerator) / float64(denominator)
}
