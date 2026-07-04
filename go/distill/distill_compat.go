// SPDX-Licence-Identifier: EUPL-1.2

package distill

import (
	"time"

	mlx "dappco.re/go/mlx"
	"dappco.re/go/mlx/dataset"
)

// ModelInfo, Tokenizer and SFTBatch are the root model-metadata, tokenizer and
// SFT batch types this distillation package operates on. Aliased here so the
// extracted package reads against the engine's types; distill depends on mlx
// one-way (the root never imports distill) so there is no import cycle.
type (
	ModelInfo = mlx.ModelInfo
	Tokenizer = mlx.Tokenizer
	SFTBatch  = mlx.SFTBatch
	Batch     = mlx.Batch
)

// BuildDatasetBatches is the engine's dataset-batch builder, re-bound here so
// the extracted package calls it by name — function values, unlike types,
// cannot be aliased, so a package var holds the reference.
var BuildDatasetBatches = mlx.BuildDatasetBatches

// nonZeroDuration / normalizeDatasetBatchConfig are small leaf helpers carried
// with the package on extraction (unexported root helpers in training.go /
// dataset_stream.go, not importable across the package boundary).
func nonZeroDuration(duration time.Duration) time.Duration {
	if duration <= 0 {
		return time.Nanosecond
	}
	return duration
}

func normalizeDatasetBatchConfig(cfg dataset.BatchConfig) dataset.BatchConfig {
	if cfg.BatchSize <= 0 {
		cfg.BatchSize = 1
	}
	return cfg
}
