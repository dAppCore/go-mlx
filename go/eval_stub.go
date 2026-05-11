// SPDX-Licence-Identifier: EUPL-1.2

//go:build !(darwin && arm64) || nomlx

package mlx

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/mlx/lora"
)

// NewModelEvalRunner returns an eval runner that reports native unavailability.
func NewModelEvalRunner(model *Model) EvalRunner {
	return EvalRunner{
		Info: func(ctx context.Context) ModelInfo {
			if err := ctx.Err(); err != nil || model == nil {
				return ModelInfo{}
			}
			return model.Info()
		},
		Tokenizer: func(ctx context.Context) *Tokenizer {
			if err := ctx.Err(); err != nil || model == nil {
				return nil
			}
			return model.Tokenizer()
		},
		LoadAdapter: func(context.Context, string) (lora.AdapterInfo, error) {
			return lora.AdapterInfo{}, unsupportedBuildError()
		},
		EvaluateBatch: func(context.Context, SFTBatch) (EvalBatchMetrics, error) {
			return EvalBatchMetrics{}, core.NewError("mlx: native dataset eval requires darwin/arm64 MLX support")
		},
	}
}
