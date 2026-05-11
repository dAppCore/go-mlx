// SPDX-Licence-Identifier: EUPL-1.2

//go:build !(darwin && arm64) || nomlx

package mlx

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference/eval"
)

// NewModelEvalRunner returns an eval runner that reports native unavailability.
func NewModelEvalRunner(_ *Model) eval.Runner {
	return eval.Runner{
		EvaluateBatch: func(context.Context, eval.Batch) (eval.BatchMetrics, error) {
			return eval.BatchMetrics{}, core.NewError("mlx: native dataset eval requires darwin/arm64 MLX support")
		},
	}
}
