// SPDX-Licence-Identifier: EUPL-1.2

//go:build !(darwin && arm64) || nomlx

package mlx

import (
	"context"

	"dappco.re/go/mlx/dataset"
)

// TrainSFT returns unsupported on builds without native MLX.
func (m *Model) TrainSFT(_ context.Context, _ dataset.Dataset, _ SFTConfig) (*SFTResult, error) {
	return nil, unsupportedBuildError()
}
