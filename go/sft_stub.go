// SPDX-Licence-Identifier: EUPL-1.2

//go:build !(darwin && arm64) || nomlx

package mlx

import "context"

// TrainSFT returns unsupported on builds without native MLX.
func (m *Model) TrainSFT(_ context.Context, _ SFTDataset, _ SFTConfig) (*SFTResult, error) {
	return nil, unsupportedBuildError()
}
