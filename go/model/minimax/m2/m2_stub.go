// SPDX-Licence-Identifier: EUPL-1.2

//go:build !(darwin && arm64) || nomlx

package m2

import core "dappco.re/go"

// DispatchPackedExpertsMetal requires the native Metal backend.
func DispatchPackedExpertsMetal(_ [][]float32, _ []RouterDecision, _ map[int]PackedExpertWeights) ([][]float32, error) {
	return nil, core.NewError("mlx: MiniMax M2 packed expert dispatch requires darwin/arm64 native MLX support")
}

// DispatchPackedExpertsFromSafetensorsMetal requires the native Metal backend.
func DispatchPackedExpertsFromSafetensorsMetal(_ TensorPlan, _ []string, _ int, _ [][]float32, _ []RouterDecision) ([][]float32, error) {
	return nil, core.NewError("mlx: MiniMax M2 packed expert dispatch requires darwin/arm64 native MLX support")
}

// ForwardLazyExpertLoadMetal requires the native Metal backend.
func ForwardLazyExpertLoadMetal(_ [][]float32, _ LazyExpertLoad) (PackedLayerForwardResult, error) {
	return PackedLayerForwardResult{}, core.NewError("mlx: MiniMax M2 packed layer forward requires darwin/arm64 native MLX support")
}

// ForwardPackedLayerMetal requires the native Metal backend.
func ForwardPackedLayerMetal(_ PackedLayerForwardOptions) (PackedLayerForwardResult, error) {
	return PackedLayerForwardResult{}, core.NewError("mlx: MiniMax M2 packed layer forward requires darwin/arm64 native MLX support")
}

// ForwardPackedLayerFromSafetensorsMetal requires the native Metal backend.
func ForwardPackedLayerFromSafetensorsMetal(_ PackedLayerForwardOptions) (PackedLayerForwardResult, error) {
	return PackedLayerForwardResult{}, core.NewError("mlx: MiniMax M2 packed layer forward requires darwin/arm64 native MLX support")
}
