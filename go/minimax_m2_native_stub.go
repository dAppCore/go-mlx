// SPDX-Licence-Identifier: EUPL-1.2

//go:build !(darwin && arm64) || nomlx

package mlx

import core "dappco.re/go"

// DispatchMiniMaxM2PackedExpertsMetal requires the native Metal backend.
func DispatchMiniMaxM2PackedExpertsMetal(_ [][]float32, _ []MiniMaxM2RouterDecision, _ map[int]MiniMaxM2PackedExpertWeights) ([][]float32, error) {
	return nil, core.NewError("mlx: MiniMax M2 packed expert dispatch requires darwin/arm64 native MLX support")
}

// DispatchMiniMaxM2PackedExpertsFromSafetensorsMetal requires the native Metal backend.
func DispatchMiniMaxM2PackedExpertsFromSafetensorsMetal(_ MiniMaxM2TensorPlan, _ []string, _ int, _ [][]float32, _ []MiniMaxM2RouterDecision) ([][]float32, error) {
	return nil, core.NewError("mlx: MiniMax M2 packed expert dispatch requires darwin/arm64 native MLX support")
}

// ForwardMiniMaxM2LazyExpertLoadMetal requires the native Metal backend.
func ForwardMiniMaxM2LazyExpertLoadMetal(_ [][]float32, _ MiniMaxM2LazyExpertLoad) (MiniMaxM2PackedLayerForwardResult, error) {
	return MiniMaxM2PackedLayerForwardResult{}, core.NewError("mlx: MiniMax M2 packed layer forward requires darwin/arm64 native MLX support")
}

// ForwardMiniMaxM2PackedLayerMetal requires the native Metal backend.
func ForwardMiniMaxM2PackedLayerMetal(_ MiniMaxM2PackedLayerForwardOptions) (MiniMaxM2PackedLayerForwardResult, error) {
	return MiniMaxM2PackedLayerForwardResult{}, core.NewError("mlx: MiniMax M2 packed layer forward requires darwin/arm64 native MLX support")
}

// ForwardMiniMaxM2PackedLayerFromSafetensorsMetal requires the native Metal backend.
func ForwardMiniMaxM2PackedLayerFromSafetensorsMetal(_ MiniMaxM2PackedLayerForwardOptions) (MiniMaxM2PackedLayerForwardResult, error) {
	return MiniMaxM2PackedLayerForwardResult{}, core.NewError("mlx: MiniMax M2 packed layer forward requires darwin/arm64 native MLX support")
}
