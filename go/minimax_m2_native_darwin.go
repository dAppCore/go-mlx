// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && !nomlx

package mlx

import (
	"dappco.re/go/mlx/model/minimax/m2"
)

// DispatchMiniMaxM2PackedExpertsMetal applies router-selected MiniMax
// M2 packed experts using fused JANG/JANGTQ projection kernels.
//
//	out, err := mlx.DispatchMiniMaxM2PackedExpertsMetal(hidden, decisions, experts)
func DispatchMiniMaxM2PackedExpertsMetal(hidden [][]float32, decisions []MiniMaxM2RouterDecision, experts map[int]MiniMaxM2PackedExpertWeights) ([][]float32, error) {
	return m2.DispatchPackedExpertsMetal(hidden, decisions, experts)
}

// DispatchMiniMaxM2PackedExpertsFromSafetensorsMetal loads the
// router-selected packed experts from safetensors shards and executes
// the fused Metal dispatch.
//
//	out, err := mlx.DispatchMiniMaxM2PackedExpertsFromSafetensorsMetal(plan, files, layer, hidden, decisions)
func DispatchMiniMaxM2PackedExpertsFromSafetensorsMetal(plan MiniMaxM2TensorPlan, weightFiles []string, layer int, hidden [][]float32, decisions []MiniMaxM2RouterDecision) ([][]float32, error) {
	return m2.DispatchPackedExpertsFromSafetensorsMetal(plan, weightFiles, layer, hidden, decisions)
}

// ForwardMiniMaxM2LazyExpertLoadMetal executes an already-routed lazy
// expert load with the native packed projection kernels.
//
//	result, err := mlx.ForwardMiniMaxM2LazyExpertLoadMetal(hidden, load)
func ForwardMiniMaxM2LazyExpertLoadMetal(hidden [][]float32, load MiniMaxM2LazyExpertLoad) (MiniMaxM2PackedLayerForwardResult, error) {
	return m2.ForwardLazyExpertLoadMetal(hidden, load)
}

// ForwardMiniMaxM2PackedLayerMetal routes hidden states through a
// MiniMax M2 packed MoE layer skeleton, lazily resolving selected
// experts from safetensors and emitting router probe events.
//
//	result, err := mlx.ForwardMiniMaxM2PackedLayerMetal(opts)
func ForwardMiniMaxM2PackedLayerMetal(opts MiniMaxM2PackedLayerForwardOptions) (MiniMaxM2PackedLayerForwardResult, error) {
	return m2.ForwardPackedLayerMetal(opts)
}

// ForwardMiniMaxM2PackedLayerFromSafetensorsMetal reads the dense
// router gate, computes router scores, then runs the packed layer
// skeleton with lazy expert resolution.
//
//	result, err := mlx.ForwardMiniMaxM2PackedLayerFromSafetensorsMetal(opts)
func ForwardMiniMaxM2PackedLayerFromSafetensorsMetal(opts MiniMaxM2PackedLayerForwardOptions) (MiniMaxM2PackedLayerForwardResult, error) {
	return m2.ForwardPackedLayerFromSafetensorsMetal(opts)
}
