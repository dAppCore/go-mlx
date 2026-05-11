// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"dappco.re/go/inference/quant/jang"
	"dappco.re/go/mlx/model/minimax/m2"
	"dappco.re/go/mlx/probe"
)

// Legacy aliases — the canonical MiniMax M2 implementation lives at
// dappco.re/go/mlx/model/minimax/m2/. mlx-root callers keep their
// existing MiniMaxM2* surface via these aliases.
type (
	MiniMaxM2Config                    = m2.Config
	MiniMaxM2TensorRole                = m2.TensorRole
	MiniMaxM2TensorSpec                = m2.TensorSpec
	MiniMaxM2TensorPlan                = m2.TensorPlan
	MiniMaxM2RouterDecision            = m2.RouterDecision
	MiniMaxM2ExpertFunc                = m2.ExpertFunc
	MiniMaxM2PackedExpertWeights       = m2.PackedExpertWeights
	MiniMaxM2RouterWeights             = m2.RouterWeights
	MiniMaxM2PackedLayerForwardOptions = m2.PackedLayerForwardOptions
	MiniMaxM2PackedLayerForwardResult  = m2.PackedLayerForwardResult
	MiniMaxM2LazyExpertLoad            = m2.LazyExpertLoad
	MiniMaxM2DenseProjectionTensor     = m2.DenseProjectionTensor
	MiniMaxM2DenseExpertWeights        = m2.DenseExpertWeights
	MiniMaxM2ResolvedTensor            = m2.ResolvedTensor
	MiniMaxM2LayerForwardSkeleton      = m2.LayerForwardSkeleton
	JANGPackedProjectionTensor         = m2.JANGPackedProjectionTensor
)

// Tensor role constants forwarded from the m2 package.
const (
	MiniMaxM2TensorRoleAttentionQ = m2.TensorRoleAttentionQ
	MiniMaxM2TensorRoleAttentionK = m2.TensorRoleAttentionK
	MiniMaxM2TensorRoleAttentionV = m2.TensorRoleAttentionV
	MiniMaxM2TensorRoleAttentionO = m2.TensorRoleAttentionO
	MiniMaxM2TensorRoleRouterGate = m2.TensorRoleRouterGate
	MiniMaxM2TensorRoleRouterBias = m2.TensorRoleRouterBias
	MiniMaxM2TensorRoleExpertGate = m2.TensorRoleExpertGate
	MiniMaxM2TensorRoleExpertUp   = m2.TensorRoleExpertUp
	MiniMaxM2TensorRoleExpertDown = m2.TensorRoleExpertDown
)

// ParseMiniMaxM2Config parses a HuggingFace MiniMax M2 config payload.
//
//	cfg, err := mlx.ParseMiniMaxM2Config(data)
func ParseMiniMaxM2Config(data []byte) (MiniMaxM2Config, error) {
	return m2.ParseConfig(data)
}

// BuildMiniMaxM2TensorPlan builds the MiniMax M2 tensor plan from
// config and optional JANG quantization metadata.
//
//	plan, err := mlx.BuildMiniMaxM2TensorPlan(cfg, jangInfo)
func BuildMiniMaxM2TensorPlan(cfg MiniMaxM2Config, info *jang.Info) (MiniMaxM2TensorPlan, error) {
	return m2.BuildTensorPlan(cfg, info)
}

// RouteMiniMaxM2Tokens produces deterministic top-k expert routing decisions.
//
//	decisions, err := mlx.RouteMiniMaxM2Tokens(cfg, scores, bias)
func RouteMiniMaxM2Tokens(cfg MiniMaxM2Config, scores [][]float32, bias []float32) ([]MiniMaxM2RouterDecision, error) {
	return m2.RouteTokens(cfg, scores, bias)
}

// DispatchMiniMaxM2Experts applies fake expert functions for fixture
// dispatch tests.
//
//	out, err := mlx.DispatchMiniMaxM2Experts(hidden, decisions, experts)
func DispatchMiniMaxM2Experts(hidden [][]float32, decisions []MiniMaxM2RouterDecision, experts map[int]MiniMaxM2ExpertFunc) ([][]float32, error) {
	return m2.DispatchExperts(hidden, decisions, experts)
}

// LoadMiniMaxM2PackedExpertsForDecisionsFromSafetensors loads only the
// routed-selected packed experts from safetensors shards.
//
//	experts, err := mlx.LoadMiniMaxM2PackedExpertsForDecisionsFromSafetensors(plan, files, layer, decisions)
func LoadMiniMaxM2PackedExpertsForDecisionsFromSafetensors(plan MiniMaxM2TensorPlan, weightFiles []string, layer int, decisions []MiniMaxM2RouterDecision) (map[int]MiniMaxM2PackedExpertWeights, error) {
	return m2.LoadPackedExpertsForDecisions(plan, weightFiles, layer, decisions)
}

// LoadMiniMaxM2LazyExpertsForHiddenFromSafetensors routes hidden states
// and loads only the routed packed experts.
//
//	load, err := mlx.LoadMiniMaxM2LazyExpertsForHiddenFromSafetensors(plan, files, layer, hidden, tokens, sink)
func LoadMiniMaxM2LazyExpertsForHiddenFromSafetensors(plan MiniMaxM2TensorPlan, weightFiles []string, layer int, hidden [][]float32, tokenIDs []int32, sink probe.Sink) (MiniMaxM2LazyExpertLoad, error) {
	return m2.LoadLazyExpertsForHidden(plan, weightFiles, layer, hidden, tokenIDs, sink)
}

// LoadMiniMaxM2PackedExpertsFromSafetensors loads packed experts by ID.
//
//	experts, err := mlx.LoadMiniMaxM2PackedExpertsFromSafetensors(plan, files, layer, ids)
func LoadMiniMaxM2PackedExpertsFromSafetensors(plan MiniMaxM2TensorPlan, weightFiles []string, layer int, expertIDs []int) (map[int]MiniMaxM2PackedExpertWeights, error) {
	return m2.LoadPackedExperts(plan, weightFiles, layer, expertIDs)
}

// DequantizeJANGPackedProjection dequantises a packed JANG projection
// tensor into a dense host-side projection.
//
//	dense, err := mlx.DequantizeJANGPackedProjection(tensor)
func DequantizeJANGPackedProjection(tensor JANGPackedProjectionTensor) (MiniMaxM2DenseProjectionTensor, error) {
	return m2.DequantizeJANGPackedProjection(tensor)
}

// LoadMiniMaxM2RouterFromSafetensors loads the dense router projection
// for one MiniMax M2 MoE layer.
//
//	router, err := mlx.LoadMiniMaxM2RouterFromSafetensors(plan, files, layer)
func LoadMiniMaxM2RouterFromSafetensors(plan MiniMaxM2TensorPlan, weightFiles []string, layer int) (MiniMaxM2RouterWeights, error) {
	return m2.LoadRouter(plan, weightFiles, layer)
}

// ProjectMiniMaxM2RouterScores projects hidden states through the
// dense router weights to produce per-expert scores.
//
//	scores, err := mlx.ProjectMiniMaxM2RouterScores(hidden, router)
func ProjectMiniMaxM2RouterScores(hidden [][]float32, router MiniMaxM2RouterWeights) ([][]float32, error) {
	return m2.ProjectRouterScores(hidden, router)
}

// BuildMiniMaxM2LayerForwardSkeletonFromSafetensors resolves first-layer
// MiniMax M2 attention + router tensors from safetensors headers.
//
//	skel, err := mlx.BuildMiniMaxM2LayerForwardSkeletonFromSafetensors(plan, files, layer)
func BuildMiniMaxM2LayerForwardSkeletonFromSafetensors(plan MiniMaxM2TensorPlan, weightFiles []string, layer int) (MiniMaxM2LayerForwardSkeleton, error) {
	return m2.BuildLayerForwardSkeleton(plan, weightFiles, layer)
}

// MiniMaxM2RouterProbeEvents emits router-decision probe events for a layer.
//
//	events := mlx.MiniMaxM2RouterProbeEvents(layer, tokenIDs, decisions)
func MiniMaxM2RouterProbeEvents(layer int, tokenIDs []int32, decisions []MiniMaxM2RouterDecision) []probe.Event {
	return m2.RouterProbeEvents(layer, tokenIDs, decisions)
}
