// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && !nomlx

package mlx

import (
	"math"

	core "dappco.re/go"
)

// DispatchMiniMaxM2PackedExpertsMetal applies router-selected MiniMax M2
// packed experts using fused JANG/JANGTQ projection kernels for gate, up, and
// down projections. It is intentionally host-shaped for bring-up fixtures and
// model-loader validation; full model execution keeps tensors on device.
func DispatchMiniMaxM2PackedExpertsMetal(hidden [][]float32, decisions []MiniMaxM2RouterDecision, experts map[int]MiniMaxM2PackedExpertWeights) ([][]float32, error) {
	out := make([][]float32, len(hidden))
	for _, decision := range decisions {
		if decision.TokenIndex < 0 || decision.TokenIndex >= len(hidden) {
			return nil, core.NewError(core.Sprintf("mlx: MiniMax M2 packed dispatch token index %d out of range", decision.TokenIndex))
		}
		if len(decision.ExpertIDs) != len(decision.Weights) {
			return nil, core.NewError("mlx: MiniMax M2 packed dispatch expert/weight length mismatch")
		}
		for i, expertID := range decision.ExpertIDs {
			expert, ok := experts[expertID]
			if !ok {
				return nil, core.NewError(core.Sprintf("mlx: MiniMax M2 packed dispatch missing expert %d", expertID))
			}
			result, err := runMiniMaxM2PackedExpertMetal(hidden[decision.TokenIndex], expert)
			if err != nil {
				return nil, core.E("minimax_m2.packed_dispatch", core.Sprintf("expert %d", expertID), err)
			}
			if out[decision.TokenIndex] == nil {
				out[decision.TokenIndex] = make([]float32, len(result))
			}
			if len(result) != len(out[decision.TokenIndex]) {
				return nil, core.NewError("mlx: MiniMax M2 packed dispatch expert output shape mismatch")
			}
			for j, value := range result {
				out[decision.TokenIndex][j] += decision.Weights[i] * value
			}
		}
	}
	return out, nil
}

// DispatchMiniMaxM2PackedExpertsFromSafetensorsMetal loads the router-selected
// packed experts from safetensors shards and executes the fused Metal dispatch.
func DispatchMiniMaxM2PackedExpertsFromSafetensorsMetal(plan MiniMaxM2TensorPlan, weightFiles []string, layer int, hidden [][]float32, decisions []MiniMaxM2RouterDecision) ([][]float32, error) {
	experts, err := LoadMiniMaxM2PackedExpertsForDecisionsFromSafetensors(plan, weightFiles, layer, decisions)
	if err != nil {
		return nil, err
	}
	return DispatchMiniMaxM2PackedExpertsMetal(hidden, decisions, experts)
}

// ForwardMiniMaxM2LazyExpertLoadMetal executes an already-routed lazy expert
// load with the native packed projection kernels.
func ForwardMiniMaxM2LazyExpertLoadMetal(hidden [][]float32, load MiniMaxM2LazyExpertLoad) (MiniMaxM2PackedLayerForwardResult, error) {
	output, err := DispatchMiniMaxM2PackedExpertsMetal(hidden, load.Decisions, load.Experts)
	if err != nil {
		return MiniMaxM2PackedLayerForwardResult{}, err
	}
	return MiniMaxM2PackedLayerForwardResult{
		Output:            output,
		Decisions:         append([]MiniMaxM2RouterDecision(nil), load.Decisions...),
		SelectedExpertIDs: append([]int(nil), load.SelectedExpertIDs...),
		LoadedPackedBytes: load.LoadedPackedBytes,
		ProbeEvents:       append([]ProbeEvent(nil), load.ProbeEvents...),
	}, nil
}

// ForwardMiniMaxM2PackedLayerMetal routes hidden states through a MiniMax M2
// packed MoE layer skeleton, lazily resolving selected experts from safetensors
// and emitting router probe events.
func ForwardMiniMaxM2PackedLayerMetal(opts MiniMaxM2PackedLayerForwardOptions) (MiniMaxM2PackedLayerForwardResult, error) {
	if len(opts.Hidden) != len(opts.RouterScores) {
		return MiniMaxM2PackedLayerForwardResult{}, core.NewError(core.Sprintf("mlx: MiniMax M2 packed layer hidden rows %d, router rows %d", len(opts.Hidden), len(opts.RouterScores)))
	}
	decisions, err := RouteMiniMaxM2Tokens(opts.Plan.Config, opts.RouterScores, opts.RouterBias)
	if err != nil {
		return MiniMaxM2PackedLayerForwardResult{}, err
	}
	experts, err := LoadMiniMaxM2PackedExpertsForDecisionsFromSafetensors(opts.Plan, opts.WeightFiles, opts.Layer, decisions)
	if err != nil {
		return MiniMaxM2PackedLayerForwardResult{}, err
	}
	output, err := DispatchMiniMaxM2PackedExpertsMetal(opts.Hidden, decisions, experts)
	if err != nil {
		return MiniMaxM2PackedLayerForwardResult{}, err
	}
	events := MiniMaxM2RouterProbeEvents(opts.Layer, opts.TokenIDs, decisions)
	for _, event := range events {
		if opts.ProbeSink != nil {
			opts.ProbeSink.EmitProbe(event)
		}
	}
	return MiniMaxM2PackedLayerForwardResult{
		Output:            output,
		Decisions:         decisions,
		SelectedExpertIDs: miniMaxM2DecisionExpertIDsSorted(decisions),
		LoadedPackedBytes: miniMaxM2PackedExpertLoadedBytes(experts),
		ProbeEvents:       events,
	}, nil
}

// ForwardMiniMaxM2PackedLayerFromSafetensorsMetal reads the dense router gate,
// computes router scores, then runs the packed layer skeleton with lazy expert
// resolution.
func ForwardMiniMaxM2PackedLayerFromSafetensorsMetal(opts MiniMaxM2PackedLayerForwardOptions) (MiniMaxM2PackedLayerForwardResult, error) {
	if len(opts.RouterBias) == 0 {
		load, err := LoadMiniMaxM2LazyExpertsForHiddenFromSafetensors(opts.Plan, opts.WeightFiles, opts.Layer, opts.Hidden, opts.TokenIDs, opts.ProbeSink)
		if err != nil {
			return MiniMaxM2PackedLayerForwardResult{}, err
		}
		return ForwardMiniMaxM2LazyExpertLoadMetal(opts.Hidden, load)
	}
	router, err := LoadMiniMaxM2RouterFromSafetensors(opts.Plan, opts.WeightFiles, opts.Layer)
	if err != nil {
		return MiniMaxM2PackedLayerForwardResult{}, err
	}
	scores, err := ProjectMiniMaxM2RouterScores(opts.Hidden, router)
	if err != nil {
		return MiniMaxM2PackedLayerForwardResult{}, err
	}
	opts.RouterScores = scores
	if len(opts.RouterBias) == 0 {
		opts.RouterBias = router.Bias
	}
	return ForwardMiniMaxM2PackedLayerMetal(opts)
}

func runMiniMaxM2PackedExpertMetal(hidden []float32, expert MiniMaxM2PackedExpertWeights) ([]float32, error) {
	inputShape := []int32{1, int32(len(hidden))}
	gate, err := projectMiniMaxM2PackedTensorMetal(expert.GateProj, hidden, inputShape)
	if err != nil {
		return nil, core.E("minimax_m2.packed_expert", "gate_proj", err)
	}
	up, err := projectMiniMaxM2PackedTensorMetal(expert.UpProj, hidden, inputShape)
	if err != nil {
		return nil, core.E("minimax_m2.packed_expert", "up_proj", err)
	}
	if len(gate.Values) != len(up.Values) {
		return nil, core.NewError(core.Sprintf("mlx: MiniMax M2 packed expert gate/up size mismatch %d != %d", len(gate.Values), len(up.Values)))
	}
	activated := make([]float32, len(gate.Values))
	for i := range activated {
		activated[i] = miniMaxM2SwiGLU(gate.Values[i], up.Values[i])
	}
	downShape := []int32{1, int32(len(activated))}
	down, err := projectMiniMaxM2PackedTensorMetal(expert.DownProj, activated, downShape)
	if err != nil {
		return nil, core.E("minimax_m2.packed_expert", "down_proj", err)
	}
	return down.Values, nil
}

func projectMiniMaxM2PackedTensorMetal(tensor JANGPackedProjectionTensor, input []float32, inputShape []int32) (JANGPackedProjectionResult, error) {
	return ProjectJANGPackedTensorMetalFused(tensor.Descriptor, tensor.Packed, tensor.Scales, tensor.Biases, input, inputShape, tensor.Bias)
}

func miniMaxM2SwiGLU(gate, up float32) float32 {
	return float32(float64(gate)/(1+math.Exp(float64(-gate)))) * up
}
