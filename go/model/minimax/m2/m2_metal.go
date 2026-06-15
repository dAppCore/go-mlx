// SPDX-Licence-Identifier: EUPL-1.2

package m2

import (
	"math"

	core "dappco.re/go"
	mlxjang "dappco.re/go/mlx/quant/jang"
)

// DispatchPackedExpertsMetal applies router-selected MiniMax M2
// packed experts using fused JANG/JANGTQ projection kernels for gate, up, and
// down projections. It is intentionally host-shaped for bring-up fixtures and
// model-loader validation; full model execution keeps tensors on device.
func DispatchPackedExpertsMetal(hidden [][]float32, decisions []RouterDecision, experts map[int]PackedExpertWeights) ([][]float32, error) {
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
			result, err := runPackedExpertMetal(hidden[decision.TokenIndex], expert)
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

// DispatchPackedExpertsFromSafetensorsMetal loads the router-selected
// packed experts from safetensors shards and executes the fused Metal dispatch.
func DispatchPackedExpertsFromSafetensorsMetal(plan TensorPlan, weightFiles []string, layer int, hidden [][]float32, decisions []RouterDecision) ([][]float32, error) {
	experts, err := LoadPackedExpertsForDecisions(plan, weightFiles, layer, decisions)
	if err != nil {
		return nil, err
	}
	return DispatchPackedExpertsMetal(hidden, decisions, experts)
}

// ForwardLazyExpertLoadMetal executes an already-routed lazy expert
// load with the native packed projection kernels.
func ForwardLazyExpertLoadMetal(hidden [][]float32, load LazyExpertLoad) (PackedLayerForwardResult, error) {
	output, err := DispatchPackedExpertsMetal(hidden, load.Decisions, load.Experts)
	if err != nil {
		return PackedLayerForwardResult{}, err
	}
	return PackedLayerForwardResult{
		Output:            output,
		Decisions:         core.SliceClone(load.Decisions),
		SelectedExpertIDs: core.SliceClone(load.SelectedExpertIDs),
		LoadedPackedBytes: load.LoadedPackedBytes,
		ProbeEvents:       core.SliceClone(load.ProbeEvents),
	}, nil
}

// ForwardPackedLayerMetal routes hidden states through a MiniMax M2
// packed MoE layer skeleton, lazily resolving selected experts from safetensors
// and emitting router probe events.
func ForwardPackedLayerMetal(opts PackedLayerForwardOptions) (PackedLayerForwardResult, error) {
	if len(opts.Hidden) != len(opts.RouterScores) {
		return PackedLayerForwardResult{}, core.NewError(core.Sprintf("mlx: MiniMax M2 packed layer hidden rows %d, router rows %d", len(opts.Hidden), len(opts.RouterScores)))
	}
	decisions, err := RouteTokens(opts.Plan.Config, opts.RouterScores, opts.RouterBias)
	if err != nil {
		return PackedLayerForwardResult{}, err
	}
	experts, err := LoadPackedExpertsForDecisions(opts.Plan, opts.WeightFiles, opts.Layer, decisions)
	if err != nil {
		return PackedLayerForwardResult{}, err
	}
	output, err := DispatchPackedExpertsMetal(opts.Hidden, decisions, experts)
	if err != nil {
		return PackedLayerForwardResult{}, err
	}
	events := RouterProbeEvents(opts.Layer, opts.TokenIDs, decisions)
	for _, event := range events {
		if opts.ProbeSink != nil {
			opts.ProbeSink.EmitProbe(event)
		}
	}
	return PackedLayerForwardResult{
		Output:            output,
		Decisions:         decisions,
		SelectedExpertIDs: decisionExpertIDsSorted(decisions),
		LoadedPackedBytes: packedExpertLoadedBytes(experts),
		ProbeEvents:       events,
	}, nil
}

// ForwardPackedLayerFromSafetensorsMetal reads the dense router gate,
// computes router scores, then runs the packed layer skeleton with lazy expert
// resolution.
func ForwardPackedLayerFromSafetensorsMetal(opts PackedLayerForwardOptions) (PackedLayerForwardResult, error) {
	if len(opts.RouterBias) == 0 {
		load, err := LoadLazyExpertsForHidden(opts.Plan, opts.WeightFiles, opts.Layer, opts.Hidden, opts.TokenIDs, opts.ProbeSink)
		if err != nil {
			return PackedLayerForwardResult{}, err
		}
		return ForwardLazyExpertLoadMetal(opts.Hidden, load)
	}
	router, err := LoadRouter(opts.Plan, opts.WeightFiles, opts.Layer)
	if err != nil {
		return PackedLayerForwardResult{}, err
	}
	scores, err := ProjectRouterScores(opts.Hidden, router)
	if err != nil {
		return PackedLayerForwardResult{}, err
	}
	opts.RouterScores = scores
	if len(opts.RouterBias) == 0 {
		opts.RouterBias = router.Bias
	}
	return ForwardPackedLayerMetal(opts)
}

func runPackedExpertMetal(hidden []float32, expert PackedExpertWeights) ([]float32, error) {
	inputShape := []int32{1, int32(len(hidden))}
	gate, err := projectPackedTensorMetal(expert.GateProj, hidden, inputShape)
	if err != nil {
		return nil, core.E("minimax_m2.packed_expert", "gate_proj", err)
	}
	up, err := projectPackedTensorMetal(expert.UpProj, hidden, inputShape)
	if err != nil {
		return nil, core.E("minimax_m2.packed_expert", "up_proj", err)
	}
	if len(gate.Values) != len(up.Values) {
		return nil, core.NewError(core.Sprintf("mlx: MiniMax M2 packed expert gate/up size mismatch %d != %d", len(gate.Values), len(up.Values)))
	}
	activated := make([]float32, len(gate.Values))
	for i := range activated {
		activated[i] = swiGLU(gate.Values[i], up.Values[i])
	}
	downShape := []int32{1, int32(len(activated))}
	down, err := projectPackedTensorMetal(expert.DownProj, activated, downShape)
	if err != nil {
		return nil, core.E("minimax_m2.packed_expert", "down_proj", err)
	}
	return down.Values, nil
}

func projectPackedTensorMetal(tensor JANGPackedProjectionTensor, input []float32, inputShape []int32) (mlxjang.PackedProjectionResult, error) {
	return mlxjang.ProjectPackedTensorFused(tensor.Descriptor, tensor.Packed, tensor.Scales, tensor.Biases, input, inputShape, tensor.Bias)
}

func swiGLU(gate, up float32) float32 {
	return float32(float64(gate)/(1+math.Exp(float64(-gate)))) * up
}
