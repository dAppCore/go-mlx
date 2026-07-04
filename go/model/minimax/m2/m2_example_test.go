// SPDX-Licence-Identifier: EUPL-1.2

package m2

import (
	"fmt"
)

// ExampleResolvedTensor_EstimatedBytes shows a dense tensor's byte estimate
// from its dtype width and shape.
func ExampleResolvedTensor_EstimatedBytes() {
	tensor := ResolvedTensor{DType: "f32", Shape: []uint64{2, 3}}
	fmt.Println(tensor.EstimatedBytes())
	// Output: 24
}

// ExampleResolvedTensor_EstimatedBytes_packed shows that a packed tensor
// reports its packed byte count and ignores the dtype/shape estimate.
func ExampleResolvedTensor_EstimatedBytes_packed() {
	tensor := ResolvedTensor{DType: "f32", Shape: []uint64{2, 3}, PackedBytes: 7}
	fmt.Println(tensor.EstimatedBytes())
	// Output: 7
}

// ExampleLayerForwardSkeleton_EstimatedBytes sums the router gate, attention
// projections, and optional router bias bytes a first forward pass needs.
func ExampleLayerForwardSkeleton_EstimatedBytes() {
	bias := ResolvedTensor{DType: "f32", Shape: []uint64{3}}
	skeleton := LayerForwardSkeleton{
		RouterGate: ResolvedTensor{DType: "f32", Shape: []uint64{3, 4}},
		Attention: []ResolvedTensor{
			{PackedBytes: 16},
			{PackedBytes: 8},
		},
		RouterBias: &bias,
	}
	fmt.Println(skeleton.EstimatedBytes())
	// Output: 84
}

// ExampleParseConfig parses the MiniMax M2 config subset and shows the
// defaulted scoring function.
func ExampleParseConfig() {
	cfg, err := ParseConfig([]byte(`{
		"model_type": "minimax_m2",
		"hidden_size": 3072,
		"num_local_experts": 256,
		"num_experts_per_tok": 8
	}`))
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(cfg.ModelType, cfg.HiddenSize, cfg.NumLocalExperts, cfg.ScoringFunc)
	// Output: minimax_m2 3072 256 sigmoid
}

// ExampleBuildTensorPlan builds a model-wide MiniMax M2 tensor plan from a
// config plus a nil JANG info (the builder synthesises a default JANGTQ packed
// profile). The plan carries the routed-expert bit-width through.
func ExampleBuildTensorPlan() {
	plan, err := BuildTensorPlan(Config{
		ModelType:          "minimax_m2",
		HiddenSize:         3072,
		IntermediateSize:   1536,
		NumHiddenLayers:    62,
		NumLocalExperts:    256,
		NumExpertsPerToken: 8,
	}, nil)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(plan.Quantization.Format)
	fmt.Println(plan.Quantization.RoleBits["routed_expert"])
	// Output:
	// mxtq
	// 2
}

// ExampleTensorPlan_LayerTensorSpecs lists the canonical tensor names a
// MiniMax M2 layer/expert pass expects. The path is the documentation: each
// spec's Name is the checkpoint key the loader resolves. With UseRoutingBias
// the plan carries 9 specs (4 attention + router gate + 3 expert projections
// + router bias).
func ExampleTensorPlan_LayerTensorSpecs() {
	plan, err := BuildTensorPlan(Config{
		ModelType:          "minimax_m2",
		HiddenSize:         4,
		IntermediateSize:   8,
		NumHiddenLayers:    1,
		NumAttentionHeads:  2,
		NumKeyValueHeads:   1,
		HeadDim:            2,
		NumLocalExperts:    16,
		NumExpertsPerToken: 2,
		UseRoutingBias:     true,
	}, nil)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	specs, err := plan.LayerTensorSpecs(0, 0)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(len(specs))
	fmt.Println(specs[4].Name) // the router gate
	fmt.Println(specs[5].Name) // expert 0 gate_proj
	// Output:
	// 9
	// model.layers.0.block_sparse_moe.gate.weight
	// model.layers.0.block_sparse_moe.experts.0.gate_proj.weight
}

// ExampleTensorPlan_ValidateTensorNames shows the loader pre-flight: given the
// set of tensor names present in a checkpoint, ValidateTensorNames reports the
// first missing required tensor by name rather than failing mid-load.
func ExampleTensorPlan_ValidateTensorNames() {
	plan, err := BuildTensorPlan(Config{
		ModelType:          "minimax_m2",
		HiddenSize:         4,
		IntermediateSize:   8,
		NumHiddenLayers:    1,
		NumAttentionHeads:  2,
		NumKeyValueHeads:   1,
		HeadDim:            2,
		NumLocalExperts:    16,
		NumExpertsPerToken: 2,
	}, nil)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	// A checkpoint that has the attention + router tensors but is missing the
	// expert projections fails the pre-flight with a named diagnostic.
	err = plan.ValidateTensorNames(map[string]bool{
		"model.layers.0.self_attn.q_proj.weight":      true,
		"model.layers.0.self_attn.k_proj.weight":      true,
		"model.layers.0.self_attn.v_proj.weight":      true,
		"model.layers.0.self_attn.o_proj.weight":      true,
		"model.layers.0.block_sparse_moe.gate.weight": true,
	})
	fmt.Println(err)
	// Output:
	// mlx: MiniMax M2 tensor plan missing required tensors: model.layers.0.block_sparse_moe.experts.0.gate_proj.weight, model.layers.0.block_sparse_moe.experts.0.up_proj.weight, model.layers.0.block_sparse_moe.experts.0.down_proj.weight
}
