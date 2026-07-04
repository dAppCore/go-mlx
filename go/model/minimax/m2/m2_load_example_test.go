// SPDX-Licence-Identifier: EUPL-1.2

package m2

import (
	"fmt"

	"dappco.re/go/inference/quant/jang"
)

// ExampleLoadPackedExpertsForDecisions shows the pre-IO guard: with no
// safetensors weight files the loader rejects before touching disk, so the
// usage shape is documented without a model fixture.
func ExampleLoadPackedExpertsForDecisions() {
	plan, err := BuildTensorPlan(Config{
		ModelType: "minimax_m2", HiddenSize: 2, IntermediateSize: 2,
		NumHiddenLayers: 1, NumAttentionHeads: 1, NumKeyValueHeads: 1,
		HeadDim: 2, NumLocalExperts: 3, NumExpertsPerToken: 1,
	}, nil)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	_, err = LoadPackedExpertsForDecisions(plan, nil, 0, []RouterDecision{
		{TokenIndex: 0, ExpertIDs: []int{1}, Weights: []float32{1}},
	})
	fmt.Println(err)
	// Output: mlx: MiniMax M2 packed expert loading requires safetensors weight files
}

// ExampleLoadLazyExpertsForHidden shows the router-load guard: an empty weight
// file set fails before routing, documenting the entry point device-free.
func ExampleLoadLazyExpertsForHidden() {
	plan, err := BuildTensorPlan(Config{
		ModelType: "minimax_m2", HiddenSize: 2, IntermediateSize: 2,
		NumHiddenLayers: 1, NumAttentionHeads: 1, NumKeyValueHeads: 1,
		HeadDim: 2, NumLocalExperts: 3, NumExpertsPerToken: 1,
	}, nil)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	_, err = LoadLazyExpertsForHidden(plan, nil, 0, [][]float32{{1, 0}}, nil, nil)
	fmt.Println(err)
	// Output: mlx: MiniMax M2 router loading requires safetensors weight files
}

// ExampleLoadPackedExperts shows the weight-file pre-flight that guards every
// packed expert read.
func ExampleLoadPackedExperts() {
	plan, err := BuildTensorPlan(Config{
		ModelType: "minimax_m2", HiddenSize: 2, IntermediateSize: 2,
		NumHiddenLayers: 1, NumAttentionHeads: 1, NumKeyValueHeads: 1,
		HeadDim: 2, NumLocalExperts: 3, NumExpertsPerToken: 1,
	}, nil)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	_, err = LoadPackedExperts(plan, nil, 0, []int{0})
	fmt.Println(err)
	// Output: mlx: MiniMax M2 packed expert loading requires safetensors weight files
}

// ExampleDequantizeJANGPackedProjection expands one affine-packed projection
// back to dense float weights. With scale 1 and bias 0 the dequantized weight
// equals the original quantised values. This path is pure host arithmetic — no
// safetensors file and no device.
func ExampleDequantizeJANGPackedProjection() {
	desc := jang.PackedTensorDescriptor{
		Name:          "model.layers.0.block_sparse_moe.experts.0.gate_proj.weight",
		Type:          "jangtq",
		Format:        "mxtq",
		Role:          jang.TensorRoleRoutedExpert,
		Shape:         []uint64{2, 2},
		Elements:      4,
		Bits:          2,
		GroupSize:     4,
		Groups:        1,
		PackedBytes:   1,
		ValuesPerByte: 4,
		ScaleCount:    1,
		BiasCount:     1,
		BitOrder:      jang.BitOrderLSB0,
		Encoding:      jang.EncodingAffine,
	}
	packed, err := jang.PackQuantizedValues(desc, []uint8{0, 1, 2, 3})
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	dense, err := DequantizeJANGPackedProjection(JANGPackedProjectionTensor{
		Descriptor: desc,
		Packed:     packed,
		Scales:     []float32{1},
		Biases:     []float32{0},
	})
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(dense.Weight)
	// Output: [0 1 2 3]
}

// ExampleLazyExpertLoad_DequantizedExperts shows that an empty lazy load
// dequantizes to an empty dense expert map — the device-free, fixture-free
// shape of the expansion step.
func ExampleLazyExpertLoad_DequantizedExperts() {
	load := LazyExpertLoad{}
	dense, err := load.DequantizedExperts()
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(len(dense))
	// Output: 0
}

// ExampleLoadRouter shows the dense-router weight-file guard.
func ExampleLoadRouter() {
	plan, err := BuildTensorPlan(Config{
		ModelType: "minimax_m2", HiddenSize: 2, IntermediateSize: 2,
		NumHiddenLayers: 1, NumAttentionHeads: 1, NumKeyValueHeads: 1,
		HeadDim: 2, NumLocalExperts: 3, NumExpertsPerToken: 1,
	}, nil)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	_, err = LoadRouter(plan, nil, 0)
	fmt.Println(err)
	// Output: mlx: MiniMax M2 router loading requires safetensors weight files
}

// ExampleBuildLayerForwardSkeleton shows the metadata-only skeleton resolver's
// weight-file pre-flight (it reads safetensors headers only, never payloads).
func ExampleBuildLayerForwardSkeleton() {
	plan, err := BuildTensorPlan(Config{
		ModelType: "minimax_m2", HiddenSize: 2, IntermediateSize: 2,
		NumHiddenLayers: 1, NumAttentionHeads: 1, NumKeyValueHeads: 1,
		HeadDim: 2, NumLocalExperts: 3, NumExpertsPerToken: 1,
	}, nil)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	_, err = BuildLayerForwardSkeleton(plan, nil, 0)
	fmt.Println(err)
	// Output: mlx: MiniMax M2 layer skeleton requires safetensors weight files
}
