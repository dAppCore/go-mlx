// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import core "dappco.re/go"

func ExampleNewLinear() {
	weight := FromValues([]float32{1, 0, 0, 1}, 2, 2)
	bias := FromValues([]float32{10, 20}, 2)
	defer Free(weight, bias)

	layer := NewLinear(weight, bias)
	core.Println(layer.Weight == weight, layer.Bias == bias, layer.LoRA == nil)
	// Output: true true true
}

func ExampleNewQuantizedLinear() {
	weight := FromValues([]uint32{0, 1}, 1, 2)
	scales := FromValues([]float32{0.5}, 1)
	biases := FromValues([]float32{0}, 1)
	defer Free(weight, scales, biases)

	layer := NewQuantizedLinear(weight, scales, biases, nil, 64, 4)
	core.Println(layer.Weight == weight, layer.Scales == scales, layer.GroupSize, layer.Bits, layer.QuantizationMode)
	// Output: true true 64 4 affine
}

func ExampleNewSwitchLinear() {
	weight := FromValues([]float32{1, 0, 0, 1}, 1, 2, 2)
	defer Free(weight)

	layer := NewSwitchLinear(weight, nil)
	defer Free(layer.WeightT)

	core.Println(layer.Weight == weight, layer.Bias == nil, layer.WeightT.Shape())
	// Output: true true [1 2 2]
}

func ExampleNewQuantizedSwitchLinear() {
	weight := FromValues([]uint32{0, 1}, 1, 1, 2)
	scales := FromValues([]float32{0.5}, 1, 1)
	biases := FromValues([]float32{0}, 1, 1)
	defer Free(weight, scales, biases)

	layer := NewQuantizedSwitchLinear(weight, scales, biases, nil, 64, 4)
	core.Println(layer.Weight == weight, layer.Scales == scales, layer.GroupSize, layer.Bits, layer.QuantizationMode)
	// Output: true true 64 4 affine
}

func ExampleLinear_Forward() {
	input := FromValues([]float32{1, 2, 3}, 1, 3)
	weight := FromValues([]float32{1, 0, 0, 0, 1, 0}, 2, 3)
	bias := FromValues([]float32{10, 20}, 2)
	defer Free(input, weight, bias)

	out := NewLinear(weight, bias).Forward(input)
	defer Free(out)
	Materialize(out)

	core.Println(out.Shape(), out.Floats())
	// Output: [1 2] [11 22]
}

func ExampleSwitchLinear_Forward() {
	input := FromValues([]float32{1, 2}, 1, 1, 2)
	weight := FromValues([]float32{
		1, 0,
		0, 1,
	}, 1, 2, 2)
	expert := FromValues([]int32{0}, 1, 1)
	defer Free(input, weight, expert)

	layer := NewSwitchLinear(weight, nil)
	defer Free(layer.WeightT)
	out := layer.Forward(input, expert)
	defer Free(out)
	Materialize(out)

	core.Println(out.Shape(), out.Floats())
	// Output: [1 1 1 2] [1 2]
}

func ExampleEmbedding_Forward() {
	weight := FromValues([]float32{
		0, 0,
		1, 1,
		2, 2,
	}, 3, 2)
	tokens := FromValues([]int32{2, 1}, 2)
	defer Free(weight, tokens)

	out := (&Embedding{Weight: weight}).Forward(tokens)
	defer Free(out)
	Materialize(out)

	core.Println(out.Shape(), out.Floats())
	// Output: [2 2] [2 2 1 1]
}

func ExampleEmbedding_AsLinear() {
	weight := FromValues([]float32{1, 2, 3, 4}, 2, 2)
	defer Free(weight)

	layer := (&Embedding{Weight: weight}).AsLinear()
	core.Println(layer.Weight == weight, layer.Bias == nil)
	// Output: true true
}

func ExampleRMSNormModule_Forward() {
	input := FromValues([]float32{3, 4}, 1, 2)
	weight := FromValues([]float32{1, 1}, 2)
	defer Free(input, weight)

	out := (&RMSNormModule{Weight: weight}).Forward(input, 1e-6)
	defer Free(out)
	Materialize(out)

	core.Println(out.Shape(), core.Sprintf("%.2f %.2f", out.Floats()[0], out.Floats()[1]))
	// Output: [1 2] 0.85 1.13
}

func ExampleRepeatKV() {
	input := FromValues([]float32{1, 2, 3, 4}, 1, 2, 1, 2)
	defer Free(input)

	out := RepeatKV(input, 2)
	defer Free(out)
	Materialize(out)

	core.Println(out.Shape(), out.Floats())
	// Output: [1 4 1 2] [1 2 1 2 3 4 3 4]
}
