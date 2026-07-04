// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import core "dappco.re/go"

func ExampleNewVectorArray() {
	q := FromValues([]float32{1, 2}, 2)
	v := FromValues([]float32{3, 4}, 2)
	defer Free(q, v)

	arrays := NewVectorArray()
	defer arrays.Free()
	arrays.Append(q)
	arrays.Append(v)

	core.Println(arrays.Size())
	// Output: 2
}

func ExampleNewVectorArrayFromValue() {
	weights := FromValues([]float32{0.5, 1.5}, 2)
	defer Free(weights)

	arrays := NewVectorArrayFromValue(weights)
	defer arrays.Free()
	got := arrays.Get(0)
	defer Free(got)
	Materialize(got)

	core.Println(arrays.Size(), got.Floats())
	// Output: 1 [0.5 1.5]
}

func ExampleVectorArray_SetValue() {
	oldWeights := FromValue(float32(1))
	newWeights := FromValues([]float32{2, 3}, 2)
	defer Free(oldWeights, newWeights)

	arrays := NewVectorArrayFromValue(oldWeights)
	defer arrays.Free()
	arrays.SetValue(newWeights)
	got := arrays.Get(0)
	defer Free(got)
	Materialize(got)

	core.Println(arrays.Size(), got.Floats())
	// Output: 1 [2 3]
}

func ExampleVectorArray_Append() {
	a := FromValue(float32(4))
	b := FromValue(float32(8))
	defer Free(a, b)

	arrays := NewVectorArray()
	defer arrays.Free()
	arrays.Append(a)
	arrays.Append(b)

	core.Println(arrays.Size())
	// Output: 2
}

func ExampleVectorArray_Size() {
	activations := FromValues([]float32{1, 2, 3}, 3)
	defer Free(activations)

	arrays := NewVectorArrayFromValue(activations)
	defer arrays.Free()

	core.Println(arrays.Size())
	// Output: 1
}

func ExampleVectorArray_Get() {
	gradients := FromValues([]float32{0.25, 0.75}, 2)
	defer Free(gradients)

	arrays := NewVectorArrayFromValue(gradients)
	defer arrays.Free()
	got := arrays.Get(0)
	defer Free(got)
	Materialize(got)

	core.Println(got.Shape(), got.Floats())
	// Output: [2] [0.25 0.75]
}

func ExampleVectorArray_Free() {
	value := FromValue(float32(1))
	defer Free(value)

	arrays := NewVectorArrayFromValue(value)
	core.Println(arrays.Size())
	arrays.Free()
	core.Println(arrays.ctx.ctx == nil)
	// Output:
	// 1
	// true
}

func ExampleNewVectorString() {
	names := NewVectorString()
	defer names.Free()
	names.Append("q_proj")
	names.Append("v_proj")

	core.Println(names.Size(), names.Get(0), names.Get(1))
	// Output: 2 q_proj v_proj
}

func ExampleNewVectorStringFromValue() {
	names := NewVectorStringFromValue("adapter.alpha")
	defer names.Free()

	core.Println(names.Size(), names.Get(0))
	// Output: 1 adapter.alpha
}

func ExampleNewVectorStringFromSlice() {
	names := NewVectorStringFromSlice([]string{"q_proj", "v_proj", "o_proj"})
	defer names.Free()

	core.Println(names.Size(), names.Get(2))
	// Output: 3 o_proj
}

func ExampleVectorString_Append() {
	names := NewVectorString()
	defer names.Free()
	names.Append("lora_a")
	names.Append("lora_b")

	core.Println(names.Size(), names.Get(0), names.Get(1))
	// Output: 2 lora_a lora_b
}

func ExampleVectorString_Size() {
	names := NewVectorStringFromSlice([]string{"mlp.gate_proj", "mlp.up_proj"})
	defer names.Free()

	core.Println(names.Size())
	// Output: 2
}

func ExampleVectorString_Get() {
	names := NewVectorStringFromSlice([]string{
		"model.layers.0.self_attn.q_proj",
		"model.layers.0.self_attn.v_proj",
	})
	defer names.Free()

	core.Println(names.Get(0))
	// Output: model.layers.0.self_attn.q_proj
}

func ExampleVectorString_Free() {
	names := NewVectorStringFromValue("model.layers.0.mlp.down_proj")
	core.Println(names.Size())
	names.Free()
	core.Println(names.ctx.ctx == nil)
	// Output:
	// 1
	// true
}
