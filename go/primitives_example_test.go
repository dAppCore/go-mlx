// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	core "dappco.re/go"
	"dappco.re/go/inference"
)

func ExampleValueAndGrad() {
	grad := ValueAndGrad(func(inputs []*Array) []*Array {
		return []*Array{inputs[0]}
	}, 0)
	defer grad.Free()

	core.Println(grad != nil)
	// Output: true
}

func ExampleNewAdamW() {
	optimizer := NewAdamW(1e-4)

	core.Println(optimizer.LR, optimizer.Beta1, optimizer.Beta2, optimizer.PackedState)
	// Output: 0.0001 0.9 0.999 true
}

func ExampleCrossEntropyLoss() {
	logits := FromValues([]float32{0, 2}, 1, 1, 2)
	targets := FromValues([]int32{1}, 1, 1)
	defer Free(logits, targets)

	loss := CrossEntropyLoss(logits, targets)
	defer Free(loss)
	Materialize(loss)

	core.Println(loss.Valid(), loss.NumDims(), loss.Size())
	// Output: true 0 1
}

func ExampleMaskedCrossEntropyLoss() {
	logits := FromValues([]float32{0, 2, 3, 1}, 1, 2, 2)
	targets := FromValues([]int32{1, 0}, 1, 2)
	mask := FromValues([]float32{1, 0}, 1, 2)
	defer Free(logits, targets, mask)

	loss := MaskedCrossEntropyLoss(logits, targets, mask)
	defer Free(loss)
	Materialize(loss)

	core.Println(loss.Valid(), loss.NumDims(), loss.Size())
	// Output: true 0 1
}

func ExampleCheckpoint() {
	checkpointed := Checkpoint(func(inputs []*Array) []*Array {
		return inputs
	})

	core.Println(checkpointed != nil)
	// Output: true
}

func ExampleFromValues() {
	tokens := FromValues([]int32{1, 2, 3}, 1, 3)
	defer Free(tokens)
	Materialize(tokens)

	core.Println(tokens.Shape(), tokens.Ints())
	// Output: [1 3] [1 2 3]
}

func ExampleMaterialize() {
	values := FromValues([]float32{1, 2}, 2)
	defer Free(values)

	Materialize(values)

	core.Println(values.Floats())
	// Output: [1 2]
}

func ExampleFree() {
	values := FromValues([]float32{1, 2, 3, 4}, 2, 2)
	bytes := values.NumBytes()

	Free(values)

	core.Println(bytes)
	// Output: 16
}

func ExampleZeros() {
	values := Zeros([]int32{1, 3}, DTypeFloat32)
	defer Free(values)
	Materialize(values)

	core.Println(values.Shape(), values.Floats())
	// Output: [1 3] [0 0 0]
}

func Example_trainingAdapterApplyLoRA() {
	result := inference.LoadTrainable("/models/gemma4")
	if !result.OK {
		return
	}
	trainable := result.Value.(inference.TrainableModel)
	defer trainable.Close()

	adapter := trainable.ApplyLoRA(inference.LoRAConfig{
		Rank:       8,
		Alpha:      16,
		TargetKeys: []string{"q_proj", "v_proj", "o_proj"},
		BFloat16:   true,
	})
	_ = adapter
}

func Example_trainingAdapterEncode() {
	result := inference.LoadTrainable("/models/gemma4")
	if !result.OK {
		return
	}
	trainable := result.Value.(inference.TrainableModel)
	defer trainable.Close()

	ids := trainable.Encode("adapter training sample")
	_ = ids
}

func Example_trainingAdapterDecode() {
	result := inference.LoadTrainable("/models/gemma4")
	if !result.OK {
		return
	}
	trainable := result.Value.(inference.TrainableModel)
	defer trainable.Close()

	text := trainable.Decode([]int32{0})
	_ = text
}

func Example_trainingAdapterNumLayers() {
	result := inference.LoadTrainable("/models/gemma4")
	if !result.OK {
		return
	}
	trainable := result.Value.(inference.TrainableModel)
	defer trainable.Close()

	layers := trainable.NumLayers()
	_ = layers
}

func Example_trainingAdapterInternalModel() {
	result := inference.LoadTrainable("/models/gemma4")
	if !result.OK {
		return
	}
	trainable := result.Value.(inference.TrainableModel)
	defer trainable.Close()

	internal := TrainingModel(trainable)
	_ = internal
}

func ExampleConcreteAdapter() {
	result := inference.LoadTrainable("/models/gemma4")
	if !result.OK {
		return
	}
	trainable := result.Value.(inference.TrainableModel)
	defer trainable.Close()

	adapter := trainable.ApplyLoRA(inference.LoRAConfig{Rank: 8, Alpha: 16})
	concrete := ConcreteAdapter(adapter)
	_ = concrete.SortedNames()
}

func ExampleTrainingModel() {
	result := inference.LoadTrainable("/models/gemma4")
	if !result.OK {
		return
	}
	trainable := result.Value.(inference.TrainableModel)
	defer trainable.Close()

	internal := TrainingModel(trainable)
	_ = internal.NumLayers()
}
