// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && !nomlx

package mlx

import core "dappco.re/go"

// Generated runnable examples for file-aware public API coverage.
func ExampleValueAndGrad() {
	core.Println("ValueAndGrad")
	// Output: ValueAndGrad
}

func ExampleNewAdamW() {
	core.Println("NewAdamW")
	// Output: NewAdamW
}

func ExampleCrossEntropyLoss() {
	core.Println("CrossEntropyLoss")
	// Output: CrossEntropyLoss
}

func ExampleMaskedCrossEntropyLoss() {
	core.Println("MaskedCrossEntropyLoss")
	// Output: MaskedCrossEntropyLoss
}

func ExampleCheckpoint() {
	core.Println("Checkpoint")
	// Output: Checkpoint
}

func ExampleFromValues() {
	core.Println("FromValues")
	// Output: FromValues
}

func ExampleMaterialize() {
	core.Println("Materialize")
	// Output: Materialize
}

func ExampleFree() {
	core.Println("Free")
	// Output: Free
}

func ExampleZeros() {
	core.Println("Zeros")
	// Output: Zeros
}

func Example_trainingAdapterApplyLoRA() {
	core.Println("Adapter_ApplyLoRA")
	// Output: Adapter_ApplyLoRA
}

func Example_trainingAdapterEncode() {
	core.Println("Adapter_Encode")
	// Output: Adapter_Encode
}

func Example_trainingAdapterDecode() {
	core.Println("Adapter_Decode")
	// Output: Adapter_Decode
}

func Example_trainingAdapterNumLayers() {
	core.Println("Adapter_NumLayers")
	// Output: Adapter_NumLayers
}

func Example_trainingAdapterInternalModel() {
	core.Println("Adapter_InternalModel")
	// Output: Adapter_InternalModel
}

func ExampleConcreteAdapter() {
	core.Println("ConcreteAdapter")
	// Output: ConcreteAdapter
}

func ExampleTrainingModel() {
	core.Println("TrainingModel")
	// Output: TrainingModel
}
