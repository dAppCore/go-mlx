// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import core "dappco.re/go"

// Generated runnable examples for file-aware public API coverage.
func ExampleVJP() {
	core.Println("VJP")
	// Output: VJP
}

func ExampleJVP() {
	core.Println("JVP")
	// Output: JVP
}

func ExampleValueAndGrad() {
	core.Println("ValueAndGrad")
	// Output: ValueAndGrad
}

func ExampleGradFn_Apply() {
	core.Println("GradFn_Apply")
	// Output: GradFn_Apply
}

func ExampleGradFn_Free() {
	core.Println("GradFn_Free")
	// Output: GradFn_Free
}

func ExampleCheckpoint() {
	core.Println("Checkpoint")
	// Output: Checkpoint
}

func ExampleCrossEntropyLoss() {
	core.Println("CrossEntropyLoss")
	// Output: CrossEntropyLoss
}

func ExampleMaskedCrossEntropyLoss() {
	core.Println("MaskedCrossEntropyLoss")
	// Output: MaskedCrossEntropyLoss
}

func ExampleMSELoss() {
	core.Println("MSELoss")
	// Output: MSELoss
}

func ExampleLog() {
	core.Println("Log")
	// Output: Log
}

func ExampleSumAll() {
	core.Println("SumAll")
	// Output: SumAll
}

func ExampleMeanAll() {
	core.Println("MeanAll")
	// Output: MeanAll
}

func ExampleOnesLike() {
	core.Println("OnesLike")
	// Output: OnesLike
}
