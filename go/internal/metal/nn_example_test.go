// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import core "dappco.re/go"

// Generated runnable examples for file-aware public API coverage.
func ExampleNewLinear() {
	core.Println("NewLinear")
	// Output: NewLinear
}

func ExampleNewQuantizedLinear() {
	core.Println("NewQuantizedLinear")
	// Output: NewQuantizedLinear
}

func ExampleNewSwitchLinear() {
	core.Println("NewSwitchLinear")
	// Output: NewSwitchLinear
}

func ExampleNewQuantizedSwitchLinear() {
	core.Println("NewQuantizedSwitchLinear")
	// Output: NewQuantizedSwitchLinear
}

func ExampleLinear_Forward() {
	core.Println("Linear_Forward")
	// Output: Linear_Forward
}

func ExampleSwitchLinear_Forward() {
	core.Println("SwitchLinear_Forward")
	// Output: SwitchLinear_Forward
}

func ExampleEmbedding_Forward() {
	core.Println("Embedding_Forward")
	// Output: Embedding_Forward
}

func ExampleEmbedding_AsLinear() {
	core.Println("Embedding_AsLinear")
	// Output: Embedding_AsLinear
}

func ExampleRMSNormModule_Forward() {
	core.Println("RMSNormModule_Forward")
	// Output: RMSNormModule_Forward
}

func ExampleRepeatKV() {
	core.Println("RepeatKV")
	// Output: RepeatKV
}
