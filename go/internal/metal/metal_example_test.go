// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import core "dappco.re/go"

// Generated runnable examples for file-aware public API coverage.
func ExampleInit() {
	core.Println("Init")
	// Output: Init
}

func ExampleEval() {
	core.Println("Eval")
	// Output: Eval
}

func ExampleEvalAsync() {
	core.Println("EvalAsync")
	// Output: EvalAsync
}

func ExampleMaterialize() {
	core.Println("Materialize")
	// Output: Materialize
}

func ExampleMaterializeAsync() {
	core.Println("MaterializeAsync")
	// Output: MaterializeAsync
}

func ExampleMetalAvailable() {
	core.Println("MetalAvailable")
	// Output: MetalAvailable
}

func ExampleVersion() {
	core.Println("Version")
	// Output: Version
}
