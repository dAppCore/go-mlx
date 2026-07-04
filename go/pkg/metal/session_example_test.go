// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import core "dappco.re/go"

func ExampleSessionHandle() {
	core.Println("SessionHandle")
	// Output: SessionHandle
}

func ExampleModelSession() {
	core.Println("ModelSession")
	// Output: ModelSession
}

func ExampleModel_NewSession() {
	core.Println("Model_NewSession")
	// Output: Model_NewSession
}

func ExampleModelSession_Prefill() {
	core.Println("ModelSession_Prefill")
	// Output: ModelSession_Prefill
}

func ExampleModelSession_AppendPrompt() {
	core.Println("ModelSession_AppendPrompt")
	// Output: ModelSession_AppendPrompt
}

func ExampleModelSession_Generate() {
	core.Println("ModelSession_Generate")
	// Output: ModelSession_Generate
}

func ExampleModelSession_CaptureKV() {
	core.Println("ModelSession_CaptureKV")
	// Output: ModelSession_CaptureKV
}

func ExampleModelSession_Fork() {
	core.Println("ModelSession_Fork")
	// Output: ModelSession_Fork
}

func ExampleModelSession_Reset() {
	core.Println("ModelSession_Reset")
	// Output: ModelSession_Reset
}

func ExampleModelSession_Close() {
	core.Println("ModelSession_Close")
	// Output: ModelSession_Close
}

func ExampleModelSession_Err() {
	core.Println("ModelSession_Err")
	// Output: ModelSession_Err
}
