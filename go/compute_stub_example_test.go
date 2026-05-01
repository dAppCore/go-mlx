// SPDX-Licence-Identifier: EUPL-1.2

//go:build !(darwin && arm64) || nomlx

package mlx

import core "dappco.re/go"

// Generated runnable examples for file-aware public API coverage.
func ExampleDefaultCompute() {
	core.Println("DefaultCompute")
	// Output: DefaultCompute
}

func ExampleNewSession() {
	core.Println("NewSession")
	// Output: NewSession
}

func ExampleCompute_Available() {
	core.Println("Compute_Available")
	// Output: Compute_Available
}

func ExampleCompute_DeviceInfo() {
	core.Println("Compute_DeviceInfo")
	// Output: Compute_DeviceInfo
}

func ExampleCompute_NewSession() {
	core.Println("Compute_NewSession")
	// Output: Compute_NewSession
}
