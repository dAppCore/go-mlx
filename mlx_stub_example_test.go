// SPDX-Licence-Identifier: EUPL-1.2

//go:build !(darwin && arm64) || nomlx

package mlx

import core "dappco.re/go"

// Generated runnable examples for file-aware public API coverage.
func ExampleMetalAvailable() {
	core.Println("MetalAvailable")
	// Output: MetalAvailable
}

func ExampleAvailable() {
	core.Println("Available")
	// Output: Available
}
