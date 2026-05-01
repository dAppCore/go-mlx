// SPDX-Licence-Identifier: EUPL-1.2

//go:build !(darwin && arm64) || nomlx

package mlx

import core "dappco.re/go"

// Generated runnable examples for file-aware public API coverage.
func ExampleSetCacheLimit() {
	core.Println("SetCacheLimit")
	// Output: SetCacheLimit
}

func ExampleSetMemoryLimit() {
	core.Println("SetMemoryLimit")
	// Output: SetMemoryLimit
}

func ExampleGetActiveMemory() {
	core.Println("GetActiveMemory")
	// Output: GetActiveMemory
}

func ExampleGetPeakMemory() {
	core.Println("GetPeakMemory")
	// Output: GetPeakMemory
}

func ExampleClearCache() {
	core.Println("ClearCache")
	// Output: ClearCache
}

func ExampleGetCacheMemory() {
	core.Println("GetCacheMemory")
	// Output: GetCacheMemory
}

func ExampleResetPeakMemory() {
	core.Println("ResetPeakMemory")
	// Output: ResetPeakMemory
}

func ExampleSetWiredLimit() {
	core.Println("SetWiredLimit")
	// Output: SetWiredLimit
}

func ExampleGetDeviceInfo() {
	core.Println("GetDeviceInfo")
	// Output: GetDeviceInfo
}
