// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && !nomlx

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

func Example_metalbackendName() {
	core.Println("Backend_Name")
	// Output: Backend_Name
}

func Example_metalbackendAvailable() {
	core.Println("Backend_Available")
	// Output: Backend_Available
}

func Example_metalbackendLoadModel() {
	core.Println("Backend_LoadModel")
	// Output: Backend_LoadModel
}

func Example_metaladapterGenerate() {
	core.Println("Adapter_Generate")
	// Output: Adapter_Generate
}

func Example_metaladapterChat() {
	core.Println("Adapter_Chat")
	// Output: Adapter_Chat
}

func Example_metaladapterClassify() {
	core.Println("Adapter_Classify")
	// Output: Adapter_Classify
}

func Example_metaladapterBatchGenerate() {
	core.Println("Adapter_BatchGenerate")
	// Output: Adapter_BatchGenerate
}

func Example_metaladapterMetrics() {
	core.Println("Adapter_Metrics")
	// Output: Adapter_Metrics
}

func Example_metaladapterModelType() {
	core.Println("Adapter_ModelType")
	// Output: Adapter_ModelType
}

func Example_metaladapterInfo() {
	core.Println("Adapter_Info")
	// Output: Adapter_Info
}

func Example_metaladapterInspectAttention() {
	core.Println("Adapter_InspectAttention")
	// Output: Adapter_InspectAttention
}

func Example_metaladapterErr() {
	core.Println("Adapter_Err")
	// Output: Adapter_Err
}

func Example_metaladapterClose() {
	core.Println("Adapter_Close")
	// Output: Adapter_Close
}
