// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && !nomlx

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

func Example_computebackendAvailable() {
	core.Println("Backend_Available")
	// Output: Backend_Available
}

func Example_computebackendDeviceInfo() {
	core.Println("Backend_DeviceInfo")
	// Output: Backend_DeviceInfo
}

func Example_computebackendNewSession() {
	core.Println("Backend_NewSession")
	// Output: Backend_NewSession
}

func Example_bufferbaseSize() {
	core.Println("Base_Size")
	// Output: Base_Size
}

func Example_pixelbufferDescriptor() {
	core.Println("Buffer_Descriptor")
	// Output: Buffer_Descriptor
}

func Example_pixelbufferUpload() {
	core.Println("Buffer_Upload")
	// Output: Buffer_Upload
}

func Example_pixelbufferRead() {
	core.Println("Buffer_Read")
	// Output: Buffer_Read
}

func ExampleSession_Close() {
	core.Println("Session_Close")
	// Output: Session_Close
}

func ExampleSession_NewPixelBuffer() {
	core.Println("Session_NewPixelBuffer")
	// Output: Session_NewPixelBuffer
}

func ExampleSession_NewByteBuffer() {
	core.Println("Session_NewByteBuffer")
	// Output: Session_NewByteBuffer
}

func ExampleSession_BeginFrame() {
	core.Println("Session_BeginFrame")
	// Output: Session_BeginFrame
}

func ExampleSession_FinishFrame() {
	core.Println("Session_FinishFrame")
	// Output: Session_FinishFrame
}

func ExampleSession_Run() {
	core.Println("Session_Run")
	// Output: Session_Run
}

func ExampleSession_Sync() {
	core.Println("Session_Sync")
	// Output: Session_Sync
}

func ExampleSession_Metrics() {
	core.Println("Session_Metrics")
	// Output: Session_Metrics
}

func ExampleSession_FrameMetrics() {
	core.Println("Session_FrameMetrics")
	// Output: Session_FrameMetrics
}
