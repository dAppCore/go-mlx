// SPDX-Licence-Identifier: EUPL-1.2

//go:build !(darwin && arm64) || nomlx

package mlx

import core "dappco.re/go"

// JANGPackedProjectionResult is unavailable on unsupported builds except for
// carrying the API shape.
type JANGPackedProjectionResult struct {
	Values []float32 `json:"values"`
	Shape  []int32   `json:"shape"`
}

// DequantizeJANGPackedTensorMetal requires the native Metal backend.
func DequantizeJANGPackedTensorMetal(_ JANGPackedTensorDescriptor, _ []byte, _, _ []float32) ([]float32, error) {
	return nil, core.NewError("mlx: JANG Metal dequant requires darwin/arm64 native MLX support")
}

// ProjectJANGPackedTensorMetal requires the native Metal backend.
func ProjectJANGPackedTensorMetal(_ JANGPackedTensorDescriptor, _ []byte, _, _, _ []float32, _ []int32, _ []float32) (JANGPackedProjectionResult, error) {
	return JANGPackedProjectionResult{}, core.NewError("mlx: JANG Metal packed projection requires darwin/arm64 native MLX support")
}

// ProjectJANGPackedTensorMetalFused requires the native Metal backend.
func ProjectJANGPackedTensorMetalFused(_ JANGPackedTensorDescriptor, _ []byte, _, _, _ []float32, _ []int32, _ []float32) (JANGPackedProjectionResult, error) {
	return JANGPackedProjectionResult{}, core.NewError("mlx: JANG Metal fused packed projection requires darwin/arm64 native MLX support")
}
