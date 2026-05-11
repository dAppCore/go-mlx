// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && !nomlx

package mlx

import (
	core "dappco.re/go"
	"dappco.re/go/mlx/internal/metal"
)

// JANGPackedProjectionResult is the host result from a descriptor-level packed
// projection parity run.
type JANGPackedProjectionResult struct {
	Values []float32 `json:"values"`
	Shape  []int32   `json:"shape"`
}

// DequantizeJANGPackedTensorMetal expands a JANG/JANGTQ packed tensor with the
// native Metal path and returns host floats. It is intended for parity checks
// and loader bring-up before the packed expert GEMM path consumes GPU arrays
// directly.
func DequantizeJANGPackedTensorMetal(desc JANGPackedTensorDescriptor, packed []byte, scales, biases []float32) ([]float32, error) {
	if err := ValidateJANGPackedTensor(desc, packed, scales, biases); err != nil {
		return nil, err
	}
	shape, err := jangMetalShape(desc.Shape)
	if err != nil {
		return nil, err
	}
	packedArray := metal.FromValues(packed, len(packed))
	scalesArray := metal.FromValues(scales, len(scales))
	biasesArray := metal.FromValues(biases, len(biases))
	defer metal.Free(packedArray, scalesArray, biasesArray)

	out, err := metal.DequantizeJANGPacked(packedArray, scalesArray, biasesArray, shape, desc.GroupSize, desc.Bits)
	if err != nil {
		return nil, err
	}
	defer metal.Free(out)
	metal.Materialize(out)
	return out.Floats(), nil
}

// ProjectJANGPackedTensorMetal computes input @ dequantized(desc).T with an
// optional projection bias. It is a composed bring-up path for packed expert
// projections before fused packed-dequant matmul lands.
func ProjectJANGPackedTensorMetal(desc JANGPackedTensorDescriptor, packed []byte, scales, biases, input []float32, inputShape []int32, bias []float32) (JANGPackedProjectionResult, error) {
	return projectJANGPackedTensorMetal(desc, packed, scales, biases, input, inputShape, bias, false)
}

// ProjectJANGPackedTensorMetalFused computes input @ dequantized(desc).T
// directly from packed bytes, avoiding dense dequantized weight materialisation.
func ProjectJANGPackedTensorMetalFused(desc JANGPackedTensorDescriptor, packed []byte, scales, biases, input []float32, inputShape []int32, bias []float32) (JANGPackedProjectionResult, error) {
	return projectJANGPackedTensorMetal(desc, packed, scales, biases, input, inputShape, bias, true)
}

func projectJANGPackedTensorMetal(desc JANGPackedTensorDescriptor, packed []byte, scales, biases, input []float32, inputShape []int32, bias []float32, fused bool) (JANGPackedProjectionResult, error) {
	if err := ValidateJANGPackedTensor(desc, packed, scales, biases); err != nil {
		return JANGPackedProjectionResult{}, err
	}
	weightShape, err := jangMetalShape(desc.Shape)
	if err != nil {
		return JANGPackedProjectionResult{}, err
	}
	if len(weightShape) != 2 {
		return JANGPackedProjectionResult{}, core.NewError("mlx: JANG packed projection weight shape must be [out, in]")
	}
	inputElements, err := jangMetalShapeElements(inputShape)
	if err != nil {
		return JANGPackedProjectionResult{}, err
	}
	if inputElements != len(input) {
		return JANGPackedProjectionResult{}, core.NewError(core.Sprintf("mlx: JANG packed projection input length %d, expected %d", len(input), inputElements))
	}
	if inputShape[len(inputShape)-1] != weightShape[1] {
		return JANGPackedProjectionResult{}, core.NewError(core.Sprintf("mlx: JANG packed projection input last dimension %d, expected %d", inputShape[len(inputShape)-1], weightShape[1]))
	}
	outputShape := append([]int32(nil), inputShape...)
	outputShape[len(outputShape)-1] = weightShape[0]
	if len(bias) > 0 && len(bias) != int(weightShape[0]) {
		return JANGPackedProjectionResult{}, core.NewError(core.Sprintf("mlx: JANG packed projection bias length %d, expected %d", len(bias), weightShape[0]))
	}

	packedArray := metal.FromValues(packed, len(packed))
	scalesArray := metal.FromValues(scales, len(scales))
	biasesArray := metal.FromValues(biases, len(biases))
	inputArray := metal.FromValues(input, int32SliceToInts(inputShape)...)
	var biasArray *metal.Array
	if len(bias) > 0 {
		biasArray = metal.FromValues(bias, len(bias))
	}
	defer metal.Free(packedArray, scalesArray, biasesArray, inputArray, biasArray)

	var out *metal.Array
	if fused {
		out, err = metal.JANGPackedLinearFused(inputArray, packedArray, scalesArray, biasesArray, biasArray, weightShape, desc.GroupSize, desc.Bits)
	} else {
		out, err = metal.JANGPackedLinear(inputArray, packedArray, scalesArray, biasesArray, biasArray, weightShape, desc.GroupSize, desc.Bits)
	}
	if err != nil {
		return JANGPackedProjectionResult{}, err
	}
	defer metal.Free(out)
	metal.Materialize(out)
	return JANGPackedProjectionResult{Values: out.Floats(), Shape: outputShape}, nil
}

func jangMetalShape(shape []uint64) ([]int32, error) {
	if len(shape) == 0 {
		return nil, core.NewError("mlx: JANG Metal dequant shape is required")
	}
	out := make([]int32, len(shape))
	for i, dim := range shape {
		if dim == 0 || dim > uint64(^uint32(0)>>1) {
			return nil, core.NewError("mlx: JANG Metal dequant shape is invalid")
		}
		out[i] = int32(dim)
	}
	return out, nil
}

func jangMetalShapeElements(shape []int32) (int, error) {
	if len(shape) == 0 {
		return 0, core.NewError("mlx: JANG packed projection input shape is required")
	}
	elements := 1
	maxIntValue := int(^uint(0) >> 1)
	for _, dim := range shape {
		if dim <= 0 {
			return 0, core.NewError("mlx: JANG packed projection input shape is invalid")
		}
		if elements > maxIntValue/int(dim) {
			return 0, core.NewError("mlx: JANG packed projection input shape is too large")
		}
		elements *= int(dim)
	}
	return elements, nil
}

func int32SliceToInts(values []int32) []int {
	out := make([]int, len(values))
	for i, value := range values {
		out[i] = int(value)
	}
	return out
}
