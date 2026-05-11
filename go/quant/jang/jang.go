// SPDX-Licence-Identifier: EUPL-1.2


// Package jang holds the Metal-side JANG/JANGTQ dequant + projection kernels.
//
//	out, _ := jang.DequantizePackedTensor(desc, packed, scales, biases)
package jang

import (
	core "dappco.re/go"
	infjang "dappco.re/go/inference/quant/jang"
	"dappco.re/go/mlx/internal/metal"
)

//	res, _ := jang.ProjectPackedTensor(desc, packed, scales, biases, input, shape, bias)
type PackedProjectionResult struct {
	Values []float32 `json:"values"`
	Shape  []int32   `json:"shape"`
}

//	out, _ := jang.DequantizePackedTensor(desc, packed, scales, biases)
func DequantizePackedTensor(desc infjang.PackedTensorDescriptor, packed []byte, scales, biases []float32) ([]float32, error) {
	if err := infjang.ValidatePackedTensor(desc, packed, scales, biases); err != nil {
		return nil, err
	}
	shape, err := MetalShape(desc.Shape)
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

//	res, _ := jang.ProjectPackedTensor(desc, packed, scales, biases, input, shape, bias)
func ProjectPackedTensor(desc infjang.PackedTensorDescriptor, packed []byte, scales, biases, input []float32, inputShape []int32, bias []float32) (PackedProjectionResult, error) {
	return projectPackedTensor(desc, packed, scales, biases, input, inputShape, bias, false)
}

//	res, _ := jang.ProjectPackedTensorFused(desc, packed, scales, biases, input, shape, bias)
func ProjectPackedTensorFused(desc infjang.PackedTensorDescriptor, packed []byte, scales, biases, input []float32, inputShape []int32, bias []float32) (PackedProjectionResult, error) {
	return projectPackedTensor(desc, packed, scales, biases, input, inputShape, bias, true)
}

func projectPackedTensor(desc infjang.PackedTensorDescriptor, packed []byte, scales, biases, input []float32, inputShape []int32, bias []float32, fused bool) (PackedProjectionResult, error) {
	if err := infjang.ValidatePackedTensor(desc, packed, scales, biases); err != nil {
		return PackedProjectionResult{}, err
	}
	weightShape, err := MetalShape(desc.Shape)
	if err != nil {
		return PackedProjectionResult{}, err
	}
	if len(weightShape) != 2 {
		return PackedProjectionResult{}, core.NewError("jang: packed projection weight shape must be [out, in]")
	}
	inputElements, err := ShapeElements(inputShape)
	if err != nil {
		return PackedProjectionResult{}, err
	}
	if inputElements != len(input) {
		return PackedProjectionResult{}, core.NewError(core.Sprintf("jang: packed projection input length %d, expected %d", len(input), inputElements))
	}
	if inputShape[len(inputShape)-1] != weightShape[1] {
		return PackedProjectionResult{}, core.NewError(core.Sprintf("jang: packed projection input last dimension %d, expected %d", inputShape[len(inputShape)-1], weightShape[1]))
	}
	outputShape := append([]int32(nil), inputShape...)
	outputShape[len(outputShape)-1] = weightShape[0]
	if len(bias) > 0 && len(bias) != int(weightShape[0]) {
		return PackedProjectionResult{}, core.NewError(core.Sprintf("jang: packed projection bias length %d, expected %d", len(bias), weightShape[0]))
	}

	packedArray := metal.FromValues(packed, len(packed))
	scalesArray := metal.FromValues(scales, len(scales))
	biasesArray := metal.FromValues(biases, len(biases))
	inputArray := metal.FromValues(input, Int32SliceToInts(inputShape)...)
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
		return PackedProjectionResult{}, err
	}
	defer metal.Free(out)
	metal.Materialize(out)
	return PackedProjectionResult{Values: out.Floats(), Shape: outputShape}, nil
}

func MetalShape(shape []uint64) ([]int32, error) {
	if len(shape) == 0 {
		return nil, core.NewError("jang: metal dequant shape is required")
	}
	out := make([]int32, len(shape))
	for i, dim := range shape {
		if dim == 0 || dim > uint64(^uint32(0)>>1) {
			return nil, core.NewError("jang: metal dequant shape is invalid")
		}
		out[i] = int32(dim)
	}
	return out, nil
}

func ShapeElements(shape []int32) (int, error) {
	if len(shape) == 0 {
		return 0, core.NewError("jang: packed projection input shape is required")
	}
	elements := 1
	maxInt := int(^uint(0) >> 1)
	for _, dim := range shape {
		if dim <= 0 {
			return 0, core.NewError("jang: packed projection input shape is invalid")
		}
		if elements > maxInt/int(dim) {
			return 0, core.NewError("jang: packed projection input shape is too large")
		}
		elements *= int(dim)
	}
	return elements, nil
}

func Int32SliceToInts(values []int32) []int {
	out := make([]int, len(values))
	for i, value := range values {
		out[i] = int(value)
	}
	return out
}
