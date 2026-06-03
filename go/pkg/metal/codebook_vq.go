// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import core "dappco.re/go"

// CodebookVQMatVec computes input @ dequantized(weight).T plus optional bias
// for a VQ/codebook-compressed matrix. Codes are unpacked integer code IDs,
// codebook is [codebook_size, code_dim], and weightShape is [out, in].
func CodebookVQMatVec(input, codes, codebook, bias *Array, weightShape []int32, codeDim int) (*Array, error) {
	if err := validateCodebookVQMatVecInputs(input, codes, codebook, bias, weightShape, codeDim); err != nil {
		return nil, err
	}
	outDim := int(weightShape[0])
	inDim := int(weightShape[1])
	rows := input.Size() / inDim
	codebookSize := codebook.Dim(0)
	hasBias := bias != nil && bias.Valid()
	source := core.Sprintf(`uint elem = thread_position_in_grid.x;
uint out_col = elem %% uint(%d);
uint row = elem / uint(%d);
float sum = 0.0f;
for (uint in_col = 0; in_col < uint(%d); in_col++) {
	uint weight_index = out_col * uint(%d) + in_col;
	uint code_index = weight_index / uint(%d);
	uint code_offset = weight_index %% uint(%d);
	uint code_id = uint(codes[code_index]);
	if (code_id < uint(%d)) {
		float w = codebook[code_id * uint(%d) + code_offset];
		sum += x[row * uint(%d) + in_col] * w;
	}
}
out[elem] = sum%s;`, outDim, outDim, inDim, inDim, codeDim, codeDim, codebookSize, codeDim, inDim, codebookVQBiasSource(hasBias))

	inputNames := []string{"x", "codes", "codebook"}
	inputs := []*Array{input, codes, codebook}
	if hasBias {
		inputNames = append(inputNames, "bias")
		inputs = append(inputs, bias)
	}
	kernel := NewMetalKernel(core.Sprintf("codebook_vq_matvec_dim_%d_bias_%t", codeDim, hasBias), inputNames, []string{"out"}, source, "", true, false)
	defer kernel.Free()

	out, err := kernel.DispatchOne(
		MetalKernelGrid{GridX: rows * outDim, GridY: 1, GridZ: 1, TGX: 256, TGY: 1, TGZ: 1},
		codebookVQOutputShape(input.Shape(), weightShape[0]), DTypeFloat32,
		inputs...,
	)
	if err != nil {
		return nil, core.E("mlx.CodebookVQMatVec", "apply Metal kernel", err)
	}
	return out, nil
}

func validateCodebookVQMatVecInputs(input, codes, codebook, bias *Array, weightShape []int32, codeDim int) error {
	if input == nil || !input.Valid() {
		return core.NewError("mlx: codebook VQ matvec requires input")
	}
	if codes == nil || !codes.Valid() {
		return core.NewError("mlx: codebook VQ matvec requires codes")
	}
	if codebook == nil || !codebook.Valid() {
		return core.NewError("mlx: codebook VQ matvec requires codebook")
	}
	if input.Dtype() != DTypeFloat32 {
		return core.NewError("mlx: codebook VQ matvec input must be float32")
	}
	if !codebookVQCodeDType(codes.Dtype()) {
		return core.NewError("mlx: codebook VQ matvec codes must be uint8, uint16, or uint32")
	}
	if codebook.Dtype() != DTypeFloat32 {
		return core.NewError("mlx: codebook VQ matvec codebook must be float32")
	}
	if len(weightShape) != 2 || weightShape[0] <= 0 || weightShape[1] <= 0 {
		return core.NewError("mlx: codebook VQ matvec weight shape must be [out, in]")
	}
	if codeDim <= 0 {
		return core.NewError("mlx: codebook VQ matvec code_dim must be positive")
	}
	outDim := int(weightShape[0])
	inDim := int(weightShape[1])
	elements := outDim * inDim
	if elements%codeDim != 0 {
		return core.NewError(core.Sprintf("mlx: codebook VQ matvec weight elements %d must be divisible by code_dim %d", elements, codeDim))
	}
	if input.NumDims() == 0 || input.Dim(input.NumDims()-1) != inDim {
		return core.NewError(core.Sprintf("mlx: codebook VQ matvec input last dimension %d, expected %d", input.Dim(input.NumDims()-1), inDim))
	}
	if codes.Size() != elements/codeDim {
		return core.NewError(core.Sprintf("mlx: codebook VQ matvec code count %d, expected %d", codes.Size(), elements/codeDim))
	}
	if codebook.NumDims() != 2 || codebook.Dim(1) != codeDim {
		return core.NewError(core.Sprintf("mlx: codebook VQ matvec codebook shape %+v, expected [entries %d]", codebook.Shape(), codeDim))
	}
	if bias != nil && bias.Valid() {
		if bias.Dtype() != DTypeFloat32 {
			return core.NewError("mlx: codebook VQ matvec bias must be float32")
		}
		if bias.Size() != outDim {
			return core.NewError(core.Sprintf("mlx: codebook VQ matvec bias size %d, expected %d", bias.Size(), outDim))
		}
	}
	return nil
}

func codebookVQOutputShape(inputShape []int32, outDim int32) []int32 {
	out := append([]int32(nil), inputShape...)
	out[len(out)-1] = outDim
	return out
}

func codebookVQCodeDType(dtype DType) bool {
	return dtype == DTypeUint8 || dtype == DTypeUint16 || dtype == DTypeUint32
}

func codebookVQBiasSource(hasBias bool) string {
	if !hasBias {
		return ""
	}
	return " + bias[out_col]"
}
