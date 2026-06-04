// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import core "dappco.re/go"

// DequantizeAutoRoundPacked expands an LSB-first AutoRound packed tensor using
// affine per-group scales and zero-points.
func DequantizeAutoRoundPacked(packed, scales, zeroPoints *Array, outputShape []int32, groupSize, bits, qMin int) (*Array, error) {
	elements, err := validateAutoRoundPackedDequantInputs(packed, scales, zeroPoints, outputShape, groupSize, bits)
	if err != nil {
		return nil, err
	}

	source := core.Sprintf(`uint elem = thread_position_in_grid.x;
uint bit_offset = elem * uint(%d);
uint byte_index = bit_offset >> 3;
uint bit_shift = bit_offset & 7;
uint word = uint(packed[byte_index]);
if (bit_shift + uint(%d) > 8u) {
	word = word | (uint(packed[byte_index + 1]) << 8);
}
uint raw = (word >> bit_shift) & uint(%d);
int q = int(raw) + int(%d);
uint group = elem / uint(%d);
out[elem] = (float(q) - zero_points[group]) * scales[group];`, bits, bits, (1<<bits)-1, qMin, groupSize)

	kernel := NewMetalKernel(core.Sprintf("autoround_dequant_bits_%d_group_%d_qmin_%s", bits, groupSize, autoRoundQMinKernelSuffix(qMin)), []string{"packed", "scales", "zero_points"}, []string{"out"}, source, "", true, false)
	defer kernel.Free()

	out, err := kernel.DispatchOne(
		MetalKernelGrid{GridX: elements, GridY: 1, GridZ: 1, TGX: 256, TGY: 1, TGZ: 1},
		outputShape, DTypeFloat32,
		packed, scales, zeroPoints,
	)
	if err != nil {
		return nil, core.E("mlx.DequantizeAutoRoundPacked", "apply Metal kernel", err)
	}
	return out, nil
}

// AutoRoundPackedLinear computes input @ dequantized(weight).T plus optional
// bias for AutoRound native weight-only packs.
func AutoRoundPackedLinear(input, packed, scales, zeroPoints, bias *Array, weightShape []int32, groupSize, bits, qMin int) (*Array, error) {
	if err := validateAutoRoundPackedLinearInputs(input, bias, weightShape); err != nil {
		return nil, err
	}
	weight, err := DequantizeAutoRoundPacked(packed, scales, zeroPoints, weightShape, groupSize, bits, qMin)
	if err != nil {
		return nil, err
	}
	weightT := Transpose(weight)
	out := Matmul(input, weightT)
	Free(weight, weightT)
	if bias != nil && bias.Valid() {
		oldOut := out
		out = Add(out, bias)
		Free(oldOut)
	}
	return out, nil
}

// AutoRoundPackedLinearFused computes input @ dequantized(weight).T plus
// optional bias without materialising the dense dequantized weight.
func AutoRoundPackedLinearFused(input, packed, scales, zeroPoints, bias *Array, weightShape []int32, groupSize, bits, qMin int) (*Array, error) {
	if err := validateAutoRoundPackedLinearInputs(input, bias, weightShape); err != nil {
		return nil, err
	}
	if _, err := validateAutoRoundPackedDequantInputs(packed, scales, zeroPoints, weightShape, groupSize, bits); err != nil {
		return nil, err
	}
	outShape := jangPackedLinearOutputShape(input.Shape(), weightShape[0])
	rows := input.Size() / int(weightShape[1])
	outDim := int(weightShape[0])
	inDim := int(weightShape[1])
	source := core.Sprintf(`uint elem = thread_position_in_grid.x;
uint out_col = elem %% uint(%d);
uint row = elem / uint(%d);
float sum = 0.0f;
for (uint in_col = 0; in_col < uint(%d); in_col++) {
	uint weight_index = out_col * uint(%d) + in_col;
	uint bit_offset = weight_index * uint(%d);
	uint byte_index = bit_offset >> 3;
	uint bit_shift = bit_offset & 7;
	uint word = uint(packed[byte_index]);
	if (bit_shift + uint(%d) > 8u) {
		word = word | (uint(packed[byte_index + 1]) << 8);
	}
	uint raw = (word >> bit_shift) & uint(%d);
	int q = int(raw) + int(%d);
	uint group = weight_index / uint(%d);
	float w = (float(q) - zero_points[group]) * scales[group];
	sum += x[row * uint(%d) + in_col] * w;
}
out[elem] = sum%s;`, outDim, outDim, inDim, inDim, bits, bits, (1<<bits)-1, qMin, groupSize, inDim, jangPackedLinearBiasSource(bias != nil && bias.Valid()))

	inputNames := []string{"x", "packed", "scales", "zero_points"}
	inputs := []*Array{input, packed, scales, zeroPoints}
	if bias != nil && bias.Valid() {
		inputNames = append(inputNames, "proj_bias")
		inputs = append(inputs, bias)
	}
	kernel := NewMetalKernel(core.Sprintf("autoround_packed_linear_fused_bits_%d_group_%d_qmin_%s_bias_%t", bits, groupSize, autoRoundQMinKernelSuffix(qMin), bias != nil && bias.Valid()), inputNames, []string{"out"}, source, "", true, false)
	defer kernel.Free()

	out, err := kernel.DispatchOne(
		MetalKernelGrid{GridX: rows * outDim, GridY: 1, GridZ: 1, TGX: 256, TGY: 1, TGZ: 1},
		outShape, DTypeFloat32,
		inputs...,
	)
	if err != nil {
		return nil, core.E("mlx.AutoRoundPackedLinearFused", "apply Metal kernel", err)
	}
	return out, nil
}

func validateAutoRoundPackedDequantInputs(packed, scales, zeroPoints *Array, outputShape []int32, groupSize, bits int) (int, error) {
	if packed == nil || !packed.Valid() {
		return 0, core.NewError("mlx: AutoRound dequant requires packed uint8 input")
	}
	if scales == nil || !scales.Valid() || zeroPoints == nil || !zeroPoints.Valid() {
		return 0, core.NewError("mlx: AutoRound dequant requires scale and zero-point inputs")
	}
	if packed.Dtype() != DTypeUint8 {
		return 0, core.NewError("mlx: AutoRound dequant packed input must be uint8")
	}
	if scales.Dtype() != DTypeFloat32 || zeroPoints.Dtype() != DTypeFloat32 {
		return 0, core.NewError("mlx: AutoRound dequant scales and zero-points must be float32")
	}
	if !validAutoRoundPackedBits(bits) {
		return 0, core.NewError(core.Sprintf("mlx: AutoRound dequant unsupported bits %d", bits))
	}
	if groupSize <= 0 {
		return 0, core.NewError("mlx: AutoRound dequant group size must be positive")
	}
	elements, err := jangOutputElements(outputShape)
	if err != nil {
		return 0, err
	}
	expectedPacked := (elements*bits + 7) / 8
	if packed.Size() != expectedPacked {
		return 0, core.NewError(core.Sprintf("mlx: AutoRound dequant packed length %d, expected %d", packed.Size(), expectedPacked))
	}
	expectedGroups := (elements + groupSize - 1) / groupSize
	if scales.Size() != expectedGroups {
		return 0, core.NewError(core.Sprintf("mlx: AutoRound dequant scale count %d, expected %d", scales.Size(), expectedGroups))
	}
	if zeroPoints.Size() != expectedGroups {
		return 0, core.NewError(core.Sprintf("mlx: AutoRound dequant zero-point count %d, expected %d", zeroPoints.Size(), expectedGroups))
	}
	return elements, nil
}

func validateAutoRoundPackedLinearInputs(input, bias *Array, weightShape []int32) error {
	if input == nil || !input.Valid() {
		return core.NewError("mlx: AutoRound packed linear requires input")
	}
	if input.Dtype() != DTypeFloat32 {
		return core.NewError("mlx: AutoRound packed linear input must be float32")
	}
	if len(weightShape) != 2 || weightShape[0] <= 0 || weightShape[1] <= 0 {
		return core.NewError("mlx: AutoRound packed linear weight shape must be [out, in]")
	}
	if input.NumDims() == 0 || int32(input.Dim(input.NumDims()-1)) != weightShape[1] {
		return core.NewError(core.Sprintf("mlx: AutoRound packed linear input last dimension %d, expected %d", input.Dim(input.NumDims()-1), weightShape[1]))
	}
	if bias != nil && bias.Valid() {
		if bias.Dtype() != DTypeFloat32 {
			return core.NewError("mlx: AutoRound packed linear bias must be float32")
		}
		if bias.Size() != int(weightShape[0]) {
			return core.NewError(core.Sprintf("mlx: AutoRound packed linear bias size %d, expected %d", bias.Size(), weightShape[0]))
		}
	}
	return nil
}

func validAutoRoundPackedBits(bits int) bool {
	switch bits {
	case 2, 3, 4, 8:
		return true
	default:
		return false
	}
}

func autoRoundQMinKernelSuffix(qMin int) string {
	if qMin < 0 {
		return core.Sprintf("n%d", -qMin)
	}
	return core.Sprintf("%d", qMin)
}
