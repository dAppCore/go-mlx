// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import core "dappco.re/go"

// DequantizeJANGPacked expands an LSB-first JANG/JANGTQ packed tensor using
// affine per-group scales and biases. It is the first native MXTQ building
// block for MiniMax-style routed expert weights.
func DequantizeJANGPacked(packed, scales, biases *Array, outputShape []int32, groupSize, bits int) (*Array, error) {
	elements, err := validateJANGPackedDequantInputs(packed, scales, biases, outputShape, groupSize, bits)
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
uint q = (word >> bit_shift) & uint(%d);
uint group = elem / uint(%d);
out[elem] = float(q) * scales[group] + biases[group];`, bits, bits, (1<<bits)-1, groupSize)

	kernel := NewMetalKernel(core.Sprintf("jang_dequant_bits_%d_group_%d", bits, groupSize), []string{"packed", "scales", "biases"}, []string{"out"}, source, "", true, false)
	defer kernel.Free()

	cfg := NewMetalKernelConfig()
	defer cfg.Free()
	cfg.SetGrid(elements, 1, 1)
	cfg.SetThreadGroup(256, 1, 1)
	cfg.AddOutputArg(outputShape, DTypeFloat32)

	out, err := kernel.ApplyOne(cfg, packed, scales, biases)
	if err != nil {
		return nil, core.E("mlx.DequantizeJANGPacked", "apply Metal kernel", err)
	}
	return out, nil
}

// JANGPackedLinear computes input @ dequantized(weight).T plus optional bias.
// This is an intentionally small bring-up path for packed MiniMax experts; the
// follow-up fused kernel can replace the internal dequant+matmul without
// changing call sites.
func JANGPackedLinear(input, packed, scales, biases, bias *Array, weightShape []int32, groupSize, bits int) (*Array, error) {
	if err := validateJANGPackedLinearInputs(input, bias, weightShape); err != nil {
		return nil, err
	}
	weight, err := DequantizeJANGPacked(packed, scales, biases, weightShape, groupSize, bits)
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

// JANGPackedLinearFused computes input @ dequantized(weight).T plus optional
// bias without materialising the dense dequantized weight.
func JANGPackedLinearFused(input, packed, scales, biases, bias *Array, weightShape []int32, groupSize, bits int) (*Array, error) {
	if err := validateJANGPackedLinearInputs(input, bias, weightShape); err != nil {
		return nil, err
	}
	if _, err := validateJANGPackedDequantInputs(packed, scales, biases, weightShape, groupSize, bits); err != nil {
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
	uint q = (word >> bit_shift) & uint(%d);
	uint group = weight_index / uint(%d);
	float w = float(q) * scales[group] + qbiases[group];
	sum += x[row * uint(%d) + in_col] * w;
}
out[elem] = sum%s;`, outDim, outDim, inDim, inDim, bits, bits, (1<<bits)-1, groupSize, inDim, jangPackedLinearBiasSource(bias != nil && bias.Valid()))

	inputNames := []string{"x", "packed", "scales", "qbiases"}
	inputs := []*Array{input, packed, scales, biases}
	if bias != nil && bias.Valid() {
		inputNames = append(inputNames, "proj_bias")
		inputs = append(inputs, bias)
	}
	kernel := NewMetalKernel(core.Sprintf("jang_packed_linear_fused_bits_%d_group_%d_bias_%t", bits, groupSize, bias != nil && bias.Valid()), inputNames, []string{"out"}, source, "", true, false)
	defer kernel.Free()

	cfg := NewMetalKernelConfig()
	defer cfg.Free()
	cfg.SetGrid(rows*outDim, 1, 1)
	cfg.SetThreadGroup(256, 1, 1)
	cfg.AddOutputArg(outShape, DTypeFloat32)

	out, err := kernel.ApplyOne(cfg, inputs...)
	if err != nil {
		return nil, core.E("mlx.JANGPackedLinearFused", "apply Metal kernel", err)
	}
	return out, nil
}

func validateJANGPackedDequantInputs(packed, scales, biases *Array, outputShape []int32, groupSize, bits int) (int, error) {
	if packed == nil || !packed.Valid() {
		return 0, core.NewError("mlx: JANG dequant requires packed uint8 input")
	}
	if scales == nil || !scales.Valid() || biases == nil || !biases.Valid() {
		return 0, core.NewError("mlx: JANG dequant requires scale and bias inputs")
	}
	if packed.Dtype() != DTypeUint8 {
		return 0, core.NewError("mlx: JANG dequant packed input must be uint8")
	}
	if scales.Dtype() != DTypeFloat32 || biases.Dtype() != DTypeFloat32 {
		return 0, core.NewError("mlx: JANG dequant scales and biases must be float32")
	}
	if !validJANGPackedBits(bits) {
		return 0, core.NewError(core.Sprintf("mlx: JANG dequant unsupported bits %d", bits))
	}
	if groupSize <= 0 {
		return 0, core.NewError("mlx: JANG dequant group size must be positive")
	}
	elements, err := jangOutputElements(outputShape)
	if err != nil {
		return 0, err
	}
	expectedPacked := (elements*bits + 7) / 8
	if packed.Size() != expectedPacked {
		return 0, core.NewError(core.Sprintf("mlx: JANG dequant packed length %d, expected %d", packed.Size(), expectedPacked))
	}
	expectedGroups := (elements + groupSize - 1) / groupSize
	if scales.Size() != expectedGroups {
		return 0, core.NewError(core.Sprintf("mlx: JANG dequant scale count %d, expected %d", scales.Size(), expectedGroups))
	}
	if biases.Size() != expectedGroups {
		return 0, core.NewError(core.Sprintf("mlx: JANG dequant bias count %d, expected %d", biases.Size(), expectedGroups))
	}
	return elements, nil
}

func validateJANGPackedLinearInputs(input, bias *Array, weightShape []int32) error {
	if input == nil || !input.Valid() {
		return core.NewError("mlx: JANG packed linear requires input")
	}
	if input.Dtype() != DTypeFloat32 {
		return core.NewError("mlx: JANG packed linear input must be float32")
	}
	if len(weightShape) != 2 || weightShape[0] <= 0 || weightShape[1] <= 0 {
		return core.NewError("mlx: JANG packed linear weight shape must be [out, in]")
	}
	if input.NumDims() == 0 || int32(input.Dim(input.NumDims()-1)) != weightShape[1] {
		return core.NewError(core.Sprintf("mlx: JANG packed linear input last dimension %d, expected %d", input.Dim(input.NumDims()-1), weightShape[1]))
	}
	if bias != nil && bias.Valid() {
		if bias.Dtype() != DTypeFloat32 {
			return core.NewError("mlx: JANG packed linear bias must be float32")
		}
		if bias.Size() != int(weightShape[0]) {
			return core.NewError(core.Sprintf("mlx: JANG packed linear bias size %d, expected %d", bias.Size(), weightShape[0]))
		}
	}
	return nil
}

func jangPackedLinearOutputShape(inputShape []int32, outDim int32) []int32 {
	out := append([]int32(nil), inputShape...)
	out[len(out)-1] = outDim
	return out
}

func jangPackedLinearBiasSource(hasBias bool) string {
	if !hasBias {
		return ""
	}
	return " + proj_bias[out_col]"
}

func validJANGPackedBits(bits int) bool {
	switch bits {
	case 1, 2, 3, 4, 8:
		return true
	default:
		return false
	}
}

func jangOutputElements(shape []int32) (int, error) {
	if len(shape) == 0 {
		return 0, core.NewError("mlx: JANG dequant output shape is required")
	}
	elements := 1
	maxIntValue := int(^uint(0) >> 1)
	for _, dim := range shape {
		if dim <= 0 {
			return 0, core.NewError("mlx: JANG dequant output shape dimensions must be positive")
		}
		if elements > maxIntValue/int(dim) {
			return 0, core.NewError("mlx: JANG dequant output shape is too large")
		}
		elements *= int(dim)
	}
	return elements, nil
}
