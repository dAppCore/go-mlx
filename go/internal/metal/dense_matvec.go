// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"sync"

	core "dappco.re/go"
)

func nativeMLPMatVec(input *Array, mlp *MLP) (*Array, bool, error) {
	if !nativeMLPMatVecRuntimeEnabled() {
		return nil, false, nil
	}
	if input == nil || !input.Valid() || mlp == nil {
		return nil, false, nil
	}
	activated, ok, err := quantizedDenseGELUSplitGateUpMatVec(input, mlp.GateProj, mlp.UpProj)
	if err != nil || !ok {
		return nil, ok, err
	}
	out, ok, err := quantizedDenseMatVec(activated, mlp.DownProj)
	Free(activated)
	if err != nil || !ok {
		Free(out)
		return nil, ok, err
	}
	return out, true, nil
}

func quantizedDenseMatVec(input *Array, linear *Linear) (*Array, bool, error) {
	meta, ok := validateQuantizedDenseMatVec(input, linear)
	if !ok {
		return nil, false, nil
	}
	kernel := quantizedDenseMatVecKernel(meta, linear.GroupSize, linear.Bits)

	out, err := kernel.DispatchOne(
		MetalKernelGrid{GridX: meta.outDim * 32, GridY: 1, GridZ: 1, TGX: 256, TGY: 1, TGZ: 1},
		meta.outputShape[:], DTypeFloat32,
		input, linear.Weight, linear.Scales, linear.Biases,
	)
	if err != nil {
		return nil, true, core.E("mlx.quantizedDenseMatVec", "apply Metal kernel", err)
	}
	return out, true, nil
}

func quantizedDenseGELUSplitGateUpMatVec(input *Array, gate, up *Linear) (*Array, bool, error) {
	gateMeta, ok := validateQuantizedDenseMatVec(input, gate)
	if !ok {
		return nil, false, nil
	}
	upMeta, ok := validateQuantizedDenseMatVec(input, up)
	if !ok {
		return nil, false, nil
	}
	if gateMeta != upMeta {
		return nil, true, core.NewError(core.Sprintf("mlx: quantized dense split gate/up metadata mismatch: gate=%+v up=%+v", gateMeta, upMeta))
	}

	kernel := quantizedDenseGELUSplitGateUpMatVecKernel(gateMeta, gate.GroupSize, gate.Bits)

	out, err := kernel.DispatchOne(
		MetalKernelGrid{GridX: gateMeta.outDim * 32, GridY: 1, GridZ: 1, TGX: 256, TGY: 1, TGZ: 1},
		gateMeta.outputShape[:], DTypeFloat32,
		input, gate.Weight, gate.Scales, gate.Biases, up.Weight, up.Scales, up.Biases,
	)
	if err != nil {
		return nil, true, core.E("mlx.quantizedDenseGELUSplitGateUpMatVec", "apply Metal kernel", err)
	}
	return out, true, nil
}

type quantizedDenseMatVecMeta struct {
	bits         int
	groupSize    int
	inDim        int
	outDim       int
	packedIn     int
	groups       int
	packFactor   int
	sidecarDType DType
	outputShape  [3]int32
}

func validateQuantizedDenseMatVec(input *Array, linear *Linear) (quantizedDenseMatVecMeta, bool) {
	var meta quantizedDenseMatVecMeta
	if input == nil || !input.Valid() || linear == nil || linear.LoRA != nil {
		return meta, false
	}
	if linear.Weight == nil || !linear.Weight.Valid() || linear.Scales == nil || !linear.Scales.Valid() || linear.Biases == nil || !linear.Biases.Valid() {
		return meta, false
	}
	if !isAffineQuantizationMode(linear.QuantizationMode) {
		return meta, false
	}
	if linear.Bias != nil && linear.Bias.Valid() {
		return meta, false
	}
	if linear.GroupSize <= 0 || (linear.Bits != 4 && linear.Bits != 8) {
		return meta, false
	}
	shape := input.Shape()
	if len(shape) != 3 || shape[0] != 1 || shape[1] != 1 {
		return meta, false
	}
	weightShape := linear.Weight.Shape()
	scaleShape := linear.Scales.Shape()
	biasShape := linear.Biases.Shape()
	if len(weightShape) != 2 || len(scaleShape) != 2 || len(biasShape) != 2 {
		return meta, false
	}
	packFactor := 32 / linear.Bits
	inDim := int(shape[2])
	outDim := int(weightShape[0])
	packedIn := int(weightShape[1])
	groups := inDim / linear.GroupSize
	if inDim <= 0 || outDim <= 0 || packedIn <= 0 || groups <= 0 || inDim%linear.GroupSize != 0 || packedIn*packFactor != inDim {
		return meta, false
	}
	if int(scaleShape[0]) != outDim || int(scaleShape[1]) != groups || int(biasShape[0]) != outDim || int(biasShape[1]) != groups {
		return meta, false
	}
	if linear.Scales.Dtype() != linear.Biases.Dtype() {
		return meta, false
	}
	return quantizedDenseMatVecMeta{
		bits:         linear.Bits,
		groupSize:    linear.GroupSize,
		inDim:        inDim,
		outDim:       outDim,
		packedIn:     packedIn,
		groups:       groups,
		packFactor:   packFactor,
		sidecarDType: linear.Scales.Dtype(),
		outputShape:  [3]int32{shape[0], shape[1], int32(outDim)},
	}, true
}

type quantizedDenseMatVecKernelKey struct {
	bits         int
	groupSize    int
	inDim        int
	outDim       int
	packedIn     int
	sidecarDType DType
}

var quantizedDenseMatVecKernelCache struct {
	sync.Mutex
	kernels map[quantizedDenseMatVecKernelKey]*MetalKernel
}

var quantizedDenseGELUSplitGateUpMatVecKernelCache struct {
	sync.Mutex
	kernels map[quantizedDenseMatVecKernelKey]*MetalKernel
}

func quantizedDenseMatVecKernel(meta quantizedDenseMatVecMeta, groupSize, bits int) *MetalKernel {
	key := quantizedDenseMatVecKernelKey{
		bits:         bits,
		groupSize:    groupSize,
		inDim:        meta.inDim,
		outDim:       meta.outDim,
		packedIn:     meta.packedIn,
		sidecarDType: meta.sidecarDType,
	}
	quantizedDenseMatVecKernelCache.Lock()
	defer quantizedDenseMatVecKernelCache.Unlock()
	if quantizedDenseMatVecKernelCache.kernels == nil {
		quantizedDenseMatVecKernelCache.kernels = make(map[quantizedDenseMatVecKernelKey]*MetalKernel)
	}
	if kernel := quantizedDenseMatVecKernelCache.kernels[key]; kernel != nil {
		return kernel
	}

	source := core.Sprintf(`uint out_col = thread_position_in_grid.x / 32u;
uint lane = thread_index_in_simdgroup;
float sum = 0.0f;
for (uint pack_col = lane; pack_col < uint(%d); pack_col += 32u) {
	uint packed = weight[out_col * uint(%d) + pack_col];
	uint base_in = pack_col * uint(%d);
	for (uint packed_offset = 0; packed_offset < uint(%d); packed_offset++) {
		uint in_col = base_in + packed_offset;
		uint bit_shift = packed_offset * uint(%d);
		uint q = (packed >> bit_shift) & uint(%d);
		uint group = in_col / uint(%d);
		uint scale_index = out_col * uint(%d) + group;
		float w = float(q) * float(scales[scale_index]) + float(qbiases[scale_index]);
		sum += float(x[in_col]) * w;
	}
}
sum = simd_sum(sum);
if (lane == 0u) {
	out[out_col] = sum;
}`,
		meta.packedIn,
		meta.packedIn,
		meta.packFactor,
		meta.packFactor,
		bits,
		(1<<bits)-1,
		groupSize,
		meta.groups,
	)
	header := "#include <metal_stdlib>\n#include <metal_simdgroup>\nusing namespace metal;\n"
	kernel := NewMetalKernel(
		core.Sprintf("quantized_dense_matvec_b%d_g%d_i%d_o%d_p%d_s%d", bits, groupSize, meta.inDim, meta.outDim, meta.packedIn, meta.sidecarDType),
		[]string{"x", "weight", "scales", "qbiases"},
		[]string{"out"},
		source,
		header,
		true,
		false,
	)
	quantizedDenseMatVecKernelCache.kernels[key] = kernel
	return kernel
}

func quantizedDenseGELUSplitGateUpMatVecKernel(meta quantizedDenseMatVecMeta, groupSize, bits int) *MetalKernel {
	key := quantizedDenseMatVecKernelKey{
		bits:         bits,
		groupSize:    groupSize,
		inDim:        meta.inDim,
		outDim:       meta.outDim,
		packedIn:     meta.packedIn,
		sidecarDType: meta.sidecarDType,
	}
	quantizedDenseGELUSplitGateUpMatVecKernelCache.Lock()
	defer quantizedDenseGELUSplitGateUpMatVecKernelCache.Unlock()
	if quantizedDenseGELUSplitGateUpMatVecKernelCache.kernels == nil {
		quantizedDenseGELUSplitGateUpMatVecKernelCache.kernels = make(map[quantizedDenseMatVecKernelKey]*MetalKernel)
	}
	if kernel := quantizedDenseGELUSplitGateUpMatVecKernelCache.kernels[key]; kernel != nil {
		return kernel
	}

	source := core.Sprintf(`uint out_col = thread_position_in_grid.x / 32u;
uint lane = thread_index_in_simdgroup;
float gate_sum = 0.0f;
float up_sum = 0.0f;
for (uint pack_col = lane; pack_col < uint(%d); pack_col += 32u) {
	uint gate_packed = gate_weight[out_col * uint(%d) + pack_col];
	uint up_packed = up_weight[out_col * uint(%d) + pack_col];
	uint base_in = pack_col * uint(%d);
	for (uint packed_offset = 0; packed_offset < uint(%d); packed_offset++) {
		uint in_col = base_in + packed_offset;
		uint bit_shift = packed_offset * uint(%d);
		uint gate_q = (gate_packed >> bit_shift) & uint(%d);
		uint up_q = (up_packed >> bit_shift) & uint(%d);
		uint group = in_col / uint(%d);
		uint scale_index = out_col * uint(%d) + group;
		float gate_w = float(gate_q) * float(gate_scales[scale_index]) + float(gate_qbiases[scale_index]);
		float up_w = float(up_q) * float(up_scales[scale_index]) + float(up_qbiases[scale_index]);
		float input_value = float(x[in_col]);
		gate_sum += input_value * gate_w;
		up_sum += input_value * up_w;
	}
}
gate_sum = simd_sum(gate_sum);
up_sum = simd_sum(up_sum);
if (lane == 0u) {
	float gate_cube = gate_sum * gate_sum * gate_sum;
	float gelu = 0.5f * gate_sum * (1.0f + tanh(0.7978845608028654f * (gate_sum + 0.044715f * gate_cube)));
	out[out_col] = gelu * up_sum;
}`,
		meta.packedIn,
		meta.packedIn,
		meta.packedIn,
		meta.packFactor,
		meta.packFactor,
		bits,
		(1<<bits)-1,
		(1<<bits)-1,
		groupSize,
		meta.groups,
	)
	header := "#include <metal_stdlib>\n#include <metal_simdgroup>\nusing namespace metal;\n"
	kernel := NewMetalKernel(
		core.Sprintf("quantized_dense_gelu_split_gate_up_matvec_b%d_g%d_i%d_o%d_p%d_s%d", bits, groupSize, meta.inDim, meta.outDim, meta.packedIn, meta.sidecarDType),
		[]string{"x", "gate_weight", "gate_scales", "gate_qbiases", "up_weight", "up_scales", "up_qbiases"},
		[]string{"out"},
		source,
		header,
		true,
		false,
	)
	quantizedDenseGELUSplitGateUpMatVecKernelCache.kernels[key] = kernel
	return kernel
}
