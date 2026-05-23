// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"sync"

	core "dappco.re/go"
)

// Constant validation errors hoisted to package vars — each previously
// allocated a fresh core.NewError on the (rare but hot under churn)
// validation path. MoE expert-ID matvec validation fires on every
// dispatch under the explicit Gemma 4 opt-in gate; W8-J / W9-K shaved
// the validate path further, so the per-call allocation here is the
// last remaining alloc on the route's validation hot path.
var (
	errEIMWSumRouteWeightsDtype  = core.NewError("mlx: quantized expert id weighted matvec sum route weights must be float32")
	errEIMWSumNeedRouteWeights   = core.NewError("mlx: quantized expert id weighted matvec sum requires route weights")
	errEIMWeightDtype            = core.NewError("mlx: quantized expert id matvec weight must be uint32")
	errEIMScalesBiasesDtype      = core.NewError("mlx: quantized expert id matvec scales and biases must be float32, float16, or bfloat16")
	errEIMScalesBiasesShape      = core.NewError("mlx: quantized expert id matvec scales and biases must be [experts, out, groups]")
	errEIMNeedWeightScalesBiases = core.NewError("mlx: quantized expert id matvec requires weight, scales, and biases")
	errEIMNeedInput              = core.NewError("mlx: quantized expert id matvec requires input")
	errEIMNeedExpertIDs          = core.NewError("mlx: quantized expert id matvec requires expert ids")
	errEIMInputDtype             = core.NewError("mlx: quantized expert id matvec input must be float32")
	errEIMGroupSizeInvalid       = core.NewError("mlx: quantized expert id matvec group size must be positive")
	errEIMExpertIDsDtype         = core.NewError("mlx: quantized expert id matvec expert ids must be int32 or uint32")
	errEIMDimsInvalid            = core.NewError("mlx: quantized expert id matvec dimensions must be positive")
)

// quantizedExpertIDMatVec is a correctness scaffold for llama.cpp-style
// expert-ID matvec work. It consumes MLX affine-packed quantized expert rows and
// produces one route row per expert id. One SIMD group reduces each routed
// output row; the helper is internal and only wired into Gemma 4 behind an
// explicit opt-in gate.
func quantizedExpertIDMatVec(input, weight, scales, biases, expertIDs *Array, groupSize, bits int) (*Array, error) {
	meta, err := validateQuantizedExpertIDMatVec(input, weight, scales, biases, expertIDs, groupSize, bits)
	if err != nil {
		return nil, err
	}

	kernel := quantizedExpertIDMatVecKernel(meta, groupSize, bits)

	out, err := kernel.DispatchOne(
		MetalKernelGrid{GridX: meta.routes * meta.outDim * 32, GridY: 1, GridZ: 1, TGX: 256, TGY: 1, TGZ: 1},
		[]int32{int32(meta.routes), int32(meta.outDim)}, DTypeFloat32,
		input, weight, scales, biases, expertIDs,
	)
	if err != nil {
		return nil, core.E("mlx.quantizedExpertIDMatVec", "apply Metal kernel", err)
	}
	return out, nil
}

// quantizedExpertIDGELUGateUpMatVec computes GELU(gate) * up directly from a
// fused gate_up expert projection. It avoids materialising the two projection
// halves and the separate GELU/multiply graph nodes on single-token MoE decode.
func quantizedExpertIDGELUGateUpMatVec(input, weight, scales, biases, expertIDs *Array, groupSize, bits int) (*Array, error) {
	meta, err := validateQuantizedExpertIDMatVec(input, weight, scales, biases, expertIDs, groupSize, bits)
	if err != nil {
		return nil, err
	}
	if meta.outDim%2 != 0 {
		return nil, core.NewError(core.Sprintf("mlx: quantized expert id gate/up matvec output dim %d must be even", meta.outDim))
	}

	kernel := quantizedExpertIDGELUGateUpMatVecKernel(meta, groupSize, bits)

	out, err := kernel.DispatchOne(
		MetalKernelGrid{GridX: meta.routes * (meta.outDim / 2) * 32, GridY: 1, GridZ: 1, TGX: 256, TGY: 1, TGZ: 1},
		[]int32{int32(meta.routes), int32(meta.outDim / 2)}, DTypeFloat32,
		input, weight, scales, biases, expertIDs,
	)
	if err != nil {
		return nil, core.E("mlx.quantizedExpertIDGELUGateUpMatVec", "apply Metal kernel", err)
	}
	return out, nil
}

// quantizedExpertIDGELUSplitGateUpMatVec computes GELU(gate) * up directly
// when Gemma 4 stores gate and up expert projections as separate quantized
// tensors. The active MLX 26B A4B q4 safetensors use this split layout.
func quantizedExpertIDGELUSplitGateUpMatVec(input, gateWeight, gateScales, gateBiases, upWeight, upScales, upBiases, expertIDs *Array, groupSize, bits int) (*Array, error) {
	gateMeta, err := validateQuantizedExpertIDMatVec(input, gateWeight, gateScales, gateBiases, expertIDs, groupSize, bits)
	if err != nil {
		return nil, err
	}
	upMeta, err := validateQuantizedExpertIDMatVec(input, upWeight, upScales, upBiases, expertIDs, groupSize, bits)
	if err != nil {
		return nil, err
	}
	if gateMeta != upMeta {
		return nil, core.NewError(core.Sprintf("mlx: quantized expert id split gate/up metadata mismatch: gate=%+v up=%+v", gateMeta, upMeta))
	}

	kernel := quantizedExpertIDGELUSplitGateUpMatVecKernel(gateMeta, groupSize, bits)

	out, err := kernel.DispatchOne(
		MetalKernelGrid{GridX: gateMeta.routes * gateMeta.outDim * 32, GridY: 1, GridZ: 1, TGX: 256, TGY: 1, TGZ: 1},
		[]int32{int32(gateMeta.routes), int32(gateMeta.outDim)}, DTypeFloat32,
		input, gateWeight, gateScales, gateBiases, upWeight, upScales, upBiases, expertIDs,
	)
	if err != nil {
		return nil, core.E("mlx.quantizedExpertIDGELUSplitGateUpMatVec", "apply Metal kernel", err)
	}
	return out, nil
}

// quantizedExpertIDWeightedMatVecSum computes the routed expert matvec for each
// route and returns the weighted sum across routes. Gemma 4 uses this for the
// expert down projection under the opt-in expert-ID path.
func quantizedExpertIDWeightedMatVecSum(input, routeWeights, weight, scales, biases, expertIDs *Array, groupSize, bits int) (*Array, error) {
	meta, err := validateQuantizedExpertIDMatVec(input, weight, scales, biases, expertIDs, groupSize, bits)
	if err != nil {
		return nil, err
	}
	if routeWeights == nil || !routeWeights.Valid() {
		return nil, errEIMWSumNeedRouteWeights
	}
	if routeWeights.Dtype() != DTypeFloat32 {
		return nil, errEIMWSumRouteWeightsDtype
	}
	if routeWeights.Size() != meta.routes {
		return nil, core.NewError(core.Sprintf("mlx: quantized expert id weighted matvec sum route weight count %d, expected %d", routeWeights.Size(), meta.routes))
	}

	kernel := quantizedExpertIDWeightedMatVecSumKernel(meta, groupSize, bits)

	out, err := kernel.DispatchOne(
		MetalKernelGrid{GridX: meta.outDim * 32, GridY: 1, GridZ: 1, TGX: 256, TGY: 1, TGZ: 1},
		[]int32{int32(meta.outDim)}, DTypeFloat32,
		input, routeWeights, weight, scales, biases, expertIDs,
	)
	if err != nil {
		return nil, core.E("mlx.quantizedExpertIDWeightedMatVecSum", "apply Metal kernel", err)
	}
	return out, nil
}

type quantizedExpertIDMatVecKernelKey struct {
	bits         int
	groupSize    int
	routes       int
	inDim        int
	outDim       int
	packedIn     int
	sidecarDType DType
	sharedInput  bool
	unrolledQ4   bool
}

var quantizedExpertIDMatVecKernelCache struct {
	sync.Mutex
	kernels map[quantizedExpertIDMatVecKernelKey]*MetalKernel
}

var quantizedExpertIDWeightedMatVecSumKernelCache struct {
	sync.Mutex
	kernels map[quantizedExpertIDMatVecKernelKey]*MetalKernel
}

var quantizedExpertIDGELUGateUpMatVecKernelCache struct {
	sync.Mutex
	kernels map[quantizedExpertIDMatVecKernelKey]*MetalKernel
}

var quantizedExpertIDGELUSplitGateUpMatVecKernelCache struct {
	sync.Mutex
	kernels map[quantizedExpertIDMatVecKernelKey]*MetalKernel
}

func quantizedExpertIDMatVecKernel(meta quantizedExpertIDMatVecMeta, groupSize, bits int) *MetalKernel {
	key := quantizedExpertIDMatVecKernelKey{
		bits:         bits,
		groupSize:    groupSize,
		routes:       meta.routes,
		inDim:        meta.inDim,
		outDim:       meta.outDim,
		packedIn:     meta.packedIn,
		sidecarDType: meta.sidecarDType,
		sharedInput:  meta.sharedInput,
	}
	quantizedExpertIDMatVecKernelCache.Lock()
	defer quantizedExpertIDMatVecKernelCache.Unlock()
	if quantizedExpertIDMatVecKernelCache.kernels == nil {
		quantizedExpertIDMatVecKernelCache.kernels = make(map[quantizedExpertIDMatVecKernelKey]*MetalKernel)
	}
	if kernel := quantizedExpertIDMatVecKernelCache.kernels[key]; kernel != nil {
		return kernel
	}

	inputBase := quantizedExpertIDMatVecInputBase(meta)
	source := core.Sprintf(`uint simd_elem = thread_position_in_grid.x / 32u;
uint out_col = simd_elem %% uint(%d);
uint route = simd_elem / uint(%d);
uint expert = uint(expert_ids[route]);
uint lane = thread_index_in_simdgroup;
float sum = 0.0f;
for (uint pack_col = lane; pack_col < uint(%d); pack_col += 32u) {
	uint pack_index = (expert * uint(%d) + out_col) * uint(%d) + pack_col;
	uint packed = weight[pack_index];
	uint base_in = pack_col * uint(%d);
	for (uint packed_offset = 0; packed_offset < uint(%d); packed_offset++) {
		uint in_col = base_in + packed_offset;
		uint bit_shift = packed_offset * uint(%d);
		uint q = (packed >> bit_shift) & uint(%d);
		uint group = in_col / uint(%d);
		uint scale_index = (expert * uint(%d) + out_col) * uint(%d) + group;
		float w = float(q) * float(scales[scale_index]) + float(qbiases[scale_index]);
		sum += x[%s + in_col] * w;
	}
}
sum = simd_sum(sum);
if (lane == 0u) {
	out[simd_elem] = sum;
}`,
		meta.outDim,
		meta.outDim,
		meta.packedIn,
		meta.outDim,
		meta.packedIn,
		meta.packFactor,
		meta.packFactor,
		bits,
		(1<<bits)-1,
		groupSize,
		meta.outDim,
		meta.groups,
		inputBase,
	)
	header := "#include <metal_stdlib>\n#include <metal_simdgroup>\nusing namespace metal;\n"

	kernel := NewMetalKernel(
		core.Sprintf("quantized_expert_id_matvec_b%d_g%d_r%d_i%d_o%d_p%d_s%d_sh%t", bits, groupSize, meta.routes, meta.inDim, meta.outDim, meta.packedIn, meta.sidecarDType, meta.sharedInput),
		[]string{"x", "weight", "scales", "qbiases", "expert_ids"},
		[]string{"out"},
		source,
		header,
		true,
		false,
	)
	quantizedExpertIDMatVecKernelCache.kernels[key] = kernel
	return kernel
}

func quantizedExpertIDGELUGateUpMatVecKernel(meta quantizedExpertIDMatVecMeta, groupSize, bits int) *MetalKernel {
	key := quantizedExpertIDMatVecKernelKey{
		bits:         bits,
		groupSize:    groupSize,
		routes:       meta.routes,
		inDim:        meta.inDim,
		outDim:       meta.outDim,
		packedIn:     meta.packedIn,
		sidecarDType: meta.sidecarDType,
		sharedInput:  meta.sharedInput,
	}
	quantizedExpertIDGELUGateUpMatVecKernelCache.Lock()
	defer quantizedExpertIDGELUGateUpMatVecKernelCache.Unlock()
	if quantizedExpertIDGELUGateUpMatVecKernelCache.kernels == nil {
		quantizedExpertIDGELUGateUpMatVecKernelCache.kernels = make(map[quantizedExpertIDMatVecKernelKey]*MetalKernel)
	}
	if kernel := quantizedExpertIDGELUGateUpMatVecKernelCache.kernels[key]; kernel != nil {
		return kernel
	}

	halfOut := meta.outDim / 2
	inputBase := quantizedExpertIDMatVecInputBase(meta)
	source := core.Sprintf(`uint simd_elem = thread_position_in_grid.x / 32u;
uint out_col = simd_elem %% uint(%d);
uint route = simd_elem / uint(%d);
uint expert = uint(expert_ids[route]);
uint lane = thread_index_in_simdgroup;
float gate_sum = 0.0f;
float up_sum = 0.0f;
for (uint pack_col = lane; pack_col < uint(%d); pack_col += 32u) {
	uint gate_pack_index = (expert * uint(%d) + out_col) * uint(%d) + pack_col;
	uint up_pack_index = (expert * uint(%d) + out_col + uint(%d)) * uint(%d) + pack_col;
	uint gate_packed = weight[gate_pack_index];
	uint up_packed = weight[up_pack_index];
	uint base_in = pack_col * uint(%d);
	for (uint packed_offset = 0; packed_offset < uint(%d); packed_offset++) {
		uint in_col = base_in + packed_offset;
		uint bit_shift = packed_offset * uint(%d);
		uint group = in_col / uint(%d);
		uint gate_q = (gate_packed >> bit_shift) & uint(%d);
		uint up_q = (up_packed >> bit_shift) & uint(%d);
		uint gate_scale_index = (expert * uint(%d) + out_col) * uint(%d) + group;
		uint up_scale_index = (expert * uint(%d) + out_col + uint(%d)) * uint(%d) + group;
		float gate_w = float(gate_q) * float(scales[gate_scale_index]) + float(qbiases[gate_scale_index]);
		float up_w = float(up_q) * float(scales[up_scale_index]) + float(qbiases[up_scale_index]);
		float input_value = x[%s + in_col];
		gate_sum += input_value * gate_w;
		up_sum += input_value * up_w;
	}
}
gate_sum = simd_sum(gate_sum);
up_sum = simd_sum(up_sum);
if (lane == 0u) {
	float gate_cube = gate_sum * gate_sum * gate_sum;
	float gelu = 0.5f * gate_sum * (1.0f + tanh(0.7978845608028654f * (gate_sum + 0.044715f * gate_cube)));
	out[simd_elem] = gelu * up_sum;
}`,
		halfOut,
		halfOut,
		meta.packedIn,
		meta.outDim,
		meta.packedIn,
		meta.outDim,
		halfOut,
		meta.packedIn,
		meta.packFactor,
		meta.packFactor,
		bits,
		groupSize,
		(1<<bits)-1,
		(1<<bits)-1,
		meta.outDim,
		meta.groups,
		meta.outDim,
		halfOut,
		meta.groups,
		inputBase,
	)
	header := "#include <metal_stdlib>\n#include <metal_simdgroup>\nusing namespace metal;\n"

	kernel := NewMetalKernel(
		core.Sprintf("quantized_expert_id_gelu_gate_up_matvec_b%d_g%d_r%d_i%d_o%d_p%d_s%d_sh%t", bits, groupSize, meta.routes, meta.inDim, meta.outDim, meta.packedIn, meta.sidecarDType, meta.sharedInput),
		[]string{"x", "weight", "scales", "qbiases", "expert_ids"},
		[]string{"out"},
		source,
		header,
		true,
		false,
	)
	quantizedExpertIDGELUGateUpMatVecKernelCache.kernels[key] = kernel
	return kernel
}

func quantizedExpertIDGELUSplitGateUpMatVecKernel(meta quantizedExpertIDMatVecMeta, groupSize, bits int) *MetalKernel {
	unrolledQ4 := expertIDUnrolledQ4Enabled(bits)
	key := quantizedExpertIDMatVecKernelKey{
		bits:         bits,
		groupSize:    groupSize,
		routes:       meta.routes,
		inDim:        meta.inDim,
		outDim:       meta.outDim,
		packedIn:     meta.packedIn,
		sidecarDType: meta.sidecarDType,
		sharedInput:  meta.sharedInput,
		unrolledQ4:   unrolledQ4,
	}
	quantizedExpertIDGELUSplitGateUpMatVecKernelCache.Lock()
	defer quantizedExpertIDGELUSplitGateUpMatVecKernelCache.Unlock()
	if quantizedExpertIDGELUSplitGateUpMatVecKernelCache.kernels == nil {
		quantizedExpertIDGELUSplitGateUpMatVecKernelCache.kernels = make(map[quantizedExpertIDMatVecKernelKey]*MetalKernel)
	}
	if kernel := quantizedExpertIDGELUSplitGateUpMatVecKernelCache.kernels[key]; kernel != nil {
		return kernel
	}

	inputBase := quantizedExpertIDMatVecInputBase(meta)
	source := core.Sprintf(`uint simd_elem = thread_position_in_grid.x / 32u;
uint out_col = simd_elem %% uint(%d);
uint route = simd_elem / uint(%d);
uint expert = uint(expert_ids[route]);
uint lane = thread_index_in_simdgroup;
float gate_sum = 0.0f;
float up_sum = 0.0f;
for (uint pack_col = lane; pack_col < uint(%d); pack_col += 32u) {
	uint pack_index = (expert * uint(%d) + out_col) * uint(%d) + pack_col;
	uint gate_packed = gate_weight[pack_index];
	uint up_packed = up_weight[pack_index];
	uint base_in = pack_col * uint(%d);
	for (uint packed_offset = 0; packed_offset < uint(%d); packed_offset++) {
		uint in_col = base_in + packed_offset;
		uint bit_shift = packed_offset * uint(%d);
		uint group = in_col / uint(%d);
		uint gate_q = (gate_packed >> bit_shift) & uint(%d);
		uint up_q = (up_packed >> bit_shift) & uint(%d);
		uint scale_index = (expert * uint(%d) + out_col) * uint(%d) + group;
		float gate_w = float(gate_q) * float(gate_scales[scale_index]) + float(gate_qbiases[scale_index]);
		float up_w = float(up_q) * float(up_scales[scale_index]) + float(up_qbiases[scale_index]);
		float input_value = x[%s + in_col];
		gate_sum += input_value * gate_w;
		up_sum += input_value * up_w;
	}
}
gate_sum = simd_sum(gate_sum);
up_sum = simd_sum(up_sum);
if (lane == 0u) {
	float gate_cube = gate_sum * gate_sum * gate_sum;
	float gelu = 0.5f * gate_sum * (1.0f + tanh(0.7978845608028654f * (gate_sum + 0.044715f * gate_cube)));
	out[simd_elem] = gelu * up_sum;
}`,
		meta.outDim,
		meta.outDim,
		meta.packedIn,
		meta.outDim,
		meta.packedIn,
		meta.packFactor,
		meta.packFactor,
		bits,
		groupSize,
		(1<<bits)-1,
		(1<<bits)-1,
		meta.outDim,
		meta.groups,
		inputBase,
	)
	if unrolledQ4 {
		source = quantizedExpertIDGELUSplitGateUpMatVecKernelQ4Source(meta, groupSize, inputBase)
	}
	header := "#include <metal_stdlib>\n#include <metal_simdgroup>\nusing namespace metal;\n"

	kernel := NewMetalKernel(
		core.Sprintf("quantized_expert_id_gelu_split_gate_up_matvec_b%d_g%d_r%d_i%d_o%d_p%d_s%d_sh%t_u%t", bits, groupSize, meta.routes, meta.inDim, meta.outDim, meta.packedIn, meta.sidecarDType, meta.sharedInput, unrolledQ4),
		[]string{"x", "gate_weight", "gate_scales", "gate_qbiases", "up_weight", "up_scales", "up_qbiases", "expert_ids"},
		[]string{"out"},
		source,
		header,
		true,
		false,
	)
	quantizedExpertIDGELUSplitGateUpMatVecKernelCache.kernels[key] = kernel
	return kernel
}

func quantizedExpertIDWeightedMatVecSumKernel(meta quantizedExpertIDMatVecMeta, groupSize, bits int) *MetalKernel {
	unrolledQ4 := expertIDUnrolledQ4Enabled(bits)
	key := quantizedExpertIDMatVecKernelKey{
		bits:         bits,
		groupSize:    groupSize,
		routes:       meta.routes,
		inDim:        meta.inDim,
		outDim:       meta.outDim,
		packedIn:     meta.packedIn,
		sidecarDType: meta.sidecarDType,
		sharedInput:  meta.sharedInput,
		unrolledQ4:   unrolledQ4,
	}
	quantizedExpertIDWeightedMatVecSumKernelCache.Lock()
	defer quantizedExpertIDWeightedMatVecSumKernelCache.Unlock()
	if quantizedExpertIDWeightedMatVecSumKernelCache.kernels == nil {
		quantizedExpertIDWeightedMatVecSumKernelCache.kernels = make(map[quantizedExpertIDMatVecKernelKey]*MetalKernel)
	}
	if kernel := quantizedExpertIDWeightedMatVecSumKernelCache.kernels[key]; kernel != nil {
		return kernel
	}

	inputBase := quantizedExpertIDMatVecInputBase(meta)
	source := core.Sprintf(`uint out_col = thread_position_in_grid.x / 32u;
	uint lane = thread_index_in_simdgroup;
	float sum = 0.0f;
	for (uint route = 0; route < uint(%d); route++) {
		uint expert = uint(expert_ids[route]);
		float route_weight = route_weights[route];
		for (uint pack_col = lane; pack_col < uint(%d); pack_col += 32u) {
			uint pack_index = (expert * uint(%d) + out_col) * uint(%d) + pack_col;
			uint packed = weight[pack_index];
			uint base_in = pack_col * uint(%d);
			for (uint packed_offset = 0; packed_offset < uint(%d); packed_offset++) {
				uint in_col = base_in + packed_offset;
				uint bit_shift = packed_offset * uint(%d);
				uint q = (packed >> bit_shift) & uint(%d);
				uint group = in_col / uint(%d);
				uint scale_index = (expert * uint(%d) + out_col) * uint(%d) + group;
				float w = float(q) * float(scales[scale_index]) + float(qbiases[scale_index]);
				sum += route_weight * x[%s + in_col] * w;
			}
		}
	}
	sum = simd_sum(sum);
	if (lane == 0u) {
		out[out_col] = sum;
	}`,
		meta.routes,
		meta.packedIn,
		meta.outDim,
		meta.packedIn,
		meta.packFactor,
		meta.packFactor,
		bits,
		(1<<bits)-1,
		groupSize,
		meta.outDim,
		meta.groups,
		inputBase,
	)
	if unrolledQ4 {
		source = quantizedExpertIDWeightedMatVecSumKernelQ4Source(meta, groupSize, inputBase)
	}
	header := "#include <metal_stdlib>\n#include <metal_simdgroup>\nusing namespace metal;\n"

	kernel := NewMetalKernel(
		core.Sprintf("quantized_expert_id_weighted_matvec_sum_b%d_g%d_r%d_i%d_o%d_p%d_s%d_sh%t_u%t", bits, groupSize, meta.routes, meta.inDim, meta.outDim, meta.packedIn, meta.sidecarDType, meta.sharedInput, unrolledQ4),
		[]string{"x", "route_weights", "weight", "scales", "qbiases", "expert_ids"},
		[]string{"out"},
		source,
		header,
		true,
		false,
	)
	quantizedExpertIDWeightedMatVecSumKernelCache.kernels[key] = kernel
	return kernel
}

func expertIDUnrolledQ4Enabled(bits int) bool {
	return bits == 4 && expertIDUnrolledQ4RuntimeEnabled()
}

func quantizedExpertIDGELUSplitGateUpMatVecKernelQ4Source(meta quantizedExpertIDMatVecMeta, groupSize int, inputBase string) string {
	return core.Sprintf(`uint simd_elem = thread_position_in_grid.x / 32u;
uint out_col = simd_elem %% uint(%d);
uint route = simd_elem / uint(%d);
uint expert = uint(expert_ids[route]);
uint lane = thread_index_in_simdgroup;
float gate_sum = 0.0f;
float up_sum = 0.0f;
for (uint pack_col = lane; pack_col < uint(%d); pack_col += 32u) {
	uint pack_index = (expert * uint(%d) + out_col) * uint(%d) + pack_col;
	uint gate_packed = gate_weight[pack_index];
	uint up_packed = up_weight[pack_index];
	uint base_in = pack_col * 8u;
%s
}
gate_sum = simd_sum(gate_sum);
up_sum = simd_sum(up_sum);
if (lane == 0u) {
	float gate_cube = gate_sum * gate_sum * gate_sum;
	float gelu = 0.5f * gate_sum * (1.0f + tanh(0.7978845608028654f * (gate_sum + 0.044715f * gate_cube)));
	out[simd_elem] = gelu * up_sum;
}`,
		meta.outDim,
		meta.outDim,
		meta.packedIn,
		meta.outDim,
		meta.packedIn,
		quantizedExpertIDGELUSplitGateUpMatVecKernelQ4Body(meta, groupSize, inputBase),
	)
}

func quantizedExpertIDGELUSplitGateUpMatVecKernelQ4Body(meta quantizedExpertIDMatVecMeta, groupSize int, inputBase string) string {
	parts := make([]string, 0, 8)
	for offset := 0; offset < 8; offset++ {
		parts = append(parts, core.Sprintf(`	{
		uint in_col = base_in + uint(%d);
		uint group = in_col / uint(%d);
		uint gate_q = (gate_packed >> uint(%d)) & 15u;
		uint up_q = (up_packed >> uint(%d)) & 15u;
		uint scale_index = (expert * uint(%d) + out_col) * uint(%d) + group;
		float gate_w = float(gate_q) * float(gate_scales[scale_index]) + float(gate_qbiases[scale_index]);
		float up_w = float(up_q) * float(up_scales[scale_index]) + float(up_qbiases[scale_index]);
		float input_value = x[%s + in_col];
		gate_sum += input_value * gate_w;
		up_sum += input_value * up_w;
	}`,
			offset,
			groupSize,
			offset*4,
			offset*4,
			meta.outDim,
			meta.groups,
			inputBase,
		))
	}
	return core.Join("\n", parts...)
}

func quantizedExpertIDWeightedMatVecSumKernelQ4Source(meta quantizedExpertIDMatVecMeta, groupSize int, inputBase string) string {
	return core.Sprintf(`uint out_col = thread_position_in_grid.x / 32u;
uint lane = thread_index_in_simdgroup;
float sum = 0.0f;
for (uint route = 0; route < uint(%d); route++) {
	uint expert = uint(expert_ids[route]);
	float route_weight = route_weights[route];
	for (uint pack_col = lane; pack_col < uint(%d); pack_col += 32u) {
		uint pack_index = (expert * uint(%d) + out_col) * uint(%d) + pack_col;
		uint packed = weight[pack_index];
		uint base_in = pack_col * 8u;
%s
	}
}
sum = simd_sum(sum);
if (lane == 0u) {
	out[out_col] = sum;
}`,
		meta.routes,
		meta.packedIn,
		meta.outDim,
		meta.packedIn,
		quantizedExpertIDWeightedMatVecSumKernelQ4Body(meta, groupSize, inputBase),
	)
}

func quantizedExpertIDWeightedMatVecSumKernelQ4Body(meta quantizedExpertIDMatVecMeta, groupSize int, inputBase string) string {
	parts := make([]string, 0, 8)
	for offset := 0; offset < 8; offset++ {
		parts = append(parts, core.Sprintf(`		{
			uint in_col = base_in + uint(%d);
			uint q = (packed >> uint(%d)) & 15u;
			uint group = in_col / uint(%d);
			uint scale_index = (expert * uint(%d) + out_col) * uint(%d) + group;
			float w = float(q) * float(scales[scale_index]) + float(qbiases[scale_index]);
			sum += route_weight * x[%s + in_col] * w;
		}`,
			offset,
			offset*4,
			groupSize,
			meta.outDim,
			meta.groups,
			inputBase,
		))
	}
	return core.Join("\n", parts...)
}

type quantizedExpertIDMatVecMeta struct {
	routes       int
	inputRows    int
	experts      int
	outDim       int
	inDim        int
	packedIn     int
	groups       int
	packFactor   int
	sidecarDType DType
	sharedInput  bool
}

func validateQuantizedExpertIDMatVec(input, weight, scales, biases, expertIDs *Array, groupSize, bits int) (quantizedExpertIDMatVecMeta, error) {
	var meta quantizedExpertIDMatVecMeta
	if input == nil || !input.Valid() {
		return meta, errEIMNeedInput
	}
	if weight == nil || !weight.Valid() || scales == nil || !scales.Valid() || biases == nil || !biases.Valid() {
		return meta, errEIMNeedWeightScalesBiases
	}
	if expertIDs == nil || !expertIDs.Valid() {
		return meta, errEIMNeedExpertIDs
	}
	if input.Dtype() != DTypeFloat32 {
		return meta, errEIMInputDtype
	}
	if weight.Dtype() != DTypeUint32 {
		return meta, errEIMWeightDtype
	}
	// Resolve dtypes once per validation — Array.Dtype() is a cgo call,
	// and the old code re-resolved scales/expertIDs dtypes up to 4 times.
	// W9-K stream-resolve-once-per-call analogue for the MoE hot path.
	scalesDtype := scales.Dtype()
	biasesDtype := biases.Dtype()
	if scalesDtype != biasesDtype {
		return meta, core.NewError(core.Sprintf("mlx: quantized expert id matvec scales and biases dtype mismatch: %v/%v", scalesDtype, biasesDtype))
	}
	switch scalesDtype {
	case DTypeFloat32, DTypeFloat16, DTypeBFloat16:
		meta.sidecarDType = scalesDtype
	default:
		return meta, errEIMScalesBiasesDtype
	}
	expertIDsDtype := expertIDs.Dtype()
	if expertIDsDtype != DTypeInt32 && expertIDsDtype != DTypeUint32 {
		return meta, errEIMExpertIDsDtype
	}
	if bits != 2 && bits != 4 && bits != 8 {
		return meta, core.NewError(core.Sprintf("mlx: quantized expert id matvec unsupported bits %d", bits))
	}
	if groupSize <= 0 {
		return meta, errEIMGroupSizeInvalid
	}
	// Read dimensions via direct Dim(i) cgo to avoid Shape()'s per-call
	// []int32 heap allocation on the MoE-decode hot path. Error paths
	// still call Shape() to format the diagnostic — those only fire on
	// misuse and do not allocate on the happy path. W9-K, Wave 9.
	if input.NumDims() != 2 {
		return meta, core.NewError(core.Sprintf("mlx: quantized expert id matvec input shape %v, expected [routes, in]", input.Shape()))
	}
	if weight.NumDims() != 3 {
		return meta, core.NewError(core.Sprintf("mlx: quantized expert id matvec weight shape %v, expected [experts, out, packed_in]", weight.Shape()))
	}
	if scales.NumDims() != 3 || biases.NumDims() != 3 {
		return meta, errEIMScalesBiasesShape
	}

	meta.inputRows = input.Dim(0)
	meta.routes = expertIDs.Size()
	meta.inDim = input.Dim(1)
	meta.experts = weight.Dim(0)
	meta.outDim = weight.Dim(1)
	meta.packedIn = weight.Dim(2)
	meta.packFactor = 32 / bits
	meta.groups = meta.inDim / groupSize
	meta.sharedInput = meta.inputRows == 1 && meta.routes > 1
	if meta.inputRows <= 0 || meta.routes <= 0 || meta.inDim <= 0 || meta.experts <= 0 || meta.outDim <= 0 || meta.packedIn <= 0 {
		return meta, errEIMDimsInvalid
	}
	if meta.inputRows != 1 && meta.inputRows != meta.routes {
		return meta, core.NewError(core.Sprintf("mlx: quantized expert id matvec input row count %d must be 1 or match expert id count %d", meta.inputRows, meta.routes))
	}
	if meta.inDim%groupSize != 0 {
		return meta, core.NewError(core.Sprintf("mlx: quantized expert id matvec input dim %d must divide by group size %d", meta.inDim, groupSize))
	}
	if meta.packedIn*meta.packFactor != meta.inDim {
		return meta, core.NewError(core.Sprintf("mlx: quantized expert id matvec packed input dim %d expands to %d, expected %d", meta.packedIn, meta.packedIn*meta.packFactor, meta.inDim))
	}
	if scales.Dim(0) != meta.experts || scales.Dim(1) != meta.outDim || scales.Dim(2) != meta.groups ||
		biases.Dim(0) != meta.experts || biases.Dim(1) != meta.outDim || biases.Dim(2) != meta.groups {
		return meta, core.NewError(core.Sprintf("mlx: quantized expert id matvec scale/bias shape = %v/%v, expected [%d %d %d]", scales.Shape(), biases.Shape(), meta.experts, meta.outDim, meta.groups))
	}
	return meta, nil
}

func quantizedExpertIDMatVecInputBase(meta quantizedExpertIDMatVecMeta) string {
	if meta.sharedInput {
		return "0u"
	}
	return core.Sprintf("route * uint(%d)", meta.inDim)
}

