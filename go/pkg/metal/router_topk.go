// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"sync"

	core "dappco.re/go"
)

// The router gates carry no init-time package var — their value is the runtime
// gate the model's EngineFeatures.Apply sets, so a clear is honoured rather than
// frozen at boot. (#55 slice 3b)

func nativeMoERouterProjectionScores(input *Array, router MoERouterProjection) (*Array, bool, error) {
	return nativeMoERouterMatVecScores(input, router.Linear())
}

func nativeMoERouterMatVecScores(input *Array, proj *Linear) (*Array, bool, error) {
	meta, ok, err := validateNativeMoERouterMatVec(input, proj)
	if err != nil || !ok {
		return nil, ok, err
	}

	kernel := nativeMoERouterMatVecKernel(meta, proj.GroupSize, proj.Bits)

	out, err := kernel.DispatchOne(
		MetalKernelGrid{GridX: meta.outDim * 32, GridY: 1, GridZ: 1, TGX: 256, TGY: 1, TGZ: 1},
		[]int32{1, 1, int32(meta.outDim)}, DTypeFloat32,
		input, proj.Weight, proj.Scales, proj.Biases,
	)
	if err != nil {
		return nil, true, core.E("mlx.nativeMoERouterMatVecScores", "apply Metal kernel", err)
	}
	return out, true, nil
}

type nativeMoERouterMatVecMeta struct {
	inDim        int
	outDim       int
	packedIn     int
	groups       int
	packFactor   int
	sidecarDType DType
}

func validateNativeMoERouterMatVec(input *Array, proj *Linear) (nativeMoERouterMatVecMeta, bool, error) {
	var meta nativeMoERouterMatVecMeta
	if input == nil || !input.Valid() || proj == nil || proj.LoRA != nil {
		return meta, false, nil
	}
	if proj.Weight == nil || !proj.Weight.Valid() || proj.Scales == nil || !proj.Scales.Valid() || proj.Biases == nil || !proj.Biases.Valid() {
		return meta, false, nil
	}
	if proj.Bias != nil && proj.Bias.Valid() {
		return meta, false, nil
	}
	if proj.GroupSize <= 0 || (proj.Bits != 4 && proj.Bits != 8) {
		return meta, false, nil
	}
	shape := input.Shape()
	weightShape := proj.Weight.Shape()
	scaleShape := proj.Scales.Shape()
	biasShape := proj.Biases.Shape()
	if len(shape) != 3 || shape[0] != 1 || shape[1] != 1 || len(weightShape) != 2 || len(scaleShape) != 2 || len(biasShape) != 2 {
		return meta, false, nil
	}
	packFactor := 32 / proj.Bits
	if packFactor <= 0 {
		return meta, false, nil
	}
	inDim := int(shape[2])
	outDim := int(weightShape[0])
	packedIn := int(weightShape[1])
	groups := inDim / proj.GroupSize
	if inDim <= 0 || outDim <= 0 || packedIn <= 0 || groups <= 0 || inDim%proj.GroupSize != 0 || packedIn*packFactor != inDim {
		return meta, false, nil
	}
	if int(scaleShape[0]) != outDim || int(scaleShape[1]) != groups || int(biasShape[0]) != outDim || int(biasShape[1]) != groups {
		return meta, false, nil
	}
	if proj.Scales.Dtype() != proj.Biases.Dtype() {
		return meta, false, nil
	}
	return nativeMoERouterMatVecMeta{
		inDim:        inDim,
		outDim:       outDim,
		packedIn:     packedIn,
		groups:       groups,
		packFactor:   packFactor,
		sidecarDType: proj.Scales.Dtype(),
	}, true, nil
}

type nativeMoERouterMatVecKernelKey struct {
	bits         int
	groupSize    int
	inDim        int
	outDim       int
	packedIn     int
	sidecarDType DType
}

var nativeMoERouterMatVecKernelCache struct {
	sync.Mutex
	kernels map[nativeMoERouterMatVecKernelKey]*MetalKernel
}

func nativeMoERouterMatVecKernel(meta nativeMoERouterMatVecMeta, groupSize, bits int) *MetalKernel {
	key := nativeMoERouterMatVecKernelKey{
		bits:         bits,
		groupSize:    groupSize,
		inDim:        meta.inDim,
		outDim:       meta.outDim,
		packedIn:     meta.packedIn,
		sidecarDType: meta.sidecarDType,
	}
	nativeMoERouterMatVecKernelCache.Lock()
	defer nativeMoERouterMatVecKernelCache.Unlock()
	if nativeMoERouterMatVecKernelCache.kernels == nil {
		nativeMoERouterMatVecKernelCache.kernels = make(map[nativeMoERouterMatVecKernelKey]*MetalKernel)
	}
	if kernel := nativeMoERouterMatVecKernelCache.kernels[key]; kernel != nil {
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
		core.Sprintf("moe_router_matvec_b%d_g%d_i%d_o%d_p%d_s%d", bits, groupSize, meta.inDim, meta.outDim, meta.packedIn, meta.sidecarDType),
		[]string{"x", "weight", "scales", "qbiases"},
		[]string{"out"},
		source,
		header,
		true,
		false,
	)
	nativeMoERouterMatVecKernelCache.kernels[key] = kernel
	return kernel
}

func nativeMoERouterTopK(scores, perExpertScale *Array, topK int) (*Array, *Array, bool, error) {
	if perExpertScale == nil || !perExpertScale.Valid() {
		return nativeMoERouterTopKUnitScale(scores, topK)
	}
	if scores == nil || !scores.Valid() {
		return nil, nil, false, nil
	}
	if scores.Dtype() != DTypeFloat32 || perExpertScale.Dtype() != DTypeFloat32 {
		return nil, nil, false, nil
	}
	shape := scores.Shape()
	if len(shape) != 3 || shape[0] != 1 || shape[1] != 1 {
		return nil, nil, false, nil
	}
	experts := int(shape[2])
	if experts <= 0 || topK <= 0 || topK > experts || topK > 32 {
		return nil, nil, false, nil
	}
	if perExpertScale.Size() != experts {
		return nil, nil, false, nil
	}

	kernel := nativeMoERouterTopKKernel(experts, topK)
	cfg := NewMetalKernelConfig()
	defer cfg.Free()
	cfg.SetGrid(1, 1, 1)
	cfg.SetThreadGroup(1, 1, 1)
	outShape := []int32{1, 1, int32(topK)}
	cfg.AddOutputArg(outShape, DTypeInt32)
	cfg.AddOutputArg(outShape, DTypeFloat32)

	results, err := kernel.Apply(cfg, scores, perExpertScale)
	if err != nil {
		return nil, nil, true, core.E("mlx.nativeMoERouterTopK", "apply Metal kernel", err)
	}
	if len(results) != 2 {
		Free(results...)
		return nil, nil, true, core.NewError(core.Sprintf("mlx: native MoE router top-k returned %d outputs, expected 2", len(results)))
	}
	return results[0], results[1], true, nil
}

func nativeMoERouterTopKUnitScale(scores *Array, topK int) (*Array, *Array, bool, error) {
	if scores == nil || !scores.Valid() {
		return nil, nil, false, nil
	}
	if scores.Dtype() != DTypeFloat32 {
		return nil, nil, false, nil
	}
	shape := scores.Shape()
	if len(shape) != 3 || shape[0] != 1 || shape[1] != 1 {
		return nil, nil, false, nil
	}
	experts := int(shape[2])
	if experts <= 0 || topK <= 0 || topK > experts || topK > 32 {
		return nil, nil, false, nil
	}

	kernel := nativeMoERouterTopKUnitScaleKernel(experts, topK)
	cfg := NewMetalKernelConfig()
	defer cfg.Free()
	cfg.SetGrid(1, 1, 1)
	cfg.SetThreadGroup(1, 1, 1)
	outShape := []int32{1, 1, int32(topK)}
	cfg.AddOutputArg(outShape, DTypeInt32)
	cfg.AddOutputArg(outShape, DTypeFloat32)

	results, err := kernel.Apply(cfg, scores)
	if err != nil {
		return nil, nil, true, core.E("mlx.nativeMoERouterTopKUnitScale", "apply Metal kernel", err)
	}
	if len(results) != 2 {
		Free(results...)
		return nil, nil, true, core.NewError(core.Sprintf("mlx: native MoE router unit-scale top-k returned %d outputs, expected 2", len(results)))
	}
	return results[0], results[1], true, nil
}

type nativeMoERouterTopKKernelKey struct {
	experts int
	topK    int
}

var nativeMoERouterTopKKernelCache struct {
	sync.Mutex
	kernels map[nativeMoERouterTopKKernelKey]*MetalKernel
}

var nativeMoERouterTopKUnitScaleKernelCache struct {
	sync.Mutex
	kernels map[nativeMoERouterTopKKernelKey]*MetalKernel
}

func nativeMoERouterTopKKernel(experts, topK int) *MetalKernel {
	key := nativeMoERouterTopKKernelKey{experts: experts, topK: topK}
	nativeMoERouterTopKKernelCache.Lock()
	defer nativeMoERouterTopKKernelCache.Unlock()
	if nativeMoERouterTopKKernelCache.kernels == nil {
		nativeMoERouterTopKKernelCache.kernels = make(map[nativeMoERouterTopKKernelKey]*MetalKernel)
	}
	if kernel := nativeMoERouterTopKKernelCache.kernels[key]; kernel != nil {
		return kernel
	}

	source := core.Sprintf(`float best_values[%d];
uint best_indices[%d];
for (uint i = 0; i < uint(%d); i++) {
	best_values[i] = -3.402823466e+38f;
	best_indices[i] = 0u;
}
for (uint expert = 0; expert < uint(%d); expert++) {
	float score = float(scores[expert]);
	for (uint slot = 0; slot < uint(%d); slot++) {
		bool better = score > best_values[slot] || (score == best_values[slot] && expert < best_indices[slot]);
		if (!better) {
			continue;
		}
		for (uint move = uint(%d) - 1u; move > slot; move--) {
			best_values[move] = best_values[move - 1u];
			best_indices[move] = best_indices[move - 1u];
		}
		best_values[slot] = score;
		best_indices[slot] = expert;
		break;
	}
}
float max_value = best_values[0];
float denom = 0.0f;
for (uint i = 0; i < uint(%d); i++) {
	denom += exp(best_values[i] - max_value);
}
for (uint i = 0; i < uint(%d); i++) {
	uint expert = best_indices[i];
	float weight = exp(best_values[i] - max_value) / denom;
	top_indices[i] = int(expert);
	top_weights[i] = weight * float(per_expert_scale[expert]);
}`,
		topK,
		topK,
		topK,
		experts,
		topK,
		topK,
		topK,
		topK,
	)
	header := "#include <metal_stdlib>\nusing namespace metal;\n"
	kernel := NewMetalKernel(
		core.Sprintf("moe_router_topk_e%d_k%d", experts, topK),
		[]string{"scores", "per_expert_scale"},
		[]string{"top_indices", "top_weights"},
		source,
		header,
		true,
		false,
	)
	nativeMoERouterTopKKernelCache.kernels[key] = kernel
	return kernel
}

func nativeMoERouterTopKUnitScaleKernel(experts, topK int) *MetalKernel {
	key := nativeMoERouterTopKKernelKey{experts: experts, topK: topK}
	nativeMoERouterTopKUnitScaleKernelCache.Lock()
	defer nativeMoERouterTopKUnitScaleKernelCache.Unlock()
	if nativeMoERouterTopKUnitScaleKernelCache.kernels == nil {
		nativeMoERouterTopKUnitScaleKernelCache.kernels = make(map[nativeMoERouterTopKKernelKey]*MetalKernel)
	}
	if kernel := nativeMoERouterTopKUnitScaleKernelCache.kernels[key]; kernel != nil {
		return kernel
	}

	source := core.Sprintf(`float best_values[%d];
uint best_indices[%d];
for (uint i = 0; i < uint(%d); i++) {
	best_values[i] = -3.402823466e+38f;
	best_indices[i] = 0u;
}
for (uint expert = 0; expert < uint(%d); expert++) {
	float score = float(scores[expert]);
	for (uint slot = 0; slot < uint(%d); slot++) {
		bool better = score > best_values[slot] || (score == best_values[slot] && expert < best_indices[slot]);
		if (!better) {
			continue;
		}
		for (uint move = uint(%d) - 1u; move > slot; move--) {
			best_values[move] = best_values[move - 1u];
			best_indices[move] = best_indices[move - 1u];
		}
		best_values[slot] = score;
		best_indices[slot] = expert;
		break;
	}
}
float max_value = best_values[0];
float denom = 0.0f;
for (uint i = 0; i < uint(%d); i++) {
	denom += exp(best_values[i] - max_value);
}
for (uint i = 0; i < uint(%d); i++) {
	top_indices[i] = int(best_indices[i]);
	top_weights[i] = exp(best_values[i] - max_value) / denom;
}`,
		topK,
		topK,
		topK,
		experts,
		topK,
		topK,
		topK,
		topK,
	)
	header := "#include <metal_stdlib>\nusing namespace metal;\n"
	kernel := NewMetalKernel(
		core.Sprintf("moe_router_topk_unit_e%d_k%d", experts, topK),
		[]string{"scores"},
		[]string{"top_indices", "top_weights"},
		source,
		header,
		true,
		false,
	)
	nativeMoERouterTopKUnitScaleKernelCache.kernels[key] = kernel
	return kernel
}
