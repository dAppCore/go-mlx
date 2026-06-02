// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"sync"

	core "dappco.re/go"
)

var enableNativeGemma4RouterTopK = core.Env("GO_MLX_ENABLE_NATIVE_GEMMA4_ROUTER_TOPK") == "1"
var enableNativeGemma4RouterMatVec = core.Env("GO_MLX_ENABLE_NATIVE_GEMMA4_ROUTER_MATVEC") == "1"

func nativeGemma4RouterTopKEnabled() bool {
	return enableNativeGemma4RouterTopK || nativeGemma4RouterTopKRuntimeEnabled()
}

func nativeGemma4RouterMatVecEnabled() bool {
	return enableNativeGemma4RouterMatVec || nativeGemma4RouterMatVecRuntimeEnabled()
}

func nativeGemma4RouterMatVecScores(input *Array, proj *Linear) (*Array, bool, error) {
	if !nativeGemma4RouterMatVecEnabled() {
		return nil, false, nil
	}
	return nativeMoERouterMatVecScores(input, proj)
}

// MoERouterProjection is the model-family neutral representation of a router
// projection. Qwen, Mixtral, GPT-OSS, Kimi, Gemma 4, and MiniMax wrap this
// shape differently in their loaders, but the per-token projection is the same
// quantized hidden -> expert-score matvec.
type MoERouterProjection struct {
	Weight    *Array
	Scales    *Array
	Biases    *Array
	GroupSize int
	Bits      int
}

func (r MoERouterProjection) Linear() *Linear {
	if r.Weight == nil || !r.Weight.Valid() {
		return nil
	}
	if r.Scales != nil && r.Scales.Valid() {
		return NewQuantizedLinear(r.Weight, r.Scales, r.Biases, nil, r.GroupSize, r.Bits)
	}
	return NewLinear(r.Weight, nil)
}

func nativeMoERouterProjectionScores(input *Array, router MoERouterProjection) (*Array, bool, error) {
	return nativeMoERouterMatVecScores(input, router.Linear())
}

func nativeMoERouterMatVecScores(input *Array, proj *Linear) (*Array, bool, error) {
	meta, ok, err := validateNativeGemma4RouterMatVec(input, proj)
	if err != nil || !ok {
		return nil, ok, err
	}

	kernel := nativeGemma4RouterMatVecKernel(meta, proj.GroupSize, proj.Bits)

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

type nativeGemma4RouterMatVecMeta struct {
	inDim        int
	outDim       int
	packedIn     int
	groups       int
	packFactor   int
	sidecarDType DType
}

func validateNativeGemma4RouterMatVec(input *Array, proj *Linear) (nativeGemma4RouterMatVecMeta, bool, error) {
	var meta nativeGemma4RouterMatVecMeta
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
	return nativeGemma4RouterMatVecMeta{
		inDim:        inDim,
		outDim:       outDim,
		packedIn:     packedIn,
		groups:       groups,
		packFactor:   packFactor,
		sidecarDType: proj.Scales.Dtype(),
	}, true, nil
}

type nativeGemma4RouterMatVecKernelKey struct {
	bits         int
	groupSize    int
	inDim        int
	outDim       int
	packedIn     int
	sidecarDType DType
}

var nativeGemma4RouterMatVecKernelCache struct {
	sync.Mutex
	kernels map[nativeGemma4RouterMatVecKernelKey]*MetalKernel
}

func nativeGemma4RouterMatVecKernel(meta nativeGemma4RouterMatVecMeta, groupSize, bits int) *MetalKernel {
	key := nativeGemma4RouterMatVecKernelKey{
		bits:         bits,
		groupSize:    groupSize,
		inDim:        meta.inDim,
		outDim:       meta.outDim,
		packedIn:     meta.packedIn,
		sidecarDType: meta.sidecarDType,
	}
	nativeGemma4RouterMatVecKernelCache.Lock()
	defer nativeGemma4RouterMatVecKernelCache.Unlock()
	if nativeGemma4RouterMatVecKernelCache.kernels == nil {
		nativeGemma4RouterMatVecKernelCache.kernels = make(map[nativeGemma4RouterMatVecKernelKey]*MetalKernel)
	}
	if kernel := nativeGemma4RouterMatVecKernelCache.kernels[key]; kernel != nil {
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
	nativeGemma4RouterMatVecKernelCache.kernels[key] = kernel
	return kernel
}

func nativeGemma4RouterTopK(scores, perExpertScale *Array, topK int) (*Array, *Array, bool, error) {
	if !nativeGemma4RouterTopKEnabled() {
		return nil, nil, false, nil
	}
	return nativeMoERouterTopK(scores, perExpertScale, topK)
}

func nativeMoERouterTopK(scores, perExpertScale *Array, topK int) (*Array, *Array, bool, error) {
	if scores == nil || !scores.Valid() || perExpertScale == nil || !perExpertScale.Valid() {
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

	kernel := nativeGemma4RouterTopKKernel(experts, topK)
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

type nativeGemma4RouterTopKKernelKey struct {
	experts int
	topK    int
}

var nativeGemma4RouterTopKKernelCache struct {
	sync.Mutex
	kernels map[nativeGemma4RouterTopKKernelKey]*MetalKernel
}

func nativeGemma4RouterTopKKernel(experts, topK int) *MetalKernel {
	key := nativeGemma4RouterTopKKernelKey{experts: experts, topK: topK}
	nativeGemma4RouterTopKKernelCache.Lock()
	defer nativeGemma4RouterTopKKernelCache.Unlock()
	if nativeGemma4RouterTopKKernelCache.kernels == nil {
		nativeGemma4RouterTopKKernelCache.kernels = make(map[nativeGemma4RouterTopKKernelKey]*MetalKernel)
	}
	if kernel := nativeGemma4RouterTopKKernelCache.kernels[key]; kernel != nil {
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
	nativeGemma4RouterTopKKernelCache.kernels[key] = kernel
	return kernel
}
