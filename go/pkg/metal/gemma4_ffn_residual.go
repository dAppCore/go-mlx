// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"sync"

	core "dappco.re/go"
)

func nativeGemma4FFNResidual(residual, local, expert, localNorm, expertNorm, combinedNorm *Array, eps float32) (*Array, bool, error) {
	if !nativeGemma4FFNResidualRuntimeEnabled() {
		return nil, false, nil
	}
	meta, ok := validateNativeGemma4FFNResidual(residual, local, expert, localNorm, expertNorm, combinedNorm, eps)
	if !ok {
		return nil, false, nil
	}

	kernel := nativeGemma4FFNResidualKernel(meta)

	out, err := kernel.DispatchOne(
		MetalKernelGrid{GridX: 256, GridY: 1, GridZ: 1, TGX: 256, TGY: 1, TGZ: 1},
		meta.outputShape[:], DTypeFloat32,
		residual, local, expert, localNorm, expertNorm, combinedNorm,
	)
	if err != nil {
		return nil, true, core.E("mlx.nativeGemma4FFNResidual", "apply Metal kernel", err)
	}
	return out, true, nil
}

type nativeGemma4FFNResidualMeta struct {
	hidden            int
	residualDType     DType
	localDType        DType
	expertDType       DType
	localNormDType    DType
	expertNormDType   DType
	combinedNormDType DType
	outputShape       [3]int32
}

func validateNativeGemma4FFNResidual(residual, local, expert, localNorm, expertNorm, combinedNorm *Array, eps float32) (nativeGemma4FFNResidualMeta, bool) {
	var meta nativeGemma4FFNResidualMeta
	if residual == nil || local == nil || expert == nil || localNorm == nil || expertNorm == nil || combinedNorm == nil {
		return meta, false
	}
	if !residual.Valid() || !local.Valid() || !expert.Valid() || !localNorm.Valid() || !expertNorm.Valid() || !combinedNorm.Valid() {
		return meta, false
	}
	if eps != 1e-6 {
		return meta, false
	}
	shape := residual.Shape()
	if len(shape) != 3 || shape[0] != 1 || shape[1] != 1 || shape[2] <= 0 {
		return meta, false
	}
	for _, arr := range []*Array{local, expert} {
		arrShape := arr.Shape()
		if len(arrShape) != len(shape) {
			return meta, false
		}
		for i := range shape {
			if arrShape[i] != shape[i] {
				return meta, false
			}
		}
	}
	hidden := int(shape[2])
	for _, norm := range []*Array{localNorm, expertNorm, combinedNorm} {
		if norm.NumDims() != 1 || norm.Dim(0) != hidden {
			return meta, false
		}
	}
	return nativeGemma4FFNResidualMeta{
		hidden:            hidden,
		residualDType:     residual.Dtype(),
		localDType:        local.Dtype(),
		expertDType:       expert.Dtype(),
		localNormDType:    localNorm.Dtype(),
		expertNormDType:   expertNorm.Dtype(),
		combinedNormDType: combinedNorm.Dtype(),
		outputShape:       [3]int32{1, 1, int32(hidden)},
	}, true
}

type nativeGemma4FFNResidualKernelKey struct {
	hidden            int
	residualDType     DType
	localDType        DType
	expertDType       DType
	localNormDType    DType
	expertNormDType   DType
	combinedNormDType DType
}

var nativeGemma4FFNResidualKernelCache struct {
	sync.Mutex
	kernels map[nativeGemma4FFNResidualKernelKey]*MetalKernel
}

func nativeGemma4FFNResidualKernel(meta nativeGemma4FFNResidualMeta) *MetalKernel {
	key := nativeGemma4FFNResidualKernelKey{
		hidden:            meta.hidden,
		residualDType:     meta.residualDType,
		localDType:        meta.localDType,
		expertDType:       meta.expertDType,
		localNormDType:    meta.localNormDType,
		expertNormDType:   meta.expertNormDType,
		combinedNormDType: meta.combinedNormDType,
	}
	nativeGemma4FFNResidualKernelCache.Lock()
	defer nativeGemma4FFNResidualKernelCache.Unlock()
	if nativeGemma4FFNResidualKernelCache.kernels == nil {
		nativeGemma4FFNResidualKernelCache.kernels = make(map[nativeGemma4FFNResidualKernelKey]*MetalKernel)
	}
	if kernel := nativeGemma4FFNResidualKernelCache.kernels[key]; kernel != nil {
		return kernel
	}

	source := core.Sprintf(`uint tid = thread_position_in_threadgroup.x;
	threadgroup float local_sums[256];
	threadgroup float expert_sums[256];
	threadgroup float combined_sums[256];

	float local_sum = 0.0f;
	float expert_sum = 0.0f;
	for (uint col = tid; col < uint(%d); col += 256u) {
		float local_value = float(local[col]);
		float expert_value = float(expert[col]);
		local_sum += local_value * local_value;
		expert_sum += expert_value * expert_value;
	}
	local_sums[tid] = local_sum;
	expert_sums[tid] = expert_sum;
	threadgroup_barrier(mem_flags::mem_threadgroup);

	for (uint stride = 128u; stride > 0u; stride >>= 1u) {
		if (tid < stride) {
			local_sums[tid] += local_sums[tid + stride];
			expert_sums[tid] += expert_sums[tid + stride];
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
	}

	float local_inv = rsqrt(local_sums[0] / float(%d) + 0.000001f);
	float expert_inv = rsqrt(expert_sums[0] / float(%d) + 0.000001f);
	float combined_sum = 0.0f;
	for (uint col = tid; col < uint(%d); col += 256u) {
		float local_value = float(local[col]) * local_inv * float(local_norm[col]);
		float expert_value = float(expert[col]) * expert_inv * float(expert_norm[col]);
		float combined_value = local_value + expert_value;
		combined_sum += combined_value * combined_value;
	}
	combined_sums[tid] = combined_sum;
	threadgroup_barrier(mem_flags::mem_threadgroup);

	for (uint stride = 128u; stride > 0u; stride >>= 1u) {
		if (tid < stride) {
			combined_sums[tid] += combined_sums[tid + stride];
		}
		threadgroup_barrier(mem_flags::mem_threadgroup);
	}

	float combined_inv = rsqrt(combined_sums[0] / float(%d) + 0.000001f);
	for (uint col = tid; col < uint(%d); col += 256u) {
		float local_value = float(local[col]) * local_inv * float(local_norm[col]);
		float expert_value = float(expert[col]) * expert_inv * float(expert_norm[col]);
		float combined_value = (local_value + expert_value) * combined_inv * float(combined_norm[col]);
		out[col] = float(residual[col]) + combined_value;
	}`,
		meta.hidden,
		meta.hidden,
		meta.hidden,
		meta.hidden,
		meta.hidden,
		meta.hidden,
	)
	header := "#include <metal_stdlib>\nusing namespace metal;\n"
	kernel := NewMetalKernel(
		core.Sprintf("gemma4_ffn_residual_h%d_rd%d_ld%d_ed%d_lnd%d_end%d_cnd%d", meta.hidden, meta.residualDType, meta.localDType, meta.expertDType, meta.localNormDType, meta.expertNormDType, meta.combinedNormDType),
		[]string{"residual", "local", "expert", "local_norm", "expert_norm", "combined_norm"},
		[]string{"out"},
		source,
		header,
		true,
		false,
	)
	nativeGemma4FFNResidualKernelCache.kernels[key] = kernel
	return kernel
}
