// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"sync"

	core "dappco.re/go"
)

// AffineQuantPrefersGemm reports whether this affine-quantized linear decodes
// faster through MLX quantized_matmul (which auto-selects its internal qmv at
// M=1) than through the custom QuantizedDenseMatVec kernel. AX-11 head-to-head
// at dim 2048 + 6144 (BenchmarkQuantDecodeOrdering): gemm wins q4 +44%,
// q8 +37%, bitstream-q6 2.6× — the custom q6 kernel achieves ~319 GB/s where
// the q8 kernel achieves ~871, which is where the bandwidth-impossible
// q8-faster-than-q6 serve inversion lived. Only legacy-packed q6
// (packedIn×5 == inDim, the pre-bitstream layout) stays on the native kernel:
// MLX's gemm cannot read that layout. Likewise non-standard group sizes: MLX
// ships qmv kernels only for groups 32/64/128 (gs=4 dies at Eval with "Unable
// to load kernel affine_qmv_float_gs_4_…"), while the native kernel handles
// any group size.
func AffineQuantPrefersGemm(linear *Linear) bool {
	if linear == nil || !IsAffineQuantizationMode(linear.QuantizationMode) {
		return false
	}
	switch linear.GroupSize {
	case 32, 64, 128:
	default:
		return false
	}
	switch linear.Bits {
	case 4, 8:
		return true
	case 6:
		if linear.Weight == nil || !linear.Weight.Valid() || linear.Scales == nil || !linear.Scales.Valid() {
			return false
		}
		inDim := linear.Scales.Dim(1) * linear.GroupSize
		return linear.Weight.Dim(1)*5 != inDim
	default:
		return false
	}
}

func nativeMLPMatVec(input *Array, mlp *MLP) (*Array, bool, error) {
	if !nativeMLPMatVecRuntimeEnabled() {
		return nil, false, nil
	}
	if input == nil || !input.Valid() || mlp == nil {
		return nil, false, nil
	}
	// q6-affine MLPs fall back to the per-linear gemm path (AffineQuantPrefersGemm);
	// q4/q8 use the fused gate+up+down kernel, which avoids materialising the
	// gate/up intermediates.
	for _, l := range []*Linear{mlp.GateProj, mlp.UpProj, mlp.DownProj} {
		if l != nil && l.Bits == 6 && AffineQuantPrefersGemm(l) {
			return nil, false, nil
		}
	}
	activated, ok, err := quantizedDenseGELUSplitGateUpMatVec(input, mlp.GateProj, mlp.UpProj)
	if err != nil || !ok {
		return nil, ok, err
	}
	out, ok, err := QuantizedDenseMatVec(activated, mlp.DownProj)
	Free(activated)
	if err != nil || !ok {
		Free(out)
		return nil, ok, err
	}
	return out, true, nil
}

func QuantizedDenseMatVec(input *Array, linear *Linear) (*Array, bool, error) {
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
		return nil, true, core.E("mlx.QuantizedDenseMatVec", "apply Metal kernel", err)
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

// maxDecodeMatVecBatch is the largest sequence length the batched quantized
// matvec accepts. Single-token decode is rows=1; the MTP verify forward is a
// small batch (draft block + carry, typically 2-3). Beyond this, prefill-style
// generic GEMM is the right tool, so the matvec declines and the caller falls
// back. The kernel holds one float accumulator per row in registers, so this
// also bounds register pressure.
const maxDecodeMatVecBatch = 8

type quantizedDenseMatVecMeta struct {
	bits         int
	groupSize    int
	inDim        int
	outDim       int
	packedIn     int
	groups       int
	packFactor   int
	rows         int
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
	if !IsAffineQuantizationMode(linear.QuantizationMode) {
		return meta, false
	}
	if linear.Bias != nil && linear.Bias.Valid() {
		return meta, false
	}
	if linear.GroupSize <= 0 || (linear.Bits != 4 && linear.Bits != 6 && linear.Bits != 8) {
		return meta, false
	}
	var inputShapeBuf [MaxTensorRank]int32
	shape := input.ShapeInto(inputShapeBuf[:0])
	if len(shape) != 3 || shape[0] != 1 || shape[1] < 1 || shape[1] > maxDecodeMatVecBatch {
		return meta, false
	}
	rows := int(shape[1])
	// The q6 bitstream/group-64 kernels have bespoke single-row sources; only
	// the standard kernel is row-batched, so decline a multi-row q6 weight.
	if rows > 1 && linear.Bits == 6 {
		return meta, false
	}
	// The batched kernel indexes x[r*inDim + in_col]; that row stride is only
	// valid for a row-contiguous input. Decline otherwise (generic GEMM copes).
	if rows > 1 && !input.IsRowContiguous() {
		return meta, false
	}
	var weightShapeBuf [MaxTensorRank]int32
	var scaleShapeBuf [MaxTensorRank]int32
	var biasShapeBuf [MaxTensorRank]int32
	weightShape := linear.Weight.ShapeInto(weightShapeBuf[:0])
	scaleShape := linear.Scales.ShapeInto(scaleShapeBuf[:0])
	biasShape := linear.Biases.ShapeInto(biasShapeBuf[:0])
	if len(weightShape) != 2 || len(scaleShape) != 2 || len(biasShape) != 2 {
		return meta, false
	}
	packFactor := 32 / linear.Bits
	inDim := int(shape[2])
	outDim := int(weightShape[0])
	packedIn := int(weightShape[1])
	groups := inDim / linear.GroupSize
	expectedPackedIn := quantizedDenseMatVecPackedIn(inDim, linear.Bits)
	legacyPacked := packedIn*packFactor == inDim
	bitstreamPacked := packedIn == expectedPackedIn
	if linear.Bits == 6 && bitstreamPacked && !legacyPacked && !nativeQ6BitstreamMatVecRuntimeEnabled() {
		return meta, false
	}
	if inDim <= 0 || outDim <= 0 || packedIn <= 0 || groups <= 0 || expectedPackedIn <= 0 || inDim%linear.GroupSize != 0 || (!legacyPacked && !bitstreamPacked) {
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
		rows:         rows,
		sidecarDType: linear.Scales.Dtype(),
		outputShape:  [3]int32{1, int32(rows), int32(outDim)},
	}, true
}

func quantizedDenseMatVecPackedIn(inDim, bits int) int {
	if inDim <= 0 || bits <= 0 {
		return 0
	}
	return (inDim*bits + 31) / 32
}

type quantizedDenseMatVecKernelKey struct {
	bits         int
	groupSize    int
	inDim        int
	outDim       int
	packedIn     int
	rows         int
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
		rows:         meta.rows,
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

	// Row-batched matvec: each thread owns one out_col and, for every packed
	// weight word it loads + dequantises, fans that weight across all ROWS
	// token-rows. The weight stream (the decode bottleneck) is paid once for the
	// whole batch — that is what makes a small MTP-verify batch as cheap as a
	// single-token decode. ROWS=1 is byte-identical to the prior single-row form.
	// One quantised matvec for every bit width. Each lane loads ONE weight word
	// and unpacks every value that STARTS in it (q4/q8: packFactor values, the
	// coalesced one-load-many-values fast path; q6 and any bits that do not divide
	// 32: ~packFactor values plus a boundary value whose high bits straddle into
	// the next word — pulled in only then). The straddle branch is never taken for
	// q4/q8, so they keep their throughput; q6 folds in here instead of a bespoke
	// kernel, and the next quant we add (q3/q5) falls out for free.
	source := core.Sprintf(`uint out_col = thread_position_in_grid.x / 32u;
if (out_col >= uint(%d)) {
	return;
}
uint lane = thread_index_in_simdgroup;
uint row_base = out_col * uint(%d);
float sum[%d];
for (uint r = 0u; r < uint(%d); r++) { sum[r] = 0.0f; }
for (uint pack_col = lane; pack_col < uint(%d); pack_col += 32u) {
	uint w0 = weight[row_base + pack_col];
	uint base_bit = pack_col * 32u;
	uint in_col = (base_bit + uint(%d) - 1u) / uint(%d);
	uint vbit = in_col * uint(%d);
	for (; vbit < base_bit + 32u && in_col < uint(%d); in_col++, vbit += uint(%d)) {
		uint bit_shift = vbit - base_bit;
		uint q = w0 >> bit_shift;
		if (bit_shift + uint(%d) > 32u && pack_col + 1u < uint(%d)) {
			q |= weight[row_base + pack_col + 1u] << (32u - bit_shift);
		}
		q &= uint(%d);
		uint group = in_col / uint(%d);
		uint scale_index = out_col * uint(%d) + group;
		float w = float(q) * float(scales[scale_index]) + float(qbiases[scale_index]);
		for (uint r = 0u; r < uint(%d); r++) {
			sum[r] += float(x[r * uint(%d) + in_col]) * w;
		}
	}
}
for (uint r = 0u; r < uint(%d); r++) {
	float s = simd_sum(sum[r]);
	if (lane == 0u) {
		out[r * uint(%d) + out_col] = s;
	}
}`,
		meta.outDim,
		meta.packedIn,
		meta.rows,
		meta.rows,
		meta.packedIn,
		bits,
		bits,
		bits,
		meta.inDim,
		bits,
		bits,
		meta.packedIn,
		(1<<bits)-1,
		groupSize,
		meta.groups,
		meta.rows,
		meta.inDim,
		meta.rows,
		meta.outDim,
	)
	// q6 packs 6-bit values across 32-bit word boundaries, so the unified
	// word-coalesced loop pays a per-value straddle branch that q4/q8 never hit.
	// The bitstream kernel walks values directly, and the group-64 variant
	// precomputes each lane's fixed bit position once (every group shares it),
	// recovering the throughput the unified path loses on this one packing — the
	// same fast-path split the GELU gate/up matvec keeps. Single-row only (the
	// rows>1 q6 weight is declined upstream), matching these single-row sources.
	if bits == 6 {
		source = quantizedDenseMatVecKernelQ6Source(meta, groupSize)
		if groupSize == 64 && meta.packedIn == meta.groups*12 {
			source = quantizedDenseMatVecKernelQ6Group64Source(meta)
		}
	}
	header := "#include <metal_stdlib>\n#include <metal_simdgroup>\nusing namespace metal;\n"
	kernel := NewMetalKernel(
		core.Sprintf("quantized_dense_matvec_b%d_g%d_i%d_o%d_p%d_r%d_s%d", bits, groupSize, meta.inDim, meta.outDim, meta.packedIn, meta.rows, meta.sidecarDType),
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
		rows:         meta.rows,
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

	// Row-batched gate/up GELU-split matvec: each dequantised gate+up weight word
	// is fanned across all ROWS token-rows so the weight stream is paid once for
	// the small decode batch. ROWS=1 is byte-identical to the prior single-row form.
	source := core.Sprintf(`uint out_col = thread_position_in_grid.x / 32u;
if (out_col >= uint(%d)) {
	return;
}
uint lane = thread_index_in_simdgroup;
float gate_sum[%d];
float up_sum[%d];
for (uint r = 0u; r < uint(%d); r++) { gate_sum[r] = 0.0f; up_sum[r] = 0.0f; }
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
		for (uint r = 0u; r < uint(%d); r++) {
			float input_value = float(x[r * uint(%d) + in_col]);
			gate_sum[r] += input_value * gate_w;
			up_sum[r] += input_value * up_w;
		}
	}
}
for (uint r = 0u; r < uint(%d); r++) {
	float gs = simd_sum(gate_sum[r]);
	float us = simd_sum(up_sum[r]);
	if (lane == 0u) {
		float gate_cube = gs * gs * gs;
		float gelu = 0.5f * gs * (1.0f + tanh(0.7978845608028654f * (gs + 0.044715f * gate_cube)));
		out[r * uint(%d) + out_col] = gelu * us;
	}
}`,
		meta.outDim,
		meta.rows,
		meta.rows,
		meta.rows,
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
		meta.rows,
		meta.inDim,
		meta.rows,
		meta.outDim,
	)
	if bits == 6 {
		source = quantizedDenseGELUSplitGateUpMatVecKernelQ6Source(meta, groupSize)
		if groupSize == 64 && meta.packedIn == meta.groups*12 {
			source = quantizedDenseGELUSplitGateUpMatVecKernelQ6Group64Source(meta)
		}
	}
	header := "#include <metal_stdlib>\n#include <metal_simdgroup>\nusing namespace metal;\n"
	kernel := NewMetalKernel(
		core.Sprintf("quantized_dense_gelu_split_gate_up_matvec_b%d_g%d_i%d_o%d_p%d_r%d_s%d", bits, groupSize, meta.inDim, meta.outDim, meta.packedIn, meta.rows, meta.sidecarDType),
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
