// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import core "dappco.re/go"

func quantizedDenseMatVecKernelQ6Group64Source(meta quantizedDenseMatVecMeta) string {
	return core.Sprintf(`uint out_col = thread_position_in_grid.x / 32u;
if (out_col >= uint(%d)) {
	return;
}
uint lane = thread_index_in_simdgroup;
uint row_base = out_col * uint(%d);
uint lane_bit_offset = lane * 6u;
uint lane_pack_col = lane_bit_offset >> 5u;
uint lane_bit_shift = lane_bit_offset & 31u;
float sum = 0.0f;
for (uint group = 0; group < uint(%d); group++) {
	uint scale_index = out_col * uint(%d) + group;
	float scale = float(scales[scale_index]);
	float qbias = float(qbiases[scale_index]);
	uint group_pack_base = row_base + group * 12u;
	uint group_in_base = group * 64u;

	uint qbits0 = weight[group_pack_base + lane_pack_col] >> lane_bit_shift;
	if (lane_bit_shift > 26u) {
		qbits0 |= weight[group_pack_base + lane_pack_col + 1u] << (32u - lane_bit_shift);
	}
	uint q0 = qbits0 & 63u;
	float w0 = float(q0) * scale + qbias;
	sum += float(x[group_in_base + lane]) * w0;

	uint qbits1 = weight[group_pack_base + lane_pack_col + 6u] >> lane_bit_shift;
	if (lane_bit_shift > 26u) {
		qbits1 |= weight[group_pack_base + lane_pack_col + 7u] << (32u - lane_bit_shift);
	}
	uint q1 = qbits1 & 63u;
	float w1 = float(q1) * scale + qbias;
	sum += float(x[group_in_base + lane + 32u]) * w1;
}
sum = simd_sum(sum);
if (lane == 0u) {
	out[out_col] = sum;
}`,
		meta.outDim,
		meta.packedIn,
		meta.groups,
		meta.groups,
	)
}

func quantizedDenseMatVecKernelQ6Source(meta quantizedDenseMatVecMeta, groupSize int) string {
	return core.Sprintf(`uint out_col = thread_position_in_grid.x / 32u;
if (out_col >= uint(%d)) {
	return;
}
uint lane = thread_index_in_simdgroup;
uint row_base = out_col * uint(%d);
float sum = 0.0f;
for (uint in_col = lane; in_col < uint(%d); in_col += 32u) {
	uint bit_offset = in_col * 6u;
	uint pack_col = bit_offset >> 5u;
	uint bit_shift = bit_offset & 31u;
	uint qbits = weight[row_base + pack_col] >> bit_shift;
	if (bit_shift > 26u && pack_col + 1u < uint(%d)) {
		qbits |= weight[row_base + pack_col + 1u] << (32u - bit_shift);
	}
	uint q = qbits & 63u;
	uint group = in_col / uint(%d);
	uint scale_index = out_col * uint(%d) + group;
	float w = float(q) * float(scales[scale_index]) + float(qbiases[scale_index]);
	sum += float(x[in_col]) * w;
}
sum = simd_sum(sum);
if (lane == 0u) {
	out[out_col] = sum;
}`,
		meta.outDim,
		meta.packedIn,
		meta.inDim,
		meta.packedIn,
		groupSize,
		meta.groups,
	)
}

func quantizedDenseGELUSplitGateUpMatVecKernelQ6Group64Source(meta quantizedDenseMatVecMeta) string {
	return core.Sprintf(`uint out_col = thread_position_in_grid.x / 32u;
if (out_col >= uint(%d)) {
	return;
}
uint lane = thread_index_in_simdgroup;
uint row_base = out_col * uint(%d);
uint lane_bit_offset = lane * 6u;
uint lane_pack_col = lane_bit_offset >> 5u;
uint lane_bit_shift = lane_bit_offset & 31u;
float gate_sum = 0.0f;
float up_sum = 0.0f;
for (uint group = 0; group < uint(%d); group++) {
	uint scale_index = out_col * uint(%d) + group;
	float gate_scale = float(gate_scales[scale_index]);
	float gate_qbias = float(gate_qbiases[scale_index]);
	float up_scale = float(up_scales[scale_index]);
	float up_qbias = float(up_qbiases[scale_index]);
	uint group_pack_base = row_base + group * 12u;
	uint group_in_base = group * 64u;

	uint gate_qbits0 = gate_weight[group_pack_base + lane_pack_col] >> lane_bit_shift;
	uint up_qbits0 = up_weight[group_pack_base + lane_pack_col] >> lane_bit_shift;
	if (lane_bit_shift > 26u) {
		uint spill_shift = 32u - lane_bit_shift;
		gate_qbits0 |= gate_weight[group_pack_base + lane_pack_col + 1u] << spill_shift;
		up_qbits0 |= up_weight[group_pack_base + lane_pack_col + 1u] << spill_shift;
	}
	uint gate_q0 = gate_qbits0 & 63u;
	uint up_q0 = up_qbits0 & 63u;
	float input0 = float(x[group_in_base + lane]);
	gate_sum += input0 * (float(gate_q0) * gate_scale + gate_qbias);
	up_sum += input0 * (float(up_q0) * up_scale + up_qbias);

	uint gate_qbits1 = gate_weight[group_pack_base + lane_pack_col + 6u] >> lane_bit_shift;
	uint up_qbits1 = up_weight[group_pack_base + lane_pack_col + 6u] >> lane_bit_shift;
	if (lane_bit_shift > 26u) {
		uint spill_shift = 32u - lane_bit_shift;
		gate_qbits1 |= gate_weight[group_pack_base + lane_pack_col + 7u] << spill_shift;
		up_qbits1 |= up_weight[group_pack_base + lane_pack_col + 7u] << spill_shift;
	}
	uint gate_q1 = gate_qbits1 & 63u;
	uint up_q1 = up_qbits1 & 63u;
	float input1 = float(x[group_in_base + lane + 32u]);
	gate_sum += input1 * (float(gate_q1) * gate_scale + gate_qbias);
	up_sum += input1 * (float(up_q1) * up_scale + up_qbias);
}
gate_sum = simd_sum(gate_sum);
up_sum = simd_sum(up_sum);
if (lane == 0u) {
	float gate_cube = gate_sum * gate_sum * gate_sum;
	float gelu = 0.5f * gate_sum * (1.0f + tanh(0.7978845608028654f * (gate_sum + 0.044715f * gate_cube)));
	out[out_col] = gelu * up_sum;
}`,
		meta.outDim,
		meta.packedIn,
		meta.groups,
		meta.groups,
	)
}

func quantizedDenseGELUSplitGateUpMatVecKernelQ6Source(meta quantizedDenseMatVecMeta, groupSize int) string {
	return core.Sprintf(`uint out_col = thread_position_in_grid.x / 32u;
if (out_col >= uint(%d)) {
	return;
}
uint lane = thread_index_in_simdgroup;
uint row_base = out_col * uint(%d);
float gate_sum = 0.0f;
float up_sum = 0.0f;
for (uint in_col = lane; in_col < uint(%d); in_col += 32u) {
	uint bit_offset = in_col * 6u;
	uint pack_col = bit_offset >> 5u;
	uint bit_shift = bit_offset & 31u;
	uint gate_qbits = gate_weight[row_base + pack_col] >> bit_shift;
	uint up_qbits = up_weight[row_base + pack_col] >> bit_shift;
	if (bit_shift > 26u && pack_col + 1u < uint(%d)) {
		uint spill_shift = 32u - bit_shift;
		gate_qbits |= gate_weight[row_base + pack_col + 1u] << spill_shift;
		up_qbits |= up_weight[row_base + pack_col + 1u] << spill_shift;
	}
	uint gate_q = gate_qbits & 63u;
	uint up_q = up_qbits & 63u;
	uint group = in_col / uint(%d);
	uint scale_index = out_col * uint(%d) + group;
	float gate_w = float(gate_q) * float(gate_scales[scale_index]) + float(gate_qbiases[scale_index]);
	float up_w = float(up_q) * float(up_scales[scale_index]) + float(up_qbiases[scale_index]);
	float input_value = float(x[in_col]);
	gate_sum += input_value * gate_w;
	up_sum += input_value * up_w;
}
gate_sum = simd_sum(gate_sum);
up_sum = simd_sum(up_sum);
if (lane == 0u) {
	float gate_cube = gate_sum * gate_sum * gate_sum;
	float gelu = 0.5f * gate_sum * (1.0f + tanh(0.7978845608028654f * (gate_sum + 0.044715f * gate_cube)));
	out[out_col] = gelu * up_sum;
}`,
		meta.outDim,
		meta.packedIn,
		meta.inDim,
		meta.packedIn,
		groupSize,
		meta.groups,
	)
}
