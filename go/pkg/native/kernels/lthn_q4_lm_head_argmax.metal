// SPDX-Licence-Identifier: EUPL-1.2

#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

constant constexpr uint lthn_bf16_lm_head_rows_per_tile = 8;
constant constexpr uint lthn_bf16_logits_argmax_rows_per_tile = 256;

static inline bool lthn_lm_head_row_suppressed(uint row, device const int* suppress, int suppress_count) {
  for (int i = 0; i < suppress_count; i++) {
    if (suppress[i] == int(row)) {
      return true;
    }
  }
  return false;
}

// BF16 direct greedy path. It scores up to eight vocab rows per tile against
// one bf16 hidden vector and writes only the tile-local best row. Scores are
// rounded to bf16 before comparison so the selected token matches model.Greedy
// over the existing full BF16 logits row.
kernel void lthn_bf16_lm_head_argmax_tiles_bf16(
    device const bfloat* x       [[buffer(0)]],
    device const bfloat* weight  [[buffer(1)]],
    device float*        values  [[buffer(2)]],
    device int*          indices [[buffer(3)]],
    constant int&        d_model [[buffer(4)]],
    constant int&        vocab   [[buffer(5)]],
    device const int*    suppress [[buffer(6)]],
    constant int&        suppress_count [[buffer(7)]],
    uint tile [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint row_in_tile [[simdgroup_index_in_threadgroup]]) {
  if (d_model <= 0 || vocab <= 0) return;

  uint row = tile * lthn_bf16_lm_head_rows_per_tile + row_in_tile;
  float partial = 0.0f;
  if (row < uint(vocab)) {
    for (uint col = lane; col < uint(d_model); col += 32u) {
      partial += float(x[col]) * float(weight[row * uint(d_model) + col]);
    }
  }

  float score = simd_sum(partial);
  if (lane == 0u) {
    bool masked = row >= uint(vocab) || lthn_lm_head_row_suppressed(row, suppress, suppress_count);
    score = !masked ? float(bfloat(score)) : -INFINITY;
  }

  threadgroup float tile_values[lthn_bf16_lm_head_rows_per_tile];
  threadgroup int tile_indices[lthn_bf16_lm_head_rows_per_tile];
  if (lane == 0u) {
    tile_values[row_in_tile] = score;
    tile_indices[row_in_tile] = (row < uint(vocab)) ? int(row) : -1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (row_in_tile == 0u && lane == 0u) {
    float best = -INFINITY;
    int best_idx = -1;
    for (uint i = 0; i < lthn_bf16_lm_head_rows_per_tile; i++) {
      float v = tile_values[i];
      int idx = tile_indices[i];
      if (idx >= 0 && (v > best || (v == best && (best_idx < 0 || idx < best_idx)))) {
        best = v;
        best_idx = idx;
      }
    }
    values[tile] = best;
    indices[tile] = best_idx;
  }
}

// Argmax over an already-materialised bf16 logits row. Quant direct greedy uses
// this after the proven MLX affine_qmv_bfloat16_t projection, so native avoids
// full-logit host readback without duplicating q4 dot-product numerics.
kernel void lthn_bf16_logits_argmax_tiles_bf16(
    device const bfloat* logits [[buffer(0)]],
    device float*        values [[buffer(1)]],
    device int*          indices [[buffer(2)]],
    constant int&        vocab [[buffer(3)]],
    device const int*    suppress [[buffer(4)]],
    constant int&        suppress_count [[buffer(5)]],
    uint tile [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
  if (vocab <= 0 || lane >= 32u) return;

  uint start = tile * lthn_bf16_logits_argmax_rows_per_tile;
  uint end = min(start + lthn_bf16_logits_argmax_rows_per_tile, uint(vocab));
  float local_best = -INFINITY;
  int local_idx = -1;
  for (uint row = start + lane; row < end; row += 32u) {
    if (lthn_lm_head_row_suppressed(row, suppress, suppress_count)) {
      continue;
    }
    float score = float(logits[row]);
    if (score > local_best || (score == local_best && (local_idx < 0 || int(row) < local_idx))) {
      local_best = score;
      local_idx = int(row);
    }
  }

  threadgroup float lane_values[32];
  threadgroup int lane_indices[32];
  lane_values[lane] = local_best;
  lane_indices[lane] = local_idx;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (lane != 0u) return;
  float best = -INFINITY;
  int best_idx = -1;
  for (uint i = 0; i < 32u; i++) {
    float v = lane_values[i];
    int idx = lane_indices[i];
    if (idx >= 0 && (v > best || (v == best && (best_idx < 0 || idx < best_idx)))) {
      best = v;
      best_idx = idx;
    }
  }
  values[tile] = best;
  indices[tile] = best_idx;
}

// Stage 2: merge tile-local candidates to one token id. One threadgroup scans
// the small candidate vector; only the final token id is copied to the host.
kernel void lthn_argmax_merge_f32(
    device const float* values  [[buffer(0)]],
    device const int*   indices [[buffer(1)]],
    device int*         out     [[buffer(2)]],
    constant int&       n       [[buffer(3)]],
    uint lane [[thread_index_in_threadgroup]]) {
  if (n <= 0 || lane >= 32) return;

  float local_best = -INFINITY;
  int local_idx = -1;
  for (uint i = lane; i < uint(n); i += 32u) {
    float v = values[i];
    int idx = indices[i];
    if (idx >= 0 && (v > local_best || (v == local_best && (local_idx < 0 || idx < local_idx)))) {
      local_best = v;
      local_idx = idx;
    }
  }

  threadgroup float lane_values[32];
  threadgroup int lane_indices[32];
  lane_values[lane] = local_best;
  lane_indices[lane] = local_idx;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (lane != 0) return;
  float best = -INFINITY;
  int best_idx = -1;
  for (uint i = 0; i < 32u; i++) {
    float v = lane_values[i];
    int idx = lane_indices[i];
    if (idx >= 0 && (v > best || (v == best && (best_idx < 0 || idx < best_idx)))) {
      best = v;
      best_idx = idx;
    }
  }
  out[0] = best_idx;
}
