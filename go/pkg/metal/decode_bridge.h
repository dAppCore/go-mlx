// SPDX-Licence-Identifier: EUPL-1.2

#pragma once

#include "mlx/c/mlx.h"

#ifdef __cplusplus
extern "C" {
#endif

void go_mlx_set_fixed_attention_diagnostics(
    int fixed_wide_matmul_attention,
    int fixed_row_cache_update);

typedef struct go_mlx_gemma4_layer_args_ {
  mlx_array x;
  mlx_array prev_keys;
  mlx_array prev_values;
  mlx_array per_layer_input;
  mlx_array fixed_mask;

  mlx_array input_norm;
  mlx_array post_attn_norm;
  mlx_array pre_ff_norm;
  mlx_array pre_ff_norm2;
  mlx_array post_ff_norm1;
  mlx_array post_ff_norm2;
  mlx_array post_ff_norm;
  mlx_array post_per_layer_input_norm;
  mlx_array layer_scalar;

  mlx_array q_weight;
  mlx_array q_scales;
  mlx_array q_biases;
  mlx_array k_weight;
  mlx_array k_scales;
  mlx_array k_biases;
  mlx_array v_weight;
  mlx_array v_scales;
  mlx_array v_biases;
  mlx_array o_weight;
  mlx_array o_scales;
  mlx_array o_biases;
  mlx_array q_norm;
  mlx_array k_norm;
  mlx_array rope_freqs;
  int q_group_size;
  int q_bits;
  int k_group_size;
  int k_bits;
  int v_group_size;
  int v_bits;
  int o_group_size;
  int o_bits;

  mlx_array mlp_gate_weight;
  mlx_array mlp_gate_scales;
  mlx_array mlp_gate_biases;
  int mlp_gate_group_size;
  int mlp_gate_bits;
  mlx_array mlp_up_weight;
  mlx_array mlp_up_scales;
  mlx_array mlp_up_biases;
  int mlp_up_group_size;
  int mlp_up_bits;
  mlx_array mlp_down_weight;
  mlx_array mlp_down_scales;
  mlx_array mlp_down_biases;
  int mlp_down_group_size;
  int mlp_down_bits;

  mlx_array router_weight;
  mlx_array router_scales;
  mlx_array router_biases;
  mlx_array router_scale;
  mlx_array router_per_expert_scale;
  int router_group_size;
  int router_bits;

  mlx_array expert_gate_weight;
  mlx_array expert_gate_scales;
  mlx_array expert_gate_biases;
  mlx_array expert_gate_bias;
  mlx_array expert_up_weight;
  mlx_array expert_up_scales;
  mlx_array expert_up_biases;
  mlx_array expert_up_bias;
  mlx_array expert_gate_up_weight;
  mlx_array expert_gate_up_scales;
  mlx_array expert_gate_up_biases;
  mlx_array expert_gate_up_bias;
  mlx_array expert_down_weight;
  mlx_array expert_down_scales;
  mlx_array expert_down_biases;
  mlx_array expert_down_bias;

  mlx_array per_layer_gate_weight;
  mlx_array per_layer_gate_scales;
  mlx_array per_layer_gate_biases;
  int per_layer_gate_group_size;
  int per_layer_gate_bits;
  mlx_array per_layer_projection_weight;
  mlx_array per_layer_projection_scales;
  mlx_array per_layer_projection_biases;
  int per_layer_projection_group_size;
  int per_layer_projection_bits;

  int has_prev;
  int owns_kv;
  int fixed_kv;
  int has_fixed_mask;
  int has_per_layer_input;
  int num_attention_heads;
  int num_key_value_heads;
  int head_dim;
  int rope_dims;
  int has_rope_freqs;
  int has_moe;
  int use_k_eq_v;
  int has_router_scale_scaled;
  int router_top_k;
  int expert_gate_group_size;
  int expert_gate_bits;
  int expert_up_group_size;
  int expert_up_bits;
  int expert_gate_up_group_size;
  int expert_gate_up_bits;
  int expert_down_group_size;
  int expert_down_bits;
  int offset;
  float rope_base;
  float attention_scale;
  float router_eps;
  float router_root_size;
} go_mlx_gemma4_layer_args;

typedef struct go_mlx_gemma4_fixed_attention_args_ {
  mlx_array x;
  mlx_array residual;
  mlx_array key_cache;
  mlx_array value_cache;
  mlx_array offset;
  mlx_array scale;
  mlx_array mask;

  mlx_array q_weight;
  mlx_array q_scales;
  mlx_array q_biases;
  mlx_array k_weight;
  mlx_array k_scales;
  mlx_array k_biases;
  mlx_array v_weight;
  mlx_array v_scales;
  mlx_array v_biases;
  mlx_array o_weight;
  mlx_array o_scales;
  mlx_array o_biases;
  mlx_array q_norm;
  mlx_array k_norm;
  mlx_array post_attn_norm;
  mlx_array rope_freqs;

  int has_mask;
  int num_attention_heads;
  int num_key_value_heads;
  int head_dim;
  int rope_dims;
  int has_rope_freqs;
  float rope_base;
} go_mlx_gemma4_fixed_attention_args;

int go_mlx_gemma4_decode_layer(
    mlx_array* out,
    mlx_array* new_keys,
    mlx_array* new_values,
    const go_mlx_gemma4_layer_args* args,
    const mlx_stream stream);

int go_mlx_gemma4_fixed_owner_attention(
    mlx_array* out,
    mlx_array* new_keys,
    mlx_array* new_values,
    const go_mlx_gemma4_fixed_attention_args* args,
    const mlx_stream stream);

int go_mlx_gemma4_fixed_owner_attention_residual(
    mlx_array* out,
    mlx_array* new_keys,
    mlx_array* new_values,
    const go_mlx_gemma4_fixed_attention_args* args,
    const mlx_stream stream);

int go_mlx_compiled_rms_norm_residual(
    mlx_array* out,
    const mlx_array residual,
    const mlx_array input,
    const mlx_array norm_weight,
    const mlx_stream stream);

int go_mlx_compiled_fixed_single_token_attention(
    mlx_array* out,
    mlx_array* new_keys,
    mlx_array* new_values,
    const mlx_array query,
    const mlx_array key_cache,
    const mlx_array value_cache,
    const mlx_array key,
    const mlx_array value,
    const mlx_array offset,
    const mlx_array scale,
    const mlx_array mask,
    const int has_mask,
    const mlx_stream stream);

int go_mlx_compiled_fixed_sliding_single_token_attention(
    mlx_array* out,
    mlx_array* new_keys,
    mlx_array* new_values,
    const mlx_array query,
    const mlx_array key_cache,
    const mlx_array value_cache,
    const mlx_array key,
    const mlx_array value,
    const mlx_array scale,
    const mlx_array shift_indices,
    const mlx_array last_index,
    const mlx_stream stream);

int go_mlx_native_paged_single_token_attention(
    mlx_array* out,
    const mlx_array query,
    const mlx_array* key_pages,
    const mlx_array* value_pages,
    const int page_count,
    const float scale,
    const mlx_stream stream);

#ifdef __cplusplus
}
#endif
