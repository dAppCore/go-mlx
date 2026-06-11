// SPDX-Licence-Identifier: EUPL-1.2

#pragma once

#include "mlx/c/mlx.h"

#ifdef __cplusplus
extern "C" {
#endif

void go_mlx_set_fixed_attention_diagnostics(
    int fixed_wide_matmul_attention,
    int fixed_row_cache_update);

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

// go_mlx_ensure_thread_streams registers GPU command encoders for the given
// streams on the CURRENT OS thread and sets the first as the thread's
// default. MLX 0.31.2 encodes GPU graphs on the CALLING thread with
// per-thread command encoders (registered idempotently by gpu::new_stream);
// Go goroutines migrate across OS threads, so any eval-class entry must
// ensure the executing thread owns encoders for every stream the graph can
// touch. Returns 0 on success.
int go_mlx_ensure_thread_streams(const mlx_stream* streams, size_t n);

#ifdef __cplusplus
}
#endif
