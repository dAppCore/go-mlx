// SPDX-Licence-Identifier: EUPL-1.2
//
// Fused quantized LM-head + top-k bridge (#95): computes the q4 affine
// lm-head matrix-vector product AND the global top-k in one Metal pass +
// an in-graph tile merge — the full vocab logits row is never materialised.
// Kernel design ported from MTPLX (Apache-2.0) mtplx/kernels/lm_head_topk.py.

#ifndef GO_MLX_LM_HEAD_TOPK_BRIDGE_H
#define GO_MLX_LM_HEAD_TOPK_BRIDGE_H

#include "mlx/c/array.h"
#include "mlx/c/stream.h"

#ifdef __cplusplus
extern "C" {
#endif

// go_mlx_q4_lm_head_topk runs the fused q4-g{group_size} lm-head top-k:
// x [K] (or [1,K]) bf16/f16/f32, w [N, K/8] uint32-packed 4-bit, scales and
// biases [N, K/group_size]. Writes values [top_k] float32 (descending) and
// indices [top_k] int32. Returns non-zero on error (mlx_error carries it).
int go_mlx_q4_lm_head_topk(
    mlx_array* out_values,
    mlx_array* out_indices,
    const mlx_array x,
    const mlx_array w,
    const mlx_array scales,
    const mlx_array biases,
    int group_size,
    int top_k,
    int num_simdgroups,
    int subtiles,
    const mlx_stream stream);

#ifdef __cplusplus
}
#endif

#endif // GO_MLX_LM_HEAD_TOPK_BRIDGE_H
