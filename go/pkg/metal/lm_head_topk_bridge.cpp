// SPDX-Licence-Identifier: EUPL-1.2
//
// Fused quantized LM-head + top-k (#95). Stage 1 is a JIT Metal kernel:
// threadgroup-tiled q4 affine matrix-vector with an in-kernel insertion
// top-k per tile (the mlx qmv nibble trick: masked nibbles stay in place,
// x pre-scaled by 1/16^i compensates; affine bias folds via the x-sum).
// Stage 2 merges the per-tile candidates in-graph (argpartition + argsort)
// so the full vocab logits row never exists anywhere.
//
// Kernel design ported from MTPLX (Apache-2.0), mtplx/kernels/lm_head_topk.py
// — credit Youssof AL. Adapted: C++ string templating, keyed kernel cache,
// in-graph second stage, and an explicit K %% BLOCK_SIZE eligibility rule
// (the tile loop carries no K tail guard by design).

#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "lm_head_topk_bridge.h"
#include "mlx/c/error.h"
#include "mlx/c/private/mlx.h"
#include "mlx/fast.h"
#include "mlx/mlx.h"

namespace {

using ArrayVector = std::vector<mlx::core::array>;

std::string q4_topk_header(int group_size, int top_k, int num_simdgroups, int subtiles) {
  std::string h = R"(
        using namespace metal;

        constant constexpr int PACK_FACTOR = 8;
        constant constexpr int PACKS_PER_THREAD = 2;
        constant constexpr int VALUES_PER_THREAD = PACK_FACTOR * PACKS_PER_THREAD;
        constant constexpr int BYTES_PER_PACK = 4;
        constant constexpr int BLOCK_SIZE = VALUES_PER_THREAD * 32;
        constant constexpr int RESULTS_PER_SIMDGROUP = 4;
)";
  h += "        constant constexpr int NUM_SIMDGROUPS = " +
      std::to_string(num_simdgroups) + ";\n";
  h += "        constant constexpr int BN = RESULTS_PER_SIMDGROUP * NUM_SIMDGROUPS;\n";
  h += "        constant constexpr int TOPK = " + std::to_string(top_k) + ";\n";
  h += "        constant constexpr int GS = " + std::to_string(group_size) + ";\n";
  h += "        constant constexpr int SUBTILES = " + std::to_string(subtiles) + ";\n";
  h += "        constant constexpr int OUTS_PER_TILE = BN * SUBTILES;\n";
  h += R"(
        constant constexpr int SCALE_STEP_PER_THREAD = GS / VALUES_PER_THREAD;

        template <typename T>
        inline float load_vector4_exact_topk(const device T* x, thread float* x_thread) {
          float sum = 0.0f;
          for (int i = 0; i < VALUES_PER_THREAD; i += 4) {
            sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
            x_thread[i] = x[i];
            x_thread[i + 1] = x[i + 1] / 16.0f;
            x_thread[i + 2] = x[i + 2] / 256.0f;
            x_thread[i + 3] = x[i + 3] / 4096.0f;
          }
          return sum;
        }

        inline float qdot4_exact_topk(
            const device uint8_t* w,
            const thread float* x_thread,
            float scale,
            float bias,
            float sum) {
          const device uint16_t* ws = (const device uint16_t*)w;
          float accum = 0.0f;
          for (int i = 0; i < (VALUES_PER_THREAD / 4); ++i) {
            uint16_t packed = ws[i];
            accum +=
              x_thread[4 * i] * float(packed & 0x000f) +
              x_thread[4 * i + 1] * float(packed & 0x00f0) +
              x_thread[4 * i + 2] * float(packed & 0x0f00) +
              x_thread[4 * i + 3] * float(packed & 0xf000);
          }
          return scale * accum + sum * bias;
        }
)";
  return h;
}

std::string q4_topk_source(int top_k, int num_simdgroups, int subtiles) {
  std::string s = R"(
        uint tile = threadgroup_position_in_grid.x;
        uint simd_gid = simdgroup_index_in_threadgroup;
        uint simd_lid = thread_index_in_simdgroup;
        int K = int(K_size);
        int N = int(N_size);
        int in_vec_size_w = K * BYTES_PER_PACK / PACK_FACTOR;
        int in_vec_size_g = K / GS;
)";
  s += "        int tile_base = int(tile) * OUTS_PER_TILE;\n";
  s += "        threadgroup float top_values[" + std::to_string(top_k) + "];\n";
  s += "        threadgroup int top_indices[" + std::to_string(top_k) + "];\n";
  s += "        threadgroup float cand_values[" + std::to_string(num_simdgroups * 4) + "];\n";
  s += "        threadgroup int cand_indices[" + std::to_string(num_simdgroups * 4) + "];\n";
  s += R"(
        if (simd_gid == 0 && simd_lid == 0) {
          for (int i = 0; i < TOPK; ++i) {
            top_values[i] = -INFINITY;
            top_indices[i] = -1;
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float x_thread[VALUES_PER_THREAD];
        for (int subtile = 0; subtile < SUBTILES; ++subtile) {
          int out_row = tile_base + subtile * BN + int(simd_gid) * RESULTS_PER_SIMDGROUP;
          float result[RESULTS_PER_SIMDGROUP] = {0.0f};

          const device uint8_t* w_base =
            (const device uint8_t*)w + out_row * in_vec_size_w
            + int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
          const device T* scales_base =
            scales + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
          const device T* biases_base =
            biases + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;

          for (int k_block = 0; k_block < K; k_block += BLOCK_SIZE) {
            const device T* x_ptr = x + k_block + int(simd_lid) * VALUES_PER_THREAD;
            float x_sum = load_vector4_exact_topk<T>(x_ptr, x_thread);
            const device uint8_t* w_block =
              w_base + k_block * BYTES_PER_PACK / PACK_FACTOR;
            const device T* scales_block = scales_base + k_block / GS;
            const device T* biases_block = biases_base + k_block / GS;

            for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
              int n = out_row + row;
              if (n < N) {
                const device uint8_t* wl = w_block + row * in_vec_size_w;
                const device T* sl = scales_block + row * in_vec_size_g;
                const device T* bl = biases_block + row * in_vec_size_g;
                result[row] += qdot4_exact_topk(
                  wl, x_thread, float(sl[0]), float(bl[0]), x_sum);
              }
            }
          }

          float summed[RESULTS_PER_SIMDGROUP];
          for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
            summed[row] = simd_sum(result[row]);
          }
          if (simd_lid == 0) {
            for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
              int cand = int(simd_gid) * RESULTS_PER_SIMDGROUP + row;
              int n = out_row + row;
              cand_values[cand] = (n < N) ? summed[row] : -INFINITY;
              cand_indices[cand] = (n < N) ? n : -1;
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);

          if (simd_gid == 0 && simd_lid == 0) {
            for (int cand = 0; cand < BN; ++cand) {
              float value = cand_values[cand];
              int index = cand_indices[cand];
              if (index < 0) {
                continue;
              }
              for (int pos = 0; pos < TOPK; ++pos) {
                if (value > top_values[pos]) {
                  for (int shift = TOPK - 1; shift > pos; --shift) {
                    top_values[shift] = top_values[shift - 1];
                    top_indices[shift] = top_indices[shift - 1];
                  }
                  top_values[pos] = value;
                  top_indices[pos] = index;
                  break;
                }
              }
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (simd_gid == 0 && simd_lid == 0) {
          for (int i = 0; i < TOPK; ++i) {
            values[int(tile) * TOPK + i] = top_values[i];
            indices[int(tile) * TOPK + i] = top_indices[i];
          }
        }
)";
  return s;
}

struct Q4TopKKernelKey {
  int group_size;
  int top_k;
  int num_simdgroups;
  int subtiles;
  int dtype;
  bool operator<(const Q4TopKKernelKey& o) const {
    return std::tie(group_size, top_k, num_simdgroups, subtiles, dtype) <
        std::tie(o.group_size, o.top_k, o.num_simdgroups, o.subtiles, o.dtype);
  }
};

const mlx::core::fast::CustomKernelFunction& q4_topk_kernel(
    int group_size,
    int top_k,
    int num_simdgroups,
    int subtiles,
    mlx::core::Dtype dtype) {
  static std::mutex mu;
  static std::map<Q4TopKKernelKey, mlx::core::fast::CustomKernelFunction> cache;
  std::lock_guard<std::mutex> lock(mu);
  Q4TopKKernelKey key{
      group_size, top_k, num_simdgroups, subtiles, static_cast<int>(dtype.val())};
  auto found = cache.find(key);
  if (found != cache.end()) {
    return found->second;
  }
  auto name = "go_mlx_q4_lm_head_topk_gs" + std::to_string(group_size) +
      "_k" + std::to_string(top_k) + "_sg" + std::to_string(num_simdgroups) +
      "_st" + std::to_string(subtiles);
  auto fn = mlx::core::fast::metal_kernel(
      name,
      {"x", "w", "scales", "biases", "K_size", "N_size"},
      {"values", "indices"},
      q4_topk_source(top_k, num_simdgroups, subtiles),
      q4_topk_header(group_size, top_k, num_simdgroups, subtiles),
      /* ensure_row_contiguous */ true,
      /* atomic_outputs */ false);
  auto inserted = cache.emplace(key, std::move(fn));
  return inserted.first->second;
}

} // namespace

extern "C" int go_mlx_q4_lm_head_topk(
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
    const mlx_stream stream) {
  try {
    (void)stream;
    auto xv = mlx_array_get_(x);
    auto wv = mlx_array_get_(w);
    auto sv = mlx_array_get_(scales);
    auto bv = mlx_array_get_(biases);

    auto flat_x = mlx::core::reshape(xv, mlx::core::Shape{-1});
    const int k_size = static_cast<int>(flat_x.shape(0));
    const int n_size = static_cast<int>(wv.shape(0));
    if (top_k < 1 || top_k > 64) {
      throw std::runtime_error("mlx: lm-head top_k must be in [1, 64]");
    }
    if (n_size < top_k) {
      throw std::runtime_error("mlx: lm-head rows are fewer than top_k");
    }
    // BLOCK_SIZE = 16 values/thread * 32 lanes; the tile loop has no K tail
    // guard, so a non-multiple K would read out of bounds.
    if (k_size <= 0 || (k_size % 512) != 0) {
      throw std::runtime_error("mlx: lm-head K must be a positive multiple of 512");
    }
    if (group_size != 32 && group_size != 64 && group_size != 128) {
      throw std::runtime_error("mlx: lm-head group_size must be 32, 64 or 128");
    }
    if (num_simdgroups != 2 && num_simdgroups != 4 && num_simdgroups != 8) {
      throw std::runtime_error("mlx: lm-head num_simdgroups must be 2, 4 or 8");
    }
    if (subtiles < 1) {
      throw std::runtime_error("mlx: lm-head subtiles must be >= 1");
    }

    const int bn = 4 * num_simdgroups;
    const int outs_per_tile = bn * subtiles;
    const int tile_count = (n_size + outs_per_tile - 1) / outs_per_tile;

    const auto& fn = q4_topk_kernel(
        group_size, top_k, num_simdgroups, subtiles, flat_x.dtype());

    ArrayVector inputs = {
        flat_x,
        wv,
        sv,
        bv,
        mlx::core::array(k_size),
        mlx::core::array(n_size)};
    std::vector<mlx::core::Shape> out_shapes = {
        mlx::core::Shape{tile_count, top_k},
        mlx::core::Shape{tile_count, top_k}};
    std::vector<mlx::core::Dtype> out_dtypes = {
        mlx::core::float32, mlx::core::int32};
    std::vector<std::pair<std::string, mlx::core::fast::TemplateArg>> tmpl = {
        {"T", flat_x.dtype()}};

    auto tiles = fn(
        inputs,
        out_shapes,
        out_dtypes,
        std::make_tuple(32 * tile_count, num_simdgroups, 1),
        std::make_tuple(32, num_simdgroups, 1),
        tmpl,
        std::nullopt,
        false,
        {});

    // Stage 2: merge per-tile candidates in-graph. Padding slots carry
    // -INFINITY so they sort behind every real row (N >= top_k guarantees
    // enough real candidates).
    auto flat_v = mlx::core::reshape(tiles[0], mlx::core::Shape{-1});
    auto flat_i = mlx::core::reshape(tiles[1], mlx::core::Shape{-1});
    auto neg = mlx::core::negative(flat_v);
    auto part = mlx::core::argpartition(neg, top_k - 1, 0);
    auto sel = mlx::core::slice(
        part, mlx::core::Shape{0}, mlx::core::Shape{top_k});
    auto top_v = mlx::core::take(flat_v, sel, 0);
    auto top_i = mlx::core::take(flat_i, sel, 0);
    auto order = mlx::core::argsort(mlx::core::negative(top_v), 0);
    mlx_array_set_(*out_values, mlx::core::take(top_v, order, 0));
    mlx_array_set_(*out_indices, mlx::core::take(top_i, order, 0));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
