// SPDX-Licence-Identifier: EUPL-1.2

#include <array>
#include <atomic>
#include <cstdint>
#include <exception>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "decode_bridge.h"
#include "mlx/c/error.h"
#include "mlx/c/private/mlx.h"
#include "mlx/compile.h"
#include "mlx/fast.h"
#include "mlx/backend/gpu/eval.h"
#include "mlx/mlx.h"


extern "C" int go_mlx_ensure_thread_streams(const mlx_stream* streams, size_t n) {
  try {
    for (size_t i = 0; i < n; ++i) {
      auto& s = mlx_stream_get_(streams[i]);
      if (s.device == mlx::core::Device::gpu) {
        // Idempotent per-thread encoder registration (try_emplace inside).
        mlx::core::gpu::new_stream(s);
      }
      if (i == 0) {
        mlx::core::set_default_stream(s);
      }
    }
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

namespace {

using ArrayVector = std::vector<mlx::core::array>;

std::atomic<bool> g_fixed_wide_matmul_attention{false};
std::atomic<bool> g_fixed_row_cache_update{false};

mlx::core::array last_token_logits(const mlx::core::array& logits) {
  const auto ndim = static_cast<int>(logits.ndim());
  if (ndim <= 0) {
    throw std::runtime_error("mlx: logits rank is invalid");
  }
  if (ndim == 1) {
    return mlx::core::reshape(logits, mlx::core::Shape{1, logits.shape(0)});
  }

  const auto seq_axis = ndim == 2 ? 0 : ndim - 2;
  const auto seq_len = logits.shape(seq_axis);
  if (seq_len <= 0) {
    throw std::runtime_error("mlx: logits sequence is empty");
  }

  mlx::core::Shape starts(ndim, 0);
  mlx::core::Shape stops = logits.shape();
  starts[seq_axis] = seq_len - 1;
  stops[seq_axis] = seq_len;

  auto last = mlx::core::slice(logits, starts, stops);
  return mlx::core::reshape(
      last,
      mlx::core::Shape{1, last.shape(static_cast<int>(last.ndim()) - 1)});
}

const std::function<ArrayVector(const ArrayVector&)>& compiled_greedy_decode_token() {
  static const auto fn = mlx::core::compile(
      [](const ArrayVector& inputs) -> ArrayVector {
        if (inputs.empty()) {
          throw std::runtime_error("mlx: decode token inputs are empty");
        }
        auto last = last_token_logits(inputs[0]);
        return {mlx::core::argmax(last, -1, false)};
      },
      false);
  return fn;
}

mlx::core::array softcap30(const mlx::core::array& logits) {
  auto scale = mlx::core::array(30.0f, logits.dtype());
  auto scaled = mlx::core::divide(logits, scale);
  auto capped = mlx::core::tanh(scaled);
  return mlx::core::multiply(capped, scale);
}

mlx::core::array suppress_token_logits(
    const mlx::core::array& logits,
    const mlx::core::array& suppress_token_ids) {
  if (suppress_token_ids.size() == 0) {
    return logits;
  }
  auto update_shape = logits.shape();
  if (update_shape.empty()) {
    throw std::runtime_error("mlx: suppress-token logits rank is invalid");
  }
  update_shape.back() = suppress_token_ids.size();
  auto indices = mlx::core::reshape(suppress_token_ids, update_shape);
  auto updates = mlx::core::full(
      update_shape,
      -std::numeric_limits<float>::infinity(),
      logits.dtype());
  return mlx::core::put_along_axis(logits, indices, updates, -1);
}

const std::function<ArrayVector(const ArrayVector&)>&
compiled_dense_last_logits_softcap30() {
  static const auto fn = mlx::core::compile(
      [](const ArrayVector& inputs) -> ArrayVector {
        if (inputs.size() != 3) {
          throw std::runtime_error("mlx: dense last-logits inputs are invalid");
        }
        auto normed = mlx::core::fast::rms_norm(inputs[0], inputs[1], 1e-6f);
        auto weight_t = mlx::core::transpose(inputs[2]);
        auto logits = mlx::core::matmul(normed, weight_t);
        return {softcap30(logits)};
      },
      true);
  return fn;
}

const std::function<ArrayVector(const ArrayVector&)>&
compiled_q4_g64_last_logits_softcap30() {
  static const auto fn = mlx::core::compile(
      [](const ArrayVector& inputs) -> ArrayVector {
        if (inputs.size() != 5) {
          throw std::runtime_error("mlx: q4 last-logits inputs are invalid");
        }
        auto normed = mlx::core::fast::rms_norm(inputs[0], inputs[1], 1e-6f);
        auto logits = mlx::core::quantized_matmul(
            normed,
            inputs[2],
            inputs[3],
            inputs[4],
            true,
            64,
            4,
            "affine");
        return {softcap30(logits)};
      },
      true);
  return fn;
}

const std::function<ArrayVector(const ArrayVector&)>&
compiled_dense_last_token() {
  static const auto fn = mlx::core::compile(
      [](const ArrayVector& inputs) -> ArrayVector {
        if (inputs.size() != 3) {
          throw std::runtime_error("mlx: dense last-token inputs are invalid");
        }
        auto normed = mlx::core::fast::rms_norm(inputs[0], inputs[1], 1e-6f);
        auto weight_t = mlx::core::transpose(inputs[2]);
        auto logits = mlx::core::matmul(normed, weight_t);
        return {mlx::core::argmax(logits, -1, false)};
      },
      true);
  return fn;
}

const std::function<ArrayVector(const ArrayVector&)>&
compiled_dense_last_token_suppressed() {
  static const auto fn = mlx::core::compile(
      [](const ArrayVector& inputs) -> ArrayVector {
        if (inputs.size() != 4) {
          throw std::runtime_error("mlx: dense suppressed last-token inputs are invalid");
        }
        auto normed = mlx::core::fast::rms_norm(inputs[0], inputs[1], 1e-6f);
        auto weight_t = mlx::core::transpose(inputs[2]);
        auto logits = mlx::core::matmul(normed, weight_t);
        logits = suppress_token_logits(logits, inputs[3]);
        return {mlx::core::argmax(logits, -1, false)};
      },
      true);
  return fn;
}

const std::function<ArrayVector(const ArrayVector&)>&
compiled_q4_g64_last_token() {
  static const auto fn = mlx::core::compile(
      [](const ArrayVector& inputs) -> ArrayVector {
        if (inputs.size() != 5) {
          throw std::runtime_error("mlx: q4 last-token inputs are invalid");
        }
        auto normed = mlx::core::fast::rms_norm(inputs[0], inputs[1], 1e-6f);
        auto logits = mlx::core::quantized_matmul(
            normed,
            inputs[2],
            inputs[3],
            inputs[4],
            true,
            64,
            4,
            "affine");
        return {mlx::core::argmax(logits, -1, false)};
      },
      true);
  return fn;
}

const std::function<ArrayVector(const ArrayVector&)>&
compiled_q4_g64_last_token_suppressed() {
  static const auto fn = mlx::core::compile(
      [](const ArrayVector& inputs) -> ArrayVector {
        if (inputs.size() != 6) {
          throw std::runtime_error("mlx: q4 suppressed last-token inputs are invalid");
        }
        auto normed = mlx::core::fast::rms_norm(inputs[0], inputs[1], 1e-6f);
        auto logits = mlx::core::quantized_matmul(
            normed,
            inputs[2],
            inputs[3],
            inputs[4],
            true,
            64,
            4,
            "affine");
        logits = suppress_token_logits(logits, inputs[5]);
        return {mlx::core::argmax(logits, -1, false)};
      },
      true);
  return fn;
}

template <int Bits>
const std::function<ArrayVector(const ArrayVector&)>&
compiled_quant_g64_last_token() {
  static const auto fn = mlx::core::compile(
      [](const ArrayVector& inputs) -> ArrayVector {
        if (inputs.size() != 5) {
          throw std::runtime_error("mlx: quantized last-token inputs are invalid");
        }
        auto normed = mlx::core::fast::rms_norm(inputs[0], inputs[1], 1e-6f);
        auto logits = mlx::core::quantized_matmul(
            normed,
            inputs[2],
            inputs[3],
            inputs[4],
            true,
            64,
            Bits,
            "affine");
        return {mlx::core::argmax(logits, -1, false)};
      },
      true);
  return fn;
}

template <int Bits>
const std::function<ArrayVector(const ArrayVector&)>&
compiled_quant_g64_last_token_suppressed() {
  static const auto fn = mlx::core::compile(
      [](const ArrayVector& inputs) -> ArrayVector {
        if (inputs.size() != 6) {
          throw std::runtime_error("mlx: quantized suppressed last-token inputs are invalid");
        }
        auto normed = mlx::core::fast::rms_norm(inputs[0], inputs[1], 1e-6f);
        auto logits = mlx::core::quantized_matmul(
            normed,
            inputs[2],
            inputs[3],
            inputs[4],
            true,
            64,
            Bits,
            "affine");
        logits = suppress_token_logits(logits, inputs[5]);
        return {mlx::core::argmax(logits, -1, false)};
      },
      true);
  return fn;
}

const std::function<ArrayVector(const ArrayVector&)>&
compiled_q8_g64_last_token() {
  return compiled_quant_g64_last_token<8>();
}

const std::function<ArrayVector(const ArrayVector&)>&
compiled_q8_g64_last_token_suppressed() {
  return compiled_quant_g64_last_token_suppressed<8>();
}

const std::function<ArrayVector(const ArrayVector&)>&
compiled_q6_g64_last_token() {
  return compiled_quant_g64_last_token<6>();
}

const std::function<ArrayVector(const ArrayVector&)>&
compiled_q6_g64_last_token_suppressed() {
  return compiled_quant_g64_last_token_suppressed<6>();
}

mlx::core::array gelu_approx(const mlx::core::array& x) {
  auto x2 = mlx::core::multiply(x, x);
  auto x3 = mlx::core::multiply(x2, x);
  auto inner = mlx::core::add(
      x,
      mlx::core::multiply(x3, mlx::core::array(0.044715f, x.dtype())));
  auto scaled = mlx::core::multiply(
      inner,
      mlx::core::array(0.7978845608028654f, x.dtype()));
  auto t = mlx::core::tanh(scaled);
  auto one_plus = mlx::core::add(t, mlx::core::array(1.0f, x.dtype()));
  auto half_x = mlx::core::multiply(x, mlx::core::array(0.5f, x.dtype()));
  return mlx::core::multiply(half_x, one_plus);
}

mlx::core::array dense_linear(
    const mlx::core::array& x,
    const mlx::core::array& weight) {
  return mlx::core::matmul(x, mlx::core::transpose(weight));
}

mlx::core::array q4_g64_linear(
    const mlx::core::array& x,
    const mlx::core::array& weight,
    const mlx::core::array& scales,
    const mlx::core::array& biases) {
  return mlx::core::quantized_matmul(
      x,
      weight,
      scales,
      biases,
      true,
      64,
      4,
      "affine");
}

mlx::core::array repeat_kv(const mlx::core::array& input, int factor) {
  if (factor <= 1) {
    return input;
  }
  const auto shape = input.shape();
  if (shape.size() != 4) {
    throw std::runtime_error("mlx: repeat_kv expects rank-4 K/V tensors");
  }
  auto expanded = mlx::core::expand_dims(input, 2);
  auto broadcasted = mlx::core::broadcast_to(
      expanded,
      mlx::core::Shape{shape[0], shape[1], factor, shape[2], shape[3]});
  return mlx::core::reshape(
      broadcasted,
      mlx::core::Shape{shape[0], shape[1] * factor, shape[2], shape[3]});
}

mlx::core::array single_token_causal_mask(
    int capacity,
    const mlx::core::array& offset,
    mlx::core::Dtype dtype) {
  // The mask must promote to the attention's result type: a float32 mask
  // over half-precision K/V is rejected by scaled_dot_product_attention,
  // so build it at the cache dtype.
  auto idx = mlx::core::arange(0, capacity, 1);
  auto reshaped = mlx::core::reshape(
      idx,
      mlx::core::Shape{1, 1, 1, capacity});
  auto valid = mlx::core::less_equal(reshaped, offset);
  return mlx::core::where(
      valid,
      mlx::core::array(0.0f, dtype),
      mlx::core::array(mlx::core::finfo(dtype).min, dtype));
}

mlx::core::array single_token_cache_update(
    const mlx::core::array& cache,
    const mlx::core::array& token,
    const mlx::core::array& offset) {
  auto offset_index = mlx::core::reshape(
      offset,
      mlx::core::Shape{1, 1, 1, 1});
  auto indices = mlx::core::broadcast_to(offset_index, token.shape());
  return mlx::core::put_along_axis(cache, indices, token, 2);
}

mlx::core::array single_token_cache_row_update(
    const mlx::core::array& cache,
    const mlx::core::array& token,
    const mlx::core::array& offset) {
  const auto shape = cache.shape();
  if (shape.size() != 4 || token.shape().size() != 4) {
    throw std::runtime_error("mlx: row fixed cache update expects rank-4 tensors");
  }
  auto cache_rows = mlx::core::reshape(
      mlx::core::transpose(cache, {0, 2, 1, 3}),
      mlx::core::Shape{shape[0], shape[2], shape[1] * shape[3]});
  auto token_rows = mlx::core::reshape(
      mlx::core::transpose(token, {0, 2, 1, 3}),
      mlx::core::Shape{shape[0], 1, shape[1] * shape[3]});
  auto offset_index = mlx::core::reshape(
      offset,
      mlx::core::Shape{1, 1, 1});
  auto indices = mlx::core::broadcast_to(offset_index, token_rows.shape());
  auto updated_rows = mlx::core::put_along_axis(cache_rows, indices, token_rows, 1);
  auto updated = mlx::core::reshape(
      updated_rows,
      mlx::core::Shape{shape[0], shape[2], shape[1], shape[3]});
  return mlx::core::transpose(updated, {0, 2, 1, 3});
}

mlx::core::array sliding_single_token_cache_update(
    const mlx::core::array& cache,
    const mlx::core::array& token,
    const mlx::core::array& shift_indices,
    const mlx::core::array& last_index) {
  const auto shape = cache.shape();
  if (shape.size() != 4 || token.shape().size() != 4) {
    throw std::runtime_error("mlx: sliding fixed cache update expects rank-4 tensors");
  }
  if (shape[2] <= 0) {
    throw std::runtime_error("mlx: sliding fixed cache capacity is empty");
  }
  auto shifted = mlx::core::take(cache, shift_indices, 2);
  auto index = mlx::core::reshape(
      last_index,
      mlx::core::Shape{1, 1, 1, 1});
  auto indices = mlx::core::broadcast_to(index, token.shape());
  return mlx::core::put_along_axis(shifted, indices, token, 2);
}

const std::function<ArrayVector(const ArrayVector&)>&
compiled_fixed_single_token_attention() {
  static const auto fn = mlx::core::compile(
      [](const ArrayVector& inputs) -> ArrayVector {
        if (inputs.size() != 7) {
          throw std::runtime_error("mlx: fixed single-token attention inputs are invalid");
        }
        auto updated_keys = single_token_cache_update(inputs[1], inputs[3], inputs[5]);
        auto updated_values = single_token_cache_update(inputs[2], inputs[4], inputs[5]);
        auto mask = single_token_causal_mask(updated_keys.shape(2), inputs[5], updated_keys.dtype());
        auto scaled_query = mlx::core::multiply(inputs[0], mlx::core::astype(inputs[6], inputs[0].dtype()));
        auto out = mlx::core::fast::scaled_dot_product_attention(
            scaled_query,
            updated_keys,
            updated_values,
            1.0f,
            "array",
            std::optional<mlx::core::array>{mask});
        return {out, updated_keys, updated_values};
      },
      true);
  return fn;
}

const std::function<ArrayVector(const ArrayVector&)>&
compiled_fixed_single_token_attention_row_update() {
  static const auto fn = mlx::core::compile(
      [](const ArrayVector& inputs) -> ArrayVector {
        if (inputs.size() != 7) {
          throw std::runtime_error("mlx: row fixed single-token attention inputs are invalid");
        }
        auto updated_keys = single_token_cache_row_update(inputs[1], inputs[3], inputs[5]);
        auto updated_values = single_token_cache_row_update(inputs[2], inputs[4], inputs[5]);
        auto mask = single_token_causal_mask(updated_keys.shape(2), inputs[5], updated_keys.dtype());
        auto scaled_query = mlx::core::multiply(inputs[0], mlx::core::astype(inputs[6], inputs[0].dtype()));
        auto out = mlx::core::fast::scaled_dot_product_attention(
            scaled_query,
            updated_keys,
            updated_values,
            1.0f,
            "array",
            std::optional<mlx::core::array>{mask});
        return {out, updated_keys, updated_values};
      },
      true);
  return fn;
}

const std::function<ArrayVector(const ArrayVector&)>&
compiled_fixed_sliding_single_token_attention() {
  static const auto fn = mlx::core::compile(
      [](const ArrayVector& inputs) -> ArrayVector {
        if (inputs.size() != 8) {
          throw std::runtime_error("mlx: fixed sliding single-token attention inputs are invalid");
        }
        auto updated_keys = sliding_single_token_cache_update(inputs[1], inputs[3], inputs[6], inputs[7]);
        auto updated_values = sliding_single_token_cache_update(inputs[2], inputs[4], inputs[6], inputs[7]);
        auto scaled_query = mlx::core::multiply(inputs[0], mlx::core::astype(inputs[5], inputs[0].dtype()));
        auto out = mlx::core::fast::scaled_dot_product_attention(
            scaled_query,
            updated_keys,
            updated_values,
            1.0f);
        return {out, updated_keys, updated_values};
      },
      true);
  return fn;
}

const std::function<ArrayVector(const ArrayVector&)>&
compiled_fixed_single_token_attention_masked() {
  static const auto fn = mlx::core::compile(
      [](const ArrayVector& inputs) -> ArrayVector {
        if (inputs.size() != 8) {
          throw std::runtime_error("mlx: fixed single-token masked attention inputs are invalid");
        }
        auto updated_keys = single_token_cache_update(inputs[1], inputs[3], inputs[5]);
        auto updated_values = single_token_cache_update(inputs[2], inputs[4], inputs[5]);
        auto scaled_query = mlx::core::multiply(inputs[0], mlx::core::astype(inputs[6], inputs[0].dtype()));
        auto out = mlx::core::fast::scaled_dot_product_attention(
            scaled_query,
            updated_keys,
            updated_values,
            1.0f,
            "array",
            std::optional<mlx::core::array>{inputs[7]});
        return {out, updated_keys, updated_values};
      },
      true);
  return fn;
}

const std::function<ArrayVector(const ArrayVector&)>&
compiled_fixed_single_token_attention_row_update_masked() {
  static const auto fn = mlx::core::compile(
      [](const ArrayVector& inputs) -> ArrayVector {
        if (inputs.size() != 8) {
          throw std::runtime_error("mlx: row fixed single-token masked attention inputs are invalid");
        }
        auto updated_keys = single_token_cache_row_update(inputs[1], inputs[3], inputs[5]);
        auto updated_values = single_token_cache_row_update(inputs[2], inputs[4], inputs[5]);
        auto scaled_query = mlx::core::multiply(inputs[0], mlx::core::astype(inputs[6], inputs[0].dtype()));
        auto out = mlx::core::fast::scaled_dot_product_attention(
            scaled_query,
            updated_keys,
            updated_values,
            1.0f,
            "array",
            std::optional<mlx::core::array>{inputs[7]});
        return {out, updated_keys, updated_values};
      },
      true);
  return fn;
}

const std::function<ArrayVector(const ArrayVector&)>&
compiled_fixed_single_token_attention_matmul() {
  static const auto fn = mlx::core::compile(
      [](const ArrayVector& inputs) -> ArrayVector {
        if (inputs.size() != 7) {
          throw std::runtime_error("mlx: fixed single-token matmul attention inputs are invalid");
        }
        auto updated_keys = single_token_cache_update(inputs[1], inputs[3], inputs[5]);
        auto updated_values = single_token_cache_update(inputs[2], inputs[4], inputs[5]);
        auto scaled_query = mlx::core::multiply(inputs[0], mlx::core::astype(inputs[6], inputs[0].dtype()));

        auto keys = updated_keys;
        auto values = updated_values;
        const auto query_heads = scaled_query.shape(1);
        const auto key_heads = keys.shape(1);
        if (query_heads % key_heads != 0) {
          throw std::runtime_error("mlx: query heads must be a multiple of key heads");
        }
        const auto repeat_factor = query_heads / key_heads;
        if (repeat_factor > 1) {
          keys = repeat_kv(keys, repeat_factor);
          values = repeat_kv(values, repeat_factor);
        }

        auto key_t = mlx::core::transpose(keys, {0, 1, 3, 2});
        auto scores = mlx::core::matmul(scaled_query, key_t);
        auto mask = single_token_causal_mask(updated_keys.shape(2), inputs[5], updated_keys.dtype());
        scores = mlx::core::add(scores, mask);
        auto weights = mlx::core::softmax(scores, std::vector<int>{-1}, true);
        auto out = mlx::core::matmul(weights, values);
        return {out, updated_keys, updated_values};
      },
      true);
  return fn;
}

const std::function<ArrayVector(const ArrayVector&)>&
compiled_fixed_single_token_attention_matmul_masked() {
  static const auto fn = mlx::core::compile(
      [](const ArrayVector& inputs) -> ArrayVector {
        if (inputs.size() != 8) {
          throw std::runtime_error("mlx: fixed single-token masked matmul attention inputs are invalid");
        }
        auto updated_keys = single_token_cache_update(inputs[1], inputs[3], inputs[5]);
        auto updated_values = single_token_cache_update(inputs[2], inputs[4], inputs[5]);
        auto scaled_query = mlx::core::multiply(inputs[0], mlx::core::astype(inputs[6], inputs[0].dtype()));

        auto keys = updated_keys;
        auto values = updated_values;
        const auto query_heads = scaled_query.shape(1);
        const auto key_heads = keys.shape(1);
        if (query_heads % key_heads != 0) {
          throw std::runtime_error("mlx: query heads must be a multiple of key heads");
        }
        const auto repeat_factor = query_heads / key_heads;
        if (repeat_factor > 1) {
          keys = repeat_kv(keys, repeat_factor);
          values = repeat_kv(values, repeat_factor);
        }

        auto key_t = mlx::core::transpose(keys, {0, 1, 3, 2});
        auto scores = mlx::core::matmul(scaled_query, key_t);
        scores = mlx::core::add(scores, inputs[7]);
        auto weights = mlx::core::softmax(scores, std::vector<int>{-1}, true);
        auto out = mlx::core::matmul(weights, values);
        return {out, updated_keys, updated_values};
      },
      true);
  return fn;
}

mlx::core::array paged_single_token_attention_impl(
    const mlx::core::array& query,
    const ArrayVector& key_pages,
    const ArrayVector& value_pages,
    float scale) {
  if (key_pages.empty() || key_pages.size() != value_pages.size()) {
    throw std::runtime_error("mlx: paged attention page arrays are invalid");
  }
  if (key_pages.size() == 1) {
    return mlx::core::fast::scaled_dot_product_attention(
        query,
        key_pages[0],
        value_pages[0],
        scale);
  }

  ArrayVector score_pages;
  score_pages.reserve(key_pages.size());
  std::optional<mlx::core::array> global_max;
  for (size_t i = 0; i < key_pages.size(); i++) {
    auto key = key_pages[i];
    auto value = value_pages[i];
    if (key.ndim() != 4 || value.ndim() != 4 || query.ndim() != 4) {
      throw std::runtime_error("mlx: paged attention expects rank-4 tensors");
    }
    const auto query_heads = query.shape(1);
    const auto key_heads = key.shape(1);
    if (key_heads <= 0 || query_heads % key_heads != 0) {
      throw std::runtime_error("mlx: paged attention query heads must be a multiple of key heads");
    }
    const auto repeat_factor = query_heads / key_heads;
    if (repeat_factor > 1 && key_heads != 1) {
      key = repeat_kv(key, repeat_factor);
      value = repeat_kv(value, repeat_factor);
    }

    auto key_t = mlx::core::transpose(key, {0, 1, 3, 2});
    auto score = mlx::core::matmul(query, key_t);
    if (scale != 1.0f) {
      score = mlx::core::multiply(score, mlx::core::array(scale, score.dtype()));
    }
    auto page_max = mlx::core::max(score, -1, true);
    if (global_max.has_value()) {
      global_max = mlx::core::maximum(global_max.value(), page_max);
    } else {
      global_max = page_max;
    }
    score_pages.push_back(score);
  }

  std::optional<mlx::core::array> denom;
  std::optional<mlx::core::array> weighted;
  for (size_t i = 0; i < score_pages.size(); i++) {
    auto value = value_pages[i];
    const auto query_heads = query.shape(1);
    const auto value_heads = value.shape(1);
    const auto repeat_factor = value_heads > 0 ? query_heads / value_heads : 1;
    if (repeat_factor > 1 && value_heads != 1) {
      value = repeat_kv(value, repeat_factor);
    }

    auto shifted = mlx::core::subtract(score_pages[i], global_max.value());
    auto exp_score = mlx::core::exp(shifted);
    auto page_denom = mlx::core::sum(exp_score, -1, true);
    auto page_weighted = mlx::core::matmul(exp_score, value);
    if (denom.has_value()) {
      denom = mlx::core::add(denom.value(), page_denom);
      weighted = mlx::core::add(weighted.value(), page_weighted);
    } else {
      denom = page_denom;
      weighted = page_weighted;
    }
  }
  return mlx::core::divide(weighted.value(), denom.value());
}

using PagedAttentionCompileKey =
    std::tuple<int, int, int, int, int, int, int, int, int, int>;

const std::function<ArrayVector(const ArrayVector&)>&
compiled_paged_single_token_attention(
    int page_count,
    int query_heads,
    int key_heads,
    int value_heads,
    int page_tokens,
    int head_dim,
    int dtype_id) {
  if (page_count < 2 || query_heads <= 0 || key_heads <= 0 ||
      value_heads <= 0 || page_tokens <= 0 || head_dim <= 0 ||
      query_heads % key_heads != 0 || query_heads % value_heads != 0) {
    throw std::runtime_error("mlx: compiled paged attention signature is invalid");
  }
  const PagedAttentionCompileKey key{
      page_count,
      query_heads,
      key_heads,
      value_heads,
      query_heads / key_heads,
      query_heads / value_heads,
      page_tokens,
      head_dim,
      dtype_id,
      0};
  static std::mutex mu;
  static std::map<PagedAttentionCompileKey, std::function<ArrayVector(const ArrayVector&)>> cache;
  std::lock_guard<std::mutex> lock(mu);
  auto found = cache.find(key);
  if (found != cache.end()) {
    return found->second;
  }
  const int key_repeat = query_heads / key_heads;
  const int value_repeat = query_heads / value_heads;
  auto fn = mlx::core::compile(
      [page_count, key_repeat, value_repeat](const ArrayVector& inputs) -> ArrayVector {
        if (inputs.size() != static_cast<size_t>(2 + (page_count * 2))) {
          throw std::runtime_error("mlx: compiled paged attention inputs are invalid");
        }
        const auto& query = inputs[0];
        const auto& scale = inputs[1];

        ArrayVector score_pages;
        score_pages.reserve(static_cast<size_t>(page_count));
        std::optional<mlx::core::array> global_max;
        for (int i = 0; i < page_count; i++) {
          auto key = inputs[2 + static_cast<size_t>(i)];
          if (key.ndim() != 4 || query.ndim() != 4) {
            throw std::runtime_error("mlx: compiled paged attention expects rank-4 tensors");
          }
          if (key_repeat > 1) {
            key = repeat_kv(key, key_repeat);
          }

          auto key_t = mlx::core::transpose(key, {0, 1, 3, 2});
          auto score = mlx::core::matmul(query, key_t);
          score = mlx::core::multiply(score, scale);
          auto page_max = mlx::core::max(score, -1, true);
          if (global_max.has_value()) {
            global_max = mlx::core::maximum(global_max.value(), page_max);
          } else {
            global_max = page_max;
          }
          score_pages.push_back(score);
        }

        std::optional<mlx::core::array> denom;
        std::optional<mlx::core::array> weighted;
        for (int i = 0; i < page_count; i++) {
          auto value = inputs[2 + static_cast<size_t>(page_count + i)];
          if (value.ndim() != 4 || query.ndim() != 4) {
            throw std::runtime_error("mlx: compiled paged value tensors must be rank-4");
          }
          if (value_repeat > 1) {
            value = repeat_kv(value, value_repeat);
          }

          auto shifted = mlx::core::subtract(score_pages[i], global_max.value());
          auto exp_score = mlx::core::exp(shifted);
          auto page_denom = mlx::core::sum(exp_score, -1, true);
          auto page_weighted = mlx::core::matmul(exp_score, value);
          if (denom.has_value()) {
            denom = mlx::core::add(denom.value(), page_denom);
            weighted = mlx::core::add(weighted.value(), page_weighted);
          } else {
            denom = page_denom;
            weighted = page_weighted;
          }
        }
        return {mlx::core::divide(weighted.value(), denom.value())};
      },
      true);
  auto inserted = cache.emplace(key, std::move(fn));
  return inserted.first->second;
}

bool paged_single_token_attention_uniform_shape(
    const mlx::core::array& query,
    const ArrayVector& keys,
    const ArrayVector& values) {
  if (query.ndim() != 4 || keys.empty() || keys.size() != values.size()) {
    return false;
  }
  const auto key_shape = keys[0].shape();
  const auto value_shape = values[0].shape();
  if (key_shape.size() != 4 || value_shape.size() != 4 ||
      key_shape[0] != query.shape(0) ||
      key_shape[3] != query.shape(3) ||
      value_shape[0] != query.shape(0) ||
      value_shape[3] != query.shape(3)) {
    return false;
  }
  for (size_t i = 0; i < keys.size(); i++) {
    if (keys[i].shape() != key_shape || values[i].shape() != value_shape) {
      return false;
    }
  }
  return true;
}

bool fixed_wide_matmul_attention_enabled() {
  return g_fixed_wide_matmul_attention.load(std::memory_order_relaxed);
}

bool fixed_row_cache_update_enabled() {
  return g_fixed_row_cache_update.load(std::memory_order_relaxed);
}

struct Gemma4LayerState {
  std::optional<mlx::core::array> keys;
  std::optional<mlx::core::array> values;
};

const std::function<ArrayVector(const ArrayVector&)>& compiled_dense_mlp_gelu() {
  static const auto fn = mlx::core::compile(
      [](const ArrayVector& inputs) -> ArrayVector {
        if (inputs.size() != 4) {
          throw std::runtime_error("mlx: dense MLP inputs are invalid");
        }
        auto gate = dense_linear(inputs[0], inputs[1]);
        auto up = dense_linear(inputs[0], inputs[2]);
        auto activated = mlx::core::multiply(gelu_approx(gate), up);
        return {dense_linear(activated, inputs[3])};
      },
      true);
  return fn;
}

const std::function<ArrayVector(const ArrayVector&)>& compiled_q4_g64_mlp_gelu() {
  static const auto fn = mlx::core::compile(
      [](const ArrayVector& inputs) -> ArrayVector {
        if (inputs.size() != 10) {
          throw std::runtime_error("mlx: q4 MLP inputs are invalid");
        }
        auto gate = q4_g64_linear(inputs[0], inputs[1], inputs[2], inputs[3]);
        auto up = q4_g64_linear(inputs[0], inputs[4], inputs[5], inputs[6]);
        auto activated = mlx::core::multiply(gelu_approx(gate), up);
        return {q4_g64_linear(activated, inputs[7], inputs[8], inputs[9])};
      },
      true);
  return fn;
}

} // namespace

extern "C" void go_mlx_set_fixed_attention_diagnostics(
    int fixed_wide_matmul_attention,
    int fixed_row_cache_update) {
  g_fixed_wide_matmul_attention.store(
      fixed_wide_matmul_attention != 0,
      std::memory_order_relaxed);
  g_fixed_row_cache_update.store(
      fixed_row_cache_update != 0,
      std::memory_order_relaxed);
}

extern "C" int go_mlx_compiled_greedy_decode_token(
    mlx_array* res,
    const mlx_array logits,
    const mlx_stream stream) {
  try {
    (void)stream;
    ArrayVector inputs = {mlx_array_get_(logits)};
    auto outputs = compiled_greedy_decode_token()(inputs);
    mlx_array_set_(*res, std::move(outputs[0]));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int go_mlx_compiled_fixed_single_token_attention(
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
    const mlx_stream stream) {
  try {
    (void)stream;
    ArrayVector inputs = {
        mlx_array_get_(query),
        mlx_array_get_(key_cache),
        mlx_array_get_(value_cache),
        mlx_array_get_(key),
        mlx_array_get_(value),
        mlx_array_get_(offset),
        mlx_array_get_(scale)};
    if (has_mask) {
      inputs.push_back(mlx_array_get_(mask));
    }
    const auto use_matmul = mlx_array_get_(key_cache).shape(3) >= 512 &&
        fixed_wide_matmul_attention_enabled();
    const auto use_row_update = !use_matmul && fixed_row_cache_update_enabled();
    const auto& fn = use_matmul
        ? (has_mask
            ? compiled_fixed_single_token_attention_matmul_masked()
            : compiled_fixed_single_token_attention_matmul())
        : use_row_update
            ? (has_mask
                ? compiled_fixed_single_token_attention_row_update_masked()
                : compiled_fixed_single_token_attention_row_update())
        : (has_mask
            ? compiled_fixed_single_token_attention_masked()
            : compiled_fixed_single_token_attention());
    auto outputs = fn(inputs);
    mlx_array_set_(*out, std::move(outputs[0]));
    mlx_array_set_(*new_keys, std::move(outputs[1]));
    mlx_array_set_(*new_values, std::move(outputs[2]));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int go_mlx_compiled_fixed_sliding_single_token_attention(
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
    const mlx_stream stream) {
  try {
    (void)stream;
    ArrayVector inputs = {
        mlx_array_get_(query),
        mlx_array_get_(key_cache),
        mlx_array_get_(value_cache),
        mlx_array_get_(key),
        mlx_array_get_(value),
        mlx_array_get_(scale),
        mlx_array_get_(shift_indices),
        mlx_array_get_(last_index)};
    auto outputs = compiled_fixed_sliding_single_token_attention()(inputs);
    mlx_array_set_(*out, std::move(outputs[0]));
    mlx_array_set_(*new_keys, std::move(outputs[1]));
    mlx_array_set_(*new_values, std::move(outputs[2]));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int go_mlx_native_paged_single_token_attention(
    mlx_array* out,
    const mlx_array query,
    const mlx_array* key_pages,
    const mlx_array* value_pages,
    const int page_count,
    const float scale,
    const mlx_stream stream) {
  try {
    (void)stream;
    if (key_pages == nullptr || value_pages == nullptr || page_count <= 0) {
      throw std::runtime_error("mlx: native paged attention pages are invalid");
    }
    ArrayVector keys;
    ArrayVector values;
    keys.reserve(static_cast<size_t>(page_count));
    values.reserve(static_cast<size_t>(page_count));
    for (int i = 0; i < page_count; i++) {
      keys.push_back(mlx_array_get_(key_pages[i]));
      values.push_back(mlx_array_get_(value_pages[i]));
    }
    auto query_array = mlx_array_get_(query);
    if (page_count == 1) {
      auto output = paged_single_token_attention_impl(
          query_array,
          keys,
          values,
          scale);
      mlx_array_set_(*out, std::move(output));
    } else if (paged_single_token_attention_uniform_shape(query_array, keys, values)) {
      ArrayVector inputs;
      inputs.reserve(static_cast<size_t>(2 + (page_count * 2)));
      inputs.push_back(query_array);
      inputs.emplace_back(scale, query_array.dtype());
      inputs.insert(inputs.end(), keys.begin(), keys.end());
      inputs.insert(inputs.end(), values.begin(), values.end());
      auto outputs = compiled_paged_single_token_attention(
          page_count,
          query_array.shape(1),
          keys[0].shape(1),
          values[0].shape(1),
          keys[0].shape(2),
          query_array.shape(3),
          static_cast<int>(query_array.dtype().val()))(inputs);
      mlx_array_set_(*out, std::move(outputs[0]));
    } else {
      auto output = paged_single_token_attention_impl(
          query_array,
          keys,
          values,
          scale);
      mlx_array_set_(*out, std::move(output));
    }
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int go_mlx_compiled_dense_last_logits_softcap30(
    mlx_array* res,
    const mlx_array hidden,
    const mlx_array norm_weight,
    const mlx_array output_weight,
    const mlx_stream stream) {
  try {
    (void)stream;
    ArrayVector inputs = {
        mlx_array_get_(hidden),
        mlx_array_get_(norm_weight),
        mlx_array_get_(output_weight)};
    auto outputs = compiled_dense_last_logits_softcap30()(inputs);
    mlx_array_set_(*res, std::move(outputs[0]));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int go_mlx_compiled_q4_g64_last_logits_softcap30(
    mlx_array* res,
    const mlx_array hidden,
    const mlx_array norm_weight,
    const mlx_array output_weight,
    const mlx_array output_scales,
    const mlx_array output_biases,
    const mlx_stream stream) {
  try {
    (void)stream;
    ArrayVector inputs = {
        mlx_array_get_(hidden),
        mlx_array_get_(norm_weight),
        mlx_array_get_(output_weight),
        mlx_array_get_(output_scales),
        mlx_array_get_(output_biases)};
    auto outputs = compiled_q4_g64_last_logits_softcap30()(inputs);
    mlx_array_set_(*res, std::move(outputs[0]));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int go_mlx_compiled_dense_last_token(
    mlx_array* res,
    const mlx_array hidden,
    const mlx_array norm_weight,
    const mlx_array output_weight,
    const mlx_stream stream) {
  try {
    (void)stream;
    ArrayVector inputs = {
        mlx_array_get_(hidden),
        mlx_array_get_(norm_weight),
        mlx_array_get_(output_weight)};
    auto outputs = compiled_dense_last_token()(inputs);
    mlx_array_set_(*res, std::move(outputs[0]));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int go_mlx_compiled_dense_last_token_suppressed(
    mlx_array* res,
    const mlx_array hidden,
    const mlx_array norm_weight,
    const mlx_array output_weight,
    const mlx_array suppress_token_ids,
    const mlx_stream stream) {
  try {
    (void)stream;
    ArrayVector inputs = {
        mlx_array_get_(hidden),
        mlx_array_get_(norm_weight),
        mlx_array_get_(output_weight),
        mlx_array_get_(suppress_token_ids)};
    auto outputs = compiled_dense_last_token_suppressed()(inputs);
    mlx_array_set_(*res, std::move(outputs[0]));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int go_mlx_compiled_q4_g64_last_token(
    mlx_array* res,
    const mlx_array hidden,
    const mlx_array norm_weight,
    const mlx_array output_weight,
    const mlx_array output_scales,
    const mlx_array output_biases,
    const mlx_stream stream) {
  try {
    (void)stream;
    ArrayVector inputs = {
        mlx_array_get_(hidden),
        mlx_array_get_(norm_weight),
        mlx_array_get_(output_weight),
        mlx_array_get_(output_scales),
        mlx_array_get_(output_biases)};
    auto outputs = compiled_q4_g64_last_token()(inputs);
    mlx_array_set_(*res, std::move(outputs[0]));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int go_mlx_compiled_q4_g64_last_token_suppressed(
    mlx_array* res,
    const mlx_array hidden,
    const mlx_array norm_weight,
    const mlx_array output_weight,
    const mlx_array output_scales,
    const mlx_array output_biases,
    const mlx_array suppress_token_ids,
    const mlx_stream stream) {
  try {
    (void)stream;
    ArrayVector inputs = {
        mlx_array_get_(hidden),
        mlx_array_get_(norm_weight),
        mlx_array_get_(output_weight),
        mlx_array_get_(output_scales),
        mlx_array_get_(output_biases),
        mlx_array_get_(suppress_token_ids)};
    auto outputs = compiled_q4_g64_last_token_suppressed()(inputs);
    mlx_array_set_(*res, std::move(outputs[0]));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int go_mlx_compiled_q8_g64_last_token(
    mlx_array* res,
    const mlx_array hidden,
    const mlx_array norm_weight,
    const mlx_array output_weight,
    const mlx_array output_scales,
    const mlx_array output_biases,
    const mlx_stream stream) {
  try {
    (void)stream;
    ArrayVector inputs = {
        mlx_array_get_(hidden),
        mlx_array_get_(norm_weight),
        mlx_array_get_(output_weight),
        mlx_array_get_(output_scales),
        mlx_array_get_(output_biases)};
    auto outputs = compiled_q8_g64_last_token()(inputs);
    mlx_array_set_(*res, std::move(outputs[0]));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int go_mlx_compiled_q8_g64_last_token_suppressed(
    mlx_array* res,
    const mlx_array hidden,
    const mlx_array norm_weight,
    const mlx_array output_weight,
    const mlx_array output_scales,
    const mlx_array output_biases,
    const mlx_array suppress_token_ids,
    const mlx_stream stream) {
  try {
    (void)stream;
    ArrayVector inputs = {
        mlx_array_get_(hidden),
        mlx_array_get_(norm_weight),
        mlx_array_get_(output_weight),
        mlx_array_get_(output_scales),
        mlx_array_get_(output_biases),
        mlx_array_get_(suppress_token_ids)};
    auto outputs = compiled_q8_g64_last_token_suppressed()(inputs);
    mlx_array_set_(*res, std::move(outputs[0]));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int go_mlx_compiled_q6_g64_last_token(
    mlx_array* res,
    const mlx_array hidden,
    const mlx_array norm_weight,
    const mlx_array output_weight,
    const mlx_array output_scales,
    const mlx_array output_biases,
    const mlx_stream stream) {
  try {
    (void)stream;
    ArrayVector inputs = {
        mlx_array_get_(hidden),
        mlx_array_get_(norm_weight),
        mlx_array_get_(output_weight),
        mlx_array_get_(output_scales),
        mlx_array_get_(output_biases)};
    auto outputs = compiled_q6_g64_last_token()(inputs);
    mlx_array_set_(*res, std::move(outputs[0]));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int go_mlx_compiled_q6_g64_last_token_suppressed(
    mlx_array* res,
    const mlx_array hidden,
    const mlx_array norm_weight,
    const mlx_array output_weight,
    const mlx_array output_scales,
    const mlx_array output_biases,
    const mlx_array suppress_token_ids,
    const mlx_stream stream) {
  try {
    (void)stream;
    ArrayVector inputs = {
        mlx_array_get_(hidden),
        mlx_array_get_(norm_weight),
        mlx_array_get_(output_weight),
        mlx_array_get_(output_scales),
        mlx_array_get_(output_biases),
        mlx_array_get_(suppress_token_ids)};
    auto outputs = compiled_q6_g64_last_token_suppressed()(inputs);
    mlx_array_set_(*res, std::move(outputs[0]));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int go_mlx_compiled_dense_mlp_gelu(
    mlx_array* res,
    const mlx_array input,
    const mlx_array gate_weight,
    const mlx_array up_weight,
    const mlx_array down_weight,
    const mlx_stream stream) {
  try {
    (void)stream;
    ArrayVector inputs = {
        mlx_array_get_(input),
        mlx_array_get_(gate_weight),
        mlx_array_get_(up_weight),
        mlx_array_get_(down_weight)};
    auto outputs = compiled_dense_mlp_gelu()(inputs);
    mlx_array_set_(*res, std::move(outputs[0]));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int go_mlx_compiled_q4_g64_mlp_gelu(
    mlx_array* res,
    const mlx_array input,
    const mlx_array gate_weight,
    const mlx_array gate_scales,
    const mlx_array gate_biases,
    const mlx_array up_weight,
    const mlx_array up_scales,
    const mlx_array up_biases,
    const mlx_array down_weight,
    const mlx_array down_scales,
    const mlx_array down_biases,
    const mlx_stream stream) {
  try {
    (void)stream;
    ArrayVector inputs = {
        mlx_array_get_(input),
        mlx_array_get_(gate_weight),
        mlx_array_get_(gate_scales),
        mlx_array_get_(gate_biases),
        mlx_array_get_(up_weight),
        mlx_array_get_(up_scales),
        mlx_array_get_(up_biases),
        mlx_array_get_(down_weight),
        mlx_array_get_(down_scales),
        mlx_array_get_(down_biases)};
    auto outputs = compiled_q4_g64_mlp_gelu()(inputs);
    mlx_array_set_(*res, std::move(outputs[0]));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
