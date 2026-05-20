// SPDX-Licence-Identifier: EUPL-1.2

#include <cstdlib>
#include <cstdint>
#include <exception>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "decode_bridge.h"
#include "mlx/c/error.h"
#include "mlx/c/private/mlx.h"
#include "mlx/compile.h"
#include "mlx/fast.h"
#include "mlx/mlx.h"

namespace {

using ArrayVector = std::vector<mlx::core::array>;

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

const std::function<ArrayVector(const ArrayVector&)>&
compiled_rms_norm_residual() {
  static const auto fn = mlx::core::compile(
      [](const ArrayVector& inputs) -> ArrayVector {
        if (inputs.size() != 3) {
          throw std::runtime_error("mlx: residual RMSNorm inputs are invalid");
        }
        auto normed = mlx::core::fast::rms_norm(inputs[1], inputs[2], 1e-6f);
        return {mlx::core::add(inputs[0], normed)};
      },
      true);
  return fn;
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

std::optional<int> optional_positive_int(int value) {
  if (value <= 0) {
    return std::nullopt;
  }
  return value;
}

bool valid_array(mlx_array arr) {
  return arr.ctx != nullptr;
}

mlx::core::array get_required(mlx_array arr, const char* name) {
  if (!valid_array(arr)) {
    throw std::runtime_error(std::string("mlx: missing Gemma 4 layer input: ") + name);
  }
  return mlx_array_get_(arr);
}

mlx::core::array layer_linear(
    const mlx::core::array& x,
    mlx_array weight,
    mlx_array scales,
    mlx_array biases,
    const char* name) {
  auto w = get_required(weight, name);
  if (valid_array(scales)) {
    return q4_g64_linear(x, w, mlx_array_get_(scales), mlx_array_get_(biases));
  }
  return dense_linear(x, w);
}

mlx::core::array layer_linear_quantized(
    const mlx::core::array& x,
    mlx_array weight,
    mlx_array scales,
    mlx_array biases,
    int group_size,
    int bits,
    const char* name) {
  auto w = get_required(weight, name);
  if (valid_array(scales)) {
    return mlx::core::quantized_matmul(
        x,
        w,
        mlx_array_get_(scales),
        mlx_array_get_(biases),
        true,
        optional_positive_int(group_size),
        optional_positive_int(bits),
        "affine");
  }
  return dense_linear(x, w);
}

mlx::core::array switch_linear(
    const mlx::core::array& x,
    mlx_array weight,
    mlx_array scales,
    mlx_array biases,
    mlx_array bias,
    const mlx::core::array& expert_indices,
    int group_size,
    int bits,
    const char* name) {
  auto w = get_required(weight, name);
  std::optional<mlx::core::array> out;
  if (valid_array(scales)) {
    out = mlx::core::gather_qmm(
        x,
        w,
        mlx_array_get_(scales),
        valid_array(biases) ? std::optional<mlx::core::array>{mlx_array_get_(biases)} : std::nullopt,
        std::nullopt,
        expert_indices,
        true,
        optional_positive_int(group_size),
        optional_positive_int(bits),
        "affine",
        false);
  } else {
    auto weight_t = mlx::core::transpose(w, {0, 2, 1});
    out = mlx::core::gather_mm(
        x,
        weight_t,
        std::nullopt,
        expert_indices,
        false);
  }
  auto result = *out;
  if (valid_array(bias)) {
    auto gathered_bias = mlx::core::take(mlx_array_get_(bias), expert_indices, 0);
    auto expanded_bias = mlx::core::expand_dims(
        gathered_bias,
        static_cast<int>(gathered_bias.ndim()) - 1);
    result = mlx::core::add(result, expanded_bias);
  }
  return result;
}

mlx::core::array slice_last_dim(
    const mlx::core::array& a,
    int start,
    int stop) {
  const auto ndim = static_cast<int>(a.ndim());
  mlx::core::Shape starts(ndim, 0);
  auto stops = a.shape();
  starts[ndim - 1] = start;
  stops[ndim - 1] = stop;
  return mlx::core::slice(a, starts, stops);
}

std::pair<mlx::core::array, mlx::core::array> split_last_dim(
    const mlx::core::array& a) {
  const auto ndim = static_cast<int>(a.ndim());
  const auto last = a.shape(ndim - 1);
  if (last % 2 != 0) {
    throw std::runtime_error("mlx: split_last_dim requires an even last dimension");
  }
  const auto mid = last / 2;
  return {slice_last_dim(a, 0, mid), slice_last_dim(a, mid, last)};
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

mlx::core::array gelu_gate_mul(
    const mlx::core::array& gate,
    const mlx::core::array& up) {
  return mlx::core::multiply(gelu_approx(gate), up);
}

mlx::core::array apply_gemma4_rope(
    const mlx::core::array& x,
    const go_mlx_gemma4_layer_args& args,
    const mlx::core::array& offset) {
  if (args.has_rope_freqs) {
    return mlx::core::fast::rope(
        x,
        args.head_dim,
        false,
        std::nullopt,
        1.0f,
        offset,
        mlx_array_get_(args.rope_freqs));
  }
  return mlx::core::fast::rope(
      x,
      args.rope_dims,
      false,
      args.rope_base,
      1.0f,
      offset);
}

mlx::core::array concat_cache_token(
    const mlx::core::array& previous,
    const mlx::core::array& current) {
  if (previous.shape().empty()) {
    return current;
  }
  return mlx::core::concatenate({previous, current}, 2);
}

mlx::core::array single_token_causal_mask(
    int capacity,
    const mlx::core::array& offset) {
  auto idx = mlx::core::arange(0, capacity, 1);
  auto reshaped = mlx::core::reshape(
      idx,
      mlx::core::Shape{1, 1, 1, capacity});
  auto valid = mlx::core::less_equal(reshaped, offset);
  return mlx::core::where(
      valid,
      mlx::core::array(0.0f),
      mlx::core::array(-1e9f));
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
        auto mask = single_token_causal_mask(updated_keys.shape(2), inputs[5]);
        auto scaled_query = mlx::core::multiply(inputs[0], inputs[6]);
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
        auto mask = single_token_causal_mask(updated_keys.shape(2), inputs[5]);
        auto scaled_query = mlx::core::multiply(inputs[0], inputs[6]);
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
        auto scaled_query = mlx::core::multiply(inputs[0], inputs[5]);
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
        auto scaled_query = mlx::core::multiply(inputs[0], inputs[6]);
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
        auto scaled_query = mlx::core::multiply(inputs[0], inputs[6]);
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

mlx::core::array apply_gemma4_fixed_attention_rope(
    const mlx::core::array& x,
    const go_mlx_gemma4_fixed_attention_args& args,
    const mlx::core::array& offset) {
  if (args.has_rope_freqs) {
    return mlx::core::fast::rope(
        x,
        args.head_dim,
        false,
        std::nullopt,
        1.0f,
        offset,
        mlx_array_get_(args.rope_freqs));
  }
  return mlx::core::fast::rope(
      x,
      args.rope_dims,
      false,
      args.rope_base,
      1.0f,
      offset);
}

ArrayVector gemma4_fixed_owner_attention_impl(
    const go_mlx_gemma4_fixed_attention_args& args) {
  auto x = get_required(args.x, "x");
  auto key_cache = get_required(args.key_cache, "key_cache");
  auto value_cache = get_required(args.value_cache, "value_cache");
  auto offset = get_required(args.offset, "offset");
  auto scale = get_required(args.scale, "scale");
  const auto B = x.shape(0);
  const auto L = x.shape(1);

  auto q_proj = layer_linear(
      x,
      args.q_weight,
      args.q_scales,
      args.q_biases,
      "q_weight");
  auto q = mlx::core::as_strided(
      q_proj,
      mlx::core::Shape{B, args.num_attention_heads, L, args.head_dim},
      mlx::core::Strides{
          L * args.num_attention_heads * args.head_dim,
          args.head_dim,
          args.num_attention_heads * args.head_dim,
          1},
      0);
  q = mlx::core::fast::rms_norm(
      q,
      get_required(args.q_norm, "q_norm"),
      1e-6f);
  q = apply_gemma4_fixed_attention_rope(q, args, offset);

  auto k_proj = layer_linear(
      x,
      args.k_weight,
      args.k_scales,
      args.k_biases,
      "k_weight");
  auto k = mlx::core::as_strided(
      k_proj,
      mlx::core::Shape{B, args.num_key_value_heads, L, args.head_dim},
      mlx::core::Strides{
          L * args.num_key_value_heads * args.head_dim,
          args.head_dim,
          args.num_key_value_heads * args.head_dim,
          1},
      0);
  k = mlx::core::fast::rms_norm(
      k,
      get_required(args.k_norm, "k_norm"),
      1e-6f);
  k = apply_gemma4_fixed_attention_rope(k, args, offset);

  auto v_proj = layer_linear(
      x,
      args.v_weight,
      args.v_scales,
      args.v_biases,
      "v_weight");
  auto v = mlx::core::as_strided(
      v_proj,
      mlx::core::Shape{B, args.num_key_value_heads, L, args.head_dim},
      mlx::core::Strides{
          L * args.num_key_value_heads * args.head_dim,
          args.head_dim,
          args.num_key_value_heads * args.head_dim,
          1},
      0);
  v = mlx::core::fast::rms_norm(v, std::nullopt, 1e-6f);

  auto updated_keys = single_token_cache_update(key_cache, k, offset);
  auto updated_values = single_token_cache_update(value_cache, v, offset);
  auto scaled_query = mlx::core::multiply(q, scale);
  std::optional<mlx::core::array> mask;
  if (args.has_mask) {
    mask = mlx_array_get_(args.mask);
  } else {
    mask = single_token_causal_mask(updated_keys.shape(2), offset);
  }
  auto attn = mlx::core::fast::scaled_dot_product_attention(
      scaled_query,
      updated_keys,
      updated_values,
      1.0f,
      "array",
      mask);

  auto transposed = mlx::core::transpose(attn, {0, 2, 1, 3});
  auto reshaped = mlx::core::reshape(
      transposed,
      mlx::core::Shape{B, L, args.num_attention_heads * args.head_dim});
  auto out = layer_linear(
      reshaped,
      args.o_weight,
      args.o_scales,
      args.o_biases,
      "o_weight");
  return {out, updated_keys, updated_values};
}

ArrayVector gemma4_q4_fixed_owner_attention_graph(
    const ArrayVector& inputs,
    bool has_rope_freqs,
    bool with_residual) {
  const auto x = inputs[0];
  const auto key_cache = inputs[1];
  const auto value_cache = inputs[2];
  const auto offset = inputs[3];
  const auto scale = inputs[4];
  const auto B = x.shape(0);
  const auto L = x.shape(1);
  const auto head_dim = key_cache.shape(3);
  const auto num_key_value_heads = key_cache.shape(1);

  auto q_proj = q4_g64_linear(x, inputs[5], inputs[6], inputs[7]);
  const auto num_attention_heads = q_proj.shape(2) / head_dim;
  auto q_reshaped = mlx::core::reshape(
      q_proj,
      mlx::core::Shape{B, L, num_attention_heads, head_dim});
  auto q = mlx::core::transpose(q_reshaped, {0, 2, 1, 3});
  q = mlx::core::fast::rms_norm(q, inputs[17], 1e-6f);

  auto k_proj = q4_g64_linear(x, inputs[8], inputs[9], inputs[10]);
  auto k_reshaped = mlx::core::reshape(
      k_proj,
      mlx::core::Shape{B, L, num_key_value_heads, head_dim});
  auto k = mlx::core::transpose(k_reshaped, {0, 2, 1, 3});
  k = mlx::core::fast::rms_norm(k, inputs[18], 1e-6f);

  auto v_proj = q4_g64_linear(x, inputs[11], inputs[12], inputs[13]);
  auto v_reshaped = mlx::core::reshape(
      v_proj,
      mlx::core::Shape{B, L, num_key_value_heads, head_dim});
  auto v = mlx::core::transpose(v_reshaped, {0, 2, 1, 3});
  v = mlx::core::fast::rms_norm(v, std::nullopt, 1e-6f);

  int mask_index = 19;
  if (has_rope_freqs) {
    q = mlx::core::fast::rope(
        q,
        head_dim,
        false,
        std::nullopt,
        1.0f,
        offset,
        inputs[19]);
    k = mlx::core::fast::rope(
        k,
        head_dim,
        false,
        std::nullopt,
        1.0f,
        offset,
        inputs[19]);
    mask_index = 20;
  } else {
    q = mlx::core::fast::rope(
        q,
        head_dim,
        false,
        10000.0f,
        1.0f,
        offset);
    k = mlx::core::fast::rope(
        k,
        head_dim,
        false,
        10000.0f,
        1.0f,
        offset);
  }

  auto updated_keys = single_token_cache_update(key_cache, k, offset);
  auto updated_values = single_token_cache_update(value_cache, v, offset);
  auto scaled_query = mlx::core::multiply(q, scale);
  auto attn = mlx::core::fast::scaled_dot_product_attention(
      scaled_query,
      updated_keys,
      updated_values,
      1.0f,
      "array",
      std::optional<mlx::core::array>{inputs[mask_index]});

  auto transposed = mlx::core::transpose(attn, {0, 2, 1, 3});
  auto reshaped = mlx::core::reshape(
      transposed,
      mlx::core::Shape{B, L, num_attention_heads * head_dim});
  auto out = q4_g64_linear(reshaped, inputs[14], inputs[15], inputs[16]);
  if (with_residual) {
    auto normed = mlx::core::fast::rms_norm(
        out,
        inputs[mask_index + 2],
        1e-6f);
    out = mlx::core::add(inputs[mask_index + 1], normed);
  }
  return {out, updated_keys, updated_values};
}

const std::function<ArrayVector(const ArrayVector&)>&
compiled_gemma4_q4_fixed_owner_attention_default_rope_masked() {
  static const auto fn = mlx::core::compile(
      [](const ArrayVector& inputs) -> ArrayVector {
        if (inputs.size() != 20) {
          throw std::runtime_error("mlx: Gemma 4 q4 fixed owner attention inputs are invalid");
        }
        return gemma4_q4_fixed_owner_attention_graph(inputs, false, false);
      },
      true);
  return fn;
}

const std::function<ArrayVector(const ArrayVector&)>&
compiled_gemma4_q4_fixed_owner_attention_freqs_masked() {
  static const auto fn = mlx::core::compile(
      [](const ArrayVector& inputs) -> ArrayVector {
        if (inputs.size() != 21) {
          throw std::runtime_error("mlx: Gemma 4 q4 fixed owner attention freqs inputs are invalid");
        }
        return gemma4_q4_fixed_owner_attention_graph(inputs, true, false);
      },
      true);
  return fn;
}

const std::function<ArrayVector(const ArrayVector&)>&
compiled_gemma4_q4_fixed_owner_attention_residual_default_rope_masked() {
  static const auto fn = mlx::core::compile(
      [](const ArrayVector& inputs) -> ArrayVector {
        if (inputs.size() != 22) {
          throw std::runtime_error("mlx: Gemma 4 q4 fixed owner attention residual inputs are invalid");
        }
        return gemma4_q4_fixed_owner_attention_graph(inputs, false, true);
      },
      true);
  return fn;
}

const std::function<ArrayVector(const ArrayVector&)>&
compiled_gemma4_q4_fixed_owner_attention_residual_freqs_masked() {
  static const auto fn = mlx::core::compile(
      [](const ArrayVector& inputs) -> ArrayVector {
        if (inputs.size() != 23) {
          throw std::runtime_error("mlx: Gemma 4 q4 fixed owner attention residual freqs inputs are invalid");
        }
        return gemma4_q4_fixed_owner_attention_graph(inputs, true, true);
      },
      true);
  return fn;
}

bool q4_fixed_owner_attention_linear_available(
    mlx_array weight,
    mlx_array scales,
    mlx_array biases) {
  return valid_array(weight) && valid_array(scales) && valid_array(biases);
}

bool q4_fixed_owner_attention_available(
    const go_mlx_gemma4_fixed_attention_args& args) {
  if (!args.has_mask || args.head_dim >= 512) {
    return false;
  }
  if (!q4_fixed_owner_attention_linear_available(args.q_weight, args.q_scales, args.q_biases) ||
      !q4_fixed_owner_attention_linear_available(args.k_weight, args.k_scales, args.k_biases) ||
      !q4_fixed_owner_attention_linear_available(args.v_weight, args.v_scales, args.v_biases) ||
      !q4_fixed_owner_attention_linear_available(args.o_weight, args.o_scales, args.o_biases)) {
    return false;
  }
  if (!valid_array(args.x) || !valid_array(args.key_cache) ||
      !valid_array(args.value_cache) || !valid_array(args.offset) ||
      !valid_array(args.scale) || !valid_array(args.q_norm) ||
      !valid_array(args.k_norm) || !valid_array(args.mask)) {
    return false;
  }
  if (args.has_rope_freqs) {
    return valid_array(args.rope_freqs);
  }
  return args.rope_dims == args.head_dim && args.rope_base == 10000.0f;
}

bool q4_fixed_owner_attention_residual_available(
    const go_mlx_gemma4_fixed_attention_args& args) {
  return q4_fixed_owner_attention_available(args) &&
      valid_array(args.residual) &&
      valid_array(args.post_attn_norm);
}

ArrayVector gemma4_q4_fixed_owner_attention_impl(
    const go_mlx_gemma4_fixed_attention_args& args) {
  ArrayVector inputs = {
      mlx_array_get_(args.x),
      mlx_array_get_(args.key_cache),
      mlx_array_get_(args.value_cache),
      mlx_array_get_(args.offset),
      mlx_array_get_(args.scale),
      mlx_array_get_(args.q_weight),
      mlx_array_get_(args.q_scales),
      mlx_array_get_(args.q_biases),
      mlx_array_get_(args.k_weight),
      mlx_array_get_(args.k_scales),
      mlx_array_get_(args.k_biases),
      mlx_array_get_(args.v_weight),
      mlx_array_get_(args.v_scales),
      mlx_array_get_(args.v_biases),
      mlx_array_get_(args.o_weight),
      mlx_array_get_(args.o_scales),
      mlx_array_get_(args.o_biases),
      mlx_array_get_(args.q_norm),
      mlx_array_get_(args.k_norm)};
  if (args.has_rope_freqs) {
    inputs.push_back(mlx_array_get_(args.rope_freqs));
    inputs.push_back(mlx_array_get_(args.mask));
    return compiled_gemma4_q4_fixed_owner_attention_freqs_masked()(inputs);
  }
  inputs.push_back(mlx_array_get_(args.mask));
  return compiled_gemma4_q4_fixed_owner_attention_default_rope_masked()(inputs);
}

ArrayVector gemma4_q4_fixed_owner_attention_residual_impl(
    const go_mlx_gemma4_fixed_attention_args& args) {
  ArrayVector inputs = {
      mlx_array_get_(args.x),
      mlx_array_get_(args.key_cache),
      mlx_array_get_(args.value_cache),
      mlx_array_get_(args.offset),
      mlx_array_get_(args.scale),
      mlx_array_get_(args.q_weight),
      mlx_array_get_(args.q_scales),
      mlx_array_get_(args.q_biases),
      mlx_array_get_(args.k_weight),
      mlx_array_get_(args.k_scales),
      mlx_array_get_(args.k_biases),
      mlx_array_get_(args.v_weight),
      mlx_array_get_(args.v_scales),
      mlx_array_get_(args.v_biases),
      mlx_array_get_(args.o_weight),
      mlx_array_get_(args.o_scales),
      mlx_array_get_(args.o_biases),
      mlx_array_get_(args.q_norm),
      mlx_array_get_(args.k_norm)};
  if (args.has_rope_freqs) {
    inputs.push_back(mlx_array_get_(args.rope_freqs));
    inputs.push_back(mlx_array_get_(args.mask));
    inputs.push_back(mlx_array_get_(args.residual));
    inputs.push_back(mlx_array_get_(args.post_attn_norm));
    return compiled_gemma4_q4_fixed_owner_attention_residual_freqs_masked()(inputs);
  }
  inputs.push_back(mlx_array_get_(args.mask));
  inputs.push_back(mlx_array_get_(args.residual));
  inputs.push_back(mlx_array_get_(args.post_attn_norm));
  return compiled_gemma4_q4_fixed_owner_attention_residual_default_rope_masked()(inputs);
}

ArrayVector gemma4_fixed_owner_attention_residual_impl(
    const go_mlx_gemma4_fixed_attention_args& args) {
  auto outputs = gemma4_fixed_owner_attention_impl(args);
  auto normed = mlx::core::fast::rms_norm(
      outputs[0],
      get_required(args.post_attn_norm, "post_attn_norm"),
      1e-6f);
  auto out = mlx::core::add(
      get_required(args.residual, "residual"),
      normed);
  return {out, outputs[1], outputs[2]};
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
        auto scaled_query = mlx::core::multiply(inputs[0], inputs[6]);

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
        auto mask = single_token_causal_mask(updated_keys.shape(2), inputs[5]);
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
        auto scaled_query = mlx::core::multiply(inputs[0], inputs[6]);

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

bool fixed_wide_matmul_attention_enabled() {
  const char* value = std::getenv("GO_MLX_ENABLE_FIXED_WIDE_MATMUL_ATTENTION");
  return value != nullptr && std::string(value) == "1";
}

bool fixed_row_cache_update_enabled() {
  const char* value = std::getenv("GO_MLX_ENABLE_FIXED_ROW_CACHE_UPDATE");
  return value != nullptr && std::string(value) == "1";
}

std::pair<mlx::core::array, mlx::core::array> gemma4_router_topk(
    const mlx::core::array& h,
    const go_mlx_gemma4_layer_args& args) {
  auto router_scale = get_required(args.router_scale, "router_scale");
  if (!args.has_router_scale_scaled) {
    router_scale = mlx::core::multiply(
        router_scale,
        mlx::core::array(args.router_root_size, router_scale.dtype()));
  }
  auto normed = mlx::core::fast::rms_norm(
      h,
      router_scale,
      args.router_eps);
  auto expert_scores = layer_linear_quantized(
      normed,
      args.router_weight,
      args.router_scales,
      args.router_biases,
      args.router_group_size,
      args.router_bits,
      "router_weight");
  const auto num_experts = expert_scores.shape(
      static_cast<int>(expert_scores.ndim()) - 1);
  auto top_k = args.router_top_k;
  if (top_k <= 0 || top_k > num_experts) {
    top_k = num_experts;
  }
  const auto kth = num_experts - top_k;
  auto partitioned = mlx::core::argpartition(expert_scores, kth, -1);
  auto top_k_indices = slice_last_dim(partitioned, kth, num_experts);
  auto top_k_weights = mlx::core::take_along_axis(expert_scores, top_k_indices, -1);
  auto weights = mlx::core::softmax(top_k_weights, std::vector<int>{-1}, false);
  if (valid_array(args.router_per_expert_scale)) {
    auto per_expert_scale = mlx::core::take(
        mlx_array_get_(args.router_per_expert_scale),
        top_k_indices,
        0);
    weights = mlx::core::multiply(weights, per_expert_scale);
  }
  return {top_k_indices, weights};
}

mlx::core::array gemma4_experts_graph(
    const mlx::core::array& x,
    const mlx::core::array& top_k_indices,
    const mlx::core::array& top_k_weights,
    const go_mlx_gemma4_layer_args& args) {
  auto expanded1 = mlx::core::expand_dims(x, 2);
  auto expanded = mlx::core::expand_dims(expanded1, 2);

  std::optional<mlx::core::array> gate;
  std::optional<mlx::core::array> up;
  if (valid_array(args.expert_gate_up_weight)) {
    auto gate_up = switch_linear(
        expanded,
        args.expert_gate_up_weight,
        args.expert_gate_up_scales,
        args.expert_gate_up_biases,
        args.expert_gate_up_bias,
        top_k_indices,
        args.expert_gate_up_group_size,
        args.expert_gate_up_bits,
        "expert_gate_up_weight");
    auto split = split_last_dim(gate_up);
    gate = split.first;
    up = split.second;
  } else {
    gate = switch_linear(
        expanded,
        args.expert_gate_weight,
        args.expert_gate_scales,
        args.expert_gate_biases,
        args.expert_gate_bias,
        top_k_indices,
        args.expert_gate_group_size,
        args.expert_gate_bits,
        "expert_gate_weight");
    up = switch_linear(
        expanded,
        args.expert_up_weight,
        args.expert_up_scales,
        args.expert_up_biases,
        args.expert_up_bias,
        top_k_indices,
        args.expert_up_group_size,
        args.expert_up_bits,
        "expert_up_weight");
  }
  auto activated = gelu_gate_mul(*gate, *up);
  auto down = switch_linear(
      activated,
      args.expert_down_weight,
      args.expert_down_scales,
      args.expert_down_biases,
      args.expert_down_bias,
      top_k_indices,
      args.expert_down_group_size,
      args.expert_down_bits,
      "expert_down_weight");
  auto down_squeezed = mlx::core::squeeze(down, 3);
  auto weights_expanded = mlx::core::expand_dims(top_k_weights, 3);
  auto weighted = mlx::core::multiply(weights_expanded, down_squeezed);
  return mlx::core::sum(weighted, -2, false);
}

mlx::core::array gemma4_mlp_graph(
    const mlx::core::array& x,
    const go_mlx_gemma4_layer_args& args) {
  auto gate = layer_linear_quantized(
      x,
      args.mlp_gate_weight,
      args.mlp_gate_scales,
      args.mlp_gate_biases,
      args.mlp_gate_group_size,
      args.mlp_gate_bits,
      "mlp_gate_weight");
  auto up = layer_linear_quantized(
      x,
      args.mlp_up_weight,
      args.mlp_up_scales,
      args.mlp_up_biases,
      args.mlp_up_group_size,
      args.mlp_up_bits,
      "mlp_up_weight");
  auto activated = gelu_gate_mul(gate, up);
  return layer_linear_quantized(
      activated,
      args.mlp_down_weight,
      args.mlp_down_scales,
      args.mlp_down_biases,
      args.mlp_down_group_size,
      args.mlp_down_bits,
      "mlp_down_weight");
}

mlx::core::array gemma4_ffn_residual_graph(
    const mlx::core::array& h,
    const go_mlx_gemma4_layer_args& args) {
  if (args.has_moe) {
    auto h1_in = mlx::core::fast::rms_norm(
        h,
        get_required(args.pre_ff_norm, "pre_ff_norm"),
        1e-6f);
    auto h1 = gemma4_mlp_graph(h1_in, args);
    auto h1_normed = mlx::core::fast::rms_norm(
        h1,
        get_required(args.post_ff_norm1, "post_ff_norm1"),
        1e-6f);

    auto h2_in = mlx::core::fast::rms_norm(
        h,
        get_required(args.pre_ff_norm2, "pre_ff_norm2"),
        1e-6f);
    auto router = gemma4_router_topk(h, args);
    auto h2 = gemma4_experts_graph(h2_in, router.first, router.second, args);
    auto h2_normed = mlx::core::fast::rms_norm(
        h2,
        get_required(args.post_ff_norm2, "post_ff_norm2"),
        1e-6f);

    auto combined = mlx::core::add(h1_normed, h2_normed);
    return mlx::core::fast::rms_norm(
        combined,
        get_required(args.post_ff_norm, "post_ff_norm"),
        1e-6f);
  }

  auto ff_in = mlx::core::fast::rms_norm(
      h,
      get_required(args.pre_ff_norm, "pre_ff_norm"),
      1e-6f);
  auto ff = gemma4_mlp_graph(ff_in, args);
  return mlx::core::fast::rms_norm(
      ff,
      get_required(args.post_ff_norm, "post_ff_norm"),
      1e-6f);
}

ArrayVector gemma4_decode_layer_impl_with_state(
    const go_mlx_gemma4_layer_args& args,
    const mlx::core::array& x,
    const mlx::core::array& prev_keys,
    const mlx::core::array& prev_values) {
  auto residual = x;
  auto offset = mlx::core::array(args.offset);

  auto normed = mlx::core::fast::rms_norm(
      x,
      get_required(args.input_norm, "input_norm"),
      1e-6f);
  const auto B = normed.shape(0);
  const auto L = normed.shape(1);

  auto q_proj = layer_linear_quantized(
      normed,
      args.q_weight,
      args.q_scales,
      args.q_biases,
      args.q_group_size,
      args.q_bits,
      "q_weight");
  auto q = mlx::core::as_strided(
      q_proj,
      mlx::core::Shape{B, args.num_attention_heads, L, args.head_dim},
      mlx::core::Strides{
          L * args.num_attention_heads * args.head_dim,
          args.head_dim,
          args.num_attention_heads * args.head_dim,
          1},
      0);
  q = mlx::core::fast::rms_norm(
      q,
      get_required(args.q_norm, "q_norm"),
      1e-6f);

  std::optional<mlx::core::array> keys;
  std::optional<mlx::core::array> values;
  if (args.owns_kv) {
    auto k_proj = layer_linear_quantized(
        normed,
        args.k_weight,
        args.k_scales,
        args.k_biases,
        args.k_group_size,
        args.k_bits,
        "k_weight");
    auto k = mlx::core::as_strided(
        k_proj,
        mlx::core::Shape{B, args.num_key_value_heads, L, args.head_dim},
        mlx::core::Strides{
            L * args.num_key_value_heads * args.head_dim,
            args.head_dim,
            args.num_key_value_heads * args.head_dim,
            1},
        0);
    k = mlx::core::fast::rms_norm(
        k,
        get_required(args.k_norm, "k_norm"),
        1e-6f);
    k = apply_gemma4_rope(k, args, offset);

    mlx::core::array v = k;
    if (!args.use_k_eq_v) {
      auto v_proj = layer_linear_quantized(
          normed,
          args.v_weight,
          args.v_scales,
          args.v_biases,
          args.v_group_size,
          args.v_bits,
          "v_weight");
      v = mlx::core::as_strided(
          v_proj,
          mlx::core::Shape{B, args.num_key_value_heads, L, args.head_dim},
          mlx::core::Strides{
              L * args.num_key_value_heads * args.head_dim,
              args.head_dim,
              args.num_key_value_heads * args.head_dim,
              1},
          0);
    }
    v = mlx::core::fast::rms_norm(v, std::nullopt, 1e-6f);
    if (args.fixed_kv) {
      keys = single_token_cache_update(prev_keys, k, offset);
      values = single_token_cache_update(prev_values, v, offset);
    } else if (args.has_prev) {
      keys = concat_cache_token(prev_keys, k);
      values = concat_cache_token(prev_values, v);
    } else {
      keys = k;
      values = v;
    }
  } else {
    keys = prev_keys;
    values = prev_values;
  }

  q = apply_gemma4_rope(q, args, offset);
  mlx::core::array attn = q;
  if (args.fixed_kv) {
    auto scaled_q = mlx::core::multiply(
        q,
        mlx::core::array(args.attention_scale, q.dtype()));
    std::optional<mlx::core::array> mask;
    if (args.has_fixed_mask) {
      mask = get_required(args.fixed_mask, "fixed_mask");
    } else {
      mask = single_token_causal_mask((*keys).shape(2), offset);
    }
    attn = mlx::core::fast::scaled_dot_product_attention(
        scaled_q,
        *keys,
        *values,
        1.0f,
        "array",
        mask);
  } else {
    attn = mlx::core::fast::scaled_dot_product_attention(
        q,
        *keys,
        *values,
        args.attention_scale);
  }
  auto transposed = mlx::core::transpose(attn, {0, 2, 1, 3});
  auto reshaped = mlx::core::reshape(
      transposed,
      mlx::core::Shape{B, L, args.num_attention_heads * args.head_dim});
  auto attn_out = layer_linear_quantized(
      reshaped,
      args.o_weight,
      args.o_scales,
      args.o_biases,
      args.o_group_size,
      args.o_bits,
      "o_weight");

  auto attn_normed = mlx::core::fast::rms_norm(
      attn_out,
      get_required(args.post_attn_norm, "post_attn_norm"),
      1e-6f);
  auto h = mlx::core::add(residual, attn_normed);

  auto ff_residual = gemma4_ffn_residual_graph(h, args);

  auto h_next = mlx::core::add(h, ff_residual);
  if (args.has_per_layer_input) {
    auto layer_gate = layer_linear_quantized(
        h_next,
        args.per_layer_gate_weight,
        args.per_layer_gate_scales,
        args.per_layer_gate_biases,
        args.per_layer_gate_group_size,
        args.per_layer_gate_bits,
        "per_layer_gate_weight");
    auto layer_mul = gelu_gate_mul(
        layer_gate,
        get_required(args.per_layer_input, "per_layer_input"));
    auto layer_projected = layer_linear_quantized(
        layer_mul,
        args.per_layer_projection_weight,
        args.per_layer_projection_scales,
        args.per_layer_projection_biases,
        args.per_layer_projection_group_size,
        args.per_layer_projection_bits,
        "per_layer_projection_weight");
    auto layer_normed = mlx::core::fast::rms_norm(
        layer_projected,
        get_required(args.post_per_layer_input_norm, "post_per_layer_input_norm"),
        1e-6f);
    h_next = mlx::core::add(h_next, layer_normed);
  }
  h_next = mlx::core::multiply(
      h_next,
      get_required(args.layer_scalar, "layer_scalar"));

  if (args.owns_kv) {
    return {h_next, *keys, *values};
  }
  return {h_next};
}

ArrayVector gemma4_decode_layer_impl(const go_mlx_gemma4_layer_args& args) {
  return gemma4_decode_layer_impl_with_state(
      args,
      get_required(args.x, "x"),
      get_required(args.prev_keys, "prev_keys"),
      get_required(args.prev_values, "prev_values"));
}

struct Gemma4LayerState {
  std::optional<mlx::core::array> keys;
  std::optional<mlx::core::array> values;
};

enum class Gemma4KVPath {
  Shared,
  Owner,
};

Gemma4KVPath gemma4_kv_path(const go_mlx_gemma4_layer_args& args) {
  switch (args.owns_kv) {
    case 0:
      return Gemma4KVPath::Shared;
    case 1:
      return Gemma4KVPath::Owner;
    default:
      throw std::runtime_error("mlx: Gemma 4 layer KV ownership flag is invalid");
      std::unreachable();
  }
}

mlx::core::array gemma4_fixed_greedy_token_impl(
    const go_mlx_gemma4_model_greedy_args& model_args,
    mlx_array* new_keys,
    mlx_array* new_values) {
  if (model_args.layer_count <= 0) {
    throw std::runtime_error("mlx: Gemma 4 model greedy layer count is invalid");
  }
  if (model_args.layers == nullptr || model_args.previous_kvs == nullptr) {
    throw std::runtime_error("mlx: Gemma 4 model greedy layer metadata is missing");
  }

  auto h = get_required(model_args.hidden, "hidden");
  std::vector<Gemma4LayerState> states(static_cast<size_t>(model_args.layer_count));
  for (int i = 0; i < model_args.layer_count; i++) {
    auto layer_args = model_args.layers[i];
    const auto kv_path = gemma4_kv_path(layer_args);
    mlx::core::array prev_keys = get_required(layer_args.prev_keys, "prev_keys");
    mlx::core::array prev_values = get_required(layer_args.prev_values, "prev_values");
    switch (kv_path) {
      case Gemma4KVPath::Shared: {
        const int prev = model_args.previous_kvs[i];
        if (prev < 0 || prev >= i ||
            !states[static_cast<size_t>(prev)].keys.has_value() ||
            !states[static_cast<size_t>(prev)].values.has_value()) {
          throw std::runtime_error("mlx: Gemma 4 model greedy shared KV owner is invalid");
        }
        prev_keys = *states[static_cast<size_t>(prev)].keys;
        prev_values = *states[static_cast<size_t>(prev)].values;
        break;
      }
      case Gemma4KVPath::Owner:
        break;
      default:
        throw std::runtime_error("mlx: Gemma 4 model greedy KV path is invalid");
        std::unreachable();
    }

    auto outputs = gemma4_decode_layer_impl_with_state(
        layer_args,
        h,
        prev_keys,
        prev_values);
    h = outputs[0];
    if (layer_args.owns_kv) {
      if (outputs.size() != 3) {
        throw std::runtime_error("mlx: Gemma 4 model greedy owner layer returned invalid KV outputs");
      }
      states[static_cast<size_t>(i)].keys = std::move(outputs[1]);
      states[static_cast<size_t>(i)].values = std::move(outputs[2]);
    }
  }

  for (int i = 0; i < model_args.layer_count; i++) {
    if (!states[static_cast<size_t>(i)].keys.has_value()) {
      continue;
    }
    mlx_array_set_(new_keys[i], std::move(*states[static_cast<size_t>(i)].keys));
    mlx_array_set_(new_values[i], std::move(*states[static_cast<size_t>(i)].values));
  }

  auto normed = mlx::core::fast::rms_norm(
      h,
      get_required(model_args.final_norm, "final_norm"),
      1e-6f);
  mlx::core::array logits = normed;
  if (model_args.output_quantized) {
    logits = q4_g64_linear(
        normed,
        get_required(model_args.output_weight, "output_weight"),
        get_required(model_args.output_scales, "output_scales"),
        get_required(model_args.output_biases, "output_biases"));
  } else {
    logits = dense_linear(
        normed,
        get_required(model_args.output_weight, "output_weight"));
  }
  if (model_args.has_suppress_token_ids) {
    logits = suppress_token_logits(
        logits,
        get_required(model_args.suppress_token_ids, "suppress_token_ids"));
  }
  return mlx::core::argmax(logits, -1, false);
}

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

extern "C" int go_mlx_gemma4_decode_layer(
    mlx_array* out,
    mlx_array* new_keys,
    mlx_array* new_values,
    const go_mlx_gemma4_layer_args* args,
    const mlx_stream stream) {
  try {
    (void)stream;
    if (args == nullptr) {
      throw std::runtime_error("mlx: Gemma 4 layer args are nil");
    }
    auto outputs = gemma4_decode_layer_impl(*args);
    mlx_array_set_(*out, std::move(outputs[0]));
    if (args->owns_kv) {
      mlx_array_set_(*new_keys, std::move(outputs[1]));
      mlx_array_set_(*new_values, std::move(outputs[2]));
    }
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int go_mlx_gemma4_fixed_greedy_token(
    mlx_array* token,
    mlx_array* new_keys,
    mlx_array* new_values,
    const go_mlx_gemma4_model_greedy_args* args,
    const mlx_stream stream) {
  try {
    (void)stream;
    if (args == nullptr) {
      throw std::runtime_error("mlx: Gemma 4 model greedy args are nil");
    }
    auto out = gemma4_fixed_greedy_token_impl(*args, new_keys, new_values);
    mlx_array_set_(*token, std::move(out));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int go_mlx_compiled_rms_norm_residual(
    mlx_array* out,
    const mlx_array residual,
    const mlx_array input,
    const mlx_array norm_weight,
    const mlx_stream stream) {
  try {
    (void)stream;
    ArrayVector inputs = {
        mlx_array_get_(residual),
        mlx_array_get_(input),
        mlx_array_get_(norm_weight)};
    auto outputs = compiled_rms_norm_residual()(inputs);
    mlx_array_set_(*out, std::move(outputs[0]));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int go_mlx_gemma4_fixed_owner_attention(
    mlx_array* out,
    mlx_array* new_keys,
    mlx_array* new_values,
    const go_mlx_gemma4_fixed_attention_args* args,
    const mlx_stream stream) {
  try {
    (void)stream;
    if (args == nullptr) {
      throw std::runtime_error("mlx: Gemma 4 fixed attention args are nil");
    }
    auto outputs = q4_fixed_owner_attention_available(*args)
        ? gemma4_q4_fixed_owner_attention_impl(*args)
        : gemma4_fixed_owner_attention_impl(*args);
    mlx_array_set_(*out, std::move(outputs[0]));
    mlx_array_set_(*new_keys, std::move(outputs[1]));
    mlx_array_set_(*new_values, std::move(outputs[2]));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int go_mlx_gemma4_fixed_owner_attention_residual(
    mlx_array* out,
    mlx_array* new_keys,
    mlx_array* new_values,
    const go_mlx_gemma4_fixed_attention_args* args,
    const mlx_stream stream) {
  try {
    (void)stream;
    if (args == nullptr) {
      throw std::runtime_error("mlx: Gemma 4 fixed attention residual args are nil");
    }
    auto outputs = q4_fixed_owner_attention_residual_available(*args)
        ? gemma4_q4_fixed_owner_attention_residual_impl(*args)
        : gemma4_fixed_owner_attention_residual_impl(*args);
    mlx_array_set_(*out, std::move(outputs[0]));
    mlx_array_set_(*new_keys, std::move(outputs[1]));
    mlx_array_set_(*new_values, std::move(outputs[2]));
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
