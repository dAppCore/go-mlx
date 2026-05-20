// SPDX-Licence-Identifier: EUPL-1.2

#include <exception>
#include <vector>

#include "mlx/c/error.h"
#include "mlx/c/private/mlx.h"
#include "mlx/compile.h"
#include "mlx/mlx.h"

namespace {

using ArrayVector = std::vector<mlx::core::array>;

mlx::core::array scalar_like(const mlx::core::array& x, float value) {
  return mlx::core::array(value, x.dtype());
}

mlx::core::array gelu_approx(
    const mlx::core::array& x,
    mlx::core::StreamOrDevice s = {}) {
  auto x2 = mlx::core::multiply(x, x, s);
  auto x3 = mlx::core::multiply(x2, x, s);
  auto inner = mlx::core::add(
      x,
      mlx::core::multiply(x3, scalar_like(x, 0.044715f), s),
      s);
  auto scaled = mlx::core::multiply(
      inner,
      scalar_like(x, 0.7978845608028654f),
      s);
  auto t = mlx::core::tanh(scaled, s);
  auto one_plus = mlx::core::add(t, scalar_like(x, 1.0f), s);
  auto half_x = mlx::core::multiply(x, scalar_like(x, 0.5f), s);
  return mlx::core::multiply(half_x, one_plus, s);
}

const std::function<ArrayVector(const ArrayVector&)>& compiled_gelu_gate_mul() {
  static const auto fn = mlx::core::compile(
      [](const ArrayVector& inputs) -> ArrayVector {
        return {mlx::core::multiply(gelu_approx(inputs[0]), inputs[1])};
      },
      true);
  return fn;
}

const std::function<ArrayVector(const ArrayVector&)>& compiled_silu_gate_mul() {
  static const auto fn = mlx::core::compile(
      [](const ArrayVector& inputs) -> ArrayVector {
        auto sigmoid = mlx::core::sigmoid(inputs[0]);
        auto activated = mlx::core::multiply(inputs[0], sigmoid);
        return {mlx::core::multiply(activated, inputs[1])};
      },
      true);
  return fn;
}

} // namespace

extern "C" int go_mlx_gelu_gate_mul(
    mlx_array* res,
    const mlx_array gate,
    const mlx_array up,
    const mlx_stream stream) {
  try {
    (void)stream;
    ArrayVector inputs = {mlx_array_get_(gate), mlx_array_get_(up)};
    auto outputs = compiled_gelu_gate_mul()(inputs);
    mlx_array_set_(*res, std::move(outputs[0]));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int go_mlx_silu_gate_mul(
    mlx_array* res,
    const mlx_array gate,
    const mlx_array up,
    const mlx_stream stream) {
  try {
    (void)stream;
    ArrayVector inputs = {mlx_array_get_(gate), mlx_array_get_(up)};
    auto outputs = compiled_silu_gate_mul()(inputs);
    mlx_array_set_(*res, std::move(outputs[0]));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
