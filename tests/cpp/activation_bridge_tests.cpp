// SPDX-Licence-Identifier: EUPL-1.2
//
// Correctness tests for go-mlx's fused gated-activation kernels:
// go/pkg/metal/activation_bridge.cpp — go_mlx_gelu_gate_mul / go_mlx_silu_gate_mul.
//
// These bridges fuse the SwiGLU/GeGLU elementwise tail — activation(gate) * up —
// into a single compiled MLX graph so the gated-MLP middle runs as one kernel.
// Each has an exact unfused reference: we recompute gelu(gate)*up and
// silu(gate)*up from generic mlx::core primitives and assert the fused output
// matches within fp tolerance.
//
// As with the decode + lm-head tests, inputs are built in mlx::core, crossed
// over the mlx_array C ABI to call the real extern "C" entry points, and read
// back for comparison. The kernels are NOT modified for the test.

#include "doctest/doctest.h"

#include <cmath>
#include <vector>

#include "mlx/c/array.h"
#include "mlx/c/private/array.h"
#include "mlx/c/stream.h"
#include "mlx/mlx.h"

using namespace mlx::core;

// activation_bridge exposes no header; the entries are extern "C". Declare them.
extern "C" {
int go_mlx_gelu_gate_mul(mlx_array* res, const mlx_array gate,
                         const mlx_array up, const mlx_stream stream);
int go_mlx_silu_gate_mul(mlx_array* res, const mlx_array gate,
                         const mlx_array up, const mlx_stream stream);
}

namespace {

mlx_array wrap(const array& a) { return mlx_array_new_(a); }

// gelu_approx matching the bridge exactly:
// 0.5*x*(1 + tanh(0.79788456*(x + 0.044715*x^3)))
array gelu_approx_ref(const array& x) {
  array x3 = multiply(multiply(x, x), x);
  array inner = add(x, multiply(x3, array(0.044715f)));
  array t = tanh(multiply(inner, array(0.7978845608028654f)));
  return multiply(multiply(x, array(0.5f)), add(t, array(1.0f)));
}

// silu/swish: x * sigmoid(x).
array silu_ref(const array& x) { return multiply(x, sigmoid(x)); }

// Drive a (gate, up) -> result bridge and read the output as floats.
std::vector<float> run_gate_mul(
    int (*fn)(mlx_array*, mlx_array, mlx_array, mlx_stream), const array& gate,
    const array& up, int* rc_out) {
  mlx_array res = mlx_array_new_();
  mlx_array g = wrap(gate), u = wrap(up);
  mlx_stream s = mlx_default_gpu_stream_new();
  int rc = fn(&res, g, u, s);
  *rc_out = rc;
  std::vector<float> out;
  if (rc == 0) {
    array& r = mlx_array_get_(res);
    eval(r);
    out.assign(r.data<float>(), r.data<float>() + r.size());
  }
  mlx_array_free(res);
  mlx_array_free(g);
  mlx_array_free(u);
  mlx_stream_free(s);
  return out;
}

}  // namespace

TEST_CASE("activation: gelu gate-mul matches independent gelu(gate)*up") {
  if (!is_available(Device::gpu)) {
    WARN_MESSAGE(false, "Metal GPU unavailable — skipping activation test");
    return;
  }

  const int N = 1, FF = 320;  // single-token gated-MLP row
  random::seed(51);
  array gate = random::normal({N, FF}) * array(1.5f);  // wide range hits both
  array up = random::normal({N, FF}) * array(0.7f);     // tanh saturation arms
  eval(gate);
  eval(up);

  array ref = multiply(gelu_approx_ref(gate), up);
  eval(ref);

  int rc = -1;
  std::vector<float> got = run_gate_mul(&go_mlx_gelu_gate_mul, gate, up, &rc);
  REQUIRE(rc == 0);
  REQUIRE(got.size() == static_cast<size_t>(N * FF));
  const float* rp = ref.data<float>();
  for (size_t i = 0; i < got.size(); ++i)
    CHECK(got[i] == doctest::Approx(rp[i]).epsilon(2e-3));
}

TEST_CASE("activation: silu gate-mul matches independent silu(gate)*up") {
  if (!is_available(Device::gpu)) {
    WARN_MESSAGE(false, "Metal GPU unavailable — skipping activation test");
    return;
  }

  const int N = 1, FF = 320;
  random::seed(53);
  array gate = random::normal({N, FF}) * array(1.5f);
  array up = random::normal({N, FF}) * array(0.7f);
  eval(gate);
  eval(up);

  array ref = multiply(silu_ref(gate), up);
  eval(ref);

  int rc = -1;
  std::vector<float> got = run_gate_mul(&go_mlx_silu_gate_mul, gate, up, &rc);
  REQUIRE(rc == 0);
  REQUIRE(got.size() == static_cast<size_t>(N * FF));
  const float* rp = ref.data<float>();
  for (size_t i = 0; i < got.size(); ++i)
    CHECK(got[i] == doctest::Approx(rp[i]).epsilon(2e-3));
}

TEST_CASE("activation: gate-mul is shapeless — a second shape reuses the graph") {
  if (!is_available(Device::gpu)) {
    WARN_MESSAGE(false, "Metal GPU unavailable — skipping activation test");
    return;
  }

  // The compiled graphs are built shapeless (compile(..., /*shapeless=*/true)),
  // so the same cached function must serve a different shape on the next call.
  // Exercise a multi-row [N>1, FF] input through both bridges.
  const int N = 3, FF = 128;
  random::seed(55);
  array gate = random::normal({N, FF}) * array(1.2f);
  array up = random::normal({N, FF}) * array(0.6f);
  eval(gate);
  eval(up);

  array gelu_ref = multiply(gelu_approx_ref(gate), up);
  array silu_ref_v = multiply(silu_ref(gate), up);
  eval(gelu_ref);
  eval(silu_ref_v);

  int rc = -1;
  std::vector<float> g = run_gate_mul(&go_mlx_gelu_gate_mul, gate, up, &rc);
  REQUIRE(rc == 0);
  REQUIRE(g.size() == static_cast<size_t>(N * FF));
  const float* grp = gelu_ref.data<float>();
  for (size_t i = 0; i < g.size(); ++i)
    CHECK(g[i] == doctest::Approx(grp[i]).epsilon(2e-3));

  std::vector<float> sv = run_gate_mul(&go_mlx_silu_gate_mul, gate, up, &rc);
  REQUIRE(rc == 0);
  REQUIRE(sv.size() == static_cast<size_t>(N * FF));
  const float* srp = silu_ref_v.data<float>();
  for (size_t i = 0; i < sv.size(); ++i)
    CHECK(sv[i] == doctest::Approx(srp[i]).epsilon(2e-3));
}
