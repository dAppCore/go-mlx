// SPDX-Licence-Identifier: EUPL-1.2
//
// Correctness tests for go-mlx's fused/compiled single-token DECODE kernels
// (#65/#90/#91/#93): go/pkg/metal/decode_bridge.cpp.
//
// decode_bridge fuses the per-token decode tail — RMSNorm → (quantised) lm-head
// matmul → softcap / argmax / token suppression — into compiled MLX graphs so
// the hot single-token path runs as one kernel sequence. The graphs are built
// from standard MLX ops, so each has an exact unfused reference: we reproduce
// the same op chain in the C++ mlx::core domain and assert the fused output
// matches (logits within fp tolerance; argmax token-id exactly).
//
// As with the lm-head tests, inputs are built in mlx::core, crossed over the
// mlx_array C ABI to call the real extern "C" entry points, and read back for
// comparison. The kernels are NOT modified for the test.

#include "doctest/doctest.h"

#include <cmath>
#include <vector>

#include "mlx/c/array.h"
#include "mlx/c/private/array.h"
#include "mlx/c/stream.h"
#include "mlx/mlx.h"

using namespace mlx::core;

// decode_bridge declares only the attention entry points in its header; the
// last-token / last-logits family is extern "C" but not header-exposed, so we
// forward-declare the ones we exercise here.
extern "C" {
int go_mlx_compiled_dense_last_logits_softcap30(
    mlx_array* res, const mlx_array hidden, const mlx_array norm_weight,
    const mlx_array output_weight, const mlx_stream stream);
int go_mlx_compiled_dense_last_token(
    mlx_array* res, const mlx_array hidden, const mlx_array norm_weight,
    const mlx_array output_weight, const mlx_stream stream);
int go_mlx_compiled_q4_g64_last_token(
    mlx_array* res, const mlx_array hidden, const mlx_array norm_weight,
    const mlx_array output_weight, const mlx_array output_scales,
    const mlx_array output_biases, const mlx_stream stream);
int go_mlx_compiled_q8_g64_last_token(
    mlx_array* res, const mlx_array hidden, const mlx_array norm_weight,
    const mlx_array output_weight, const mlx_array output_scales,
    const mlx_array output_biases, const mlx_stream stream);
int go_mlx_compiled_q6_g64_last_token(
    mlx_array* res, const mlx_array hidden, const mlx_array norm_weight,
    const mlx_array output_weight, const mlx_array output_scales,
    const mlx_array output_biases, const mlx_stream stream);
int go_mlx_compiled_dense_last_token_suppressed(
    mlx_array* res, const mlx_array hidden, const mlx_array norm_weight,
    const mlx_array output_weight, const mlx_array suppress_token_ids,
    const mlx_stream stream);
int go_mlx_compiled_q4_g64_last_token_suppressed(
    mlx_array* res, const mlx_array hidden, const mlx_array norm_weight,
    const mlx_array output_weight, const mlx_array output_scales,
    const mlx_array output_biases, const mlx_array suppress_token_ids,
    const mlx_stream stream);
int go_mlx_compiled_dense_mlp_gelu(
    mlx_array* res, const mlx_array input, const mlx_array gate_weight,
    const mlx_array up_weight, const mlx_array down_weight,
    const mlx_stream stream);
}

namespace {

mlx_array wrap(const array& a) { return mlx_array_new_(a); }

// RMSNorm matching mlx::core::fast::rms_norm(x, w, eps): x / sqrt(mean(x^2)+eps)
// then elementwise * w, computed with the generic ops so it is an independent
// reference for the fused fast-kernel path.
array rms_norm_ref(const array& x, const array& w, float eps) {
  array ms = mean(multiply(x, x), /*axis=*/-1, /*keepdims=*/true);
  array inv = rsqrt(add(ms, array(eps)));
  return multiply(multiply(x, inv), w);
}

// gelu_approx matching the bridge: 0.5*x*(1 + tanh(0.79788456*(x + 0.044715*x^3)))
array gelu_approx_ref(const array& x) {
  array x3 = multiply(multiply(x, x), x);
  array inner = add(x, multiply(x3, array(0.044715f)));
  array t = tanh(multiply(inner, array(0.7978845608028654f)));
  return multiply(multiply(x, array(0.5f)), add(t, array(1.0f)));
}

} // namespace

TEST_CASE("decode: dense last-logits softcap30 matches reference") {
  if (!is_available(Device::gpu)) {
    WARN_MESSAGE(false, "Metal GPU unavailable — skipping decode test");
    return;
  }

  const int H = 256;  // hidden size
  const int V = 512;  // vocab
  random::seed(11);
  array hidden = random::normal({1, H}) * array(0.5f);
  array norm_w = random::normal({H}) * array(0.1f) + array(1.0f);
  array out_w = random::normal({V, H}) * array(0.05f);  // [V,H], used transposed
  eval(hidden);
  eval(norm_w);
  eval(out_w);

  // Reference: rms_norm → matmul(normed, out_w^T) → 30*tanh(logits/30).
  array normed = rms_norm_ref(hidden, norm_w, 1e-6f);
  array logits = matmul(normed, transpose(out_w));
  array thirty = array(30.0f);
  array ref = multiply(tanh(divide(logits, thirty)), thirty);
  eval(ref);

  mlx_array res = mlx_array_new_();
  mlx_array h = wrap(hidden), nw = wrap(norm_w), ow = wrap(out_w);
  mlx_stream s = mlx_default_gpu_stream_new();
  int rc = go_mlx_compiled_dense_last_logits_softcap30(&res, h, nw, ow, s);
  REQUIRE(rc == 0);

  array& got = mlx_array_get_(res);
  eval(got);
  REQUIRE(got.size() == static_cast<size_t>(V));
  const float* gp = got.data<float>();
  const float* rp = ref.data<float>();
  for (int i = 0; i < V; ++i) {
    CHECK(gp[i] == doctest::Approx(rp[i]).epsilon(2e-3));
    // softcap30 bounds every logit to (-30, 30).
    CHECK(std::abs(gp[i]) < 30.0f + 1e-3f);
  }

  mlx_array_free(res);
  mlx_array_free(h);
  mlx_array_free(nw);
  mlx_array_free(ow);
  mlx_stream_free(s);
}

TEST_CASE("decode: dense last-token argmax matches reference") {
  if (!is_available(Device::gpu)) {
    WARN_MESSAGE(false, "Metal GPU unavailable — skipping decode test");
    return;
  }

  const int H = 256;
  const int V = 1024;
  random::seed(13);
  array hidden = random::normal({1, H}) * array(0.5f);
  array norm_w = random::normal({H}) * array(0.1f) + array(1.0f);
  array out_w = random::normal({V, H}) * array(0.05f);
  eval(hidden);
  eval(norm_w);
  eval(out_w);

  array normed = rms_norm_ref(hidden, norm_w, 1e-6f);
  array logits = matmul(normed, transpose(out_w));
  array ref_tok = argmax(logits, -1, false);
  eval(ref_tok);

  mlx_array res = mlx_array_new_();
  mlx_array h = wrap(hidden), nw = wrap(norm_w), ow = wrap(out_w);
  mlx_stream s = mlx_default_gpu_stream_new();
  int rc = go_mlx_compiled_dense_last_token(&res, h, nw, ow, s);
  REQUIRE(rc == 0);

  array& got = mlx_array_get_(res);
  eval(got);
  // argmax over a continuous random distribution: a single unambiguous winner.
  CHECK(got.item<uint32_t>() == ref_tok.item<uint32_t>());

  mlx_array_free(res);
  mlx_array_free(h);
  mlx_array_free(nw);
  mlx_array_free(ow);
  mlx_stream_free(s);
}

TEST_CASE("decode: q4 g64 last-token argmax matches dequantised reference") {
  if (!is_available(Device::gpu)) {
    WARN_MESSAGE(false, "Metal GPU unavailable — skipping decode test");
    return;
  }

  const int H = 256;  // must be a multiple of group_size (64)
  const int V = 512;
  random::seed(17);
  array hidden = random::normal({1, H}) * array(0.5f);
  array norm_w = random::normal({H}) * array(0.1f) + array(1.0f);
  array out_w_full = random::normal({V, H}) * array(0.05f);
  eval(hidden);
  eval(norm_w);
  eval(out_w_full);

  std::vector<array> q = quantize(out_w_full, 64, 4);
  array w = q[0], scales = q[1], biases = q[2];
  eval(w);
  eval(scales);
  eval(biases);

  // Reference uses the SAME quantised weights via quantized_matmul (transpose
  // = true, g=64, 4-bit, affine) — exactly what the fused graph builds — so the
  // comparison isolates the fused kernel from the quantiser.
  array normed = rms_norm_ref(hidden, norm_w, 1e-6f);
  array logits = quantized_matmul(normed, w, scales, biases, true, 64, 4, "affine");
  array ref_tok = argmax(logits, -1, false);
  eval(ref_tok);

  mlx_array res = mlx_array_new_();
  mlx_array h = wrap(hidden), nw = wrap(norm_w), ow = wrap(w),
            sc = wrap(scales), bi = wrap(biases);
  mlx_stream s = mlx_default_gpu_stream_new();
  int rc = go_mlx_compiled_q4_g64_last_token(&res, h, nw, ow, sc, bi, s);
  REQUIRE(rc == 0);

  array& got = mlx_array_get_(res);
  eval(got);
  CHECK(got.item<uint32_t>() == ref_tok.item<uint32_t>());

  mlx_array_free(res);
  mlx_array_free(h);
  mlx_array_free(nw);
  mlx_array_free(ow);
  mlx_array_free(sc);
  mlx_array_free(bi);
  mlx_stream_free(s);
}

TEST_CASE("decode: q8/q6 g64 last-token argmax matches dequantised reference") {
  if (!is_available(Device::gpu)) {
    WARN_MESSAGE(false, "Metal GPU unavailable — skipping decode test");
    return;
  }

  // Same fused-vs-quantized_matmul oracle as the q4 case; the q8 and q6 paths
  // are template instantiations (Bits=8/6) of the same compiled decode graph.
  using Entry = int (*)(mlx_array*, mlx_array, mlx_array, mlx_array, mlx_array,
                        mlx_array, mlx_stream);
  struct Variant {
    const char* name;
    int bits;
    Entry fn;
  };
  const Variant variants[] = {
      {"q8", 8, &go_mlx_compiled_q8_g64_last_token},
      {"q6", 6, &go_mlx_compiled_q6_g64_last_token},
  };

  const int H = 256, V = 512;
  for (const Variant& var : variants) {
    CAPTURE(var.name);
    random::seed(19 + var.bits);
    array hidden = random::normal({1, H}) * array(0.5f);
    array norm_w = random::normal({H}) * array(0.1f) + array(1.0f);
    array out_w_full = random::normal({V, H}) * array(0.05f);
    eval(hidden);
    eval(norm_w);
    eval(out_w_full);

    std::vector<array> q = quantize(out_w_full, 64, var.bits);
    array w = q[0], scales = q[1], biases = q[2];
    eval(w);
    eval(scales);
    eval(biases);

    array normed = rms_norm_ref(hidden, norm_w, 1e-6f);
    array logits = quantized_matmul(normed, w, scales, biases, true, 64,
                                    var.bits, "affine");
    array ref_tok = argmax(logits, -1, false);
    eval(ref_tok);

    mlx_array res = mlx_array_new_();
    mlx_array h = wrap(hidden), nw = wrap(norm_w), ow = wrap(w),
              sc = wrap(scales), bi = wrap(biases);
    mlx_stream s = mlx_default_gpu_stream_new();
    int rc = var.fn(&res, h, nw, ow, sc, bi, s);
    REQUIRE(rc == 0);
    array& got = mlx_array_get_(res);
    eval(got);
    CHECK(got.item<uint32_t>() == ref_tok.item<uint32_t>());

    mlx_array_free(res);
    mlx_array_free(h);
    mlx_array_free(nw);
    mlx_array_free(ow);
    mlx_array_free(sc);
    mlx_array_free(bi);
    mlx_stream_free(s);
  }
}

TEST_CASE("decode: dense suppressed last-token forces a non-argmax winner") {
  if (!is_available(Device::gpu)) {
    WARN_MESSAGE(false, "Metal GPU unavailable — skipping decode test");
    return;
  }

  const int H = 256, V = 512;
  random::seed(23);
  array hidden = random::normal({1, H}) * array(0.5f);
  array norm_w = random::normal({H}) * array(0.1f) + array(1.0f);
  array out_w = random::normal({V, H}) * array(0.05f);
  eval(hidden);
  eval(norm_w);
  eval(out_w);

  // The unsuppressed argmax is the natural winner; suppress exactly that token
  // and the kernel must return the SECOND-best — a strong test of the in-graph
  // put_along_axis(-inf) suppression, not just the matmul.
  array normed = rms_norm_ref(hidden, norm_w, 1e-6f);
  array logits = matmul(normed, transpose(out_w));  // [1, V]
  array top1 = argmax(logits, -1, false);
  eval(top1);
  uint32_t top1_id = top1.item<uint32_t>();

  // suppress_token_ids: a 1-D int32 array of token ids to mask to -inf. The
  // bridge reshapes it to [..., n_suppress] and put_along_axis(-inf). We mirror
  // that exactly: one id (the natural argmax), so the winner must shift.
  array suppress = reshape(astype(top1, int32), {1});  // [1]
  array idx = reshape(suppress, {1, 1});               // [1, n_suppress=1]
  array updates = full({1, 1}, -std::numeric_limits<float>::infinity(), float32);
  array masked = put_along_axis(logits, idx, updates, -1);
  array ref_tok = argmax(masked, -1, false);
  eval(ref_tok);
  uint32_t ref_id = ref_tok.item<uint32_t>();
  REQUIRE(ref_id != top1_id);  // suppression genuinely moved the winner

  mlx_array res = mlx_array_new_();
  mlx_array h = wrap(hidden), nw = wrap(norm_w), ow = wrap(out_w),
            sp = wrap(suppress);
  mlx_stream s = mlx_default_gpu_stream_new();
  int rc = go_mlx_compiled_dense_last_token_suppressed(&res, h, nw, ow, sp, s);
  REQUIRE(rc == 0);
  array& got = mlx_array_get_(res);
  eval(got);
  CHECK(got.item<uint32_t>() == ref_id);
  CHECK(got.item<uint32_t>() != top1_id);

  mlx_array_free(res);
  mlx_array_free(h);
  mlx_array_free(nw);
  mlx_array_free(ow);
  mlx_array_free(sp);
  mlx_stream_free(s);
}

TEST_CASE("decode: dense MLP gelu (SwiGLU) matches reference") {
  if (!is_available(Device::gpu)) {
    WARN_MESSAGE(false, "Metal GPU unavailable — skipping decode test");
    return;
  }

  const int H = 128;   // model dim
  const int FF = 256;  // intermediate dim
  random::seed(29);
  array x = random::normal({1, H}) * array(0.5f);
  array gate_w = random::normal({FF, H}) * array(0.05f);  // [FF,H] (transposed in linear)
  array up_w = random::normal({FF, H}) * array(0.05f);
  array down_w = random::normal({H, FF}) * array(0.05f);  // [H,FF]
  eval(x);
  eval(gate_w);
  eval(up_w);
  eval(down_w);

  // Reference SwiGLU: down( gelu(x·gateᵀ) * (x·upᵀ) ).
  array gate = matmul(x, transpose(gate_w));
  array up = matmul(x, transpose(up_w));
  array activated = multiply(gelu_approx_ref(gate), up);
  array ref = matmul(activated, transpose(down_w));
  eval(ref);

  mlx_array res = mlx_array_new_();
  mlx_array xa = wrap(x), gw = wrap(gate_w), uw = wrap(up_w), dw = wrap(down_w);
  mlx_stream s = mlx_default_gpu_stream_new();
  int rc = go_mlx_compiled_dense_mlp_gelu(&res, xa, gw, uw, dw, s);
  REQUIRE(rc == 0);

  array& got = mlx_array_get_(res);
  eval(got);
  REQUIRE(got.size() == static_cast<size_t>(H));
  const float* gp = got.data<float>();
  const float* rp = ref.data<float>();
  for (int i = 0; i < H; ++i) {
    CHECK(gp[i] == doctest::Approx(rp[i]).epsilon(2e-3));
  }

  mlx_array_free(res);
  mlx_array_free(xa);
  mlx_array_free(gw);
  mlx_array_free(uw);
  mlx_array_free(dw);
  mlx_stream_free(s);
}
