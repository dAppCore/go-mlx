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
#include <limits>
#include <vector>

#include "mlx/c/array.h"
#include "mlx/c/private/array.h"
#include "mlx/c/stream.h"
#include "mlx/mlx.h"

// The attention entry points ARE header-exposed; the last-token / last-logits
// family is extern "C" but not, so it is forward-declared below.
#include "decode_bridge.h"

using namespace mlx::core;

// decode_bridge declares only the attention entry points in its header; the
// last-token / last-logits family is extern "C" but not header-exposed, so we
// forward-declare the ones we exercise here.
extern "C" {
int go_mlx_compiled_greedy_decode_token(
    mlx_array* res, const mlx_array logits, const mlx_stream stream);
int go_mlx_compiled_dense_last_logits_softcap30(
    mlx_array* res, const mlx_array hidden, const mlx_array norm_weight,
    const mlx_array output_weight, const mlx_stream stream);
int go_mlx_compiled_q4_g64_last_logits_softcap30(
    mlx_array* res, const mlx_array hidden, const mlx_array norm_weight,
    const mlx_array output_weight, const mlx_array output_scales,
    const mlx_array output_biases, const mlx_stream stream);
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
int go_mlx_compiled_q8_g64_last_token_suppressed(
    mlx_array* res, const mlx_array hidden, const mlx_array norm_weight,
    const mlx_array output_weight, const mlx_array output_scales,
    const mlx_array output_biases, const mlx_array suppress_token_ids,
    const mlx_stream stream);
int go_mlx_compiled_q6_g64_last_token_suppressed(
    mlx_array* res, const mlx_array hidden, const mlx_array norm_weight,
    const mlx_array output_weight, const mlx_array output_scales,
    const mlx_array output_biases, const mlx_array suppress_token_ids,
    const mlx_stream stream);
int go_mlx_compiled_dense_mlp_gelu(
    mlx_array* res, const mlx_array input, const mlx_array gate_weight,
    const mlx_array up_weight, const mlx_array down_weight,
    const mlx_stream stream);
int go_mlx_compiled_q4_g64_mlp_gelu(
    mlx_array* res, const mlx_array input, const mlx_array gate_weight,
    const mlx_array gate_scales, const mlx_array gate_biases,
    const mlx_array up_weight, const mlx_array up_scales,
    const mlx_array up_biases, const mlx_array down_weight,
    const mlx_array down_scales, const mlx_array down_biases,
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

// --- single-token attention reference ---------------------------------------
//
// One independent oracle for the whole fixed_single_token_attention family. The
// kernel has THREE internal mechanisms (fast::scaled_dot_product_attention, the
// explicit matmul+softmax wide path, and the row-reshape cache update) selected
// by the global diagnostics + head_dim, but they all compute the SAME maths:
//   1. write the new key/value into the cache at `offset` along the token axis,
//   2. repeat KV heads to match the query heads (GQA),
//   3. scale the query, score = q·kᵀ, add the mask, softmax, out = w·v.
// We reassemble that from unfused primitives so the reference is independent of
// every fused leaf at once. `mask` is the additive bias added to the scores
// (causal `where(idx<=offset, 0, -inf)` or any external mask the caller passes).
//
// cache shapes: [B, n_kv_heads, capacity, head_dim]; query: [B, n_q_heads, 1,
// head_dim]; new key/value: [B, n_kv_heads, 1, head_dim]; offset: scalar int32.
array single_token_attention_ref(
    const array& query,        // [B, Hq, 1, D]
    const array& key_cache,    // [B, Hkv, C, D]
    const array& value_cache,  // [B, Hkv, C, D]
    const array& new_key,      // [B, Hkv, 1, D]
    const array& new_value,    // [B, Hkv, 1, D]
    const array& offset,       // scalar int32 (token index to write at)
    float scale,
    const array& add_mask) {   // [1,1,1,C] or broadcastable additive bias
  // Write new K/V at `offset` along the token axis (axis 2) via put_along_axis,
  // exactly as the kernel's single_token_cache_update does.
  array off_idx = reshape(offset, Shape{1, 1, 1, 1});
  array k_indices = broadcast_to(off_idx, new_key.shape());
  array v_indices = broadcast_to(off_idx, new_value.shape());
  array keys = put_along_axis(key_cache, k_indices, new_key, 2);
  array values = put_along_axis(value_cache, v_indices, new_value, 2);

  // GQA repeat: tile each KV head query_heads/key_heads times.
  const int hq = query.shape(1);
  const int hkv = keys.shape(1);
  const int factor = hq / hkv;
  if (factor > 1) {
    const int b = keys.shape(0), c = keys.shape(2), d = keys.shape(3);
    array ke = expand_dims(keys, 2);
    keys = reshape(broadcast_to(ke, Shape{b, hkv, factor, c, d}),
                   Shape{b, hkv * factor, c, d});
    array ve = expand_dims(values, 2);
    values = reshape(broadcast_to(ve, Shape{b, hkv, factor, c, d}),
                     Shape{b, hkv * factor, c, d});
  }

  array sq = multiply(query, array(scale, query.dtype()));
  array scores = matmul(sq, transpose(keys, {0, 1, 3, 2}));  // [B,Hq,1,C]
  scores = add(scores, astype(add_mask, scores.dtype()));
  array weights = softmax(scores, std::vector<int>{-1}, true);
  return matmul(weights, values);  // [B,Hq,1,D]
}

// Build the causal additive mask the kernel builds internally for the unmasked
// path: where(token_idx <= offset, 0, finfo.min). Shape [1,1,1,C].
array causal_mask_ref(int capacity, const array& offset, Dtype dtype) {
  array idx = reshape(arange(0, capacity, 1), Shape{1, 1, 1, capacity});
  array valid = less_equal(idx, offset);
  return where(valid, array(0.0f, dtype),
               array(finfo(dtype).min, dtype));
}

// Read a fused mlx_array result back into a flat float32 vector for comparison.
std::vector<float> to_floats(mlx_array& a) {
  array& r = mlx_array_get_(a);
  array rf = astype(r, float32);
  eval(rf);
  return std::vector<float>(rf.data<float>(), rf.data<float>() + rf.size());
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

// =============================================================================
// Single-token ATTENTION family (#65/#90/#91/#93) — the real custom decode
// logic. go_mlx_compiled_fixed_single_token_attention is ONE entry with SIX
// leaves selected by (use_matmul, use_row_update, has_mask):
//   use_matmul     = key_cache.shape(3) >= 512 && fixed_wide_matmul_attention
//   use_row_update = !use_matmul && fixed_row_cache_update
//   has_mask       picks the _masked sibling.
// The three mechanisms compute identical maths, so single_token_attention_ref
// (assembled from unfused primitives) is an independent oracle for all of them.
// To make a wiring bug FAIL: invalid/future cache slots (token index > offset)
// are poisoned with large values — if the mask polarity or offset binding is
// wrong the output explodes and diverges from the reference.
// =============================================================================

namespace {

// Run go_mlx_compiled_fixed_single_token_attention and return the attention
// output (out[0]) as floats. Drives the EXACT shipped entry point.
std::vector<float> run_fixed_attention(
    const array& query, const array& key_cache, const array& value_cache,
    const array& new_key, const array& new_value, const array& offset,
    float scale, const array& mask, bool has_mask, int* rc_out) {
  mlx_array out = mlx_array_new_();
  mlx_array nk = mlx_array_new_();
  mlx_array nv = mlx_array_new_();
  mlx_array q = wrap(query), kc = wrap(key_cache), vc = wrap(value_cache),
            k = wrap(new_key), v = wrap(new_value), off = wrap(offset);
  array scale_arr = array(scale);
  eval(scale_arr);
  mlx_array sc = wrap(scale_arr), mk = wrap(mask);
  mlx_stream s = mlx_default_gpu_stream_new();

  int rc = go_mlx_compiled_fixed_single_token_attention(
      &out, &nk, &nv, q, kc, vc, k, v, off, sc, mk, has_mask ? 1 : 0, s);
  *rc_out = rc;
  std::vector<float> got;
  if (rc == 0) got = to_floats(out);

  mlx_array_free(out);
  mlx_array_free(nk);
  mlx_array_free(nv);
  mlx_array_free(q);
  mlx_array_free(kc);
  mlx_array_free(vc);
  mlx_array_free(k);
  mlx_array_free(v);
  mlx_array_free(off);
  mlx_array_free(sc);
  mlx_array_free(mk);
  mlx_stream_free(s);
  return got;
}

// Build a [B,Hkv,C,D] cache whose VALID slots (token index <= offset) are small
// random values and whose FUTURE slots (> offset) are poisoned huge, so a
// broken causal mask leaks the poison and diverges from the oracle.
struct PoisonedCache {
  array keys = array(0.0f);
  array values = array(0.0f);
};
PoisonedCache poisoned_cache(int B, int Hkv, int C, int D, int offset,
                             unsigned seed) {
  random::seed(seed);
  array kbase = random::normal({B, Hkv, C, D}) * array(0.3f);
  array vbase = random::normal({B, Hkv, C, D}) * array(0.3f);
  array idx = reshape(arange(0, C, 1), Shape{1, 1, C, 1});
  array future = greater(idx, array(offset));  // token index > offset
  array poison = array(1.0e4f);
  PoisonedCache pc;
  pc.keys = where(future, poison, kbase);
  pc.values = where(future, poison, vbase);
  eval(pc.keys);
  eval(pc.values);
  return pc;
}

}  // namespace

TEST_CASE("decode: fixed single-token attention — all six dispatch leaves") {
  if (!is_available(Device::gpu)) {
    WARN_MESSAGE(false, "Metal GPU unavailable — skipping attention test");
    return;
  }

  const int B = 1, Hkv = 2, C = 8, D = 16;  // small head_dim: matmul leaf OFF
  const int Hq = 4;                          // GQA factor 2 -> exercises repeat_kv
  const int offset = 3;                      // slots 0..3 valid, 4..7 poisoned
  const float scale = 0.25f;

  // (use_row_update, has_mask) for the small-head_dim leaves; the matmul leaves
  // are driven separately below because they need head_dim >= 512.
  struct Leaf { int row_update; int has_mask; const char* name; };
  const Leaf leaves[] = {
      {0, 0, "default-sdpa"},
      {0, 1, "default-masked"},
      {1, 0, "row-update"},
      {1, 1, "row-update-masked"},
  };

  unsigned seed = 101;
  for (const Leaf& leaf : leaves) {
    CAPTURE(leaf.name);
    // Isolate: set diagnostics for THIS leaf, reset to 0 after the case so the
    // global atomics never leak into a later doctest case.
    go_mlx_set_fixed_attention_diagnostics(0, leaf.row_update);

    random::seed(seed);
    array query = random::normal({B, Hq, 1, D}) * array(0.4f);
    array new_key = random::normal({B, Hkv, 1, D}) * array(0.3f);
    array new_value = random::normal({B, Hkv, 1, D}) * array(0.3f);
    array offset_a = array(static_cast<int32_t>(offset));
    eval(query);
    eval(new_key);
    eval(new_value);
    eval(offset_a);
    PoisonedCache pc = poisoned_cache(B, Hkv, C, D, offset, seed + 1);

    // EXTERNAL mask (has_mask): causal up to offset, but additionally mask a
    // middle slot (1) so the test proves the kernel applies the SUPPLIED mask,
    // not an internal causal one. Unmasked path: the kernel builds its own
    // causal mask, so the oracle uses the matching causal mask.
    array base_mask = causal_mask_ref(C, offset_a, float32);
    array one_hot =
        reshape(equal(arange(0, C, 1), array(1)), Shape{1, 1, 1, C});
    array mask = leaf.has_mask
                     ? where(one_hot, array(finfo(float32).min), base_mask)
                     : base_mask;
    eval(mask);

    array ref = single_token_attention_ref(query, pc.keys, pc.values, new_key,
                                            new_value, offset_a, scale, mask);
    eval(ref);

    int rc = -1;
    std::vector<float> got = run_fixed_attention(
        query, pc.keys, pc.values, new_key, new_value, offset_a, scale, mask,
        leaf.has_mask != 0, &rc);
    REQUIRE(rc == 0);
    REQUIRE(got.size() == ref.size());
    const float* rp = ref.data<float>();
    for (size_t i = 0; i < got.size(); ++i) {
      CHECK(got[i] == doctest::Approx(rp[i]).epsilon(3e-3));
      // Poison would push |out| toward 1e4; a correct mask keeps it small.
      CHECK(std::abs(got[i]) < 50.0f);
    }
    go_mlx_set_fixed_attention_diagnostics(0, 0);  // reset global state
    ++seed;
  }
}

TEST_CASE("decode: fixed single-token attention — wide matmul leaf (head_dim>=512)") {
  if (!is_available(Device::gpu)) {
    WARN_MESSAGE(false, "Metal GPU unavailable — skipping attention test");
    return;
  }

  // The matmul path is gated on key_cache.shape(3) >= 512 AND the toggle. Use a
  // single KV head + single query head here (the repeat_kv branch is covered by
  // the GQA case above) to keep the wide-D tensors affordable.
  const int B = 1, Hkv = 1, Hq = 1, C = 4, D = 512;
  const int offset = 2;
  const float scale = 0.04f;

  for (int has_mask = 0; has_mask <= 1; ++has_mask) {
    CAPTURE(has_mask);
    go_mlx_set_fixed_attention_diagnostics(/*wide_matmul=*/1, 0);

    unsigned seed = 211 + static_cast<unsigned>(has_mask);
    random::seed(seed);
    array query = random::normal({B, Hq, 1, D}) * array(0.1f);
    array new_key = random::normal({B, Hkv, 1, D}) * array(0.1f);
    array new_value = random::normal({B, Hkv, 1, D}) * array(0.1f);
    array offset_a = array(static_cast<int32_t>(offset));
    eval(query);
    eval(new_key);
    eval(new_value);
    eval(offset_a);
    PoisonedCache pc = poisoned_cache(B, Hkv, C, D, offset, seed + 1);

    array mask = causal_mask_ref(C, offset_a, float32);
    if (has_mask) {
      array one_hot = reshape(equal(arange(0, C, 1), array(1)),
                              Shape{1, 1, 1, C});
      mask = where(one_hot, array(finfo(float32).min), mask);
    }
    eval(mask);

    array ref = single_token_attention_ref(query, pc.keys, pc.values, new_key,
                                            new_value, offset_a, scale, mask);
    eval(ref);

    int rc = -1;
    std::vector<float> got = run_fixed_attention(
        query, pc.keys, pc.values, new_key, new_value, offset_a, scale, mask,
        has_mask != 0, &rc);
    REQUIRE(rc == 0);
    REQUIRE(got.size() == ref.size());
    const float* rp = ref.data<float>();
    for (size_t i = 0; i < got.size(); ++i) {
      CHECK(got[i] == doctest::Approx(rp[i]).epsilon(3e-3));
      CHECK(std::abs(got[i]) < 50.0f);
    }
    go_mlx_set_fixed_attention_diagnostics(0, 0);
  }
}

TEST_CASE("decode: fixed sliding-window single-token attention") {
  if (!is_available(Device::gpu)) {
    WARN_MESSAGE(false, "Metal GPU unavailable — skipping attention test");
    return;
  }

  // Sliding window: no causal mask. The cache is gathered by shift_indices then
  // the new token is placed at last_index; attention is over the WHOLE window.
  // Reference: gather cache, scatter the token in, attend over all C slots with
  // a zero (all-valid) additive mask.
  const int B = 1, H = 2, C = 6, D = 16;
  const float scale = 0.25f;
  go_mlx_set_fixed_attention_diagnostics(0, 0);

  random::seed(311);
  array query = random::normal({B, H, 1, D}) * array(0.4f);
  array new_key = random::normal({B, H, 1, D}) * array(0.3f);
  array new_value = random::normal({B, H, 1, D}) * array(0.3f);
  array key_cache = random::normal({B, H, C, D}) * array(0.3f);
  array value_cache = random::normal({B, H, C, D}) * array(0.3f);
  // Roll the window left by one: shift_indices = [1,2,3,4,5,5]; last slot is
  // overwritten by the new token (last_index = C-1).
  std::vector<int32_t> shifts = {1, 2, 3, 4, 5, 5};
  array shift_indices = array(shifts.data(), {C}, int32);
  array last_index = array(static_cast<int32_t>(C - 1));
  eval(query);
  eval(new_key);
  eval(new_value);
  eval(key_cache);
  eval(value_cache);
  eval(shift_indices);
  eval(last_index);

  // Reference window cache: take(cache, shift_indices, axis=2), then write the
  // new token at last_index (axis 2).
  array shifted_k = take(key_cache, shift_indices, 2);
  array shifted_v = take(value_cache, shift_indices, 2);
  array all_valid = zeros({1, 1, 1, C}, float32);
  array ref = single_token_attention_ref(query, shifted_k, shifted_v, new_key,
                                          new_value, last_index, scale,
                                          all_valid);
  eval(ref);

  mlx_array out = mlx_array_new_(), nk = mlx_array_new_(), nv = mlx_array_new_();
  mlx_array q = wrap(query), kc = wrap(key_cache), vc = wrap(value_cache),
            k = wrap(new_key), v = wrap(new_value);
  array scale_arr = array(scale);
  eval(scale_arr);
  mlx_array sc = wrap(scale_arr), si = wrap(shift_indices), li = wrap(last_index);
  mlx_stream s = mlx_default_gpu_stream_new();
  int rc = go_mlx_compiled_fixed_sliding_single_token_attention(
      &out, &nk, &nv, q, kc, vc, k, v, sc, si, li, s);
  REQUIRE(rc == 0);
  std::vector<float> got = to_floats(out);
  REQUIRE(got.size() == ref.size());
  const float* rp = ref.data<float>();
  for (size_t i = 0; i < got.size(); ++i) {
    CHECK(got[i] == doctest::Approx(rp[i]).epsilon(3e-3));
  }

  mlx_array_free(out);
  mlx_array_free(nk);
  mlx_array_free(nv);
  mlx_array_free(q);
  mlx_array_free(kc);
  mlx_array_free(vc);
  mlx_array_free(k);
  mlx_array_free(v);
  mlx_array_free(sc);
  mlx_array_free(si);
  mlx_array_free(li);
  mlx_stream_free(s);
}

// Paged attention reference: concat K/V across pages on the token axis (axis 2),
// repeat KV heads to query heads, full softmax. Matches all three kernel paths
// (single page == sdpa; uniform pages == compiled online-softmax; non-uniform
// pages == impl online-softmax) because online softmax is exact.
namespace {
std::vector<float> paged_attention_ref(const array& query,
                                       const std::vector<array>& keys,
                                       const std::vector<array>& values,
                                       float scale) {
  array k_all = keys[0];
  array v_all = values[0];
  for (size_t i = 1; i < keys.size(); ++i) {
    k_all = concatenate({k_all, keys[i]}, 2);
    v_all = concatenate({v_all, values[i]}, 2);
  }
  const int hq = query.shape(1), hkv = k_all.shape(1), factor = hq / hkv;
  if (factor > 1) {
    const int b = k_all.shape(0), c = k_all.shape(2), d = k_all.shape(3);
    k_all = reshape(broadcast_to(expand_dims(k_all, 2),
                                 Shape{b, hkv, factor, c, d}),
                    Shape{b, hkv * factor, c, d});
    v_all = reshape(broadcast_to(expand_dims(v_all, 2),
                                 Shape{b, hkv, factor, c, d}),
                    Shape{b, hkv * factor, c, d});
  }
  array scores = matmul(query, transpose(k_all, {0, 1, 3, 2}));
  scores = multiply(scores, array(scale, scores.dtype()));
  array weights = softmax(scores, std::vector<int>{-1}, true);
  array out = matmul(weights, v_all);
  array of = astype(out, float32);
  eval(of);
  return std::vector<float>(of.data<float>(), of.data<float>() + of.size());
}

std::vector<float> run_paged(const array& query,
                             const std::vector<array>& keys,
                             const std::vector<array>& values, float scale,
                             int* rc_out) {
  std::vector<mlx_array> kp, vp;
  for (const array& k : keys) kp.push_back(wrap(k));
  for (const array& v : values) vp.push_back(wrap(v));
  mlx_array out = mlx_array_new_();
  mlx_array q = wrap(query);
  mlx_stream s = mlx_default_gpu_stream_new();
  int rc = go_mlx_native_paged_single_token_attention(
      &out, q, kp.data(), vp.data(), static_cast<int>(keys.size()), scale, s);
  *rc_out = rc;
  std::vector<float> got;
  if (rc == 0) got = to_floats(out);
  mlx_array_free(out);
  mlx_array_free(q);
  for (mlx_array a : kp) mlx_array_free(a);
  for (mlx_array a : vp) mlx_array_free(a);
  mlx_stream_free(s);
  return got;
}
}  // namespace

TEST_CASE("decode: native paged single-token attention — all three page paths") {
  if (!is_available(Device::gpu)) {
    WARN_MESSAGE(false, "Metal GPU unavailable — skipping attention test");
    return;
  }

  const int B = 1, D = 16;
  const float scale = 0.25f;

  SUBCASE("single page (sdpa path)") {
    const int Hq = 2, Hkv = 2, C = 5;
    random::seed(401);
    array query = random::normal({B, Hq, 1, D}) * array(0.4f);
    array key = random::normal({B, Hkv, C, D}) * array(0.3f);
    array value = random::normal({B, Hkv, C, D}) * array(0.3f);
    eval(query);
    eval(key);
    eval(value);
    std::vector<float> ref = paged_attention_ref(query, {key}, {value}, scale);
    int rc = -1;
    std::vector<float> got = run_paged(query, {key}, {value}, scale, &rc);
    REQUIRE(rc == 0);
    REQUIRE(got.size() == ref.size());
    for (size_t i = 0; i < got.size(); ++i)
      CHECK(got[i] == doctest::Approx(ref[i]).epsilon(3e-3));
  }

  SUBCASE("two uniform pages with GQA (compiled online-softmax path)") {
    const int Hq = 4, Hkv = 2, C = 4;  // GQA factor 2 -> repeat_kv in the graph
    random::seed(403);
    array query = random::normal({B, Hq, 1, D}) * array(0.4f);
    array k0 = random::normal({B, Hkv, C, D}) * array(0.3f);
    array v0 = random::normal({B, Hkv, C, D}) * array(0.3f);
    array k1 = random::normal({B, Hkv, C, D}) * array(0.3f);
    array v1 = random::normal({B, Hkv, C, D}) * array(0.3f);
    eval(query); eval(k0); eval(v0); eval(k1); eval(v1);
    std::vector<float> ref =
        paged_attention_ref(query, {k0, k1}, {v0, v1}, scale);
    int rc = -1;
    std::vector<float> got = run_paged(query, {k0, k1}, {v0, v1}, scale, &rc);
    REQUIRE(rc == 0);
    REQUIRE(got.size() == ref.size());
    for (size_t i = 0; i < got.size(); ++i)
      CHECK(got[i] == doctest::Approx(ref[i]).epsilon(3e-3));
  }

  SUBCASE("two non-uniform pages, MQA single KV head (impl online-softmax path)") {
    // Different per-page token counts -> the uniform-shape guard fails and the
    // bridge takes paged_single_token_attention_impl. key_heads == 1 (MQA)
    // additionally exercises the `key_heads != 1` skip in repeat_kv.
    const int Hq = 4, Hkv = 1, C0 = 3, C1 = 5;
    random::seed(405);
    array query = random::normal({B, Hq, 1, D}) * array(0.4f);
    array k0 = random::normal({B, Hkv, C0, D}) * array(0.3f);
    array v0 = random::normal({B, Hkv, C0, D}) * array(0.3f);
    array k1 = random::normal({B, Hkv, C1, D}) * array(0.3f);
    array v1 = random::normal({B, Hkv, C1, D}) * array(0.3f);
    eval(query); eval(k0); eval(v0); eval(k1); eval(v1);
    std::vector<float> ref =
        paged_attention_ref(query, {k0, k1}, {v0, v1}, scale);
    int rc = -1;
    std::vector<float> got = run_paged(query, {k0, k1}, {v0, v1}, scale, &rc);
    REQUIRE(rc == 0);
    REQUIRE(got.size() == ref.size());
    for (size_t i = 0; i < got.size(); ++i)
      CHECK(got[i] == doctest::Approx(ref[i]).epsilon(3e-3));
  }
}

// =============================================================================
// Cheap fill-ins — the remaining lm-head-tail variants, same unfused oracle as
// the dense cases above (#90 fused decode tail).
// =============================================================================

TEST_CASE("decode: greedy decode token — rank 1/2/3 logits all argmax last row") {
  if (!is_available(Device::gpu)) {
    WARN_MESSAGE(false, "Metal GPU unavailable — skipping decode test");
    return;
  }
  // last_token_logits has three rank branches; feed each. For rank>=2 the last
  // sequence position is the one argmaxed.
  const int V = 64;
  random::seed(31);

  SUBCASE("rank 1 logits") {
    array logits = random::normal({V});
    eval(logits);
    array ref = argmax(reshape(logits, {1, V}), -1, false);
    eval(ref);
    mlx_array res = mlx_array_new_();
    mlx_array lg = wrap(logits);
    mlx_stream s = mlx_default_gpu_stream_new();
    int rc = go_mlx_compiled_greedy_decode_token(&res, lg, s);
    REQUIRE(rc == 0);
    array& got = mlx_array_get_(res);
    eval(got);
    CHECK(got.item<uint32_t>() == ref.item<uint32_t>());
    mlx_array_free(res); mlx_array_free(lg); mlx_stream_free(s);
  }

  SUBCASE("rank 2 logits [S,V] — last row") {
    const int S = 3;
    array logits = random::normal({S, V});
    eval(logits);
    array last = slice(logits, Shape{S - 1, 0}, Shape{S, V});
    array ref = argmax(reshape(last, {1, V}), -1, false);
    eval(ref);
    mlx_array res = mlx_array_new_();
    mlx_array lg = wrap(logits);
    mlx_stream s = mlx_default_gpu_stream_new();
    int rc = go_mlx_compiled_greedy_decode_token(&res, lg, s);
    REQUIRE(rc == 0);
    array& got = mlx_array_get_(res);
    eval(got);
    CHECK(got.item<uint32_t>() == ref.item<uint32_t>());
    mlx_array_free(res); mlx_array_free(lg); mlx_stream_free(s);
  }

  SUBCASE("rank 3 logits [B,S,V] — last seq position") {
    const int Bb = 1, S = 3;
    array logits = random::normal({Bb, S, V});
    eval(logits);
    array last = slice(logits, Shape{0, S - 1, 0}, Shape{Bb, S, V});
    array ref = argmax(reshape(last, {1, V}), -1, false);
    eval(ref);
    mlx_array res = mlx_array_new_();
    mlx_array lg = wrap(logits);
    mlx_stream s = mlx_default_gpu_stream_new();
    int rc = go_mlx_compiled_greedy_decode_token(&res, lg, s);
    REQUIRE(rc == 0);
    array& got = mlx_array_get_(res);
    eval(got);
    CHECK(got.item<uint32_t>() == ref.item<uint32_t>());
    mlx_array_free(res); mlx_array_free(lg); mlx_stream_free(s);
  }
}

TEST_CASE("decode: q4 g64 last-logits softcap30 matches dequantised reference") {
  if (!is_available(Device::gpu)) {
    WARN_MESSAGE(false, "Metal GPU unavailable — skipping decode test");
    return;
  }
  const int H = 256, V = 512;
  random::seed(37);
  array hidden = random::normal({1, H}) * array(0.5f);
  array norm_w = random::normal({H}) * array(0.1f) + array(1.0f);
  array out_w_full = random::normal({V, H}) * array(0.05f);
  eval(hidden); eval(norm_w); eval(out_w_full);
  std::vector<array> q = quantize(out_w_full, 64, 4);
  array w = q[0], scales = q[1], biases = q[2];
  eval(w); eval(scales); eval(biases);

  array normed = rms_norm_ref(hidden, norm_w, 1e-6f);
  array logits = quantized_matmul(normed, w, scales, biases, true, 64, 4, "affine");
  array thirty = array(30.0f);
  array ref = multiply(tanh(divide(logits, thirty)), thirty);
  eval(ref);

  mlx_array res = mlx_array_new_();
  mlx_array h = wrap(hidden), nw = wrap(norm_w), ow = wrap(w),
            sc = wrap(scales), bi = wrap(biases);
  mlx_stream s = mlx_default_gpu_stream_new();
  int rc = go_mlx_compiled_q4_g64_last_logits_softcap30(&res, h, nw, ow, sc, bi, s);
  REQUIRE(rc == 0);
  array& got = mlx_array_get_(res);
  eval(got);
  REQUIRE(got.size() == static_cast<size_t>(V));
  const float* gp = got.data<float>();
  const float* rp = ref.data<float>();
  for (int i = 0; i < V; ++i) {
    CHECK(gp[i] == doctest::Approx(rp[i]).epsilon(2e-3));
    CHECK(std::abs(gp[i]) < 30.0f + 1e-3f);
  }
  mlx_array_free(res); mlx_array_free(h); mlx_array_free(nw);
  mlx_array_free(ow); mlx_array_free(sc); mlx_array_free(bi); mlx_stream_free(s);
}

TEST_CASE("decode: quantised suppressed last-token (q4/q8/q6) shifts the winner") {
  if (!is_available(Device::gpu)) {
    WARN_MESSAGE(false, "Metal GPU unavailable — skipping decode test");
    return;
  }
  using Entry = int (*)(mlx_array*, mlx_array, mlx_array, mlx_array, mlx_array,
                        mlx_array, mlx_array, mlx_stream);
  struct Variant { const char* name; int bits; Entry fn; };
  const Variant variants[] = {
      {"q4", 4, &go_mlx_compiled_q4_g64_last_token_suppressed},
      {"q8", 8, &go_mlx_compiled_q8_g64_last_token_suppressed},
      {"q6", 6, &go_mlx_compiled_q6_g64_last_token_suppressed},
  };
  const int H = 256, V = 512;
  for (const Variant& var : variants) {
    CAPTURE(var.name);
    random::seed(41 + var.bits);
    array hidden = random::normal({1, H}) * array(0.5f);
    array norm_w = random::normal({H}) * array(0.1f) + array(1.0f);
    array out_w_full = random::normal({V, H}) * array(0.05f);
    eval(hidden); eval(norm_w); eval(out_w_full);
    std::vector<array> q = quantize(out_w_full, 64, var.bits);
    array w = q[0], scales = q[1], biases = q[2];
    eval(w); eval(scales); eval(biases);

    array normed = rms_norm_ref(hidden, norm_w, 1e-6f);
    array logits = quantized_matmul(normed, w, scales, biases, true, 64,
                                    var.bits, "affine");
    array top1 = argmax(logits, -1, false);
    eval(top1);
    uint32_t top1_id = top1.item<uint32_t>();
    array suppress = reshape(astype(top1, int32), {1});
    array idx = reshape(suppress, {1, 1});
    array updates = full({1, 1}, -std::numeric_limits<float>::infinity(), float32);
    array masked = put_along_axis(logits, idx, updates, -1);
    array ref_tok = argmax(masked, -1, false);
    eval(ref_tok);
    uint32_t ref_id = ref_tok.item<uint32_t>();
    REQUIRE(ref_id != top1_id);

    mlx_array res = mlx_array_new_();
    mlx_array h = wrap(hidden), nw = wrap(norm_w), ow = wrap(w),
              sc = wrap(scales), bi = wrap(biases), sp = wrap(suppress);
    mlx_stream s = mlx_default_gpu_stream_new();
    int rc = var.fn(&res, h, nw, ow, sc, bi, sp, s);
    REQUIRE(rc == 0);
    array& got = mlx_array_get_(res);
    eval(got);
    CHECK(got.item<uint32_t>() == ref_id);
    CHECK(got.item<uint32_t>() != top1_id);
    mlx_array_free(res); mlx_array_free(h); mlx_array_free(nw);
    mlx_array_free(ow); mlx_array_free(sc); mlx_array_free(bi);
    mlx_array_free(sp); mlx_stream_free(s);
  }
}

TEST_CASE("decode: q4 g64 MLP gelu (SwiGLU) matches dequantised reference") {
  if (!is_available(Device::gpu)) {
    WARN_MESSAGE(false, "Metal GPU unavailable — skipping decode test");
    return;
  }
  const int H = 128, FF = 256;  // both multiples of group_size 64
  random::seed(43);
  array x = random::normal({1, H}) * array(0.5f);
  array gate_w = random::normal({FF, H}) * array(0.05f);
  array up_w = random::normal({FF, H}) * array(0.05f);
  array down_w = random::normal({H, FF}) * array(0.05f);
  eval(x); eval(gate_w); eval(up_w); eval(down_w);

  std::vector<array> qg = quantize(gate_w, 64, 4);
  std::vector<array> qu = quantize(up_w, 64, 4);
  std::vector<array> qd = quantize(down_w, 64, 4);
  for (array* a : {&qg[0], &qg[1], &qg[2], &qu[0], &qu[1], &qu[2],
                   &qd[0], &qd[1], &qd[2]})
    eval(*a);

  // Reference uses the SAME quantised weights via quantized_matmul, so the only
  // thing under test is the fused gate/up/gelu/down chain, not the quantiser.
  auto qlin = [](const array& in, const array& w, const array& s,
                 const array& b) {
    return quantized_matmul(in, w, s, b, true, 64, 4, "affine");
  };
  array gate = qlin(x, qg[0], qg[1], qg[2]);
  array up = qlin(x, qu[0], qu[1], qu[2]);
  array activated = multiply(gelu_approx_ref(gate), up);
  array ref = qlin(activated, qd[0], qd[1], qd[2]);
  eval(ref);

  mlx_array res = mlx_array_new_();
  mlx_array xa = wrap(x);
  mlx_array gw = wrap(qg[0]), gs = wrap(qg[1]), gb = wrap(qg[2]);
  mlx_array uw = wrap(qu[0]), us = wrap(qu[1]), ub = wrap(qu[2]);
  mlx_array dw = wrap(qd[0]), ds = wrap(qd[1]), db = wrap(qd[2]);
  mlx_stream s = mlx_default_gpu_stream_new();
  int rc = go_mlx_compiled_q4_g64_mlp_gelu(&res, xa, gw, gs, gb, uw, us, ub,
                                           dw, ds, db, s);
  REQUIRE(rc == 0);
  array& got = mlx_array_get_(res);
  eval(got);
  REQUIRE(got.size() == static_cast<size_t>(H));
  const float* gp = got.data<float>();
  const float* rp = ref.data<float>();
  for (int i = 0; i < H; ++i)
    CHECK(gp[i] == doctest::Approx(rp[i]).epsilon(3e-3));

  mlx_array_free(res); mlx_array_free(xa);
  mlx_array_free(gw); mlx_array_free(gs); mlx_array_free(gb);
  mlx_array_free(uw); mlx_array_free(us); mlx_array_free(ub);
  mlx_array_free(dw); mlx_array_free(ds); mlx_array_free(db);
  mlx_stream_free(s);
}

TEST_CASE("decode: fixed attention rejects a malformed input (contract error path)") {
  if (!is_available(Device::gpu)) {
    WARN_MESSAGE(false, "Metal GPU unavailable — skipping decode test");
    return;
  }
  // A rank-3 key cache (should be rank-4) trips the row-update contract guard,
  // so the bridge must mlx_error + return non-zero rather than crash. This
  // covers the catch/return-1 path the happy cases never reach. Also flips on
  // GO_MLX_BRIDGE_SHAPE_TRACE for this one call to cover the trace fprintf.
  go_mlx_set_fixed_attention_diagnostics(0, /*row_update=*/1);
  setenv("GO_MLX_BRIDGE_SHAPE_TRACE", "1", 1);

  const int B = 1, H = 2, D = 8;
  array query = random::normal({B, H, 1, D});
  array bad_cache = random::normal({B, H, D});  // rank 3 — invalid
  array new_kv = random::normal({B, H, 1, D});
  array offset_a = array(static_cast<int32_t>(0));
  array mask = zeros({1, 1, 1, 1}, float32);
  eval(query); eval(bad_cache); eval(new_kv); eval(offset_a); eval(mask);

  int rc = 0;
  std::vector<float> got = run_fixed_attention(
      query, bad_cache, bad_cache, new_kv, new_kv, offset_a, 1.0f, mask, false,
      &rc);
  CHECK(rc != 0);  // contract violation reported, process not aborted
  CHECK(got.empty());

  unsetenv("GO_MLX_BRIDGE_SHAPE_TRACE");
  go_mlx_set_fixed_attention_diagnostics(0, 0);
}
