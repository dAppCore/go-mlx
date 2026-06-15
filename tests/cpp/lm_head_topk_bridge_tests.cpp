// SPDX-Licence-Identifier: EUPL-1.2
//
// Correctness tests for the go-mlx fused quantised lm-head + top-k kernel
// (#95): go/pkg/metal/lm_head_topk_bridge.cpp, entry point
// go_mlx_q4_lm_head_topk.
//
// The kernel fuses a q4-affine lm-head matrix-vector product with a global
// top-k in a single Metal pass plus an in-graph tile merge — the full vocab
// logits row is never materialised. These tests assert the fused output equals
// the unfused reference path (dequantise → full mat-vec → argsort top-k),
// which is the only meaningful definition of "correct" for a fused kernel.
//
// We build the inputs in the C++ mlx::core domain (so the quantised layout is
// produced by mlx::core::quantize, matching exactly what the kernel expects —
// hand-packing nibbles would desync from the affine layout), cross the mlx_array
// C ABI to call the actual shipped extern "C" entry point, then read the result
// back as an mlx::core::array for comparison. The kernel is NOT reshaped for
// the test; the test adapts to the kernel.

#include "doctest/doctest.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "mlx/c/array.h"
#include "mlx/c/private/array.h"
#include "mlx/c/stream.h"
#include "mlx/mlx.h"

// The bridge under test (go-mlx authored).
#include "lm_head_topk_bridge.h"

using namespace mlx::core;

namespace {

// Wrap a C++ array as an mlx_array (borrowing into a fresh ctx). Caller frees.
mlx_array wrap(const array& a) {
  return mlx_array_new_(a);
}

// Reference (unfused) top-k of the q4-affine lm-head: dequantise the packed
// weights, do the full [N,K]·[K] mat-vec, then argsort for the descending
// top-k values + indices. This is the oracle the fused kernel must match.
struct Reference {
  std::vector<float> values;  // descending, length top_k
  std::vector<int> indices;   // matching argmax order
};

Reference reference_topk(
    const array& x_full,        // [K] float32
    const array& w_packed,      // [N, K/8] uint32
    const array& scales,        // [N, K/group_size]
    const array& biases,        // [N, K/group_size]
    int group_size,
    int top_k) {
  // Dequantise to a dense [N, K] float matrix and run the full mat-vec.
  array deq = dequantize(w_packed, scales, biases, group_size, /*bits=*/4);
  array x_col = reshape(astype(x_full, float32), {-1, 1});  // [K,1]
  array logits = reshape(matmul(deq, x_col), {-1});         // [N]

  // Descending top-k via argsort on the negated logits.
  array order = argsort(negative(logits), 0);
  array idx = slice(order, Shape{0}, Shape{top_k});
  array vals = take(logits, idx, 0);
  eval(vals);
  eval(idx);

  Reference ref;
  ref.values.assign(vals.data<float>(), vals.data<float>() + top_k);
  ref.indices.assign(idx.data<int>(), idx.data<int>() + top_k);
  return ref;
}

// Run the fused kernel via the real extern "C" entry point and read results.
struct Fused {
  std::vector<float> values;
  std::vector<int> indices;
  int rc;
};

Fused fused_topk(
    const array& x_full,
    const array& w_packed,
    const array& scales,
    const array& biases,
    int group_size,
    int top_k,
    int num_simdgroups,
    int subtiles) {
  mlx_array x = wrap(x_full);
  mlx_array w = wrap(w_packed);
  mlx_array s = wrap(scales);
  mlx_array b = wrap(biases);
  mlx_array out_values = mlx_array_new_();
  mlx_array out_indices = mlx_array_new_();
  mlx_stream stream = mlx_default_gpu_stream_new();

  int rc = go_mlx_q4_lm_head_topk(
      &out_values, &out_indices, x, w, s, b, group_size, top_k,
      num_simdgroups, subtiles, stream);

  Fused f;
  f.rc = rc;
  if (rc == 0) {
    array& v = mlx_array_get_(out_values);
    array& i = mlx_array_get_(out_indices);
    eval(v);
    eval(i);
    f.values.assign(v.data<float>(), v.data<float>() + top_k);
    f.indices.assign(i.data<int>(), i.data<int>() + top_k);
  }

  mlx_array_free(x);
  mlx_array_free(w);
  mlx_array_free(s);
  mlx_array_free(b);
  mlx_array_free(out_values);
  mlx_array_free(out_indices);
  mlx_stream_free(stream);
  return f;
}

// Quantise a dense float weight to the q4-affine layout the kernel consumes.
struct Quantised {
  array w_packed = array(0.0f);
  array scales = array(0.0f);
  array biases = array(0.0f);
};

Quantised quantise_q4(const array& w_full, int group_size) {
  std::vector<array> q = quantize(w_full, group_size, /*bits=*/4);
  Quantised out{q[0], q[1], q[2]};
  eval(out.w_packed);
  eval(out.scales);
  eval(out.biases);
  return out;
}

} // namespace

TEST_CASE("q4 lm-head top-k: fused matches unfused reference") {
  if (!is_available(Device::gpu)) {
    WARN_MESSAGE(false, "Metal GPU unavailable — skipping fused lm-head test");
    return;
  }

  // K must be a positive multiple of 512 (no K-tail guard in the tile loop);
  // N >= top_k; group_size in {32,64,128}; num_simdgroups in {2,4,8}.
  const int K = 512;
  const int N = 128;
  const int group_size = 64;
  const int top_k = 8;
  const int num_simdgroups = 4;
  const int subtiles = 1;

  random::seed(42);
  // Small magnitudes so dequant error stays well below the logit gaps; keeps
  // the top-k ordering unambiguous (no tie-driven index churn).
  array w_full = random::normal({N, K}) * array(0.1f);
  array x_full = random::normal({K});
  eval(w_full);
  eval(x_full);

  Quantised q = quantise_q4(w_full, group_size);
  // Oracle reads the SAME quantised weights the kernel sees, so dequant error
  // is shared — the comparison isolates the kernel, not the quantiser.
  Reference ref = reference_topk(x_full, q.w_packed, q.scales, q.biases,
                                 group_size, top_k);
  Fused got = fused_topk(x_full, q.w_packed, q.scales, q.biases, group_size,
                         top_k, num_simdgroups, subtiles);

  REQUIRE(got.rc == 0);
  REQUIRE(got.values.size() == static_cast<size_t>(top_k));
  REQUIRE(got.indices.size() == static_cast<size_t>(top_k));

  // Values: descending, and equal to the reference within fp tolerance.
  for (int i = 0; i < top_k; ++i) {
    CHECK(got.values[i] == doctest::Approx(ref.values[i]).epsilon(1e-3));
    if (i > 0) {
      CHECK(got.values[i] <= got.values[i - 1] + 1e-4f);
    }
  }

  // Indices: same SET as the reference (membership, robust to any tie ordering).
  std::vector<int> got_sorted = got.indices;
  std::vector<int> ref_sorted = ref.indices;
  std::sort(got_sorted.begin(), got_sorted.end());
  std::sort(ref_sorted.begin(), ref_sorted.end());
  CHECK(got_sorted == ref_sorted);
}

TEST_CASE("q4 lm-head top-k: matches reference across group sizes and top_k") {
  if (!is_available(Device::gpu)) {
    WARN_MESSAGE(false, "Metal GPU unavailable — skipping fused lm-head test");
    return;
  }

  struct Case {
    int K, N, group_size, top_k, num_simdgroups, subtiles;
  };
  // K a multiple of 512; sweep the three legal group sizes, a couple of top_k,
  // a larger vocab that forces multiple tiles (N=1024 > one tile's outputs),
  // and num_simdgroups 2/8 to exercise the per-tile candidate merge.
  const Case cases[] = {
      {512, 128, 32, 4, 2, 1},
      {512, 256, 128, 16, 8, 1},
      {1024, 1024, 64, 8, 4, 2},
      {512, 130, 64, 1, 4, 1},
  };

  int c = 0;
  for (const Case& tc : cases) {
    CAPTURE(c);
    CAPTURE(tc.K);
    CAPTURE(tc.N);
    CAPTURE(tc.group_size);
    CAPTURE(tc.top_k);
    random::seed(100 + c);
    array w_full = random::normal({tc.N, tc.K}) * array(0.1f);
    array x_full = random::normal({tc.K});
    eval(w_full);
    eval(x_full);

    Quantised q = quantise_q4(w_full, tc.group_size);
    Reference ref = reference_topk(x_full, q.w_packed, q.scales, q.biases,
                                   tc.group_size, tc.top_k);
    Fused got = fused_topk(x_full, q.w_packed, q.scales, q.biases,
                           tc.group_size, tc.top_k, tc.num_simdgroups,
                           tc.subtiles);

    REQUIRE(got.rc == 0);
    REQUIRE(got.values.size() == static_cast<size_t>(tc.top_k));
    for (int i = 0; i < tc.top_k; ++i) {
      CHECK(got.values[i] == doctest::Approx(ref.values[i]).epsilon(1e-3));
    }
    std::vector<int> gs = got.indices, rs = ref.indices;
    std::sort(gs.begin(), gs.end());
    std::sort(rs.begin(), rs.end());
    CHECK(gs == rs);
    ++c;
  }
}

TEST_CASE("q4 lm-head top-k: rejects out-of-contract inputs") {
  if (!is_available(Device::gpu)) {
    WARN_MESSAGE(false, "Metal GPU unavailable — skipping fused lm-head test");
    return;
  }

  // A valid baseline set we then perturb one constraint at a time. The kernel
  // must return non-zero (rc==1) for each — these are the guard branches.
  const int K = 512, N = 128, group_size = 64;
  random::seed(7);
  array w_full = random::normal({N, K}) * array(0.1f);
  array x_full = random::normal({K});
  eval(w_full);
  eval(x_full);
  Quantised q = quantise_q4(w_full, group_size);

  auto call = [&](int gsz, int top_k, int simd, int subt, const array& x) {
    return fused_topk(x, q.w_packed, q.scales, q.biases, gsz, top_k, simd,
                      subt).rc;
  };

  SUBCASE("top_k below range") { CHECK(call(group_size, 0, 4, 1, x_full) == 1); }
  SUBCASE("top_k above 64") { CHECK(call(group_size, 65, 4, 1, x_full) == 1); }
  SUBCASE("top_k exceeds rows") {
    // n_size < top_k: a 4-row weight with top_k=8 (top_k still within [1,64]).
    array small_w = random::normal({4, K}) * array(0.1f);
    eval(small_w);
    Quantised sq = quantise_q4(small_w, group_size);
    CHECK(fused_topk(x_full, sq.w_packed, sq.scales, sq.biases, group_size, 8,
                     4, 1).rc == 1);
  }
  SUBCASE("bad group_size") { CHECK(call(48, 8, 4, 1, x_full) == 1); }
  SUBCASE("bad num_simdgroups") { CHECK(call(group_size, 8, 3, 1, x_full) == 1); }
  SUBCASE("subtiles < 1") { CHECK(call(group_size, 8, 4, 0, x_full) == 1); }
  SUBCASE("K not a multiple of 512") {
    array bad_x = random::normal({256});  // K=256, not a multiple of 512
    eval(bad_x);
    CHECK(call(group_size, 8, 4, 1, bad_x) == 1);
  }
}
