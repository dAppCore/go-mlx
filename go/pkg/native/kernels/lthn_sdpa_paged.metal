// SPDX-Licence-Identifier: EUPL-1.2

#include <metal_stdlib>
using namespace metal;

typedef bfloat bf16;

struct PagedSDPADims {
  uint nHeads;
  uint nKVHeads;
  uint headDim;
  uint pageLen;
  uint kHeadStride;
  uint kSeqStride;
  uint vHeadStride;
  uint vSeqStride;
  float scale;
};

[[kernel]] void lthn_sdpa_paged_update_bf16(
    const device bf16* q      [[buffer(0)]],
    const device bf16* kPage  [[buffer(1)]],
    const device bf16* vPage  [[buffer(2)]],
    device float*      maxs   [[buffer(3)]],
    device float*      denoms [[buffer(4)]],
    device float*      acc    [[buffer(5)]],
    const constant PagedSDPADims& D [[buffer(6)]],
    uint h [[thread_position_in_grid]]) {
  if (h >= D.nHeads) return;

  const uint gqa = D.nHeads / D.nKVHeads;
  const uint kvh = h / gqa;
  const device bf16* qh = q + h * D.headDim;
  const device bf16* kh = kPage + kvh * D.kHeadStride;
  const device bf16* vh = vPage + kvh * D.vHeadStride;

  float pageMax = -3.0e38f;
  for (uint t = 0; t < D.pageLen; t++) {
    float dot = 0.0f;
    const device bf16* kt = kh + t * D.kSeqStride;
    for (uint d = 0; d < D.headDim; d++) {
      dot += float(qh[d]) * float(kt[d]);
    }
    pageMax = max(pageMax, dot * D.scale);
  }

  const uint accOff = h * D.headDim;
  const float oldMax = maxs[h];
  const float oldDenom = denoms[h];
  const float newMax = max(oldMax, pageMax);
  const float oldScale = oldDenom > 0.0f ? exp(oldMax - newMax) : 0.0f;
  float denom = oldDenom * oldScale;
  for (uint d = 0; d < D.headDim; d++) {
    acc[accOff + d] *= oldScale;
  }

  for (uint t = 0; t < D.pageLen; t++) {
    float dot = 0.0f;
    const device bf16* kt = kh + t * D.kSeqStride;
    for (uint d = 0; d < D.headDim; d++) {
      dot += float(qh[d]) * float(kt[d]);
    }
    const float p = exp(dot * D.scale - newMax);
    denom += p;
    const device bf16* vt = vh + t * D.vSeqStride;
    for (uint d = 0; d < D.headDim; d++) {
      acc[accOff + d] += p * float(vt[d]);
    }
  }

  maxs[h] = newMax;
  denoms[h] = denom;
}

[[kernel]] void lthn_sdpa_paged_finalise_bf16(
    const device float* denoms  [[buffer(0)]],
    const device float* acc     [[buffer(1)]],
    device bf16*        out     [[buffer(2)]],
    const constant uint& headDim [[buffer(3)]],
    const constant uint& total   [[buffer(4)]],
    uint i [[thread_position_in_grid]]) {
  if (i >= total) return;
  const uint h = i / headDim;
  const float denom = denoms[h];
  out[i] = denom > 0.0f ? bf16(acc[i] / denom) : bf16(0.0f);
}
