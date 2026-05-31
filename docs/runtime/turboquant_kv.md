<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# TurboQuant KV Implementation Note

Status: research implementation for the explicit `turboquant` cache mode. This
is not a default path. The current code has a versioned page payload, a
physical 3.5-bit/channel reference layout using a 3-bit regular / 4-bit outlier
split, and a reference restore bridge that dequantizes compressed pages back
into MLX arrays before attention. Pinned restore and compressed-attention
kernels are still open work.

Source basis: `/Users/snider/Downloads/2504.19874v1.pdf`, especially Algorithm
1 `TurboQuantmse`, Algorithm 2 `TurboQuantprod`, and the KV-cache compression
experiments. The current planner estimate uses `3.5` bits per KV element as the
paper-backed hypothesis to validate, not as a production guarantee.

## Current go-mlx Cache Shape

Native K/V tensors are rank-4 MLX arrays:

```text
[batch, kv_heads, seq_len, head_dim]
```

The active cache families expose that shape differently:

- `KVCache`, `RotatingKVCache`, and `FixedKVCache` store one K array and one V
  array per cache.
- `PagedKVCache` stores `kPages` and `vPages`, each page still shaped as
  `[batch, kv_heads, page_len, head_dim]`. The default page size is `2048`;
  Gemma 4 local sliding caches cap at the native local window, normally `512`,
  while global owner layers carry the long retained context.
- `KVSnapshot` version `4` stores native byte slabs per logical layer via
  `KeyBytes`/`KeyShape` and `ValueBytes`/`ValueShape`. Version `5` adds
  explicit `CacheMode` plus opaque TurboQuant page payloads so compressed KV
  state can survive the public `kv.Snapshot` binary format and root/Metal
  conversion without being mistaken for fp16, q8, or paged K/V slabs.
- Native slab restore already has a zero-copy pinned raw-byte path through
  `fromPinnedRawBytes`.
- `fromPinnedRawBytesStrided` and the external `go-cgo` C++23 `mdspan` helper
  are the right substrate for future State-file pages that should be viewed
  without reshuffling.

TurboQuant must preserve this logical shape. Compression changes only the
physical page payload and the attention/dequant path.

## Algorithm Mapping

TurboQuant works on vectors in `R^d`; for go-mlx, one vector is one token row:

```text
cache page vector = cache[layer or cache_index][kind K/V][batch][head][token][:]
d = head_dim
```

The paper assumes unit vectors. K/V rows are not guaranteed to be unit length,
so each encoded vector stores a norm. Zero vectors use a zero-norm sentinel and
skip rotation/quantisation.

### K path: `TurboQuantprod`

Keys participate directly in attention score inner products, so they should use
the paper's inner-product path:

1. Normalize key vector `k` into `k_hat` and store `||k||`.
2. Apply `TurboQuantmse` with `b - 1` bits per coordinate:
   - deterministic rotation seed produces `Pi`;
   - `y = Pi * k_hat`;
   - each coordinate stores the nearest centroid index.
3. Reconstruct the MSE approximation and compute residual
   `r = k_hat - DeQuantmse(idx)`.
4. Store `qjl = sign(S * r)` plus `||r||`.
5. During attention, keep the query vector high precision and estimate
   `q dot k` from the MSE reconstruction plus the QJL residual correction,
   scaled by the stored key norm.

The first correctness implementation may dequantize K pages back to fp16/bf16
before calling existing attention. The production implementation should consume
compressed K pages in native attention so retained global pages are not
expanded for every decode step.

### V path: `TurboQuantmse`

Values are multiplied by attention weights rather than used as lookup keys for
an inner-product search. They should start with the MSE path:

1. Normalize value vector `v` and store `||v||`.
2. Rotate with the same deterministic rotation family, scoped separately for V.
3. Store nearest-centroid indices for each coordinate.
4. Dequantize by centroid lookup, inverse rotation, and norm rescale.

If long-output quality shows value reconstruction error dominates, add a
`TurboQuantprod` V experiment behind a separate gate instead of changing the
default TurboQuant design.

## Outlier Split

The paper's `2.5` and `3.5` bit KV results come from splitting channels into
outlier and non-outlier sets and applying independent TurboQuant instances at
different bit widths. go-mlx should make that explicit metadata:

```text
outlier_policy:
  kind: channel_mask
  dimension: head_dim
  mask_bits: packed bitset
  normal_bits: N
  outlier_bits: M
  effective_bits: weighted_average(normal_bits, outlier_bits)
```

Do not hard-code a channel count from another model family. Gemma 4 E2B/E4B
needs its own calibration sweep over K and V rows, reported separately for
local and global caches.

## Physical Layout

Use a versioned TurboQuant physical layout instead of overloading q8 or paged
snapshots. Older or malformed payloads still fail closed through the exact
layout/codec/version checks.

Each compressed page should carry:

- schema version and codec name, for example `turboquant-kv-v1`;
- model identity, architecture, cache layout hash, and tokenizer/config hashes;
- `cache_index`, logical layer index, layer type, and shared-KV owner identity;
- logical shape `[batch, kv_heads, seq_len, head_dim]`;
- logical token offset, page token count, page size, and local-window cap;
- K codec metadata: algorithm `turboquantprod`, effective bits, rotation seed,
  QJL seed, codebook id, norm policy, residual-norm policy, outlier policy,
  packed centroid indices, packed QJL signs, vector norms, residual norms;
- V codec metadata: algorithm `turboquantmse`, effective bits, rotation seed,
  codebook id, norm policy, outlier policy, packed centroid indices, vector
  norms;
- byte alignment and endian marker.

Payloads should be page-local and appendable. A State file can then index pages
by token range without materializing a full context. Public State blocks treat
opaque compressed payload snapshots as whole blocks unless a native Metal block
source has already emitted block-specific payload pages; this avoids silently
splitting a bit-packed page at the wrong token boundary. For Metal, align binary
payload sections to at least a cache-line boundary and keep K and V page
payloads independently addressable so the first implementation can dequantize
one side without touching the other.

## Restore Strategy

Implement restore in three stages:

1. **Reference restore:** read compressed pages, dequantize to MLX arrays, and
   reuse the existing attention paths. This validates schema, quality, and
   retained-State behaviour before optimizing. `TurboQuantKVCache` now owns
   compressed `TurboQuantKVReferencePagePayload` pages and regenerates arrays as
   the compatibility bridge.
2. **Pinned page restore:** memory-map the State payload, pin the relevant
   compressed page bytes, and wrap the page as MLX data or C++23 `mdspan`
   views. This removes copy pressure but may still dequantize before attention.
3. **Compressed attention:** keep K pages compressed through score computation.
   Query vectors stay high precision; the native kernel applies centroid and
   QJL corrections while walking compressed pages.

At every stage, local Gemma 4 caches must remain bounded to their configured
sliding window. Only global owner layers should show retained long-context
growth.

## Integration Points

- `go/internal/metal.TurboQuantKVPageLayout` is the first concrete metadata
  contract for `turboquant-kv-v1` pages. It validates rank-4 logical shape,
  exact layout version, K=`TurboQuantprod`, V=`TurboQuantmse`, QJL seed
  presence for keys, outlier masks, and effective-bit accounting.
- `memory.KVCacheModeTurboQuant` remains opt-in and never selected by
  `NewPlan` until quality gates pass.
- `scaleKVElements(..., KVCacheModeTurboQuant)` is a lower-bound data estimate
  at `3.5` bits per element. Once metadata is real, planner estimates must add
  norms, QJL residual norms, seeds/codebook ids, outlier masks, and page index
  overhead.
- `go/internal/metal.TurboQuantKVCache` exists beside `PagedKVCache`, not hidden
  inside q8. It is selected only by the explicit `turboquant` cache mode. The
  reference cache now emits K=`TurboQuantprod` and V=`TurboQuantmse` payloads
  with deterministic 3-bit regular channels and 4-bit outlier channels over the
  high half of the head dimension. The stored codec metadata names the
  outlier split as `outlier_policy=high-half-head-dim-v1`, records
  `norm_policy=explicit-vector-norm-bf16-v1` for K and V, and records
  `residual_norm_policy=explicit-vector-residual-norm-bf16-v1` for K because
  only `TurboQuantprod` carries the QJL residual path. The bit split gives
  `3500` effective bits/milli for both K and V in the stored layout.
- Snapshot, prompt-cache, and public State restore accept TurboQuant only when
  the page schema version matches exactly; older, empty, or partial snapshots
  fail clearly. `kv.Snapshot` v5 keeps compressed page payloads opaque at the
  portable layer and preserves them through State block save/load.
- Driver reports must label TurboQuant separately from `fp16`, `q8`,
  `k-q8-v-q4`, `paged`, and `fixed`.

Current focused benchmark on the M3 Ultra dev target:

```text
BenchmarkTurboQuantKVCache_Update_D128_T8              88428 ns/op 165193 B/op 234 allocs/op
BenchmarkTurboQuantKVCache_SnapshotRestore_D128_T8     34084 ns/op  65806 B/op  86 allocs/op
BenchmarkTurboQuantKVReferencePage_Encode_D128_T8      31623 ns/op  75776 B/op  98 allocs/op
BenchmarkTurboQuantKVReferencePage_DecodeBase_D128_T8  15903 ns/op  49152 B/op  50 allocs/op
BenchmarkTurboQuantKVReferencePage_EstimateKeys_D128_T8 14493 ns/op 36896 B/op  41 allocs/op
BenchmarkTurboQuantKVReferencePage_PackedPayload       15227 ns/op   8416 B/op  46 allocs/op
BenchmarkTurboQuantKVReferencePage_DecodePayload       13602 ns/op   6144 B/op  26 allocs/op
BenchmarkTurboQuantKVReferencePage_DecodePayloadArrays 33574 ns/op  63657 B/op  80 allocs/op
```

These are reference-path costs, not production-kernel targets.

## Validation Matrix

Minimum pre-promotion checks:

- CPU/reference round trips for MSE K/V rows, zero vectors, bad shapes, and
  packed bitstreams.
- Seeded statistical test that the K-side `TurboQuantprod` estimator is
  unbiased within tolerance over random query/key pairs.
- Metadata tests for outlier masks, effective-bit accounting, and page
  alignment.
- Restore tests proving unsupported TurboQuant snapshots fail closed, then
  versioned snapshots restore through the reference path.
- Greedy generation parity/quality checks against fp16 or paged cache on short
  prompts before any long-context run.
- Retained workflow tests at the normal `30k`-`40k` opencode-sized target and
  the `100k` stress lane, reporting restore, raw decode, wall time, peak memory,
  estimated energy, and long-output coherence.
- Focused benchmarks only: page encode, page dequant, pinned restore, and
  compressed attention. Avoid broad cache bench sweeps that accumulate MLX
  memory across unrelated cases.

Promotion requires TurboQuant to beat the accepted retained-State baseline on
memory or wall-clock without visible quality drift. It should not be promoted
for a short-context decode number alone.
