<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# go-mlx Upstream TODO

This file is the short upstream request list for making the State `.kv`
container path real instead of a smoke-test packer.

Active optimisation work must stay on the paged retained-State path. Do not use
context-length cutoffs or fixed Gemma 4 K/V lanes for current benchmarks unless
the user explicitly asks to reproduce old diagnostic rows. Runtime and tests
should describe accepted contexts by the real workflow shape: 32k opencode
seeds, 100k retained-State growth, or the model window.

## Current handover checkpoint

Status on `dev`, 2026-05-25: recent pushed handover commits include `ca0aad3`
(`perf(metal): create decode token inputs directly`), `ac64e4d`
(`docs(runtime): clarify benchmark handover state`), and `138dec1`
(`docs(todo): add current handover checkpoint`). The tree was clean after those
pushes to `homelab`, `origin`, and `github`.

Use `GOAL.md` as the detailed historical ledger, but treat missing
`docs/runtime/2026-*` artefact links as archived notes unless the report is
regenerated and checked in again. Fresh working reports may still live under
`/private/tmp/go-mlx-goal/reports` during active tuning.

Next code work should be one contained change at a time, with focused tests and
benchmarks before commit. Stay on the accepted paged retained-State path:
no fixed-cache default, no context-family cutoff, no forced compaction during
benchmarks, no native paged-attention promotion without a real retained
workflow win, and no sampler/lookahead changes unless the retained-session
state-advance parity guard is extended first.

Default CLI polish in progress: keep `driver-profile` aligned with
`DefaultProductionLane()` for the plain fast-lane shape unless a caller sets an
explicit flag. Do not reintroduce the older one-run, 32-token smoke default as a
production acceptance path.

## P0 - Enchantrix `pkg/trix`: streaming container API

Status: landed on Enchantrix branch `dev/go-mlx-trix-stream` at `14d89c2`;
`go/go.mod` currently consumes the pseudo-version from that commit.

`go-mlx` needs to pack large State logs without loading the full `.mvlog` into a
Go `[]byte`. The current `trix.Encode` API accepts a `Trix{Payload: []byte}`,
which is fine for small files but wrong for 30k-128k State windows.

The branch adds streaming helpers while preserving the existing API:

```go
func EncodeStream(header map[string]interface{}, magicNumber string, payload io.Reader, w io.Writer) (int64, error)
func DecodeHeader(r io.Reader, magicNumber string) (header map[string]interface{}, payload io.Reader, err error)
func DecodeStream(r io.Reader, magicNumber string, payload io.Writer) (header map[string]interface{}, n int64, err error)
```

Acceptance:

- Same wire format as RFC-0002:
  `[magic:4][version:1][header_len:4][json_header][payload]`
- Custom 4-byte magic still supported.
- Header max-size validation still enforced.
- Payload is copied with `io.Copy`, not `io.ReadAll`.
- `DecodeHeader` leaves the reader positioned at the payload so go-mlx can later
  stream or mmap the tail directly.
- Tests include a payload larger than 64 MiB and prove bounded allocations.

## P0 - Enchantrix `pkg/trix`: payload offset helper

Status: landed on Enchantrix branch `dev/go-mlx-trix-stream` at `14d89c2`.

For direct State restore we need the byte offset of the binary tail.

The branch adds:

```go
type HeaderInfo struct {
    Header        map[string]interface{}
    PayloadOffset int64
    PayloadBytes  int64 // optional when the reader is seekable
}

func ReadHeaderInfo(r io.ReaderAt, magicNumber string) (HeaderInfo, error)
```

Acceptance:

- Works with `*os.File`.
- Does not read the payload.
- Validates magic, version, and header length.
- Returns the exact offset immediately after the JSON header.

## P0 - go-inference `state/filestore`: relocatable segment aliases and embedded regions

Status: segment aliases were pushed to `external/go-inference` dev at
`303e835` as `OpenWithSegmentAlias(ctx, path, canonicalSegment)`. Embedded
regions were pushed at `e1ce07a`, and mapped borrowed chunks at `41a48af`. The
current dev branch now has the read-only embedded-region path
`OpenRegionWithSegmentAlias(ctx, path, payloadOffset, payloadBytes,
canonicalSegment)` plus borrowed byte reads via `BorrowBytes` /
`BorrowRefBytes`. The large-payload store-open allocation fix landed at
`e05c165` as `perf(state): bound filestore open preallocation`.

The current file-backed State store validates `ChunkRef.Segment` against the
opened store path. That is correct for safety, but a `.kv` container extracted
to a temporary path fails because the folded State block refs still point at
the original segment path.

The safe alias/open options are:

```go
func OpenWithSegmentAlias(ctx context.Context, path string, canonicalSegment string) (*Store, error)
func OpenRegionWithSegmentAlias(ctx context.Context, path string, payloadOffset int64, payloadBytes int64, canonicalSegment string) (*Store, error)
func BorrowRefBytes(ctx context.Context, store Store, ref ChunkRef) (BorrowedChunk, error)
```

Acceptance:

- `ResolveRefBytes` accepts refs whose `Segment` equals either the physical
  opened path or the explicit canonical segment alias.
- The default `Open` behaviour remains strict and unchanged.
- Alias mode is opt-in and covered by tests for matching alias, physical path,
  and wrong segment rejection.
- Region mode keeps frame offsets relative to the embedded State payload while
  reading from `payloadOffset + frame_offset` inside the `.kv` container.
- Region mode is read-only so a wake from a packed State file cannot append
  chunks into the middle of a container.
- Region borrows are mmap-backed on Darwin/Linux/BSD targets and fall back to a
  copy where mmap is unavailable, keeping the public State contract portable.
- The store still writes new refs using the physical path unless an explicit
  write-segment option is also provided.

Current go-mlx bridge: direct `.kv` wake reads the Trix header without touching
the payload, opens the `.kv` file itself as a read-only State region using the
payload offset and byte length, and keeps the original `state_store_path` as the
canonical segment alias. This removes the temporary `.mvlog` materialisation
step while preserving strict segment validation. Raw State block loading now
uses borrowed bytes first, so native KV tensor slices parsed from a `.kv` region
can flow into the existing pinned MLX array restore path without a per-block
heap copy. The first real retained wake proof is now recorded in `GOAL.md`:
the packed `.kv` wake cut wake-phase Go heap allocation from about `49.45 MB`
to `157 KB` while keeping decode flat on the same 658-token folded state. The
follow-up store-open proof is also recorded in `GOAL.md`: the same packed
`440 MB` State payload now opens with `17 KB` of total Go allocation instead of
about `481 MB`.

## P1 - Enchantrix `pkg/trix`: no default transforms for State KV

The State `.kv` format must keep the payload raw by default. Compression and
encryption can be optional later, but the first production path needs the binary
tail to remain byte-for-byte identical to the `.mvlog` input so it can become a
zero-copy mmap/pinned view later.

Status: covered by the Enchantrix streaming tests; keep this as a contract for
future transform support.

Acceptance:

- The streaming encode/decode tests assert payload byte equality.
- No implicit sigil, compression, checksum string conversion, or encryption is
  applied unless the caller explicitly asks for it.

## P1 - Borg: raw Trix file/container helpers

Borg is helpful for DataNode-backed packaging, but go-mlx needs a raw-file State
container, not a tarred DataNode, for the hot path.

Helpful additions:

```go
func ToRawTrix(header map[string]interface{}, magic string, payload io.Reader, w io.Writer) (int64, error)
func FromRawTrixHeader(r io.ReaderAt, magic string) (trix.HeaderInfo, error)
```

Acceptance:

- Delegates to Enchantrix streaming Trix helpers.
- Does not tar, encrypt, compress, or allocate the full payload.
- Keeps Borg's current DataNode helpers unchanged.

## P2 - Poindexter: State index sidecar shape

Less urgent, but useful once `.kv` files can hold multiple State segments or
reference other State files.

Desired shape:

```json
{
  "kind": "go-mlx/state-index",
  "states": [
    {
      "id": "session-1-fold-1",
      "path": "session-1.kv",
      "index_uri": "mlx://state-ramp/fold/1/folded/index",
      "token_count": 206,
      "payload_offset": 1234,
      "payload_bytes": 80511040
    }
  ]
}
```

Acceptance:

- A tiny API can append and query State entries by `index_uri`.
- It can point at one `.kv` file or many `.kv` files.
- It avoids reading the binary State payload.

## Current go-mlx bridge state

`go-mlx` is adding a `state-pack` CLI that uses
`forge.lthn.ai/Snider/Enchantrix/pkg/trix` with magic `KVST` and header kind
`go-mlx/state-kv`.

That bridge proves the JSON-head/binary-tail format with streaming pack and
header-only wake. The current wake path uses the `.kv` payload offset directly
through `OpenRegionWithSegmentAlias`, so it no longer creates a temporary
`.mvlog` copy. Raw State block payloads are now borrowed from the mmap-backed
region where the platform supports it and are handed into the existing pinned
MLX array restore path. The next proof point is no longer "does `.kv` wake
without copying blocks" or "does store-open avoid giant heap preallocation";
both now do. The next useful target is retained decode graph/materialisation:
the request-context traces still show the dominant per-token bucket in
`sample_eval`, where lazy MLX materialises the current one-token forward graph
and sampler.

Do not reintroduce any arbitrary context boundary or production fixed-cache
default while chasing this. Context size can select chunking and
overflow/compact limits, but it must not select a different K/V family or
invent a fixed-cache budget for benchmark convenience. The overflow/compact
threshold must also stay unarmed during ordinary benchmarks: retained growth is
limited by the requested target unless a fold store is configured for explicit
overflow compaction.

Current retained decode evidence: the real async prefetch runtime gate and the
new `prefetch` token-phase bucket prove the old large `other` bucket is the
async next-logits materialisation boundary. On the 2026-05-24 two-turn
request-context trace, `prefetch` averages about `6.33 ms/token`, while
`sample_eval` is about `3.28 ms/token` and `forward` about `1.56 ms/token`.
The dirty-KV prefetch pass now evaluates next logits with only the cache arrays
touched by the most recent token update. This is accepted because it improves
the same 10-turn retained request-context row from `84.633` to `86.125 tok/s`
raw decode and from `72.744` to `73.839 tok/s` effective throughput while
preserving paged K/V, bounded 512-token local windows, and no fixed caches.
The rejected prepared-sampler prefetch probe confirms that splitting the
deterministic top-k/top-p candidate graph is still too small: it improved a
sampler-only microbench but regressed the real retained trace to `81.338 tok/s`
and left `sample_eval` around `3.37 ms/token`. The next optimisation should
still target the larger MLX graph/eval boundary directly without changing the
paged retained-State semantics.
The 2026-05-25 native suppressed top-k/top-p sampler wrapper confirms the same
boundary issue from the other direction: a C++ compiled sampler/suppression
wrapper slightly helped one isolated suppressed microbench but regressed the
same-output two-turn retained trace from `91.599` to `86.285` raw tok/s. Keep
sampler changes inside the accepted Go/compiled sampler shape until a larger
stable logits/eval boundary is available.
Direct `RandomCategorical` benches now exist for the 32k and 262k vocab
sampler edge. They are for attribution only: the zero-key handle probe remains
rejected because the retained request-context row regressed even though the
isolated wrapper benchmark moved slightly.
The sampled-token lookahead variant is also rejected: trying to materialise the
next sampled token inside the prefetch boundary caused the gated trace to end
turn 1 with `empty_visible_output` and `0` generated tokens, while the same
rebuilt binary with the gate off completed normally. Any future lookahead work
needs a first-token token/RNG parity harness before it is allowed near the
retained benchmark lane.
The scalar sampled-token sync variant is also rejected for production: a direct
`next.Int()` materialisation microbench beat the explicit `Eval(next)` row, but
the matched two-turn retained trace regressed from `91.024` raw tok/s to
`89.175` raw tok/s and from `81.968` effective tok/s to `80.465`. Keep the
benchmark probe; keep production on explicit sampled-token eval.
The guarded combined sample/logits eval boundary is now benchmarked too. It
only moved the suppressed Gemma-sized row from `516.277us` to `511.315us`, and
the retained-shaped logits+dirty-K/V row from `517.691us` to `515.825us`. That
is useful attribution but too small to justify a second runtime lookahead probe
after the previous retained failure.
The attention query dtype cast is also now defended by evidence. Mixed
`Q=float32`, `K/V=float16` SDPA is correct, but the retained fast-concat shape
is much slower without the cast (`8` pages: `435.944us` cast vs `640.400us`
mixed; `16` pages: `645.359us` cast vs `995.736us` mixed) and uses more MLX
active-cache memory. Do not remove `attentionQueryForKV` as apparent
boilerplate.
That harness now exists as `TestSample_PrefetchTokenEvalParity_Good`: it proves
normal guarded sampling and combined `EvalAsync(logits, sampled_token)`
materialisation return the same first token under the same seed. Future
lookahead work must extend this guard to the retained-session state-advance
boundary before running full request-context traces.
`TestModelSession_PrefetchTokenStateAdvanceParity_Good` now covers that
retained-session boundary with a paged cache: normal two-token generation must
match a manual path that advances state and evaluates next logits, the next
sampled token, and dirty K/V together. Future lookahead work can build on this
guard, but still must prove the full retained request-context trace before it
is considered for production.

Trace timing now keeps the default `TraceTokenPhases` path on the same combined
`EvalAsync(logits + dirty K/V)` boundary as production generation. The older
split timing smoke at
`/private/tmp/go-mlx-goal/reports/2026-05-24-trace-prefetch-split-smoke.json`
remains useful attribution evidence only: it showed dirty-cache prefetch was
about `9.124 us`, but it measured a split eval shape that production does not
use. Current trace rows should read `prefetch_logits` as the whole combined
prefetch boundary when logits are present; `prefetch_cache` is reserved for
cache-only diagnostics. The two-turn opencode proof is recorded in `GOAL.md`
and keeps paged/no-fixed/no-context-cutoff invariants.

The zero-empty-handle SDPA cleanup is also recorded in `GOAL.md`. It removes
per-attention empty native handle allocation for absent masks/sinks, but the
matched production-shaped trace is neutral (`91.599` raw tok/s versus
`91.608` before), so it is a cleanup rather than a parity milestone.
The concat parent-slice cleanup follows the same pattern: `Concatenate` no
longer allocates a Go `inputs` slice for `newArray`, because `newArray` no
longer stores parent references. Focused Metal benches moved
`BenchmarkPromptCache_KVConcat_16Pages_256Each` from `128 B/op` and
`1 alloc/op` to `0 B/op` and `0 allocs/op`; paged fast-concat K+V moved from
`2 allocs/op` (`128 B/op` at 8 pages, `256 B/op` at 16 pages) to `0 allocs/op`.
This is retained as a hot-path allocation cleanup, not as evidence that the
owner-layer attention materialisation gap is closed.
`Eval`/`EvalAsync` also now hand a pooled contiguous run of output handles to a
native helper instead of issuing one cgo append call per output. The stack
buffer variant was rejected because it regressed Go allocations; the pooled
variant keeps `BenchmarkAsyncDecodePrefetchTrace_CombinedDirtyKV` in the same
`1 alloc/op` profile and moves the focused prefetch bench from the previous
`160.024-179.131 us/op` band to `164.487-165.937 us/op`. Treat it as cgo
boundary hygiene only; it does not replace the larger logits/materialisation
fusion target.
The prefetch benchmark now also measures the production non-trace boundary and
keeps the cache slice outside the hot loop. The corrected Metal row records
production combined prefetch at `177.954 us/op`, `512 B/op`, `1 alloc/op`, trace
combined at `175.221 us/op`, `512 B/op`, `1 alloc/op`, and trace split at
`184.888 us/op`, `560 B/op`, `3 allocs/op`. A slice-only internal prefetch/eval
patch was tested and reverted because it kept the same `512 B/op`, `1 alloc/op`
while moving the combined trace row from `173.397 us/op` to `176.224 us/op`.
Do not chase that varargs/cache-slice shape; the remaining target is still the
larger MLX logits/materialisation boundary.
`CompiledFunc.CallOne` now moves the one-input/one-output closure apply path
into one C helper. The focused compiled sampler row improves from
`496.546 us/op`, `8 B/op`, `1 alloc/op` to `450.085 us/op`, `0 B/op`,
`0 allocs/op`; production-shaped suppressed sampler rows improve to the
`475-486 us/op`, `7-8 B/op`, `1 alloc/op` band. This is accepted as a
sampler/materialisation boundary cleanup, but still needs a retained
request-context rerun before it can be counted as a workflow parity milestone.
That retained rerun now exists:
`2026-05-25-state-ramp-request-context-callone-helper-go-mlx-gemma4-e2b-4bit-opencode-30k-r10-g1024.json`.
It keeps the same `10/10`, `4476` visible-token output shape and paged/no-fixed
cache invariants, improves raw decode from `87.483` to `87.687 tok/s`, and
drops `sample_eval` from `3.305ms/token` to `3.274ms/token`. The wall delta is
only `16ms`, so this is accepted cleanup evidence, not a parity close. The
dominant remaining bucket is still `prefetch_logits` at about `6.726ms/token`.
The next concat cleanup is now accepted at the two-array boundary only:
`concatenate2` builds its temporary MLX vector on the C stack and keeps the same
graph. The 16-page fast-concat mixed-query bench median moved from about
`627.381 us/op` to `601.880 us/op`, while the prompt-cache concat median stayed
allocation-neutral and moved from about `238.422 us/op` to `236.052 us/op`.
Do not revive the broader Go handle-array `mlx_vector_array_new_data` attempt:
it regressed the same benches to `1152 B/op` and `2305-2308 B/op`, so multi-page
concat still needs a true C-side page-list owner rather than a Go slice handoff.
Two scalar C-side page-list variants were also rejected: 64 slots was too heavy,
and 32 slots covered the current `24` max-page request-context trace but left the
actual 16-page fast-concat SDPA median around `623.972 us/op` versus the accepted
two-array helper's `601.880 us/op` row. Prompt-cache-only concat wins do not
justify a retained decode change.
`PagedKVCache` dirty-state marking now uses a fixed pair helper instead of the
old variadic helper on per-token updates. Focused tests pass, and
`BenchmarkPagedKVCache_UpdateBorrowedPages_To128` is allocation-stable while
moving from the sweep's `1129903 ns/op` to repeated rows around
`1072846-1077538 ns/op`. This is small paged-State hygiene, not a parity close.
Decode continuation inputs now use a direct rank-2 int32 constructor instead of
`fromSingleInt32` followed by `Reshape2(..., 1, 1)`. This removes the
per-token reshape graph node from `Model.Generate`, retained
`ModelSession.Generate`, prompt-cache exact replay, split continuation, and the
Gemma 4 assistant continuation paths. Focused shape/continuation tests pass; the
matched constructor microbench moves from about `745-760 ns/op`, `8 B/op`, and
`1 alloc/op` to about `310-319 ns/op`, `0 B/op`, and `0 allocs/op`. This is a
contained handover-safe cleanup, not a new runner-parity row.
Two adjacent probes are rejected there too: zero-value random key handles
regressed the matched trace to `90.113` raw tok/s, and yielding retained-session
tokens before async prefetch regressed it to `88.045` raw tok/s despite the
nicer first-token timestamp. Do not revive either as a default-path cleanup.

The per-token eval boundary now detaches logits together with caches after the
sampled token is materialised. That should reduce graph lifetime pressure while
preserving the paged retained-State semantics. The matched 30k request-context
retained run and the uncapped 100k stress proof are now recorded in `GOAL.md`;
the 100k boundary trace with paged-concat native event details is also recorded
there. Follow-up probes rejected native paged attention and forced single-token
last-logits defaults for the production lane: both failed to improve the
10-turn retained workflow. The next optimisation should aim at a fused
logits/materialisation boundary or sampler/eval fusion, not at reviving
fixed-cache, native paged attention, forced last-logits, or context-cutoff
behaviour.
