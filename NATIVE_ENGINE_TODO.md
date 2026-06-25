<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# Native Engine TODO

This is the active worklist for moving `pkg/native` toward replacing
`pkg/metal`. It is paired with the fresh 2026-06-22 `GOAL.md`; older goal notes
should be treated as stale unless the evidence is re-verified here. The rule for
this list is simple: copy the proven engine contracts from `pkg/metal`, remove
the CGO/MLX dependency, and keep the native route measurably faster or simpler.
Do not add new gates or settings to hide missing native behaviour.

## Current Focus

- [ ] Port the `pkg/metal` prompt-cache/session engine semantics into
  `pkg/native`:
  - exact token-prefix reuse
  - cache clear/reset behaviour
  - prefix truncation/compaction semantics
  - state restore hooks where the root API already exposes them
- [ ] Make native decode replay the default hot route:
  - keep ICB/session replay for eligible dense paths
  - remove real-model fallbacks caused by missing native engine features
  - preserve byte parity against the non-ICB path
- [ ] Move PLE fully into the native engine:
  - keep per-token PLE inputs GPU-resident
  - keep PLE gates inside the recorded/replayed path
  - avoid returning hot PLE tensors through host `[]byte`
- [ ] Move MoE fully into the native engine:
  - GPU router scores
  - GPU top-k selection
  - grouped expert matvec without host readback
  - ICB eligibility for MoE once router readback is gone
- [ ] Add native async decode prefetch semantics copied from `pkg/metal`:
  - next hidden/logits pre-evaluation boundary
  - dirty-cache materialisation boundary
  - per-token timing evidence equivalent to `prefetch`
- [ ] Close native KV-cache feature parity:
  - fixed cache
  - paged cache
  - rotating/sliding cache
  - quantized/native raw cache restore
  - prefix copy/view/restore helpers
- [ ] Finish no-copy quant residency:
  - prove aligned packed weight views are safe across layers
  - keep misaligned fallback contained and measured
  - remove duplicate resident copies where byte parity is proven
- [ ] Mirror `pkg/metal` benchmarks for the native engine:
  - prefill
  - decode
  - ICB vs fallback
  - PLE
  - MoE
  - head/sampler
  - cache update/restore
  - command-buffer waits and host bytes copied per token

## Live Evidence - 2026-06-22

- Default package coverage with metallib:
  `go test ./go/pkg/native -coverprofile=/private/tmp/go-mlx-native-cover.out
  -covermode=count -count=1` passes in 32.535s at 85.2%.
- `metal_runtime` package coverage:
  `go test -tags metal_runtime ./go/pkg/native
  -coverprofile=/private/tmp/go-mlx-native-metal-cover.out -covermode=count
  -count=1` passes in 39.531s at 91.8%.
- Current `>=95%` package coverage target is not met. The function-level gaps
  are broad, mostly outside the first session/cache lane: audio encoder/tower
  branches, vision tower/projector, MTP, training backward/LoRA, Mamba/composed
  loader branches, ICB-only helpers, and remaining prompt-cache/state branches.
- `pkg/metal` still exposes a fuller session engine surface than native:
  `Model.NewSession()` consumes a `metal.SessionHandle` with prompt/token
  prefill, append, generate, `CaptureKV`, `RangeKVBlocks`, `RestoreKV`, block
  restore fallbacks, and prompt-cache restoration from KV snapshots/blocks.
  The native text wrapper currently retains a private prompt-cache session but
  does not yet expose the public root session/KV snapshot handle. This is the
  next concrete parity feature to copy without adding a gate or setting.
- Audio feature extraction/helper coverage added:
  `go test ./go/pkg/native -run
  'Test(AudioFeature|AudioRFFT|HTKMel|AudioHelpers|RMSRowsHost|AudioPositionTable|ReLUF32|Conv2dF32|LayerNormF32|Mamba2EpsFromConfig|NativeTokenModelVocab)'
  -count=1` passes. This covers the host audio feature config/extractor, HTK
  mel filterbank, radix-2 RFFT, audio clamp/activation/RMS helpers, F32 Conv2d
  and LayerNorm parity wrappers, Mamba2 eps probing, and the native token-model
  vocab accessor without model assets.
- Additional deterministic helper/audio coverage:
  `go test ./go/pkg/native -run
  'Test(SelectProj|VisionGridForPatchCount|VisionPoolerBranches|VisionStandardize|VisionProjectorNoProjectionNormalisesRows|SlideWindowBounds|ArchPLE|AudioEncodeAndSubsampleF32InputGuards|NativeTokenModelSpecialLoaderErrors|ResetResidentBufsForTestClearsCache|AudioSubsampleF32|AudioEncodeNoLayers)'
  -count=1` passes. This covers LoRA projection selection, vision grid/pool/
  standardise/projector helper branches, sliding-window bounds, native PLE
  payload/layer validation, special loader error routes, resident buffer reset,
  the real fp32 audio subsampler path, and the no-layer audio encoder
  composition path.
- Prompt-cache/state validation coverage added:
  `go test ./go/pkg/native -run
  'Test(PromptCacheInputGuards|WarmPromptCacheReusesResidentIDBacking|WarmPromptCachePrefillsResidentPrefix|GenerateCachedReusesResidentIDBacking|CompactCacheReusesRetainedIDBacking|SessionStateNoRuntimeValidation|SessionStateSerializeZeroLayerCachedIDs|NativeBackendDecodeForwardRejectsPLEWholeSequence)'
  -count=1` passes. The red run for
  `TestWarmPromptCacheReusesResidentIDBacking` showed the second warm prefix
  changed the resident id backing pointer; the green path keeps the backing
  slice across re-warms while still rewinding the KV cursor and visible prefix.
- New warm prompt-cache benchmark:
  `go test ./go/pkg/native -run '^$' -bench
  '^BenchmarkWarmPromptCacheRetainedIDs$' -benchmem -benchtime=5x -count=3`
  reports about 2.60-2.99 ms/op, 1,244,089-1,244,176 B/op, and
  28,715 allocs/op on the tiny synthetic fixture. The benchmark is now in the
  prepared `_bench_test.go` lane for repeated warm-prefix resource tracking.
- Focused resident-head benchmark evidence:
  `BenchmarkLMHeadQuant` reports about 356-377 us/op with 115-138 KB
  `rss-grow-B/op`; `BenchmarkHeadEncoderQuant` reports about 378-388 us/op
  with 84-134 KB `rss-grow-B/op`. At this synthetic size the resident head is
  not faster, but its steady resident runs reduce RSS growth while proving the
  packed 32 MB head is not re-uploaded per token.
- Decode microbenchmarks from the same run:
  fallback `BenchmarkDecodeForwardArchOneLayerTwoTokens` is about
  0.93-1.06 ms/op with 193 KB/op and 4456 allocs/op; ICB is about
  1.72-2.36 ms/op with 164 KB/op and 3685 allocs/op. ICB currently saves
  allocations/bytes here, but is slower on this one-layer synthetic case.
- Prompt-cache state restore regression was driven red before the fix:
  `TestSessionStateRoundTripRestoresCachedPrefixMetadata` restored
  `CachedPrefixLen = 0`, then passed after native state snapshots started
  carrying cached token-id metadata.
- Focused prompt-cache/state suite:
  `go test ./go/pkg/native -run
  'Test(SessionStateRoundTrip|SessionStateSerializeCachedPrefixAllocationBudget|GenerateCachedPrefixReuse|WarmPromptCachePrefillsResidentPrefix|GenerateCachedReusesResidentIDBacking|CompactCacheContinuation|CompactCacheReusesRetainedIDBacking|CompactCacheAllocationBudget|ClearPromptCacheDropsNativePrefixState)'
  -count=1` passes in 0.946s.
- Root native text-model prompt-cache forwarding:
  `go test ./go -run 'TestNativeTextModelWarmPromptCacheUsesCachedSession'
  -count=1` passes in 0.461s. The red run failed because
  `nativeTextModel` had no `WarmPromptCache` or `ClearPromptCache` methods; the
  green path opens a retained native prompt-cache session, warms it with
  tokenized IDs, routes the next greedy generation through `GenerateCached`,
  and clears the retained session on request.
- Root native text-model chunk prompt-cache forwarding:
  `go test ./go -run '^TestNativeTextModelWarmPromptCacheChunksUsesCachedSession$'
  -bench '^BenchmarkNativeTextModelWarmPromptCache(Chunks|FallbackJoin)$'
  -benchmem -count=3` passes. `WarmPromptCacheChunks` now copies metal's chunk
  tokenisation semantics (strip implicit BOS after the first non-empty chunk)
  into the retained native prompt-cache session. The chunk route improved from
  the first implementation's 8,352 B/op and 398 allocs/op to 7,336 B/op and
  391 allocs/op after preallocating the token accumulator, but on the tiny
  repeated-token fixture it remains slower and more allocation-heavy than the
  joined fallback (about 2.40-2.60 us/op, 6,072 B/op, 14 allocs/op). Treat this
  as feature parity plus a partial resource win, not a global speed win.
- Native token-model BF16 embed resource guard:
  `go test -tags metal_runtime ./go/pkg/native -run
  'TestNative(TokenModel_ContractParity|BF16TokenModelEmbedSingleTokenAllocationBudget)'
  -count=1` passes in 0.588s. The red run measured two allocations per
  single-token `NativeTokenModel.Embed`; the green path uses the direct
  one-token BF16 helper and is guarded at <=1 allocation.
- Native token-model quant embed resource guard:
  `go test -tags metal_runtime ./go/pkg/native -run
  'Test(NativeTokenModel_QuantContractParity|NativeQuantTokenModelEmbedSingleTokenAllocationBudget|NativeBF16TokenModelEmbedSingleTokenAllocationBudget|EmbedLMHeadQuant)'
  -count=1` passes in 0.555s. The red run measured two allocations per
  single-token quant `NativeTokenModel.Embed`; the green path shares the direct
  one-token quant row helper with the quant session path and is guarded at <=1
  allocation.
- `BenchmarkNativeTokenModelEmbed` now reports about 106.2-109.4 ns/op,
  1.17-1.21 GB/s, 128 B/op, and 1 alloc/op for the BF16 contract embedding
  path. This is the root contract route used by native `model.Generate`, not
  only the lower-level `ArchSession` path.
- `BenchmarkNativeQuantTokenModelEmbed` now reports about 104.3-108.5 ns/op,
  1.18-1.23 GB/s, 128 B/op, and 1 alloc/op for the 4-bit quant contract
  embedding path.
- New state snapshot benchmark:
  `BenchmarkSessionStateSerializeCachedPrefix` now reports about 14.3-14.4 us/op,
  20.5-20.7 GB/s, 306,920 B/op, and 82 allocs/op for a warmed cached-prefix
  session. The previous baseline from the same benchmark was about 42-44 us/op,
  1,257,283 B/op, and 90 allocs/op, so the direct snapshot writer removes the
  large append-growth copy overhead while leaving the unavoidable resident K/V
  copy intact.
- `TestSessionStateSerializeCachedPrefixAllocationBudget` now guards the
  warmed prompt-cache state snapshot path at <=82 allocations, after a red run
  showed the old implementation at 90 allocations.
- `TestGenerateCachedReusesResidentIDBacking` now guards the prompt-cache hot
  path's resident token metadata. The red run showed a repeated cached generate
  dropped backing capacity from 11 to 7 by allocating a fresh slice; the green
  path reuses the existing backing store when capacity is available.
- `TestCompactCacheReusesRetainedIDBacking` now guards compaction's retained
  token metadata. The red run showed the retained suffix moved to a fresh
  backing array; the green path keeps a suffix view and avoids that metadata
  copy while preserving the re-prefill semantics.
- `BenchmarkCompactCacheRetainedIDs` now reports about 2.13-2.26 ms/op,
  995,989-996,005 B/op, and 22,977 allocs/op for compacting a warmed session
  down to four retained tokens. The previous measured path was 2.77-2.86 ms/op,
  1,002,834-1,002,844 B/op, and 22,992 allocs/op. The win comes from using the
  existing dense batched decode route for eligible compact re-prefill and from
  skipping final-hidden readback on compact replay. The remaining allocation
  count is still dominated by Metal command/buffer wrapper churn during required
  retained-token re-prefill.
- `TestCompactCacheAllocationBudget` now guards the compact resource path at
  <=22,980 allocations. The original compact path measured 22,992 allocations,
  so the guard catches regression without pretending the remaining Metal wrapper
  churn has been solved.
- MTP decode now has no-runtime input guard coverage:
  `go test ./go/pkg/native -run
  '^TestMTPDecode(InputGuards|BatchedTokenIdentity|DraftEqualsTargetAllocationBudget)$'
  -count=1` passes in 0.738s. The allocation guard was driven red before the
  resource change at 253,453 allocations, then passed after `MTPDecode` stopped
  allocating per-round draft/commit/verify slices.
- New MTP benchmark:
  `go test ./go/pkg/native -run '^$' -bench
  '^BenchmarkMTPDecodeDraftEqualsTarget$' -benchmem -benchtime=3x -count=3`
  moved from about 45.3-49.7 ms/op, 11.091-11.093 MB/op, and
  253,460-253,463 allocs/op to about 43.7-48.7 ms/op, 11.075-11.077 MB/op, and
  253,418-253,419 allocs/op on the same synthetic draft-equals-target fixture.
  This is a small resource win, not the final MTP engine win.
- Dense MTP prompt prefill now uses the resident batched dense prefill route
  for prompt prefix rows, then steps only the final prompt token normally to
  carry the target cursor hidden. `TestMTPDecodeDensePromptPrefillAllocationBudget`
  was driven red at 64,109 allocations, then passed after the split with a
  <=64,108 guard.
- New MTP prompt-prefill benchmark:
  `go test ./go/pkg/native -run '^$' -bench
  '^BenchmarkMTPDecode(DensePromptPrefill|DraftEqualsTarget)$' -benchmem
  -benchtime=3x -count=3` passes. `BenchmarkMTPDecodeDensePromptPrefill`
  reports about 9.46-9.68 ms/op, 2,784,117-2,784,224 B/op, and
  64,077-64,078 allocs/op on the five-token dense prompt fixture.
- PLE prompt-cache warmup no longer allocates the dense-path `[][]byte`
  embedding batch when the per-layer-input route must replay sequentially.
  The allocation guard was driven red at 4,664 allocations with
  `TestWarmPromptCachePLESequentialAllocationBudget`, then passed after the
  split at <=4,662 allocations.
- New PLE prompt-cache benchmark:
  `go test -tags metal_runtime ./go/pkg/native -run
  '^TestWarmPromptCachePLESequentialAllocationBudget$' -bench
  '^BenchmarkWarmPromptCachePLESequential$' -benchmem -benchtime=5x -count=3`
  passes and reports about 7.81-9.31 ms/op, 219,955-219,984 B/op, and
  4,662 allocs/op on the synthetic four-token E2B/E4B-style PLE fixture.

## First Implementation Lane

- [x] Strengthen native prompt-cache/session semantics first. This is the
  smallest engine feature already present in both packages and gives a safe
  place to copy `pkg/metal` behaviour without inventing a parallel API.
- [x] Add `ArchSession.ClearPromptCache()` to mirror the metal engine's
  prompt-cache clear semantics at the native session layer.
- [x] Hoist `forwardLayer` into the shared native test helper file so default
  native tests can compile without the `metal_runtime` tag.
- [x] Repair native loader/no-copy test fixtures so registry-driven directory
  loads use explicit `model_type` and realistic safetensors matrix shapes.
  This keeps Mistral/Gemma BF16, quant, MoE, PLE, and no-copy byte-identity
  gates on the same route as real checkpoints.
- [x] Preserve native prompt-cache metadata across `SerializeState` /
  `RestoreState` round trips while accepting older snapshots that only contain
  KV tensors.
- [x] Add a session-state benchmark using the prepared `_bench_test.go` path so
  allocation/resource changes have repeatable evidence.
- [x] Reduce `SerializeState` resource usage by pre-sizing the native snapshot
  and writing metadata/KV bytes directly into one output buffer.
- [x] Reuse resident prompt-cache token metadata across cached generations
  instead of allocating a fresh `cachedIDs` backing slice on every successful
  call.
- [x] Reuse retained prompt-cache token metadata during `CompactCache` instead
  of copying the kept suffix before re-prefill.
- [x] Batch dense `CompactCache` re-prefill through the existing native
  session-level batched decode route when the model shape is eligible, and skip
  final-hidden readback for compact replay because compaction only needs K/V
  cache mutation.
- [x] Add native session warm-prefix support (`ArchSession.WarmPromptCache`) so
  prompt-cache warmup can prefill K/V without forcing a throwaway generated
  token.
- [x] Wire the root package's no-cgo `nativeTextModel` wrapper to retain a
  prompt-cache-capable native session after warmup, use it for greedy
  `GenerateCached`, and clear it through the same surface.
- [x] Wire `nativeTextModel.WarmPromptCacheChunks` to the same retained native
  prompt-cache session, copying metal's chunk tokenisation rule instead of
  forcing the root fallback to concatenate every prompt chunk.
- [x] Move the BF16 `NativeTokenModel.Embed` single-token hot path onto the
  direct native embedding helper, matching the `ArchSession` path and removing
  the wrapper slice allocation from the root contract generation route.
- [x] Move the quant `NativeTokenModel.Embed` and quant `ArchSession` single-token
  hot paths onto a direct native quant row helper, removing the wrapper slice
  allocation from the root 4-bit contract generation route and session route.
- [x] Add deterministic native audio feature/helper and F32 kernel parity tests
  so the audio extractor lane is covered without requiring heavyweight model
  assets.
- [x] Add deterministic native PLE, LoRA, vision, resident-buffer reset, and
  fp32 audio encoder/subsampler tests for reachable engine helper gaps.
- [x] Reuse native warm prompt-cache resident token metadata on repeated
  `WarmPromptCache` calls instead of allocating a fresh backing slice.
- [x] Add no-runtime native state snapshot validation tests for ICB rejection,
  legacy snapshots, truncated prompt-cache metadata, trailing metadata, and
  zero-layer cached-id round trips.
- [x] Pin the backend contract that PLE models must use the incremental
  id-aware session path, not whole-sequence `DecodeForward`.
- [x] Add MTP input guard coverage, a reusable MTP fixture, an MTP decode
  benchmark, and a stack-backed verify-id buffer that removes the per-round
  draft/commit/verify slice allocations for the common small draft block.
- [x] Batch dense MTP prompt-prefix cache mutation through the resident
  session prefill route and step only the final prompt token for the cursor
  hidden, reducing the short dense prompt-prefill allocation path.
- [x] Split native prompt-cache prefill so dense sessions keep the batched
  replay path, while PLE/ICB sequential sessions embed one token at a time and
  avoid the unused dense-path embedding batch allocation.
- [ ] Continue the prompt-cache/session lane with root native model forwarding
  and remaining KV/state restore parity where the current root API already
  exposes hooks.
