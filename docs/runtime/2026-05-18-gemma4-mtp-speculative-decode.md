<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# Gemma 4 MTP Speculative Decode Lane

## Decision

Gemma 4 MTP is worth pursuing, but it is not a prefill optimisation. It is a
separate speculative-decode lane for production visible throughput.

The raw parity lane remains target-model-only go-mlx versus target-model-only
llama.cpp, with prefill and decode reported separately. A speculative run can
be a valid user-facing throughput win only when it is labelled as speculative
and compared against a matching llama.cpp speculative run where possible.

## Why It Does Not Push Prefill

Prefill is the target model ingesting the prompt and building KV state. MTP
starts helping after that point: a drafter proposes several future tokens, and
the target verifies those candidates in a wider pass. That reduces the number
of serial target decode steps when the drafter is accepted, but it does not
remove the target prefill pass over the prompt.

If a benchmark reports one combined end-to-end tokens/sec number, speculative
decode can improve the combined number when generation is long enough. The
prefill metric itself should stay roughly unchanged or slightly worse if the
assistant model also needs its own initial state.

## Model Pairing

Google publishes Gemma 4 `-assistant` checkpoints for the MTP drafter role:

- E4B target lane: `google/gemma-4-E4B-it` with
  `google/gemma-4-E4B-it-assistant`.
- Current 26B A4B lane: `google/gemma-4-26B-A4B-it` with
  `google/gemma-4-26B-A4B-it-assistant`.

Do not use the E4B assistant as evidence for the 26B A4B target lane unless the
experiment is explicitly labelled as a mismatched-drafter probe.

## llama.cpp Reference

The local Homebrew llama.cpp build and the current upstream master are not
enough by themselves for Gemma 4 assistant MTP:

- Homebrew `llama-cli` build `8990`, commit `660b1b4bd`, rejects
  `--spec-type draft-mtp`.
- Upstream master at `/private/tmp/llama.cpp`, commit `1a68ec9`, exposes
  `draft-mtp` but cannot load the 26B assistant GGUF because it does not know
  the `gemma4_assistant` architecture.
- Unmerged PR `ggml-org/llama.cpp#23211`, cloned to
  `/private/tmp/llama.cpp-pr23211`, builds and runs the attached Gemma 4 MTP
  path on Metal. It is therefore useful R&D evidence, not an upstream-stable
  comparator.

The local 26B assistant GGUF used for the successful run is:

```text
repo: AtomicChat/gemma-4-26B-A4B-it-assistant-GGUF
sha: 171ecca181ec00ed6ffacb573195aa7c644bbdc6
file: gemma-4-26B-A4B-it-assistant.Q4_K_M.gguf
architecture: gemma4_assistant
```

Target model:

```text
repo: unsloth/gemma-4-26B-A4B-it-GGUF
sha: 3365c68df1a83799b846d05324ebfadbb8cc70b3
file: gemma-4-26B-A4B-it-UD-Q4_K_M.gguf
```

## 2026-05-18 llama.cpp PR 23211 Results

All rows use the README prompt, 128 generated tokens, `temperature=0`, `top_k=0`,
`top_p=1`, `min_p=0`, `repeat_penalty=1`, `-ngl 99`, `-fa 1`, and
`-c 4096` on the same M3 Ultra.

CLI sweep:

| Lane | Prompt tok/s | Generation tok/s | Artefact |
| --- | ---: | ---: | --- |
| Target-only PR CLI | `2063.7` | `83.4` | `docs/runtime/2026-05-18-llamacpp-pr23211-gemma4-26b-a4b-q4-k-m-target-only-cli-p2204-g128.txt` |
| MTP `n_max=1` | `1611.2` | `95.3` | `docs/runtime/2026-05-18-llamacpp-pr23211-gemma4-26b-a4b-q4-k-m-mtp-nmax1-cli-p2204-g128.txt` |
| MTP `n_max=2` | `1615.7` | `100.2` | `docs/runtime/2026-05-18-llamacpp-pr23211-gemma4-26b-a4b-q4-k-m-mtp-nmax2-cli-p2204-g128.txt` |
| MTP `n_max=4` | `1620.2` | `90.7` | `docs/runtime/2026-05-18-llamacpp-pr23211-gemma4-26b-a4b-q4-k-m-mtp-nmax4-cli-p2204-g128.txt` |
| MTP `n_max=8` | `1619.2` | `61.5` | `docs/runtime/2026-05-18-llamacpp-pr23211-gemma4-26b-a4b-q4-k-m-mtp-cli-p2204-g128.txt` |

Server baseline and acceptance metrics:

| Lane | Prompt tok/s | Generation tok/s | Draft tokens | Accepted | Artefact |
| --- | ---: | ---: | ---: | ---: | --- |
| Target-only PR server | `2014.5732742465332` | `83.07814927845328` | n/a | n/a | `docs/runtime/2026-05-18-llamacpp-pr23211-gemma4-26b-a4b-q4-k-m-target-only-server-completion-p2204-g128.json` |
| MTP `n_max=2` PR server | `1562.0125388366318` | `93.76822253543413` | `101` | `75` | `docs/runtime/2026-05-18-llamacpp-pr23211-gemma4-26b-a4b-q4-k-m-mtp-nmax2-server-completion-p2204-g128.json` |

The server log reports:

```text
draft acceptance rate = 0.74257 (75 accepted / 101 generated)
statistics draft-mtp: #calls(b,g,a) = 1 51 51, #gen drafts = 51, #acc drafts = 42, #gen tokens = 101, #acc tokens = 75
```

Read:

- MTP can cross the 100 tok/s visible decode floor in llama.cpp's unmerged PR
  branch when tuned to `n_max=2`.
- It does not improve prefill. In both CLI and server runs, prompt tok/s drops
  because the assistant path adds setup and bookkeeping.
- Large draft windows are harmful here. `n_max=8` regresses generation from the
  target-only CLI's `83.4 tok/s` to `61.5 tok/s`.
- This is not raw target-model parity evidence for go-mlx. It is an R&D target:
  go-mlx needs a package-level target+assistant speculative API and the same
  proposed/accepted/rejected metrics before the lane can count as a production
  visible-throughput mode.

## go-mlx Implementation Shape

Keep this package-first and portable:

1. Add a draft/target speculative generation API without changing the existing
   single-model `Generate` contract for all drivers.
2. Load the target and assistant with a shared tokenizer check, matching chat
   template, and compatible context/settings checks.
3. Prefill target state normally; initialise any required assistant state
   separately and report that cost.
4. Draft up to `K` candidate tokens.
5. Verify the candidate block with the target in one pass.
6. Accept the matching prefix, reject the rest, and update target/assistant
   caches consistently.
7. Emit metrics: proposed tokens, accepted tokens, rejected tokens, acceptance
   rate, target verify passes, effective visible tok/s, target-only baseline
   tok/s, and prefill timings.

Correctness gate for greedy mode: with `temperature=0`, the accepted token
stream must match the target-only greedy stream exactly.

2026-05-18 code progress: go-mlx now exposes a package-first
`Model.GenerateSpeculative` target+draft reference API, plus
`LoadSpeculativePair` for loading a target beside its assistant with vocab and
tokenizer-probe compatibility checks. The fast-eval adapter feeds native token
IDs and text into the shared `dappco.re/go/inference/decode` speculative and
prompt-lookup harness. That makes acceptance metrics real for package callers
and bench reports instead of text-only generation with zero accepted/rejected
token counts.

The CLI benchmark surface can now emit the same reference metrics when the
drafter is a standalone model:

```bash
bin/lthn-mlx bench -json \
  -speculative-draft-model /path/to/gemma-4-26B-A4B-it-assistant \
  -speculative-draft-tokens 2 \
  /path/to/gemma-4-26B-A4B-it
```

The resulting `speculative_decode.metrics` JSON includes proposed draft tokens,
accepted tokens, rejected tokens, acceptance rate, visible-token tok/s,
target-token tok/s, and draft-token tok/s. This is still a reference metrics
path: go-mlx does not yet batch target verification over a drafted block or
report production visible tok/s for native target+assistant MTP.

An attempted real E2B run is captured at:

```text
docs/runtime/2026-05-18-go-mlx-gemma4-e2b-speculative-reference-bench.stderr
```

That run reaches the next concrete blocker:

```text
gemma4_assistant native MTP drafter loading is not implemented yet
```

`gemma4_assistant` is now recognised as a metadata-only architecture instead of
being misloaded as ordinary `gemma4_text`.

Follow-up code progress: `go/internal/metal.LoadGemma4Assistant` now loads and
validates Gemma 4 assistant drafter tensors separately from `InternalModel`.
That loader handles the assistant-specific `backbone_hidden_size`, centroid
metadata, `pre_projection`, `post_projection`, Q/O-only assistant layers, MLP
tensors, and optional ordered-embedding centroid/token-ordering tensors. Focused
verification passed with:

```bash
cd /Users/snider/Code/core/go-mlx/go
env GOCACHE=/private/tmp/codex-go-mlx-cache GOWORK=/Users/snider/Code/core/go-mlx/go.work go test ./internal/metal -run 'TestGemma4Assistant' -count=1
```

The same optional local-pack smoke also passed when
`GO_MLX_GEMMA4_ASSISTANT_MODEL` pointed at the local E2B assistant safetensors
snapshot and when it pointed at the local 26B A4B assistant safetensors
snapshot. That verifies the loader against the real assistant tensor layouts;
it does not yet make the assistant a standalone `InternalModel`.

Follow-up code progress: `go/internal/metal.LoadGemma4AssistantPair` now loads
and validates a Gemma 4 target beside its attached assistant. The attachment
checks the shared backbone hidden size, vocabulary, tokenizer probes, target K/V
stream layer types, and matching attention head dimensions. Focused verification
passed with:

```bash
cd /Users/snider/Code/core/go-mlx/go
env GOCACHE=/private/tmp/codex-go-mlx-cache GOWORK=/Users/snider/Code/core/go-mlx/go.work go test ./internal/metal -run 'TestGemma4Assistant' -count=1
```

Optional local-pack smokes also pass for both real model pairs:

```bash
env GOCACHE=/private/tmp/codex-go-mlx-cache GOWORK=/Users/snider/Code/core/go-mlx/go.work GO_MLX_GEMMA4_TARGET_MODEL=/Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-4bit/snapshots/99d9a53ff828d365a8ecae538e45f80a08d612cd GO_MLX_GEMMA4_ASSISTANT_MODEL=/Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-E2B-it-assistant-bf16/snapshots/a7770799b560135ebdbfae8b7f468947415003bc go test ./internal/metal -run 'TestGemma4Assistant_LoadLocalAssistantPair_Good' -count=1
env GOCACHE=/private/tmp/codex-go-mlx-cache GOWORK=/Users/snider/Code/core/go-mlx/go.work GO_MLX_GEMMA4_TARGET_MODEL=/Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-26B-A4B-it-4bit/snapshots/695690b33533b1f8b0395c1d6b4f00dc411353ef GO_MLX_GEMMA4_ASSISTANT_MODEL=/Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-26B-A4B-it-assistant-bf16/snapshots/cda74908f1dbe7d3dbd3030e66576a7d4094144f go test ./internal/metal -run 'TestGemma4Assistant_LoadLocalAssistantPair_Good' -count=1
```

The root package now uses this attachment path too: `mlx.LoadSpeculativePair`
recognises `gemma4_assistant` draft packs, attaches them to the native Gemma 4
target, and routes `SpeculativePair.Generate` into the native MTP generation loop
when the target runtime implements `GenerateGemma4Assistant`. A mocked root test
covers that routing. The optional root local-pack smoke skips when
`metal.MetalAvailable()` is false because root loading goes through
`metal.LoadAndInit`; the internal attachment smoke above does not claim a
successful root runtime load in that environment.

Follow-up code progress: `go/internal/metal.Gemma4Model` now exposes
`ForwardLastTokenLogitsAndHidden`, so the target can return final-position
logits and the matching pre-output-normalisation hidden state from the same
forward pass. `go/internal/metal.Gemma4AssistantPair.DraftStep` consumes that
target hidden state plus the last token and runs one assistant MTP step against
the target model's populated K/V caches. The step follows the llama.cpp PR
shape: embed the last token through the target embedding table, concatenate it
with the target-backbone hidden state, run the assistant pre-projection plus
Q-only assistant layers over borrowed target K/V streams, then return assistant
logits, the greedy draft token, and the post-projected backbone hidden for a
chained step. `Gemma4AssistantPair.DraftBlock` chains those steps into a
CPU-visible draft token block for the future target verifier. Ordered-embedding
centroid logits still fail closed until that path is implemented.

Follow-up code progress: `Gemma4AssistantPair.VerifyDraftBlock` now performs the
first greedy target-side accept/reject pass over proposed assistant tokens. It
clones the target K/V caches before verification, compares each draft token
against the target argmax at the accepted boundary, returns accepted/rejected
token counts, the target replacement token on mismatch, and the accepted-boundary
cache/logits/hidden state for later generation-loop integration. Rejected tokens
therefore do not pollute the live target cache.

Focused verification passed with:

```bash
cd /Users/snider/Code/core/go-mlx/go
env GOCACHE=/private/tmp/codex-go-mlx-cache GOWORK=/Users/snider/Code/core/go-mlx/go.work go test ./internal/metal -run 'TestGemma4AssistantDecode' -count=1
env GOCACHE=/private/tmp/codex-go-mlx-cache GOWORK=/Users/snider/Code/core/go-mlx/go.work go test ./internal/metal -run 'TestGemma4Assistant' -count=1
env GOCACHE=/private/tmp/codex-go-mlx-cache GOWORK=/Users/snider/Code/core/go-mlx/go.work go test . -run 'TestSpeculative' -count=1
```

The optional E2B real-pack smoke also passed with:

```bash
cd /Users/snider/Code/core/go-mlx/go
env GOCACHE=/private/tmp/codex-go-mlx-cache GOWORK=/Users/snider/Code/core/go-mlx/go.work MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib GO_MLX_GEMMA4_TARGET_MODEL=/Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-4bit/snapshots/99d9a53ff828d365a8ecae538e45f80a08d612cd GO_MLX_GEMMA4_ASSISTANT_MODEL=/Users/snider/.cache/huggingface/hub/models--mlx-community--gemma-4-E2B-it-assistant-bf16/snapshots/a7770799b560135ebdbfae8b7f468947415003bc go test ./internal/metal -run 'TestGemma4AssistantDecode_LoadLocalAssistantPairDraftStep_Good' -count=1
```

That smoke now covers both a real-pack draft step and one accepted greedy target
verification token.

Follow-up code progress: `Model.GenerateGemma4Assistant` now wires the
draft-block and verify-block primitives into a conservative greedy native MTP
generation loop. The loop pre-fills the target, drafts up to `draftTokens`
assistant tokens from the last target hidden state, verifies the proposed block
against cloned target caches, accepts the matching prefix, emits the target
replacement token on mismatch, and keeps the live cache at the accepted boundary.
It records prompt tokens, target/draft calls, proposed/accepted/rejected token
counts, and prefill/target/draft durations. The root
`SpeculativePair.Generate` path converts this native result back into the shared
`go-inference/decode` speculative metrics.

The MTP prefill path now uses hidden-aware prompt preparation. Native MTP prompt
cache entries store the final target hidden state alongside K/V and logits, so
exact repeated project-memory prompts do not have to replay the prefix. KV-only
restored memory entries still avoid replaying the full prefix: the MTP path
restores the cached K/V prefix and replays only the final suffix token required
to recover the target hidden state. Chunked prefill is also honoured for
unavoidable new context through the existing `prefill_chunk_size` setting.
Prompt-cache restore is now fixed-cache aware too, so the request-sized Gemma 4
production cache planner can wake durable K/V into fixed backing buffers instead
of disabling the cache hit and pre-filling the whole prefix again. The rejected
native router top-k probe still demonstrates the fixed-cache restore path:
after the first cold README run, the next two 2204-token prompt setups restored
from cache in about `4.7ms`.

Focused verification passed with:

```bash
cd /Users/snider/Code/core/go-mlx/go
env GOCACHE=/private/tmp/codex-go-mlx-cache GOWORK=/Users/snider/Code/core/go-mlx/go.work go test ./internal/metal -run 'TestGemma4Assistant(Decode|Generate)' -count=1
env GOCACHE=/private/tmp/codex-go-mlx-cache GOWORK=/Users/snider/Code/core/go-mlx/go.work go test . -run 'TestSpeculative' -count=1
```

Real benchmark status:

- E2B target plus `mlx-community/gemma-4-E2B-it-assistant-bf16` reaches the
  native loop but fails closed with `gemma4.assistant ordered embedding logits
  are not implemented yet`. That pack has `use_ordered_embeddings=true`, so it
  still needs the centroid/token-ordering logits path.
- 26B A4B target plus `mlx-community/gemma-4-26B-A4B-it-assistant-bf16`
  completes the native loop after fixing cloned/restored `PagedKVCache`
  `pageLens` handling. `draftTokens=2` records target-only
  `61.42236924451142 tok/s`, native MTP visible `32.207918216043666 tok/s`,
  and `8/24` draft tokens accepted. `draftTokens=1` records target-only
  `60.756648029450965 tok/s`, native MTP visible `34.89669623707289 tok/s`,
  and `6/16` accepted.

Same-short-prompt llama.cpp PR 23211 comparison:

| Lane | Prompt tok/s | Decode tok/s | Draft accepted | Artefact |
| --- | ---: | ---: | ---: | --- |
| llama.cpp target-only CLI | `361.8` | `92.0` | n/a | `docs/runtime/2026-05-18-llamacpp-pr23211-gemma4-26b-a4b-q4-k-m-target-only-cli-shortprompt-g16.txt` |
| llama.cpp MTP `n_max=1` CLI | `327.0` | `103.2` | n/a | `docs/runtime/2026-05-18-llamacpp-pr23211-gemma4-26b-a4b-q4-k-m-mtp-nmax1-cli-shortprompt-g16.txt` |
| llama.cpp MTP `n_max=2` CLI | `326.7` | `118.2` | n/a | `docs/runtime/2026-05-18-llamacpp-pr23211-gemma4-26b-a4b-q4-k-m-mtp-nmax2-cli-shortprompt-g16.txt` |
| llama.cpp target-only server | `229.16507524253308` | `88.79861030174878` | n/a | `docs/runtime/2026-05-18-llamacpp-pr23211-gemma4-26b-a4b-q4-k-m-target-only-server-shortprompt-g16.json` |
| llama.cpp MTP `n_max=2` server | `186.6193897545955` | `100.62260235205333` | `9/12` | `docs/runtime/2026-05-18-llamacpp-pr23211-gemma4-26b-a4b-q4-k-m-mtp-nmax2-server-shortprompt-g16.json` |

The current go-mlx native MTP loop is therefore rejected as the production path.
It is benchmarkable and useful R&D scaffolding, but on the same prompt it is
slower than go-mlx target-only and far behind llama.cpp MTP. The production
parity lane returns to raw target decode and the remaining same-prompt
llama.cpp gap.

## Benchmark Acceptance

Recorded MTP lanes:

| Lane | Required |
| --- | --- |
| go-mlx target-only | recorded |
| go-mlx target + assistant MTP | recorded; rejected for production |
| llama.cpp target-only | recorded |
| llama.cpp target + assistant MTP | recorded |

The expected useful number is effective visible decode tok/s, not prefill
tok/s. For the current 26B A4B work, llama.cpp MTP crosses the `100 tok/s`
visible-throughput floor, but go-mlx MTP does not. Keep the code path, but do
not count it toward production parity until acceptance/verification overhead is
solved.
