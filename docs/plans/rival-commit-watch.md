<!-- SPDX-Licence-Identifier: EUPL-1.2 -->
# Rival Inference-Engine Commit Watch

Daily digest of what shipped in rival open-source inference engines, filtered through the
go-mlx lens (temporally-aware, CONT/no-replay retained-state engine; KV/state persists and is
mounted via Wake/Sleep, not re-prefilled). Newest entry at the top.

Repos tracked: `ml-explore/mlx`, `ml-explore/mlx-lm`, `Blaizzy/mlx-vlm`,
`lmstudio-ai/mlx-engine`, `ggml-org/llama.cpp`, `vllm-project/vllm`.

---

## 2026-06-06 — window ~2026-06-04 22:09 → 2026-06-06 00:09 UTC (last ~26h)

> ⚠️ **Degraded run — Atom feeds could not be loaded.** The GitHub commit/release/tag Atom
> feeds were unreachable this run, so the per-commit detail below is **not** feed-derived.
> See "Why the feeds failed" and "Action required" at the foot of this entry. Nothing below
> should be treated as a verified commit list, and no commit hashes/PR numbers have been
> invented to fill the gap.

### ⭐ Worth a look for go-mlx

Cannot be compiled reliably this run — the feed pipeline that produces per-commit, in-window
items did not function (see below). Treating this as **"no verified actionable items"** rather
than risk surfacing fabricated or stale highlights.

The only low-confidence, search-derived hint worth flagging: `llama.cpp` cut at least one
tagged build on **5 Jun 2026** (its cadence is ~one release every few hours), so anything that
landed there — quant/k-quant, sampling, or Metal kernel work — would be the most likely place
to find something in-window. Needs the feed to confirm specifics. (KV/state, quant, Metal —
unverified.)

### Per repo

**ml-explore/mlx** — feed unavailable (fetch blocked). Verified out-of-band from the repo
landing page: latest *release* is **v0.31.2, dated 22 Apr 2026** — well outside the window, so
**no release in window**. Commit-level activity in window: unknown (feed required).

**ml-explore/mlx-lm** — feed unavailable. Search signal only: repo last updated ~**2 Jun 2026**
(outside the 26h window); PyPI still at **0.31.3 (22 Apr 2026)**. A recurring theme in recent
mlx-lm work is batch KV behaviour (e.g. defaulting to `BatchRotatingKVCache` in batch mode) —
relevant to go-mlx's KV/state surface — but **not confirmed in this window**. — quiet / unverified.

**Blaizzy/mlx-vlm** — feed unavailable. No reliable in-window signal. — unverified.

**lmstudio-ai/mlx-engine** — feed unavailable. No reliable in-window signal. — unverified.

**ggml-org/llama.cpp** — feed unavailable. Search signal only: at least one tagged build on
**5 Jun 2026** (within window); project releases roughly every few hours, so multiple commits
almost certainly landed in window. Specific titles/hashes/PRs **not verified** (feed required).
Likely-relevant areas to check once feeds work: GGUF/k-quant/imatrix, sampling, Metal kernels.

**vllm-project/vllm** — feed unavailable. Search returned inconsistent version data; no reliable
in-window signal. — unverified.

### Honest gaps

- **All six commit/release/tag Atom feeds: unavailable this run.** Not a GitHub outage — a
  sandbox constraint (below).
- Per-commit detail, exact timestamps, and short hashes/PR numbers are therefore **absent by
  design** (not fabricated).
- Release facts marked "verified" come from a successful fetch of the repo landing page; items
  marked "search signal" are fuzzy and may be stale.

### Why the feeds failed

The run is restricted to the `web_fetch` tool, which enforces a **URL-provenance allowlist**: it
will only retrieve a URL that has already appeared verbatim in the task/user message or in a
prior fetch result. The task file supplies the feed URLs as *templates*
(`https://github.com/<owner>/<repo>/commits.atom`), so the **literal** feed URLs (with real
owner/repo) never entered the allowlist, and every `*.atom` fetch returned
*"URL not in provenance set."* GitHub's Atom feed URLs are not surfaced by web search result
links or inside fetched HTML bodies (the `<link rel="alternate">` tags are stripped), so there
is no in-policy way to get them into provenance. The task forbids substituting another fetch
method (curl/wget/python/browser), so per its own fallback rule the feeds are reported as
unavailable rather than worked around.

### Action required (one-line fix for tomorrow's run)

List the **18 literal feed URLs** explicitly in the scheduled-task SKILL.md body (not as
`<owner>/<repo>` templates). Once the exact URLs appear in the task message they enter the
`web_fetch` provenance allowlist and the feed pipeline works unchanged. The URLs to hard-code:

```
https://github.com/ml-explore/mlx/commits.atom
https://github.com/ml-explore/mlx/releases.atom
https://github.com/ml-explore/mlx/tags.atom
https://github.com/ml-explore/mlx-lm/commits.atom
https://github.com/ml-explore/mlx-lm/releases.atom
https://github.com/ml-explore/mlx-lm/tags.atom
https://github.com/Blaizzy/mlx-vlm/commits.atom
https://github.com/Blaizzy/mlx-vlm/releases.atom
https://github.com/Blaizzy/mlx-vlm/tags.atom
https://github.com/lmstudio-ai/mlx-engine/commits.atom
https://github.com/lmstudio-ai/mlx-engine/releases.atom
https://github.com/lmstudio-ai/mlx-engine/tags.atom
https://github.com/ggml-org/llama.cpp/commits.atom
https://github.com/ggml-org/llama.cpp/releases.atom
https://github.com/ggml-org/llama.cpp/tags.atom
https://github.com/vllm-project/vllm/commits.atom
https://github.com/vllm-project/vllm/releases.atom
https://github.com/vllm-project/vllm/tags.atom
```

(Alternative, if you'd rather not bloat the task file: allow the run to fetch via the rendered
GitHub pages with the Claude-in-Chrome browser tool — but that contradicts the current
"web_fetch only / never substitute" rule, so the URL-listing fix above is the clean one.)
