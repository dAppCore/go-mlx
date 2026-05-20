<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# Local Discovery And Autotune

`go-mlx` exposes a metadata-first setup path for UIs that want to help people
pick local model settings without making them understand context windows, cache
modes, batch sizes, or allocator limits.

The flow is deliberately opt-in:

1. Call `DiscoverLocalRuntime` to show what this machine/backend can do.
2. Call `PlanLocalTuning` for a model/workload to get a small candidate set.
3. If the user asks for help, call `RunLocalTuning` and stream each candidate
   result into the UI.
4. Persist the winning `inference.TuningProfile`.
5. On reload, apply `TuningCandidateLoadOptions(profile.Candidate)` and use
   `inference.PlanModelReplace` to decide whether state can be reused,
   checkpointed, or compacted into a summary/new window.

The discovery path does not load weights. It reads device facts, runtime
capabilities, cache modes, and optional model-pack metadata. The expensive part
is only the user's explicit tuning run.

Architectures with metadata support but no native decode kernels are planned
onto a fallback backend instead of pretending the Metal loader can run them. In
practice this means Qwen 3.6 (`qwen3_6` / `qwen3_6_moe`) candidates use
`mlx_lm` while the native hybrid linear-attention path is still pending.

```go
report, err := mlx.DiscoverLocalRuntime(ctx, mlx.LocalDiscoveryConfig{
	ModelDirs:         []string{"/Users/me/models"},
	IncludeModels:     true,
	IncludeCandidates: true,
})
```

`RunLocalTuning` loads and closes one candidate at a time. It emits
`TuningEventCandidate` before each load and `TuningEventResult` after the smoke
bench finishes or fails, so a UI can keep updating without waiting for the whole
run.

```go
results, err := mlx.RunLocalTuning(ctx, mlx.LocalTuningRunConfig{
	ModelPath:  "/Users/me/models/qwen3",
	Workload:   inference.TuningWorkloadAgentState,
	Candidates: plan.Candidates,
	Emit: func(event inference.TuningEvent) bool {
		// update UI progress; return false to stop early
		return true
	},
})
```

Workloads are stable strings: `chat`, `coding`, `long_context`, `agent_state`,
`throughput`, and `low_latency`. Scores are transparent heuristics over measured
smoke counters, not a universal benchmark. For agent workflows the score weights
prompt-cache hit rate and KV/state restore latency because waking useful context
quickly matters more than peak single-turn decode speed.

## CLI Profile Reload

The CLI keeps the same profile shape as the package API. A setup run can persist
the selected profile:

```bash
lthn-mlx tune-run -jsonl -workload agent_state -profile-output profiles/agent-state.json /models/qwen3
```

The persisted JSON can then be inspected without loading the model:

```bash
lthn-mlx tune-profile -json profiles/agent-state.json
```

Saved profiles include the winning candidate's raw measurements, workload score,
and selection labels such as `selection_policy`, `selected_score`,
`selected_load_milliseconds`, `selected_first_token_milliseconds`,
`selected_restore_milliseconds`, `selected_decode_tokens_per_sec`,
`selected_peak_memory_bytes`, `selected_correctness_smoke_result`,
`successful_candidates`, and `selection_score_delta`. This keeps a slower
profile from being hidden behind a generic successful run: the profile records
the measured reason it won in terms a setup UI can show directly.

`driver-profile` can reload through that saved profile without repeating the
tuning search. The profile supplies the model path and candidate load settings;
explicit command flags such as `-context` and `-device` remain final overrides.

```bash
lthn-mlx driver-profile -json -profile profiles/agent-state.json -prompt "Why does retained state matter?" -max-tokens 128 -runs 3
```

When the UI wants to test another local model or cache profile, it can compare
the current saved profile against the candidate profile without loading either
model:

```bash
lthn-mlx replace-plan -json -current-profile profiles/current.json -next-profile profiles/candidate.json
```

The JSON response includes the backend-neutral `ModelReplaceRequest` plus a
conservative `ModelReplacePlan`: reuse state when model/runtime/adapter match,
checkpoint exact state when only runtime or cache settings changed, or fall back
to summary-plus-new-window when model or adapter identity changes.
