<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# Agentic Project Seed Workflow

go-mlx is the Metal implementation of the portable `go-inference/state`
contracts. The wider LTHN stack should treat the state file as a project
context seed: a durable live-prefix object that can be woken, extended, forked,
or compacted without replaying every prompt into the model.

## Roles

| Layer | Responsibility |
|-------|----------------|
| `go-inference/state` | Backend-neutral DTOs and interfaces: `WakeRequest`, `SleepRequest`, `Session`, `Forker`, `Store`, and file/URI refs. |
| go-mlx | Reference Metal runtime that restores KV blocks into a live session and sleeps the current session back to a store. |
| go-ai / go-ml / LTHN app | Orchestration policy: which project seed to wake, which findings become memory, when to save state, and when to use a text summary instead. |

## Project seed

A project seed is a slept model state containing stable context for one working
area. It is usually built from:

- Project identity: repo path, module names, active docs, current branch posture.
- Operator context: preferences, collaboration style, and durable constraints.
- System context: tool limits, build/test lanes, available runtime settings.
- Project memory: recent decisions, findings, benchmarks, and rejected paths.
- A short active task frame, if the seed is being created for a known next task.

The seed should be addressed by URI, not by filesystem convention alone, for
example `state://lthn/projects/go-mlx/seed`. The store can be an append-only
file log, memvid, object storage, or an in-memory test store.

The shared helper is `state.NewProjectSeed`:

```go
seed := state.NewProjectSeed(state.ProjectSeedOptions{
    BaseURI:   "state://lthn/projects",
    ProjectID: "core/go-mlx",
})
```

## Fast task path

1. Load the model with the requested runtime settings.
2. Open the selected state store.
3. Build a `WakeRequest` with `seed.WakeRequest(...)`.
4. Call `ForkState` or `WakeState` with the project seed index and entry URI.
5. Append the current task and fresh repo observations.
6. Run the agent loop.
7. Persist the result with one of the sleep modes below.

This avoids a large prefill at the start of every agent turn. When
`ReuseParentPrefix` is enabled, a child state writes only the changed suffix
while retaining parent links for the shared prefix.

## Sleep modes

| Mode | Use when | Behaviour |
|------|----------|-----------|
| State checkpoint | The operator wants the exact live context to continue later. | Call `SleepState` with a new entry URI and `ReuseParentPrefix=true`. |
| Reuse current seed | The operator wants findings available but not a new KV branch. | Write findings to project memory, then keep the current seed as the next wake target. |
| Summary window | Settings/model identity changed or the operator does not want durable KV state. | Summarise the task state as text and start a new window from the summary plus the project seed material. |
| Hybrid | Research or long-running workflow where portability matters. | Save both a state checkpoint and a text summary; the summary is the fallback if the KV state becomes incompatible. |

## Reload with new settings

Reload is a compatibility decision, not a blind restore:

- Safe to wake: same tokenizer identity, compatible model identity, compatible
  adapter identity, and a runtime that can restore the stored KV encoding.
- Usually safe: sampler changes, max-token limits, scheduling policy, and probe
  settings that do not change the prefix tokens.
- Do not wake blindly: tokenizer changes, model architecture/layer mismatch,
  adapter mismatch, incompatible quantisation/cache encoding, or a context
  length smaller than the saved prefix.

When compatibility is unclear, prefer the hybrid path: write a summary, open a
new session, and only use `SkipCompatibilityCheck` for explicit research runs.
The reusable check is `state.CheckWakeCompatibility(bundle, req)`.

## No-reply workflow

An agent does not always need to answer the operator. For background work,
append observations and sleep the state:

1. Wake the project seed.
2. Append inspected files, command results, and decisions.
3. Call `AppendAndSleep` or `SleepState`.
4. Store the returned `Ref` as the next task's candidate parent.

This turns "reply" into an optional UI event. The useful output is the updated
state and memory index.

## LTHN bundle binary

The LTHN app/CLI/server bundle should ship the same `cmd/mlx` command built as
`lthn-mlx`. The Taskfile target is:

```bash
task build:lthn
```

For the app bundle, use:

```bash
task build:bundle
```

That produces `bin/lthn-mlx` and the Violet sidecar in `bin/violet`.
