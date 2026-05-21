<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# state_bundle.go — Bundle envelope encode/decode

**Package**: `dappco.re/go/mlx`
**File**: `go/state_bundle.go`

## What this is

The **JSON-shaped envelope** that wraps a KV snapshot + its metadata into one portable artefact: model identity, tokenizer identity, sampler config, prompt hash, list of state refs (State video / file / inline), runtime identity. Implements the encode/decode for `inference/state.Bundle`.

A bundle is the unit a user thinks about (`"the Aurelius Meditations book-state"`); a snapshot is the bytes that bundle points at.

## Constants

```go
StateBundleVersion   = 1
StateBundleKind      = "go-mlx/state-bundle"
StateBundleRefState = "State"
```

`StateBundleKind` distinguishes our bundles from other future kinds (e.g. an LLAVA vision-context bundle would be `go-mlx/vision-bundle`). `Kind` lets a generic Store iterate all bundles and route based on type.

## What's inside

The `inference/state.Bundle` shape (re-exported from go-inference) carries:

- Schema version + creation timestamp
- `ModelIdentity` / `TokenizerIdentity` / `AdapterIdentity` / `SamplerConfig` / `RuntimeIdentity`
- `PromptHash`, prompt token count, generated token count
- `KVRefs []StateRef` (where the KV blocks live)
- `ProbeRefs []StateRef` (where probe-event traces live, if captured)
- `StateRefs []StateRef` (where bundled knowledge-pack content lives)
- Labels + Metadata maps

## Encode

```go
data, err := encodeStateBundle(bundle)         // → JSON bytes
chunkRef, err := store.PutBytes(ctx, data, opts) // → durable ref
```

JSON encoding (not protobuf, not msgpack) because:

- Bundles are infrequent (one per sleep, not per token).
- Hand-editable bundles ship in fixtures.
- Cross-tool readable (Python, Rust, browser inspector) without code-gen.

The bundle is small (KBs) so binary efficiency doesn't matter; readability does.

## Decode

```go
bundle, err := decodeStateBundle(jsonBytes)
```

Strict schema check: rejects unknown bundle kinds, unknown schema versions, missing required fields. A future v2 bundle is rejected by a v1 reader — explicit failure beats silent corruption.

## Tokenizer handoff

```go
type StateBundleTokenizer interface {
    EncodePrompt(string) ([]int32, error)
    TokenizerHash() string
}
```

A wake needs the same tokenizer the sleep used. The bundle records `TokenizerIdentity.Hash`; the wake side provides a live tokenizer that satisfies this interface. Hash mismatch → wake refuses.

This is the cleanest split — the bundle doesn't *embed* the tokenizer (would balloon the bundle and create version coupling), it just records enough identity for the wake side to confirm a match.

## Why "Bundle" vs "Snapshot"

- **Bundle** = JSON envelope + references = the portable artefact.
- **Snapshot** = the binary KV bytes a bundle's `KVRefs` point at.

A bundle can reference multiple snapshots (multi-prompt journey persisted as ordered KV slices). A snapshot is one contiguous KV span.

## Related

- [agent_memory.md](agent_memory.md) — Wake/Sleep produces/consumes bundles
- [kv_snapshot.md](kv_snapshot.md) — the snapshot referenced by bundles
- [kv_snapshot_index.md](kv_snapshot_index.md) — index across many bundles
- `../../../go-inference/docs/state/identity.md` — Bundle DTO definition
