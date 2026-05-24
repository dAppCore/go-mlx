<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# go-mlx Upstream TODO

This file is the short upstream request list for making the State `.kv`
container path real instead of a smoke-test packer.

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
`303e835` as `OpenWithSegmentAlias(ctx, path, canonicalSegment)`. The local
dev branch now also adds the read-only embedded-region path
`OpenRegionWithSegmentAlias(ctx, path, payloadOffset, payloadBytes,
canonicalSegment)`.

The current file-backed State store validates `ChunkRef.Segment` against the
opened store path. That is correct for safety, but a `.kv` container extracted
to a temporary path fails because the folded State block refs still point at
the original segment path.

The safe alias/open options are:

```go
func OpenWithSegmentAlias(ctx context.Context, path string, canonicalSegment string) (*Store, error)
func OpenRegionWithSegmentAlias(ctx context.Context, path string, payloadOffset int64, payloadBytes int64, canonicalSegment string) (*Store, error)
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
- The store still writes new refs using the physical path unless an explicit
  write-segment option is also provided.

Current go-mlx bridge: direct `.kv` wake reads the Trix header without touching
the payload, opens the `.kv` file itself as a read-only State region using the
payload offset and byte length, and keeps the original `state_store_path` as the
canonical segment alias. This removes the temporary `.mvlog` materialisation
step while preserving strict segment validation. It is still not the final
zero-copy/mmap shape because the State payload is read through the filestore
region API rather than mapped into pinned MLX-ready memory.

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
`.mvlog` copy. The remaining production step is mmap plus pinned view/MLX array
handoff so the payload window can become a zero-copy State vector source.
