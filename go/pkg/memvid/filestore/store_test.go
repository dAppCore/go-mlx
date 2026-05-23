// SPDX-Licence-Identifier: EUPL-1.2

package filestore

import (
	"bytes"
	"context"
	"strconv"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/memvid"
)

func TestCompatibilityFileStore_RoundTrip_Good(t *testing.T) {
	ctx := context.Background()
	path := core.PathJoin(t.TempDir(), "compat-state.bin")
	store, err := Create(ctx, path)
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	ref, err := store.Put(ctx, "payload", memvid.PutOptions{URI: "mlx://compat/1"})
	if err != nil {
		t.Fatalf("Put() error = %v", err)
	}
	if err := store.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}

	reopened, err := Open(ctx, path)
	if err != nil {
		t.Fatalf("Open() error = %v", err)
	}
	defer reopened.Close()

	chunk, err := memvid.Resolve(ctx, reopened, ref.ChunkID)
	if err != nil {
		t.Fatalf("Resolve() error = %v", err)
	}
	if chunk.Text != "payload" || chunk.Ref.Codec != CodecFile {
		t.Fatalf("Resolve() = %+v, want compatibility file chunk", chunk)
	}
}

// TestCompatibilityFileStore_BinaryRoundTrip_Good — bit-exact binary
// round-trip across multiple chunk sizes. The golden-path use case is
// KV cache bytes: encode → close → reopen → ResolveBytes must yield
// the original bytes byte-for-byte. This guards the State container
// contract that's load-bearing for the inference KV save/load path.
func TestCompatibilityFileStore_BinaryRoundTrip_Good(t *testing.T) {
	ctx := context.Background()
	path := core.PathJoin(t.TempDir(), "compat-binary.bin")
	store, err := Create(ctx, path)
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}

	// Cover three size classes: small (header-only), medium (single
	// page), large (multi-page) — exercises the encode/decode boundary
	// across the typical KV cache size range.
	sizes := []int{64, 4096, 64 * 1024}
	payloads := make([][]byte, len(sizes))
	refs := make([]memvid.ChunkRef, len(sizes))
	for i, size := range sizes {
		payload := make([]byte, size)
		for j := range payload {
			payload[j] = byte((j * 31) ^ size) // deterministic non-trivial pattern
		}
		payloads[i] = payload
		ref, err := store.PutBytes(ctx, payload, memvid.PutOptions{URI: "mlx://kv/" + strconv.Itoa(size)})
		if err != nil {
			t.Fatalf("PutBytes(size=%d) error = %v", size, err)
		}
		refs[i] = ref
	}
	if err := store.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}

	reopened, err := Open(ctx, path)
	if err != nil {
		t.Fatalf("Open() error = %v", err)
	}
	defer reopened.Close()

	// Bit-exact parity for every payload; order does not matter (each
	// indexed by chunk ID returned by Put).
	for i, ref := range refs {
		chunk, err := memvid.ResolveBytes(ctx, reopened, ref.ChunkID)
		if err != nil {
			t.Fatalf("ResolveBytes(chunk %d) error = %v", ref.ChunkID, err)
		}
		if !bytes.Equal(chunk.Data, payloads[i]) {
			t.Fatalf("ResolveBytes(chunk %d, size=%d) NOT bit-exact: got %d bytes, want %d bytes",
				ref.ChunkID, sizes[i], len(chunk.Data), len(payloads[i]))
		}
	}
}

// BenchmarkCompatibilityFileStore_TextRoundTrip — encode-and-resolve
// in the same store. Establishes a baseline for the Put+Resolve fused
// hot path that consumers driving a State container hit per chunk.
func BenchmarkCompatibilityFileStore_TextRoundTrip(b *testing.B) {
	ctx := context.Background()
	path := core.PathJoin(b.TempDir(), "compat-bench.bin")
	store, err := Create(ctx, path)
	if err != nil {
		b.Fatalf("Create() error = %v", err)
	}
	defer store.Close()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ref, err := store.Put(ctx, "payload bytes for round trip", memvid.PutOptions{})
		if err != nil {
			b.Fatalf("Put() error = %v", err)
		}
		chunk, err := memvid.Resolve(ctx, store, ref.ChunkID)
		if err != nil {
			b.Fatalf("Resolve() error = %v", err)
		}
		if chunk.Text == "" {
			b.Fatalf("Resolve() returned empty text")
		}
	}
}

// BenchmarkCompatibilityFileStore_BinaryResolve — pre-populated store;
// the bench loop ONLY does Resolve. Tracks the random-access cost (the
// "load by chunk_id" path Snider's KV state load hits).
func BenchmarkCompatibilityFileStore_BinaryResolve(b *testing.B) {
	ctx := context.Background()
	path := core.PathJoin(b.TempDir(), "compat-resolve.bin")
	store, err := Create(ctx, path)
	if err != nil {
		b.Fatalf("Create() error = %v", err)
	}
	defer store.Close()

	payload := make([]byte, 4096)
	for i := range payload {
		payload[i] = byte(i & 0xff)
	}
	ref, err := store.PutBytes(ctx, payload, memvid.PutOptions{})
	if err != nil {
		b.Fatalf("PutBytes() error = %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		chunk, err := memvid.ResolveBytes(ctx, store, ref.ChunkID)
		if err != nil {
			b.Fatalf("ResolveBytes() error = %v", err)
		}
		if len(chunk.Data) != 4096 {
			b.Fatalf("ResolveBytes() len=%d, want 4096", len(chunk.Data))
		}
	}
}
