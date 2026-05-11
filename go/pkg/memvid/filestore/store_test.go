// SPDX-Licence-Identifier: EUPL-1.2

package filestore

import (
	"context"
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
