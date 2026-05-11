// SPDX-Licence-Identifier: EUPL-1.2

package blockcache

import (
	"context"

	memvid "dappco.re/go/inference/state"
)

// failingMemvidWriter is a test stub that always errors on Put. Used to
// exercise the memvid-write failure path inside blockcache.WarmCache.
type failingMemvidWriter struct{}

func (failingMemvidWriter) Put(_ context.Context, _ string, _ memvid.PutOptions) (memvid.ChunkRef, error) {
	return memvid.ChunkRef{}, context.Canceled
}
