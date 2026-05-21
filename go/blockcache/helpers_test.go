// SPDX-Licence-Identifier: EUPL-1.2

package blockcache

import (
	"context"

	state "dappco.re/go/inference/state"
)

// failingStateWriter is a test stub that always errors on Put. Used to
// exercise the State-write failure path inside blockcache.WarmCache.
type failingStateWriter struct{}

func (failingStateWriter) Put(_ context.Context, _ string, _ state.PutOptions) (state.ChunkRef, error) {
	return state.ChunkRef{}, context.Canceled
}
