// SPDX-Licence-Identifier: EUPL-1.2

// Package filestore keeps the old go-mlx import path as a compatibility shim.
// New code should import dappco.re/go/inference/state/filestore directly.
package filestore

import (
	"context"

	statefile "dappco.re/go/inference/state/filestore"
)

const CodecFile = statefile.CodecFile

type Store = statefile.Store

func Create(ctx context.Context, path string) (*Store, error) {
	return statefile.Create(ctx, path)
}

func Open(ctx context.Context, path string) (*Store, error) {
	return statefile.Open(ctx, path)
}
