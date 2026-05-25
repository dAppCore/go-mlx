// SPDX-Licence-Identifier: EUPL-1.2

// Package memvid keeps the old go-mlx import path as a compatibility shim.
//
// Deprecated: import dappco.re/go/inference/state directly for State stores.
package memvid

import "dappco.re/go/inference/state"

var ErrChunkNotFound = state.ErrChunkNotFound

const (
	CodecMemory  = state.CodecMemory
	CodecQRVideo = state.CodecQRVideo
)

type Store = state.Store
type Resolver = state.Resolver
type URIResolver = state.URIResolver
type Writer = state.Writer
type BinaryResolver = state.BinaryResolver
type RefBinaryResolver = state.RefBinaryResolver
type BinaryWriter = state.BinaryWriter
type BinaryStreamWriter = state.BinaryStreamWriter
type PutOptions = state.PutOptions
type Chunk = state.Chunk
type ChunkRef = state.ChunkRef
type ChunkNotFoundError = state.ChunkNotFoundError
type URIChunkNotFoundError = state.URIChunkNotFoundError
type InMemoryStore = state.InMemoryStore

var NewInMemoryStore = state.NewInMemoryStore
var NewInMemoryStoreWithManifest = state.NewInMemoryStoreWithManifest
var Resolve = state.Resolve
var ResolveBytes = state.ResolveBytes
var ResolveRefBytes = state.ResolveRefBytes
var ResolveURI = state.ResolveURI
var MergeRef = state.MergeRef
