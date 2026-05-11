// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"

	core "dappco.re/go"
	memvid "dappco.re/go/inference/state"
)

const (
	// KVSnapshotMemvidKind identifies memvid chunks containing go-mlx KV state.
	KVSnapshotMemvidKind = "go-mlx/kv-snapshot"
	// KVSnapshotMemvidVersion is the JSON envelope schema version.
	KVSnapshotMemvidVersion = 1
)

// KVSnapshotMemvidOptions controls how KV snapshots are stored in memvid.
type KVSnapshotMemvidOptions struct {
	KVEncoding KVSnapshotEncoding
	URI        string
	Title      string
	Kind       string
	Track      string
	Tags       map[string]string
	Labels     []string
}

type kvSnapshotMemvidEnvelope struct {
	Version          int    `json:"version"`
	Kind             string `json:"kind"`
	KVVersion        int    `json:"kv_version"`
	KVEncoding       string `json:"kv_encoding,omitempty"`
	BinaryEncoding   string `json:"binary_encoding"`
	KVHash           string `json:"kv_hash"`
	Architecture     string `json:"architecture,omitempty"`
	TokenCount       int    `json:"token_count,omitempty"`
	TokenOffset      int    `json:"token_offset,omitempty"`
	GeneratedTokens  int    `json:"generated_tokens,omitempty"`
	NumLayers        int    `json:"num_layers,omitempty"`
	NumHeads         int    `json:"num_heads,omitempty"`
	SeqLen           int    `json:"seq_len,omitempty"`
	HeadDim          int    `json:"head_dim,omitempty"`
	NumQueryHeads    int    `json:"num_query_heads,omitempty"`
	PayloadByteCount int    `json:"payload_byte_count,omitempty"`
	Data             string `json:"data"`
}

// SaveMemvid writes this KV snapshot to a memvid cold store. The payload is the
// same binary format used by Save, base64 wrapped so text-oriented memvid stores
// and QR-video backends can carry it without lossy conversion.
func (s *KVSnapshot) SaveMemvid(ctx context.Context, store memvid.Writer, opts KVSnapshotMemvidOptions) (memvid.ChunkRef, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if s == nil {
		return memvid.ChunkRef{}, core.NewError("mlx: KV snapshot is nil")
	}
	if store == nil {
		return memvid.ChunkRef{}, core.NewError("mlx: memvid store is nil")
	}
	encoding, err := normalizeKVSnapshotEncoding(opts.KVEncoding)
	if err != nil {
		return memvid.ChunkRef{}, err
	}
	data, err := s.bytesWithOptions(KVSnapshotSaveOptions{KVEncoding: encoding})
	if err != nil {
		return memvid.ChunkRef{}, err
	}
	envelope := kvSnapshotMemvidEnvelope{
		Version:          KVSnapshotMemvidVersion,
		Kind:             KVSnapshotMemvidKind,
		KVVersion:        effectiveKVSnapshotVersion(s, encoding),
		KVEncoding:       string(encoding),
		BinaryEncoding:   "base64",
		KVHash:           core.SHA256Hex(data),
		Architecture:     s.Architecture,
		TokenCount:       len(s.Tokens),
		TokenOffset:      effectiveKVSnapshotTokenOffset(s),
		GeneratedTokens:  len(s.Generated),
		NumLayers:        s.NumLayers,
		NumHeads:         s.NumHeads,
		SeqLen:           s.SeqLen,
		HeadDim:          s.HeadDim,
		NumQueryHeads:    s.NumQueryHeads,
		PayloadByteCount: len(data),
		Data:             core.Base64Encode(data),
	}
	ref, err := store.Put(ctx, core.JSONMarshalString(envelope), kvSnapshotMemvidPutOptions(s, opts, envelope))
	if err != nil {
		return memvid.ChunkRef{}, core.E("KVSnapshot.SaveMemvid", "write memvid chunk", err)
	}
	return ref, nil
}

// LoadKVSnapshotFromMemvid resolves and decodes a KV snapshot from a memvid
// chunk ref.
func LoadKVSnapshotFromMemvid(ctx context.Context, store memvid.Store, ref memvid.ChunkRef) (*KVSnapshot, error) {
	return LoadKVSnapshotFromMemvidWithOptions(ctx, store, ref, KVSnapshotLoadOptions{})
}

// LoadKVSnapshotFromMemvidWithOptions resolves and decodes a KV snapshot from a
// memvid chunk ref with explicit decode options.
func LoadKVSnapshotFromMemvidWithOptions(ctx context.Context, store memvid.Store, ref memvid.ChunkRef, opts KVSnapshotLoadOptions) (*KVSnapshot, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if store == nil {
		return nil, core.NewError("mlx: memvid store is nil")
	}
	chunk, err := memvid.Resolve(ctx, store, ref.ChunkID)
	if err != nil {
		return nil, core.E("LoadKVSnapshotFromMemvid", "resolve memvid chunk", err)
	}
	var envelope kvSnapshotMemvidEnvelope
	if result := core.JSONUnmarshalString(chunk.Text, &envelope); !result.OK {
		return nil, core.E("LoadKVSnapshotFromMemvid", "parse memvid envelope", kvSnapshotResultError(result))
	}
	data, err := decodeKVSnapshotMemvidEnvelope(envelope)
	if err != nil {
		return nil, err
	}
	return parseKVSnapshotWithOptions(data, opts)
}

func decodeKVSnapshotMemvidEnvelope(envelope kvSnapshotMemvidEnvelope) ([]byte, error) {
	if envelope.Version <= 0 || envelope.Version > KVSnapshotMemvidVersion {
		return nil, core.NewError("mlx: unsupported memvid KV snapshot version")
	}
	if envelope.Kind != KVSnapshotMemvidKind {
		return nil, core.NewError("mlx: invalid memvid KV snapshot kind")
	}
	if envelope.BinaryEncoding != "base64" {
		return nil, core.NewError("mlx: unsupported memvid KV snapshot binary encoding")
	}
	decoded := core.Base64Decode(envelope.Data)
	if !decoded.OK {
		return nil, core.E("LoadKVSnapshotFromMemvid", "decode memvid KV payload", kvSnapshotResultError(decoded))
	}
	data, ok := decoded.Value.([]byte)
	if !ok {
		return nil, core.NewError("mlx: memvid KV payload decoded to non-byte data")
	}
	if envelope.PayloadByteCount > 0 && len(data) != envelope.PayloadByteCount {
		return nil, core.NewError("mlx: memvid KV payload length mismatch")
	}
	if envelope.KVHash != "" && core.SHA256Hex(data) != envelope.KVHash {
		return nil, core.NewError("mlx: memvid KV snapshot hash mismatch")
	}
	return data, nil
}

func kvSnapshotMemvidPutOptions(snapshot *KVSnapshot, opts KVSnapshotMemvidOptions, envelope kvSnapshotMemvidEnvelope) memvid.PutOptions {
	kind := opts.Kind
	if kind == "" {
		kind = KVSnapshotMemvidKind
	}
	track := opts.Track
	if track == "" {
		track = "session-kv"
	}
	tags := cloneKVSnapshotMemvidTags(opts.Tags)
	tags["kv_hash"] = envelope.KVHash
	tags["kv_encoding"] = envelope.KVEncoding
	tags["architecture"] = envelope.Architecture
	tags["token_count"] = core.Itoa(envelope.TokenCount)
	tags["payload_bytes"] = core.Itoa(envelope.PayloadByteCount)
	labels := append([]string(nil), opts.Labels...)
	labels = append(labels, "go-mlx", "kv-snapshot")
	return memvid.PutOptions{
		URI:    firstNonEmptyString(opts.URI, "mlx://kv-snapshot/"+envelope.KVHash),
		Title:  firstNonEmptyString(opts.Title, "go-mlx KV snapshot"),
		Kind:   kind,
		Track:  track,
		Tags:   tags,
		Labels: labels,
	}
}

func cloneKVSnapshotMemvidTags(input map[string]string) map[string]string {
	out := map[string]string{}
	for key, value := range input {
		out[key] = value
	}
	return out
}

func effectiveKVSnapshotVersion(snapshot *KVSnapshot, encoding KVSnapshotEncoding) int {
	version := snapshot.Version
	if version == 0 {
		version = KVSnapshotVersion
	}
	if encoding != KVSnapshotEncodingFloat32 && version < 3 {
		version = 3
	}
	return version
}

func effectiveKVSnapshotTokenOffset(snapshot *KVSnapshot) int {
	if snapshot == nil {
		return 0
	}
	if snapshot.TokenOffset != 0 {
		return snapshot.TokenOffset
	}
	return len(snapshot.Tokens)
}
