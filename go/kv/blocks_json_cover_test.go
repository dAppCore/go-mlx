// SPDX-Licence-Identifier: EUPL-1.2

package kv

import (
	"context"
	"testing"

	core "dappco.re/go"
	state "dappco.re/go/inference/state"
)

// putJSONBlock stores a JSON-base64 block envelope wrapping payload and returns
// a non-raw StateBlockRef pointing at it. The envelope's BlockIndex /
// TokenStart / TokenCount are taken from the envelope arg so tests can craft a
// stored block whose metadata disagrees with the bundle ref.
func putJSONBlock(t *testing.T, store *state.InMemoryStore, uri string, envelope kvSnapshotStateBlockEnvelope, payload []byte) state.ChunkRef {
	t.Helper()
	envelope.Version = StateBlockVersion
	envelope.Kind = KVSnapshotStateBlockKind
	envelope.BinaryEncoding = "base64"
	envelope.PayloadByteCount = len(payload)
	envelope.KVHash = core.SHA256Hex(payload)
	envelope.Data = core.Base64Encode(payload)
	ref, err := store.Put(context.Background(), core.JSONMarshalString(envelope), state.PutOptions{URI: uri})
	if err != nil {
		t.Fatalf("Put(json block) error = %v", err)
	}
	return ref
}

// twoTokenBlockPayload marshals a 2-token sub-snapshot whose tokens match the
// supplied IDs — the payload a single State block carries.
func twoTokenBlockPayload(t *testing.T, a, b int32) []byte {
	t.Helper()
	s := &Snapshot{
		Version:       SnapshotVersion,
		Architecture:  "gemma4_text",
		Tokens:        []int32{a, b},
		TokenOffset:   2,
		NumLayers:     1,
		NumHeads:      1,
		SeqLen:        2,
		HeadDim:       2,
		NumQueryHeads: 1,
		Layers: []LayerSnapshot{{Layer: 0, CacheIndex: 0, Heads: []HeadSnapshot{{
			Key:   []float32{1, 2, 3, 4},
			Value: []float32{5, 6, 7, 8},
		}}}},
	}
	data, err := s.MarshalBinary()
	if err != nil {
		t.Fatalf("MarshalBinary() error = %v", err)
	}
	return data
}

// TestBlocksJSONCover_BlockMetadataMismatch drives the block-metadata mismatch
// guard of loadAndAssembleStateBlocks: a one-block bundle whose ref is in order
// but whose stored JSON envelope carries a divergent BlockIndex.
func TestBlocksJSONCover_BlockMetadataMismatch(t *testing.T) {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)
	payload := twoTokenBlockPayload(t, 1, 2)

	// Stored envelope claims BlockIndex 9 though the ref says index 0.
	chunk := putJSONBlock(t, store, "mlx://mismatch-block", kvSnapshotStateBlockEnvelope{
		BlockIndex: 9, TokenStart: 0, TokenCount: 2,
	}, payload)

	bundle := &StateBlockBundle{
		Version:    StateBlockVersion,
		Kind:       StateBlockBundleKind,
		TokenCount: 2,
		Blocks: []StateBlockRef{{
			Index: 0, TokenStart: 0, TokenCount: 2,
			PayloadEncoding: "", // force the JSON envelope load path
			State:           chunk,
		}},
	}
	if _, err := LoadFromStateBlocks(ctx, store, bundle); err == nil {
		t.Fatal("LoadFromStateBlocks(metadata mismatch) error = nil, want metadata error")
	}

	// The prefix assembler shares the same per-block metadata check.
	if _, err := LoadPrefixFromStateBlocks(ctx, store, bundle, 1); err == nil {
		t.Fatal("LoadPrefixFromStateBlocks(metadata mismatch) error = nil, want metadata error")
	}
}

// TestBlocksJSONCover_BlockTokenCountMismatch drives the token-count mismatch
// guard: the stored payload carries two tokens but the ref claims one.
func TestBlocksJSONCover_BlockTokenCountMismatch(t *testing.T) {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)
	payload := twoTokenBlockPayload(t, 1, 2)

	chunk := putJSONBlock(t, store, "mlx://count-block", kvSnapshotStateBlockEnvelope{
		BlockIndex: 0, TokenStart: 0, TokenCount: 1,
	}, payload)

	bundle := &StateBlockBundle{
		Version:    StateBlockVersion,
		Kind:       StateBlockBundleKind,
		TokenCount: 1,
		Blocks: []StateBlockRef{{
			Index: 0, TokenStart: 0, TokenCount: 1,
			PayloadEncoding: "",
			State:           chunk,
		}},
	}
	if _, err := LoadFromStateBlocks(ctx, store, bundle); err == nil {
		t.Fatal("LoadFromStateBlocks(token count mismatch) error = nil, want count error")
	}
}

// TestBlocksJSONCover_PayloadParseError drives the snapshot-parse error arm of
// the JSON block load path (both the full and token-only loaders): the envelope
// decodes to a truncated, unparseable payload.
func TestBlocksJSONCover_PayloadParseError(t *testing.T) {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)

	// A payload that is valid base64 but a truncated snapshot (just the magic).
	truncated := []byte(kvSnapshotMagic)
	chunk := putJSONBlock(t, store, "mlx://parse-error-block", kvSnapshotStateBlockEnvelope{
		BlockIndex: 0, TokenStart: 0, TokenCount: 2,
	}, truncated)

	ref := StateBlockRef{Index: 0, TokenStart: 0, TokenCount: 2, PayloadEncoding: "", State: chunk}
	if _, err := LoadStateBlockWithOptions(ctx, store, ref, LoadOptions{}); err == nil {
		t.Fatal("LoadStateBlockWithOptions(unparseable payload) error = nil, want parse error")
	}
	if _, err := LoadStateBlockTokensWithOptions(ctx, store, ref, LoadOptions{}); err == nil {
		t.Fatal("LoadStateBlockTokensWithOptions(unparseable payload) error = nil, want parse error")
	}
}

// TestBlocksJSONCover_TokenPrefix_RawParseError drives the raw-path token parse
// error of LoadPrefixTokensFromStateBlocks: a raw block whose stored payload is
// a header-only snapshot that declares more tokens than it carries.
func TestBlocksJSONCover_TokenPrefix_RawParseError(t *testing.T) {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)

	// A snapshot header that declares two tokens (8 token bytes) but whose data
	// is cut partway through the token region: magic(8) + version(4) +
	// archLen(4) + "gemma4_text"(11) + 5×u32(20) + tokenOffset(4) +
	// tokenCount(4) = 55, tokens occupy 55..63. Cut at 57 so the declared
	// token read overruns → parseKVSnapshotTokensInto fails.
	header := twoTokenBlockPayload(t, 1, 2)
	if len(header) < 57 {
		t.Fatalf("payload too short (%d) to truncate mid-token", len(header))
	}
	truncated := header[:57]

	chunk, err := store.PutBytes(ctx, truncated, state.PutOptions{URI: "mlx://token-raw-parse"})
	if err != nil {
		t.Fatalf("PutBytes error = %v", err)
	}
	bundle := &StateBlockBundle{
		Version:    StateBlockVersion,
		Kind:       StateBlockBundleKind,
		TokenCount: 2,
		Blocks: []StateBlockRef{{
			Index: 0, TokenStart: 0, TokenCount: 2,
			PayloadEncoding:  kvSnapshotStatePayloadRaw,
			PayloadByteCount: len(truncated),
			State:            chunk,
		}},
	}
	if _, err := LoadPrefixTokensFromStateBlocks(ctx, store, bundle, 2); err == nil {
		t.Fatal("LoadPrefixTokensFromStateBlocks(raw parse error) error = nil, want parse error")
	}
}

// TestBlocksJSONCover_TokenPrefix_JSONErrors drives the JSON-path arms of
// LoadPrefixTokensFromStateBlocks: a load failure (missing chunk) and a token
// count that disagrees with the ref.
func TestBlocksJSONCover_TokenPrefix_JSONErrors(t *testing.T) {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)
	payload := twoTokenBlockPayload(t, 1, 2)

	// Token-count mismatch: the stored JSON envelope carries two tokens but the
	// ref claims three → blockTokenCount != ref.TokenCount.
	chunk := putJSONBlock(t, store, "mlx://token-count-block", kvSnapshotStateBlockEnvelope{
		BlockIndex: 0, TokenStart: 0, TokenCount: 3,
	}, payload)
	countBundle := &StateBlockBundle{
		Version:    StateBlockVersion,
		Kind:       StateBlockBundleKind,
		TokenCount: 3,
		Blocks: []StateBlockRef{{
			Index: 0, TokenStart: 0, TokenCount: 3,
			PayloadEncoding: "",
			State:           chunk,
		}},
	}
	if _, err := LoadPrefixTokensFromStateBlocks(ctx, store, countBundle, 3); err == nil {
		t.Fatal("LoadPrefixTokensFromStateBlocks(token count mismatch) error = nil, want count error")
	}

	// Load failure: a JSON ref pointing at a missing chunk.
	missingBundle := &StateBlockBundle{
		Version:    StateBlockVersion,
		Kind:       StateBlockBundleKind,
		TokenCount: 2,
		Blocks: []StateBlockRef{{
			Index: 0, TokenStart: 0, TokenCount: 2,
			PayloadEncoding: "",
			State:           state.ChunkRef{ChunkID: 987654},
		}},
	}
	if _, err := LoadPrefixTokensFromStateBlocks(ctx, store, missingBundle, 2); err == nil {
		t.Fatal("LoadPrefixTokensFromStateBlocks(missing chunk) error = nil, want load error")
	}
}

// TestBlocksJSONCover_PrefixMetadataMismatch drives the prefix assembler's
// per-block metadata mismatch guard via a crafted JSON block whose stored
// BlockIndex diverges, exercised through a multi-block prefix request.
func TestBlocksJSONCover_PrefixTokenCountMismatch(t *testing.T) {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)
	payload := twoTokenBlockPayload(t, 1, 2)

	// Stored payload carries 2 tokens; ref claims 2 but the prefix assembler's
	// token-count guard fires when the snapshot token slice disagrees with the
	// ref count — craft a ref claiming 1 token for a 2-token payload.
	chunk := putJSONBlock(t, store, "mlx://prefix-count", kvSnapshotStateBlockEnvelope{
		BlockIndex: 0, TokenStart: 0, TokenCount: 1,
	}, payload)
	bundle := &StateBlockBundle{
		Version:    StateBlockVersion,
		Kind:       StateBlockBundleKind,
		TokenCount: 2,
		Blocks: []StateBlockRef{{
			Index: 0, TokenStart: 0, TokenCount: 1,
			PayloadEncoding: "",
			State:           chunk,
		}},
	}
	if _, err := LoadPrefixFromStateBlocks(ctx, store, bundle, 1); err == nil {
		t.Fatal("LoadPrefixFromStateBlocks(token count mismatch) error = nil, want count error")
	}
}

// TestBlocksJSONCover_RawPayloadParseError drives the snapshot-parse error arm
// of the raw block load path: a raw-encoded ref whose stored bytes are a
// truncated snapshot.
func TestBlocksJSONCover_RawPayloadParseError(t *testing.T) {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)

	truncated := []byte(kvSnapshotMagic)
	chunk, err := store.PutBytes(ctx, truncated, state.PutOptions{URI: "mlx://raw-parse-error"})
	if err != nil {
		t.Fatalf("PutBytes error = %v", err)
	}
	ref := StateBlockRef{
		Index: 0, TokenStart: 0, TokenCount: 2,
		PayloadEncoding:  kvSnapshotStatePayloadRaw,
		PayloadByteCount: len(truncated),
		State:            chunk,
	}
	if _, err := loadRawKVSnapshotStateBlockWithOptions(ctx, store, ref, LoadOptions{}); err == nil {
		t.Fatal("loadRawKVSnapshotStateBlockWithOptions(unparseable) error = nil, want parse error")
	}
}
