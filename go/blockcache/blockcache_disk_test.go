// SPDX-Licence-Identifier: EUPL-1.2

package blockcache

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	state "dappco.re/go/inference/state"
)

// recordingStateWriter is a test stub that returns a fixed ChunkRef and records
// the last payload it received. It lets the State cold-store success path be
// driven with a ChunkRef whose optional fields (Codec/Segment/FrameOffset) are
// all populated, exercising the withStateLabels label-emission branches that the
// in-memory store leaves empty.
type recordingStateWriter struct {
	ref state.ChunkRef
}

func (w recordingStateWriter) Put(_ context.Context, _ string, _ state.PutOptions) (state.ChunkRef, error) {
	return w.ref, nil
}

// ---------------------------------------------------------------------------
// Unexported helper branch coverage — pure logic, no filesystem.
// ---------------------------------------------------------------------------

func TestBlockCacheHelpers_Branches(t *testing.T) {
	// stateStore on a nil receiver returns nil rather than dereferencing the
	// nil *Service (the early-return guard the public callers never hit because
	// they reject nil first).
	if (*Service)(nil).stateStore() != nil {
		t.Fatal("stateStore(nil service) != nil")
	}
	if (*Service)(nil).stateStoreEnabled() {
		t.Fatal("stateStoreEnabled(nil service) = true")
	}
	if (*Service)(nil).diskEnabled() {
		t.Fatal("diskEnabled(nil service) = true")
	}

	// blockRefs clamps a non-positive BlockSize to DefaultBlockSize. A Service
	// literal with BlockSize 0 (bypassing New's clamp) chunks a short token run
	// into a single default-sized block.
	zeroSizeService := &Service{cfg: Config{}, blocks: map[string]inference.CacheBlockRef{}}
	refs := zeroSizeService.blockRefs(inference.CacheWarmRequest{}, []int32{1, 2, 3}, nil)
	if len(refs) != 1 || refs[0].TokenCount != 3 {
		t.Fatalf("blockRefs(BlockSize=0) = %+v, want one default-sized block", refs)
	}

	// diskRecordCompatible rejects an empty-ID record outright, and rejects a
	// record whose adapter hash mismatches the configured identity.
	service := &Service{cfg: Config{
		ModelHash:     "sha256:model",
		AdapterHash:   "sha256:adapter",
		TokenizerHash: "sha256:tokenizer",
	}}
	if service.diskRecordCompatible(diskRecord{}) {
		t.Fatal("diskRecordCompatible(empty ID) = true")
	}
	if service.diskRecordCompatible(diskRecord{Ref: inference.CacheBlockRef{
		ID:          "x",
		ModelHash:   "sha256:model",
		AdapterHash: "sha256:other-adapter",
	}}) {
		t.Fatal("diskRecordCompatible(adapter mismatch) = true")
	}

	// blockRefMatchesLabels rejects an adapter_hash mismatch and a
	// tokenizer_hash mismatch (the two switch arms the existing Good test does
	// not flip individually).
	ref := inference.CacheBlockRef{ModelHash: "m", AdapterHash: "a", TokenizerHash: "t"}
	if blockRefMatchesLabels(ref, map[string]string{"adapter_hash": "other"}) {
		t.Fatal("blockRefMatchesLabels(adapter mismatch) = true")
	}
	if blockRefMatchesLabels(ref, map[string]string{"tokenizer_hash": "other"}) {
		t.Fatal("blockRefMatchesLabels(tokenizer mismatch) = true")
	}

	// cloneBlockCacheLabelsExtra clamps a negative extra to zero rather than
	// passing a negative capacity hint to make.
	cloned := cloneBlockCacheLabelsExtra(map[string]string{"a": "b"}, -4)
	if cloned["a"] != "b" {
		t.Fatalf("cloneBlockCacheLabelsExtra(extra<0) = %+v, want copied entry", cloned)
	}

	// writeStateBlock guards a nil store: a Service whose state store is unset
	// returns an explicit error instead of calling Put on nil.
	if _, err := (&Service{}).writeStateBlock(context.Background(), inference.CacheBlockRef{ID: "x"}, nil); err == nil {
		t.Fatal("writeStateBlock(nil store) error = nil")
	}
	// writeStateBlock substitutes context.Background for a nil context (the
	// documented fast path) — with a recording store the call succeeds.
	okStore := &Service{cfg: Config{StateStore: recordingStateWriter{}}}
	//nolint:staticcheck // SA1012: passing a nil Context is the path under test.
	if _, err := okStore.writeStateBlock(nil, inference.CacheBlockRef{ID: "x"}, []int32{1}); err != nil {
		t.Fatalf("writeStateBlock(nil ctx) error = %v, want nil", err)
	}
}

// TestBlockCacheHelpers_WithStateLabels exercises every optional-field arm of
// withStateLabels: a ChunkRef carrying a codec, a segment, and a frame offset
// emits the corresponding state_* labels, which the in-memory store path leaves
// unset.
func TestBlockCacheHelpers_WithStateLabels(t *testing.T) {
	labelled := withStateLabels(inference.CacheBlockRef{ID: "x"}, state.ChunkRef{
		ChunkID:        7,
		Codec:          "zstd",
		Segment:        "seg-1",
		HasFrameOffset: true,
		FrameOffset:    42,
	})
	if labelled.Labels["cold_store"] != "state" {
		t.Fatalf("cold_store label = %q, want state", labelled.Labels["cold_store"])
	}
	if labelled.Labels["state_chunk_id"] != "7" {
		t.Fatalf("state_chunk_id = %q, want 7", labelled.Labels["state_chunk_id"])
	}
	if labelled.Labels["state_codec"] != "zstd" {
		t.Fatalf("state_codec = %q, want zstd", labelled.Labels["state_codec"])
	}
	if labelled.Labels["state_segment"] != "seg-1" {
		t.Fatalf("state_segment = %q, want seg-1", labelled.Labels["state_segment"])
	}
	if labelled.Labels["state_frame_offset"] != "42" {
		t.Fatalf("state_frame_offset = %q, want 42", labelled.Labels["state_frame_offset"])
	}
}

// TestBlockCacheHelpers_SortPdqsort drives sortCacheBlockRefs past its
// insertion-sort threshold (32) so the pdqsort branch executes. The input is
// reverse-ordered by TokenStart; the result must be ascending.
func TestBlockCacheHelpers_SortPdqsort(t *testing.T) {
	const n = sortCacheBlockRefsInsertionThreshold + 8 // 40 > 32
	refs := make([]inference.CacheBlockRef, n)
	for i := range refs {
		refs[i] = inference.CacheBlockRef{
			ID:         core.Itoa(n - i),
			TokenStart: (n - i) * 2,
		}
	}
	sortCacheBlockRefs(refs)
	for i := 1; i < len(refs); i++ {
		if refs[i-1].TokenStart > refs[i].TokenStart {
			t.Fatalf("sortCacheBlockRefs(n=%d) not ascending at %d: %d > %d", n, i, refs[i-1].TokenStart, refs[i].TokenStart)
		}
	}
}
