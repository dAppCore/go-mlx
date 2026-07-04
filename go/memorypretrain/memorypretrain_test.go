// SPDX-Licence-Identifier: EUPL-1.2

package memorypretrain

import (
	"context"
	"errors"
	"math"
	"strings"
	"testing"
)

// TestMemorypretrain_EmbedFunc_Embed_Good adapts a closure into an Embedder and
// confirms the adapter forwards the context and text to the wrapped function.
func TestMemorypretrain_EmbedFunc_Embed_Good(t *testing.T) {
	var gotText string
	fn := EmbedFunc(func(_ context.Context, text string) ([]float32, error) {
		gotText = text
		return []float32{1, 0}, nil
	})
	out, err := fn.Embed(context.Background(), "hello")
	if err != nil {
		t.Fatalf("Embed() error = %v", err)
	}
	if gotText != "hello" || len(out) != 2 || out[0] != 1 {
		t.Fatalf("Embed() forwarded text=%q out=%+v, want the wrapped call", gotText, out)
	}
}

// TestMemorypretrain_EmbedFunc_Embed_Bad rejects a nil EmbedFunc receiver rather
// than panicking when Embed is called on it.
func TestMemorypretrain_EmbedFunc_Embed_Bad(t *testing.T) {
	if _, err := (EmbedFunc)(nil).Embed(context.Background(), "x"); err == nil {
		t.Fatal("EmbedFunc(nil).Embed() error = nil")
	}
}

// TestMemorypretrain_EmbedFunc_Embed_Ugly propagates the wrapped function's own
// error verbatim through the adapter.
func TestMemorypretrain_EmbedFunc_Embed_Ugly(t *testing.T) {
	wantErr := errors.New("anchor offline")
	fn := EmbedFunc(func(context.Context, string) ([]float32, error) { return nil, wantErr })
	if _, err := fn.Embed(context.Background(), "x"); !errors.Is(err, wantErr) {
		t.Fatalf("Embed() error = %v, want the wrapped error", err)
	}
}

// TestMemorypretrain_BuildBank_Good builds a hierarchical bank and confirms it
// routes a query to the nearest cluster, plus the equal-score tie-break by block
// index in the retrieval sort.
func TestMemorypretrain_BuildBank_Good(t *testing.T) {
	bank, err := BuildBank([]Block{
		{ID: "go-1", Text: "Go memory planning", Embedding: []float32{1, 0}},
		{ID: "go-2", Text: "Go cgo bridge", Embedding: []float32{0.9, 0.1}},
		{ID: "poem-1", Text: "winter proof poem", Embedding: []float32{0, 1}},
		{ID: "poem-2", Text: "autumn prayer", Embedding: []float32{0.1, 0.9}},
	}, BuildConfig{BranchingFactor: 2, MaxDepth: 2, MinClusterSize: 2, KMeansIters: 8})
	if err != nil {
		t.Fatalf("BuildBank() error = %v", err)
	}
	if bank.Dimension != 2 || len(bank.Nodes) < 3 {
		t.Fatalf("bank = %+v, want dimension and child clusters", bank)
	}
	got, err := bank.Retrieve([]float32{1, 0}, 2)
	if err != nil {
		t.Fatalf("Retrieve() error = %v", err)
	}
	if len(got) != 2 || got[0].BlockID != "go-1" || got[1].BlockID != "go-2" {
		t.Fatalf("Retrieve() = %+v, want Go cluster ordered by score", got)
	}
	t.Run("tied scores break by block index", func(t *testing.T) {
		tied, err := BuildBank([]Block{
			{ID: "first", Embedding: []float32{1, 0}},
			{ID: "second", Embedding: []float32{1, 0}},
			{ID: "third", Embedding: []float32{1, 0}},
		}, BuildConfig{BranchingFactor: 2, MaxDepth: 1, MinClusterSize: 8, KMeansIters: 4})
		if err != nil {
			t.Fatalf("BuildBank(tied scores) error = %v", err)
		}
		gotTied, err := tied.Retrieve([]float32{1, 0}, 3)
		if err != nil {
			t.Fatalf("Retrieve(tied scores) error = %v", err)
		}
		if len(gotTied) != 3 || gotTied[0].BlockIndex >= gotTied[1].BlockIndex || gotTied[1].BlockIndex >= gotTied[2].BlockIndex {
			t.Fatalf("Retrieve(tied scores) = %+v, want ascending block index on score ties", gotTied)
		}
	})
}

// TestMemorypretrain_BuildBank_Bad proves BuildBank clones its input (no aliasing)
// and rejects malformed block sets: dimension mismatch, an empty slice, a
// zero-length embedding, and non-finite (NaN / +Inf) embedding values.
func TestMemorypretrain_BuildBank_Bad(t *testing.T) {
	blocks := []Block{{ID: "a", Embedding: []float32{1, 0}, Meta: map[string]string{"source": "unit"}}}
	bank, err := BuildBank(blocks, BuildConfig{})
	if err != nil {
		t.Fatalf("BuildBank() error = %v", err)
	}
	blocks[0].Embedding[0] = 0
	blocks[0].Meta["source"] = "mutated"
	if bank.Blocks[0].Embedding[0] != 1 || bank.Blocks[0].Meta["source"] != "unit" {
		t.Fatalf("bank block aliased input: %+v", bank.Blocks[0])
	}
	if _, err := BuildBank([]Block{{Embedding: []float32{1}}, {Embedding: []float32{1, 2}}}, BuildConfig{}); err == nil {
		t.Fatal("BuildBank() dimension mismatch error = nil")
	}
	// An empty block slice is rejected before validation.
	if _, err := BuildBank(nil, BuildConfig{}); err == nil {
		t.Fatal("BuildBank(no blocks) error = nil")
	}
	// A zero-length embedding on the first block is rejected before any routing.
	if _, err := BuildBank([]Block{{Embedding: []float32{}}}, BuildConfig{}); err == nil {
		t.Fatal("BuildBank() empty embedding error = nil")
	}
	// Non-finite embedding values (NaN, +Inf) are rejected.
	if _, err := BuildBank([]Block{{Embedding: []float32{float32(math.NaN()), 0}}}, BuildConfig{}); err == nil {
		t.Fatal("BuildBank() NaN embedding error = nil")
	}
	if _, err := BuildBank([]Block{
		{Embedding: []float32{1, 0}},
		{Embedding: []float32{float32(math.Inf(1)), 0}},
	}, BuildConfig{}); err == nil {
		t.Fatal("BuildBank() infinite embedding error = nil")
	}
}

// TestMemorypretrain_BuildBank_Ugly builds a bank from fewer blocks than the
// branching factor, all sharing one embedding. kmeans is reached (MinClusterSize
// is below the block count) but degenerates: k clamps to the block count, every
// point collapses into a single cluster, so buildNode takes the single-cluster
// short-circuit and the bank is one leaf.
func TestMemorypretrain_BuildBank_Ugly(t *testing.T) {
	bank, err := BuildBank([]Block{
		{ID: "a", Embedding: []float32{1, 0}},
		{ID: "b", Embedding: []float32{1, 0}},
	}, BuildConfig{BranchingFactor: 8, MaxDepth: 3, MinClusterSize: 1, KMeansIters: 4})
	if err != nil {
		t.Fatalf("BuildBank(duplicate blocks) error = %v", err)
	}
	// Identical embeddings cannot split into multiple clusters, so the root stays
	// a single leaf holding both blocks.
	if len(bank.Nodes) != 1 || len(bank.Nodes[0].Children) != 0 || len(bank.Nodes[0].BlockIDs) != 2 {
		t.Fatalf("nodes = %+v, want a single leaf node holding both blocks", bank.Nodes)
	}
	got, err := bank.Retrieve([]float32{1, 0}, 2)
	if err != nil {
		t.Fatalf("Retrieve(duplicate blocks) error = %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("Retrieve(duplicate blocks) = %+v, want both blocks", got)
	}
}

// TestMemorypretrain_BuildBankFromCorpus_Good embeds corpus records with the
// anchor embedder, clones their metadata, and routes a query to the embedded
// record.
func TestMemorypretrain_BuildBankFromCorpus_Good(t *testing.T) {
	records := []CorpusRecord{
		{ID: "go", Text: "Go memory planning", Meta: map[string]string{"source": "docs"}},
		{ID: "poem", Text: "winter proof poem", Meta: map[string]string{"source": "creative"}},
	}
	embedder := EmbedFunc(func(_ context.Context, text string) ([]float32, error) {
		if strings.Contains(text, "Go") {
			return []float32{1, 0}, nil
		}
		return []float32{0, 1}, nil
	})
	bank, err := BuildBankFromCorpus(context.Background(), embedder, records, BuildConfig{BranchingFactor: 2, MaxDepth: 1, MinClusterSize: 2})
	if err != nil {
		t.Fatalf("BuildBankFromCorpus() error = %v", err)
	}
	if bank.Dimension != 2 || len(bank.Blocks) != 2 {
		t.Fatalf("bank dimension=%d blocks=%d, want embedded records", bank.Dimension, len(bank.Blocks))
	}
	records[0].Meta["source"] = "mutated"
	if bank.Blocks[0].ID != "go" || bank.Blocks[0].Text != "Go memory planning" || bank.Blocks[0].Meta["source"] != "docs" {
		t.Fatalf("bank block = %+v, want cloned corpus metadata", bank.Blocks[0])
	}
	got, err := bank.Retrieve([]float32{1, 0}, 1)
	if err != nil {
		t.Fatalf("Retrieve() error = %v", err)
	}
	if len(got) != 1 || got[0].BlockID != "go" {
		t.Fatalf("Retrieve() = %+v, want embedded Go record", got)
	}
}

// TestMemorypretrain_BuildBankFromCorpus_Bad rejects a nil embedder and an empty
// record set, and wraps an embedder error with the offending record index.
func TestMemorypretrain_BuildBankFromCorpus_Bad(t *testing.T) {
	if _, err := BuildBankFromCorpus(context.Background(), nil, []CorpusRecord{{Text: "x"}}, BuildConfig{}); err == nil {
		t.Fatal("BuildBankFromCorpus(nil embedder) error = nil")
	}
	if _, err := BuildBankFromCorpus(context.Background(), EmbedFunc(func(context.Context, string) ([]float32, error) {
		return []float32{1}, nil
	}), nil, BuildConfig{}); err == nil {
		t.Fatal("BuildBankFromCorpus(empty records) error = nil")
	}
	wantErr := errors.New("anchor unavailable")
	if _, err := BuildBankFromCorpus(context.Background(), EmbedFunc(func(context.Context, string) ([]float32, error) {
		return nil, wantErr
	}), []CorpusRecord{{Text: "x"}}, BuildConfig{}); err == nil || !strings.Contains(err.Error(), "embed record 0") {
		t.Fatalf("BuildBankFromCorpus(embed error) error = %v, want record context", err)
	}
}

// TestMemorypretrain_BuildBankFromCorpus_Ugly cancels the context before the
// build runs, so BuildBankFromCorpus returns the cancellation without embedding
// any record.
func TestMemorypretrain_BuildBankFromCorpus_Ugly(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	calls := 0
	_, err := BuildBankFromCorpus(ctx, EmbedFunc(func(context.Context, string) ([]float32, error) {
		calls++
		return []float32{1}, nil
	}), []CorpusRecord{{Text: "x"}}, BuildConfig{})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("BuildBankFromCorpus(cancelled) error = %v, want context.Canceled", err)
	}
	if calls != 0 {
		t.Fatalf("embed calls = %d, want cancellation before embedding", calls)
	}
}

// TestMemorypretrain_Bank_Retrieve_Good routes a query to the nearest leaf and
// returns the top-k blocks ordered by cosine score.
func TestMemorypretrain_Bank_Retrieve_Good(t *testing.T) {
	bank, err := BuildBank([]Block{
		{ID: "go-1", Text: "Go memory planning", Embedding: []float32{1, 0}},
		{ID: "go-2", Text: "Go cgo bridge", Embedding: []float32{0.9, 0.1}},
		{ID: "poem-1", Text: "winter proof poem", Embedding: []float32{0, 1}},
		{ID: "poem-2", Text: "autumn prayer", Embedding: []float32{0.1, 0.9}},
	}, BuildConfig{BranchingFactor: 2, MaxDepth: 2, MinClusterSize: 2, KMeansIters: 8})
	if err != nil {
		t.Fatalf("BuildBank() error = %v", err)
	}
	got, err := bank.Retrieve([]float32{1, 0}, 2)
	if err != nil {
		t.Fatalf("Retrieve() error = %v", err)
	}
	if len(got) != 2 || got[0].BlockID != "go-1" || got[1].BlockID != "go-2" {
		t.Fatalf("Retrieve() = %+v, want Go cluster ordered by score", got)
	}
}

// TestMemorypretrain_Bank_Retrieve_Bad covers the entry guards: a nil bank, a
// query whose dimension mismatches the bank, and a non-positive k.
func TestMemorypretrain_Bank_Retrieve_Bad(t *testing.T) {
	if _, err := (*Bank)(nil).Retrieve([]float32{1}, 1); err == nil {
		t.Fatal("Retrieve(nil) error = nil")
	}
	bank, err := BuildBank([]Block{{ID: "a", Embedding: []float32{1, 0}}}, BuildConfig{})
	if err != nil {
		t.Fatalf("BuildBank() error = %v", err)
	}
	if _, err := bank.Retrieve([]float32{1}, 1); err == nil {
		t.Fatal("Retrieve(wrong dim) error = nil")
	}
	if _, err := bank.Retrieve([]float32{1, 0}, 0); err == nil {
		t.Fatal("Retrieve(k=0) error = nil")
	}
}

// TestMemorypretrain_Bank_Retrieve_Ugly routes into a synthetic leaf that holds
// no block IDs, so Retrieve returns an empty result rather than indexing a
// missing block.
func TestMemorypretrain_Bank_Retrieve_Ugly(t *testing.T) {
	bank := &Bank{
		Dimension: 2,
		Blocks:    []Block{{ID: "a", Embedding: []float32{1, 0}}},
		Nodes:     []Node{{ID: 0, Parent: -1, Centroid: []float32{1, 0}, BlockIDs: nil}},
		Root:      0,
		Config:    BuildConfig{BranchingFactor: 2, MaxDepth: 1, MinClusterSize: 1, KMeansIters: 1},
	}
	got, err := bank.Retrieve([]float32{1, 0}, 4)
	if err != nil {
		t.Fatalf("Retrieve(empty leaf) error = %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("Retrieve(empty leaf) = %+v, want no retrievals", got)
	}
}

// TestMemorypretrain_Bank_ClusterIDs_Good routes a query and returns the per-level
// hierarchical cluster IDs, matching the IDs derived from ClusterAssignments.
func TestMemorypretrain_Bank_ClusterIDs_Good(t *testing.T) {
	bank, err := BuildBank([]Block{
		{ID: "go-1", Embedding: []float32{1, 0}},
		{ID: "go-2", Embedding: []float32{0.9, 0.1}},
		{ID: "poem-1", Embedding: []float32{0, 1}},
		{ID: "poem-2", Embedding: []float32{0.1, 0.9}},
	}, BuildConfig{BranchingFactor: 2, MaxDepth: 2, MinClusterSize: 1, KMeansIters: 8})
	if err != nil {
		t.Fatalf("BuildBank() error = %v", err)
	}
	assignments, err := bank.ClusterAssignments([]float32{1, 0})
	if err != nil {
		t.Fatalf("ClusterAssignments() error = %v", err)
	}
	ids, err := bank.ClusterIDs([]float32{1, 0})
	if err != nil {
		t.Fatalf("ClusterIDs() error = %v", err)
	}
	if len(ids) != 2 || ids[0] != assignments[0].ClusterID || ids[1] != assignments[1].ClusterID {
		t.Fatalf("ClusterIDs() = %+v, assignments=%+v", ids, assignments)
	}
}

// TestMemorypretrain_Bank_ClusterIDs_Bad rejects a nil bank from the ClusterIDs
// entry point, which delegates to ClusterIDsInto.
func TestMemorypretrain_Bank_ClusterIDs_Bad(t *testing.T) {
	if _, err := (*Bank)(nil).ClusterIDs([]float32{1, 0}); err == nil {
		t.Fatal("ClusterIDs(nil bank) error = nil")
	}
}

// TestMemorypretrain_Bank_ClusterIDs_Ugly drives the dimension-mismatch and
// missing-root guards through ClusterIDs on a structurally-empty bank.
func TestMemorypretrain_Bank_ClusterIDs_Ugly(t *testing.T) {
	empty := &Bank{Dimension: 2}
	if _, err := empty.ClusterIDs([]float32{1}); err == nil {
		t.Fatal("ClusterIDs(dim mismatch) error = nil")
	}
	if _, err := empty.ClusterIDs([]float32{1, 0}); err == nil {
		t.Fatal("ClusterIDs(no root) error = nil")
	}
}

// TestMemorypretrain_Bank_ClusterIDsInto_Good threads one buffer through repeated
// routing: the returned slice reuses dst's backing when it has capacity, and the
// cluster IDs match the allocating ClusterIDs path.
func TestMemorypretrain_Bank_ClusterIDsInto_Good(t *testing.T) {
	bank, err := BuildBank([]Block{
		{ID: "go-1", Embedding: []float32{1, 0}},
		{ID: "go-2", Embedding: []float32{0.9, 0.1}},
		{ID: "poem-1", Embedding: []float32{0, 1}},
		{ID: "poem-2", Embedding: []float32{0.1, 0.9}},
	}, BuildConfig{BranchingFactor: 2, MaxDepth: 2, MinClusterSize: 1, KMeansIters: 8})
	if err != nil {
		t.Fatalf("BuildBank() error = %v", err)
	}
	scratch := make([]int, 0, 8)
	got, err := bank.ClusterIDsInto(scratch, []float32{1, 0})
	if err != nil {
		t.Fatalf("ClusterIDsInto() error = %v", err)
	}
	want, err := bank.ClusterIDs([]float32{1, 0})
	if err != nil {
		t.Fatalf("ClusterIDs() error = %v", err)
	}
	if len(got) != len(want) || cap(got) != cap(scratch) {
		t.Fatalf("ClusterIDsInto() = %+v cap=%d, want %+v in caller buffer cap=%d", got, cap(got), want, cap(scratch))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("ClusterIDsInto()[%d] = %d, want %d (same IDs as ClusterIDs)", i, got[i], want[i])
		}
	}
	// Routing a second query through the same buffer reuses the backing array.
	again, err := bank.ClusterIDsInto(got, []float32{0, 1})
	if err != nil {
		t.Fatalf("ClusterIDsInto(reuse) error = %v", err)
	}
	if cap(again) != cap(scratch) {
		t.Fatalf("ClusterIDsInto(reuse) cap = %d, want the threaded buffer cap %d", cap(again), cap(scratch))
	}
}

// TestMemorypretrain_Bank_ClusterIDsInto_Bad rejects a nil bank from the
// allocation-free routing entry point.
func TestMemorypretrain_Bank_ClusterIDsInto_Bad(t *testing.T) {
	if _, err := (*Bank)(nil).ClusterIDsInto(nil, []float32{1, 0}); err == nil {
		t.Fatal("ClusterIDsInto(nil bank) error = nil")
	}
}

// TestMemorypretrain_Bank_ClusterIDsInto_Ugly drives the dimension-mismatch and
// missing-root guards through ClusterIDsInto on a structurally-empty bank.
func TestMemorypretrain_Bank_ClusterIDsInto_Ugly(t *testing.T) {
	empty := &Bank{Dimension: 2}
	if _, err := empty.ClusterIDsInto(nil, []float32{1}); err == nil {
		t.Fatal("ClusterIDsInto(dim mismatch) error = nil")
	}
	if _, err := empty.ClusterIDsInto(nil, []float32{1, 0}); err == nil {
		t.Fatal("ClusterIDsInto(no root) error = nil")
	}
}

// TestMemorypretrain_Bank_ClusterAssignments_Good records one assignment per
// reached level, with the hierarchical parent*branching+local cluster ID at each
// level.
func TestMemorypretrain_Bank_ClusterAssignments_Good(t *testing.T) {
	bank, err := BuildBank([]Block{
		{ID: "go-1", Embedding: []float32{1, 0}},
		{ID: "go-2", Embedding: []float32{0.9, 0.1}},
		{ID: "poem-1", Embedding: []float32{0, 1}},
		{ID: "poem-2", Embedding: []float32{0.1, 0.9}},
	}, BuildConfig{BranchingFactor: 2, MaxDepth: 2, MinClusterSize: 1, KMeansIters: 8})
	if err != nil {
		t.Fatalf("BuildBank() error = %v", err)
	}
	assignments, err := bank.ClusterAssignments([]float32{1, 0})
	if err != nil {
		t.Fatalf("ClusterAssignments() error = %v", err)
	}
	if len(assignments) != 2 {
		t.Fatalf("assignments = %+v, want one routed cluster per hierarchy level", assignments)
	}
	if assignments[0].Level != 1 || assignments[0].LocalClusterID != 0 || assignments[0].ClusterID != 0 {
		t.Fatalf("level 1 assignment = %+v, want first root child", assignments[0])
	}
	if assignments[1].Level != 2 || assignments[1].ClusterID != assignments[0].ClusterID*2+assignments[1].LocalClusterID {
		t.Fatalf("level 2 assignment = %+v after %+v, want hierarchical global id", assignments[1], assignments[0])
	}
}

// TestMemorypretrain_Bank_ClusterAssignments_Bad rejects a nil bank.
func TestMemorypretrain_Bank_ClusterAssignments_Bad(t *testing.T) {
	if _, err := (*Bank)(nil).ClusterAssignments([]float32{1, 0}); err == nil {
		t.Fatal("ClusterAssignments(nil bank) error = nil")
	}
}

// TestMemorypretrain_Bank_ClusterAssignments_Ugly drives the dimension-mismatch
// and missing-root guards on a structurally-empty bank.
func TestMemorypretrain_Bank_ClusterAssignments_Ugly(t *testing.T) {
	empty := &Bank{Dimension: 2}
	if _, err := empty.ClusterAssignments([]float32{1}); err == nil {
		t.Fatal("ClusterAssignments(dim mismatch) error = nil")
	}
	if _, err := empty.ClusterAssignments([]float32{1, 0}); err == nil {
		t.Fatal("ClusterAssignments(no root) error = nil")
	}
}

// TestMemorypretrain_GenericClusterIDs_Good returns the generic-memory fallback:
// the last cluster index at each level.
func TestMemorypretrain_GenericClusterIDs_Good(t *testing.T) {
	ids, err := GenericClusterIDs([]int{16, 256, 1024})
	if err != nil {
		t.Fatalf("GenericClusterIDs() error = %v", err)
	}
	if len(ids) != 3 || ids[0] != 15 || ids[1] != 255 || ids[2] != 1023 {
		t.Fatalf("GenericClusterIDs() = %+v, want last cluster per level", ids)
	}
}

// TestMemorypretrain_GenericClusterIDs_Bad rejects empty cluster counts.
func TestMemorypretrain_GenericClusterIDs_Bad(t *testing.T) {
	if _, err := GenericClusterIDs(nil); err == nil {
		t.Fatal("GenericClusterIDs(empty) error = nil")
	}
}

// TestMemorypretrain_GenericClusterIDs_Ugly rejects a non-positive cluster count
// at a level rather than emitting a negative index.
func TestMemorypretrain_GenericClusterIDs_Ugly(t *testing.T) {
	if _, err := GenericClusterIDs([]int{16, 0}); err == nil {
		t.Fatal("GenericClusterIDs(zero level) error = nil")
	}
	if _, err := GenericClusterIDs([]int{-1}); err == nil {
		t.Fatal("GenericClusterIDs(negative level) error = nil")
	}
}

// TestMemorypretrain_Bank_RetrieveInto_Good threads a caller buffer through
// retrieval (the returned slice reuses the scratch backing) and clamps k down to
// the available block count rather than over-slicing.
func TestMemorypretrain_Bank_RetrieveInto_Good(t *testing.T) {
	bank, err := BuildBank([]Block{
		{ID: "go-1", Text: "Go memory planning", Embedding: []float32{1, 0}},
		{ID: "go-2", Text: "Go cgo bridge", Embedding: []float32{0.9, 0.1}},
		{ID: "poem-1", Text: "winter proof poem", Embedding: []float32{0, 1}},
		{ID: "poem-2", Text: "autumn prayer", Embedding: []float32{0.1, 0.9}},
	}, BuildConfig{BranchingFactor: 2, MaxDepth: 2, MinClusterSize: 2, KMeansIters: 8})
	if err != nil {
		t.Fatalf("BuildBank() error = %v", err)
	}
	scratch := make([]Retrieval, 0, 2)
	reused, err := bank.RetrieveInto(scratch, []float32{0, 1}, 2)
	if err != nil {
		t.Fatalf("RetrieveInto() error = %v", err)
	}
	if len(reused) != 2 || reused[0].BlockID != "poem-1" || cap(reused) != cap(scratch) {
		t.Fatalf("RetrieveInto() = %+v cap=%d, want poem cluster in caller storage cap=%d", reused, cap(reused), cap(scratch))
	}
	t.Run("clamps k to available blocks", func(t *testing.T) {
		small, err := BuildBank([]Block{
			{ID: "a", Embedding: []float32{1, 0}},
			{ID: "b", Embedding: []float32{0.9, 0.1}},
		}, BuildConfig{BranchingFactor: 2, MaxDepth: 1, MinClusterSize: 2})
		if err != nil {
			t.Fatalf("BuildBank() error = %v", err)
		}
		got, err := small.RetrieveInto(nil, []float32{1, 0}, 100)
		if err != nil {
			t.Fatalf("RetrieveInto() error = %v", err)
		}
		if len(got) != 2 || got[0].BlockID != "a" {
			t.Fatalf("RetrieveInto(k=100) = %+v, want all available blocks clamped", got)
		}
	})
}

// TestMemorypretrain_Bank_RetrieveInto_Bad covers the entry guards: dimension
// mismatch, non-positive k, and a nil bank.
func TestMemorypretrain_Bank_RetrieveInto_Bad(t *testing.T) {
	if _, err := (*Bank)(nil).RetrieveInto(nil, []float32{1}, 1); err == nil {
		t.Fatal("RetrieveInto(nil bank) error = nil")
	}
	bank, err := BuildBank([]Block{{ID: "a", Embedding: []float32{1, 0}}}, BuildConfig{})
	if err != nil {
		t.Fatalf("BuildBank() error = %v", err)
	}
	if _, err := bank.RetrieveInto(nil, []float32{1}, 1); err == nil {
		t.Fatal("RetrieveInto(dim mismatch) error = nil")
	}
	if _, err := bank.RetrieveInto(nil, []float32{1, 0}, 0); err == nil {
		t.Fatal("RetrieveInto(k=0) error = nil")
	}
}

// TestMemorypretrain_Bank_RetrieveInto_Ugly routes into a synthetic empty leaf
// and confirms RetrieveInto returns the reset (empty) destination slice.
func TestMemorypretrain_Bank_RetrieveInto_Ugly(t *testing.T) {
	bank := &Bank{
		Dimension: 2,
		Blocks:    []Block{{ID: "a", Embedding: []float32{1, 0}}},
		Nodes:     []Node{{ID: 0, Parent: -1, Centroid: []float32{1, 0}, BlockIDs: nil}},
		Root:      0,
		Config:    BuildConfig{BranchingFactor: 2, MaxDepth: 1, MinClusterSize: 1, KMeansIters: 1},
	}
	got, err := bank.RetrieveInto(nil, []float32{1, 0}, 4)
	if err != nil {
		t.Fatalf("RetrieveInto(empty leaf) error = %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("RetrieveInto(empty leaf) = %+v, want no retrievals", got)
	}
}

// TestMemorypretrain_Bank_InjectAdditive_Good adds weighted retrieved memory into
// a hidden activation, covering the score-weighted blend, the uniform fallback
// when every score is clamped, and the per-block skip of a clamped negative score.
func TestMemorypretrain_Bank_InjectAdditive_Good(t *testing.T) {
	t.Run("adds retrieved memory", func(t *testing.T) {
		bank, err := BuildBank([]Block{
			{ID: "near", Embedding: []float32{1, 0}},
			{ID: "far", Embedding: []float32{0, 1}},
		}, BuildConfig{BranchingFactor: 2, MaxDepth: 1, MinClusterSize: 2})
		if err != nil {
			t.Fatalf("BuildBank() error = %v", err)
		}
		hidden := []float32{0.25, 0.5}
		dst := make([]float32, 0, 2)
		scratch := make([]Retrieval, 0, 2)
		out, retrievals, stats, err := bank.InjectAdditive(dst, hidden, []float32{1, 0}, scratch, InjectionConfig{TopK: 1, Scale: 0.5, PositiveScoresOnly: true})
		if err != nil {
			t.Fatalf("InjectAdditive() error = %v", err)
		}
		if len(retrievals) != 1 || retrievals[0].BlockID != "near" {
			t.Fatalf("retrievals = %+v, want nearest memory block", retrievals)
		}
		if !stats.Applied || stats.Retrieved != 1 || stats.Scale != 0.5 {
			t.Fatalf("stats = %+v, want applied injection", stats)
		}
		if len(out) != 2 || out[0] != 0.75 || out[1] != 0.5 || cap(out) != cap(dst) {
			t.Fatalf("out = %+v cap=%d, want hidden plus scaled memory in caller buffer cap=%d", out, cap(out), cap(dst))
		}
	})

	t.Run("uniform fallback when all scores clamped", func(t *testing.T) {
		// Every retrieved block has a negative cosine to the query, so
		// PositiveScoresOnly clamps the whole weight sum to zero and the injector
		// falls back to a uniform 1/k blend (the WeightSum==0 branch).
		bank, err := BuildBank([]Block{
			{ID: "a", Embedding: []float32{-1, 0}},
			{ID: "b", Embedding: []float32{0, -1}},
		}, BuildConfig{BranchingFactor: 2, MaxDepth: 1, MinClusterSize: 2})
		if err != nil {
			t.Fatalf("BuildBank() error = %v", err)
		}
		hidden := []float32{0, 0}
		out, retrievals, stats, err := bank.InjectAdditive(nil, hidden, []float32{1, 0}, nil, InjectionConfig{TopK: 2, Scale: 1, PositiveScoresOnly: true})
		if err != nil {
			t.Fatalf("InjectAdditive() error = %v", err)
		}
		if len(retrievals) != 2 || !stats.Applied || stats.WeightSum != 1 {
			t.Fatalf("stats = %+v retrievals=%d, want uniform fallback applied", stats, len(retrievals))
		}
		// uniform = scale/k = 0.5 per block; sum of the two block embeddings is
		// (-1,-1) so out = 0.5*(-1,-1).
		if len(out) != 2 || !approx32(out[0], -0.5) || !approx32(out[1], -0.5) {
			t.Fatalf("out = %+v, want uniform blend of both blocks", out)
		}
	})

	t.Run("skips clamped blocks in weighted blend", func(t *testing.T) {
		// One block scores positive, one scores negative; PositiveScoresOnly clamps
		// the negative to zero so the weighted blend skips it (the weight==0 continue
		// branch) while still applying the positive block.
		bank, err := BuildBank([]Block{
			{ID: "near", Embedding: []float32{1, 0}},
			{ID: "far", Embedding: []float32{-1, 0}},
		}, BuildConfig{BranchingFactor: 2, MaxDepth: 1, MinClusterSize: 2})
		if err != nil {
			t.Fatalf("BuildBank() error = %v", err)
		}
		out, retrievals, stats, err := bank.InjectAdditive(nil, []float32{0, 0}, []float32{1, 0}, nil, InjectionConfig{TopK: 2, Scale: 1, PositiveScoresOnly: true})
		if err != nil {
			t.Fatalf("InjectAdditive() error = %v", err)
		}
		if len(retrievals) != 2 || !stats.Applied {
			t.Fatalf("retrievals=%d stats=%+v, want both retrieved with positive applied", len(retrievals), stats)
		}
		// Only "near" (score 1) contributes; its weight normalises to scale=1, so
		// out == near.Embedding == (1,0). The clamped "far" block adds nothing.
		if len(out) != 2 || !approx32(out[0], 1) || !approx32(out[1], 0) {
			t.Fatalf("out = %+v, want only the positive block applied", out)
		}
	})
}

// TestMemorypretrain_Bank_InjectAdditive_Bad covers the dimension guards: a hidden
// width that mismatches the bank, and a query width that mismatches the bank.
func TestMemorypretrain_Bank_InjectAdditive_Bad(t *testing.T) {
	bank, err := BuildBank([]Block{{ID: "a", Embedding: []float32{1, 0}}}, BuildConfig{})
	if err != nil {
		t.Fatalf("BuildBank() error = %v", err)
	}
	if _, _, _, err := bank.InjectAdditive(nil, []float32{1}, []float32{1, 0}, nil, InjectionConfig{TopK: 1}); err == nil {
		t.Fatal("InjectAdditive(hidden dim mismatch) error = nil")
	}
	if _, _, _, err := bank.InjectAdditive(nil, []float32{1, 0}, []float32{1}, nil, InjectionConfig{TopK: 1}); err == nil {
		t.Fatal("InjectAdditive(query dim mismatch) error = nil")
	}
}

// TestMemorypretrain_Bank_InjectAdditive_Ugly routes into a synthetic empty leaf:
// with nothing retrieved InjectAdditive applies no memory and passes the hidden
// state through unchanged.
func TestMemorypretrain_Bank_InjectAdditive_Ugly(t *testing.T) {
	bank := &Bank{
		Dimension: 2,
		Blocks:    []Block{{ID: "a", Embedding: []float32{1, 0}}},
		Nodes:     []Node{{ID: 0, Parent: -1, Centroid: []float32{1, 0}, BlockIDs: nil}},
		Root:      0,
		Config:    BuildConfig{BranchingFactor: 2, MaxDepth: 1, MinClusterSize: 1, KMeansIters: 1},
	}
	out, retrievals, stats, err := bank.InjectAdditive(nil, []float32{3, 5}, []float32{1, 0}, nil, InjectionConfig{})
	if err != nil {
		t.Fatalf("InjectAdditive(empty leaf) error = %v", err)
	}
	if len(retrievals) != 0 || stats.Retrieved != 0 {
		t.Fatalf("InjectAdditive(empty leaf) retrievals = %+v stats = %+v, want none", retrievals, stats)
	}
	// With nothing retrieved the hidden state passes through unchanged.
	if len(out) != 2 || out[0] != 3 || out[1] != 5 {
		t.Fatalf("InjectAdditive(empty leaf) out = %+v, want the unchanged hidden state", out)
	}
}

// TestMemorypretrain_cosine_Ugly covers the zero-norm guard in the cosine helper:
// a zero vector against any vector yields zero similarity.
func TestMemorypretrain_cosine_Ugly(t *testing.T) {
	if got := cosine([]float32{0, 0}, []float32{1, 0}); got != 0 {
		t.Fatalf("cosine(zero, unit) = %v, want 0", got)
	}
	if got := cosine([]float32{1, 0}, []float32{0, 0}); got != 0 {
		t.Fatalf("cosine(unit, zero) = %v, want 0", got)
	}
}

// TestMemorypretrain_padClusterIDsWithGenericFallbackInto_Bad covers the
// per-level positive-count guard in the allocation-free padding helper and the
// empty-counts pass-through.
func TestMemorypretrain_padClusterIDsWithGenericFallbackInto_Bad(t *testing.T) {
	// A zero cluster count at a level is rejected.
	if _, err := padClusterIDsWithGenericFallbackInto(nil, []int{0}, []int{0}); err == nil {
		t.Fatal("padClusterIDsWithGenericFallbackInto(zero count) error = nil")
	}
	// Empty cluster counts pass the cluster IDs through unchanged.
	got, err := padClusterIDsWithGenericFallbackInto(nil, []int{2, 5}, nil)
	if err != nil {
		t.Fatalf("padClusterIDsWithGenericFallbackInto(no counts) error = %v", err)
	}
	if len(got) != 2 || got[0] != 2 || got[1] != 5 {
		t.Fatalf("padClusterIDsWithGenericFallbackInto(no counts) = %+v, want the input passed through", got)
	}
}

// TestMemorypretrain_bankDimension_Ugly drives the small unexported helpers in
// memorypretrain.go directly, covering the guard branches the public
// build/retrieve paths never exercise: nil-bank dimension, a cluster-id miss,
// injection-config defaulting, and the empty-block-IDs centroid.
func TestMemorypretrain_bankDimension_Ugly(t *testing.T) {
	if got := bankDimension(nil); got != 0 {
		t.Fatalf("bankDimension(nil) = %d, want 0", got)
	}
	if got := bankDimension(&Bank{Dimension: 5}); got != 5 {
		t.Fatalf("bankDimension(bank) = %d, want 5", got)
	}

	if got := localClusterID([]int{4, 9, 2}, 9); got != 1 {
		t.Fatalf("localClusterID(present) = %d, want index 1", got)
	}
	if got := localClusterID([]int{4, 9, 2}, 7); got != -1 {
		t.Fatalf("localClusterID(absent) = %d, want -1", got)
	}

	// Empty injection config takes both defaults; explicit values pass through.
	if got := normaliseInjectionConfig(InjectionConfig{}); got.TopK != 4 || got.Scale != 1 {
		t.Fatalf("normaliseInjectionConfig(empty) = %+v, want TopK 4 and Scale 1", got)
	}
	if got := normaliseInjectionConfig(InjectionConfig{TopK: 2, Scale: 0.5}); got.TopK != 2 || got.Scale != 0.5 {
		t.Fatalf("normaliseInjectionConfig(explicit) = %+v, want preserved values", got)
	}

	// No block IDs yields a zero centroid of the requested dimension.
	empty := centroidForBlocks(nil, nil, 3)
	if len(empty) != 3 || empty[0] != 0 || empty[1] != 0 || empty[2] != 0 {
		t.Fatalf("centroidForBlocks(no ids) = %+v, want zero vector of dim 3", empty)
	}
	// A populated centroid averages the referenced block embeddings.
	blocks := []Block{{Embedding: []float32{2, 0}}, {Embedding: []float32{0, 4}}}
	avg := centroidForBlocks(blocks, []int{0, 1}, 2)
	if len(avg) != 2 || avg[0] != 1 || avg[1] != 2 {
		t.Fatalf("centroidForBlocks(two blocks) = %+v, want mean [1 2]", avg)
	}
}

func siluTest(value float32) float32 {
	return value / (1 + float32(math.Exp(float64(-value))))
}

func approx32(a, b float32) bool {
	return float32(math.Abs(float64(a-b))) < 1e-5
}
