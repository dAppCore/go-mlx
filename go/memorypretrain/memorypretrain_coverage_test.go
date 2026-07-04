// SPDX-Licence-Identifier: EUPL-1.2

package memorypretrain

import (
	"testing"
)

// TestMemorypretrain_RetrieveInto_NoRootNode_Ugly reaches the missing-root guard
// in RetrieveInto: the bank passes the dimension and positive-k guards but has no
// nodes, so traversal cannot start and the call errors instead of indexing an
// empty node table.
func TestMemorypretrain_RetrieveInto_NoRootNode_Ugly(t *testing.T) {
	bank := &Bank{Dimension: 2}
	if _, err := bank.RetrieveInto(nil, []float32{1, 0}, 1); err == nil {
		t.Fatal("RetrieveInto(no nodes) error = nil, want missing-root failure")
	}
}

// TestMemorypretrain_RetrieveInto_ScoreTieOrdering_Good exercises equal-score
// retrieval: two blocks with identical embeddings land in the same root leaf and
// score identically against the query, so the comparator's score branches fall
// through to the stable BlockIndex ordering. Retrieving both confirms the tie is
// resolved deterministically in ascending block-index order.
func TestMemorypretrain_RetrieveInto_ScoreTieOrdering_Good(t *testing.T) {
	bank, err := BuildBank([]Block{
		{ID: "first", Text: "first", Embedding: []float32{1, 0}},
		{ID: "second", Text: "second", Embedding: []float32{1, 0}},
	}, BuildConfig{BranchingFactor: 2, MaxDepth: 1, MinClusterSize: 2, KMeansIters: 1})
	if err != nil {
		t.Fatalf("BuildBank() error = %v", err)
	}
	got, err := bank.Retrieve([]float32{1, 0}, 2)
	if err != nil {
		t.Fatalf("Retrieve() error = %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("Retrieve() = %+v, want both tied blocks", got)
	}
	if got[0].Score != got[1].Score {
		t.Fatalf("Retrieve() scores = %v, %v, want a tie", got[0].Score, got[1].Score)
	}
	if got[0].BlockIndex >= got[1].BlockIndex {
		t.Fatalf("Retrieve() block order = %d, %d, want ascending block-index tie-break", got[0].BlockIndex, got[1].BlockIndex)
	}
}
