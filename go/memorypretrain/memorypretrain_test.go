// SPDX-Licence-Identifier: EUPL-1.2

package memorypretrain

import "testing"

func TestBuildBank_RetrieveRoutesToNearestCluster_Good(t *testing.T) {
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
	scratch := make([]Retrieval, 0, 2)
	reused, err := bank.RetrieveInto(scratch, []float32{0, 1}, 2)
	if err != nil {
		t.Fatalf("RetrieveInto() error = %v", err)
	}
	if len(reused) != 2 || reused[0].BlockID != "poem-1" || cap(reused) != cap(scratch) {
		t.Fatalf("RetrieveInto() = %+v cap=%d, want poem cluster in caller storage cap=%d", reused, cap(reused), cap(scratch))
	}
}

func TestBuildBank_ClonesInputAndValidatesDimensions_Bad(t *testing.T) {
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
}

func TestRetrieve_Validation_Ugly(t *testing.T) {
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
