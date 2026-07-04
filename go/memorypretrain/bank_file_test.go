// SPDX-Licence-Identifier: EUPL-1.2

package memorypretrain

import (
	"testing"

	core "dappco.re/go"
)

func sampleBank(t *testing.T) *Bank {
	t.Helper()
	bank, err := BuildBank([]Block{
		{ID: "go", Text: "Go memory planning", Embedding: []float32{1, 0}, Meta: map[string]string{"source": "docs"}},
		{ID: "poem", Text: "winter proof poem", Embedding: []float32{0, 1}, Meta: map[string]string{"source": "creative"}},
	}, BuildConfig{BranchingFactor: 2, MaxDepth: 1, MinClusterSize: 2})
	if err != nil {
		t.Fatalf("BuildBank() error = %v", err)
	}
	return bank
}

// TestBankFile_Bank_Save_Good round-trips a built bank through the (*Bank).Save
// method: a nested target path is created, the file reloads, and the routed
// retrieval plus block metadata survive the round trip.
func TestBankFile_Bank_Save_Good(t *testing.T) {
	bank := sampleBank(t)
	path := core.PathJoin(t.TempDir(), "nested", "bank.json")
	if err := bank.Save(path); err != nil {
		t.Fatalf("Save() error = %v", err)
	}
	loaded, err := LoadBank(path)
	if err != nil {
		t.Fatalf("LoadBank() error = %v", err)
	}
	if loaded.Dimension != bank.Dimension || len(loaded.Blocks) != len(bank.Blocks) || len(loaded.Nodes) != len(bank.Nodes) {
		t.Fatalf("loaded bank = %+v, want round-tripped structure", loaded)
	}
	got, err := loaded.Retrieve([]float32{1, 0}, 1)
	if err != nil {
		t.Fatalf("Retrieve() error = %v", err)
	}
	if len(got) != 1 || got[0].BlockID != "go" || got[0].Text != "Go memory planning" {
		t.Fatalf("Retrieve() = %+v, want saved Go block", got)
	}
	if loaded.Blocks[0].Meta["source"] != "docs" {
		t.Fatalf("loaded meta = %+v, want saved metadata", loaded.Blocks[0].Meta)
	}
}

// TestBankFile_Bank_Save_Bad checks the method-side guards: Save delegates to
// SaveBank, so an empty path and a nil-bank receiver both surface errors.
func TestBankFile_Bank_Save_Bad(t *testing.T) {
	bank := sampleBank(t)
	if err := bank.Save(""); err == nil {
		t.Fatal("Save(empty path) error = nil")
	}
	var nilBank *Bank
	if err := nilBank.Save(core.PathJoin(t.TempDir(), "bank.json")); err == nil {
		t.Fatal("Save() on nil bank error = nil")
	}
}

// TestBankFile_Bank_Save_Ugly drives Save with a structurally invalid bank: the
// dimension is set but the node table is empty, so validateBank rejects it
// before any file is written.
func TestBankFile_Bank_Save_Ugly(t *testing.T) {
	bank := &Bank{Dimension: 2, Blocks: []Block{{Embedding: []float32{1, 0}}}}
	path := core.PathJoin(t.TempDir(), "bank.json")
	if err := bank.Save(path); err == nil {
		t.Fatal("Save(bank with no nodes) error = nil")
	}
	if core.ReadFile(path).OK {
		t.Fatal("Save() wrote a file for an invalid bank")
	}
}

// TestBankFile_SaveBank_Good writes a valid bank with SaveBank, creating the
// parent directory, and confirms the persisted envelope reloads.
func TestBankFile_SaveBank_Good(t *testing.T) {
	bank := sampleBank(t)
	path := core.PathJoin(t.TempDir(), "deep", "tree", "bank.json")
	if err := SaveBank(path, bank); err != nil {
		t.Fatalf("SaveBank() error = %v", err)
	}
	loaded, err := LoadBank(path)
	if err != nil {
		t.Fatalf("LoadBank() error = %v", err)
	}
	if loaded.Dimension != bank.Dimension {
		t.Fatalf("SaveBank() round trip dimension = %d, want %d", loaded.Dimension, bank.Dimension)
	}
}

// TestBankFile_SaveBank_Bad covers the SaveBank entry guards: an empty path and
// a nil bank both fail before any write.
func TestBankFile_SaveBank_Bad(t *testing.T) {
	if err := SaveBank("", sampleBank(t)); err == nil {
		t.Fatal("SaveBank(empty path) error = nil")
	}
	if err := SaveBank(core.PathJoin(t.TempDir(), "bank.json"), nil); err == nil {
		t.Fatal("SaveBank(nil bank) error = nil")
	}
}

// TestBankFile_SaveBank_Ugly hands SaveBank a bank that passes the nil check but
// fails deep validation (root index out of range), exercising the validate path
// inside SaveBank rather than the cheap top-level guards.
func TestBankFile_SaveBank_Ugly(t *testing.T) {
	bank := &Bank{
		Dimension: 2,
		Blocks:    []Block{{Embedding: []float32{1, 0}}},
		Nodes:     []Node{{ID: 0, Parent: -1, Centroid: []float32{1, 0}, BlockIDs: []int{0}}},
		Root:      9,
	}
	path := core.PathJoin(t.TempDir(), "bank.json")
	if err := SaveBank(path, bank); err == nil {
		t.Fatal("SaveBank(root out of range) error = nil")
	}
	if core.ReadFile(path).OK {
		t.Fatal("SaveBank() wrote a file for an invalid bank")
	}
}

// TestBankFile_LoadBank_Good builds, saves, and reloads a bank, asserting the
// reloaded structure routes a query to the originally stored block.
func TestBankFile_LoadBank_Good(t *testing.T) {
	bank := sampleBank(t)
	path := core.PathJoin(t.TempDir(), "bank.json")
	if err := SaveBank(path, bank); err != nil {
		t.Fatalf("SaveBank() error = %v", err)
	}
	loaded, err := LoadBank(path)
	if err != nil {
		t.Fatalf("LoadBank() error = %v", err)
	}
	got, err := loaded.Retrieve([]float32{0, 1}, 1)
	if err != nil {
		t.Fatalf("Retrieve() error = %v", err)
	}
	if len(got) != 1 || got[0].BlockID != "poem" {
		t.Fatalf("LoadBank() retrieve = %+v, want poem block", got)
	}
}

// TestBankFile_LoadBank_Bad covers the LoadBank entry guards and envelope
// rejections: empty path, wrong kind, unsupported version, and unparsable JSON.
func TestBankFile_LoadBank_Bad(t *testing.T) {
	if _, err := LoadBank(""); err == nil {
		t.Fatal("LoadBank(empty path) error = nil")
	}
	dir := t.TempDir()
	writeFile(t, core.PathJoin(dir, "bad-kind.json"), `{"version":1,"kind":"other","bank":{}}`)
	if _, err := LoadBank(core.PathJoin(dir, "bad-kind.json")); err == nil {
		t.Fatal("LoadBank(bad kind) error = nil")
	}
	writeFile(t, core.PathJoin(dir, "bad-version.json"), `{"version":99,"kind":"go-mlx/memorypretrain-bank","bank":{}}`)
	if _, err := LoadBank(core.PathJoin(dir, "bad-version.json")); err == nil {
		t.Fatal("LoadBank(bad version) error = nil")
	}
	writeFile(t, core.PathJoin(dir, "bad-json.json"), `{`)
	if _, err := LoadBank(core.PathJoin(dir, "bad-json.json")); err == nil {
		t.Fatal("LoadBank(bad json) error = nil")
	}
	if _, err := LoadBank(core.PathJoin(dir, "missing.json")); err == nil {
		t.Fatal("LoadBank(missing file) error = nil")
	}
}

// TestBankFile_LoadBank_Ugly reloads envelopes whose JSON parses cleanly but
// whose embedded bank fails structural validation — an out-of-range root, a
// mismatched node id, and a self-parenting root node.
func TestBankFile_LoadBank_Ugly(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, core.PathJoin(dir, "bad-root.json"), `{
  "version": 1,
  "kind": "go-mlx/memorypretrain-bank",
  "bank": {
    "dimension": 2,
    "blocks": [{"id":"a","embedding":[1,0]}],
    "nodes": [{"id":0,"depth":0,"centroid":[1,0],"block_ids":[0]}],
    "root": 7,
    "config": {"branching_factor":2,"max_depth":1,"min_cluster_size":2,"kmeans_iters":1}
  }
}`)
	if _, err := LoadBank(core.PathJoin(dir, "bad-root.json")); err == nil {
		t.Fatal("LoadBank(bad root) error = nil")
	}
	writeFile(t, core.PathJoin(dir, "bad-node.json"), `{
  "version": 1,
  "kind": "go-mlx/memorypretrain-bank",
  "bank": {
    "dimension": 2,
    "blocks": [{"id":"a","embedding":[1,0]}],
    "nodes": [{"id":4,"depth":0,"centroid":[1,0],"block_ids":[0]}],
    "root": 0,
    "config": {"branching_factor":2,"max_depth":1,"min_cluster_size":2,"kmeans_iters":1}
  }
}`)
	if _, err := LoadBank(core.PathJoin(dir, "bad-node.json")); err == nil {
		t.Fatal("LoadBank(bad node id) error = nil")
	}
	writeFile(t, core.PathJoin(dir, "bad-parent.json"), `{
  "version": 1,
  "kind": "go-mlx/memorypretrain-bank",
  "bank": {
    "dimension": 2,
    "blocks": [{"id":"a","embedding":[1,0]}],
    "nodes": [{"id":0,"parent":0,"depth":0,"centroid":[1,0],"block_ids":[0]}],
    "root": 0,
    "config": {"branching_factor":2,"max_depth":1,"min_cluster_size":2,"kmeans_iters":1}
  }
}`)
	if _, err := LoadBank(core.PathJoin(dir, "bad-parent.json")); err == nil {
		t.Fatal("LoadBank(bad root parent) error = nil")
	}
}

// TestValidateBankNode_Branches_Ugly drives every per-node guard in
// validateBankNode directly with synthetic Bank structs, covering the centroid
// dimension, child-index, block-index, self-parent, and parent-range branches
// that the file round-trip happy path never reaches.
func TestValidateBankNode_Branches_Ugly(t *testing.T) {
	base := func() *Bank {
		return &Bank{
			Dimension: 2,
			Blocks:    []Block{{ID: "a", Embedding: []float32{1, 0}}},
			Nodes: []Node{
				{ID: 0, Parent: -1, Centroid: []float32{1, 0}, Children: []int{1}, BlockIDs: nil},
				{ID: 1, Parent: 0, Centroid: []float32{0, 1}, BlockIDs: []int{0}},
			},
			Root:   0,
			Config: BuildConfig{BranchingFactor: 2, MaxDepth: 1, MinClusterSize: 1, KMeansIters: 1},
		}
	}
	// The baseline is valid so each mutation isolates a single branch.
	if err := validateBank(base()); err != nil {
		t.Fatalf("validateBank(baseline) error = %v, want nil", err)
	}

	cases := []struct {
		name   string
		mutate func(b *Bank)
	}{
		{"node id mismatch", func(b *Bank) { b.Nodes[1].ID = 9 }},
		{"non-root self parent", func(b *Bank) { b.Nodes[1].Parent = 1 }},
		{"parent out of range", func(b *Bank) { b.Nodes[1].Parent = 9 }},
		{"centroid dimension mismatch", func(b *Bank) { b.Nodes[1].Centroid = []float32{1} }},
		{"child out of range", func(b *Bank) { b.Nodes[0].Children = []int{9} }},
		{"block id out of range", func(b *Bank) { b.Nodes[1].BlockIDs = []int{9} }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			b := base()
			tc.mutate(b)
			if err := validateBank(b); err == nil {
				t.Fatalf("validateBank(%s) error = nil, want failure", tc.name)
			}
		})
	}
}

// TestValidateBank_TopLevelGuards_Bad covers the bank-level guards reached
// before per-node validation: nil bank, zero dimension, dimension/block
// mismatch, empty nodes, and out-of-range root.
func TestValidateBank_TopLevelGuards_Bad(t *testing.T) {
	if err := validateBank(nil); err == nil {
		t.Fatal("validateBank(nil) error = nil")
	}
	if err := validateBank(&Bank{Dimension: 0}); err == nil {
		t.Fatal("validateBank(zero dimension) error = nil")
	}
	if err := validateBank(&Bank{Dimension: 3, Blocks: []Block{{Embedding: []float32{1, 0}}}}); err == nil {
		t.Fatal("validateBank(dimension/block mismatch) error = nil")
	}
	noNodes := &Bank{Dimension: 2, Blocks: []Block{{Embedding: []float32{1, 0}}}}
	if err := validateBank(noNodes); err == nil {
		t.Fatal("validateBank(no nodes) error = nil")
	}
	badRoot := &Bank{
		Dimension: 2,
		Blocks:    []Block{{Embedding: []float32{1, 0}}},
		Nodes:     []Node{{ID: 0, Parent: -1, Centroid: []float32{1, 0}, BlockIDs: []int{0}}},
		Root:      9,
	}
	if err := validateBank(badRoot); err == nil {
		t.Fatal("validateBank(root out of range) error = nil")
	}
}

// TestMemoryPretrainResultError_Branches_Ugly covers the result-to-error helper:
// an OK result yields nil, a failed result carrying an error returns it
// verbatim, and a failed result carrying a non-error value falls back to the
// sentinel.
func TestMemoryPretrainResultError_Branches_Ugly(t *testing.T) {
	if err := memoryPretrainResultError(core.Result{OK: true}); err != nil {
		t.Fatalf("memoryPretrainResultError(ok) = %v, want nil", err)
	}
	wrapped := core.NewError("boom")
	if err := memoryPretrainResultError(core.Result{OK: false, Value: wrapped}); err != wrapped {
		t.Fatalf("memoryPretrainResultError(error value) = %v, want the wrapped error", err)
	}
	if err := memoryPretrainResultError(core.Result{OK: false, Value: "not-an-error"}); err != errBankFileCoreResult {
		t.Fatalf("memoryPretrainResultError(non-error value) = %v, want the sentinel", err)
	}
}

func writeFile(t *testing.T, path string, data string) {
	t.Helper()
	if result := core.WriteFile(path, []byte(data), 0o644); !result.OK {
		t.Fatalf("WriteFile(%s): %v", path, result.Value)
	}
}
