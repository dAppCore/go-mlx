// SPDX-Licence-Identifier: EUPL-1.2

package memorypretrain

import (
	"testing"

	core "dappco.re/go"
)

func TestBankFile_SaveLoadRoundTrip_Good(t *testing.T) {
	bank, err := BuildBank([]Block{
		{ID: "go", Text: "Go memory planning", Embedding: []float32{1, 0}, Meta: map[string]string{"source": "docs"}},
		{ID: "poem", Text: "winter proof poem", Embedding: []float32{0, 1}, Meta: map[string]string{"source": "creative"}},
	}, BuildConfig{BranchingFactor: 2, MaxDepth: 1, MinClusterSize: 2})
	if err != nil {
		t.Fatalf("BuildBank() error = %v", err)
	}
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

func TestBankFile_Validation_Bad(t *testing.T) {
	if err := SaveBank("", &Bank{}); err == nil {
		t.Fatal("SaveBank(empty path) error = nil")
	}
	if err := SaveBank(core.PathJoin(t.TempDir(), "bank.json"), nil); err == nil {
		t.Fatal("SaveBank(nil bank) error = nil")
	}
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
}

func TestBankFile_LoadRejectsCorruptBank_Ugly(t *testing.T) {
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

func writeFile(t *testing.T, path string, data string) {
	t.Helper()
	if result := core.WriteFile(path, []byte(data), 0o644); !result.OK {
		t.Fatalf("WriteFile(%s): %v", path, result.Value)
	}
}
