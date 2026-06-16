// SPDX-Licence-Identifier: EUPL-1.2

package memorypretrain

import (
	"testing"

	core "dappco.re/go"
)

func sampleFFNMemoryBank(t *testing.T) *FFNMemoryBank {
	t.Helper()
	bank, err := NewFFNMemoryBank(FFNMemoryConfig{
		HiddenSize:       2,
		Layers:           2,
		MemoryLevels:     []string{"1", "2"},
		FFNMemoryTokens:  []int{1, 2},
		NumClusters:      []int{2, 3},
		AddedGenericSize: 1,
	})
	if err != nil {
		t.Fatalf("NewFFNMemoryBank() error = %v", err)
	}
	return bank
}

// TestFfnMemoryFile_FFNMemoryBank_Save_Good round-trips a bank through the
// (*FFNMemoryBank).Save method, creating the parent directory and confirming a
// learned W3 value survives the reload.
func TestFfnMemoryFile_FFNMemoryBank_Save_Good(t *testing.T) {
	bank, err := NewFFNMemoryBank(FFNMemoryConfig{
		HiddenSize:       2,
		Layers:           1,
		MemoryLevels:     []string{"1"},
		FFNMemoryTokens:  []int{1},
		NumClusters:      []int{2},
		AddedGenericSize: 1,
	})
	if err != nil {
		t.Fatalf("NewFFNMemoryBank() error = %v", err)
	}
	bank.Layers[0].Levels[0].W3[0] = 0.75
	path := core.PathJoin(t.TempDir(), "memory", "ffn.json")
	if err := bank.Save(path); err != nil {
		t.Fatalf("Save() error = %v", err)
	}
	loaded, err := LoadFFNMemoryBank(path)
	if err != nil {
		t.Fatalf("LoadFFNMemoryBank() error = %v", err)
	}
	if loaded.HiddenSize != 2 || len(loaded.Layers) != 1 || loaded.Layers[0].Levels[0].W3[0] != 0.75 {
		t.Fatalf("loaded = %+v, want method-saved bank round-tripped", loaded)
	}
}

// TestFfnMemoryFile_FFNMemoryBank_Save_Bad rejects an empty path and a nil
// receiver, both of which the Save method delegates to SaveFFNMemoryBank.
func TestFfnMemoryFile_FFNMemoryBank_Save_Bad(t *testing.T) {
	if err := (*FFNMemoryBank)(nil).Save(core.PathJoin(t.TempDir(), "ffn.json")); err == nil {
		t.Fatal("Save(nil receiver) error = nil")
	}
	if err := sampleFFNMemoryBank(t).Save(""); err == nil {
		t.Fatal("Save(empty path) error = nil")
	}
}

// TestFfnMemoryFile_FFNMemoryBank_Save_Ugly drives Save with a structurally
// invalid bank (hidden size set, no layers) so validateFFNMemoryBank rejects it
// before any file is written.
func TestFfnMemoryFile_FFNMemoryBank_Save_Ugly(t *testing.T) {
	bank := &FFNMemoryBank{HiddenSize: 2}
	path := core.PathJoin(t.TempDir(), "ffn.json")
	if err := bank.Save(path); err == nil {
		t.Fatal("Save(invalid bank) error = nil")
	}
	if core.ReadFile(path).OK {
		t.Fatal("Save() wrote a file for an invalid bank")
	}
}

// TestFfnMemoryFile_SaveFFNMemoryBank_Good persists a multi-layer bank with
// learned W3 edits, creating the parent directory, and confirms the reload
// preserves both shape and the learned values.
func TestFfnMemoryFile_SaveFFNMemoryBank_Good(t *testing.T) {
	bank := sampleFFNMemoryBank(t)
	bank.Layers[1].Levels[0].W3[0] = 0.25
	bank.Layers[1].Levels[1].W3[3] = -0.5
	path := core.PathJoin(t.TempDir(), "memory", "ffn.json")
	if err := SaveFFNMemoryBank(path, bank); err != nil {
		t.Fatalf("SaveFFNMemoryBank() error = %v", err)
	}
	loaded, err := LoadFFNMemoryBank(path)
	if err != nil {
		t.Fatalf("LoadFFNMemoryBank() error = %v", err)
	}
	if loaded.HiddenSize != 2 || len(loaded.Layers) != 2 || len(loaded.Layers[1].Levels) != 2 {
		t.Fatalf("loaded = %+v, want same shape", loaded)
	}
	if loaded.Layers[1].Levels[0].W3[0] != 0.25 || loaded.Layers[1].Levels[1].W3[3] != -0.5 {
		t.Fatalf("loaded W3 values = %+v %+v, want learned values", loaded.Layers[1].Levels[0].W3[:1], loaded.Layers[1].Levels[1].W3[:4])
	}
	out, _, stats, err := loaded.AddGenericToFFNOutput(nil, []float32{1, 2}, []float32{3, 4}, 1)
	if err != nil {
		t.Fatalf("loaded AddGenericToFFNOutput() error = %v", err)
	}
	if len(out) != 2 || !stats.Applied {
		t.Fatalf("loaded output=%+v stats=%+v, want usable memory bank", out, stats)
	}
}

// TestFfnMemoryFile_SaveFFNMemoryBank_Bad covers the SaveFFNMemoryBank entry
// guards: an empty path and a nil bank both fail before any write.
func TestFfnMemoryFile_SaveFFNMemoryBank_Bad(t *testing.T) {
	if err := SaveFFNMemoryBank("", &FFNMemoryBank{}); err == nil {
		t.Fatal("SaveFFNMemoryBank(empty path) error = nil")
	}
	if err := SaveFFNMemoryBank(core.PathJoin(t.TempDir(), "nil.json"), nil); err == nil {
		t.Fatal("SaveFFNMemoryBank(nil bank) error = nil")
	}
}

// TestFfnMemoryFile_SaveFFNMemoryBank_Ugly hands SaveFFNMemoryBank a non-nil
// bank that fails deep validation (hidden size set but no config/layers), so the
// validate path inside SaveFFNMemoryBank rejects it rather than the cheap nil
// guard.
func TestFfnMemoryFile_SaveFFNMemoryBank_Ugly(t *testing.T) {
	bank := &FFNMemoryBank{HiddenSize: 2}
	path := core.PathJoin(t.TempDir(), "ffn.json")
	if err := SaveFFNMemoryBank(path, bank); err == nil {
		t.Fatal("SaveFFNMemoryBank(invalid bank) error = nil")
	}
	if core.ReadFile(path).OK {
		t.Fatal("SaveFFNMemoryBank() wrote a file for an invalid bank")
	}
}

// TestFfnMemoryFile_LoadFFNMemoryBank_Good saves a bank then reloads it through
// LoadFFNMemoryBank, confirming the reloaded bank applies generic memory.
func TestFfnMemoryFile_LoadFFNMemoryBank_Good(t *testing.T) {
	bank := sampleFFNMemoryBank(t)
	path := core.PathJoin(t.TempDir(), "ffn.json")
	if err := SaveFFNMemoryBank(path, bank); err != nil {
		t.Fatalf("SaveFFNMemoryBank() error = %v", err)
	}
	loaded, err := LoadFFNMemoryBank(path)
	if err != nil {
		t.Fatalf("LoadFFNMemoryBank() error = %v", err)
	}
	out, _, stats, err := loaded.AddGenericToFFNOutput(nil, []float32{1, 2}, []float32{3, 4}, 0)
	if err != nil {
		t.Fatalf("LoadFFNMemoryBank() reload AddGenericToFFNOutput error = %v", err)
	}
	if len(out) != 2 || !stats.Applied {
		t.Fatalf("LoadFFNMemoryBank() reload output=%+v stats=%+v, want usable bank", out, stats)
	}
}

// TestFfnMemoryFile_LoadFFNMemoryBank_Bad covers the LoadFFNMemoryBank entry
// guards and envelope rejections: empty path, wrong kind, and unsupported
// version.
func TestFfnMemoryFile_LoadFFNMemoryBank_Bad(t *testing.T) {
	dir := t.TempDir()
	if _, err := LoadFFNMemoryBank(""); err == nil {
		t.Fatal("LoadFFNMemoryBank(empty path) error = nil")
	}
	writeFile(t, core.PathJoin(dir, "bad-kind.json"), `{"version":1,"kind":"bad","bank":{}}`)
	if _, err := LoadFFNMemoryBank(core.PathJoin(dir, "bad-kind.json")); err == nil {
		t.Fatal("LoadFFNMemoryBank(bad kind) error = nil")
	}
	writeFile(t, core.PathJoin(dir, "bad-version.json"), `{"version":99,"kind":"go-mlx/memorypretrain-ffn-memory","bank":{}}`)
	if _, err := LoadFFNMemoryBank(core.PathJoin(dir, "bad-version.json")); err == nil {
		t.Fatal("LoadFFNMemoryBank(bad version) error = nil")
	}
	if _, err := LoadFFNMemoryBank(core.PathJoin(dir, "missing.json")); err == nil {
		t.Fatal("LoadFFNMemoryBank(missing file) error = nil")
	}
}

// TestFfnMemoryFile_LoadFFNMemoryBank_Ugly reloads an envelope whose JSON parses
// cleanly but whose embedded memory table fails structural validation (a level
// with empty W2/W3 weights), and an envelope whose body is not valid JSON.
func TestFfnMemoryFile_LoadFFNMemoryBank_Ugly(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, core.PathJoin(dir, "bad-shape.json"), `{
  "version": 1,
  "kind": "go-mlx/memorypretrain-ffn-memory",
  "bank": {
    "hidden_size": 2,
    "config": {
      "hidden_size": 2,
      "layers": 1,
      "memory_levels": ["1"],
      "ffn_memory_tokens": [1],
      "num_clusters": [2],
      "added_generic_size": 1
    },
    "layers": [
      {
        "layer": 0,
        "levels": [
          {"name": "1", "num_clusters": 2, "added_generic_size": 1, "memory_tokens": 1, "w1": [1], "w2": [], "w3": []}
        ]
      }
    ]
  }
}`)
	if _, err := LoadFFNMemoryBank(core.PathJoin(dir, "bad-shape.json")); err == nil {
		t.Fatal("LoadFFNMemoryBank(bad shape) error = nil")
	}
	writeFile(t, core.PathJoin(dir, "bad-json.json"), `{`)
	if _, err := LoadFFNMemoryBank(core.PathJoin(dir, "bad-json.json")); err == nil {
		t.Fatal("LoadFFNMemoryBank(bad json) error = nil")
	}
}

// TestFfnMemoryFile_validateFFNMemoryLayer_Ugly mutates a freshly-allocated
// bank's single layer to hit each layer/level guard the round-trip never trips:
// layer id, level count, level name, cluster mismatch, generic-size mismatch,
// and a non-positive token count. validateFFNMemoryLayer is reached for each
// layer through validateFFNMemoryBank.
func TestFfnMemoryFile_validateFFNMemoryLayer_Ugly(t *testing.T) {
	cfg := FFNMemoryConfig{
		HiddenSize:       2,
		Layers:           1,
		MemoryLevels:     []string{"1"},
		FFNMemoryTokens:  []int{1},
		NumClusters:      []int{2},
		AddedGenericSize: 1,
	}
	build := func() *FFNMemoryBank {
		bank, err := NewFFNMemoryBank(cfg)
		if err != nil {
			t.Fatalf("NewFFNMemoryBank() error = %v", err)
		}
		return bank
	}
	if err := validateFFNMemoryBank(build()); err != nil {
		t.Fatalf("validateFFNMemoryBank(baseline) error = %v, want nil", err)
	}
	cases := []struct {
		name   string
		mutate func(b *FFNMemoryBank)
	}{
		{"layer id mismatch", func(b *FFNMemoryBank) { b.Layers[0].Layer = 7 }},
		{"level count mismatch", func(b *FFNMemoryBank) { b.Layers[0].Levels = b.Layers[0].Levels[:0] }},
		{"level name mismatch", func(b *FFNMemoryBank) { b.Layers[0].Levels[0].Name = "wrong" }},
		{"level cluster mismatch", func(b *FFNMemoryBank) { b.Layers[0].Levels[0].NumClusters = 9 }},
		{"level generic mismatch", func(b *FFNMemoryBank) { b.Layers[0].Levels[0].AddedGenericSize = 9 }},
		{"non-positive memory tokens", func(b *FFNMemoryBank) { b.Layers[0].Levels[0].MemoryTokens = 0 }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			b := build()
			tc.mutate(b)
			if err := validateFFNMemoryBank(b); err == nil {
				t.Fatalf("validateFFNMemoryBank(%s) error = nil, want failure", tc.name)
			}
		})
	}
}

// TestFfnMemoryFile_validateFFNMemoryBank_Bad covers the bank-level guards: nil
// bank, non-positive hidden size, config hidden-size mismatch, and a layer count
// that disagrees with the config.
func TestFfnMemoryFile_validateFFNMemoryBank_Bad(t *testing.T) {
	if err := validateFFNMemoryBank(nil); err == nil {
		t.Fatal("validateFFNMemoryBank(nil) error = nil")
	}
	if err := validateFFNMemoryBank(&FFNMemoryBank{HiddenSize: 0}); err == nil {
		t.Fatal("validateFFNMemoryBank(zero hidden size) error = nil")
	}
	mismatch := &FFNMemoryBank{
		HiddenSize: 2,
		Config:     FFNMemoryConfig{HiddenSize: 4, Layers: 1, MemoryLevels: []string{"1"}, FFNMemoryTokens: []int{1}, NumClusters: []int{2}, AddedGenericSize: 1},
	}
	if err := validateFFNMemoryBank(mismatch); err == nil {
		t.Fatal("validateFFNMemoryBank(config hidden mismatch) error = nil")
	}
	layerMismatch := &FFNMemoryBank{
		HiddenSize: 2,
		Config:     FFNMemoryConfig{HiddenSize: 2, Layers: 3, MemoryLevels: []string{"1"}, FFNMemoryTokens: []int{1}, NumClusters: []int{2}, AddedGenericSize: 1},
	}
	if err := validateFFNMemoryBank(layerMismatch); err == nil {
		t.Fatal("validateFFNMemoryBank(layer count mismatch) error = nil")
	}
	// Matching hidden sizes but an internally inconsistent config (level/token
	// length mismatch) surfaces the deeper config-validation error.
	badConfig := &FFNMemoryBank{
		HiddenSize: 2,
		Config:     FFNMemoryConfig{HiddenSize: 2, Layers: 1, MemoryLevels: []string{"1", "2"}, FFNMemoryTokens: []int{1}, NumClusters: []int{2}, AddedGenericSize: 1},
	}
	if err := validateFFNMemoryBank(badConfig); err == nil {
		t.Fatal("validateFFNMemoryBank(invalid embedded config) error = nil")
	}
}
