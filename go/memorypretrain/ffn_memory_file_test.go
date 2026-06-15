// SPDX-Licence-Identifier: EUPL-1.2

package memorypretrain

import (
	"testing"

	core "dappco.re/go"
)

func TestSaveLoadFFNMemoryBank_RoundTrip_Good(t *testing.T) {
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

func TestFFNMemoryFile_SaveMethodRoundTrip_Good(t *testing.T) {
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

func TestFFNMemoryFile_SaveMethodNilReceiver_Bad(t *testing.T) {
	if err := (*FFNMemoryBank)(nil).Save(core.PathJoin(t.TempDir(), "ffn.json")); err == nil {
		t.Fatal("Save(nil receiver) error = nil")
	}
}

func TestLoadFFNMemoryBank_Validation_Bad(t *testing.T) {
	dir := t.TempDir()
	if err := SaveFFNMemoryBank("", &FFNMemoryBank{}); err == nil {
		t.Fatal("SaveFFNMemoryBank(empty path) error = nil")
	}
	if err := SaveFFNMemoryBank(core.PathJoin(dir, "nil.json"), nil); err == nil {
		t.Fatal("SaveFFNMemoryBank(nil bank) error = nil")
	}
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
}
