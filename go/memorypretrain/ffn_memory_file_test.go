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

// TestNormaliseFFNMemoryConfig_Defaults_Good proves the normaliser fills every
// empty slice with the upstream defaults and always forces ZeroInitialiseW3.
func TestNormaliseFFNMemoryConfig_Defaults_Good(t *testing.T) {
	got := normaliseFFNMemoryConfig(FFNMemoryConfig{HiddenSize: 4, Layers: 1})
	if len(got.MemoryLevels) != 4 || got.MemoryLevels[0] != "1" {
		t.Fatalf("MemoryLevels = %+v, want the four default level names", got.MemoryLevels)
	}
	if len(got.FFNMemoryTokens) != 4 || got.FFNMemoryTokens[0] != 8 {
		t.Fatalf("FFNMemoryTokens = %+v, want default token counts", got.FFNMemoryTokens)
	}
	if len(got.NumClusters) != 4 || got.NumClusters[0] != 256 {
		t.Fatalf("NumClusters = %+v, want default cluster counts", got.NumClusters)
	}
	if got.AddedGenericSize != 1 {
		t.Fatalf("AddedGenericSize = %d, want default 1", got.AddedGenericSize)
	}
	if !got.ZeroInitialiseW3 {
		t.Fatal("ZeroInitialiseW3 = false, want always forced true")
	}
	// Explicit values pass through untouched (only the empties are filled).
	custom := normaliseFFNMemoryConfig(FFNMemoryConfig{
		HiddenSize:       4,
		Layers:           1,
		MemoryLevels:     []string{"only"},
		FFNMemoryTokens:  []int{3},
		NumClusters:      []int{5},
		AddedGenericSize: 2,
	})
	if len(custom.MemoryLevels) != 1 || custom.FFNMemoryTokens[0] != 3 || custom.NumClusters[0] != 5 || custom.AddedGenericSize != 2 {
		t.Fatalf("custom config = %+v, want explicit values preserved", custom)
	}
}

// TestNewFFNMemoryBank_InvalidConfig_Bad proves NewFFNMemoryBank surfaces the
// config validation error rather than allocating a malformed bank.
func TestNewFFNMemoryBank_InvalidConfig_Bad(t *testing.T) {
	// Zero layers fails validation after normalisation (which only fills empty
	// slices, never the layer count).
	if _, err := NewFFNMemoryBank(FFNMemoryConfig{HiddenSize: 2}); err == nil {
		t.Fatal("NewFFNMemoryBank(zero layers) error = nil")
	}
	// Mismatched explicit level/token/cluster lengths also fail.
	if _, err := NewFFNMemoryBank(FFNMemoryConfig{
		HiddenSize:      2,
		Layers:          1,
		MemoryLevels:    []string{"1", "2"},
		FFNMemoryTokens: []int{1},
		NumClusters:     []int{2},
	}); err == nil {
		t.Fatal("NewFFNMemoryBank(mismatched level lengths) error = nil")
	}
}

// TestValidateFFNMemoryConfig_Branches_Bad drives each guard in
// validateFFNMemoryConfig directly: hidden size, layers, mismatched
// level/token/cluster lengths, blank level name, non-positive token count, and
// non-positive cluster count.
func TestValidateFFNMemoryConfig_Branches_Bad(t *testing.T) {
	good := FFNMemoryConfig{
		HiddenSize:       2,
		Layers:           1,
		MemoryLevels:     []string{"a", "b"},
		FFNMemoryTokens:  []int{1, 1},
		NumClusters:      []int{2, 2},
		AddedGenericSize: 1,
	}
	if err := validateFFNMemoryConfig(good); err != nil {
		t.Fatalf("validateFFNMemoryConfig(good) error = %v, want nil", err)
	}
	cases := []struct {
		name   string
		mutate func(cfg *FFNMemoryConfig)
	}{
		{"zero hidden size", func(cfg *FFNMemoryConfig) { cfg.HiddenSize = 0 }},
		{"zero layers", func(cfg *FFNMemoryConfig) { cfg.Layers = 0 }},
		{"mismatched token length", func(cfg *FFNMemoryConfig) { cfg.FFNMemoryTokens = []int{1} }},
		{"mismatched cluster length", func(cfg *FFNMemoryConfig) { cfg.NumClusters = []int{2} }},
		{"blank level name", func(cfg *FFNMemoryConfig) { cfg.MemoryLevels = []string{"a", ""} }},
		{"non-positive token count", func(cfg *FFNMemoryConfig) { cfg.FFNMemoryTokens = []int{1, 0} }},
		{"non-positive cluster count", func(cfg *FFNMemoryConfig) { cfg.NumClusters = []int{2, 0} }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := good
			tc.mutate(&cfg)
			if err := validateFFNMemoryConfig(cfg); err == nil {
				t.Fatalf("validateFFNMemoryConfig(%s) error = nil, want failure", tc.name)
			}
		})
	}
}

// TestValidateFFNMemoryLayer_Branches_Ugly mutates a freshly-allocated bank's
// single layer to hit each layer/level guard the round-trip never trips: layer
// id, level count, level name, cluster mismatch, generic-size mismatch, and a
// non-positive token count.
func TestValidateFFNMemoryLayer_Branches_Ugly(t *testing.T) {
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

// TestValidateFFNMemoryBank_TopLevelGuards_Bad covers the bank-level guards:
// nil bank, non-positive hidden size, config hidden-size mismatch, and a layer
// count that disagrees with the config.
func TestValidateFFNMemoryBank_TopLevelGuards_Bad(t *testing.T) {
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

// TestLevelHiddenDims_ZeroTokens_Ugly covers the zero-token guard in
// levelHiddenStride and levelHiddenSize, which the populated round-trip path
// never reaches.
func TestLevelHiddenDims_ZeroTokens_Ugly(t *testing.T) {
	zero := &FFNMemoryLevelWeight{Name: "z", NumClusters: 2, AddedGenericSize: 1, MemoryTokens: 0}
	if got := levelHiddenStride(zero); got != 0 {
		t.Fatalf("levelHiddenStride(zero tokens) = %d, want 0", got)
	}
	if got := levelHiddenSize(zero); got != 0 {
		t.Fatalf("levelHiddenSize(zero tokens) = %d, want 0", got)
	}
}

// TestValidateFFNMemoryLevel_WeightLengths_Bad isolates the W2 and W3
// length-mismatch branches. The round-trip and bad-shape fixtures trip the W1
// guard first, so this calls the level validator directly with a correct W1 but
// a short W2 (then W3) to reach the later checks.
func TestValidateFFNMemoryLevel_WeightLengths_Bad(t *testing.T) {
	const hiddenSize, tokens = 2, 1
	total := 3 // NumClusters 2 + AddedGenericSize 1
	w12Len := total * hiddenSize * tokens
	w3Len := total * tokens * hiddenSize
	good := func() *FFNMemoryLevelWeight {
		return &FFNMemoryLevelWeight{
			Name:             "1",
			NumClusters:      2,
			AddedGenericSize: 1,
			MemoryTokens:     tokens,
			W1:               make([]float32, w12Len),
			W2:               make([]float32, w12Len),
			W3:               make([]float32, w3Len),
		}
	}
	if err := validateFFNMemoryLevel(good(), hiddenSize, 0); err != nil {
		t.Fatalf("validateFFNMemoryLevel(good) error = %v, want nil", err)
	}
	shortW2 := good()
	shortW2.W2 = make([]float32, w12Len-1)
	if err := validateFFNMemoryLevel(shortW2, hiddenSize, 0); err == nil {
		t.Fatal("validateFFNMemoryLevel(short W2) error = nil")
	}
	shortW3 := good()
	shortW3.W3 = make([]float32, w3Len-1)
	if err := validateFFNMemoryLevel(shortW3, hiddenSize, 0); err == nil {
		t.Fatal("validateFFNMemoryLevel(short W3) error = nil")
	}
	// The cluster-id range guard rejects an out-of-range cluster.
	if err := validateFFNMemoryLevel(good(), hiddenSize, total); err == nil {
		t.Fatal("validateFFNMemoryLevel(out-of-range cluster) error = nil")
	}
}

// TestInitialiseFFNMemoryInputWeights_NonPositiveHidden_Ugly proves the
// early-return guard leaves the buffer untouched when hidden size is not
// positive.
func TestInitialiseFFNMemoryInputWeights_NonPositiveHidden_Ugly(t *testing.T) {
	weights := []float32{1, 2, 3}
	initialiseFFNMemoryInputWeights(weights, 0, 0, 0, 0)
	if weights[0] != 1 || weights[1] != 2 || weights[2] != 3 {
		t.Fatalf("weights = %+v, want untouched for non-positive hidden size", weights)
	}
}
