// SPDX-Licence-Identifier: EUPL-1.2

package memorypretrain

import (
	"testing"
)

// TestFfnMemory_NewFFNMemoryBank_Good allocates a hierarchical FFN memory table
// and confirms the linear-ramp token schedule, the always-on W3 zero init, and a
// fresh bank leaving the FFN output unchanged when memory is applied.
func TestFfnMemory_NewFFNMemoryBank_Good(t *testing.T) {
	bank, err := NewFFNMemoryBank(FFNMemoryConfig{
		HiddenSize:         4,
		Layers:             4,
		MemoryLevels:       []string{"1"},
		FFNMemoryTokens:    []int{8},
		NumClusters:        []int{2},
		LinearRampMemories: true,
		AddedGenericSize:   1,
		ZeroInitialiseW3:   true,
	})
	if err != nil {
		t.Fatalf("NewFFNMemoryBank() error = %v", err)
	}
	if got := bank.Layers[0].Levels[0].MemoryTokens; got != 4 {
		t.Fatalf("first layer memory tokens = %d, want ramped floor(2*8*1/4)", got)
	}
	if got := bank.Layers[3].Levels[0].MemoryTokens; got != 16 {
		t.Fatalf("last layer memory tokens = %d, want ramped floor(2*8*4/4)", got)
	}
	// W3 starts at zero everywhere, so applying memory to a fresh bank preserves
	// the anchor FFN output.
	out, stats, err := bank.AddToFFNOutput(nil, []float32{1, 2, 3, 4}, []float32{4, 3, 2, 1}, 0, []int{2})
	if err != nil {
		t.Fatalf("AddToFFNOutput() zero memory error = %v", err)
	}
	if len(out) != 4 || out[0] != 1 || out[3] != 4 || !stats.Applied {
		t.Fatalf("zero-initialised memory output=%+v stats=%+v, want unchanged output with applied route", out, stats)
	}
}

// TestFfnMemory_NewFFNMemoryBank_Bad proves NewFFNMemoryBank surfaces the config
// validation error rather than allocating a malformed bank.
func TestFfnMemory_NewFFNMemoryBank_Bad(t *testing.T) {
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

// TestFfnMemory_NewFFNMemoryBank_Ugly drives the linear-ramp clamp edge: a ramp
// whose floor would fall below one token is clamped up to a single token so the
// earliest layer still holds a usable memory.
func TestFfnMemory_NewFFNMemoryBank_Ugly(t *testing.T) {
	clamped, err := NewFFNMemoryBank(FFNMemoryConfig{
		HiddenSize:         2,
		Layers:             4,
		MemoryLevels:       []string{"1"},
		FFNMemoryTokens:    []int{1},
		NumClusters:        []int{2},
		LinearRampMemories: true,
		AddedGenericSize:   1,
	})
	if err != nil {
		t.Fatalf("NewFFNMemoryBank(clamped ramp) error = %v", err)
	}
	if got := clamped.Layers[0].Levels[0].MemoryTokens; got != 1 {
		t.Fatalf("clamped first layer memory tokens = %d, want clamped up to 1", got)
	}
}

// TestFfnMemory_FFNMemoryBank_AddToFFNOutput_Good drives the per-level cluster
// selection: each hierarchy level routes to its own cluster, the silu-gated
// contribution adds onto the FFN output, and the stats report both levels.
func TestFfnMemory_FFNMemoryBank_AddToFFNOutput_Good(t *testing.T) {
	bank, err := NewFFNMemoryBank(FFNMemoryConfig{
		HiddenSize:       2,
		Layers:           1,
		MemoryLevels:     []string{"1", "2"},
		FFNMemoryTokens:  []int{1, 1},
		NumClusters:      []int{2, 2},
		AddedGenericSize: 1,
	})
	if err != nil {
		t.Fatalf("NewFFNMemoryBank() error = %v", err)
	}
	level1 := &bank.Layers[0].Levels[0]
	level1.W1 = []float32{
		1, 0,
		0, 0,
		0, 0,
	}
	level1.W2 = []float32{
		0, 1,
		0, 0,
		0, 0,
	}
	level1.W3 = []float32{
		1, 2,
		0, 0,
		0, 0,
	}
	level2 := &bank.Layers[0].Levels[1]
	level2.W1 = []float32{
		0, 0,
		0, 0,
		0.5, 0,
	}
	level2.W2 = []float32{
		0, 0,
		0, 0,
		0, 2,
	}
	level2.W3 = []float32{
		0, 0,
		0, 0,
		3, 4,
	}

	out, stats, err := bank.AddToFFNOutput(nil, []float32{10, 20}, []float32{2, 1}, 0, []int{0, 2})
	if err != nil {
		t.Fatalf("AddToFFNOutput() error = %v", err)
	}
	wantLevel1 := siluTest(2) * 1
	wantLevel2 := siluTest(1) * 2
	want := []float32{10 + wantLevel1 + 3*wantLevel2, 20 + 2*wantLevel1 + 4*wantLevel2}
	if len(out) != 2 || !approx32(out[0], want[0]) || !approx32(out[1], want[1]) {
		t.Fatalf("AddToFFNOutput() = %+v, want %+v", out, want)
	}
	if stats.Layer != 0 || stats.LevelsApplied != 2 || stats.MemoryTokens != 2 || !stats.Applied {
		t.Fatalf("stats = %+v, want two applied memory levels", stats)
	}
}

// TestFfnMemory_FFNMemoryBank_AddToFFNOutput_Bad covers the dimension and route
// guards: FFN output and MLP input width mismatches, an out-of-range cluster id,
// and a cluster-ID count that disagrees with the level count.
func TestFfnMemory_FFNMemoryBank_AddToFFNOutput_Bad(t *testing.T) {
	bank, err := NewFFNMemoryBank(FFNMemoryConfig{
		HiddenSize:       4,
		Layers:           1,
		MemoryLevels:     []string{"1"},
		FFNMemoryTokens:  []int{8},
		NumClusters:      []int{2},
		AddedGenericSize: 1,
	})
	if err != nil {
		t.Fatalf("NewFFNMemoryBank() error = %v", err)
	}
	if _, _, err := bank.AddToFFNOutput(nil, []float32{1}, []float32{1, 2, 3, 4}, 0, []int{2}); err == nil {
		t.Fatal("AddToFFNOutput(output dim mismatch) error = nil")
	}
	if _, _, err := bank.AddToFFNOutput(nil, []float32{1, 2, 3, 4}, []float32{1, 2, 3, 4}, 0, []int{3}); err == nil {
		t.Fatal("AddToFFNOutput(cluster out of range) error = nil")
	}
	if _, _, err := bank.AddToFFNOutput(nil, []float32{1, 2, 3, 4}, []float32{1}, 0, []int{2}); err == nil {
		t.Fatal("AddToFFNOutput(mlp input dim mismatch) error = nil")
	}
	if _, _, err := bank.AddToFFNOutput(nil, []float32{1, 2, 3, 4}, []float32{1, 2, 3, 4}, 0, []int{2, 2}); err == nil {
		t.Fatal("AddToFFNOutput(cluster id count mismatch) error = nil")
	}
}

// TestFfnMemory_FFNMemoryBank_AddToFFNOutput_Ugly drives the nil-receiver and
// out-of-range layer guards that the populated happy path never reaches.
func TestFfnMemory_FFNMemoryBank_AddToFFNOutput_Ugly(t *testing.T) {
	bank, err := NewFFNMemoryBank(FFNMemoryConfig{
		HiddenSize:       4,
		Layers:           1,
		MemoryLevels:     []string{"1"},
		FFNMemoryTokens:  []int{8},
		NumClusters:      []int{2},
		AddedGenericSize: 1,
	})
	if err != nil {
		t.Fatalf("NewFFNMemoryBank() error = %v", err)
	}
	if _, _, err := (*FFNMemoryBank)(nil).AddToFFNOutput(nil, []float32{1, 2, 3, 4}, []float32{1, 2, 3, 4}, 0, []int{2}); err == nil {
		t.Fatal("AddToFFNOutput(nil bank) error = nil")
	}
	if _, _, err := bank.AddToFFNOutput(nil, []float32{1, 2, 3, 4}, []float32{1, 2, 3, 4}, 9, []int{2}); err == nil {
		t.Fatal("AddToFFNOutput(layer out of range) error = nil")
	}
}

// TestFfnMemory_FFNMemoryBank_ClusterCounts_Good proves ClusterCounts reports the
// learned cluster count plus the generic slot for each hierarchy level.
func TestFfnMemory_FFNMemoryBank_ClusterCounts_Good(t *testing.T) {
	bank, err := NewFFNMemoryBank(FFNMemoryConfig{
		HiddenSize:       2,
		Layers:           1,
		MemoryLevels:     []string{"1", "2"},
		FFNMemoryTokens:  []int{1, 1},
		NumClusters:      []int{2, 3},
		AddedGenericSize: 1,
	})
	if err != nil {
		t.Fatalf("NewFFNMemoryBank() error = %v", err)
	}
	counts := bank.ClusterCounts()
	if len(counts) != 2 || counts[0] != 3 || counts[1] != 4 {
		t.Fatalf("ClusterCounts() = %+v, want learned clusters plus generic slot", counts)
	}
}

// TestFfnMemory_FFNMemoryBank_ClusterCounts_Bad proves a nil bank yields no
// cluster counts rather than panicking.
func TestFfnMemory_FFNMemoryBank_ClusterCounts_Bad(t *testing.T) {
	if counts := (*FFNMemoryBank)(nil).ClusterCounts(); counts != nil {
		t.Fatalf("ClusterCounts(nil bank) = %+v, want nil", counts)
	}
}

// TestFfnMemory_FFNMemoryBank_ClusterCounts_Ugly proves a non-nil bank with no
// layers also yields no cluster counts (the empty-layer guard) rather than
// indexing a missing layer.
func TestFfnMemory_FFNMemoryBank_ClusterCounts_Ugly(t *testing.T) {
	if counts := (&FFNMemoryBank{HiddenSize: 2}).ClusterCounts(); counts != nil {
		t.Fatalf("ClusterCounts(no layers) = %+v, want nil", counts)
	}
}

// TestFfnMemory_FFNMemoryBank_GenericClusterIDs_Good proves the bank-level
// GenericClusterIDs returns the final cluster index at each level.
func TestFfnMemory_FFNMemoryBank_GenericClusterIDs_Good(t *testing.T) {
	bank, err := NewFFNMemoryBank(FFNMemoryConfig{
		HiddenSize:       2,
		Layers:           1,
		MemoryLevels:     []string{"1", "2"},
		FFNMemoryTokens:  []int{1, 1},
		NumClusters:      []int{2, 3},
		AddedGenericSize: 1,
	})
	if err != nil {
		t.Fatalf("NewFFNMemoryBank() error = %v", err)
	}
	ids, err := bank.GenericClusterIDs()
	if err != nil {
		t.Fatalf("GenericClusterIDs() error = %v", err)
	}
	if len(ids) != 2 || ids[0] != 2 || ids[1] != 3 {
		t.Fatalf("GenericClusterIDs() = %+v, want last index per level [2 3]", ids)
	}
}

// TestFfnMemory_FFNMemoryBank_GenericClusterIDs_Bad proves a bank with no layers
// has no cluster counts, so GenericClusterIDs surfaces the underlying error.
func TestFfnMemory_FFNMemoryBank_GenericClusterIDs_Bad(t *testing.T) {
	if _, err := (&FFNMemoryBank{HiddenSize: 2}).GenericClusterIDs(); err == nil {
		t.Fatal("GenericClusterIDs(no layers) error = nil")
	}
}

// TestFfnMemory_FFNMemoryBank_GenericClusterIDs_Ugly proves a nil bank likewise
// surfaces an error from the empty cluster counts rather than panicking.
func TestFfnMemory_FFNMemoryBank_GenericClusterIDs_Ugly(t *testing.T) {
	if _, err := (*FFNMemoryBank)(nil).GenericClusterIDs(); err == nil {
		t.Fatal("GenericClusterIDs(nil bank) error = nil")
	}
}

// TestFfnMemory_FFNMemoryBank_AddGenericToFFNOutput_Good applies the generic
// fallback (final cluster slot at each level) and asserts the exact silu-gated
// contribution and the selected generic cluster IDs.
func TestFfnMemory_FFNMemoryBank_AddGenericToFFNOutput_Good(t *testing.T) {
	mem, err := NewFFNMemoryBank(FFNMemoryConfig{
		HiddenSize:       2,
		Layers:           1,
		MemoryLevels:     []string{"1", "2"},
		FFNMemoryTokens:  []int{1, 1},
		NumClusters:      []int{2, 3},
		AddedGenericSize: 1,
	})
	if err != nil {
		t.Fatalf("NewFFNMemoryBank() error = %v", err)
	}
	counts := mem.ClusterCounts()
	if len(counts) != 2 || counts[0] != 3 || counts[1] != 4 {
		t.Fatalf("ClusterCounts() = %+v, want learned clusters plus generic slot", counts)
	}
	level1 := &mem.Layers[0].Levels[0]
	level1.W1 = []float32{0, 0, 0, 0, 1, 0}
	level1.W2 = []float32{0, 0, 0, 0, 0, 1}
	level1.W3 = []float32{0, 0, 0, 0, 1, 1}
	level2 := &mem.Layers[0].Levels[1]
	level2.W1 = []float32{0, 0, 0, 0, 0, 0, 0.5, 0}
	level2.W2 = []float32{0, 0, 0, 0, 0, 0, 0, 1}
	level2.W3 = []float32{0, 0, 0, 0, 0, 0, 2, 3}

	out, ids, stats, err := mem.AddGenericToFFNOutput(nil, []float32{5, 7}, []float32{2, 4}, 0)
	if err != nil {
		t.Fatalf("AddGenericToFFNOutput() error = %v", err)
	}
	wantIDs := []int{2, 3}
	wantLevel1 := siluTest(2) * 4
	wantLevel2 := siluTest(1) * 4
	want := []float32{5 + wantLevel1 + 2*wantLevel2, 7 + wantLevel1 + 3*wantLevel2}
	if len(ids) != 2 || ids[0] != wantIDs[0] || ids[1] != wantIDs[1] || len(out) != 2 || !approx32(out[0], want[0]) || !approx32(out[1], want[1]) {
		t.Fatalf("AddGenericToFFNOutput() out=%+v ids=%+v, want out=%+v ids=%+v", out, ids, want, wantIDs)
	}
	if !stats.Applied || stats.LevelsApplied != 2 {
		t.Fatalf("stats = %+v, want generic memory applied", stats)
	}
}

// TestFfnMemory_FFNMemoryBank_AddGenericToFFNOutput_Bad covers the two failure
// modes: a layer index out of range propagates the AddToFFNOutput error, and a
// bank with no layers has no generic cluster IDs to select.
func TestFfnMemory_FFNMemoryBank_AddGenericToFFNOutput_Bad(t *testing.T) {
	mem, err := NewFFNMemoryBank(FFNMemoryConfig{
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
	// Layer index out of range propagates the AddToFFNOutput error.
	if _, _, _, err := mem.AddGenericToFFNOutput(nil, []float32{1, 2}, []float32{3, 4}, 5); err == nil {
		t.Fatal("AddGenericToFFNOutput(layer out of range) error = nil")
	}
	// A bank with no layers has no generic cluster IDs to select.
	empty := &FFNMemoryBank{HiddenSize: 2}
	if _, _, _, err := empty.AddGenericToFFNOutput(nil, []float32{1, 2}, []float32{3, 4}, 0); err == nil {
		t.Fatal("AddGenericToFFNOutput(no layers) error = nil")
	}
}

// TestFfnMemory_FFNMemoryBank_AddGenericToFFNOutput_Ugly drives the nil-receiver
// guard: AddGenericToFFNOutput on a nil bank surfaces the GenericClusterIDs
// error from the empty cluster counts rather than panicking.
func TestFfnMemory_FFNMemoryBank_AddGenericToFFNOutput_Ugly(t *testing.T) {
	if _, _, _, err := (*FFNMemoryBank)(nil).AddGenericToFFNOutput(nil, []float32{1, 2}, []float32{3, 4}, 0); err == nil {
		t.Fatal("AddGenericToFFNOutput(nil bank) error = nil")
	}
}

// TestFfnMemory_FFNMemoryBank_AddRoutedToFFNOutput_Good routes a query through
// the offline clustering bank and applies the selected memories: the single-level
// case uses the retriever's cluster IDs directly, and the two-level case pads the
// unreached second level with its generic slot.
func TestFfnMemory_FFNMemoryBank_AddRoutedToFFNOutput_Good(t *testing.T) {
	t.Run("uses retriever cluster ids", func(t *testing.T) {
		router, err := BuildBank([]Block{
			{ID: "go-1", Embedding: []float32{1, 0}},
			{ID: "go-2", Embedding: []float32{0.9, 0.1}},
			{ID: "poem-1", Embedding: []float32{0, 1}},
			{ID: "poem-2", Embedding: []float32{0.1, 0.9}},
		}, BuildConfig{BranchingFactor: 2, MaxDepth: 1, MinClusterSize: 1, KMeansIters: 8})
		if err != nil {
			t.Fatalf("BuildBank() error = %v", err)
		}
		clusterIDs, err := router.ClusterIDs([]float32{1, 0})
		if err != nil {
			t.Fatalf("ClusterIDs() error = %v", err)
		}
		if len(clusterIDs) != 1 {
			t.Fatalf("clusterIDs = %+v, want one level", clusterIDs)
		}
		mem, err := NewFFNMemoryBank(FFNMemoryConfig{
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
		level := &mem.Layers[0].Levels[0]
		level.W1 = []float32{0, 0, 0, 0, 0, 0}
		level.W2 = []float32{0, 0, 0, 0, 0, 0}
		level.W3 = []float32{0, 0, 0, 0, 0, 0}
		cluster := clusterIDs[0]
		level.W1[cluster*2] = 1
		level.W2[cluster*2+1] = 1
		level.W3[cluster*2] = 2
		level.W3[cluster*2+1] = 3

		out, ids, stats, err := mem.AddRoutedToFFNOutput(nil, []float32{1, 2}, []float32{2, 4}, router, []float32{1, 0}, 0)
		if err != nil {
			t.Fatalf("AddRoutedToFFNOutput() error = %v", err)
		}
		wantContribution := siluTest(2) * 4
		want := []float32{1 + 2*wantContribution, 2 + 3*wantContribution}
		if len(ids) != 1 || ids[0] != cluster || len(out) != 2 || !approx32(out[0], want[0]) || !approx32(out[1], want[1]) {
			t.Fatalf("AddRoutedToFFNOutput() out=%+v ids=%+v, want out=%+v ids=%+v", out, ids, want, clusterIDs)
		}
		if !stats.Applied || stats.LevelsApplied != 1 {
			t.Fatalf("stats = %+v, want routed memory applied", stats)
		}
	})

	t.Run("pads unreached levels with generic", func(t *testing.T) {
		router, err := BuildBank([]Block{
			{ID: "go-1", Embedding: []float32{1, 0}},
			{ID: "go-2", Embedding: []float32{0.9, 0.1}},
			{ID: "poem-1", Embedding: []float32{0, 1}},
			{ID: "poem-2", Embedding: []float32{0.1, 0.9}},
		}, BuildConfig{BranchingFactor: 2, MaxDepth: 1, MinClusterSize: 1, KMeansIters: 8})
		if err != nil {
			t.Fatalf("BuildBank() error = %v", err)
		}
		mem, err := NewFFNMemoryBank(FFNMemoryConfig{
			HiddenSize:       2,
			Layers:           1,
			MemoryLevels:     []string{"1", "2"},
			FFNMemoryTokens:  []int{1, 1},
			NumClusters:      []int{2, 4},
			AddedGenericSize: 1,
		})
		if err != nil {
			t.Fatalf("NewFFNMemoryBank() error = %v", err)
		}
		level1 := &mem.Layers[0].Levels[0]
		level1.W1 = []float32{0, 0, 0, 0, 0, 0}
		level1.W2 = []float32{0, 0, 0, 0, 0, 0}
		level1.W3 = []float32{0, 0, 0, 0, 0, 0}
		level1.W1[0] = 1
		level1.W2[1] = 1
		level1.W3[0] = 2
		level1.W3[1] = 3
		level2 := &mem.Layers[0].Levels[1]
		level2.W1 = make([]float32, 5*2)
		level2.W2 = make([]float32, 5*2)
		level2.W3 = make([]float32, 5*2)
		genericLevel2 := 4
		level2.W1[genericLevel2*2] = 0.5
		level2.W2[genericLevel2*2+1] = 1
		level2.W3[genericLevel2*2] = 5
		level2.W3[genericLevel2*2+1] = 7

		out, ids, stats, err := mem.AddRoutedToFFNOutput(nil, []float32{1, 2}, []float32{2, 4}, router, []float32{1, 0}, 0)
		if err != nil {
			t.Fatalf("AddRoutedToFFNOutput() error = %v", err)
		}
		wantIDs := []int{0, genericLevel2}
		wantLevel1 := siluTest(2) * 4
		wantLevel2 := siluTest(1) * 4
		want := []float32{1 + 2*wantLevel1 + 5*wantLevel2, 2 + 3*wantLevel1 + 7*wantLevel2}
		if len(ids) != 2 || ids[0] != wantIDs[0] || ids[1] != wantIDs[1] || len(out) != 2 || !approx32(out[0], want[0]) || !approx32(out[1], want[1]) {
			t.Fatalf("AddRoutedToFFNOutput() out=%+v ids=%+v, want out=%+v ids=%+v", out, ids, want, wantIDs)
		}
		if !stats.Applied || stats.LevelsApplied != 2 {
			t.Fatalf("stats = %+v, want both memory levels applied", stats)
		}
	})
}

// TestFfnMemory_FFNMemoryBank_AddRoutedToFFNOutput_Bad covers the routing failure
// modes: a nil router, a query whose dimension mismatches the router, and a route
// into an out-of-range layer.
func TestFfnMemory_FFNMemoryBank_AddRoutedToFFNOutput_Bad(t *testing.T) {
	mem, err := NewFFNMemoryBank(FFNMemoryConfig{
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
	// Nil router is rejected before any routing happens.
	if _, _, _, err := mem.AddRoutedToFFNOutput(nil, []float32{1, 2}, []float32{3, 4}, nil, []float32{1, 0}, 0); err == nil {
		t.Fatal("AddRoutedToFFNOutput(nil router) error = nil")
	}
	router, err := BuildBank([]Block{
		{ID: "a", Embedding: []float32{1, 0}},
		{ID: "b", Embedding: []float32{0, 1}},
	}, BuildConfig{BranchingFactor: 2, MaxDepth: 1, MinClusterSize: 1, KMeansIters: 8})
	if err != nil {
		t.Fatalf("BuildBank() error = %v", err)
	}
	// A query whose dimension mismatches the router surfaces the routing error.
	if _, _, _, err := mem.AddRoutedToFFNOutput(nil, []float32{1, 2}, []float32{3, 4}, router, []float32{1, 0, 0}, 0); err == nil {
		t.Fatal("AddRoutedToFFNOutput(query dim mismatch) error = nil")
	}
	// A valid route into an out-of-range layer propagates the apply error.
	if _, _, _, err := mem.AddRoutedToFFNOutput(nil, []float32{1, 2}, []float32{3, 4}, router, []float32{1, 0}, 9); err == nil {
		t.Fatal("AddRoutedToFFNOutput(layer out of range) error = nil")
	}
}

// TestFfnMemory_FFNMemoryBank_AddRoutedToFFNOutput_Ugly drives the generic-pad
// overflow edge: a router with more hierarchy levels than the bank has memory
// levels routes successfully but produces more cluster IDs than the bank can pad,
// surfacing the generic-fallback padding error.
func TestFfnMemory_FFNMemoryBank_AddRoutedToFFNOutput_Ugly(t *testing.T) {
	mem, err := NewFFNMemoryBank(FFNMemoryConfig{
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
	deepRouter, err := BuildBank([]Block{
		{ID: "a", Embedding: []float32{1, 0}},
		{ID: "b", Embedding: []float32{0.9, 0.1}},
		{ID: "c", Embedding: []float32{0, 1}},
		{ID: "d", Embedding: []float32{0.1, 0.9}},
	}, BuildConfig{BranchingFactor: 2, MaxDepth: 2, MinClusterSize: 1, KMeansIters: 8})
	if err != nil {
		t.Fatalf("BuildBank(deep router) error = %v", err)
	}
	if _, _, _, err := mem.AddRoutedToFFNOutput(nil, []float32{1, 2}, []float32{3, 4}, deepRouter, []float32{1, 0}, 0); err == nil {
		t.Fatal("AddRoutedToFFNOutput(more route levels than memory levels) error = nil")
	}
}

// TestFfnMemory_normaliseFFNMemoryConfig_Good proves the normaliser fills every
// empty slice with the upstream defaults and always forces ZeroInitialiseW3.
func TestFfnMemory_normaliseFFNMemoryConfig_Good(t *testing.T) {
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

// TestFfnMemory_validateFFNMemoryConfig_Bad drives each guard in
// validateFFNMemoryConfig directly: hidden size, layers, mismatched
// level/token/cluster lengths, blank level name, non-positive token count, and
// non-positive cluster count.
func TestFfnMemory_validateFFNMemoryConfig_Bad(t *testing.T) {
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

// TestFfnMemory_levelHiddenSize_Ugly covers the zero-token guard in
// levelHiddenStride and levelHiddenSize, which the populated round-trip path
// never reaches.
func TestFfnMemory_levelHiddenSize_Ugly(t *testing.T) {
	zero := &FFNMemoryLevelWeight{Name: "z", NumClusters: 2, AddedGenericSize: 1, MemoryTokens: 0}
	if got := levelHiddenStride(zero); got != 0 {
		t.Fatalf("levelHiddenStride(zero tokens) = %d, want 0", got)
	}
	if got := levelHiddenSize(zero); got != 0 {
		t.Fatalf("levelHiddenSize(zero tokens) = %d, want 0", got)
	}
}

// TestFfnMemory_validateFFNMemoryLevel_Bad isolates the W2 and W3 length-mismatch
// branches. The round-trip and bad-shape fixtures trip the W1 guard first, so
// this calls the level validator directly with a correct W1 but a short W2 (then
// W3) to reach the later checks.
func TestFfnMemory_validateFFNMemoryLevel_Bad(t *testing.T) {
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

// TestFfnMemory_initialiseFFNMemoryInputWeights_Ugly proves the early-return
// guard leaves the buffer untouched when hidden size is not positive.
func TestFfnMemory_initialiseFFNMemoryInputWeights_Ugly(t *testing.T) {
	weights := []float32{1, 2, 3}
	initialiseFFNMemoryInputWeights(weights, 0, 0, 0, 0)
	if weights[0] != 1 || weights[1] != 2 || weights[2] != 3 {
		t.Fatalf("weights = %+v, want untouched for non-positive hidden size", weights)
	}
}
