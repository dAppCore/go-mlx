// SPDX-Licence-Identifier: EUPL-1.2

package memorypretrain

import (
	"context"
	"errors"
	"math"
	"strings"
	"testing"
)

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

func TestBank_ClusterIDsRoutePerLevel_Good(t *testing.T) {
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
	ids, err := bank.ClusterIDs([]float32{1, 0})
	if err != nil {
		t.Fatalf("ClusterIDs() error = %v", err)
	}
	if len(ids) != 2 || ids[0] != assignments[0].ClusterID || ids[1] != assignments[1].ClusterID {
		t.Fatalf("ClusterIDs() = %+v, assignments=%+v", ids, assignments)
	}
}

// TestBank_RoutingGuards_Bad covers the shared nil-bank, dimension-mismatch,
// and missing-root guards on the three routing entry points, which the happy
// paths never reach because they always build a well-formed bank.
func TestBank_RoutingGuards_Bad(t *testing.T) {
	// A structurally-empty bank: zero dimension and no nodes.
	empty := &Bank{Dimension: 2}

	// nil-bank guard.
	if _, err := (*Bank)(nil).ClusterIDsInto(nil, []float32{1, 0}); err == nil {
		t.Fatal("ClusterIDsInto(nil bank) error = nil")
	}
	if _, err := (*Bank)(nil).ClusterAssignments([]float32{1, 0}); err == nil {
		t.Fatal("ClusterAssignments(nil bank) error = nil")
	}

	// dimension-mismatch guard.
	if _, err := empty.ClusterIDsInto(nil, []float32{1}); err == nil {
		t.Fatal("ClusterIDsInto(dim mismatch) error = nil")
	}
	if _, err := empty.ClusterAssignments([]float32{1}); err == nil {
		t.Fatal("ClusterAssignments(dim mismatch) error = nil")
	}
	if _, err := empty.RetrieveInto(nil, []float32{1}, 1); err == nil {
		t.Fatal("RetrieveInto(dim mismatch) error = nil")
	}

	// missing-root guard (correct dimension, but no nodes).
	if _, err := empty.ClusterIDsInto(nil, []float32{1, 0}); err == nil {
		t.Fatal("ClusterIDsInto(no root) error = nil")
	}
	if _, err := empty.ClusterAssignments([]float32{1, 0}); err == nil {
		t.Fatal("ClusterAssignments(no root) error = nil")
	}
	if _, err := empty.RetrieveInto(nil, []float32{1, 0}, 1); err == nil {
		t.Fatal("RetrieveInto(no root) error = nil")
	}
}

// TestBank_EmptyLeaf_Ugly builds a synthetic bank whose routed leaf node holds
// no block IDs, exercising both the empty-leaf early return in RetrieveInto and
// the no-retrievals early return in InjectAdditive.
func TestBank_EmptyLeaf_Ugly(t *testing.T) {
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

// TestCosine_ZeroNorm_Ugly covers the zero-norm guard in the cosine helper: a
// zero vector against any vector yields zero similarity.
func TestCosine_ZeroNorm_Ugly(t *testing.T) {
	if got := cosine([]float32{0, 0}, []float32{1, 0}); got != 0 {
		t.Fatalf("cosine(zero, unit) = %v, want 0", got)
	}
	if got := cosine([]float32{1, 0}, []float32{0, 0}); got != 0 {
		t.Fatalf("cosine(unit, zero) = %v, want 0", got)
	}
}

// TestBuildBank_FewerBlocksThanBranching_Ugly builds a bank from fewer blocks
// than the branching factor, all sharing one embedding. kmeans is reached
// (MinClusterSize is below the block count) but degenerates: k clamps to the
// block count, every point collapses into a single cluster (leaving the other
// centroid unassigned), so buildNode takes the single-cluster short-circuit and
// the bank is one leaf.
func TestBuildBank_FewerBlocksThanBranching_Ugly(t *testing.T) {
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

// TestBuildBank_TiedScores_Good builds a bank with duplicate embeddings so the
// retrieval sort exercises the equal-score tie-break by block index.
func TestBuildBank_TiedScores_Good(t *testing.T) {
	bank, err := BuildBank([]Block{
		{ID: "first", Embedding: []float32{1, 0}},
		{ID: "second", Embedding: []float32{1, 0}},
		{ID: "third", Embedding: []float32{1, 0}},
	}, BuildConfig{BranchingFactor: 2, MaxDepth: 1, MinClusterSize: 8, KMeansIters: 4})
	if err != nil {
		t.Fatalf("BuildBank(tied scores) error = %v", err)
	}
	got, err := bank.Retrieve([]float32{1, 0}, 3)
	if err != nil {
		t.Fatalf("Retrieve(tied scores) error = %v", err)
	}
	// Identical scores resolve to ascending block index, so insertion order holds.
	if len(got) != 3 || got[0].BlockIndex >= got[1].BlockIndex || got[1].BlockIndex >= got[2].BlockIndex {
		t.Fatalf("Retrieve(tied scores) = %+v, want ascending block index on score ties", got)
	}
}

func TestGenericClusterIDs_Good(t *testing.T) {
	ids, err := GenericClusterIDs([]int{16, 256, 1024})
	if err != nil {
		t.Fatalf("GenericClusterIDs() error = %v", err)
	}
	if len(ids) != 3 || ids[0] != 15 || ids[1] != 255 || ids[2] != 1023 {
		t.Fatalf("GenericClusterIDs() = %+v, want last cluster per level", ids)
	}
	if _, err := GenericClusterIDs([]int{16, 0}); err == nil {
		t.Fatal("GenericClusterIDs(invalid) error = nil")
	}
}

func TestFFNMemoryBank_AddToFFNOutputSelectsClusterPerLevel_Good(t *testing.T) {
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

func TestFFNMemoryBank_LinearRampAndValidation_GoodBad(t *testing.T) {
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
	// A ramp that would floor below one token is clamped up to a single token, so
	// the earliest layer still holds a usable memory.
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
	out, stats, err := bank.AddToFFNOutput(nil, []float32{1, 2, 3, 4}, []float32{4, 3, 2, 1}, 0, []int{2})
	if err != nil {
		t.Fatalf("AddToFFNOutput() zero memory error = %v", err)
	}
	if len(out) != 4 || out[0] != 1 || out[3] != 4 || !stats.Applied {
		t.Fatalf("zero-initialised memory output=%+v stats=%+v, want unchanged output with applied route", out, stats)
	}
	if _, _, err := bank.AddToFFNOutput(nil, []float32{1}, []float32{1, 2, 3, 4}, 0, []int{2}); err == nil {
		t.Fatal("AddToFFNOutput(output dim mismatch) error = nil")
	}
	if _, _, err := bank.AddToFFNOutput(nil, []float32{1, 2, 3, 4}, []float32{1, 2, 3, 4}, 0, []int{3}); err == nil {
		t.Fatal("AddToFFNOutput(cluster out of range) error = nil")
	}
	// Nil receiver, MLP-input dimension mismatch, and a cluster-ID count that
	// disagrees with the level count all guard before any memory is applied.
	if _, _, err := (*FFNMemoryBank)(nil).AddToFFNOutput(nil, []float32{1, 2, 3, 4}, []float32{1, 2, 3, 4}, 0, []int{2}); err == nil {
		t.Fatal("AddToFFNOutput(nil bank) error = nil")
	}
	if _, _, err := bank.AddToFFNOutput(nil, []float32{1, 2, 3, 4}, []float32{1}, 0, []int{2}); err == nil {
		t.Fatal("AddToFFNOutput(mlp input dim mismatch) error = nil")
	}
	if _, _, err := bank.AddToFFNOutput(nil, []float32{1, 2, 3, 4}, []float32{1, 2, 3, 4}, 0, []int{2, 2}); err == nil {
		t.Fatal("AddToFFNOutput(cluster id count mismatch) error = nil")
	}
}

func TestFFNMemoryBank_AddRoutedToFFNOutputUsesRetrieverClusterIDs_Good(t *testing.T) {
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
}

func TestFFNMemoryBank_AddRoutedToFFNOutputPadsUnreachedLevelsWithGeneric_Good(t *testing.T) {
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
}

func TestFFNMemoryBank_AddGenericToFFNOutputUsesLastClusterPerLevel_Good(t *testing.T) {
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

func TestFFNMemoryBank_AddGenericToFFNOutput_Bad(t *testing.T) {
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

func TestFFNMemoryBank_AddRoutedToFFNOutput_Bad(t *testing.T) {
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
	// A router with more hierarchy levels than the bank has memory levels routes
	// successfully but produces more cluster IDs than the bank can pad, surfacing
	// the generic-fallback padding error.
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

// TestPadClusterIDsWithGenericFallbackInto_NonPositiveCount_Bad covers the
// per-level positive-count guard in the allocation-free padding helper.
func TestPadClusterIDsWithGenericFallbackInto_NonPositiveCount_Bad(t *testing.T) {
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

func TestFFNMemoryRuntime_AddTextToFFNOutputRoutesThroughEmbedder_Good(t *testing.T) {
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
	embedCalls := 0
	runtime, err := NewFFNMemoryRuntime(mem, router, EmbedFunc(func(_ context.Context, text string) ([]float32, error) {
		embedCalls++
		if text != "Go memory planning" {
			t.Fatalf("embedded text = %q, want model-side query text", text)
		}
		return []float32{1, 0}, nil
	}))
	if err != nil {
		t.Fatalf("NewFFNMemoryRuntime() error = %v", err)
	}

	out, ids, stats, err := runtime.AddTextToFFNOutput(context.Background(), nil, []float32{1, 2}, []float32{2, 4}, "Go memory planning", 0)
	if err != nil {
		t.Fatalf("AddTextToFFNOutput() error = %v", err)
	}
	wantContribution := siluTest(2) * 4
	want := []float32{1 + 2*wantContribution, 2 + 3*wantContribution}
	if embedCalls != 1 || len(ids) != 1 || ids[0] != cluster || len(out) != 2 || !approx32(out[0], want[0]) || !approx32(out[1], want[1]) {
		t.Fatalf("AddTextToFFNOutput() calls=%d out=%+v ids=%+v, want out=%+v ids=%+v", embedCalls, out, ids, want, clusterIDs)
	}
	if !stats.Applied || stats.LevelsApplied != 1 {
		t.Fatalf("stats = %+v, want routed runtime memory applied", stats)
	}
}

func TestFFNMemoryRuntime_AddTextToFFNOutputUsesGenericFallback_Good(t *testing.T) {
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
	level.W1 = []float32{0, 0, 0, 0, 1, 0}
	level.W2 = []float32{0, 0, 0, 0, 0, 1}
	level.W3 = []float32{0, 0, 0, 0, 2, 3}
	runtime, err := NewFFNMemoryRuntime(mem, nil, nil)
	if err != nil {
		t.Fatalf("NewFFNMemoryRuntime(generic) error = %v", err)
	}

	out, ids, stats, err := runtime.AddTextToFFNOutput(context.Background(), nil, []float32{5, 7}, []float32{2, 4}, "", 0)
	if err != nil {
		t.Fatalf("AddTextToFFNOutput(generic) error = %v", err)
	}
	wantContribution := siluTest(2) * 4
	want := []float32{5 + 2*wantContribution, 7 + 3*wantContribution}
	if len(ids) != 1 || ids[0] != 2 || len(out) != 2 || !approx32(out[0], want[0]) || !approx32(out[1], want[1]) {
		t.Fatalf("AddTextToFFNOutput(generic) out=%+v ids=%+v, want out=%+v ids=[2]", out, ids, want)
	}
	if !stats.Applied || stats.LevelsApplied != 1 {
		t.Fatalf("stats = %+v, want generic runtime memory applied", stats)
	}
}

func TestFFNMemoryRuntime_Validation_Bad(t *testing.T) {
	if _, err := NewFFNMemoryRuntime(nil, nil, nil); err == nil {
		t.Fatal("NewFFNMemoryRuntime(nil memory) error = nil")
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
	router, err := BuildBank([]Block{{ID: "a", Embedding: []float32{1, 0}}}, BuildConfig{})
	if err != nil {
		t.Fatalf("BuildBank() error = %v", err)
	}
	// A router without an embedder is rejected at construction.
	if _, err := NewFFNMemoryRuntime(mem, router, nil); err == nil {
		t.Fatal("NewFFNMemoryRuntime(router without embedder) error = nil")
	}
	// Nil receiver and a nil-memory runtime both error from the method.
	if _, _, _, err := (*FFNMemoryRuntime)(nil).AddTextToFFNOutput(context.Background(), nil, []float32{1, 2}, []float32{3, 4}, "", 0); err == nil {
		t.Fatal("AddTextToFFNOutput(nil receiver) error = nil")
	}
	if _, _, _, err := (&FFNMemoryRuntime{}).AddTextToFFNOutput(context.Background(), nil, []float32{1, 2}, []float32{3, 4}, "", 0); err == nil {
		t.Fatal("AddTextToFFNOutput(nil memory) error = nil")
	}
	// A runtime whose router is set but embedder was cleared after construction
	// rejects the call rather than routing without an embedder.
	if _, _, _, err := (&FFNMemoryRuntime{Memory: mem, Router: router}).AddTextToFFNOutput(context.Background(), nil, []float32{1, 2}, []float32{3, 4}, "x", 0); err == nil {
		t.Fatal("AddTextToFFNOutput(router without embedder) error = nil")
	}
}

func TestFFNMemoryRuntime_AddTextToFFNOutput_Ugly(t *testing.T) {
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
	router, err := BuildBank([]Block{
		{ID: "a", Embedding: []float32{1, 0}},
		{ID: "b", Embedding: []float32{0, 1}},
	}, BuildConfig{BranchingFactor: 2, MaxDepth: 1, MinClusterSize: 1, KMeansIters: 8})
	if err != nil {
		t.Fatalf("BuildBank() error = %v", err)
	}

	// A context cancelled before the call returns early without embedding.
	embedCalls := 0
	embedder := EmbedFunc(func(_ context.Context, _ string) ([]float32, error) {
		embedCalls++
		return []float32{1, 0}, nil
	})
	runtime, err := NewFFNMemoryRuntime(mem, router, embedder)
	if err != nil {
		t.Fatalf("NewFFNMemoryRuntime() error = %v", err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, _, _, err := runtime.AddTextToFFNOutput(ctx, nil, []float32{1, 2}, []float32{3, 4}, "x", 0); !errors.Is(err, context.Canceled) {
		t.Fatalf("AddTextToFFNOutput(cancelled) error = %v, want context.Canceled", err)
	}
	if embedCalls != 0 {
		t.Fatalf("embed calls = %d, want cancellation before embedding", embedCalls)
	}

	// An embedder error is wrapped with call context.
	failing, err := NewFFNMemoryRuntime(mem, router, EmbedFunc(func(context.Context, string) ([]float32, error) {
		return nil, errors.New("anchor offline")
	}))
	if err != nil {
		t.Fatalf("NewFFNMemoryRuntime(failing) error = %v", err)
	}
	if _, _, _, err := failing.AddTextToFFNOutput(context.Background(), nil, []float32{1, 2}, []float32{3, 4}, "x", 0); err == nil || !strings.Contains(err.Error(), "embed query text") {
		t.Fatalf("AddTextToFFNOutput(embed error) error = %v, want embed context", err)
	}

	// An embedder that cancels the context mid-call passes the pre-embed check
	// but trips the post-embed cancellation guard before routing.
	midCtx, midCancel := context.WithCancel(context.Background())
	midRuntime, err := NewFFNMemoryRuntime(mem, router, EmbedFunc(func(context.Context, string) ([]float32, error) {
		midCancel()
		return []float32{1, 0}, nil
	}))
	if err != nil {
		t.Fatalf("NewFFNMemoryRuntime(mid-cancel) error = %v", err)
	}
	if _, _, _, err := midRuntime.AddTextToFFNOutput(midCtx, nil, []float32{1, 2}, []float32{3, 4}, "x", 0); !errors.Is(err, context.Canceled) {
		t.Fatalf("AddTextToFFNOutput(cancel during embed) error = %v, want context.Canceled", err)
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

// TestMemoryPretrainHelpers_PureBranches_Ugly drives the small unexported
// helpers in memorypretrain.go directly, covering the guard branches the public
// build/retrieve paths never exercise: nil-bank dimension, a cluster-id miss,
// injection-config defaulting, and the empty-block-IDs centroid.
func TestMemoryPretrainHelpers_PureBranches_Ugly(t *testing.T) {
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

func TestBuildBankFromCorpus_EmbedsRecords_Good(t *testing.T) {
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

func TestBuildBankFromCorpus_Validation_Bad(t *testing.T) {
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
	if _, err := (EmbedFunc)(nil).Embed(context.Background(), "x"); err == nil {
		t.Fatal("EmbedFunc(nil).Embed() error = nil")
	}
}

func TestBuildBankFromCorpus_ContextCancelled_Ugly(t *testing.T) {
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

func TestInjectAdditive_AddsRetrievedMemory_Good(t *testing.T) {
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
}

func TestInjectAdditive_UniformFallbackWhenAllScoresClamped_Good(t *testing.T) {
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
}

func TestInjectAdditive_SkipsClampedBlocksInWeightedBlend_Good(t *testing.T) {
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
}

func TestRetrieveInto_ClampsKToAvailableBlocks_Good(t *testing.T) {
	// Ask for more neighbours than the routed leaf holds; RetrieveInto clamps k
	// down to the available block count rather than over-slicing.
	bank, err := BuildBank([]Block{
		{ID: "a", Embedding: []float32{1, 0}},
		{ID: "b", Embedding: []float32{0.9, 0.1}},
	}, BuildConfig{BranchingFactor: 2, MaxDepth: 1, MinClusterSize: 2})
	if err != nil {
		t.Fatalf("BuildBank() error = %v", err)
	}
	got, err := bank.RetrieveInto(nil, []float32{1, 0}, 100)
	if err != nil {
		t.Fatalf("RetrieveInto() error = %v", err)
	}
	if len(got) != 2 || got[0].BlockID != "a" {
		t.Fatalf("RetrieveInto(k=100) = %+v, want all available blocks clamped", got)
	}
}

func TestInjectAdditive_Validation_Bad(t *testing.T) {
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
