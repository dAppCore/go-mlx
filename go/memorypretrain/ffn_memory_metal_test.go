// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package memorypretrain

import (
	"testing"

	"dappco.re/go/mlx/internal/metaltest"
	"dappco.re/go/mlx/pkg/metal"
)

func TestMetalFFNMemoryAugmenter_AugmentFFNMemoryGeneric_Good(t *testing.T) {
	if !metaltest.RunMetalTests {
		t.Skip("build with -tags metal_runtime to enable Metal runtime tests")
	}
	if !metal.MetalAvailable() {
		t.Skip("Metal runtime unavailable")
	}
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
	level := &bank.Layers[0].Levels[0]
	level.W1 = []float32{0, 0, 0, 0, 1, 0}
	level.W2 = []float32{0, 0, 0, 0, 0, 1}
	level.W3 = []float32{0, 0, 0, 0, 2, 3}
	augmenter, err := NewMetalFFNMemoryAugmenter(bank, nil)
	if err != nil {
		t.Fatalf("NewMetalFFNMemoryAugmenter() error = %v", err)
	}
	ffnOutput := metal.FromValues([]float32{5, 7}, 1, 1, 2)
	mlpInput := metal.FromValues([]float32{2, 4}, 1, 1, 2)
	defer metal.Free(ffnOutput, mlpInput)

	got, applied, err := augmenter.AugmentFFNMemory(0, ffnOutput, mlpInput)
	if err != nil {
		t.Fatalf("AugmentFFNMemory() error = %v", err)
	}
	if !applied {
		t.Fatal("AugmentFFNMemory() applied = false, want true")
	}
	defer metal.Free(got)

	wantContribution := siluTest(2) * 4
	want := []float32{5 + 2*wantContribution, 7 + 3*wantContribution}
	gotValues := got.Floats()
	if len(gotValues) != len(want) || !approx32(gotValues[0], want[0]) || !approx32(gotValues[1], want[1]) {
		t.Fatalf("AugmentFFNMemory() = %+v, want %+v", gotValues, want)
	}
	if shape := got.Shape(); len(shape) != 3 || shape[0] != 1 || shape[1] != 1 || shape[2] != 2 {
		t.Fatalf("shape = %+v, want [1 1 2]", shape)
	}
}

func TestMetalFFNMemoryAugmenter_AugmentFFNMemoryPadsExplicitRoute_Good(t *testing.T) {
	if !metaltest.RunMetalTests {
		t.Skip("build with -tags metal_runtime to enable Metal runtime tests")
	}
	if !metal.MetalAvailable() {
		t.Skip("Metal runtime unavailable")
	}
	bank, err := NewFFNMemoryBank(FFNMemoryConfig{
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
	level1 := &bank.Layers[0].Levels[0]
	level1.W1 = []float32{1, 0, 0, 0, 0, 0}
	level1.W2 = []float32{0, 1, 0, 0, 0, 0}
	level1.W3 = []float32{2, 3, 0, 0, 0, 0}
	level2 := &bank.Layers[0].Levels[1]
	level2.W1 = make([]float32, 5*2)
	level2.W2 = make([]float32, 5*2)
	level2.W3 = make([]float32, 5*2)
	genericLevel2 := 4
	level2.W1[genericLevel2*2] = 0.5
	level2.W2[genericLevel2*2+1] = 1
	level2.W3[genericLevel2*2] = 5
	level2.W3[genericLevel2*2+1] = 7
	augmenter, err := NewMetalFFNMemoryAugmenter(bank, []int{0})
	if err != nil {
		t.Fatalf("NewMetalFFNMemoryAugmenter() error = %v", err)
	}
	if len(augmenter.ClusterIDs) != 2 || augmenter.ClusterIDs[0] != 0 || augmenter.ClusterIDs[1] != genericLevel2 {
		t.Fatalf("ClusterIDs = %+v, want explicit first level and generic padded second level", augmenter.ClusterIDs)
	}
	ffnOutput := metal.FromValues([]float32{1, 2}, 1, 1, 2)
	mlpInput := metal.FromValues([]float32{2, 4}, 1, 1, 2)
	defer metal.Free(ffnOutput, mlpInput)

	got, applied, err := augmenter.AugmentFFNMemory(0, ffnOutput, mlpInput)
	if err != nil {
		t.Fatalf("AugmentFFNMemory() error = %v", err)
	}
	if !applied {
		t.Fatal("AugmentFFNMemory() applied = false, want true")
	}
	defer metal.Free(got)

	wantLevel1 := siluTest(2) * 4
	wantLevel2 := siluTest(1) * 4
	want := []float32{1 + 2*wantLevel1 + 5*wantLevel2, 2 + 3*wantLevel1 + 7*wantLevel2}
	gotValues := got.Floats()
	if len(gotValues) != len(want) || !approx32(gotValues[0], want[0]) || !approx32(gotValues[1], want[1]) {
		t.Fatalf("AugmentFFNMemory() = %+v, want %+v", gotValues, want)
	}
}

func TestMetalFFNMemoryAugmenter_AugmentFFNMemoryArrayGuards_Bad(t *testing.T) {
	if !metaltest.RunMetalTests {
		t.Skip("build with -tags metal_runtime to enable Metal runtime tests")
	}
	if !metal.MetalAvailable() {
		t.Skip("Metal runtime unavailable")
	}
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
	augmenter, err := NewMetalFFNMemoryAugmenter(bank, nil)
	if err != nil {
		t.Fatalf("NewMetalFFNMemoryAugmenter() error = %v", err)
	}

	// Nil receiver and nil-memory augmenter both error before touching arrays.
	if _, _, err := (*MetalFFNMemoryAugmenter)(nil).AugmentFFNMemory(0, nil, nil); err == nil {
		t.Fatal("AugmentFFNMemory(nil receiver) error = nil")
	}
	if _, _, err := (&MetalFFNMemoryAugmenter{}).AugmentFFNMemory(0, nil, nil); err == nil {
		t.Fatal("AugmentFFNMemory(nil memory) error = nil")
	}

	valid := metal.FromValues([]float32{5, 7}, 1, 1, 2)
	defer metal.Free(valid)

	// An invalid FFN output array is rejected.
	if _, _, err := augmenter.AugmentFFNMemory(0, nil, valid); err == nil {
		t.Fatal("AugmentFFNMemory(invalid ffn output) error = nil")
	}
	// An invalid MLP input array is rejected.
	if _, _, err := augmenter.AugmentFFNMemory(0, valid, nil); err == nil {
		t.Fatal("AugmentFFNMemory(invalid mlp input) error = nil")
	}

	// Mismatched sizes between the two arrays are rejected.
	mismatch := metal.FromValues([]float32{1, 2, 3, 4}, 1, 1, 4)
	defer metal.Free(mismatch)
	if _, _, err := augmenter.AugmentFFNMemory(0, valid, mismatch); err == nil {
		t.Fatal("AugmentFFNMemory(size mismatch) error = nil")
	}

	// A total size not divisible by the hidden size is rejected.
	odd1 := metal.FromValues([]float32{1, 2, 3}, 1, 1, 3)
	odd2 := metal.FromValues([]float32{4, 5, 6}, 1, 1, 3)
	defer metal.Free(odd1, odd2)
	if _, _, err := augmenter.AugmentFFNMemory(0, odd1, odd2); err == nil {
		t.Fatal("AugmentFFNMemory(size not divisible by hidden size) error = nil")
	}
}

func TestMetalFFNMemoryAugmenter_Validation_Bad(t *testing.T) {
	if _, err := NewMetalFFNMemoryAugmenter(nil, nil); err == nil {
		t.Fatal("NewMetalFFNMemoryAugmenter(nil) error = nil")
	}
	if err := (*MetalFFNMemoryAugmenter)(nil).SetClusterIDs(nil); err == nil {
		t.Fatal("SetClusterIDs(nil receiver) error = nil")
	}
	// A non-nil augmenter with no memory bank is rejected before routing.
	if err := (&MetalFFNMemoryAugmenter{}).SetClusterIDs(nil); err == nil {
		t.Fatal("SetClusterIDs(nil memory) error = nil")
	}
	// A bank with no layers yields no cluster counts, so the generic-fallback
	// branch surfaces the GenericClusterIDs error rather than panicking.
	if err := (&MetalFFNMemoryAugmenter{Memory: &FFNMemoryBank{HiddenSize: 2}}).SetClusterIDs(nil); err == nil {
		t.Fatal("SetClusterIDs(empty bank generic fallback) error = nil")
	}
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
	if _, err := NewMetalFFNMemoryAugmenter(bank, []int{3}); err == nil {
		t.Fatal("NewMetalFFNMemoryAugmenter(out-of-range route) error = nil")
	}
	augmenter, err := NewMetalFFNMemoryAugmenter(bank, nil)
	if err != nil {
		t.Fatalf("NewMetalFFNMemoryAugmenter(generic) error = %v", err)
	}
	if err := augmenter.SetClusterIDs([]int{0, 0}); err == nil {
		t.Fatal("SetClusterIDs(too many route levels) error = nil")
	}
}
