// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package memorypretrain

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/metal"
)

func TestMetalFFNMemoryAugmenter_AugmentFFNMemoryGeneric_Good(t *testing.T) {
	if core.Getenv("GO_MLX_RUN_METAL_TESTS") != "1" {
		t.Skip("set GO_MLX_RUN_METAL_TESTS=1 to enable Metal runtime tests")
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
	if core.Getenv("GO_MLX_RUN_METAL_TESTS") != "1" {
		t.Skip("set GO_MLX_RUN_METAL_TESTS=1 to enable Metal runtime tests")
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

func TestMetalFFNMemoryAugmenter_Validation_Bad(t *testing.T) {
	if _, err := NewMetalFFNMemoryAugmenter(nil, nil); err == nil {
		t.Fatal("NewMetalFFNMemoryAugmenter(nil) error = nil")
	}
	if err := (*MetalFFNMemoryAugmenter)(nil).SetClusterIDs(nil); err == nil {
		t.Fatal("SetClusterIDs(nil receiver) error = nil")
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
