// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package memorypretrain

import (
	"testing"

	"dappco.re/go/mlx/internal/metaltest"
	"dappco.re/go/mlx/pkg/metal"
)

func metalAugmenterBank(t *testing.T) *FFNMemoryBank {
	t.Helper()
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
	return bank
}

// TestFfnMemoryMetal_NewMetalFFNMemoryAugmenter_Good constructs an augmenter
// with the generic fallback (nil route) and with an explicit first-level route,
// confirming the constructor seeds the cluster IDs in both modes.
func TestFfnMemoryMetal_NewMetalFFNMemoryAugmenter_Good(t *testing.T) {
	bank := metalAugmenterBank(t)
	generic, err := NewMetalFFNMemoryAugmenter(bank, nil)
	if err != nil {
		t.Fatalf("NewMetalFFNMemoryAugmenter(nil route) error = %v", err)
	}
	if len(generic.ClusterIDs) != 1 || generic.ClusterIDs[0] != 2 {
		t.Fatalf("NewMetalFFNMemoryAugmenter(nil route) ClusterIDs = %+v, want generic [2]", generic.ClusterIDs)
	}
	explicit, err := NewMetalFFNMemoryAugmenter(bank, []int{0})
	if err != nil {
		t.Fatalf("NewMetalFFNMemoryAugmenter(explicit route) error = %v", err)
	}
	if len(explicit.ClusterIDs) != 1 || explicit.ClusterIDs[0] != 0 {
		t.Fatalf("NewMetalFFNMemoryAugmenter(explicit route) ClusterIDs = %+v, want [0]", explicit.ClusterIDs)
	}
}

// TestFfnMemoryMetal_NewMetalFFNMemoryAugmenter_Bad rejects a nil memory bank.
func TestFfnMemoryMetal_NewMetalFFNMemoryAugmenter_Bad(t *testing.T) {
	if _, err := NewMetalFFNMemoryAugmenter(nil, nil); err == nil {
		t.Fatal("NewMetalFFNMemoryAugmenter(nil) error = nil")
	}
}

// TestFfnMemoryMetal_NewMetalFFNMemoryAugmenter_Ugly passes a structurally valid
// bank but an out-of-range route, so the constructor surfaces the SetClusterIDs
// padding error rather than returning a half-built augmenter.
func TestFfnMemoryMetal_NewMetalFFNMemoryAugmenter_Ugly(t *testing.T) {
	bank := metalAugmenterBank(t)
	if _, err := NewMetalFFNMemoryAugmenter(bank, []int{3}); err == nil {
		t.Fatal("NewMetalFFNMemoryAugmenter(out-of-range route) error = nil")
	}
}

// TestFfnMemoryMetal_MetalFFNMemoryAugmenter_SetClusterIDs_Good updates the route
// after construction: an explicit route is padded with the generic slot for the
// unaddressed level, and a nil route restores the full generic fallback.
func TestFfnMemoryMetal_MetalFFNMemoryAugmenter_SetClusterIDs_Good(t *testing.T) {
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
	augmenter, err := NewMetalFFNMemoryAugmenter(bank, nil)
	if err != nil {
		t.Fatalf("NewMetalFFNMemoryAugmenter() error = %v", err)
	}
	if err := augmenter.SetClusterIDs([]int{1}); err != nil {
		t.Fatalf("SetClusterIDs(explicit) error = %v", err)
	}
	if len(augmenter.ClusterIDs) != 2 || augmenter.ClusterIDs[0] != 1 || augmenter.ClusterIDs[1] != 4 {
		t.Fatalf("SetClusterIDs(explicit) = %+v, want explicit first level and generic padded second", augmenter.ClusterIDs)
	}
	if err := augmenter.SetClusterIDs(nil); err != nil {
		t.Fatalf("SetClusterIDs(nil) error = %v", err)
	}
	if len(augmenter.ClusterIDs) != 2 || augmenter.ClusterIDs[0] != 2 || augmenter.ClusterIDs[1] != 4 {
		t.Fatalf("SetClusterIDs(nil) = %+v, want full generic fallback [2 4]", augmenter.ClusterIDs)
	}
}

// TestFfnMemoryMetal_MetalFFNMemoryAugmenter_SetClusterIDs_Bad rejects a route
// with more levels than the bank provides.
func TestFfnMemoryMetal_MetalFFNMemoryAugmenter_SetClusterIDs_Bad(t *testing.T) {
	bank := metalAugmenterBank(t)
	augmenter, err := NewMetalFFNMemoryAugmenter(bank, nil)
	if err != nil {
		t.Fatalf("NewMetalFFNMemoryAugmenter() error = %v", err)
	}
	if err := augmenter.SetClusterIDs([]int{0, 0}); err == nil {
		t.Fatal("SetClusterIDs(too many route levels) error = nil")
	}
}

// TestFfnMemoryMetal_MetalFFNMemoryAugmenter_SetClusterIDs_Ugly exercises the
// nil-receiver, nil-memory, and empty-bank generic-fallback guards that the
// happy path never reaches.
func TestFfnMemoryMetal_MetalFFNMemoryAugmenter_SetClusterIDs_Ugly(t *testing.T) {
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
}

// TestFfnMemoryMetal_MetalFFNMemoryAugmenter_AugmentFFNMemory_Good applies memory
// through the neutral metal hook: the generic-fallback single-level path, and a
// two-level path where an explicit first-level route is padded with the generic
// second-level slot. Both assert the exact silu-gated contribution.
func TestFfnMemoryMetal_MetalFFNMemoryAugmenter_AugmentFFNMemory_Good(t *testing.T) {
	if !metaltest.RunMetalTests {
		t.Skip("build with -tags metal_runtime to enable Metal runtime tests")
	}
	if !metal.MetalAvailable() {
		t.Skip("Metal runtime unavailable")
	}
	t.Run("generic fallback", func(t *testing.T) {
		bank := metalAugmenterBank(t)
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
	})

	t.Run("explicit route padded", func(t *testing.T) {
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
	})
}

// TestFfnMemoryMetal_MetalFFNMemoryAugmenter_AugmentFFNMemory_Bad rejects nil
// receiver and nil-memory augmenters before touching any array.
func TestFfnMemoryMetal_MetalFFNMemoryAugmenter_AugmentFFNMemory_Bad(t *testing.T) {
	if !metaltest.RunMetalTests {
		t.Skip("build with -tags metal_runtime to enable Metal runtime tests")
	}
	if !metal.MetalAvailable() {
		t.Skip("Metal runtime unavailable")
	}
	// Nil receiver and nil-memory augmenter both error before touching arrays.
	if _, _, err := (*MetalFFNMemoryAugmenter)(nil).AugmentFFNMemory(0, nil, nil); err == nil {
		t.Fatal("AugmentFFNMemory(nil receiver) error = nil")
	}
	if _, _, err := (&MetalFFNMemoryAugmenter{}).AugmentFFNMemory(0, nil, nil); err == nil {
		t.Fatal("AugmentFFNMemory(nil memory) error = nil")
	}
}

// TestFfnMemoryMetal_MetalFFNMemoryAugmenter_AugmentFFNMemory_Ugly drives the
// array-shape guards: invalid output, invalid input, mismatched sizes, and a
// total size that is not divisible by the hidden size.
func TestFfnMemoryMetal_MetalFFNMemoryAugmenter_AugmentFFNMemory_Ugly(t *testing.T) {
	if !metaltest.RunMetalTests {
		t.Skip("build with -tags metal_runtime to enable Metal runtime tests")
	}
	if !metal.MetalAvailable() {
		t.Skip("Metal runtime unavailable")
	}
	bank := metalAugmenterBank(t)
	augmenter, err := NewMetalFFNMemoryAugmenter(bank, nil)
	if err != nil {
		t.Fatalf("NewMetalFFNMemoryAugmenter() error = %v", err)
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
