// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package memorypretrain

import (
	"testing"

	"dappco.re/go/mlx/internal/metaltest"
	"dappco.re/go/mlx/pkg/metal"
)

// TestFfnMemoryMetal_AugmentFFNMemory_HiddenSizeGuard_Ugly reaches the
// hidden-size guard that sits after the array-validity and size checks: the
// arrays are valid and equally sized, but the memory bank reports a non-positive
// hidden size, so AugmentFFNMemory rejects it before any cluster routing. This
// path is downstream of the live-array checks, so it is gated behind the Metal
// runtime like the other AugmentFFNMemory tests.
func TestFfnMemoryMetal_AugmentFFNMemory_HiddenSizeGuard_Ugly(t *testing.T) {
	if !metaltest.RunMetalTests {
		t.Skip("build with -tags metal_runtime to enable Metal runtime tests")
	}
	if !metal.MetalAvailable() {
		t.Skip("Metal runtime unavailable")
	}
	augmenter := &MetalFFNMemoryAugmenter{Memory: &FFNMemoryBank{HiddenSize: 0}}
	valid := metal.FromValues([]float32{5, 7}, 1, 1, 2)
	other := metal.FromValues([]float32{2, 4}, 1, 1, 2)
	defer metal.Free(valid, other)
	if _, _, err := augmenter.AugmentFFNMemory(0, valid, other); err == nil {
		t.Fatal("AugmentFFNMemory(non-positive hidden size) error = nil, want hidden-size guard failure")
	}
}

// TestFfnMemoryMetal_AugmentFFNMemory_ClusterIDDerivation_Good drives the two
// in-call cluster-ID derivation branches that the constructor-seeded happy path
// skips. With ClusterIDs cleared, the call re-derives the generic fallback and
// still applies memory; the result equals the silu-gated contribution from the
// generic slot.
func TestFfnMemoryMetal_AugmentFFNMemory_ClusterIDDerivation_Good(t *testing.T) {
	if !metaltest.RunMetalTests {
		t.Skip("build with -tags metal_runtime to enable Metal runtime tests")
	}
	if !metal.MetalAvailable() {
		t.Skip("Metal runtime unavailable")
	}
	bank := metalAugmenterBank(t)
	level := &bank.Layers[0].Levels[0]
	level.W1 = []float32{0, 0, 0, 0, 1, 0}
	level.W2 = []float32{0, 0, 0, 0, 0, 1}
	level.W3 = []float32{0, 0, 0, 0, 2, 3}
	augmenter, err := NewMetalFFNMemoryAugmenter(bank, nil)
	if err != nil {
		t.Fatalf("NewMetalFFNMemoryAugmenter() error = %v", err)
	}
	// Force the in-call generic re-derivation branch (len(clusterIDs) == 0).
	augmenter.ClusterIDs = nil

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
		t.Fatalf("AugmentFFNMemory(re-derived generic IDs) = %+v, want %+v", gotValues, want)
	}
}

// TestFfnMemoryMetal_AugmentFFNMemory_ClusterIDDerivation_Ugly drives the two
// in-call cluster-ID error returns. A non-empty but out-of-range route fails the
// pad step, and an in-range route with an out-of-range layer ID fails the
// per-row AddToFFNOutput apply. Both sit after the live-array checks.
func TestFfnMemoryMetal_AugmentFFNMemory_ClusterIDDerivation_Ugly(t *testing.T) {
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
	ffnOutput := metal.FromValues([]float32{5, 7}, 1, 1, 2)
	mlpInput := metal.FromValues([]float32{2, 4}, 1, 1, 2)
	defer metal.Free(ffnOutput, mlpInput)

	// A non-empty route with an out-of-range cluster fails the in-call pad step,
	// bypassing the SetClusterIDs validation by assigning the field directly.
	augmenter.ClusterIDs = []int{999}
	if _, _, err := augmenter.AugmentFFNMemory(0, ffnOutput, mlpInput); err == nil {
		t.Fatal("AugmentFFNMemory(out-of-range route) error = nil, want pad failure")
	}

	// A valid generic route but an out-of-range layer ID fails the per-row apply
	// inside AddToFFNOutput.
	if err := augmenter.SetClusterIDs(nil); err != nil {
		t.Fatalf("SetClusterIDs(nil) error = %v", err)
	}
	if _, _, err := augmenter.AugmentFFNMemory(5, ffnOutput, mlpInput); err == nil {
		t.Fatal("AugmentFFNMemory(out-of-range layer) error = nil, want apply failure")
	}

	// An empty-route augmenter whose bank carries a positive hidden size but no
	// layers reaches the in-call generic re-derivation and surfaces the
	// GenericClusterIDs error (no cluster counts) rather than applying memory.
	emptyBank := &MetalFFNMemoryAugmenter{Memory: &FFNMemoryBank{HiddenSize: 2}}
	if _, _, err := emptyBank.AugmentFFNMemory(0, ffnOutput, mlpInput); err == nil {
		t.Fatal("AugmentFFNMemory(empty bank generic re-derivation) error = nil, want cluster-count failure")
	}
}
