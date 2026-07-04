// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"math"
	"testing"
)

// TestLoRAMerge_FoldsDelta_Good covers mergeLinearLoRA / mergedLoRAWeight (both
// 0%): the LoRAAdapter.Merge() test in the synthetic suite only walks the
// nil-base skip branch, so a real layer with non-zero A/B never folds. Here a
// dense base + an explicit rank-1 delta folds into the base weight in place,
// and the merged weight must equal base + Scale*(B@A).
func TestLoRAMerge_FoldsDelta_Good(t *testing.T) {
	requireMetalRuntime(t)

	// Base weight [out=2, in=2] = identity-ish.
	base := NewLinear(FromValues([]float32{1, 0, 0, 1}, 2, 2), nil)
	lora := NewLoRALinear(base, 1, 2.0) // rank=1, alpha=2 → Scale=2
	// NewLoRALinear zero-inits B (identity start); set B non-zero so the fold is
	// observable. A is [rank=1, in=2], B is [out=2, rank=1].
	Free(lora.A, lora.B)
	lora.A = FromValues([]float32{1, 1}, 1, 2) // [1,2]
	lora.B = FromValues([]float32{1, 0}, 2, 1) // [2,1]
	Materialize(lora.A, lora.B)
	// delta = Scale * (B @ A) = 2 * [[1,1],[0,0]] = [[2,2],[0,0]]
	// merged = base + delta = [[3,2],[0,1]]

	// Merge() folds layer.Base.LoRA, so the adapter must wire the LoRA onto the
	// base linear (the field Forward/mergeLinearLoRA route through).
	base.LoRA = lora
	adapter := &LoRAAdapter{Layers: map[string]*LoRALinear{"q_proj": lora}}
	adapter.Merge()

	if base.LoRA != nil {
		t.Fatal("after Merge, base.LoRA should be nil")
	}
	if base.Scales != nil || base.Bits != 0 || base.GroupSize != 0 {
		t.Fatal("after Merge, dense base should carry no quant metadata")
	}
	got := base.Weight.Floats()
	want := []float32{3, 2, 0, 1}
	if len(got) != len(want) {
		t.Fatalf("merged weight len = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 1e-4 {
			t.Fatalf("merged weight[%d] = %v, want %v", i, got[i], want[i])
		}
	}
	if len(adapter.Layers) != 0 {
		t.Fatalf("Merge should clear adapter.Layers, got %d", len(adapter.Layers))
	}
}

// TestLoadLinear_DenseAndAbsent_Good covers LoadLinear (0%): the dense (no
// scales) projection-build branch and the absent-weight nil return. The
// quantized branch needs a packed checkpoint; the dense + nil branches are the
// model-agnostic core.
func TestLoadLinear_DenseAndAbsent_Good(t *testing.T) {
	requireMetalRuntime(t)

	w := FromValues([]float32{1, 2, 3, 4, 5, 6}, 2, 3)
	b := FromValues([]float32{0.5, -0.5}, 2)
	defer Free(w, b)
	weights := map[string]*Array{
		"decoder.0.in_proj.weight": w,
		"decoder.0.in_proj.bias":   b,
	}

	// Dense build: weight present, no scales → NewLinear path.
	lin := LoadLinear(weights, "decoder.0.in_proj", &QuantizationConfig{GroupSize: 64, Bits: 4})
	if lin == nil {
		t.Fatal("LoadLinear returned nil for a present dense weight")
	}
	if lin.Weight == nil {
		t.Fatal("LoadLinear built a Linear with a nil weight")
	}
	if lin.Scales != nil {
		t.Fatal("dense LoadLinear should not set Scales")
	}

	// Absent weight → nil (the caller decides if that is fatal).
	if got := LoadLinear(weights, "decoder.0.missing", nil); got != nil {
		t.Fatal("LoadLinear for an absent weight returned non-nil")
	}
}
