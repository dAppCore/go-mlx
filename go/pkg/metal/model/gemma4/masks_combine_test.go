// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"testing"

	"dappco.re/go/mlx/pkg/metal"
)

// TestMasks_gemma4CombineMasks_Good covers the three branches of the additive
// mask combiner: a nil base returns the extra mask unchanged, a nil extra
// returns the base unchanged, and two real masks combine element-wise via
// Minimum (the more-negative additive-mask value wins at each position, which
// is how a sliding-window band is intersected with a padding mask).
func TestMasks_gemma4CombineMasks_Good(t *testing.T) {
	requireMetalRuntime(t)

	t.Run("nil base returns extra", func(t *testing.T) {
		extra := metal.FromValues([]float32{0, -1e9}, 1, 2)
		defer metal.Free(extra)
		if got := gemma4CombineMasks(nil, extra); got != extra {
			t.Fatalf("gemma4CombineMasks(nil, extra) = %p, want extra %p", got, extra)
		}
	})

	t.Run("nil extra returns base", func(t *testing.T) {
		base := metal.FromValues([]float32{0, -1e9}, 1, 2)
		defer metal.Free(base)
		if got := gemma4CombineMasks(base, nil); got != base {
			t.Fatalf("gemma4CombineMasks(base, nil) = %p, want base %p", got, base)
		}
	})

	t.Run("both masks intersect element-wise", func(t *testing.T) {
		// base allows position 1, extra allows position 0; the intersection
		// blocks both (Minimum keeps the -1e9 at each masked position).
		base := metal.FromValues([]float32{-1e9, 0, -1e9, 0}, 1, 4)
		extra := metal.FromValues([]float32{0, 0, -1e9, -1e9}, 1, 4)
		out := gemma4CombineMasks(base, extra)
		if out == nil || !out.Valid() {
			t.Fatal("gemma4CombineMasks returned invalid array")
		}
		defer metal.Free(base, extra, out)
		if err := metal.Eval(out); err != nil {
			t.Fatalf("Eval: %v", err)
		}
		got := out.Floats()
		want := []float32{-1e9, 0, -1e9, -1e9}
		assertFloat32SliceClose(t, got, want, 1)
	})
}
