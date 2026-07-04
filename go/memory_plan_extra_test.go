// SPDX-Licence-Identifier: EUPL-1.2

// Extra coverage for the small pure helpers in memory_plan.go that the main
// suite leaves untouched: minPositive / maxPositive. These are private root
// helpers retained for the small-model smoke callers.

package mlx

import "testing"

func TestMinPositive_Good(t *testing.T) {
	if got := minPositive(3, 7); got != 3 {
		t.Fatalf("minPositive(3,7) = %d, want 3", got)
	}
	if got := minPositive(9, 4); got != 4 {
		t.Fatalf("minPositive(9,4) = %d, want 4", got)
	}
}

func TestMinPositive_TreatsNonPositiveAsUnset_Ugly(t *testing.T) {
	// a <= 0 → the other operand wins.
	if got := minPositive(0, 5); got != 5 {
		t.Fatalf("minPositive(0,5) = %d, want 5", got)
	}
	// b <= 0 → a wins.
	if got := minPositive(8, -1); got != 8 {
		t.Fatalf("minPositive(8,-1) = %d, want 8", got)
	}
	// equal positive values return that value.
	if got := minPositive(6, 6); got != 6 {
		t.Fatalf("minPositive(6,6) = %d, want 6", got)
	}
}

func TestMaxPositive_Good(t *testing.T) {
	if got := maxPositive(3, 7); got != 7 {
		t.Fatalf("maxPositive(3,7) = %d, want 7", got)
	}
	if got := maxPositive(9, 4); got != 9 {
		t.Fatalf("maxPositive(9,4) = %d, want 9", got)
	}
	// maxPositive does not treat non-positive specially: it is a plain max.
	if got := maxPositive(-2, -5); got != -2 {
		t.Fatalf("maxPositive(-2,-5) = %d, want -2", got)
	}
}
