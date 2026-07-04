// SPDX-Licence-Identifier: EUPL-1.2

package m2

import (
	"testing"
	"time"
)

// firstNonEmpty, firstPositive, minPositive, maxPositive, and
// nonZeroDuration are unexported arithmetic leaves used by ParseConfig and
// the residency limit math. They are reached transitively by the config /
// residency tests, but the non-positive / "unset" legs only fire on inputs
// those flows never produce. These Good/Bad/Ugly units pin each branch with
// synthetic scalars (no model, no device — AX-11).

func TestHelpers_firstNonEmpty_Good(t *testing.T) {
	if got := firstNonEmpty("", "  ", "primary", "fallback"); got != "primary" {
		t.Fatalf("firstNonEmpty() = %q, want first non-blank %q", got, "primary")
	}
}

func TestHelpers_firstNonEmpty_Bad(t *testing.T) {
	// All-blank (empty + whitespace-only) collapses to the empty string.
	if got := firstNonEmpty("", "   ", "\t"); got != "" {
		t.Fatalf("firstNonEmpty(all blank) = %q, want empty", got)
	}
}

func TestHelpers_firstNonEmpty_Ugly(t *testing.T) {
	// No arguments at all — the range loop never runs, returns empty.
	if got := firstNonEmpty(); got != "" {
		t.Fatalf("firstNonEmpty() = %q, want empty on no args", got)
	}
}

func TestHelpers_firstPositive_Good(t *testing.T) {
	if got := firstPositive(0, -3, 7, 9); got != 7 {
		t.Fatalf("firstPositive() = %d, want first >0 value 7", got)
	}
}

func TestHelpers_firstPositive_Bad(t *testing.T) {
	// Every candidate non-positive → 0 (the "no head dim, no hidden" floor).
	if got := firstPositive(0, -1, -100); got != 0 {
		t.Fatalf("firstPositive(none positive) = %d, want 0", got)
	}
}

func TestHelpers_minPositive_Good(t *testing.T) {
	if got := minPositive(8, 3); got != 3 {
		t.Fatalf("minPositive(8,3) = %d, want 3", got)
	}
	if got := minPositive(3, 8); got != 3 {
		t.Fatalf("minPositive(3,8) = %d, want 3", got)
	}
}

func TestHelpers_minPositive_Bad(t *testing.T) {
	// Non-positive operands are treated as "unset" → the other operand wins.
	if got := minPositive(0, 5); got != 5 {
		t.Fatalf("minPositive(0,5) = %d, want other operand 5", got)
	}
	if got := minPositive(5, 0); got != 5 {
		t.Fatalf("minPositive(5,0) = %d, want other operand 5", got)
	}
}

func TestHelpers_minPositive_Ugly(t *testing.T) {
	// Equal positives — neither a<b nor early-out, falls through to b.
	if got := minPositive(4, 4); got != 4 {
		t.Fatalf("minPositive(4,4) = %d, want 4", got)
	}
}

func TestHelpers_maxPositive_Good(t *testing.T) {
	if got := maxPositive(2, 9); got != 9 {
		t.Fatalf("maxPositive(2,9) = %d, want 9", got)
	}
	if got := maxPositive(9, 2); got != 9 {
		t.Fatalf("maxPositive(9,2) = %d, want 9", got)
	}
}

func TestHelpers_nonZeroDuration_Good(t *testing.T) {
	if got := nonZeroDuration(5 * time.Millisecond); got != 5*time.Millisecond {
		t.Fatalf("nonZeroDuration(5ms) = %v, want 5ms unchanged", got)
	}
}

func TestHelpers_nonZeroDuration_Bad(t *testing.T) {
	// Zero and negative both clamp to 1ns so a tok/s divisor is never zero.
	if got := nonZeroDuration(0); got != time.Nanosecond {
		t.Fatalf("nonZeroDuration(0) = %v, want 1ns floor", got)
	}
	if got := nonZeroDuration(-time.Second); got != time.Nanosecond {
		t.Fatalf("nonZeroDuration(-1s) = %v, want 1ns floor", got)
	}
}
