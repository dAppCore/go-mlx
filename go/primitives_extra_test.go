// SPDX-Licence-Identifier: EUPL-1.2

// Additional pure-facade tests for primitives.go covering the small helpers
// the main primitives_test.go skipped: nonZeroDuration (a pure clamp) and
// ConcreteAdapter (a typed unwrap that needs only a zero-value adapter, not a
// loaded model — the assertion path is independent of GPU state).

package mlx

import (
	"testing"
	"time"

	"dappco.re/go/mlx/pkg/metal"
)

func TestNonZeroDuration_Positive_Good(t *testing.T) {
	if got := nonZeroDuration(5 * time.Millisecond); got != 5*time.Millisecond {
		t.Fatalf("nonZeroDuration(5ms) = %v, want 5ms", got)
	}
}

func TestNonZeroDuration_ClampsNonPositive_Ugly(t *testing.T) {
	if got := nonZeroDuration(0); got != time.Nanosecond {
		t.Fatalf("nonZeroDuration(0) = %v, want 1ns", got)
	}
	if got := nonZeroDuration(-time.Second); got != time.Nanosecond {
		t.Fatalf("nonZeroDuration(-1s) = %v, want 1ns", got)
	}
}

func TestConcreteAdapter_RoundTrips_Good(t *testing.T) {
	// ConcreteAdapter unwraps an inference.Adapter back to its concrete
	// *metal.LoRAAdapter. A zero-value adapter satisfies inference.Adapter
	// (TotalParams + Save) and is enough to drive the type assertion — no GPU
	// math is performed by the unwrap itself.
	concrete := &metal.LoRAAdapter{}
	got := ConcreteAdapter(concrete)
	if got != concrete {
		t.Fatalf("ConcreteAdapter returned a different pointer: got %p want %p", got, concrete)
	}
}
