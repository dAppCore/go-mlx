// SPDX-Licence-Identifier: EUPL-1.2

package grpo

import (
	"testing"
	"time"
)

// TestGrpoCompat_nonZeroDuration_Good covers the floor logic in
// nonZeroDuration: a positive duration passes through unchanged, but any
// zero or negative duration is floored to a single nanosecond so a
// completed run never reports a zero (or negative) Duration. The floor
// keeps the GRPOResult.Duration field truthful for instant runs and guards
// against a clock that ran backwards.
func TestGrpoCompat_nonZeroDuration_Good(t *testing.T) {
	if got := nonZeroDuration(5 * time.Second); got != 5*time.Second {
		t.Fatalf("nonZeroDuration(5s) = %v, want passthrough", got)
	}
	if got := nonZeroDuration(0); got != time.Nanosecond {
		t.Fatalf("nonZeroDuration(0) = %v, want 1ns floor", got)
	}
	if got := nonZeroDuration(-3 * time.Second); got != time.Nanosecond {
		t.Fatalf("nonZeroDuration(-3s) = %v, want 1ns floor", got)
	}
}

// TestGrpoCompat_cloneStringMap_Good covers cloneStringMap: a nil or empty
// map clones to nil (the GRPOSample.Meta omitempty contract), and a
// populated map clones to a detached copy whose mutation does not write
// back through to the source.
func TestGrpoCompat_cloneStringMap_Good(t *testing.T) {
	if got := cloneStringMap(nil); got != nil {
		t.Fatalf("cloneStringMap(nil) = %v, want nil", got)
	}
	if got := cloneStringMap(map[string]string{}); got != nil {
		t.Fatalf("cloneStringMap(empty) = %v, want nil", got)
	}
	src := map[string]string{"id": "row-1", "split": "train"}
	got := cloneStringMap(src)
	if len(got) != 2 || got["id"] != "row-1" || got["split"] != "train" {
		t.Fatalf("cloneStringMap = %v, want a faithful copy", got)
	}
	got["id"] = "mutated"
	if src["id"] != "row-1" {
		t.Fatalf("clone aliases source: src[id] = %q, want unchanged row-1", src["id"])
	}
}
