// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import "testing"

func TestConfigHelpers_FirstPositiveInt_Good(t *testing.T) {
	if got := FirstPositiveInt(0, 0, 7, 3); got != 7 {
		t.Fatalf("FirstPositiveInt(0,0,7,3) = %d, want 7", got)
	}
	if got := FirstPositiveInt(5); got != 5 {
		t.Fatalf("FirstPositiveInt(5) = %d, want 5", got)
	}
}

func TestConfigHelpers_FirstPositiveInt_Bad(t *testing.T) {
	// No positive value present — must report zero, not a negative or a panic.
	if got := FirstPositiveInt(0, -1, -42); got != 0 {
		t.Fatalf("FirstPositiveInt(0,-1,-42) = %d, want 0", got)
	}
	if got := FirstPositiveInt(); got != 0 {
		t.Fatalf("FirstPositiveInt() = %d, want 0 for empty", got)
	}
}

func TestConfigHelpers_FirstPositiveInt_Ugly(t *testing.T) {
	// The unexported wrapper must agree with the exported form, and a leading
	// negative must be skipped to reach the first strictly-positive value.
	if got := firstPositiveInt(-9, 0, 11); got != 11 {
		t.Fatalf("firstPositiveInt(-9,0,11) = %d, want 11", got)
	}
	if firstPositiveInt(2, 4) != FirstPositiveInt(2, 4) {
		t.Fatal("firstPositiveInt and FirstPositiveInt disagree")
	}
}

func TestConfigHelpers_FirstNonEmptyString_Good(t *testing.T) {
	if got := FirstNonEmptyString("", "", "hidden_size", "x"); got != "hidden_size" {
		t.Fatalf("FirstNonEmptyString = %q, want %q", got, "hidden_size")
	}
}

func TestConfigHelpers_FirstNonEmptyString_Bad(t *testing.T) {
	// All empty — must report "" rather than picking arbitrary whitespace.
	if got := FirstNonEmptyString("", "", ""); got != "" {
		t.Fatalf("FirstNonEmptyString(empties) = %q, want empty", got)
	}
	if got := FirstNonEmptyString(); got != "" {
		t.Fatalf("FirstNonEmptyString() = %q, want empty for no args", got)
	}
}

func TestConfigHelpers_FirstNonEmptyString_Ugly(t *testing.T) {
	// Whitespace is non-empty: a "  " value is returned as-is (the helper does
	// not trim), and the unexported wrapper must agree with the exported form.
	if got := firstNonEmptyString("", "  ", "later"); got != "  " {
		t.Fatalf("firstNonEmptyString = %q, want two spaces (non-empty, untrimmed)", got)
	}
	if firstNonEmptyString("a", "b") != FirstNonEmptyString("a", "b") {
		t.Fatal("firstNonEmptyString and FirstNonEmptyString disagree")
	}
}
