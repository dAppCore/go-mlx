// SPDX-Licence-Identifier: EUPL-1.2

package substrate

import "testing"

func TestCondition_Normalize_Good(t *testing.T) {
	cases := map[string]Condition{
		"":                    CONT,
		"cont":                CONT,
		"continuous":          CONT,
		"TRAD":                TRAD,
		"traditional":         TRAD,
		"TRAD-no-replay":      TRADNoReplay,
		"trad_no_replay":      TRADNoReplay,
		"CONT-with-gap":       CONTWithGap,
		"continuous-with-gap": CONTWithGap,
		// Continuous-stream / traditional-runner long aliases — only
		// reachable through the canonical lookupCondition switch.
		"continuous-stream":     CONT,
		"traditional-runner":    TRAD,
		"traditional-no-replay": TRADNoReplay,
		"cont_with_gap":         CONTWithGap,
	}
	for input, want := range cases {
		got, err := Normalize(input)
		if err != nil {
			t.Fatalf("Normalize(%q) error = %v", input, err)
		}
		if got != want {
			t.Fatalf("Normalize(%q) = %q, want %q", input, got, want)
		}
	}
}

// TestCondition_Normalize_CaseInsensitive drives Normalize's
// matchConditionFold path: each input is a mixed/upper-case alias that
// misses the exact-match lookupCondition switch and must fall through
// to the case-folded length-bucket dispatch. Covers every alias length
// bucket (4/10/11/13/14/17/18/19/21) on the fold path.
func TestCondition_Normalize_CaseInsensitive(t *testing.T) {
	cases := map[string]Condition{
		"CONT":                  CONT,         // bucket 4
		"Trad":                  TRAD,         // bucket 4
		"CONTINUOUS":            CONT,         // bucket 10
		"TRADITIONAL":           TRAD,         // bucket 11
		"CONT-WITH-GAP":         CONTWithGap,  // bucket 13 (hyphen)
		"CONT_WITH_GAP":         CONTWithGap,  // bucket 13 (underscore)
		"TRAD-NO-REPLAY":        TRADNoReplay, // bucket 14 (hyphen)
		"TRAD_NO_REPLAY":        TRADNoReplay, // bucket 14 (underscore)
		"CONTINUOUS-STREAM":     CONT,         // bucket 17
		"TRADITIONAL-RUNNER":    TRAD,         // bucket 18
		"CONTINUOUS-WITH-GAP":   CONTWithGap,  // bucket 19
		"TRADITIONAL-NO-REPLAY": TRADNoReplay, // bucket 21
	}
	for input, want := range cases {
		got, err := Normalize(input)
		if err != nil {
			t.Fatalf("Normalize(%q) error = %v", input, err)
		}
		if got != want {
			t.Fatalf("Normalize(%q) = %q, want %q", input, got, want)
		}
	}
}

// TestCondition_Normalize_Whitespace drives the trim window in
// matchConditionFold — leading/trailing ASCII whitespace bytes are
// stripped before the alias match. Exercises isASCIISpace's true arm
// across all five whitespace bytes and the trimmed-canonical fast path.
func TestCondition_Normalize_Whitespace(t *testing.T) {
	cases := map[string]Condition{
		"  cont  ":           CONT,         // spaces, trimmed-canonical path
		"\ttrad\t":           TRAD,         // tabs
		"\ncontinuous\n":     CONT,         // newlines
		"\rTRAD\r":           TRAD,         // carriage return + fold
		"\v\fcont-with-gap":  CONTWithGap,  // vertical tab + form feed (leading)
		"  TRAD-no-replay  ": TRADNoReplay, // pad + mixed case → fold path
	}
	for input, want := range cases {
		got, err := Normalize(input)
		if err != nil {
			t.Fatalf("Normalize(%q) error = %v", input, err)
		}
		if got != want {
			t.Fatalf("Normalize(%q) = %q, want %q", input, got, want)
		}
	}
}

func TestCondition_Normalize_Bad(t *testing.T) {
	if got, err := Normalize("broken"); err == nil || got != "" {
		t.Fatalf("Normalize(broken) = %q/%v, want error", got, err)
	}
}

func TestCondition_Normalize_Ugly(t *testing.T) {
	if got := MustNormalize("broken"); got != CONT {
		t.Fatalf("MustNormalize(broken) = %q, want CONT", got)
	}
	if got := Condition("unknown").String(); got != "" {
		t.Fatalf("unknown String() = %q, want empty", got)
	}
}

// TestCondition_MustNormalize_Good exercises MustNormalize's two
// success branches: the exact-match lookupCondition path (canonical
// alias) and the case-folded matchConditionFold path (mixed case).
// The existing _Ugly test only covers the invalid → CONT fallback.
func TestCondition_MustNormalize_Good(t *testing.T) {
	cases := map[string]Condition{
		"cont":           CONT,         // lookupCondition exact-match branch
		"trad-no-replay": TRADNoReplay, // lookupCondition exact-match branch
		"TRAD":           TRAD,         // matchConditionFold branch (mixed case)
		"Continuous":     CONT,         // matchConditionFold branch (mixed case)
		"":               CONT,         // empty → CONT via lookupCondition
	}
	for input, want := range cases {
		if got := MustNormalize(input); got != want {
			t.Fatalf("MustNormalize(%q) = %q, want %q", input, got, want)
		}
	}
}

func TestCondition_TransitionSemantics_Good(t *testing.T) {
	cases := []struct {
		condition     Condition
		replay        bool
		continuous    bool
		artificialGap bool
		measureGap    bool
	}{
		{TRAD, true, false, false, true},
		{CONT, false, true, false, false},
		{TRADNoReplay, false, true, true, false},
		{CONTWithGap, false, true, true, false},
	}
	for _, tc := range cases {
		if tc.condition.RequiresReplay() != tc.replay {
			t.Fatalf("%s RequiresReplay = %v, want %v", tc.condition, tc.condition.RequiresReplay(), tc.replay)
		}
		if tc.condition.UsesContinuousState() != tc.continuous {
			t.Fatalf("%s UsesContinuousState = %v, want %v", tc.condition, tc.condition.UsesContinuousState(), tc.continuous)
		}
		if tc.condition.RequiresArtificialGap() != tc.artificialGap {
			t.Fatalf("%s RequiresArtificialGap = %v, want %v", tc.condition, tc.condition.RequiresArtificialGap(), tc.artificialGap)
		}
		if tc.condition.MeasuresPrefillGap() != tc.measureGap {
			t.Fatalf("%s MeasuresPrefillGap = %v, want %v", tc.condition, tc.condition.MeasuresPrefillGap(), tc.measureGap)
		}
	}
}

func TestCondition_All_Bad(t *testing.T) {
	got := All()
	if len(got) != 4 {
		t.Fatalf("All() len = %d, want 4", len(got))
	}
	for _, condition := range got {
		if !condition.Valid() {
			t.Fatalf("All() contains invalid condition %q", condition)
		}
	}
}

func TestCondition_Valid_Ugly(t *testing.T) {
	if Condition("").Valid() {
		t.Fatal("empty condition Valid = true")
	}
}
