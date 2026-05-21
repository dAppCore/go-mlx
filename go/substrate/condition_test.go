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
