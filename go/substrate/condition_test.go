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

// TestCondition_Normalize_Ugly drives Normalize on a degenerate-but-
// recoverable input: whitespace-and-mixed-case padding that the exact
// switch misses entirely and only the trim + case-fold slow path can
// resolve. Unlike _Bad (a clean error on truly unknown text), the ugly
// case still returns a valid condition with no error — the parser
// tolerates the mess rather than rejecting it.
func TestCondition_Normalize_Ugly(t *testing.T) {
	got, err := Normalize("\t  Continuous-With-Gap  \n")
	if err != nil {
		t.Fatalf("Normalize(padded mixed-case) error = %v", err)
	}
	if got != CONTWithGap {
		t.Fatalf("Normalize(padded mixed-case) = %q, want %q", got, CONTWithGap)
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

// TestCondition_RequiresReplay_Good asserts the full pre-registered
// transition-semantics truth table across all four conditions —
// RequiresReplay is the anchor predicate (TRAD is the only replay
// condition) and the table cross-checks it against the three sibling
// predicates so the whole transition contract is exercised in one pass.
func TestCondition_RequiresReplay_Good(t *testing.T) {
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

// --- All ----------------------------------------------------------------

// TestCondition_All_Good asserts All() returns the four pre-registered
// conditions in their fixed method order — the order callers depend on
// when presenting the enum (e.g. CLI --help listings).
func TestCondition_All_Good(t *testing.T) {
	want := []Condition{TRAD, CONT, TRADNoReplay, CONTWithGap}
	got := All()
	if len(got) != len(want) {
		t.Fatalf("All() len = %d, want %d", len(got), len(want))
	}
	for i, w := range want {
		if got[i] != w {
			t.Fatalf("All()[%d] = %q, want %q", i, got[i], w)
		}
	}
}

// TestCondition_All_Ugly exercises the documented read-only / shared-
// backing contract: All() hands back the same package-level slice on
// every call, so repeated calls in a tight loop must keep returning the
// full, stable, all-valid set rather than a drifting or mutated view.
func TestCondition_All_Ugly(t *testing.T) {
	first := All()
	for i := 0; i < 16; i++ {
		again := All()
		if len(again) != len(first) {
			t.Fatalf("All() call %d len = %d, want %d", i, len(again), len(first))
		}
		for j := range again {
			if again[j] != first[j] {
				t.Fatalf("All() call %d index %d = %q, want %q", i, j, again[j], first[j])
			}
			if !again[j].Valid() {
				t.Fatalf("All() call %d index %d invalid: %q", i, j, again[j])
			}
		}
	}
}

// --- MustNormalize ------------------------------------------------------

// TestCondition_MustNormalize_Bad covers the documented swallow: an
// unrecognised string never errors, it silently falls back to CONT so
// the caller always holds a runnable condition.
func TestCondition_MustNormalize_Bad(t *testing.T) {
	for _, input := range []string{"broken", "not-a-condition", "TR4D", "cont!"} {
		if got := MustNormalize(input); got != CONT {
			t.Fatalf("MustNormalize(%q) = %q, want CONT fallback", input, got)
		}
	}
}

// TestCondition_MustNormalize_Ugly drives the degenerate-but-recoverable
// path: whitespace-padded, mixed-case aliases that miss the exact switch
// still resolve to their true condition (not the CONT fallback) via the
// trim + case-fold slow path.
func TestCondition_MustNormalize_Ugly(t *testing.T) {
	cases := map[string]Condition{
		"  TRAD-No-Replay  ":      TRADNoReplay,
		"\tCONTINUOUS-WITH-GAP\n": CONTWithGap,
		"\rTraditional\r":         TRAD,
	}
	for input, want := range cases {
		if got := MustNormalize(input); got != want {
			t.Fatalf("MustNormalize(%q) = %q, want %q", input, got, want)
		}
	}
}

// --- Valid --------------------------------------------------------------

// TestCondition_Valid_Good asserts each of the four pre-registered
// canonical conditions reports Valid — the membership the rest of the
// runner trusts.
func TestCondition_Valid_Good(t *testing.T) {
	for _, c := range []Condition{TRAD, CONT, TRADNoReplay, CONTWithGap} {
		if !c.Valid() {
			t.Fatalf("%q Valid = false, want true", c)
		}
	}
}

// TestCondition_Valid_Bad asserts Valid rejects strings that look close
// to a condition but are not the exact canonical value — Valid checks
// the stored value as-is and does not normalise, so wrong-case and alias
// forms are not valid conditions.
func TestCondition_Valid_Bad(t *testing.T) {
	for _, c := range []Condition{"trad", "TRAD-NO-REPLAY", "continuous", "cont-with-gap "} {
		if c.Valid() {
			t.Fatalf("%q Valid = true, want false", c)
		}
	}
}

// --- String -------------------------------------------------------------

// TestCondition_String_Good asserts each canonical condition stringifies
// to its own label.
func TestCondition_String_Good(t *testing.T) {
	cases := map[Condition]string{
		TRAD:         "TRAD",
		CONT:         "CONT",
		TRADNoReplay: "TRAD-no-replay",
		CONTWithGap:  "CONT-with-gap",
	}
	for c, want := range cases {
		if got := c.String(); got != want {
			t.Fatalf("%q String() = %q, want %q", string(c), got, want)
		}
	}
}

// TestCondition_String_Bad asserts an unknown (non-canonical) condition
// stringifies to the empty string rather than echoing its raw value —
// String guards on Valid first.
func TestCondition_String_Bad(t *testing.T) {
	if got := Condition("unknown-condition").String(); got != "" {
		t.Fatalf("unknown String() = %q, want empty", got)
	}
}

// TestCondition_String_Ugly asserts that near-miss values which merely
// resemble a canonical label — wrong case, alias form, or whitespace-
// wrapped — still stringify to empty, because String never normalises;
// it only echoes values that are already exactly canonical.
func TestCondition_String_Ugly(t *testing.T) {
	for _, c := range []Condition{"trad", " TRAD", "TRAD ", "traditional", "TRAD-NO-REPLAY"} {
		if got := c.String(); got != "" {
			t.Fatalf("%q String() = %q, want empty", string(c), got)
		}
	}
}

// --- RequiresReplay (Good lives above as the transition-table anchor) ---

// TestCondition_RequiresReplay_Bad asserts the three continuous-state
// conditions do NOT require replay — only TRAD re-prefills.
func TestCondition_RequiresReplay_Bad(t *testing.T) {
	for _, c := range []Condition{CONT, TRADNoReplay, CONTWithGap} {
		if c.RequiresReplay() {
			t.Fatalf("%q RequiresReplay = true, want false", c)
		}
	}
}

// TestCondition_RequiresReplay_Ugly asserts an invalid / unset condition
// reports false rather than panicking — the predicate is total over any
// Condition value.
func TestCondition_RequiresReplay_Ugly(t *testing.T) {
	for _, c := range []Condition{"", "trad", "garbage"} {
		if c.RequiresReplay() {
			t.Fatalf("%q RequiresReplay = true, want false", c)
		}
	}
}

// --- UsesContinuousState ------------------------------------------------

// TestCondition_UsesContinuousState_Good asserts the three conditions
// that mount retained KV all report continuous state.
func TestCondition_UsesContinuousState_Good(t *testing.T) {
	for _, c := range []Condition{CONT, TRADNoReplay, CONTWithGap} {
		if !c.UsesContinuousState() {
			t.Fatalf("%q UsesContinuousState = false, want true", c)
		}
	}
}

// TestCondition_UsesContinuousState_Bad asserts TRAD — which re-prefills
// from scratch — does not use continuous state.
func TestCondition_UsesContinuousState_Bad(t *testing.T) {
	if TRAD.UsesContinuousState() {
		t.Fatal("TRAD UsesContinuousState = true, want false")
	}
}

// TestCondition_UsesContinuousState_Ugly asserts invalid / near-miss
// values report false without panicking — wrong case never counts as a
// continuous-state condition.
func TestCondition_UsesContinuousState_Ugly(t *testing.T) {
	for _, c := range []Condition{"", "cont", "CONT-WITH-GAP", "unknown"} {
		if c.UsesContinuousState() {
			t.Fatalf("%q UsesContinuousState = true, want false", c)
		}
	}
}

// --- RequiresArtificialGap ----------------------------------------------

// TestCondition_RequiresArtificialGap_Good asserts the two synthetic-gap
// conditions (TRAD-no-replay and CONT-with-gap) require the artificial
// wait.
func TestCondition_RequiresArtificialGap_Good(t *testing.T) {
	for _, c := range []Condition{TRADNoReplay, CONTWithGap} {
		if !c.RequiresArtificialGap() {
			t.Fatalf("%q RequiresArtificialGap = false, want true", c)
		}
	}
}

// TestCondition_RequiresArtificialGap_Bad asserts the two natural
// conditions (TRAD does its own replay, CONT has no gap) do not require
// an artificial gap.
func TestCondition_RequiresArtificialGap_Bad(t *testing.T) {
	for _, c := range []Condition{TRAD, CONT} {
		if c.RequiresArtificialGap() {
			t.Fatalf("%q RequiresArtificialGap = true, want false", c)
		}
	}
}

// TestCondition_RequiresArtificialGap_Ugly asserts invalid / near-miss
// values report false without panicking.
func TestCondition_RequiresArtificialGap_Ugly(t *testing.T) {
	for _, c := range []Condition{"", "cont-with-gap", "TRAD-NO-REPLAY", "??"} {
		if c.RequiresArtificialGap() {
			t.Fatalf("%q RequiresArtificialGap = true, want false", c)
		}
	}
}

// --- MeasuresPrefillGap -------------------------------------------------

// TestCondition_MeasuresPrefillGap_Good asserts TRAD — whose own replay
// work is the T_prefill source — is the condition that measures the gap.
func TestCondition_MeasuresPrefillGap_Good(t *testing.T) {
	if !TRAD.MeasuresPrefillGap() {
		t.Fatal("TRAD MeasuresPrefillGap = false, want true")
	}
}

// TestCondition_MeasuresPrefillGap_Bad asserts the three non-TRAD
// conditions do not source T_prefill from their own replay (CONT does no
// replay; the gap conditions wait artificially).
func TestCondition_MeasuresPrefillGap_Bad(t *testing.T) {
	for _, c := range []Condition{CONT, TRADNoReplay, CONTWithGap} {
		if c.MeasuresPrefillGap() {
			t.Fatalf("%q MeasuresPrefillGap = true, want false", c)
		}
	}
}

// TestCondition_MeasuresPrefillGap_Ugly asserts invalid / near-miss
// values report false without panicking.
func TestCondition_MeasuresPrefillGap_Ugly(t *testing.T) {
	for _, c := range []Condition{"", "trad", "TRAD ", "nonsense"} {
		if c.MeasuresPrefillGap() {
			t.Fatalf("%q MeasuresPrefillGap = true, want false", c)
		}
	}
}
