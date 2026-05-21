// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for substrate primitives not already covered by
// condition_bench_test.go. The existing file benches Normalize on the
// alias path + the four-condition transition sweep; this file fills in
// the gaps: canonical Normalize input, MustNormalize, Valid/String,
// All(), and the individual transition predicates so codex can read
// per-method cost rather than only the sweep aggregate.
//
// Per AX-11 — Normalize fires on every condition-bearing CLI flag +
// config-load (substrate-shift experiment runner parses condition once
// per run, but the runner re-validates the condition on each turn via
// Valid() / RequiresReplay()).
//
// Run:    go test -bench='BenchmarkSubstrate' -benchmem -run='^$' ./go/substrate

package substrate

import "testing"

// Sinks defeat compiler DCE. Keep names distinct from
// condition_bench_test.go (no sinks declared there, so no collision,
// but namespacing keeps future churn safe).
var (
	substrateBenchSinkCond  Condition
	substrateBenchSinkErr   error
	substrateBenchSinkBool  bool
	substrateBenchSinkStr   string
	substrateBenchSinkConds []Condition
)

// --- Normalize on canonical (non-alias) inputs. The existing file
// already benches the alias path; these cover the fast-path branches
// that don't trigger Lower/Trim work beyond the minimum.

func BenchmarkSubstrate_Normalize_CanonicalCONT(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		substrateBenchSinkCond, substrateBenchSinkErr = Normalize("cont")
	}
}

func BenchmarkSubstrate_Normalize_CanonicalTRAD(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		substrateBenchSinkCond, substrateBenchSinkErr = Normalize("trad")
	}
}

func BenchmarkSubstrate_Normalize_EmptyDefaultsToCONT(b *testing.B) {
	// Empty input — exercises the implicit default CONT branch + Lower
	// short-circuit.
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		substrateBenchSinkCond, substrateBenchSinkErr = Normalize("")
	}
}

func BenchmarkSubstrate_Normalize_MixedCase(b *testing.B) {
	// Mixed case — exercises Lower over a moderate-length string.
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		substrateBenchSinkCond, substrateBenchSinkErr = Normalize("Continuous-With-Gap")
	}
}

func BenchmarkSubstrate_Normalize_LeadingTrailingWhitespace(b *testing.B) {
	// Whitespace pads — exercises Trim before Lower.
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		substrateBenchSinkCond, substrateBenchSinkErr = Normalize("  trad-no-replay  ")
	}
}

func BenchmarkSubstrate_Normalize_UnsupportedError(b *testing.B) {
	// Worst-case branch: scans every case in the switch then falls
	// through to the error path with NewError allocation.
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		substrateBenchSinkCond, substrateBenchSinkErr = Normalize("unsupported-condition-string")
	}
}

// --- MustNormalize — wraps Normalize + falls back to CONT on error.
// Hit in callers that have already committed to running a condition.

func BenchmarkSubstrate_MustNormalize_Valid(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		substrateBenchSinkCond = MustNormalize("trad")
	}
}

func BenchmarkSubstrate_MustNormalize_FallbackOnError(b *testing.B) {
	// Forces the Normalize error branch + fallback to CONT.
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		substrateBenchSinkCond = MustNormalize("broken")
	}
}

// --- All — slice allocation per call. Caller-side defensive copy
// pattern; existing sweep bench uses this but doesn't time it alone.

func BenchmarkSubstrate_All(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		substrateBenchSinkConds = All()
	}
}

// --- Valid — single-switch predicate hit on every transition check.

func BenchmarkSubstrate_Valid_TRAD(b *testing.B) {
	c := TRAD
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		substrateBenchSinkBool = c.Valid()
	}
}

func BenchmarkSubstrate_Valid_CONT(b *testing.B) {
	c := CONT
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		substrateBenchSinkBool = c.Valid()
	}
}

func BenchmarkSubstrate_Valid_InvalidEmpty(b *testing.B) {
	c := Condition("")
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		substrateBenchSinkBool = c.Valid()
	}
}

func BenchmarkSubstrate_Valid_InvalidUnknown(b *testing.B) {
	c := Condition("unknown-condition")
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		substrateBenchSinkBool = c.Valid()
	}
}

// --- String — guarded conversion via Valid. The unknown branch
// short-circuits to "" with no string conversion cost.

func BenchmarkSubstrate_String_TRAD(b *testing.B) {
	c := TRAD
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		substrateBenchSinkStr = c.String()
	}
}

func BenchmarkSubstrate_String_Invalid(b *testing.B) {
	c := Condition("unknown")
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		substrateBenchSinkStr = c.String()
	}
}

// --- Individual transition predicates. The existing sweep covers
// all four in one bench; these break out the per-call cost so codex
// can see which predicate is cheapest (simple equality) vs which
// must scan a 3-condition switch list.

func BenchmarkSubstrate_RequiresReplay_TRAD(b *testing.B) {
	c := TRAD
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		substrateBenchSinkBool = c.RequiresReplay()
	}
}

func BenchmarkSubstrate_RequiresReplay_CONT(b *testing.B) {
	c := CONT
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		substrateBenchSinkBool = c.RequiresReplay()
	}
}

func BenchmarkSubstrate_UsesContinuousState_CONT(b *testing.B) {
	c := CONT
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		substrateBenchSinkBool = c.UsesContinuousState()
	}
}

func BenchmarkSubstrate_UsesContinuousState_TRAD(b *testing.B) {
	c := TRAD
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		substrateBenchSinkBool = c.UsesContinuousState()
	}
}

func BenchmarkSubstrate_RequiresArtificialGap_CONTWithGap(b *testing.B) {
	c := CONTWithGap
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		substrateBenchSinkBool = c.RequiresArtificialGap()
	}
}

func BenchmarkSubstrate_RequiresArtificialGap_TRAD(b *testing.B) {
	c := TRAD
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		substrateBenchSinkBool = c.RequiresArtificialGap()
	}
}

func BenchmarkSubstrate_MeasuresPrefillGap_TRAD(b *testing.B) {
	c := TRAD
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		substrateBenchSinkBool = c.MeasuresPrefillGap()
	}
}

func BenchmarkSubstrate_MeasuresPrefillGap_CONT(b *testing.B) {
	c := CONT
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		substrateBenchSinkBool = c.MeasuresPrefillGap()
	}
}
