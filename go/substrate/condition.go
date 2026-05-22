// SPDX-Licence-Identifier: EUPL-1.2

// Package substrate defines the pre-registered substrate-shift experiment
// conditions from host-uk/core/plans/rfc/research/experiments/worf/02-method.md.
package substrate

import core "dappco.re/go"

// Condition is one substrate level from the substrate-shift experiment.
type Condition string

const (
	// TRAD re-prefills the full conversation prefix on each turn.
	TRAD Condition = "TRAD"
	// CONT mounts the prior KV state directly with no artificial gap.
	CONT Condition = "CONT"
	// TRADNoReplay waits for the TRAD prefill gap but keeps the CONT KV state.
	TRADNoReplay Condition = "TRAD-no-replay"
	// CONTWithGap keeps the CONT KV state but waits for the TRAD prefill gap.
	CONTWithGap Condition = "CONT-with-gap"
)

// allConditions is the package-init shared slice backing All(). The
// substrate-shift experiment treats the four pre-registered conditions
// as a fixed enum — sharing one allocation across every All() call
// drops the 64 B/op slice alloc on the hot transition-sweep path
// (BenchmarkConditionTransition_FourConditions calls All() once at
// setup but the runner re-validates conditions on every turn, so the
// substrate.All() form has been observed in tight loops). Treat the
// returned slice as read-only; callers needing mutation must slices.Clone.
var allConditions = []Condition{TRAD, CONT, TRADNoReplay, CONTWithGap}

// All returns the four pre-registered substrate conditions in method order.
// The returned slice is read-only — callers must not mutate it.
//
//	for _, c := range substrate.All() { c.Valid() }
func All() []Condition {
	return allConditions
}

// Normalize parses user input into a canonical substrate condition.
func Normalize(value string) (Condition, error) {
	// Fast path: already-canonical inputs (the dominant case for
	// CLI flags + config-loaded values) skip the Trim+Lower
	// allocation pair entirely.
	if c, ok := lookupCondition(value); ok {
		return c, nil
	}
	if c, ok := lookupCondition(core.Lower(core.Trim(value))); ok {
		return c, nil
	}
	return "", core.NewError("substrate: unsupported condition: " + value)
}

// MustNormalize parses user input and falls back to CONT when invalid.
func MustNormalize(value string) Condition {
	if c, ok := lookupCondition(value); ok {
		return c
	}
	if c, ok := lookupCondition(core.Lower(core.Trim(value))); ok {
		return c
	}
	return CONT
}

// lookupCondition returns the canonical Condition for one of the
// recognised aliases or false for any other input. Held as a single
// switch so Normalize / MustNormalize share the alias-table.
func lookupCondition(value string) (Condition, bool) {
	switch value {
	case "", "cont", "continuous", "continuous-stream":
		return CONT, true
	case "trad", "traditional", "traditional-runner":
		return TRAD, true
	case "trad-no-replay", "trad_no_replay", "traditional-no-replay":
		return TRADNoReplay, true
	case "cont-with-gap", "cont_with_gap", "continuous-with-gap":
		return CONTWithGap, true
	default:
		return "", false
	}
}

// Valid reports whether the condition is one of the four pre-registered levels.
func (c Condition) Valid() bool {
	switch c {
	case TRAD, CONT, TRADNoReplay, CONTWithGap:
		return true
	default:
		return false
	}
}

// String returns the canonical condition label.
func (c Condition) String() string {
	if !c.Valid() {
		return ""
	}
	return string(c)
}

// RequiresReplay reports whether the next turn must re-prefill the full prefix.
func (c Condition) RequiresReplay() bool {
	return c == TRAD
}

// UsesContinuousState reports whether the next turn should mount retained KV.
func (c Condition) UsesContinuousState() bool {
	switch c {
	case CONT, TRADNoReplay, CONTWithGap:
		return true
	default:
		return false
	}
}

// RequiresArtificialGap reports whether the runner must wait for T_prefill
// without doing replay work.
func (c Condition) RequiresArtificialGap() bool {
	switch c {
	case TRADNoReplay, CONTWithGap:
		return true
	default:
		return false
	}
}

// MeasuresPrefillGap reports whether the condition's own replay work is the
// source for T_prefill samples.
func (c Condition) MeasuresPrefillGap() bool {
	return c == TRAD
}
