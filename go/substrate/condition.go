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

// All returns the four pre-registered substrate conditions in method order.
func All() []Condition {
	return []Condition{TRAD, CONT, TRADNoReplay, CONTWithGap}
}

// Normalize parses user input into a canonical substrate condition.
func Normalize(value string) (Condition, error) {
	switch core.Lower(core.Trim(value)) {
	case "", "cont", "continuous", "continuous-stream":
		return CONT, nil
	case "trad", "traditional", "traditional-runner":
		return TRAD, nil
	case "trad-no-replay", "trad_no_replay", "traditional-no-replay":
		return TRADNoReplay, nil
	case "cont-with-gap", "cont_with_gap", "continuous-with-gap":
		return CONTWithGap, nil
	default:
		return "", core.NewError("substrate: unsupported condition: " + value)
	}
}

// MustNormalize parses user input and falls back to CONT when invalid.
func MustNormalize(value string) Condition {
	condition, err := Normalize(value)
	if err != nil {
		return CONT
	}
	return condition
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
