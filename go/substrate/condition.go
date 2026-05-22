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
	// CLI flags + config-loaded values) skip any Trim+Lower work.
	if c, ok := lookupCondition(value); ok {
		return c, nil
	}
	// Case-insensitive + whitespace-tolerant path. matchConditionFold
	// walks the input bytes once — trims ASCII whitespace and
	// case-folds in-place — instead of allocating a Trim+Lower copy.
	if c, ok := matchConditionFold(value); ok {
		return c, nil
	}
	// Splitting the prefix into Operation + the input into Message
	// saves the prefix+value string concat (Pattern 10-ish): Err's
	// rendered form (Operation: Message) builds the printed string
	// at .Error() time, not at construction. The slow path drops
	// one of the two allocations.
	return "", core.E("substrate: unsupported condition", value, nil)
}

// MustNormalize parses user input and falls back to CONT when invalid.
func MustNormalize(value string) Condition {
	if c, ok := lookupCondition(value); ok {
		return c
	}
	if c, ok := matchConditionFold(value); ok {
		return c
	}
	return CONT
}

// lookupCondition returns the canonical Condition for one of the
// recognised aliases or false for any other input. Held as a single
// switch so Normalize / MustNormalize share the alias-table.
//
// When adding a new alias, mirror it into matchConditionFold's
// length-bucket switch below so the case-insensitive path stays in
// step with the exact-match path.
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

// matchConditionFold performs the same lookup as lookupCondition but
// against a whitespace-trimmed, case-folded view of value — without
// allocating the trimmed/lowered copy. Walks input once to find the
// trim window, tries the zero-alloc canonical switch on the trimmed
// substring (covers the all-lowercase-with-whitespace path), then
// dispatches on length to compare against the small set of aliases
// of that exact length under ASCII case fold (Pattern 8 — length
// bucket instead of full-table sweep).
func matchConditionFold(value string) (Condition, bool) {
	lo, hi := 0, len(value)
	for lo < hi && isASCIISpace(value[lo]) {
		lo++
	}
	for hi > lo && isASCIISpace(value[hi-1]) {
		hi--
	}
	trimmed := value[lo:hi]
	// Whitespace-only path: trimmed input matches a canonical alias
	// directly via the switch. Saves the table sweep when the only
	// transformation needed was whitespace removal.
	if c, ok := lookupCondition(trimmed); ok {
		return c, true
	}
	// Length-bucket dispatch: each canonical alias has a fixed length,
	// so a switch on len(trimmed) narrows the candidate set to at most
	// two per length without any iteration. Within a bucket, fall
	// through to equalASCIIFold byte-walks against the short candidate
	// list. The compiler turns the outer switch into a jump table.
	switch len(trimmed) {
	case 4:
		if equalASCIIFold(trimmed, "cont") {
			return CONT, true
		}
		if equalASCIIFold(trimmed, "trad") {
			return TRAD, true
		}
	case 10:
		if equalASCIIFold(trimmed, "continuous") {
			return CONT, true
		}
	case 11:
		if equalASCIIFold(trimmed, "traditional") {
			return TRAD, true
		}
	case 13:
		if equalASCIIFold(trimmed, "cont-with-gap") {
			return CONTWithGap, true
		}
		if equalASCIIFold(trimmed, "cont_with_gap") {
			return CONTWithGap, true
		}
	case 14:
		if equalASCIIFold(trimmed, "trad-no-replay") {
			return TRADNoReplay, true
		}
		if equalASCIIFold(trimmed, "trad_no_replay") {
			return TRADNoReplay, true
		}
	case 17:
		if equalASCIIFold(trimmed, "continuous-stream") {
			return CONT, true
		}
	case 18:
		if equalASCIIFold(trimmed, "traditional-runner") {
			return TRAD, true
		}
	case 19:
		if equalASCIIFold(trimmed, "continuous-with-gap") {
			return CONTWithGap, true
		}
	case 21:
		if equalASCIIFold(trimmed, "traditional-no-replay") {
			return TRADNoReplay, true
		}
	}
	return "", false
}

// isASCIISpace reports whether b is one of the five ASCII whitespace
// bytes recognised by strings.TrimSpace's fast path. Mirrors that
// inlinable set so matchConditionFold can avoid the runtime call.
func isASCIISpace(b byte) bool {
	switch b {
	case ' ', '\t', '\n', '\v', '\f', '\r':
		return true
	default:
		return false
	}
}

// equalASCIIFold reports whether s and lower are byte-equal under
// ASCII case folding. lower MUST be lowercase ASCII (all
// foldAliases entries are). Faster than strings.EqualFold because
// it skips Unicode case-folding work the alias table never needs.
func equalASCIIFold(s, lower string) bool {
	// Length-equality is the caller's contract (matchConditionFold
	// pre-checks), so the loop walks both strings in lockstep.
	for i := 0; i < len(s); i++ {
		c := s[i]
		// ASCII uppercase folds to lower by OR-ing 0x20. Any non-
		// ASCII or non-letter byte must match exactly.
		if c >= 'A' && c <= 'Z' {
			c |= 0x20
		}
		if c != lower[i] {
			return false
		}
	}
	return true
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
