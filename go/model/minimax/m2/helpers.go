// SPDX-Licence-Identifier: EUPL-1.2

package m2

import (
	"time"

	core "dappco.re/go"
)

// firstNonEmpty returns the first non-empty string after trimming whitespace.
//
//	value := firstNonEmpty(primary, fallback)
func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if core.Trim(value) != "" {
			return value
		}
	}
	return ""
}

// firstPositive returns the first positive value from a list.
//
//	n := firstPositive(headDim*heads, hidden)
func firstPositive(values ...int) int {
	for _, value := range values {
		if value > 0 {
			return value
		}
	}
	return 0
}

// nonZeroDuration returns d if positive, else 1 nanosecond. Kept private
// to the m2 package; the canonical exported helper lives at
// dappco.re/go/inference/bench.NonZeroDuration.
//
//	d := nonZeroDuration(elapsed)
func nonZeroDuration(d time.Duration) time.Duration {
	if d <= 0 {
		return time.Nanosecond
	}
	return d
}

// maxPositive returns the larger of a and b, but always at least the
// other operand when one is non-positive. Kept private to m2.
//
//	n := maxPositive(a, 1)
func maxPositive(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// minPositive returns the smaller of a and b, treating non-positive as
// "unset" (the other operand wins). Kept private to m2.
//
//	n := minPositive(a, b)
func minPositive(a, b int) int {
	if a <= 0 {
		return b
	}
	if b <= 0 {
		return a
	}
	if a < b {
		return a
	}
	return b
}
