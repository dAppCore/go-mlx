// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "time"

// piece_timing.go is a decode-piece GPU-time diagnostic: where does the per-token wall go across the three
// GPU pieces — the PLE projection, the ICB layer stack, and the head (final norm + lm_head)? Each piece does
// its own Commit+WaitUntilCompleted, so the wall-clock of the call is ~its GPU time. Off in production
// (pieceTimingOn=false → ptStart returns the zero Time and ptEnd is a no-op; the compiler inlines both to a
// bool check, no allocation). A test flips it on, resets pieceNs, decodes, and reads the split.
var (
	pieceTimingOn bool
	pieceNs       [3]int64 // [0]=PLE  [1]=ICB layer stack  [2]=head
)

func ptStart() time.Time {
	if pieceTimingOn {
		return time.Now()
	}
	return time.Time{}
}

func ptEnd(idx int, t time.Time) {
	if pieceTimingOn {
		pieceNs[idx] += int64(time.Since(t))
	}
}
