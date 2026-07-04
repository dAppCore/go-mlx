// SPDX-Licence-Identifier: EUPL-1.2

package m2

import (
	"testing"
)

// These tests close the remaining statement-coverage gaps in m2_route.go.
// compareExpertScoresDesc is the package-level comparator slices.SortFunc uses
// to order router experts; whether a given pair is visited is sort-internal, so
// the ID-tie-break legs are pinned by calling the comparator directly (the
// house style used by the existing residency limit/dtype table tests). No IO,
// no device (AX-11).

func TestM2RouteCover_compareExpertScoresDesc_TieBreaks(t *testing.T) {
	// Equal scores, a.ID < b.ID → a sorts first (the ID ascending tie-break).
	if got := compareExpertScoresDesc(expertScore{ID: 1, Score: 0.5}, expertScore{ID: 2, Score: 0.5}); got != -1 {
		t.Fatalf("compare(equal score, ID 1 vs 2) = %d, want -1", got)
	}
	// Equal scores, a.ID > b.ID → b sorts first.
	if got := compareExpertScoresDesc(expertScore{ID: 5, Score: 0.5}, expertScore{ID: 2, Score: 0.5}); got != 1 {
		t.Fatalf("compare(equal score, ID 5 vs 2) = %d, want 1", got)
	}
	// Identical score and ID → 0 (the stable equal case).
	if got := compareExpertScoresDesc(expertScore{ID: 3, Score: 0.5}, expertScore{ID: 3, Score: 0.5}); got != 0 {
		t.Fatalf("compare(identical) = %d, want 0", got)
	}
	// Sanity on the score-dominant legs so the comparator's primary order is
	// also asserted here, not only the tie-breaks.
	if got := compareExpertScoresDesc(expertScore{ID: 9, Score: 0.9}, expertScore{ID: 0, Score: 0.1}); got != -1 {
		t.Fatalf("compare(higher score first) = %d, want -1", got)
	}
	if got := compareExpertScoresDesc(expertScore{ID: 0, Score: 0.1}, expertScore{ID: 9, Score: 0.9}); got != 1 {
		t.Fatalf("compare(lower score) = %d, want 1", got)
	}
}
