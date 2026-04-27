// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import "testing"

func TestAttentionSnapshotHasQueries_Good(t *testing.T) {
	if (&AttentionSnapshot{}).HasQueries() {
		t.Fatal("HasQueries() = true, want false for empty snapshot")
	}

	snapshot := &AttentionSnapshot{
		Queries: [][][]float32{{{1, 2, 3}}},
	}
	if !snapshot.HasQueries() {
		t.Fatal("HasQueries() = false, want true when queries are present")
	}
}
