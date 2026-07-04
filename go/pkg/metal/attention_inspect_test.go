// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"testing"
)

func TestAttentionCacheIndexByLayer_DefaultModel_Good(t *testing.T) {
	got := attentionCacheIndexByLayer(&fakeModel{numLayers: 4}, 4, 4)
	want := []int{0, 1, 2, 3}
	for i, wantIdx := range want {
		if got[i] != wantIdx {
			t.Fatalf("cache index for layer %d = %d, want %d", i, got[i], wantIdx)
		}
	}
}
