// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import "testing"

// TestCompiledLayer_gemma4AttnBandFor_Good covers the compiled-decode attention
// band sizing: the band starts at 256 and doubles until it covers the needed
// key span, then clamps to the cache capacity. It is a pure size calculation,
// so the table drives every branch (already >= 256, one doubling, several
// doublings, and the capacity clamp) with no Metal runtime.
func TestCompiledLayer_gemma4AttnBandFor_Good(t *testing.T) {
	cases := []struct {
		name     string
		needed   int32
		capacity int32
		want     int32
	}{
		{name: "needed below floor stays at 256", needed: 10, capacity: 4096, want: 256},
		{name: "needed equal to floor stays at 256", needed: 256, capacity: 4096, want: 256},
		{name: "one doubling to 512", needed: 257, capacity: 4096, want: 512},
		{name: "several doublings to 2048", needed: 1500, capacity: 4096, want: 2048},
		{name: "clamped down to capacity", needed: 5000, capacity: 4096, want: 4096},
		{name: "floor itself clamped below 256", needed: 10, capacity: 64, want: 64},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := gemma4AttnBandFor(tc.needed, tc.capacity); got != tc.want {
				t.Fatalf("gemma4AttnBandFor(%d, %d) = %d, want %d", tc.needed, tc.capacity, got, tc.want)
			}
		})
	}
}
