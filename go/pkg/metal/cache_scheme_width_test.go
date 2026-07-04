// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"testing"

	scheme "dappco.re/go/mlx/pkg/scheme"
)

// TestRegisterCachePreservingWidth_DriverKeepsPlannerWidth_Good pins the
// width-stripping bug class: pkg/metal's driver registrations overwrite the
// registry's width-carrying builtin stubs, and without forwarding, the memory
// planner silently sizes those modes on the ×2 default lane. Post-init, every
// driver-registered mode with a builtin width must still expose it AND still
// expose the driver compute surface.
func TestRegisterCachePreservingWidth_DriverKeepsPlannerWidth_Good(t *testing.T) {
	for _, tc := range []struct {
		mode    string
		num     uint64
		den     uint64
		roundUp bool
	}{
		{"q8", 1, 1, false},
		{"k-q8-v-q4", 3, 4, false},
		{"turboquant", 7, 16, true},
		{"default", 2, 1, false},
		{"fixed", 2, 1, false},
		{"paged", 2, 1, false},
	} {
		resolved, ok := scheme.CacheFor(tc.mode)
		if !ok {
			t.Fatalf("%s: not registered", tc.mode)
		}
		width, ok := resolved.(scheme.CacheWidth)
		if !ok {
			t.Fatalf("%s: driver registration stripped scheme.CacheWidth (%T)", tc.mode, resolved)
		}
		num, den, roundUp := width.KVBytesPerElement()
		if num != tc.num || den != tc.den || roundUp != tc.roundUp {
			t.Fatalf("%s: width = %d/%d roundUp=%v, want %d/%d roundUp=%v", tc.mode, num, den, roundUp, tc.num, tc.den, tc.roundUp)
		}
		if _, ok := resolved.(CacheCompute); !ok {
			t.Fatalf("%s: width preservation masked the driver CacheCompute surface (%T)", tc.mode, resolved)
		}
	}
}

// TestRegisterCachePreservingWidth_NoPriorWidth_Bad covers the modes with no
// builtin width (compaction, mla-latent): they register as-is and the width
// probe misses cleanly.
func TestRegisterCachePreservingWidth_NoPriorWidth_Bad(t *testing.T) {
	resolved, ok := scheme.CacheFor(string(KVCacheModeCompaction))
	if !ok {
		t.Fatal("compaction not registered")
	}
	if _, ok := resolved.(scheme.CacheWidth); ok {
		t.Fatalf("compaction unexpectedly carries a width (%T) — did builtin grow one? update this pin", resolved)
	}
	if _, ok := resolved.(CacheCompute); !ok {
		t.Fatalf("compaction lost its compute surface (%T)", resolved)
	}
}
