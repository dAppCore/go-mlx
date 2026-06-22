// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"
	"testing"
)

// TestNativeDecodeReproducibilityOneLoad discriminates the reproducible token-9 divergence seen across
// session loads: is it an order-dependent STATE bug (decode #1 leaves global state corrupting decode #2),
// or load-time (alignment / a per-load global)? Both host path, ONE model load, two prefills at a reset
// position. Differ → within-load state corruption. Match → the divergence is per-LOAD, not within-load.
func TestNativeDecodeReproducibilityOneLoad(t *testing.T) {
	dir := os.Getenv("NATIVE_BENCH_DIR")
	if dir == "" {
		t.Skip("set NATIVE_BENCH_DIR to a real gemma4 checkpoint dir")
	}
	pleResidentDisabled = true // host path for both decodes
	defer func() { pleResidentDisabled = false }()
	s, err := LoadDir(dir, 256)
	if err != nil {
		t.Fatalf("LoadDir: %v", err)
	}
	defer func() { _ = s.Close() }()
	prompt := []int32{2, 1841, 689, 573, 6182, 576}
	a, err := s.Generate(prompt, 48, -1)
	if err != nil {
		t.Fatalf("gen a: %v", err)
	}
	s.pos = 0 // reset position: a second fresh prefill on the SAME load (overwrites the KV cache)
	b, err := s.Generate(prompt, 48, -1)
	if err != nil {
		t.Fatalf("gen b: %v", err)
	}
	for i := range a {
		if i < len(b) && a[i] != b[i] {
			t.Fatalf("ONE LOAD, two prefills diverge at token %d (%d != %d) → order-dependent STATE within a load", i, a[i], b[i])
		}
	}
	t.Logf("one-load two-prefill reproducible over %d tokens → the cross-LOAD divergence is per-load (alignment / per-load global), not within-load state", len(a))
}
