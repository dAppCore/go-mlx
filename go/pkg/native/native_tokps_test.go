// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"
	"testing"
	"time"
)

// TestNativeDecodeTokPerSec measures the pkg/native (no-cgo, ICB-replay) decode throughput on a REAL
// gemma4 checkpoint, to compare against the cgo pkg/metal path's tg512 baseline (~169 tok/s on e2b). This
// is the instrument the host-vs-GPU thesis actually needs — the 1-layer micro-bench is too small to show
// the ICB encode-bypass win. Env-guarded (NATIVE_BENCH_DIR); a functional perf run on a real model.
func TestNativeDecodeTokPerSec(t *testing.T) {
	dir := os.Getenv("NATIVE_BENCH_DIR")
	if dir == "" {
		t.Skip("set NATIVE_BENCH_DIR to a real gemma4 checkpoint dir")
	}
	sess, err := LoadDir(dir, 1024)
	if err != nil {
		t.Fatalf("LoadDir: %v", err)
	}
	defer func() { _ = sess.Close() }()

	prompt := []int32{2, 1841, 689, 573, 6182, 576} // short prompt; greedy decode is timing-only
	// Warmup: records the per-session ICB + primes the GPU/shaders (excluded from the measurement).
	if _, err := sess.Generate(prompt, 24, -1); err != nil {
		t.Fatalf("warmup: %v", err)
	}
	const N = 512
	start := time.Now()
	gen, err := sess.Generate(prompt, N, -1)
	wall := time.Since(start)
	if err != nil {
		t.Fatalf("measure: %v", err)
	}
	tps := float64(len(gen)) / wall.Seconds()
	t.Logf("native decode: %d tokens in %v = %.1f tok/s  (ICB eligible=%v) — cgo pkg/metal baseline ~169 tok/s",
		len(gen), wall.Round(time.Millisecond), tps, sess.icbEligible())
}
