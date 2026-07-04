// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for split_native_runtime.go — the local Metal-side split
// runtime path. Per AX-11 — nativeSplitLocalRuntimeReady fires on
// every public method entry (Prefill / ForwardAttention / Sample /
// DecodeToken). ForwardAttention + Sample dominate the steady-state
// decode loop (one of each per layer per token), so any per-call
// allocation compounds linearly with generation length × layer count.
//
// Run:    go test -bench='BenchmarkNativeSplitLocalRuntime' -benchmem -run='^$' ./go

package mlx

import (
	"context"
	"testing"
)

// Sinks defeat compiler DCE.
var (
	nativeSplitBenchSinkErr error
)

// nativeSplitBenchRuntime returns a runtime in the steady-state shape
// the ready-check sees mid-decode (slicePath populated, no model load
// attempted). The benchmarks exercise the *guard* path; the actual
// metal load/decode is gated behind cgo and outside the benchmark
// surface here.
func nativeSplitBenchRuntime() *NativeSplitLocalRuntime {
	return &NativeSplitLocalRuntime{
		slicePath: "/fake/slice/path",
	}
}

// --- nativeSplitLocalRuntimeReady: the per-call guard ---

func BenchmarkNativeSplitLocalRuntime_Ready_Background(b *testing.B) {
	runtime := nativeSplitBenchRuntime()
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		nativeSplitBenchSinkErr = nativeSplitLocalRuntimeReady(ctx, runtime)
	}
}

func BenchmarkNativeSplitLocalRuntime_Ready_NilCtx(b *testing.B) {
	runtime := nativeSplitBenchRuntime()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// nil-ctx path — exercises the ctx=context.Background()
		// normalisation branch the helper has to carry for callers
		// that didn't plumb a real context.
		nativeSplitBenchSinkErr = nativeSplitLocalRuntimeReady(nil, runtime) //nolint:staticcheck // SA1012: nil ctx is the path under test
	}
}
