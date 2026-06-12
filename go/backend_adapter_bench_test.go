// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the root-package adapter constructor. NewMLXBackend is
// the canonical entry point a host process calls when wiring an
// already-loaded Metal model behind the inference.Adapter shape. The
// load itself is backend-specific (and Metal in production), but the
// constructor's option-cloning + type assertions + adapter.New wrap
// run on every host boot regardless of backend.
//
// Per AX-11 — the constructor fires once per backend instantiation but
// runs in the boot-critical path; the option append and the
// type-assertion failure branch both pay constant alloc cost.
//
// Run:    go test -bench='BenchmarkAdapterRoot' -benchmem -run='^$' ./go

package mlx

import (
	"testing"

	"dappco.re/go/inference"
)

// Sinks defeat compiler DCE. Distinct names from root_bench_test.go.
var (
	adapterBenchSinkErr     error
	adapterBenchSinkAdapter any
)

// withStubBackend swaps in a stubBackend so NewMLXBackend can run
// without a live Metal runtime. The defer restores any previously
// registered "metal" backend so concurrent benches don't interfere.
//
//	defer withStubBackend(b)()
func withStubBackend(b *testing.B) func() {
	b.Helper()
	old, hadOld := inference.Get("metal")
	backend := &stubBackend{model: &stubTextModel{}}
	inference.Register(backend)
	return func() {
		if hadOld {
			inference.Register(old)
		}
	}
}

func BenchmarkAdapterRoot_NewMLXBackend_NoLoadOptions(b *testing.B) {
	restore := withStubBackend(b)
	defer restore()
	const path = "/tmp/bench-model"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		a, err := NewMLXBackend(path)
		adapterBenchSinkAdapter = a
		adapterBenchSinkErr = err
	}
}

func BenchmarkAdapterRoot_NewMLXBackend_SingleContextOpt(b *testing.B) {
	restore := withStubBackend(b)
	defer restore()
	const path = "/tmp/bench-model"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		a, err := NewMLXBackend(path, inference.WithContextLen(4096))
		adapterBenchSinkAdapter = a
		adapterBenchSinkErr = err
	}
}

// Realistic boot-path option set — context length + a few additional
// inference loader hints. Stresses the append([]LoadOption(nil), ...)
// + append(..., WithBackend("metal")) reshape that NewMLXBackend
// does on every call.
func BenchmarkAdapterRoot_NewMLXBackend_TypicalOptSet(b *testing.B) {
	restore := withStubBackend(b)
	defer restore()
	const path = "/tmp/bench-model"
	opts := []inference.LoadOption{
		inference.WithContextLen(4096),
		inference.WithContextLen(8192),
		inference.WithContextLen(16384),
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		a, err := NewMLXBackend(path, opts...)
		adapterBenchSinkAdapter = a
		adapterBenchSinkErr = err
	}
}
