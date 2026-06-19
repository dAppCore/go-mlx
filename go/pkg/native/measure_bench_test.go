// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

// TODO(v090): replace with real Benchmark<Symbol> (AX-11 synthetic micro-benches) for measure.go.
func BenchmarkMeasure_Scaffold(b *testing.B) {
	b.Skip("scaffold: measure.go benchmarks pending")
}
