// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

// TODO(v090): replace with real Benchmark<Symbol> (AX-11 synthetic micro-benches) for qmv.go.
func BenchmarkQmv_Scaffold(b *testing.B) {
	b.Skip("scaffold: qmv.go benchmarks pending")
}
