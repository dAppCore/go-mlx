// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

// TODO(v090): replace with real Benchmark<Symbol> (AX-11 synthetic micro-benches) for rope.go.
func BenchmarkRope_Scaffold(b *testing.B) {
	b.Skip("scaffold: rope.go benchmarks pending")
}
