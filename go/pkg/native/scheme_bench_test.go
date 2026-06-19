// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

// TODO(v090): replace with real Benchmark<Symbol> (AX-11 synthetic micro-benches) for scheme.go.
func BenchmarkScheme_Scaffold(b *testing.B) {
	b.Skip("scaffold: scheme.go benchmarks pending")
}
