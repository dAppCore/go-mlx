// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

// TODO(v090): replace with real Benchmark<Symbol> (AX-11 synthetic micro-benches) for per_layer_input.go.
func BenchmarkPerLayerInput_Scaffold(b *testing.B) {
	b.Skip("scaffold: per_layer_input.go benchmarks pending")
}
