// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

// TODO(v090): replace with real Benchmark<Symbol> (AX-11 synthetic micro-benches) for decode_forward_arch_quant.go.
func BenchmarkDecodeForwardArchQuant_Scaffold(b *testing.B) {
	b.Skip("scaffold: decode_forward_arch_quant.go benchmarks pending")
}
