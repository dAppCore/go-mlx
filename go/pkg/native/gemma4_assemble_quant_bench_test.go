// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

// TODO(v090): replace with real Benchmark<Symbol> (AX-11 synthetic micro-benches) for gemma4_assemble_quant.go.
func BenchmarkGemma4AssembleQuant_Scaffold(b *testing.B) {
	b.Skip("scaffold: gemma4_assemble_quant.go benchmarks pending")
}
