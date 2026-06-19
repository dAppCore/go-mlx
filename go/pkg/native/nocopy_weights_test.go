// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

// TODO(v090): replace with real TestNocopyWeights_<Symbol>_{Good,Bad,Ugly} exercising each
// public symbol in nocopy_weights.go (invoke + assert; skip-without-metallib for GPU ops).
func TestNocopyWeights_Scaffold(t *testing.T) {
	t.Skip("scaffold: nocopy_weights.go tests pending")
}
