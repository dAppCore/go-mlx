// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

// TODO(v090): replace with real TestModelQuant_<Symbol>_{Good,Bad,Ugly} exercising each
// public symbol in model_quant.go (invoke + assert; skip-without-metallib for GPU ops).
func TestModelQuant_Scaffold(t *testing.T) {
	t.Skip("scaffold: model_quant.go tests pending")
}
