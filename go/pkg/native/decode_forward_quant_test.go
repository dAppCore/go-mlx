// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

// TODO(v090): replace with real TestDecodeForwardQuant_<Symbol>_{Good,Bad,Ugly} exercising each
// public symbol in decode_forward_quant.go (invoke + assert; skip-without-metallib for GPU ops).
func TestDecodeForwardQuant_Scaffold(t *testing.T) {
	t.Skip("scaffold: decode_forward_quant.go tests pending")
}
