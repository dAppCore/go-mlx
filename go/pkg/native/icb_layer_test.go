// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

// TODO(v090): replace with real TestIcbLayer_<Symbol>_{Good,Bad,Ugly} exercising each
// public symbol in icb_layer.go (invoke + assert; skip-without-metallib for GPU ops).
func TestIcbLayer_Scaffold(t *testing.T) {
	t.Skip("scaffold: icb_layer.go tests pending")
}
