// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

// TODO(v090): replace with real TestQmv_<Symbol>_{Good,Bad,Ugly} exercising each
// public symbol in qmv.go (invoke + assert; skip-without-metallib for GPU ops).
func TestQmv_Scaffold(t *testing.T) {
	t.Skip("scaffold: qmv.go tests pending")
}
