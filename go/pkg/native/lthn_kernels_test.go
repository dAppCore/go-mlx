// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

// TODO(v090): replace with real TestLthnKernels_<Symbol>_{Good,Bad,Ugly} exercising each
// public symbol in lthn_kernels.go (invoke + assert; skip-without-metallib for GPU ops).
func TestLthnKernels_Scaffold(t *testing.T) {
	t.Skip("scaffold: lthn_kernels.go tests pending")
}
