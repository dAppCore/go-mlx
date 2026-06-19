// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

// TODO(v090): replace with real TestMistralAssemble_<Symbol>_{Good,Bad,Ugly} exercising each
// public symbol in mistral_assemble.go (invoke + assert; skip-without-metallib for GPU ops).
func TestMistralAssemble_Scaffold(t *testing.T) {
	t.Skip("scaffold: mistral_assemble.go tests pending")
}
