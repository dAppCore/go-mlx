// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

// TODO(v090): replace with real TestLoadShared_<Symbol>_{Good,Bad,Ugly} exercising each
// public symbol in load_shared.go (invoke + assert; skip-without-metallib for GPU ops).
func TestLoadShared_Scaffold(t *testing.T) {
	t.Skip("scaffold: load_shared.go tests pending")
}
