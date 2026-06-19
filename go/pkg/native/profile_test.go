// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

// TODO(v090): replace with real TestProfile_<Symbol>_{Good,Bad,Ugly} exercising each
// public symbol in profile.go (invoke + assert; skip-without-metallib for GPU ops).
func TestProfile_Scaffold(t *testing.T) {
	t.Skip("scaffold: profile.go tests pending")
}
