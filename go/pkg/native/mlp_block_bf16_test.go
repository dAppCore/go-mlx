// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

// TODO(v090): replace with real TestMlpBlockBf16_<Symbol>_{Good,Bad,Ugly} exercising each
// public symbol in mlp_block_bf16.go (invoke + assert; skip-without-metallib for GPU ops).
func TestMlpBlockBf16_Scaffold(t *testing.T) {
	t.Skip("scaffold: mlp_block_bf16.go tests pending")
}
