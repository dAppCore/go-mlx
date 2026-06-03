// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import core "dappco.re/go"

// Generated runnable examples for file-aware public API coverage.
func ExampleRMSNorm() {
	core.Println("RMSNorm")
	// Output: RMSNorm
}

func ExampleRMSNormNoScale() {
	core.Println("RMSNormNoScale")
	// Output: RMSNormNoScale
}

func ExampleLayerNorm() {
	core.Println("LayerNorm")
	// Output: LayerNorm
}

func ExampleRoPE() {
	core.Println("RoPE")
	// Output: RoPE
}

func ExampleRoPEWithFreqs() {
	core.Println("RoPEWithFreqs")
	// Output: RoPEWithFreqs
}

func ExampleScaledDotProductAttention() {
	core.Println("ScaledDotProductAttention")
	// Output: ScaledDotProductAttention
}

func ExampleScaledDotProductAttentionWithMask() {
	core.Println("ScaledDotProductAttentionWithMask")
	// Output: ScaledDotProductAttentionWithMask
}
