// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import (
	core "dappco.re/go"
)

// ExampleNormalizeQuantType shows how a free-form quant-type label is folded
// to the canonical lower-snake form the rest of the package keys on.
func ExampleNormalizeQuantType() {
	core.Println(NormalizeQuantType("Q4_K_M"))
	core.Println(NormalizeQuantType("Q5-K M"))
	core.Println(NormalizeQuantType("  BF16  "))
	// Output:
	// q4_k_m
	// q5_k_m
	// bf16
}
