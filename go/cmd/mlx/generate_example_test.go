// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	core "dappco.re/go"
	mlx "dappco.re/go/mlx"
)

// Example_resolvedDraftBlock shows how the -draft-block flag resolves for the
// MTP lane: a positive value is used verbatim, and 0 (or any non-positive)
// falls back to the engine's default draft block so the loader picks it.
//
//	lthn-mlx generate -draft-block 0 <model>   # → engine default
func Example_resolvedDraftBlock() {
	core.Println(resolvedDraftBlock(7))
	core.Println(resolvedDraftBlock(0) == mlx.MTPDefaultDraftBlock)
	// Output:
	// 7
	// true
}
