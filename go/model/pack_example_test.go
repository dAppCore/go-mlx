// SPDX-Licence-Identifier: EUPL-1.2

package model_test

import (
	"fmt"

	"dappco.re/go/mlx/model"
)

// ExampleSupportsArchitecture screens architecture names against the registered
// profiles in dappco.re/go/mlx/profile — the predicate the loader uses to decide
// whether a candidate model has a known runtime profile before any bytes are
// read. Names carrying a profile (gemma-4 text, qwen3) report true; an unknown
// name reports false. The match is case-insensitive, so an upper-cased alias
// resolves the same as its canonical form.
func ExampleSupportsArchitecture() {
	for _, arch := range []string{"gemma4_text", "qwen3", "QWEN3", "totally_unknown_arch"} {
		fmt.Printf("%-22s -> %v\n", arch, model.SupportsArchitecture(arch))
	}
	// Output:
	// gemma4_text            -> true
	// qwen3                  -> true
	// QWEN3                  -> true
	// totally_unknown_arch   -> false
}
