// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal_test

import (
	"fmt"

	metal "dappco.re/go/mlx/pkg/metal"
)

// FirstPositiveInt picks the first usable value while normalising nested
// config.json shapes that overload the same dimension under several aliases —
// the unset (zero) aliases fall through to the populated one.
func ExampleFirstPositiveInt() {
	headDim := metal.FirstPositiveInt(0 /* head_dim unset */, 0 /* attention_head_dim unset */, 256 /* derived */)
	fmt.Println(headDim)
	// Output: 256
}

// FirstNonEmptyString resolves a config string from a list of aliases, taking
// the first one that is actually present.
func ExampleFirstNonEmptyString() {
	modelType := metal.FirstNonEmptyString("" /* model_type unset */, "gemma3" /* architectures[0] */)
	fmt.Println(modelType)
	// Output: gemma3
}
