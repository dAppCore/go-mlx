// SPDX-Licence-Identifier: EUPL-1.2

package score_test

import (
	"fmt"

	"dappco.re/go/mlx/pkg/score"
)

func ExampleDoubleMetaphone() {
	// Cross-orthographic spellings collapse to the same primary code.
	p1, _, _ := score.DoubleMetaphone("Smith")
	p2, _, _ := score.DoubleMetaphone("Smyth")
	fmt.Println("Smith primary:", p1)
	fmt.Println("Smyth primary:", p2)
	fmt.Println("equal:", p1 == p2)
	// Output:
	// Smith primary: SM0
	// Smyth primary: SM0
	// equal: true
}

func ExamplePhoneticEquivalent() {
	fmt.Println(score.PhoneticEquivalent("Catherine", "Katherine"))
	fmt.Println(score.PhoneticEquivalent("dog", "cat"))
	// Output:
	// true
	// false
}

func ExamplePhoneticContains() {
	// The LEK-class artifact: "Cina-Gia'a" carries "China" phonetically
	// even though no character substring of "China" appears.
	fmt.Println(score.PhoneticContains("Cina-Gia'a", "China"))
	// A single-phoneme needle is rejected (floor = 2 phonemes).
	fmt.Println(score.PhoneticContains("anything", "I"))
	// Output:
	// true
	// false
}
