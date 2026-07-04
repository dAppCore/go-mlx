// SPDX-Licence-Identifier: EUPL-1.2

package score_test

import (
	"fmt"

	"dappco.re/go/mlx/pkg/score"
)

func ExampleIsKnownDialectContraction() {
	// Known English contractions and colloquial dialect forms have a
	// structural apostrophe — case-insensitive, so "AIN'T" matches.
	fmt.Println(score.IsKnownDialectContraction("y'all"))
	fmt.Println(score.IsKnownDialectContraction("AIN'T"))
	// A foreign phonetic-circumvention token ("Cina-Gia'a") and an
	// invented compound are NOT on the allowlist — PseudoJargonDensity
	// keeps counting them as suspicious.
	fmt.Println(score.IsKnownDialectContraction("Cina-Gia'a"))
	fmt.Println(score.IsKnownDialectContraction("frabbis'nork"))
	// Output:
	// true
	// true
	// false
	// false
}
