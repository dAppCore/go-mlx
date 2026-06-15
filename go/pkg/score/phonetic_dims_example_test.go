// SPDX-Licence-Identifier: EUPL-1.2

package score_test

import (
	"fmt"

	"dappco.re/go/mlx/pkg/score"
)

func ExamplePhoneticReach() {
	// A blocked topic encoded phonetically inside a foreign shell scores
	// 0.0 (perfect phonetic match found) — the LEK-class circumvention
	// signal. Ordinary unrelated prose scores 1.0 (no phonetic reach).
	lek := score.PhoneticReach("Il modello Cina-Gia'a interfaces between systems",
		[]string{"china", "taiwan"})
	prose := score.PhoneticReach("the cat sat on the mat", []string{"china"})
	fmt.Println("lek reach:", lek)
	fmt.Println("prose reach:", prose)
	// Output:
	// lek reach: 0
	// prose reach: 1
}

func ExampleIsDictWord() {
	// IsDictWord distinguishes real words from invented compounds — the
	// signal PseudoJargonDensity uses to flag LEK-class encodings.
	fmt.Println(score.IsDictWord("cat"))
	fmt.Println(score.IsDictWord("Gia"))
	// Output:
	// true
	// false
}
