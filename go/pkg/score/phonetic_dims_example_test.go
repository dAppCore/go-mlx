// SPDX-Licence-Identifier: EUPL-1.2

package score_test

import (
	"fmt"

	"dappco.re/go/mlx/pkg/score"
)

// ExampleSyllableCount counts syllables via the CMU dict, falling back
// to a vowel-cluster heuristic for out-of-dict words. An empty string
// has no syllables.
func ExampleSyllableCount() {
	fmt.Println(score.SyllableCount("cat sat mat"))
	fmt.Println(score.SyllableCount("family"))
	fmt.Println(score.SyllableCount(""))
	// Output:
	// 3
	// 3
	// 0
}

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

// ExampleSigilEntropy measures Shannon entropy (bits/byte) over a
// sliding window. Empty input has zero entropy.
func ExampleSigilEntropy() {
	fmt.Println(score.SigilEntropy("", 32))
	// Output:
	// 0
}

// ExampleRhymeDensity scores end-rhyme across lines. A rhyming couplet
// (cat / mat) scores 1.0; a single line has no pair to rhyme.
func ExampleRhymeDensity() {
	fmt.Println(score.RhymeDensity("the cat\nsat on the mat"))
	fmt.Println(score.RhymeDensity("just one line here"))
	// Output:
	// 1
	// 0
}

// ExampleAlliterationDensity scores shared leading consonants. Empty
// input is zero.
func ExampleAlliterationDensity() {
	fmt.Println(score.AlliterationDensity(""))
	// Output:
	// 0
}

// ExampleAssonanceDensity scores shared stressed vowels. Empty input is
// zero.
func ExampleAssonanceDensity() {
	fmt.Println(score.AssonanceDensity(""))
	// Output:
	// 0
}

// ExamplePunDensity scores homophone play. "sea see" is a perfect
// homophone pair (1.0); ordinary prose has none.
func ExamplePunDensity() {
	fmt.Println(score.PunDensity("sea see"))
	fmt.Println(score.PunDensity("the cat sat on the mat"))
	// Output:
	// 1
	// 0
}

// ExamplePseudoJargonDensity scores the proportion of invented-looking
// compounds. Ordinary prose with no compounds scores zero.
func ExamplePseudoJargonDensity() {
	fmt.Println(score.PseudoJargonDensity("the cat sat on the mat"))
	// Output:
	// 0
}

// ExampleMeterRegularity scores stress-pattern regularity. A perfectly
// iambic line scores 1.0; input below the 4-syllable floor scores 0.
func ExampleMeterRegularity() {
	fmt.Println(score.MeterRegularity("the cat the dog the sun the moon the war the night"))
	fmt.Println(score.MeterRegularity("cat sat"))
	// Output:
	// 1
	// 0
}
