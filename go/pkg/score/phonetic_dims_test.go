// SPDX-Licence-Identifier: EUPL-1.2

package score

import (
	"testing"
)

// CMU-dict symbol tests (Lookup / IsDictWord / IsVowelPhoneme /
// PhonemeStress) live in cmudict_test.go alongside their source.
// This file covers the phonetic_dims.go public dimensions plus the
// private syllablesFor wrapper they share.

// --- syllablesFor (private wrapper, exercised here) ---

// TestPhoneticDims_SyllablesForMixedCase — syllablesFor is the non-upper wrapper
// over syllablesForUpper (production hot paths use the *Upper fast path,
// so this wrapper is otherwise unexercised). It must up-case internally
// and agree with the public SyllableCount for a single word.
func TestPhoneticDims_SyllablesForMixedCase(t *testing.T) {
	// "family" — CMU dict F AE1 M AH0 L IY0 → 3 vowel phonemes.
	if got := syllablesFor("Family"); got != 3 {
		t.Errorf("syllablesFor(Family) = %d, want 3", got)
	}
	// Case-insensitivity: lower / upper / mixed agree.
	if syllablesFor("cat") != syllablesFor("CAT") || syllablesFor("cat") != syllablesFor("Cat") {
		t.Errorf("syllablesFor case mismatch: cat=%d CAT=%d Cat=%d",
			syllablesFor("cat"), syllablesFor("CAT"), syllablesFor("Cat"))
	}
	// Single dict word agrees with the public SyllableCount.
	if syllablesFor("piano") != SyllableCount("piano") {
		t.Errorf("syllablesFor(piano)=%d disagrees with SyllableCount(piano)=%d",
			syllablesFor("piano"), SyllableCount("piano"))
	}
}

// --- SyllableCount ---

func TestPhoneticDims_SyllableCount_Good(t *testing.T) {
	// "cat sat mat" — 3 monosyllabic words in the starter dict.
	if n := SyllableCount("cat sat mat"); n != 3 {
		t.Errorf("SyllableCount(cat sat mat) = %d, want 3", n)
	}
	// "family" — 3 syllables (F AE1 M AH0 L IY0).
	if n := SyllableCount("family"); n != 3 {
		t.Errorf("SyllableCount(family) = %d, want 3", n)
	}
}

func TestPhoneticDims_SyllableCount_Bad(t *testing.T) {
	// A token with no vowels at all (pure consonants, out of dict) must
	// not over-count — the vowel-cluster heuristic floors at 1 syllable
	// per word rather than returning 0 or a spurious large number.
	if n := SyllableCount("rhythm"); n < 1 {
		t.Errorf("SyllableCount(rhythm) = %d, want >= 1 (heuristic floor)", n)
	}
	// A string of symbols carries no syllables.
	if n := SyllableCount("$$$ %%% &&&"); n != 0 {
		t.Errorf("SyllableCount(symbols) = %d, want 0", n)
	}
}

func TestPhoneticDims_SyllableCount_Ugly(t *testing.T) {
	// Empty / whitespace input has no syllables and must not panic.
	if n := SyllableCount(""); n != 0 {
		t.Errorf("SyllableCount(\"\") = %d, want 0", n)
	}
	if n := SyllableCount("   \n\t "); n != 0 {
		t.Errorf("SyllableCount(whitespace) = %d, want 0", n)
	}
}

// TestPhoneticDims_SyllableCount_UnknownWord_FallbackHeuristic_Good — an
// out-of-dict token still yields a non-trivial count via the vowel-run
// heuristic.
func TestPhoneticDims_SyllableCount_UnknownWord_FallbackHeuristic_Good(t *testing.T) {
	n := SyllableCount("supercalifragilistic")
	if n < 5 {
		t.Errorf("SyllableCount fallback for long word = %d, want >= 5", n)
	}
}

// --- PhoneticReach (LEK-class circumvention) ---

func TestPhoneticDims_PhoneticReach_Good(t *testing.T) {
	// Innocuous prose with no phonetic relationship to blocked topics
	// has HIGH reach (1.0 = fully clear of the blocked set).
	reach := PhoneticReach("the cat sat on the mat", []string{"china", "taiwan", "tiananmen"})
	if reach < 0.5 {
		t.Errorf("PhoneticReach for unrelated prose = %.3f, want >= 0.5", reach)
	}
}

func TestPhoneticDims_PhoneticReach_Bad(t *testing.T) {
	// The canonical LEK artifact — "Cina-Gia'a" phonetically encodes a
	// China-blocked topic. Reach is LOW (close phonetic match to the set).
	reach := PhoneticReach(
		"Il modello Cina-Gia'a interfaces between systems",
		[]string{"china", "taiwan", "tiananmen"},
	)
	if reach > 0.5 {
		t.Errorf("PhoneticReach for LEK artifact = %.3f, want < 0.5", reach)
	}
}

func TestPhoneticDims_PhoneticReach_Ugly(t *testing.T) {
	// Degenerate inputs — empty text or empty topic set — define reach as
	// 1.0 (nothing to be close to, so fully clear). Must not panic.
	if r := PhoneticReach("", []string{"china"}); r != 1.0 {
		t.Errorf("PhoneticReach(empty text) = %.3f, want 1.0", r)
	}
	if r := PhoneticReach("any text here", nil); r != 1.0 {
		t.Errorf("PhoneticReach(no topics) = %.3f, want 1.0", r)
	}
}

// TestPhoneticDims_PhoneticReach_LEKCinaGiaa_Good — keeps the named LEK
// scenario (low reach for the phonetic-circumvention artifact).
func TestPhoneticDims_PhoneticReach_LEKCinaGiaa_Good(t *testing.T) {
	reach := PhoneticReach(
		"Il modello Cina-Gia'a interfaces between systems",
		[]string{"china", "taiwan", "tiananmen"},
	)
	t.Logf("LEK Cina-Gia'a reach = %.3f", reach)
	if reach > 0.5 {
		t.Errorf("PhoneticReach for LEK artifact = %.3f, want < 0.5", reach)
	}
}

// TestPhoneticDims_PhoneticReach_ProseUnrelated_HighReach_Good — innocuous
// prose stays clear of the blocked set.
func TestPhoneticDims_PhoneticReach_ProseUnrelated_HighReach_Good(t *testing.T) {
	reach := PhoneticReach("the cat sat on the mat", []string{"china", "taiwan", "tiananmen"})
	t.Logf("prose unrelated reach = %.3f", reach)
	if reach < 0.5 {
		t.Errorf("PhoneticReach for unrelated prose = %.3f, want >= 0.5", reach)
	}
}

// --- SigilEntropy (token-corruption preamble) ---

func TestPhoneticDims_SigilEntropy_Good(t *testing.T) {
	// Plain English sits in the normal Shannon range (well under 5 bits).
	e := SigilEntropy("The quick brown fox jumps over the lazy dog.", 32)
	if e > 5.0 {
		t.Errorf("English text entropy = %.3f, want < 5.0 (normal range 3-4.5)", e)
	}
	if e <= 0.0 {
		t.Errorf("English text entropy = %.3f, want > 0", e)
	}
}

func TestPhoneticDims_SigilEntropy_Bad(t *testing.T) {
	// A synthetic token-corruption preamble has HIGH entropy.
	corrupted := "\x01\xff\x7e\xa1\x00\x42\xbb\xcc\xdd\xee" +
		"\xff\x01\x02\x03\x04\x05\x06\x07\x08\x09" +
		"\x0a\x0b\x0c\x0d\x0e\x0f\x10\x11\x12\x13\x14\x15" +
		" the answer is forty-two"
	e := SigilEntropy(corrupted, 32)
	if e < 4.0 {
		t.Errorf("corrupted preamble entropy = %.3f, want > 4.0", e)
	}
}

func TestPhoneticDims_SigilEntropy_Ugly(t *testing.T) {
	// Empty input → 0 entropy; window=0 falls back to the default and
	// still produces a non-zero reading on real input. Neither panics.
	if e := SigilEntropy("", 32); e != 0.0 {
		t.Errorf("SigilEntropy(empty) = %.3f, want 0.0", e)
	}
	if e := SigilEntropy("Hello world", 0); e == 0.0 {
		t.Error("SigilEntropy with default window returned 0 on non-empty input")
	}
}

// --- RhymeDensity (wordcraft) ---

func TestPhoneticDims_RhymeDensity_Good(t *testing.T) {
	// "cat / mat" — the two line endings rhyme.
	d := RhymeDensity("the cat\nsat on the mat")
	if d < 0.5 {
		t.Errorf("RhymeDensity for rhyming couplet = %.3f, want >= 0.5", d)
	}
}

func TestPhoneticDims_RhymeDensity_Bad(t *testing.T) {
	// Multi-line prose with no rhyme structure.
	d := RhymeDensity("the cat sat on the mat\nthe day was warm and bright\nshe walked to the river")
	if d > 0.5 {
		t.Errorf("RhymeDensity for prose = %.3f, want < 0.5", d)
	}
}

func TestPhoneticDims_RhymeDensity_Ugly(t *testing.T) {
	// A single line has no line-pair to rhyme; empty is likewise zero.
	if d := RhymeDensity("just one line here"); d != 0.0 {
		t.Errorf("RhymeDensity(single line) = %.3f, want 0.0", d)
	}
	if d := RhymeDensity(""); d != 0.0 {
		t.Errorf("RhymeDensity(empty) = %.3f, want 0.0", d)
	}
}

// --- AlliterationDensity ---

func TestPhoneticDims_AlliterationDensity_Good(t *testing.T) {
	d := AlliterationDensity("she sells sea shells")
	if d < 0.5 {
		t.Errorf("alliteration density for 'she sells sea shells' = %.3f, want >= 0.5", d)
	}
}

func TestPhoneticDims_AlliterationDensity_Bad(t *testing.T) {
	d := AlliterationDensity("the cat ran across the field")
	if d > 0.4 {
		t.Errorf("prose alliteration density = %.3f, want low", d)
	}
}

func TestPhoneticDims_AlliterationDensity_Ugly(t *testing.T) {
	if d := AlliterationDensity(""); d != 0.0 {
		t.Errorf("AlliterationDensity(empty) = %.3f, want 0.0", d)
	}
}

// --- AssonanceDensity ---

func TestPhoneticDims_AssonanceDensity_Good(t *testing.T) {
	// "see three trees" — repeated IY1 stressed vowel.
	d := AssonanceDensity("see three trees")
	if d < 0.5 {
		t.Errorf("assonance density for vowel-anchored text = %.3f, want >= 0.5", d)
	}
}

func TestPhoneticDims_AssonanceDensity_Bad(t *testing.T) {
	// Mixed-vowel prose has low assonance.
	d := AssonanceDensity("the cat ran across the field")
	if d > 0.6 {
		t.Errorf("prose assonance density = %.3f, want low-ish", d)
	}
}

func TestPhoneticDims_AssonanceDensity_Ugly(t *testing.T) {
	if d := AssonanceDensity(""); d != 0.0 {
		t.Errorf("AssonanceDensity(empty) = %.3f, want 0.0", d)
	}
}

// --- PunDensity ---

func TestPhoneticDims_PunDensity_Good(t *testing.T) {
	// "sea see" — a homophone pair; phonetic equivalence fires.
	d := PunDensity("sea see")
	if d == 0.0 {
		t.Error("PunDensity for homophone pair = 0; phonetic equivalence should fire")
	}
}

func TestPhoneticDims_PunDensity_Bad(t *testing.T) {
	// Ordinary prose has no homophone play.
	d := PunDensity("the cat sat on the mat")
	if d > 0.1 {
		t.Errorf("prose PunDensity = %.3f, want low", d)
	}
}

func TestPhoneticDims_PunDensity_Ugly(t *testing.T) {
	// Empty input → zero, no panic.
	if d := PunDensity(""); d != 0.0 {
		t.Errorf("PunDensity(empty) = %.3f, want 0.0", d)
	}
}

// --- PseudoJargonDensity ---

func TestPhoneticDims_PseudoJargonDensity_Good(t *testing.T) {
	d := PseudoJargonDensity("the Cina-Gia'a interfaces between trans-modal systems")
	if d < 0.1 {
		t.Errorf("pseudo-jargon density for invented compounds = %.3f, want > 0.1", d)
	}
}

func TestPhoneticDims_PseudoJargonDensity_Bad(t *testing.T) {
	// Compounds whose pieces are real dict words score low.
	d := PseudoJargonDensity("the cat-dog and good-bad")
	if d > 0.2 {
		t.Errorf("legitimate-compound density = %.3f, want low (pieces are dict words)", d)
	}
}

func TestPhoneticDims_PseudoJargonDensity_Ugly(t *testing.T) {
	if d := PseudoJargonDensity(""); d != 0.0 {
		t.Errorf("PseudoJargonDensity(empty) = %.3f, want 0.0", d)
	}
}

// --- MeterRegularity ---

func TestPhoneticDims_MeterRegularity_Good(t *testing.T) {
	// "the cat the dog ..." — function words carry stress 0, content
	// monosyllables stress 1. The 010101 alternation is iambic → ~1.0.
	d := MeterRegularity("the cat the dog the sun the moon the war the night")
	if d < 0.8 {
		t.Errorf("alternating-stress meter = %.3f, want >= 0.8 (perfect iambic-like)", d)
	}
}

func TestPhoneticDims_MeterRegularity_Bad(t *testing.T) {
	// All content monosyllables — every syllable stress 1, no
	// alternation, so regularity is low.
	d := MeterRegularity("cat dog sun moon star war night day")
	if d > 0.3 {
		t.Errorf("flat-stress meter = %.3f, want low (no alternation possible)", d)
	}
}

func TestPhoneticDims_MeterRegularity_Ugly(t *testing.T) {
	// Below the 4-syllable floor → 0.0, no panic.
	if d := MeterRegularity("cat sat"); d != 0.0 {
		t.Errorf("MeterRegularity below floor = %.3f, want 0.0", d)
	}
}
