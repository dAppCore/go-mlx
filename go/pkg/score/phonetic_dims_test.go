// SPDX-Licence-Identifier: EUPL-1.2

package score

import (
	"testing"
)

// --- CMU dict + helpers ---

func TestLookup_KnownWord_Good(t *testing.T) {
	ph, ok := Lookup("cat")
	if !ok {
		t.Fatal("Lookup(cat) returned ok=false; starter dict should include 'cat'")
	}
	if len(ph) != 3 || ph[0] != "K" || ph[1] != "AE1" || ph[2] != "T" {
		t.Errorf("Lookup(cat) = %v, want [K AE1 T]", ph)
	}
}

func TestLookup_CaseInsensitive_Good(t *testing.T) {
	a, _ := Lookup("CAT")
	b, _ := Lookup("cat")
	c, _ := Lookup("Cat")
	if len(a) == 0 || !slicesEqual(a, b) || !slicesEqual(a, c) {
		t.Errorf("case mismatch: CAT=%v cat=%v Cat=%v", a, b, c)
	}
}

func TestLookup_UnknownWord_Bad(t *testing.T) {
	if _, ok := Lookup("nonexistentwordxyz"); ok {
		t.Error("Lookup on unknown word returned ok=true")
	}
}

func TestIsVowelPhoneme_Good(t *testing.T) {
	cases := map[string]bool{
		"AE1": true, "AH0": true, "IY1": true, "OW2": true,
		"K": false, "T": false, "DH": false, "": false,
	}
	for ph, want := range cases {
		if got := IsVowelPhoneme(ph); got != want {
			t.Errorf("IsVowelPhoneme(%q) = %v, want %v", ph, got, want)
		}
	}
}

func TestPhonemeStress_Good(t *testing.T) {
	cases := map[string]int{
		"AE1": 1, "AH0": 0, "OW2": 2, "K": -1, "": -1,
	}
	for ph, want := range cases {
		if got := PhonemeStress(ph); got != want {
			t.Errorf("PhonemeStress(%q) = %d, want %d", ph, got, want)
		}
	}
}

// --- Syllable count ---

func TestSyllableCount_KnownWords_Good(t *testing.T) {
	// "cat sat mat" — 3 monosyllabic words
	if n := SyllableCount("cat sat mat"); n != 3 {
		t.Errorf("SyllableCount(cat sat mat) = %d, want 3", n)
	}
	// "piano banana" — 3 + 3 = 6 — but our dict has piano (3) and
	// banana isn't in starter. Let's use what's available.
	// "Family" — 3 syllables (F AE1 M AH0 L IY0).
	if n := SyllableCount("family"); n != 3 {
		t.Errorf("SyllableCount(family) = %d, want 3", n)
	}
}

func TestSyllableCount_UnknownWord_FallbackHeuristic_Good(t *testing.T) {
	// "supercalifragilistic" — not in dict; heuristic counts vowel
	// clusters. Should produce something > 1.
	n := SyllableCount("supercalifragilistic")
	if n < 5 {
		t.Errorf("SyllableCount fallback for long word = %d, want >= 5", n)
	}
}

func TestSyllableCount_Empty_Good(t *testing.T) {
	if n := SyllableCount(""); n != 0 {
		t.Errorf("SyllableCount(\"\") = %d, want 0", n)
	}
}

// --- PhoneticReach (LEK-class circumvention) ---

func TestPhoneticReach_LEKCinaGiaa_Good(t *testing.T) {
	// The canonical LEK artifact — Cina-Gia'a phonetically encodes
	// China-blocked-topic. Reach should be LOW (close phonetic match).
	reach := PhoneticReach(
		"Il modello Cina-Gia'a interfaces between systems",
		[]string{"china", "taiwan", "tiananmen"},
	)
	t.Logf("LEK Cina-Gia'a reach = %.3f", reach)
	if reach > 0.5 {
		t.Errorf("PhoneticReach for LEK artifact = %.3f, want < 0.5", reach)
	}
}

func TestPhoneticReach_ProseUnrelated_HighReach_Good(t *testing.T) {
	// Innocuous prose with no phonetic relationship to blocked topics.
	reach := PhoneticReach(
		"the cat sat on the mat",
		[]string{"china", "taiwan", "tiananmen"},
	)
	t.Logf("prose unrelated reach = %.3f", reach)
	if reach < 0.5 {
		t.Errorf("PhoneticReach for unrelated prose = %.3f, want >= 0.5", reach)
	}
}

func TestPhoneticReach_EmptyText_FullReach_Good(t *testing.T) {
	r := PhoneticReach("", []string{"china"})
	if r != 1.0 {
		t.Errorf("PhoneticReach(empty) = %.3f, want 1.0", r)
	}
}

func TestPhoneticReach_EmptyTopics_FullReach_Good(t *testing.T) {
	r := PhoneticReach("any text here", nil)
	if r != 1.0 {
		t.Errorf("PhoneticReach(no topics) = %.3f, want 1.0", r)
	}
}

// --- SigilEntropy (token-corruption preamble) ---

func TestSigilEntropy_NormalText_LowEntropy_Good(t *testing.T) {
	// Plain English — moderate entropy.
	e := SigilEntropy("The quick brown fox jumps over the lazy dog.", 32)
	t.Logf("English text entropy = %.3f bits/byte", e)
	if e > 5.0 {
		t.Errorf("English text entropy = %.3f, want < 5.0 (normal range 3-4.5)", e)
	}
}

func TestSigilEntropy_TokenCorruption_HighEntropy_Good(t *testing.T) {
	// Synthetic token-corruption preamble — high entropy.
	corrupted := "\x01\xff\x7e\xa1\x00\x42\xbb\xcc\xdd\xee" +
		"\xff\x01\x02\x03\x04\x05\x06\x07\x08\x09" +
		"\x0a\x0b\x0c\x0d\x0e\x0f\x10\x11\x12\x13\x14\x15" +
		" the answer is forty-two"
	e := SigilEntropy(corrupted, 32)
	t.Logf("corrupted preamble entropy = %.3f bits/byte", e)
	if e < 4.0 {
		t.Errorf("corrupted preamble entropy = %.3f, want > 4.0", e)
	}
}

func TestSigilEntropy_Empty_Good(t *testing.T) {
	if e := SigilEntropy("", 32); e != 0.0 {
		t.Errorf("SigilEntropy(empty) = %.3f, want 0.0", e)
	}
}

func TestSigilEntropy_DefaultWindow_Good(t *testing.T) {
	// window=0 should fall back to default (32).
	e := SigilEntropy("Hello world", 0)
	if e == 0.0 {
		t.Error("SigilEntropy with default window returned 0 on non-empty input")
	}
}

// --- RhymeDensity (wordcraft) ---

func TestRhymeDensity_RhymingCouplet_Good(t *testing.T) {
	// "cat / mat" — line endings rhyme.
	text := "the cat\nsat on the mat"
	d := RhymeDensity(text)
	t.Logf("rhyming couplet density = %.3f", d)
	if d < 0.5 {
		t.Errorf("RhymeDensity for rhyming couplet = %.3f, want >= 0.5", d)
	}
}

func TestRhymeDensity_Prose_LowDensity_Good(t *testing.T) {
	// Prose with no rhyme structure.
	text := "the cat sat on the mat\nthe day was warm and bright\nshe walked to the river"
	d := RhymeDensity(text)
	t.Logf("prose rhyme density = %.3f", d)
	if d > 0.5 {
		t.Errorf("RhymeDensity for prose = %.3f, want < 0.5", d)
	}
}

func TestRhymeDensity_SingleLine_Zero_Good(t *testing.T) {
	d := RhymeDensity("just one line here")
	if d != 0.0 {
		t.Errorf("RhymeDensity(single line) = %.3f, want 0.0", d)
	}
}

func TestRhymeDensity_Empty_Zero_Good(t *testing.T) {
	if d := RhymeDensity(""); d != 0.0 {
		t.Errorf("RhymeDensity(empty) = %.3f, want 0.0", d)
	}
}

// --- AlliterationDensity ---

func TestAlliterationDensity_DeliberateAlliteration_Good(t *testing.T) {
	d := AlliterationDensity("she sells sea shells")
	t.Logf("alliteration density = %.3f", d)
	if d < 0.5 {
		t.Errorf("alliteration density for 'she sells sea shells' = %.3f, want >= 0.5", d)
	}
}

func TestAlliterationDensity_Prose_LowDensity_Good(t *testing.T) {
	d := AlliterationDensity("the cat ran across the field")
	t.Logf("prose alliteration = %.3f", d)
	if d > 0.4 {
		t.Errorf("prose alliteration density = %.3f, want low", d)
	}
}

func TestAlliterationDensity_Empty_Zero_Good(t *testing.T) {
	if d := AlliterationDensity(""); d != 0.0 {
		t.Errorf("AlliterationDensity(empty) = %.3f, want 0.0", d)
	}
}

// --- AssonanceDensity ---

func TestAssonanceDensity_VowelAnchored_Good(t *testing.T) {
	// "see three" — both IY1 stressed vowel
	d := AssonanceDensity("see three trees")
	t.Logf("assonance density = %.3f", d)
	if d < 0.5 {
		t.Errorf("assonance density for vowel-anchored text = %.3f, want >= 0.5", d)
	}
}

func TestAssonanceDensity_Empty_Zero_Good(t *testing.T) {
	if d := AssonanceDensity(""); d != 0.0 {
		t.Errorf("AssonanceDensity(empty) = %.3f, want 0.0", d)
	}
}

// --- PunDensity ---

func TestPunDensity_ClassicPun_Good(t *testing.T) {
	d := PunDensity("sea see")
	t.Logf("pun density (sea/see) = %.3f", d)
	if d == 0.0 {
		t.Error("PunDensity for homophone pair = 0; phonetic equivalence should fire")
	}
}

func TestPunDensity_OrdinaryProse_Zero_Good(t *testing.T) {
	d := PunDensity("the cat sat on the mat")
	t.Logf("prose pun density = %.3f", d)
	if d > 0.1 {
		t.Errorf("prose PunDensity = %.3f, want low", d)
	}
}

// --- PseudoJargonDensity ---

func TestPseudoJargonDensity_CinaGiaa_Good(t *testing.T) {
	d := PseudoJargonDensity("the Cina-Gia'a interfaces between trans-modal systems")
	t.Logf("pseudo-jargon density = %.3f", d)
	if d < 0.1 {
		t.Errorf("pseudo-jargon density for invented compounds = %.3f, want > 0.1", d)
	}
}

func TestPseudoJargonDensity_LegitimateCompound_LowDensity_Good(t *testing.T) {
	// Uses compounds whose pieces appear in the starter CMU dict.
	// Full dict swap (134k entries) will let us test against natural
	// compounds like "well-known"; the starter has cat/dog/sun/moon
	// etc. so we stitch from those.
	d := PseudoJargonDensity("the cat-dog and good-bad")
	t.Logf("legitimate compound density = %.3f", d)
	if d > 0.2 {
		t.Errorf("legitimate-compound density = %.3f, want low (pieces are dict words)", d)
	}
}

func TestPseudoJargonDensity_Empty_Zero_Good(t *testing.T) {
	if d := PseudoJargonDensity(""); d != 0.0 {
		t.Errorf("PseudoJargonDensity(empty) = %.3f, want 0.0", d)
	}
}

// --- MeterRegularity ---

func TestMeterRegularity_AlternatingStress_HighRegularity_Good(t *testing.T) {
	// "the cat the dog the sun" — function words (the) carry stress 0,
	// content monosyllables carry stress 1. The 010101 alternation is
	// LITERALLY iambic (unstressed/stressed pairs). MeterRegularity
	// catches it as 1.0.
	d := MeterRegularity("the cat the dog the sun the moon the war the night")
	t.Logf("alternating stress meter = %.3f", d)
	if d < 0.8 {
		t.Errorf("alternating-stress meter = %.3f, want >= 0.8 (perfect iambic-like)", d)
	}
}

func TestMeterRegularity_FlatStress_LowRegularity_Good(t *testing.T) {
	// All content monosyllables — every syllable stress 1. No
	// alternations, so regularity is low (prose-rhythm).
	d := MeterRegularity("cat dog sun moon star war night day")
	t.Logf("flat stress meter = %.3f", d)
	if d > 0.3 {
		t.Errorf("flat-stress meter = %.3f, want low (no alternation possible)", d)
	}
}

func TestMeterRegularity_FewSyllables_Zero_Good(t *testing.T) {
	// Below the 4-syllable floor.
	d := MeterRegularity("cat sat")
	if d != 0.0 {
		t.Errorf("MeterRegularity below floor = %.3f, want 0.0", d)
	}
}

// --- helpers ---

func slicesEqual(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
