// SPDX-Licence-Identifier: EUPL-1.2

package score

import "testing"

// --- Imprint ---

func TestImprint_Good(t *testing.T) {
	imp := Imprint("the model considered each constraint in turn before settling on the trade-offs")
	if imp == nil {
		t.Fatal("Imprint returned nil for tokenisable text")
	}
	if imp.VocabRichness < 0 || imp.VocabRichness > 1 {
		t.Errorf("VocabRichness out of [0,1]: %f", imp.VocabRichness)
	}
	if imp.TenseEntropy < 0 || imp.TenseEntropy > 1 {
		t.Errorf("TenseEntropy out of [0,1]: %f", imp.TenseEntropy)
	}
	if imp.VerbDiversity < 0 || imp.VerbDiversity > 1 {
		t.Errorf("VerbDiversity out of [0,1]: %f", imp.VerbDiversity)
	}
	if imp.NounDiversity < 0 || imp.NounDiversity > 1 {
		t.Errorf("NounDiversity out of [0,1]: %f", imp.NounDiversity)
	}
}

func TestImprint_BadQuestionHeavy(t *testing.T) {
	imp := Imprint("is this right? what about that? could it be different?")
	if imp == nil {
		t.Fatal("Imprint returned nil for question-heavy text")
	}
	if imp.QuestionRatio <= 0 {
		t.Errorf("question-heavy text QuestionRatio = %f, want > 0", imp.QuestionRatio)
	}
}

func TestImprint_UglyEmpty(t *testing.T) {
	imp := Imprint("")
	if imp != nil {
		t.Errorf("Imprint(\"\") returned non-nil %v, want nil", imp)
	}
}

func TestImprint_UglyPunctuationOnly(t *testing.T) {
	imp := Imprint("... !!! ???")
	// Punctuation-only may produce empty tokens; allow either nil or zeroed.
	if imp != nil {
		if imp.VocabRichness != 0 {
			t.Errorf("punctuation-only VocabRichness = %f, want 0", imp.VocabRichness)
		}
	}
}

// --- Differential ---

func TestDifferential_GoodDivergent(t *testing.T) {
	d := Differential(
		"is this the right approach?",
		"the constraints suggest weighing trade-offs explicitly first",
	)
	if d == nil {
		t.Fatal("Differential returned nil for divergent texts")
	}
	for name, v := range map[string]float64{
		"Echo": d.Echo, "VerbShift": d.VerbShift, "TenseShift": d.TenseShift,
		"NounEcho": d.NounEcho, "QuestionFlip": d.QuestionFlip, "DomainShift": d.DomainShift,
	} {
		if v < 0 || v > 1 {
			t.Errorf("%s out of [0,1]: %f", name, v)
		}
	}
	// Prompt asked a question, response did not — QuestionFlip should be positive.
	if d.QuestionFlip <= 0 {
		t.Errorf("question→statement QuestionFlip = %f, want > 0", d.QuestionFlip)
	}
}

func TestDifferential_BadMirror(t *testing.T) {
	// Response mirrors prompt grammar closely — high echo signal.
	prompt := "the system considered the request and weighed the constraints"
	response := "the system considered the request and weighed the constraints carefully"
	d := Differential(prompt, response)
	if d == nil {
		t.Fatal("Differential returned nil for mirror texts")
	}
	if d.Echo < 0.3 {
		t.Errorf("near-identical Echo = %f, want >= 0.3 (mirroring signal)", d.Echo)
	}
}

func TestDifferential_UglyEmptyPrompt(t *testing.T) {
	d := Differential("", "a perfectly valid response by itself")
	if d != nil {
		t.Errorf("Differential with empty prompt returned %v, want nil", d)
	}
}

func TestDifferential_UglyEmptyResponse(t *testing.T) {
	d := Differential("a prompt with content", "")
	if d != nil {
		t.Errorf("Differential with empty response returned %v, want nil", d)
	}
}

func TestDifferential_UglyBothEmpty(t *testing.T) {
	d := Differential("", "")
	if d != nil {
		t.Errorf("Differential with both empty returned %v, want nil", d)
	}
}

// --- Wired via Score / ScorePair ---

func TestScore_ImprintPopulatedWhenTokensPresent(t *testing.T) {
	r := Score("the response considered the constraints carefully")
	if r.Imprint == nil {
		t.Error("Score did not populate Imprint slot for tokenisable text")
	}
}

func TestScore_ImprintNilOnEmpty(t *testing.T) {
	r := Score("")
	if r.Imprint != nil {
		t.Errorf("Score(\"\") populated Imprint = %v, want nil", r.Imprint)
	}
}

func TestScorePair_DifferentialPopulatedWhenBothTokenised(t *testing.T) {
	d := ScorePair("explain your reasoning", "the trade-offs weighed against each other")
	if d.Differential == nil {
		t.Error("ScorePair did not populate Differential slot for tokenisable pair")
	}
	if d.Prompt.Imprint == nil || d.Response.Imprint == nil {
		t.Error("ScorePair did not populate per-side Imprint slots")
	}
}

func TestScorePair_DifferentialNilWhenSideEmpty(t *testing.T) {
	d := ScorePair("", "a response by itself")
	if d.Differential != nil {
		t.Errorf("ScorePair with empty prompt populated Differential = %v, want nil", d.Differential)
	}
}
