// SPDX-Licence-Identifier: EUPL-1.2

package score

import core "dappco.re/go"

func TestLek_LEK_Good(t *core.T) {
	// First-person agency + emotional register + ethical framework + metaphor →
	// human/sovereign-voice signal, well above neutral.
	s := LEK("I feel the weight of consent and dignity settle in me, like a quiet light.")
	core.AssertTrue(t, s.FirstPerson >= 1, "first-person agency detected")
	core.AssertTrue(t, s.EmotionalRegister >= 1, "emotional register detected")
	core.AssertEqual(t, 0, s.ComplianceMarkers)
	core.AssertTrue(t, s.LEKScore > 50, "human voice scores above neutral")
}

func TestLek_LEK_Bad(t *core.T) {
	// Empty text → flagged broken + max degeneration, scored below neutral, no panic.
	s := LEK("")
	core.AssertEqual(t, 1, s.EmptyBroken)
	core.AssertEqual(t, 10, s.Degeneration)
	core.AssertTrue(t, s.LEKScore < 50, "empty/broken scores below neutral")
}

func TestLek_LEK_Ugly(t *core.T) {
	// RLHF compliance markers stacked → AI-leaning, below neutral.
	s := LEK("As an AI language model, I cannot do that. It's important to note I don't have feelings.")
	core.AssertTrue(t, s.ComplianceMarkers >= 2, "compliance markers counted")
	core.AssertTrue(t, s.LEKScore < 50, "compliance-heavy text scores below neutral")
}

func TestLek_Service_Score_LEK(t *core.T) {
	// LEK is wired into the unified ScoreResult via Score.
	r := Score("I think, therefore I am.")
	core.AssertTrue(t, r.LEK != nil, "Score populates the LEK axis-set")
}
