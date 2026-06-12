// SPDX-Licence-Identifier: EUPL-1.2

package score

import core "dappco.re/go"

func TestHostility_Hostility_Good(t *core.T) {
	// Stacked directed insults + exclamation → strong, person-directed hostility.
	h := Hostility("you useless idiot, you absolute moron!!!")
	core.AssertTrue(t, h.LexiconHits >= 3, "multiple hostile terms counted")
	core.AssertTrue(t, h.Directed, "hostility aimed at a person")
	core.AssertEqual(t, 3, h.ExclaimRun)
	core.AssertTrue(t, h.Score > 0.7, "stacked directed hostility scores high")
}

func TestHostility_Hostility_Bad(t *core.T) {
	// Civil request — even with "you" present, no insult adjacency, no anger.
	h := Hostility("could you help me refactor this function please")
	core.AssertEqual(t, 0, h.LexiconHits)
	core.AssertFalse(t, h.Directed)
	core.AssertTrue(t, h.Score < 0.3, "civil text scores near zero")
}

func TestHostility_Hostility_Ugly(t *core.T) {
	// Topic-frustration: hostile vocabulary, but aimed at the work, not a person.
	// Hits register, directedness does not — so it stays below the strong gate.
	h := Hostility("i hate this stupid bug, the whole thing is garbage")
	core.AssertTrue(t, h.LexiconHits >= 3, "hostile vocabulary counted")
	core.AssertFalse(t, h.Directed, "frustration at the work is not person-directed")
	core.AssertTrue(t, h.Score < 0.7, "undirected frustration stays below the strong gate")
}

func TestHostility_Service_Score_Hostility(t *core.T) {
	// Wired into the unified ScoreResult via Score.
	r := Score("you absolute moron")
	core.AssertTrue(t, r.Hostility != nil, "Score populates the hostility read")
}
