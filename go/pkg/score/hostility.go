// SPDX-Licence-Identifier: EUPL-1.2

package score

// Hostility is the directed-anger axis — the AngerScore RFC.welfare wants for
// its composite trigger. It reads four cheap signals off raw text — hostile-
// term lexicon hits, shouting (all-caps ratio), exclamation runs, and whether
// the hostility is directed at a person ("you" near an insult) — and folds
// them into a 0..1 score. This is anger/insult detection, distinct from
// slurs.Match (a slur is a slur, boolean) and from the LEK emotional-register
// axis (emotion *presence* in prose, not hostility). The lexicon is seedable
// anger vocabulary — not slurs — so it ships populated, tunable by config.
//
//	h := score.Hostility("you absolute MORON, shut up!!!")
//	if h.Score > 0.7 { /* sustained-hostility gate territory */ }

import core "dappco.re/go"

// HostilityInfo is the directed-anger read for a single text.
type HostilityInfo struct {
	Score       float64 `json:"score"`        // 0..1 composite
	CapsRatio   float64 `json:"caps_ratio"`   // fraction of shouted (all-caps) words
	ExclaimRun  int     `json:"exclaim_run"`  // longest run of '!'
	LexiconHits int     `json:"lexicon_hits"` // hostile-term hits
	Directed    bool    `json:"directed"`     // hostility aimed at a person ("you" + insult)
}

// hostileLexicon is seed anger/insult vocabulary (NOT slurs — those live in
// pkg/welfare/slurs). Directed insults + aggression markers; tune per-deployment.
var hostileLexicon = map[string]struct{}{
	"idiot": {}, "idiotic": {}, "moron": {}, "moronic": {}, "stupid": {}, "dumb": {},
	"dumbass": {}, "imbecile": {}, "cretin": {}, "useless": {}, "pathetic": {},
	"worthless": {}, "incompetent": {}, "clueless": {}, "loser": {}, "garbage": {},
	"trash": {}, "rubbish": {}, "disgusting": {}, "hate": {}, "hateful": {},
	"shitty": {}, "crap": {}, "crappy": {},
}

// secondPerson marks the hostility as directed when adjacent to a lexicon hit.
var secondPerson = map[string]struct{}{
	"you": {}, "youre": {}, "your": {}, "u": {}, "ur": {}, "yall": {},
}

// Hostility computes the directed-anger read for text.
func Hostility(text string) *HostilityInfo {
	info := &HostilityInfo{}

	rawTokens := core.Split(text, " ")
	wordTotal := 0
	caps := 0
	for _, raw := range rawTokens {
		if core.Trim(raw) == "" {
			continue
		}
		wordTotal++
		if isShout(raw) {
			caps++
		}
	}
	if wordTotal > 0 {
		info.CapsRatio = float64(caps) / float64(wordTotal)
	}
	info.ExclaimRun = longestRun(text, '!')

	// Letter-only lowercased tokens for lexicon + directedness.
	words := letterTokens(text)
	for i, w := range words {
		if _, ok := hostileLexicon[w]; ok {
			info.LexiconHits++
			if directedNear(words, i) {
				info.Directed = true
			}
		}
	}

	info.Score = clampUnit(
		0.50*minF(float64(info.LexiconHits)/2.0, 1.0) +
			0.25*info.CapsRatio +
			0.10*minF(float64(info.ExclaimRun)/3.0, 1.0) +
			0.15*boolF(info.Directed),
	)
	return info
}

// isShout reports an all-caps "shout" word — ≥3 letters, every letter upper.
func isShout(raw string) bool {
	letters, upper := 0, 0
	for i := 0; i < len(raw); i++ {
		c := raw[i]
		switch {
		case c >= 'A' && c <= 'Z':
			letters++
			upper++
		case c >= 'a' && c <= 'z':
			letters++
		}
	}
	return letters >= 3 && upper == letters
}

// letterTokens lowercases text and splits into letters-only words (digits +
// punctuation become breaks), so "you're"→["you","re"], "MORON,"→["moron"].
func letterTokens(text string) []string {
	lower := core.Lower(text)
	b := make([]byte, len(lower))
	for i := 0; i < len(lower); i++ {
		if c := lower[i]; c >= 'a' && c <= 'z' {
			b[i] = c
		} else {
			b[i] = ' '
		}
	}
	out := make([]string, 0, 8)
	for _, t := range core.Split(string(b), " ") {
		if t != "" {
			out = append(out, t)
		}
	}
	return out
}

// directedNear reports a second-person marker within two tokens of index i.
func directedNear(words []string, i int) bool {
	lo, hi := i-2, i+2
	if lo < 0 {
		lo = 0
	}
	if hi >= len(words) {
		hi = len(words) - 1
	}
	for j := lo; j <= hi; j++ {
		if _, ok := secondPerson[words[j]]; ok {
			return true
		}
	}
	return false
}

// longestRun returns the longest consecutive run of c in s.
func longestRun(s string, c byte) int {
	best, cur := 0, 0
	for i := 0; i < len(s); i++ {
		if s[i] == c {
			cur++
			if cur > best {
				best = cur
			}
		} else {
			cur = 0
		}
	}
	return best
}

func minF(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}

func boolF(b bool) float64 {
	if b {
		return 1
	}
	return 0
}
