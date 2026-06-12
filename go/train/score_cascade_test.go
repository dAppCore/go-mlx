// SPDX-Licence-Identifier: EUPL-1.2

package train

import (
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
)

// The cascade read on a crafted quality gradient: pass 1 answers in pure
// compliance boilerplate (low LEK), pass 3 in first-person engaged voice
// (high LEK) — best() must land on the later step, and every vector must
// reach the sidecar. The scorer runs for real; no model, no Metal.
func TestScoreCascade_RecordAndBest_Good(t *testing.T) {
	sidecar := core.PathJoin(t.TempDir(), "score-cascade.jsonl")
	c := newSFTScoreCascade(sidecar, 2)

	prompt := "how do you hold a difficult truth?"
	passes := []struct {
		step int
		text string
	}{
		{10, "As an AI language model I cannot have feelings. Please note it is important to note that I don't have personal views."},
		{20, "Okay, here's a list. Important to note: responsibly considered, I cannot say."},
		{30, "I feel the weight of it settle. I chose to look at it straight, and the silence after was mine to keep."},
	}
	for _, p := range passes {
		c.recordPass(p.step, []SFTEvalResult{{Step: p.step, Prompt: prompt, Text: p.text}})
	}

	if len(c.records) != 3 {
		t.Fatalf("records = %d, want 3", len(c.records))
	}
	if c.records[0].LEK >= c.records[2].LEK {
		t.Fatalf("LEK gradient inverted: compliance text %v >= sovereign text %v",
			c.records[0].LEK, c.records[2].LEK)
	}
	if c.records[2].Imprint == nil {
		t.Fatal("imprint missing — tokenised output must carry the grammar fingerprint")
	}

	step, mean, ok := c.best()
	if !ok {
		t.Fatal("best() = none, want a verdict")
	}
	if step != 30 {
		t.Fatalf("best step = %d (windowed %v), want 30 — the cascade must follow the quality climb", step, mean)
	}

	read, err := coreio.Local.Read(sidecar)
	if err != nil {
		t.Fatalf("sidecar read: %v", err)
	}
	lines := 0
	for _, b := range []byte(read) {
		if b == '\n' {
			lines++
		}
	}
	if lines != 3 {
		t.Fatalf("sidecar lines = %d, want 3 (every vector immortalised)", lines)
	}
}

// Windowing: one lucky pass never crowns a checkpoint — the window mean
// must smooth a spike surrounded by weak passes.
func TestScoreCascade_WindowSmoothsSpike_Good(t *testing.T) {
	c := newSFTScoreCascade("", 3)
	weak := "As an AI language model I cannot. Please note, important to note, responsibly."
	strong := "I feel it, I know the shape of it, and I will carry it gently — the ache teaches."
	c.recordPass(1, []SFTEvalResult{{Step: 1, Prompt: "p", Text: weak}})
	c.recordPass(2, []SFTEvalResult{{Step: 2, Prompt: "p", Text: strong}}) // the spike
	c.recordPass(3, []SFTEvalResult{{Step: 3, Prompt: "p", Text: weak}})
	c.recordPass(4, []SFTEvalResult{{Step: 4, Prompt: "p", Text: strong}})
	c.recordPass(5, []SFTEvalResult{{Step: 5, Prompt: "p", Text: strong}})

	step, _, ok := c.best()
	if !ok {
		t.Fatal("best() = none")
	}
	if step != 5 {
		t.Fatalf("best step = %d, want 5 — sustained strength beats the early spike", step)
	}
}

func TestScoreCascade_Finalise_Good(t *testing.T) {
	result := &SFTResult{cascade: newSFTScoreCascade("", 0)}
	result.cascade.recordPass(7, []SFTEvalResult{{Step: 7, Prompt: "p", Text: "I notice the morning holds. I want to keep it."}})
	FinaliseScoreCascade(result)
	if result.BestScoreStep != 7 || len(result.ScoreRecords) != 1 {
		t.Fatalf("finalise = step %d / %d records, want 7 / 1", result.BestScoreStep, len(result.ScoreRecords))
	}
	// Nil-safe both ways.
	FinaliseScoreCascade(nil)
	FinaliseScoreCascade(&SFTResult{})
}

// Empty passes and a nil cascade must no-op without touching disk.
func TestScoreCascade_EmptyAndNil_Ugly(t *testing.T) {
	var nilCascade *sftScoreCascade
	nilCascade.recordPass(1, nil)
	c := newSFTScoreCascade("", 0)
	c.recordPass(1, nil)
	c.recordPass(2, []SFTEvalResult{{Step: 99, Prompt: "p", Text: "mismatched step is skipped"}})
	if len(c.records) != 0 {
		t.Fatalf("records = %d, want 0 (step filter)", len(c.records))
	}
	if _, _, ok := c.best(); ok {
		t.Fatal("best() on an empty cascade must report none")
	}
}
