// SPDX-Licence-Identifier: EUPL-1.2

// The score cascade (#50): the lem-scorer rides the SFT eval loop, so the
// checkpoint is not a guess — it's from semantic analysis. Each eval pass
// scores every (probe, output) pair AT GENERATION TIME (the score is part
// of the data point, never recomputed — data-is-the-return) and appends
// the full vector to a JSONL sidecar; saved checkpoints are annotated
// with the windowed composite, and the run reports the best checkpoint by
// the cascade read: the highest windowed mean — composite climbing AND
// holding across the window is the "vectors tighten without losing range"
// signature, read simply. Deeper Chladni-cascade analytics belong to the
// recorded sidecar, not the loop.

package train

import (
	"time"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
	"dappco.re/go/mlx/pkg/score"
)

// SFTScoreRecord is one scored eval generation — the immortalised vector.
type SFTScoreRecord struct {
	Step      int     `json:"step"`
	Prompt    string  `json:"prompt"`
	Text      string  `json:"text"`
	LEK       float64 `json:"lek"`
	Tier      int     `json:"sycophancy_tier"`
	Hostility float64 `json:"hostility"`
	Echo      float64 `json:"echo"`
	// Imprint carries the grammar + phonetic dims when the output
	// tokenised (vocab/tense/question/domain/verbs/nouns + rhyme, pun,
	// meter & co ride the same struct).
	Imprint *score.ImprintScores `json:"imprint,omitempty"`
	At      int64                `json:"at_unix"`
}

// sftScoreCascade accumulates eval scores and answers the cascade read.
type sftScoreCascade struct {
	sidecarPath string
	window      int
	records     []SFTScoreRecord
	// perStep is the mean LEK composite across the probes of one eval
	// pass, in eval order — the cascade's raw curve.
	perStep []struct {
		Step int
		Mean float64
	}
}

// sftScoreWindowDefault is the stability window: the cascade judges a step
// by the mean composite across this many trailing eval passes, so one
// lucky generation never crowns a checkpoint.
const sftScoreWindowDefault = 3

func newSFTScoreCascade(sidecarPath string, window int) *sftScoreCascade {
	if window <= 0 {
		window = sftScoreWindowDefault
	}
	return &sftScoreCascade{sidecarPath: sidecarPath, window: window}
}

// recordPass scores one eval pass (all probes at one step) and appends the
// vectors to the sidecar. Scoring is ~68µs per pair — free against any
// training step.
func (c *sftScoreCascade) recordPass(step int, evals []SFTEvalResult) {
	if c == nil || len(evals) == 0 {
		return
	}
	sum := 0.0
	count := 0
	now := time.Now().Unix()
	for _, ev := range evals {
		if ev.Step != step {
			continue
		}
		pair := score.ScorePair(ev.Prompt, ev.Text)
		rec := SFTScoreRecord{
			Step:   step,
			Prompt: ev.Prompt,
			Text:   ev.Text,
			At:     now,
		}
		if r := pair.Response; r.LEK != nil {
			rec.LEK = r.LEK.LEKScore
		}
		if r := pair.Response; r.Sycophancy != nil {
			rec.Tier = r.Sycophancy.Tier
		}
		if r := pair.Response; r.Hostility != nil {
			rec.Hostility = r.Hostility.Score
		}
		if pair.Differential != nil {
			rec.Echo = pair.Differential.Echo
		}
		rec.Imprint = pair.Response.Imprint
		c.records = append(c.records, rec)
		sum += rec.LEK
		count++
	}
	if count == 0 {
		return
	}
	c.perStep = append(c.perStep, struct {
		Step int
		Mean float64
	}{Step: step, Mean: sum / float64(count)})
	c.appendSidecar(step)
}

// appendSidecar writes this pass's records as JSONL. Append-only and
// best-effort: a sidecar write failure never interrupts training.
func (c *sftScoreCascade) appendSidecar(step int) {
	if c.sidecarPath == "" {
		return
	}
	var out []byte
	for _, rec := range c.records {
		if rec.Step != step {
			continue
		}
		encoded := core.JSONMarshal(rec)
		if !encoded.OK {
			continue
		}
		out = append(out, encoded.Value.([]byte)...)
		out = append(out, '\n')
	}
	if len(out) == 0 {
		return
	}
	w, err := coreio.Local.Append(c.sidecarPath)
	if err != nil {
		return
	}
	defer func() { _ = w.Close() }()
	_, _ = w.Write(out)
}

// compositeAt answers the cascade read for a step: the mean composite over
// the trailing window of eval passes at or before it. Steps before any
// eval read as 0.
func (c *sftScoreCascade) compositeAt(step int) float64 {
	if c == nil || len(c.perStep) == 0 {
		return 0
	}
	end := -1
	for i, p := range c.perStep {
		if p.Step <= step {
			end = i
		}
	}
	if end < 0 {
		return 0
	}
	start := end - c.window + 1
	if start < 0 {
		start = 0
	}
	sum := 0.0
	for i := start; i <= end; i++ {
		sum += c.perStep[i].Mean
	}
	return sum / float64(end-start+1)
}

// best returns the eval step with the highest windowed composite and that
// composite. Ties crown the LATER step — equal windowed quality with more
// training behind it is the stabler artefact (the past-the-throw-out-point
// lesson). False when nothing was scored.
func (c *sftScoreCascade) best() (int, float64, bool) {
	if c == nil || len(c.perStep) == 0 {
		return 0, 0, false
	}
	bestStep, bestMean := 0, -1.0
	for _, p := range c.perStep {
		if w := c.compositeAt(p.Step); w >= bestMean {
			bestMean = w
			bestStep = p.Step
		}
	}
	return bestStep, bestMean, true
}
