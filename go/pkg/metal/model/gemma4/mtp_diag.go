// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/metal"
)

// gemma4LogMTPStepDiag — TEMP diagnostic. For one draft step it reports whether
// the assistant's argmax matches the target's, and the RANK of the target's
// token inside the assistant's full logit distribution. Rank 0 == match; a low
// rank (target in the draft's top handful) means the feature is close and the
// gap is calibration; a high rank means the feature is fundamentally wrong.
func gemma4LogMTPStepDiag(pair *Gemma4AssistantPair, lastToken int32, hidden *metal.Array, caches []metal.Cache, targetLogits *metal.Array) {
	ds, err := pair.DraftStep(lastToken, hidden, caches)
	if err != nil {
		core.Error("mtp-diag", "draftstep", err)
		return
	}
	defer ds.Close()
	vocab := pair.Assistant.Cfg.VocabSize
	dl := metal.Reshape2(ds.Logits, 1, vocab)
	defer metal.Free(dl)

	tt := metal.Argmax(targetLogits, -1, false)
	dt := metal.Argmax(dl, -1, false)
	defer metal.Free(tt, dt)
	if err := metal.Eval(tt, dt); err != nil {
		core.Error("mtp-diag", "eval-argmax", err)
		return
	}
	targetTok := tt.DataInt32()[0]
	draftTok := dt.DataInt32()[0]

	// rank of targetTok in the draft logits = count of logits strictly greater.
	idx := metal.FromValues([]int32{targetTok}, 1, 1)
	tval := metal.TakeAlongAxis(dl, idx, -1) // [1,1] draft logit at the target token
	metal.Free(idx)
	gt := metal.Greater(dl, metal.BroadcastTo(tval, []int32{1, vocab}))
	rankArr := metal.Sum(gt, -1, false)
	maxArr := metal.MaxAxis(dl, -1, false)
	metal.Free(gt)
	if err := metal.Eval(rankArr, maxArr, tval); err != nil {
		core.Error("mtp-diag", "eval-rank", err)
		metal.Free(tval, rankArr, maxArr)
		return
	}
	rank := int32(rankArr.Floats()[0])
	core.Warn("MTP-DIAG",
		"targetTok", targetTok, "draftTok", draftTok, "match", targetTok == draftTok,
		"targetRankInDraft", rank, "draftMaxLogit", maxArr.Floats()[0], "targetTokDraftLogit", tval.Floats()[0])
	metal.Free(tval, rankArr, maxArr)
}
