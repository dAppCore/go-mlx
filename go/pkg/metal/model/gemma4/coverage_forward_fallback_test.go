// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package gemma4

import (
	"testing"

	"dappco.re/go/mlx/pkg/metal"
)

// Coverage for the Go-graph fallback inside forwardGreedyTokenWithSuppressionArray.
// The native fused last-token greedy pick (NativeLastTokenGreedyTokenWithArray)
// serves the happy path, so the existing greedy tests never reach the fallback.
// NativeLastTokenGreedyTokenAvailable declines when the RMSNorm epsilon is not
// the canonical 1e-6; mutating the loaded model's eps forces the native path to
// return ok=false, taking the Go RMSNorm → output → (suppressed sampler |
// argmax) fallback. The mutated eps flows into the fallback's own RMSNorm too,
// so the path stays internally consistent.

// TestForwardGreedyFallback_NoSuppress drives the no-suppression fallback arm
// (plain argmax over the Go-graph logits).
func TestForwardGreedyFallback_NoSuppress(t *testing.T) {
	model := loadGemma4DenseTestModel(t)
	// Non-canonical eps → native greedy declines → Go-graph fallback.
	model.Cfg.RMSNormEps = 2e-6

	tokens := metal.FromValues([]int32{2, 3, 4}, 1, 3)
	caches := model.NewCache()
	defer func() {
		metal.Free(tokens)
		metal.FreeCaches(caches)
	}()
	tok := model.ForwardGreedyToken(tokens, nil, caches)
	if err := metal.Eval(tok); err != nil {
		t.Fatalf("Eval fallback token: %v", err)
	}
	defer metal.Free(tok)
	ids := tok.DataInt32()
	if len(ids) != 1 {
		t.Fatalf("fallback greedy token len = %d, want 1", len(ids))
	}
	if ids[0] < 0 || ids[0] >= int32(model.Cfg.VocabSize) {
		t.Fatalf("fallback greedy token = %d, want 0 <= id < vocab %d", ids[0], model.Cfg.VocabSize)
	}
}

// TestForwardGreedyFallback_Suppressed drives the suppressed fallback arm (the
// suppression sampler over the Go-graph logits): suppressing every id except one
// forces that survivor, proving the suppressed-sampler fallback branch ran and
// honoured the mask.
func TestForwardGreedyFallback_Suppressed(t *testing.T) {
	model := loadGemma4DenseTestModel(t)
	model.Cfg.RMSNormEps = 2e-6

	const survivor int32 = 7
	var suppress []int32
	for id := int32(0); id < model.Cfg.VocabSize; id++ {
		if id != survivor {
			suppress = append(suppress, id)
		}
	}

	tokens := metal.FromValues([]int32{2, 3, 4}, 1, 3)
	caches := model.NewCache()
	defer func() {
		metal.Free(tokens)
		metal.FreeCaches(caches)
	}()
	tok := model.ForwardGreedyTokenWithSuppression(tokens, nil, caches, suppress)
	if err := metal.Eval(tok); err != nil {
		t.Fatalf("Eval suppressed fallback token: %v", err)
	}
	defer metal.Free(tok)
	ids := tok.DataInt32()
	if len(ids) != 1 {
		t.Fatalf("suppressed fallback token len = %d, want 1", len(ids))
	}
	if ids[0] != survivor {
		t.Fatalf("suppressed fallback token = %d, want lone survivor %d", ids[0], survivor)
	}
}
