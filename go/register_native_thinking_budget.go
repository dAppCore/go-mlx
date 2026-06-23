// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package mlx

import (
	"dappco.re/go/inference"
	"dappco.re/go/mlx/pkg/tokenizer"
)

const (
	nativeGemma4ChannelOpenMarker  = "<|channel>"
	nativeGemma4ChannelCloseMarker = "<channel|>"
)

// nativeThinkingBudgetTracker is the native serve sibling of pkg/metal's
// thinking-budget state machine. It observes the selected token and returns the
// token that must be committed into the decode cache.
type nativeThinkingBudgetTracker struct {
	budget     int
	openID     int32
	closeID    int32
	inThought  bool
	count      int
	forced     bool
	everForced bool
}

func (t *nativeThinkingBudgetTracker) observe(id int32) int32 {
	if t == nil || t.budget <= 0 || t.openID == t.closeID {
		return id
	}
	if t.inThought && !t.forced && t.count >= t.budget && id != t.closeID {
		t.inThought = false
		t.forced = true
		t.everForced = true
		return t.closeID
	}
	switch id {
	case t.openID:
		t.inThought = true
		t.count = 0
		t.forced = false
	case t.closeID:
		t.inThought = false
	default:
		if t.inThought {
			t.count++
		}
	}
	return id
}

func (t *nativeThinkingBudgetTracker) forcedClose() bool {
	return t != nil && t.everForced
}

func newNativeThinkingBudgetTracker(tok *tokenizer.Tokenizer, cfg inference.GenerateConfig) *nativeThinkingBudgetTracker {
	if cfg.ThinkingBudget <= 0 || tok == nil {
		return nil
	}
	open, openOK := nativeGemma4SpecialTokenID(tok, nativeGemma4ChannelOpenMarker)
	close, closeOK := nativeGemma4SpecialTokenID(tok, nativeGemma4ChannelCloseMarker)
	if !openOK || !closeOK || open == close {
		return nil
	}
	return &nativeThinkingBudgetTracker{budget: cfg.ThinkingBudget, openID: open, closeID: close}
}

func nativeGemma4SpecialTokenID(tok *tokenizer.Tokenizer, marker string) (int32, bool) {
	ids := tok.Encode(marker)
	if tok.HasBOSToken() && len(ids) > 0 && ids[0] == tok.BOSToken() {
		ids = ids[1:]
	}
	if len(ids) != 1 {
		return 0, false
	}
	return ids[0], true
}
