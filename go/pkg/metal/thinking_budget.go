// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

// Thinking budget (#99): cap the tokens a reasoning model spends inside its
// thought channel, then force the channel-close token so the model is
// conditioned into its visible answer. Without it a reasoning model can burn
// its entire token allowance thinking and never emit a visible answer — the
// "content empty / finish=length mid-thought" failure.
//
// gemma4 emits an atomic open token (<|channel>) and an atomic close token
// (<channel|>); the budget counts tokens between them and forces the close
// when exceeded. The mechanism is a pure state machine (observe one committed
// token, return the token to actually emit) so the decode loops hook it with
// a single call and it is testable with no model. Budgeted requests opt out
// of the pipelined decode lane (which speculates the next token before the
// host can intervene) and ride the serial loop, where forcing is exact.

package metal

// ThinkingChannelModel is the optional capability a family model implements to
// expose its thought-channel delimiter tokens. A model that does not implement
// it simply has no thinking budget (the tracker stays inert).
type ThinkingChannelModel interface {
	// ThinkingChannelTokens returns the atomic open and close token IDs of
	// the model's thought channel; ok is false when the model has none.
	ThinkingChannelTokens() (open, close int32, ok bool)
}

// thinkingBudgetTracker enforces the cap. Zero budget or unresolved channel
// tokens make it inert (observe returns its input unchanged).
type thinkingBudgetTracker struct {
	budget    int
	openID    int32
	closeID   int32
	inThought bool
	count     int
	forced    bool // close already forced for the CURRENT channel
	everForced bool // forced at least once this generation (for metrics)
}

// observe takes the token the decode loop is about to commit and returns the
// token to actually emit. Normally that is the same token; when the thought
// channel is open and the budget is spent, it returns the close token instead
// (the loop forwards that, conditioning the model into its answer). It forces
// at most once per open channel, then stands down until the next open.
func (t *thinkingBudgetTracker) observe(id int32) int32 {
	if t == nil || t.budget <= 0 || t.openID == t.closeID {
		return id
	}
	// Over budget inside an open channel, and the model isn't closing on its
	// own this step: force the close.
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

// forcedClose reports whether the budget cut a thought channel short during
// this generation — surfaced on Metrics so a caller can see the budget bit.
func (t *thinkingBudgetTracker) forcedClose() bool {
	return t != nil && t.everForced
}

// newThinkingBudgetTracker builds the tracker for a generation, or nil when
// the budget is off or the model exposes no thought channel. Called once per
// generation, before the decode loop — never in the token hot path.
func (m *Model) newThinkingBudgetTracker(cfg GenerateConfig) *thinkingBudgetTracker {
	if cfg.ThinkingBudget <= 0 || m == nil || m.model == nil {
		return nil
	}
	channel, ok := m.model.(ThinkingChannelModel)
	if !ok {
		return nil
	}
	open, close, ok := channel.ThinkingChannelTokens()
	if !ok || open == close {
		return nil
	}
	return &thinkingBudgetTracker{budget: cfg.ThinkingBudget, openID: open, closeID: close}
}
