// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"iter"
	"slices"
	"sync"

	core "dappco.re/go"
	"dappco.re/go/inference"
	state "dappco.re/go/inference/state"
	"dappco.re/go/inference/state/agent"
	"dappco.re/go/inference/bundle"
	"dappco.re/go/mlx/chat"
)

// ConversationContinuityOptions configures the no-prompt-replay chat loop:
// each stateless chat request is matched to the conversation whose retained
// state covers its message prefix, woken (RAM-resident first, state-store
// second), appended with only the new turns, and slept back after the turn.
//
//	store, _ := filestore.Open(ctx, "~/Lethean/data/state/conversations.kv")
//	cc, _ := mlx.EnableConversationContinuity(tm, mlx.ConversationContinuityOptions{Store: store})
type ConversationContinuityOptions struct {
	// Store is the durable state store. It must also implement state.Writer
	// so finished turns can sleep; filestore and the in-memory store both do.
	Store state.Store
	// MaxResident caps RAM-resident conversations; older conversations are
	// closed on eviction and wake from the store on their next turn. 0 = 4.
	MaxResident int
	// EntryPrefix namespaces conversation state entry URIs. "" = "mlx://conversation/".
	EntryPrefix string
}

// ContinuityStats counts the paths conversation turns took — the boot notice
// and tests read these.
type ContinuityStats struct {
	FreshConversations int // prefilled from scratch (no matching state)
	ResidentTurns      int // continued on a RAM-resident session
	StoreWakes         int // woken from the state store
	Sleeps             int // turns slept to the store
	StatelessFallbacks int // requests served by the stateless path
}

// ConversationContinuity keeps conversations resident across stateless chat
// requests. Create with NewConversationContinuity or wire into a loaded text
// model with EnableConversationContinuity.
type ConversationContinuity struct {
	target conversationContinuityTarget
	store  state.Store
	writer state.Writer
	prefix string
	max    int

	mu       sync.Mutex
	resident map[string]*residentConversation
	order    []string // oldest first, for eviction
	stats    ContinuityStats
}

type conversationContinuityTarget interface {
	NewContinuitySession() (*ModelSession, error)
	FormatContinuityChat(messages []inference.Message, thinking *bool, continuation bool) string
	ContinuityNative() any
}

type modelContinuityTarget struct {
	model *Model
}

func (t modelContinuityTarget) NewContinuitySession() (*ModelSession, error) {
	if t.model == nil {
		return nil, errModelNil
	}
	return t.model.NewSession()
}

func (t modelContinuityTarget) FormatContinuityChat(messages []inference.Message, thinking *bool, continuation bool) string {
	if t.model == nil {
		return ""
	}
	return t.model.formatChatTurns(messages, thinking, continuation)
}

func (t modelContinuityTarget) ContinuityNative() any {
	if t.model == nil {
		return nil
	}
	return t.model.Native()
}

type residentConversation struct {
	session *ModelSession
	busy    bool
	dead    bool
	// Parent chain for incremental sleeps — the previous turn's slept URIs.
	parentEntry  string
	parentBundle string
	parentIndex  string
}

// NewConversationContinuity builds the manager for a loaded model.
func NewConversationContinuity(model *Model, opts ConversationContinuityOptions) (*ConversationContinuity, error) {
	if model == nil {
		return nil, core.E("mlx.NewConversationContinuity", "model is nil", nil)
	}
	return newConversationContinuityForTarget("mlx.NewConversationContinuity", modelContinuityTarget{model: model}, opts)
}

func newConversationContinuityForTarget(scope string, target conversationContinuityTarget, opts ConversationContinuityOptions) (*ConversationContinuity, error) {
	if target == nil {
		return nil, core.E(scope, "model is nil", nil)
	}
	if opts.Store == nil {
		return nil, core.E(scope, "state store is nil", nil)
	}
	// Block-diffusion models decode canvases against a per-request prefill —
	// the AR session machinery (retained KV, per-turn sleep/wake) does not
	// apply, and running it on the diffusion trunk is the #77 serve-book OOM.
	// The serve falls back to stateless chat, which routes through
	// Model.Generate's block-diffusion lane.
	if bd, ok := target.ContinuityNative().(interface{ BlockDiffusionCapable() bool }); ok && bd.BlockDiffusionCapable() {
		return nil, core.E(scope, "block-diffusion model decodes per request — continuity does not apply; the diffusion route serves it directly", nil)
	}
	writer, ok := opts.Store.(state.Writer)
	if !ok {
		return nil, core.E(scope, "state store does not implement state.Writer", nil)
	}
	maxResident := opts.MaxResident
	if maxResident <= 0 {
		maxResident = 4
	}
	prefix := opts.EntryPrefix
	if prefix == "" {
		prefix = "mlx://conversation/"
	}
	return &ConversationContinuity{
		target:   target,
		store:    opts.Store,
		writer:   writer,
		prefix:   prefix,
		max:      maxResident,
		resident: make(map[string]*residentConversation, maxResident),
	}, nil
}

// Stats returns a snapshot of the turn-path counters.
func (c *ConversationContinuity) Stats() ContinuityStats {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.stats
}

// conversationTurnSplit returns the index where the request's new turn
// begins: the trailing run of user/tool messages. Everything before it is the
// prefix a prior turn's retained state covers.
func conversationTurnSplit(messages []inference.Message) int {
	end := len(messages)
	for end > 0 {
		switch chat.NormaliseRole(messages[end-1].Role) {
		case "user", "tool":
			end--
		default:
			return end
		}
	}
	return end
}

// conversationKey hashes a message prefix into the state key a finished turn
// stores under and the next request looks up by.
func conversationKey(messages []inference.Message) string {
	builder := core.NewBuilder()
	for _, msg := range messages {
		builder.WriteString(chat.NormaliseRole(msg.Role))
		builder.WriteString("\x00")
		builder.WriteString(msg.Content)
		builder.WriteString("\x01")
	}
	return bundle.HashString(builder.String())
}

// Chat runs one continuity turn and reports whether it accepted the request.
// A false return means the caller serves the request statelessly — continuity
// never breaks serving; it declines (no trailing user turn, the conversation
// is mid-turn elsewhere, or wake/prefill failed) and the stateless path is
// always correct, just slower.
func (c *ConversationContinuity) Chat(ctx context.Context, messages []inference.Message, opts ...inference.GenerateOption) (iter.Seq[inference.Token], bool) {
	if c == nil || len(messages) == 0 {
		return nil, false
	}
	if inferenceMessagesCarryImages(messages) {
		// Continuity sessions carry text turns in woken KV; image turns go
		// through the stateless vision lane.
		return nil, false
	}
	cfg := inference.ApplyGenerateOpts(opts)
	conv, tailStart, err := c.acquire(ctx, messages)
	if err != nil {
		core.Error("mlx: conversation continuity declined; serving statelessly", "error", err)
		c.mu.Lock()
		c.stats.StatelessFallbacks++
		c.mu.Unlock()
		return nil, false
	}

	// Prefill before committing to the streamed sequence so failures here
	// still fall back to the stateless path.
	var prefillErr error
	if tailStart == 0 {
		prefillErr = conv.session.Prefill(c.target.FormatContinuityChat(messages, cfg.EnableThinking, false))
	} else {
		prefillErr = conv.session.AppendPrompt(c.target.FormatContinuityChat(messages[tailStart:], cfg.EnableThinking, true))
	}
	if prefillErr != nil {
		core.Error("mlx: conversation continuity prefill failed; serving statelessly", "error", prefillErr)
		conv.session.Close()
		c.mu.Lock()
		c.stats.StatelessFallbacks++
		c.mu.Unlock()
		return nil, false
	}

	return func(yield func(inference.Token) bool) {
		reply := core.NewBuilder()
		for token := range conv.session.GenerateStream(ctx, rootGenerateOptions(cfg)...) {
			reply.WriteString(token.Text)
			if !yield(inference.Token{ID: token.ID, Text: token.Text}) {
				break
			}
		}
		if err := conv.session.Err(); err != nil {
			core.Error("mlx: conversation continuity generation failed", "error", err)
			conv.dead = true
		}
		// A client that disconnected mid-stream received exactly the tokens
		// generated so far, so its next request's prefix matches the partial
		// state — sleeping it is correct, not a compromise.
		c.finishTurn(ctx, conv, messages, reply.String())
	}, true
}

// acquire resolves the session a request rides: RAM-resident match, store
// wake, or a fresh session. tailStart is the index of the first message that
// still needs prefilling (0 = the whole conversation).
func (c *ConversationContinuity) acquire(ctx context.Context, messages []inference.Message) (*residentConversation, int, error) {
	split := conversationTurnSplit(messages)
	if split == len(messages) {
		return nil, 0, core.E("mlx.ConversationContinuity", "request has no trailing user turn", nil)
	}

	if split > 0 {
		key := conversationKey(messages[:split])
		c.mu.Lock()
		if conv := c.resident[key]; conv != nil {
			if conv.busy {
				c.mu.Unlock()
				return nil, 0, core.E("mlx.ConversationContinuity", "conversation is mid-turn", nil)
			}
			conv.busy = true
			delete(c.resident, key)
			c.removeOrderLocked(key)
			c.stats.ResidentTurns++
			c.mu.Unlock()
			return conv, split, nil
		}
		c.mu.Unlock()

		entryURI := c.prefix + key
		indexURI := entryURI + "/index"
		if _, idxErr := agent.LoadStateIndex(ctx, c.store, indexURI); idxErr == nil {
			sess, err := c.target.NewContinuitySession()
			if err != nil {
				return nil, 0, err
			}
			if _, err := sess.WakeAgentMemory(ctx, c.store, agent.WakeOptions{IndexURI: indexURI, EntryURI: entryURI}); err != nil {
				sess.Close()
				return nil, 0, core.E("mlx.ConversationContinuity", "wake conversation state", err)
			}
			c.mu.Lock()
			c.stats.StoreWakes++
			c.mu.Unlock()
			return &residentConversation{
				session:      sess,
				busy:         true,
				parentEntry:  entryURI,
				parentBundle: entryURI + "/bundle",
				parentIndex:  indexURI,
			}, split, nil
		} else {
			var notFound *state.URIChunkNotFoundError
			if !core.As(idxErr, &notFound) {
				return nil, 0, core.E("mlx.ConversationContinuity", "probe conversation state", idxErr)
			}
		}
	}

	sess, err := c.target.NewContinuitySession()
	if err != nil {
		return nil, 0, err
	}
	c.mu.Lock()
	c.stats.FreshConversations++
	c.mu.Unlock()
	return &residentConversation{session: sess, busy: true}, 0, nil
}

// finishTurn sleeps the grown state under the key the NEXT request will look
// up (the conversation including this turn's reply), re-registers the session
// RAM-resident, and evicts beyond the cap. Sleep failure keeps the
// conversation RAM-resident only — turns keep working, durability resumes on
// the next successful sleep.
func (c *ConversationContinuity) finishTurn(ctx context.Context, conv *residentConversation, messages []inference.Message, reply string) {
	if conv.dead {
		conv.session.Close()
		return
	}
	full := append(slices.Clone(messages), inference.Message{Role: "assistant", Content: reply})
	key := conversationKey(full)
	entryURI := c.prefix + key
	sleepOpts := agent.SleepOptions{EntryURI: entryURI, Title: "conversation"}
	if conv.parentEntry != "" {
		sleepOpts.ParentEntryURI = conv.parentEntry
		sleepOpts.ParentBundleURI = conv.parentBundle
		sleepOpts.ParentIndexURI = conv.parentIndex
		sleepOpts.ReuseParentPrefix = true
		// The parent IS this session's own prior sleep and the session is
		// append-only between turns — the prefix is identical by
		// construction, so the sleep captures only the new turn's blocks.
		sleepOpts.ReuseParentPrefixTrusted = true
	}
	if report, err := conv.session.SleepAgentMemory(ctx, c.writer, sleepOpts); err != nil {
		core.Error("mlx: conversation sleep failed; conversation stays RAM-resident only", "error", err)
	} else {
		conv.parentEntry = report.EntryURI
		conv.parentBundle = report.BundleURI
		conv.parentIndex = report.IndexURI
		c.mu.Lock()
		c.stats.Sleeps++
		c.mu.Unlock()
	}

	c.mu.Lock()
	conv.busy = false
	c.resident[key] = conv
	c.order = append(c.order, key)
	for len(c.order) > c.max {
		oldest := c.order[0]
		evicted := c.resident[oldest]
		if evicted == nil || evicted.busy {
			break
		}
		c.order = c.order[1:]
		delete(c.resident, oldest)
		evicted.session.Close()
	}
	c.mu.Unlock()
}

func (c *ConversationContinuity) removeOrderLocked(key string) {
	for i, existing := range c.order {
		if existing == key {
			c.order = append(c.order[:i], c.order[i+1:]...)
			return
		}
	}
}

// rootGenerateOptions translates the inference-level request knobs onto the
// session generate options. EnableThinking is honoured at format time.
func rootGenerateOptions(cfg inference.GenerateConfig) []inference.GenerateOption {
	opts := make([]inference.GenerateOption, 0, 6)
	if cfg.MaxTokens > 0 {
		opts = append(opts, inference.WithMaxTokens(cfg.MaxTokens))
	}
	opts = append(opts, inference.WithTemperature(cfg.Temperature))
	if cfg.TopK > 0 {
		opts = append(opts, inference.WithTopK(cfg.TopK))
	}
	if cfg.TopP > 0 {
		opts = append(opts, inference.WithTopP(cfg.TopP))
	}
	if len(cfg.StopTokens) > 0 {
		opts = append(opts, inference.WithStopTokens(cfg.StopTokens...))
	}
	if cfg.RepeatPenalty > 0 && cfg.RepeatPenalty != 1 {
		opts = append(opts, inference.WithRepeatPenalty(cfg.RepeatPenalty))
	}
	if cfg.ThinkingBudget > 0 {
		opts = append(opts, inference.WithThinkingBudget(cfg.ThinkingBudget))
	}
	return opts
}

// EnableConversationContinuity wires the no-prompt-replay conversation loop
// into a loaded text model's chat path. Requests the manager declines are
// served statelessly, so enabling it never breaks serving.
//
//	cc, err := mlx.EnableConversationContinuity(tm, mlx.ConversationContinuityOptions{Store: store})
func EnableConversationContinuity(tm inference.TextModel, opts ConversationContinuityOptions) (*ConversationContinuity, error) {
	attachable, ok := tm.(conversationContinuityAttachable)
	if !ok {
		return nil, core.E("mlx.EnableConversationContinuity", "text model does not support conversation continuity", nil)
	}
	return attachable.enableConversationContinuity(opts)
}

type conversationContinuityAttachable interface {
	enableConversationContinuity(ConversationContinuityOptions) (*ConversationContinuity, error)
}

func (adapter *metaladapter) enableConversationContinuity(opts ConversationContinuityOptions) (*ConversationContinuity, error) {
	if adapter == nil {
		return nil, core.E("mlx.EnableConversationContinuity", "model is nil", nil)
	}
	continuity, err := NewConversationContinuity(adapter.rootModel(), opts)
	if err != nil {
		return nil, err
	}
	adapter.continuity = continuity
	return continuity, nil
}
