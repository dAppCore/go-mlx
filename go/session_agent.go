// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"iter"
	"strconv"

	core "dappco.re/go"
	"dappco.re/go/inference"
	state "dappco.re/go/inference/state"
	"dappco.re/go/inference/state/agent"
	session "dappco.re/go/inference/state/session"
)

// AgentMemoryFoldOptions controls how an exhausted live context is checkpointed
// and folded into a fresh summary-plus-tail state.
type AgentMemoryFoldOptions struct {
	Summary           string
	RecentTail        string
	FoldedPrompt      string
	PrefillChunkBytes int
	Checkpoint        agent.SleepOptions
	Folded            agent.SleepOptions
}

// AgentMemoryFoldReport describes the checkpointed exhausted state and the
// fresh folded state that should be used for subsequent turns.
type AgentMemoryFoldReport struct {
	Checkpoint        *agent.SleepReport `json:"checkpoint,omitempty"`
	Folded            *agent.SleepReport `json:"folded,omitempty"`
	SummaryBytes      int                `json:"summary_bytes,omitempty"`
	RecentTailBytes   int                `json:"recent_tail_bytes,omitempty"`
	FoldedPromptBytes int                `json:"folded_prompt_bytes,omitempty"`
}

// Hoisted sentinel errors. Each of these is returned multiple times from
// the agent-memory lifecycle entry points; promoting them to package vars
// removes per-call allocation in the validation hot path. errMLXModelNil
// is shared with backend.go (same error message across many call sites).
var (
	errAgentMemoryStoreNil       = core.NewError("mlx: state store is nil")
	errAgentMemoryExhaustedNil   = core.NewError("mlx: exhausted model session is nil")
	errAgentMemoryFoldEmpty      = core.NewError("mlx: folded State requires summary, recent tail, or folded prompt")
	errAgentMemoryForkNeedsStore = core.NewError("mlx: inference State fork requires state.Store")
)

// WakeAgentMemory creates a new session from a durable indexed KV prefix.
func (m *Model) WakeAgentMemory(ctx context.Context, store state.Store, opts agent.WakeOptions) (*ModelSession, *agent.WakeReport, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	session, err := m.NewSession()
	if err != nil {
		return nil, nil, err
	}
	report, err := session.WakeAgentMemory(ctx, store, opts)
	if err != nil {
		if closeErr := session.Close(); closeErr != nil {
			return nil, nil, core.ErrorJoin(err, closeErr)
		}
		return nil, nil, err
	}
	return session, report, nil
}

// Wake is a lifecycle alias for WakeAgentMemory.
func (m *Model) Wake(ctx context.Context, store state.Store, opts agent.WakeOptions) (*ModelSession, *agent.WakeReport, error) {
	return m.WakeAgentMemory(ctx, store, opts)
}

// ForkFromBundle creates an independent session from a durable indexed KV
// bundle entry. It is equivalent to waking from that bundle without mutating an
// existing session.
func (m *Model) ForkFromBundle(ctx context.Context, store state.Store, opts agent.WakeOptions) (*ModelSession, *agent.WakeReport, error) {
	return m.WakeAgentMemory(ctx, store, opts)
}

// ForkState implements the backend-neutral go-inference agent-memory contract.
func (m *Model) ForkState(ctx context.Context, req inference.AgentMemoryWakeRequest) (inference.AgentMemorySession, *inference.AgentMemoryWakeResult, error) {
	store, ok := req.Store.(state.Store)
	if !ok {
		return nil, nil, errAgentMemoryForkNeedsStore
	}
	sess, report, err := m.ForkFromBundle(ctx, store, session.WakeOptionsFromInference(req))
	if err != nil {
		return nil, nil, err
	}
	return sess, session.ToInferenceWakeResult(report), nil
}

// FoldAgentMemory checkpoints an exhausted retained state, creates a fresh
// session from summary-plus-tail text, and persists that folded state with
// parent lineage back to the checkpoint.
func (m *Model) FoldAgentMemory(ctx context.Context, exhausted *ModelSession, store state.Writer, opts AgentMemoryFoldOptions) (*ModelSession, *AgentMemoryFoldReport, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if m == nil || m.model == nil {
		return nil, nil, errMLXModelNil
	}
	if !exhausted.Valid() {
		return nil, nil, errAgentMemoryExhaustedNil
	}
	if store == nil {
		return nil, nil, errAgentMemoryStoreNil
	}
	prompt := agentMemoryFoldedPrompt(opts)
	// Empty-string fast path. agentMemoryFoldedPrompt returns "" when
	// none of summary/tail/FoldedPrompt are supplied; only a user-passed
	// whitespace-only FoldedPrompt reaches the slow Trim path.
	if prompt == "" || core.Trim(prompt) == "" {
		return nil, nil, errAgentMemoryFoldEmpty
	}
	report := &AgentMemoryFoldReport{
		SummaryBytes:      len(opts.Summary),
		RecentTailBytes:   len(opts.RecentTail),
		FoldedPromptBytes: len(prompt),
	}
	checkpoint, err := exhausted.SleepAgentMemory(ctx, store, opts.Checkpoint)
	if err != nil {
		return nil, report, err
	}
	report.Checkpoint = checkpoint
	folded, err := m.NewSession()
	if err != nil {
		return nil, report, err
	}
	if err := folded.PrefillChunks(ctx, agentMemoryTextChunks(prompt, opts.PrefillChunkBytes)); err != nil {
		if closeErr := folded.Close(); closeErr != nil {
			return nil, report, core.ErrorJoin(err, closeErr)
		}
		return nil, report, err
	}
	foldedOpts := foldedAgentMemorySleepOptions(opts.Folded, checkpoint, report)
	foldedReport, err := folded.SleepAgentMemory(ctx, store, foldedOpts)
	if err != nil {
		if closeErr := folded.Close(); closeErr != nil {
			return nil, report, core.ErrorJoin(err, closeErr)
		}
		return nil, report, err
	}
	report.Folded = foldedReport
	return folded, report, nil
}

func agentMemoryFoldedPrompt(opts AgentMemoryFoldOptions) string {
	// Empty-string fast path on FoldedPrompt — skip the Trim function
	// call when the user passed nothing at all. The hot caller
	// (FoldAgentMemory in libraries that build summary+tail explicitly)
	// almost always hits this branch.
	if opts.FoldedPrompt != "" && core.Trim(opts.FoldedPrompt) != "" {
		return opts.FoldedPrompt
	}
	// Skip Trim on already-empty Summary / RecentTail — the dominant case
	// in callers that rebuild the fold prompt with no checkpoint summary
	// yet (e.g. the bare error-path FoldAgentMemory call). Same outcome,
	// no function-call cost.
	if opts.Summary == "" && opts.RecentTail == "" {
		return ""
	}
	summary := core.Trim(opts.Summary)
	tail := core.Trim(opts.RecentTail)
	if summary == "" && tail == "" {
		return ""
	}
	// Static headers (~315 chars) + per-section wrappers (~30 each)
	// + content. Pre-sizing avoids 2-3 internal slice growths.
	size := 315
	if summary != "" {
		size += 24 + len(summary)
	}
	if tail != "" {
		size += 28 + len(tail)
	}
	builder := core.NewBuilder()
	builder.Grow(size)
	builder.WriteString("The previous retained context window reached its live-token budget and has been compacted into this folded state.\n\n")
	if summary != "" {
		builder.WriteString("<summary>\n")
		builder.WriteString(summary)
		builder.WriteString("\n</summary>\n\n")
	}
	if tail != "" {
		builder.WriteString("<recent_tail>\n")
		builder.WriteString(tail)
		builder.WriteString("\n</recent_tail>\n\n")
	}
	builder.WriteString("Use the summary as durable memory and the recent tail as the immediate continuation point. Do not assume the full exhausted context is still present.")
	return builder.String()
}

// foldedAgentMemorySleepOptions writes the "folded_state" meta and the
// "folded-state" label that session.shouldPrefillFoldedAgentMemory reads
// at wake — the producer/consumer pair spans the mlx and session packages.
func foldedAgentMemorySleepOptions(opts agent.SleepOptions, checkpoint *agent.SleepReport, report *AgentMemoryFoldReport) agent.SleepOptions {
	if opts.Title == "" {
		opts.Title = "folded State"
	}
	if checkpoint != nil {
		if opts.ParentEntryURI == "" {
			opts.ParentEntryURI = checkpoint.EntryURI
		}
		if opts.ParentBundleURI == "" {
			opts.ParentBundleURI = checkpoint.BundleURI
		}
		if opts.ParentIndexURI == "" {
			opts.ParentIndexURI = checkpoint.IndexURI
		}
	}
	opts.Meta = cloneStringMap(opts.Meta)
	opts.Meta = addAgentMemoryFoldMeta(opts.Meta, "folded_state", "true")
	if checkpoint != nil {
		opts.Meta = addAgentMemoryFoldMeta(opts.Meta, "folded_from_entry_uri", checkpoint.EntryURI)
	}
	if report != nil {
		opts.Meta = addAgentMemoryFoldMeta(opts.Meta, "summary_bytes", strconv.Itoa(report.SummaryBytes))
		opts.Meta = addAgentMemoryFoldMeta(opts.Meta, "recent_tail_bytes", strconv.Itoa(report.RecentTailBytes))
		opts.Meta = addAgentMemoryFoldMeta(opts.Meta, "folded_prompt_bytes", strconv.Itoa(report.FoldedPromptBytes))
	}
	cloned := make([]string, len(opts.Labels), len(opts.Labels)+1)
	copy(cloned, opts.Labels)
	opts.Labels = append(cloned, "folded-state")
	return opts
}

func addAgentMemoryFoldMeta(meta map[string]string, key, value string) map[string]string {
	// Fast path: empty input is the dominant case for absent fields.
	// Skip the core.Trim allocation entirely. Whitespace-only values
	// still fall through to the slow path below.
	if value == "" {
		return meta
	}
	if core.Trim(value) == "" {
		return meta
	}
	if meta == nil {
		meta = map[string]string{}
	}
	if meta[key] == "" {
		meta[key] = value
	}
	return meta
}

func agentMemoryTextChunks(text string, chunkBytes int) iter.Seq[string] {
	return func(yield func(string) bool) {
		if text == "" {
			return
		}
		if chunkBytes <= 0 || len(text) <= chunkBytes {
			yield(text)
			return
		}
		// Byte-level scan with rune-boundary alignment. The previous
		// implementation drove a `range text` loop which paid for full
		// UTF-8 decoding on every rune — N decodes per chunk to find
		// the boundary one rune past chunkBytes. Here we jump directly
		// to start+chunkBytes and only advance past UTF-8 continuation
		// bytes (top two bits 10xxxxxx) until we hit a rune-start byte.
		// Identical chunk boundaries, but O(text_bytes) byte compares
		// instead of O(text_bytes) full rune decodes.
		start := 0
		for start < len(text) {
			end := start + chunkBytes
			if end >= len(text) {
				yield(text[start:])
				return
			}
			for end < len(text) && text[end]&0xC0 == 0x80 {
				end++
			}
			if !yield(text[start:end]) {
				return
			}
			start = end
		}
	}
}
