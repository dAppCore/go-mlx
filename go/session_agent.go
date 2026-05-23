// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"iter"
	"strconv"

	core "dappco.re/go"
	"dappco.re/go/inference"
	state "dappco.re/go/inference/state"
	"dappco.re/go/mlx/agent"
	mlxbundle "dappco.re/go/mlx/bundle"
	"dappco.re/go/mlx/kv"
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

const foldedAgentMemoryPrefillWakeMaxTokens = 16 * 1024

// Hoisted sentinel errors. Each of these is returned multiple times from
// the agent-memory lifecycle entry points; promoting them to package vars
// removes per-call allocation in the validation hot path. errMLXModelNil
// is shared with backend.go (same error message across many call sites).
var (
	errAgentMemorySessionNil       = core.NewError("mlx: model session is nil")
	errAgentMemoryStoreNil         = core.NewError("mlx: state store is nil")
	errAgentMemoryExhaustedNil     = core.NewError("mlx: exhausted model session is nil")
	errAgentMemoryFoldEmpty        = core.NewError("mlx: folded State requires summary, recent tail, or folded prompt")
	errAgentMemoryFoldPlanNil      = core.NewError("mlx: folded State wake plan is nil")
	errAgentMemoryFoldNoTokens     = core.NewError("mlx: folded State prefill wake loaded no tokens")
	errAgentMemoryForkNeedsStore   = core.NewError("mlx: inference State fork requires state.Store")
	errAgentMemoryWakeNeedsStore   = core.NewError("mlx: inference agent memory wake requires state.Store")
	errAgentMemorySleepNeedsStore  = core.NewError("mlx: inference State sleep requires state.Writer")
	errAgentMemoryReuseNeedsReader = core.NewError("mlx: State parent-prefix reuse requires a readable state store")
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
	session, report, err := m.ForkFromBundle(ctx, store, agentMemoryWakeOptionsFromInference(req))
	if err != nil {
		return nil, nil, err
	}
	return session, toInferenceAgentMemoryWakeResult(report), nil
}

// WakeAgentMemory restores this session from a durable indexed KV prefix.
func (s *ModelSession) WakeAgentMemory(ctx context.Context, store state.Store, opts agent.WakeOptions) (*agent.WakeReport, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if s == nil || s.session == nil {
		return nil, errAgentMemorySessionNil
	}
	plan, err := agent.PlanWake(ctx, store, opts, modelInfoToMemory(s.info))
	if err != nil {
		return nil, err
	}
	// Cache the prefix length — consumed by metalKVSnapshotBlockSource and
	// LoadPrefixFromStateBlocksWithOptions on the two non-folded paths, and
	// re-read inside shouldPrefillFoldedAgentMemory's bounds check.
	prefixTokens := plan.Entry.PrefixTokens()
	if shouldPrefillFoldedAgentMemory(plan.Entry) {
		if err := s.prefillFoldedAgentMemory(ctx, store, plan, opts); err != nil {
			return nil, err
		}
		plan.Report.RestoreStrategy = "folded-prefill"
		s.agentMemory = agent.CloneWakeReport(plan.Report)
		return plan.Report, nil
	}
	if restorer, ok := s.session.(nativeSessionKVBlockRestorer); ok {
		source, err := metalKVSnapshotBlockSource(ctx, store, plan.Bundle, prefixTokens)
		if err != nil {
			return nil, err
		}
		if err := restorer.RestoreKVBlocks(ctx, source); err != nil {
			return nil, err
		}
		plan.Report.RestoreStrategy = "kv-blocks"
		s.agentMemory = agent.CloneWakeReport(plan.Report)
		return plan.Report, nil
	}
	snapshot, err := kv.LoadPrefixFromStateBlocksWithOptions(ctx, store, plan.Bundle, prefixTokens, opts.LoadOptions)
	if err != nil {
		return nil, err
	}
	if err := s.RestoreKV(snapshot); err != nil {
		return nil, err
	}
	plan.Report.RestoreStrategy = "snapshot"
	s.agentMemory = agent.CloneWakeReport(plan.Report)
	return plan.Report, nil
}

// Wake is a lifecycle alias for WakeAgentMemory.
func (s *ModelSession) Wake(ctx context.Context, store state.Store, opts agent.WakeOptions) (*agent.WakeReport, error) {
	return s.WakeAgentMemory(ctx, store, opts)
}

func shouldPrefillFoldedAgentMemory(entry agent.StateIndexEntry) bool {
	prefix := entry.PrefixTokens()
	if prefix <= 0 || prefix > foldedAgentMemoryPrefillWakeMaxTokens {
		return false
	}
	if meta := entry.Meta["folded_state"]; meta != "" {
		// Canonical-form fast path. foldedAgentMemorySleepOptions writes
		// "true" verbatim — the round-trip producer / consumer pairing
		// hits the byte-equal branch and skips Lower + Trim work.
		if meta == "true" || core.Lower(core.Trim(meta)) == "true" {
			return true
		}
	}
	for _, label := range entry.Labels {
		if label == "" {
			continue
		}
		// Canonical-form fast path. foldedAgentMemorySleepOptions appends
		// "folded-state" verbatim — same round-trip pairing argument.
		if label == "folded-state" || core.Lower(core.Trim(label)) == "folded-state" {
			return true
		}
	}
	return false
}

func (s *ModelSession) prefillFoldedAgentMemory(ctx context.Context, store state.Store, plan *agent.WakePlan, opts agent.WakeOptions) error {
	if s == nil || s.session == nil {
		return errAgentMemorySessionNil
	}
	if plan == nil || plan.Bundle == nil {
		return errAgentMemoryFoldPlanNil
	}
	loadOpts := opts.LoadOptions
	if plan.Bundle.KVEncoding == kv.EncodingNative {
		loadOpts.RawKVOnly = true
	}
	tokens, err := kv.LoadPrefixTokensFromStateBlocksWithOptions(ctx, store, plan.Bundle, plan.Entry.PrefixTokens(), loadOpts)
	if err != nil {
		return core.E("mlx: folded State prefill wake", "load tokens", err)
	}
	if len(tokens) == 0 {
		return errAgentMemoryFoldNoTokens
	}
	if err := s.PrefillTokens(ctx, tokens); err != nil {
		return core.E("mlx: folded State prefill wake", "prefill", err)
	}
	return nil
}

// WakeState implements the backend-neutral go-inference agent-memory contract.
func (s *ModelSession) WakeState(ctx context.Context, req inference.AgentMemoryWakeRequest) (*inference.AgentMemoryWakeResult, error) {
	store, ok := req.Store.(state.Store)
	if !ok {
		return nil, errAgentMemoryWakeNeedsStore
	}
	report, err := s.WakeAgentMemory(ctx, store, agentMemoryWakeOptionsFromInference(req))
	if err != nil {
		return nil, err
	}
	return toInferenceAgentMemoryWakeResult(report), nil
}

// SleepAgentMemory streams this session's current KV state to State blocks,
// then writes a bundle manifest and one-entry wake index.
func (s *ModelSession) SleepAgentMemory(ctx context.Context, store state.Writer, opts agent.SleepOptions) (*agent.SleepReport, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if s == nil || s.session == nil {
		return nil, errAgentMemorySessionNil
	}
	if store == nil {
		return nil, errAgentMemoryStoreNil
	}
	entryURI, bundleURI, indexURI, err := agent.SleepURIs(opts)
	if err != nil {
		return nil, err
	}
	if opts.ModelInfo.Architecture == "" {
		opts.ModelInfo = modelInfoToMemory(s.info)
	}
	// Hoist the s.agentMemory nil check — was repeated three times in
	// independent branch predicates. Single load + reused alias lets the
	// three assignments share one pointer dereference each.
	if parent := s.agentMemory; parent != nil {
		if opts.ParentEntryURI == "" {
			opts.ParentEntryURI = parent.EntryURI
		}
		if opts.ParentBundleURI == "" {
			opts.ParentBundleURI = parent.BundleURI
		}
		if opts.ParentIndexURI == "" {
			opts.ParentIndexURI = parent.IndexURI
		}
	}
	blockOpts := agent.SleepBlockOptions(opts, bundleURI)
	if opts.ReuseParentPrefix && blockOpts.ReusePrefix == nil {
		readStore, ok := store.(state.Store)
		if !ok {
			return nil, errAgentMemoryReuseNeedsReader
		}
		parentBundle, err := kv.LoadStateBlockBundle(ctx, readStore, opts.ParentBundleURI)
		if err != nil {
			return nil, err
		}
		blockOpts.ReusePrefix = parentBundle
		if blockOpts.ReusePrefixTokens <= 0 {
			blockOpts.ReusePrefixTokens = parentBundle.TokenCount
		}
	}
	bundle, err := s.SaveKVBlocksToState(ctx, store, blockOpts)
	if err != nil {
		return nil, err
	}
	bundleRef, err := kv.SaveStateBlockBundle(ctx, store, bundle, bundleURI)
	if err != nil {
		return nil, err
	}
	index, err := agent.NewSleepIndex(bundle, opts, entryURI, bundleURI)
	if err != nil {
		return nil, err
	}
	indexRef, err := agent.SaveStateIndex(ctx, store, index, indexURI)
	if err != nil {
		return nil, err
	}
	report := agent.NewSleepReport(index, bundle, opts, entryURI, bundleURI, indexURI, bundleRef, indexRef)
	s.agentMemory = agent.WakeReportFromSleep(report)
	return report, nil
}

// Sleep is a lifecycle alias for SleepAgentMemory.
func (s *ModelSession) Sleep(ctx context.Context, store state.Writer, opts agent.SleepOptions) (*agent.SleepReport, error) {
	return s.SleepAgentMemory(ctx, store, opts)
}

// SleepState implements the backend-neutral go-inference agent-memory contract.
func (s *ModelSession) SleepState(ctx context.Context, req inference.AgentMemorySleepRequest) (*inference.AgentMemorySleepResult, error) {
	store, ok := req.Store.(state.Writer)
	if !ok {
		return nil, errAgentMemorySleepNeedsStore
	}
	report, err := s.SleepAgentMemory(ctx, store, agentMemorySleepOptionsFromInference(req))
	if err != nil {
		return nil, err
	}
	return toInferenceAgentMemorySleepResult(report), nil
}

// AppendAndSleepAgentMemory appends new prompt material and then streams the
// resulting state to durable storage without forcing a generation/reply step.
func (s *ModelSession) AppendAndSleepAgentMemory(ctx context.Context, prompt string, store state.Writer, opts agent.SleepOptions) (*agent.SleepReport, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if err := s.AppendPrompt(prompt); err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return s.SleepAgentMemory(ctx, store, opts)
}

// AppendAndSleep is a lifecycle alias for AppendAndSleepAgentMemory.
func (s *ModelSession) AppendAndSleep(ctx context.Context, prompt string, store state.Writer, opts agent.SleepOptions) (*agent.SleepReport, error) {
	return s.AppendAndSleepAgentMemory(ctx, prompt, store, opts)
}

// GenerateAndSleepAgentMemory generates an answer from the current retained
// state and streams the post-answer KV state to durable storage.
func (s *ModelSession) GenerateAndSleepAgentMemory(ctx context.Context, store state.Writer, opts agent.SleepOptions, generateOpts ...GenerateOption) (string, *agent.SleepReport, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return "", nil, err
	}
	if s == nil || s.session == nil {
		return "", nil, errAgentMemorySessionNil
	}
	builder := core.NewBuilder()
	// Generations typically produce hundreds of tokens of text. Pre-grow
	// the backing slice to skip the early 64 -> 128 -> 256 -> 512 -> 1024
	// reallocations during token streaming.
	builder.Grow(1024)
	cfg := toMetalGenerateConfig(applyGenerateOptions(generateOpts))
	for tok := range s.session.Generate(ctx, cfg) {
		builder.WriteString(tok.Text)
	}
	if err := s.session.Err(); err != nil {
		return builder.String(), nil, err
	}
	if err := ctx.Err(); err != nil {
		return builder.String(), nil, err
	}
	report, err := s.SleepAgentMemory(ctx, store, opts)
	if err != nil {
		return builder.String(), nil, err
	}
	return builder.String(), report, nil
}

// GenerateAndSleep is a lifecycle alias for GenerateAndSleepAgentMemory.
func (s *ModelSession) GenerateAndSleep(ctx context.Context, store state.Writer, opts agent.SleepOptions, generateOpts ...GenerateOption) (string, *agent.SleepReport, error) {
	return s.GenerateAndSleepAgentMemory(ctx, store, opts, generateOpts...)
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
	if exhausted == nil || exhausted.session == nil {
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

func agentMemoryWakeOptionsFromInference(req inference.AgentMemoryWakeRequest) agent.WakeOptions {
	return agent.WakeOptions{
		IndexURI:               req.IndexURI,
		EntryURI:               req.EntryURI,
		Tokenizer:              stateBundleTokenizerFromInference(req.Tokenizer),
		SkipCompatibilityCheck: req.SkipCompatibilityCheck,
	}
}

func agentMemorySleepOptionsFromInference(req inference.AgentMemorySleepRequest) agent.SleepOptions {
	return agent.SleepOptions{
		EntryURI:          req.EntryURI,
		BundleURI:         req.BundleURI,
		IndexURI:          req.IndexURI,
		ParentEntryURI:    req.ParentEntryURI,
		ParentBundleURI:   req.ParentBundleURI,
		ParentIndexURI:    req.ParentIndexURI,
		Title:             req.Title,
		Model:             req.Model.ID,
		ModelPath:         req.Model.Path,
		ModelInfo:         modelInfoToMemory(modelInfoFromInferenceIdentity(req.Model)),
		Tokenizer:         stateBundleTokenizerFromInference(req.Tokenizer),
		ReuseParentPrefix: req.ReuseParentPrefix,
		BlockOptions: kv.StateBlockOptions{
			BlockSize:  req.BlockSize,
			KVEncoding: kv.Encoding(req.Encoding),
		},
		Labels: agentMemoryLabelsFromInference(req.Labels),
		Meta:   agentMemoryMetadataFromInference(req),
	}
}

func stateBundleTokenizerFromInference(tokenizer inference.TokenizerIdentity) mlxbundle.Tokenizer {
	return mlxbundle.NormaliseTokenizer(mlxbundle.Tokenizer{
		Kind:         tokenizer.Kind,
		Path:         tokenizer.Path,
		Hash:         tokenizer.Hash,
		BOS:          tokenizer.BOSID,
		EOS:          tokenizer.EOSID,
		ChatTemplate: tokenizer.ChatTemplate,
	})
}

func modelInfoFromInferenceIdentity(model inference.ModelIdentity) ModelInfo {
	return ModelInfo{
		Architecture:  model.Architecture,
		VocabSize:     model.VocabSize,
		NumLayers:     model.NumLayers,
		HiddenSize:    model.HiddenSize,
		QuantBits:     model.QuantBits,
		QuantGroup:    model.QuantGroup,
		ContextLength: model.ContextLength,
	}
}

func toInferenceAgentMemoryWakeResult(report *agent.WakeReport) *inference.AgentMemoryWakeResult {
	if report == nil {
		return nil
	}
	return &inference.AgentMemoryWakeResult{
		Entry: inference.AgentMemoryRef{
			URI:        report.EntryURI,
			BundleURI:  report.BundleURI,
			IndexURI:   report.IndexURI,
			Title:      report.Title,
			Hash:       report.SnapshotHash,
			TokenStart: 0,
			TokenCount: report.PrefixTokens,
		},
		Bundle:       agentMemoryStateRef(report.BundleURI, kv.StateBlockBundleKind, report.SnapshotHash, ""),
		Index:        agentMemoryStateRef(report.IndexURI, agent.StateIndexKind, report.IndexHash, ""),
		PrefixTokens: report.PrefixTokens,
		BundleTokens: report.BundleTokens,
		BlockSize:    report.BlockSize,
		BlocksRead:   report.BlocksRead,
	}
}

func toInferenceAgentMemorySleepResult(report *agent.SleepReport) *inference.AgentMemorySleepResult {
	if report == nil {
		return nil
	}
	// Hoist the KVEncoding string conversion — same value is consumed by
	// both the Bundle ref and the top-level Encoding field.
	encoding := string(report.KVEncoding)
	return &inference.AgentMemorySleepResult{
		Entry: inference.AgentMemoryRef{
			URI:        report.EntryURI,
			BundleURI:  report.BundleURI,
			IndexURI:   report.IndexURI,
			Title:      report.Title,
			Hash:       report.SnapshotHash,
			TokenStart: 0,
			TokenCount: report.TokenCount,
		},
		Parent: inference.AgentMemoryRef{
			URI:       report.ParentEntryURI,
			BundleURI: report.ParentBundleURI,
			IndexURI:  report.ParentIndexURI,
		},
		Bundle:        agentMemoryStateRef(report.BundleURI, kv.StateBlockBundleKind, report.SnapshotHash, encoding),
		Index:         agentMemoryStateRef(report.IndexURI, agent.StateIndexKind, report.IndexHash, ""),
		TokenCount:    report.TokenCount,
		BlockSize:     report.BlockSize,
		BlocksWritten: report.BlocksWritten,
		BlocksReused:  report.BlocksReused,
		Encoding:      encoding,
	}
}

func agentMemoryStateRef(uri, kind, hash, encoding string) inference.StateRef {
	return inference.StateRef{
		Kind:     kind,
		URI:      uri,
		Hash:     hash,
		Encoding: encoding,
	}
}

func agentMemoryLabelsFromInference(labels map[string]string) []string {
	if len(labels) == 0 {
		return nil
	}
	out := make([]string, 0, len(labels))
	// Tiny-N fast path: a single label avoids the size-pass + Builder
	// scaffolding (which only pays off when we have >=2 non-empty values
	// to share a backing buffer). Direct `key + "=" + value` allocates
	// once for the result string — same shape as the previous code,
	// without the per-iteration count overhead.
	if len(labels) == 1 {
		for key, value := range labels {
			if value == "" {
				out = append(out, key)
			} else {
				out = append(out, key+"="+value)
			}
		}
		return out
	}
	// Multi-entry path: build all "key=value" strings into a single
	// backing buffer, then slice that buffer into the []string output.
	// Saves one allocation per non-empty value vs the previous shape
	// (which alloced a fresh string per concat). Two-pass: size first
	// so the Builder buffer lands at the exact right capacity and the
	// growth ladder (8 -> 16 -> 32 ...) never kicks in.
	size := 0
	for key, value := range labels {
		if value == "" {
			continue
		}
		size += len(key) + 1 + len(value)
	}
	if size == 0 {
		// All-empty fast path — every entry aliases the map key.
		for key := range labels {
			out = append(out, key)
		}
		core.SliceSort(out)
		return out
	}
	var builder core.Builder
	builder.Grow(size)
	for key, value := range labels {
		if value == "" {
			out = append(out, key)
			continue
		}
		start := builder.Len()
		builder.WriteString(key)
		builder.WriteByte('=')
		builder.WriteString(value)
		// builder.String() returns the underlying buffer via unsafe —
		// every Grow-bounded write leaves earlier slices pinned to the
		// same backing memory, so it is safe to take a sub-slice here.
		out = append(out, builder.String()[start:])
	}
	core.SliceSort(out)
	return out
}

func agentMemoryMetadataFromInference(req inference.AgentMemorySleepRequest) map[string]string {
	// Pre-size the destination map. The 9 optional adapter/runtime fields
	// dominate the entry count — counting empties first lets us hand
	// runtime.makemap_small the exact capacity, replacing the addAgent
	// loop's incremental zero-cap growth.
	extras := 0
	if req.Adapter.Hash != "" {
		extras++
	}
	if req.Adapter.Path != "" {
		extras++
	}
	if req.Adapter.Format != "" {
		extras++
	}
	if req.Adapter.Rank != 0 {
		extras++
	}
	if req.Adapter.Alpha != 0 {
		extras++
	}
	if req.Runtime.Backend != "" {
		extras++
	}
	if req.Runtime.Device != "" {
		extras++
	}
	if req.Runtime.CacheMode != "" {
		extras++
	}
	if req.Runtime.Version != "" {
		extras++
	}
	if extras == 0 {
		// Nothing to fold in — defer to the existing clone, which
		// returns nil if req.Metadata is also empty (the common
		// idle-keepalive request shape).
		return cloneStringMap(req.Metadata)
	}
	// Fast path: no user-supplied metadata. Every adapter/runtime key is
	// fresh, so the addAgentMemoryMetadata 'meta[key] == ""' idempotence
	// read is wasted work — direct writes shave one map-probe per non-
	// empty field. Whitespace-only values still need to be filtered
	// (preserving addAgentMemoryMetadata's Trim safety check) — fields
	// like Adapter.Path can legitimately arrive as '   ' from upstream.
	if req.Metadata == nil {
		meta := make(map[string]string, extras)
		if v := req.Adapter.Hash; v != "" && core.Trim(v) != "" {
			meta["adapter_hash"] = v
		}
		if v := req.Adapter.Path; v != "" && core.Trim(v) != "" {
			meta["adapter_path"] = v
		}
		if v := req.Adapter.Format; v != "" && core.Trim(v) != "" {
			meta["adapter_format"] = v
		}
		if req.Adapter.Rank != 0 {
			meta["adapter_rank"] = strconv.Itoa(req.Adapter.Rank)
		}
		if req.Adapter.Alpha != 0 {
			meta["adapter_alpha"] = strconv.FormatFloat(float64(req.Adapter.Alpha), 'g', -1, 32)
		}
		if v := req.Runtime.Backend; v != "" && core.Trim(v) != "" {
			meta["runtime_backend"] = v
		}
		if v := req.Runtime.Device; v != "" && core.Trim(v) != "" {
			meta["runtime_device"] = v
		}
		if v := req.Runtime.CacheMode; v != "" && core.Trim(v) != "" {
			meta["runtime_cache_mode"] = v
		}
		if v := req.Runtime.Version; v != "" && core.Trim(v) != "" {
			meta["runtime_version"] = v
		}
		return meta
	}
	dst := make(map[string]string, len(req.Metadata)+extras)
	for k, v := range req.Metadata {
		dst[k] = v
	}
	// addAgentMemoryMetadata-equivalent inline writes — same idempotence
	// rule (don't overwrite caller-supplied keys) but skip the function
	// call. The Trim guard runs only for non-empty values (the counting
	// loop above already filtered v=="" out of extras, so the && short-
	// circuit makes Trim a one-time check per field).
	if v := req.Adapter.Hash; v != "" && dst["adapter_hash"] == "" && core.Trim(v) != "" {
		dst["adapter_hash"] = v
	}
	if v := req.Adapter.Path; v != "" && dst["adapter_path"] == "" && core.Trim(v) != "" {
		dst["adapter_path"] = v
	}
	if v := req.Adapter.Format; v != "" && dst["adapter_format"] == "" && core.Trim(v) != "" {
		dst["adapter_format"] = v
	}
	if req.Adapter.Rank != 0 && dst["adapter_rank"] == "" {
		dst["adapter_rank"] = strconv.Itoa(req.Adapter.Rank)
	}
	if req.Adapter.Alpha != 0 && dst["adapter_alpha"] == "" {
		dst["adapter_alpha"] = strconv.FormatFloat(float64(req.Adapter.Alpha), 'g', -1, 32)
	}
	if v := req.Runtime.Backend; v != "" && dst["runtime_backend"] == "" && core.Trim(v) != "" {
		dst["runtime_backend"] = v
	}
	if v := req.Runtime.Device; v != "" && dst["runtime_device"] == "" && core.Trim(v) != "" {
		dst["runtime_device"] = v
	}
	if v := req.Runtime.CacheMode; v != "" && dst["runtime_cache_mode"] == "" && core.Trim(v) != "" {
		dst["runtime_cache_mode"] = v
	}
	if v := req.Runtime.Version; v != "" && dst["runtime_version"] == "" && core.Trim(v) != "" {
		dst["runtime_version"] = v
	}
	return dst
}

func addAgentMemoryMetadata(meta map[string]string, key, value string) map[string]string {
	// Fast path: empty input is the dominant case for optional adapter
	// + runtime fields. Skip the core.Trim allocation entirely.
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
