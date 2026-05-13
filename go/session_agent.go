// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference"
	memvid "dappco.re/go/inference/state"
	"dappco.re/go/mlx/agent"
	mlxbundle "dappco.re/go/mlx/bundle"
	"dappco.re/go/mlx/kv"
)

// WakeAgentMemory creates a new session from a durable indexed KV prefix.
func (m *Model) WakeAgentMemory(ctx context.Context, store memvid.Store, opts agent.WakeOptions) (*ModelSession, *agent.WakeReport, error) {
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
func (m *Model) Wake(ctx context.Context, store memvid.Store, opts agent.WakeOptions) (*ModelSession, *agent.WakeReport, error) {
	return m.WakeAgentMemory(ctx, store, opts)
}

// ForkFromBundle creates an independent session from a durable indexed KV
// bundle entry. It is equivalent to waking from that bundle without mutating an
// existing session.
func (m *Model) ForkFromBundle(ctx context.Context, store memvid.Store, opts agent.WakeOptions) (*ModelSession, *agent.WakeReport, error) {
	return m.WakeAgentMemory(ctx, store, opts)
}

// ForkState implements the backend-neutral go-inference agent-memory contract.
func (m *Model) ForkState(ctx context.Context, req inference.AgentMemoryWakeRequest) (inference.AgentMemorySession, *inference.AgentMemoryWakeResult, error) {
	store, ok := req.Store.(memvid.Store)
	if !ok {
		return nil, nil, core.NewError("mlx: inference agent memory fork requires memvid.Store")
	}
	session, report, err := m.ForkFromBundle(ctx, store, agentMemoryWakeOptionsFromInference(req))
	if err != nil {
		return nil, nil, err
	}
	return session, toInferenceAgentMemoryWakeResult(report), nil
}

// WakeAgentMemory restores this session from a durable indexed KV prefix.
func (s *ModelSession) WakeAgentMemory(ctx context.Context, store memvid.Store, opts agent.WakeOptions) (*agent.WakeReport, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if s == nil || s.session == nil {
		return nil, core.NewError("mlx: model session is nil")
	}
	plan, err := agent.PlanWake(ctx, store, opts, modelInfoToMemory(s.info))
	if err != nil {
		return nil, err
	}
	if restorer, ok := s.session.(nativeSessionKVBlockRestorer); ok {
		source, err := metalKVSnapshotBlockSource(ctx, store, plan.Bundle, plan.Entry.PrefixTokens())
		if err != nil {
			return nil, err
		}
		if err := restorer.RestoreKVBlocks(ctx, source); err != nil {
			return nil, err
		}
		s.agentMemory = agent.CloneWakeReport(plan.Report)
		return plan.Report, nil
	}
	snapshot, err := kv.LoadPrefixFromMemvidBlocksWithOptions(ctx, store, plan.Bundle, plan.Entry.PrefixTokens(), opts.LoadOptions)
	if err != nil {
		return nil, err
	}
	if err := s.RestoreKV(snapshot); err != nil {
		return nil, err
	}
	s.agentMemory = agent.CloneWakeReport(plan.Report)
	return plan.Report, nil
}

// Wake is a lifecycle alias for WakeAgentMemory.
func (s *ModelSession) Wake(ctx context.Context, store memvid.Store, opts agent.WakeOptions) (*agent.WakeReport, error) {
	return s.WakeAgentMemory(ctx, store, opts)
}

// WakeState implements the backend-neutral go-inference agent-memory contract.
func (s *ModelSession) WakeState(ctx context.Context, req inference.AgentMemoryWakeRequest) (*inference.AgentMemoryWakeResult, error) {
	store, ok := req.Store.(memvid.Store)
	if !ok {
		return nil, core.NewError("mlx: inference agent memory wake requires memvid.Store")
	}
	report, err := s.WakeAgentMemory(ctx, store, agentMemoryWakeOptionsFromInference(req))
	if err != nil {
		return nil, err
	}
	return toInferenceAgentMemoryWakeResult(report), nil
}

// SleepAgentMemory streams this session's current KV state to memvid blocks,
// then writes a bundle manifest and one-entry wake index.
func (s *ModelSession) SleepAgentMemory(ctx context.Context, store memvid.Writer, opts agent.SleepOptions) (*agent.SleepReport, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if s == nil || s.session == nil {
		return nil, core.NewError("mlx: model session is nil")
	}
	if store == nil {
		return nil, core.NewError("mlx: memvid store is nil")
	}
	entryURI, bundleURI, indexURI, err := agent.SleepURIs(opts)
	if err != nil {
		return nil, err
	}
	if opts.ModelInfo.Architecture == "" {
		opts.ModelInfo = modelInfoToMemory(s.info)
	}
	if opts.ParentEntryURI == "" && s.agentMemory != nil {
		opts.ParentEntryURI = s.agentMemory.EntryURI
	}
	if opts.ParentBundleURI == "" && s.agentMemory != nil {
		opts.ParentBundleURI = s.agentMemory.BundleURI
	}
	if opts.ParentIndexURI == "" && s.agentMemory != nil {
		opts.ParentIndexURI = s.agentMemory.IndexURI
	}
	blockOpts := agent.SleepBlockOptions(opts, bundleURI)
	if opts.ReuseParentPrefix && blockOpts.ReusePrefix == nil {
		readStore, ok := store.(memvid.Store)
		if !ok {
			return nil, core.NewError("mlx: agent memory parent-prefix reuse requires a readable memvid store")
		}
		parentBundle, err := kv.LoadMemvidBlockBundle(ctx, readStore, opts.ParentBundleURI)
		if err != nil {
			return nil, err
		}
		blockOpts.ReusePrefix = parentBundle
		if blockOpts.ReusePrefixTokens <= 0 {
			blockOpts.ReusePrefixTokens = parentBundle.TokenCount
		}
	}
	bundle, err := s.SaveKVBlocksToMemvid(ctx, store, blockOpts)
	if err != nil {
		return nil, err
	}
	bundleRef, err := kv.SaveMemvidBlockBundle(ctx, store, bundle, bundleURI)
	if err != nil {
		return nil, err
	}
	index, err := agent.NewSleepIndex(bundle, opts, entryURI, bundleURI)
	if err != nil {
		return nil, err
	}
	indexRef, err := agent.SaveMemvidIndex(ctx, store, index, indexURI)
	if err != nil {
		return nil, err
	}
	report := agent.NewSleepReport(index, bundle, opts, entryURI, bundleURI, indexURI, bundleRef, indexRef)
	s.agentMemory = agent.WakeReportFromSleep(report)
	return report, nil
}

// Sleep is a lifecycle alias for SleepAgentMemory.
func (s *ModelSession) Sleep(ctx context.Context, store memvid.Writer, opts agent.SleepOptions) (*agent.SleepReport, error) {
	return s.SleepAgentMemory(ctx, store, opts)
}

// SleepState implements the backend-neutral go-inference agent-memory contract.
func (s *ModelSession) SleepState(ctx context.Context, req inference.AgentMemorySleepRequest) (*inference.AgentMemorySleepResult, error) {
	store, ok := req.Store.(memvid.Writer)
	if !ok {
		return nil, core.NewError("mlx: inference agent memory sleep requires memvid.Writer")
	}
	report, err := s.SleepAgentMemory(ctx, store, agentMemorySleepOptionsFromInference(req))
	if err != nil {
		return nil, err
	}
	return toInferenceAgentMemorySleepResult(report), nil
}

// AppendAndSleepAgentMemory appends new prompt material and then streams the
// resulting state to durable storage without forcing a generation/reply step.
func (s *ModelSession) AppendAndSleepAgentMemory(ctx context.Context, prompt string, store memvid.Writer, opts agent.SleepOptions) (*agent.SleepReport, error) {
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
func (s *ModelSession) AppendAndSleep(ctx context.Context, prompt string, store memvid.Writer, opts agent.SleepOptions) (*agent.SleepReport, error) {
	return s.AppendAndSleepAgentMemory(ctx, prompt, store, opts)
}

// GenerateAndSleepAgentMemory generates an answer from the current retained
// state and streams the post-answer KV state to durable storage.
func (s *ModelSession) GenerateAndSleepAgentMemory(ctx context.Context, store memvid.Writer, opts agent.SleepOptions, generateOpts ...GenerateOption) (string, *agent.SleepReport, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return "", nil, err
	}
	if s == nil || s.session == nil {
		return "", nil, core.NewError("mlx: model session is nil")
	}
	builder := core.NewBuilder()
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
func (s *ModelSession) GenerateAndSleep(ctx context.Context, store memvid.Writer, opts agent.SleepOptions, generateOpts ...GenerateOption) (string, *agent.SleepReport, error) {
	return s.GenerateAndSleepAgentMemory(ctx, store, opts, generateOpts...)
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
		BlockOptions: kv.MemvidBlockOptions{
			BlockSize:  req.BlockSize,
			KVEncoding: kv.Encoding(req.Encoding),
		},
		Labels: agentMemoryLabelsFromInference(req.Labels),
		Meta:   cloneStringMap(req.Metadata),
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
		Bundle:       agentMemoryStateRef(report.BundleURI, kv.MemvidBlockBundleKind, report.SnapshotHash, ""),
		Index:        agentMemoryStateRef(report.IndexURI, agent.MemvidIndexKind, report.IndexHash, ""),
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
		Bundle:        agentMemoryStateRef(report.BundleURI, kv.MemvidBlockBundleKind, report.SnapshotHash, string(report.KVEncoding)),
		Index:         agentMemoryStateRef(report.IndexURI, agent.MemvidIndexKind, report.IndexHash, ""),
		TokenCount:    report.TokenCount,
		BlockSize:     report.BlockSize,
		BlocksWritten: report.BlocksWritten,
		BlocksReused:  report.BlocksReused,
		Encoding:      string(report.KVEncoding),
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
	for key, value := range labels {
		if value == "" {
			out = append(out, key)
			continue
		}
		out = append(out, key+"="+value)
	}
	core.SliceSort(out)
	return out
}
