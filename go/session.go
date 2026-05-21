// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"iter"

	"dappco.re/go/mlx/blockcache"

	core "dappco.re/go"
	"dappco.re/go/inference/parser"
	memvid "dappco.re/go/inference/state"
	"dappco.re/go/mlx/agent"
	"dappco.re/go/mlx/bundle"
	"dappco.re/go/mlx/internal/metal"
	"dappco.re/go/mlx/kv"
)

type nativeModelSessionFactory interface {
	NewSession() metal.SessionHandle
}

type nativeSessionRestorer interface {
	RestoreKV(context.Context, *metal.KVSnapshot) error
}

type nativeSessionKVBlockRestorer interface {
	RestoreKVBlocks(context.Context, metal.KVSnapshotBlockSource) error
}

type nativeSessionKVSnapshotterWithOptions interface {
	CaptureKVWithOptions(context.Context, metal.KVSnapshotCaptureOptions) (*metal.KVSnapshot, error)
}

type nativeSessionChunkPrefiller interface {
	PrefillChunks(context.Context, iter.Seq[string]) error
}

type nativeSessionChunkAppender interface {
	AppendPromptChunks(context.Context, iter.Seq[string]) error
}

type nativeSessionTokenPrefiller interface {
	PrefillTokens(context.Context, []int32) error
}

type nativeSessionTokenAppender interface {
	AppendTokens(context.Context, []int32) error
}

// ModelSession is a persistent model-state handle with retained KV cache.
type ModelSession struct {
	session     metal.SessionHandle
	info        ModelInfo
	tok         *Tokenizer
	agentMemory *agent.WakeReport
}

// NewSession creates a persistent session for prefill, generation, KV capture, and forking.
func (m *Model) NewSession() (*ModelSession, error) {
	if m == nil || m.model == nil {
		return nil, core.NewError("mlx: model is nil")
	}
	factory, ok := m.model.(nativeModelSessionFactory)
	if !ok {
		return nil, core.NewError("mlx: native model does not support sessions")
	}
	session := factory.NewSession()
	if session == nil {
		return nil, core.NewError("mlx: native model returned nil session")
	}
	return &ModelSession{session: session, info: m.Info(), tok: m.Tokenizer()}, nil
}

// NewSessionFromKV creates a persistent session restored from a KV snapshot.
func (m *Model) NewSessionFromKV(snapshot *kv.Snapshot) (*ModelSession, error) {
	session, err := m.NewSession()
	if err != nil {
		return nil, err
	}
	if err := session.RestoreKV(snapshot); err != nil {
		if closeErr := session.Close(); closeErr != nil {
			return nil, core.ErrorJoin(err, closeErr)
		}
		return nil, err
	}
	return session, nil
}

// NewSessionFromBundle creates a persistent session restored from a state bundle.
func (m *Model) NewSessionFromBundle(b *bundle.Bundle) (*ModelSession, error) {
	if b == nil {
		return nil, core.NewError("mlx: state bundle is nil")
	}
	if err := bundle.CheckCompatibility(modelInfoToBundle(m.Info()), b); err != nil {
		return nil, err
	}
	snapshot, err := b.Snapshot()
	if err != nil {
		return nil, err
	}
	return m.NewSessionFromKV(snapshot)
}

// Prefill loads prompt into the retained session KV state.
func (s *ModelSession) Prefill(prompt string) error {
	if s == nil || s.session == nil {
		return core.NewError("mlx: model session is nil")
	}
	return s.session.Prefill(context.Background(), prompt)
}

// PrefillChunks loads bounded prompt chunks into the retained session KV state.
func (s *ModelSession) PrefillChunks(ctx context.Context, chunks iter.Seq[string]) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if s == nil || s.session == nil {
		return core.NewError("mlx: model session is nil")
	}
	if prefiller, ok := s.session.(nativeSessionChunkPrefiller); ok {
		return prefiller.PrefillChunks(ctx, chunks)
	}
	return s.Prefill(promptChunksToString(chunks))
}

// PrefillTokens loads model-native token IDs into the retained session KV state.
func (s *ModelSession) PrefillTokens(ctx context.Context, tokens []int32) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if s == nil || s.session == nil {
		return core.NewError("mlx: model session is nil")
	}
	if prefiller, ok := s.session.(nativeSessionTokenPrefiller); ok {
		return prefiller.PrefillTokens(ctx, append([]int32(nil), tokens...))
	}
	return core.NewError("mlx: native model session does not support token prefill")
}

// AppendPrompt appends prompt tokens to the retained session KV state without
// replaying the existing prefix.
func (s *ModelSession) AppendPrompt(prompt string) error {
	if s == nil || s.session == nil {
		return core.NewError("mlx: model session is nil")
	}
	return s.session.AppendPrompt(context.Background(), prompt)
}

// AppendPromptChunks appends bounded prompt chunks to the retained session KV
// state without replaying the existing prefix.
func (s *ModelSession) AppendPromptChunks(ctx context.Context, chunks iter.Seq[string]) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if s == nil || s.session == nil {
		return core.NewError("mlx: model session is nil")
	}
	if appender, ok := s.session.(nativeSessionChunkAppender); ok {
		return appender.AppendPromptChunks(ctx, chunks)
	}
	return s.AppendPrompt(promptChunksToString(chunks))
}

// AppendTokens appends model-native token IDs to the retained session KV state
// without replaying the existing prefix.
func (s *ModelSession) AppendTokens(ctx context.Context, tokens []int32) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if s == nil || s.session == nil {
		return core.NewError("mlx: model session is nil")
	}
	if appender, ok := s.session.(nativeSessionTokenAppender); ok {
		return appender.AppendTokens(ctx, append([]int32(nil), tokens...))
	}
	return core.NewError("mlx: native model session does not support token append")
}

// Generate produces a buffered string from the retained session state.
func (s *ModelSession) Generate(opts ...GenerateOption) (string, error) {
	if s == nil || s.session == nil {
		return "", core.NewError("mlx: model session is nil")
	}
	cfg := applyGenerateOptions(opts)
	filter := parser.NewProcessor(cfg.Thinking, parserHint(s.info))
	builder := core.NewBuilder()
	for tok := range s.session.Generate(context.Background(), toMetalGenerateConfig(cfg)) {
		builder.WriteString(filter.Process(sessionParserTokenText(s.tok, tok)))
	}
	builder.WriteString(filter.Flush())
	if err := s.session.Err(); err != nil {
		return "", err
	}
	return builder.String(), nil
}

// GenerateStream streams tokens from the retained session state.
func (s *ModelSession) GenerateStream(ctx context.Context, opts ...GenerateOption) <-chan Token {
	out := make(chan Token)
	go func() {
		defer close(out)
		if s == nil || s.session == nil {
			return
		}
		if ctx == nil {
			ctx = context.Background()
		}
		cfg := applyGenerateOptions(opts)
		filter := parser.NewProcessor(cfg.Thinking, parserHint(s.info))
		for tok := range s.session.Generate(ctx, toMetalGenerateConfig(cfg)) {
			if ctx.Err() != nil {
				return
			}
			text := filter.Process(sessionParserTokenText(s.tok, tok))
			if text == "" {
				continue
			}
			select {
			case out <- Token{ID: tok.ID, Value: text, Text: text}:
			case <-ctx.Done():
				return
			}
		}
		if text := filter.Flush(); text != "" {
			select {
			case out <- Token{Value: text, Text: text}:
			case <-ctx.Done():
				return
			}
		}
	}()
	return out
}

func sessionParserTokenText(tok *Tokenizer, token metal.Token) string {
	if tok != nil {
		if text := tok.IDToken(token.ID); sessionParserControlToken(text) {
			return text
		}
	}
	return token.Text
}

func sessionParserControlToken(text string) bool {
	if text == "" {
		return false
	}
	return core.Contains(text, "<|channel>") ||
		core.Contains(text, "<channel|>") ||
		core.Contains(text, "<start_of_turn>") ||
		core.Contains(text, "<end_of_turn>") ||
		core.Contains(text, "<think>") ||
		core.Contains(text, "</think>") ||
		core.Contains(text, "<thinking>") ||
		core.Contains(text, "</thinking>") ||
		core.Contains(text, "<thought>") ||
		core.Contains(text, "</thought>") ||
		core.Contains(text, "<reasoning>") ||
		core.Contains(text, "</reasoning>") ||
		core.Contains(text, "<analysis>") ||
		core.Contains(text, "</analysis>")
}

// CaptureKV copies the current retained KV cache tensors to CPU memory.
func (s *ModelSession) CaptureKV() (*kv.Snapshot, error) {
	return s.CaptureKVWithOptions(kv.CaptureOptions{})
}

// CaptureKVWithOptions copies the current retained KV cache tensors to CPU
// memory with explicit capture options.
func (s *ModelSession) CaptureKVWithOptions(opts kv.CaptureOptions) (*kv.Snapshot, error) {
	if s == nil || s.session == nil {
		return nil, core.NewError("mlx: model session is nil")
	}
	var (
		snapshot *metal.KVSnapshot
		err      error
	)
	if snapshotter, ok := s.session.(nativeSessionKVSnapshotterWithOptions); ok {
		snapshot, err = snapshotter.CaptureKVWithOptions(context.Background(), toMetalKVSnapshotCaptureOptions(opts))
	} else {
		snapshot, err = s.session.CaptureKV(context.Background())
	}
	if err != nil {
		return nil, err
	}
	root := toRootKVSnapshot(snapshot)
	if opts.RawKVOnly {
		kv.DropFloat32(root)
	}
	return root, nil
}

// kv.Analyze captures and analyses the current retained KV state.
func (s *ModelSession) AnalyzeKV() (*kv.Analysis, error) {
	snapshot, err := s.CaptureKV()
	if err != nil {
		return nil, err
	}
	return kv.Analyze(snapshot), nil
}

// SaveKV captures and writes the current retained KV state to path.
func (s *ModelSession) SaveKV(path string) error {
	snapshot, err := s.CaptureKV()
	if err != nil {
		return err
	}
	return snapshot.Save(path)
}

// RestoreKV replaces the retained session state with a restorable KV snapshot.
func (s *ModelSession) RestoreKV(snapshot *kv.Snapshot) error {
	if s == nil || s.session == nil {
		return core.NewError("mlx: model session is nil")
	}
	if snapshot == nil {
		return core.NewError("mlx: KV snapshot is nil")
	}
	restorer, ok := s.session.(nativeSessionRestorer)
	if !ok {
		return core.NewError("mlx: native model session does not support KV restore")
	}
	if err := restorer.RestoreKV(context.Background(), toMetalKVSnapshot(snapshot)); err != nil {
		return err
	}
	s.agentMemory = nil
	return nil
}

// LoadKV reads a KV snapshot from path and restores it into the session.
func (s *ModelSession) LoadKV(path string) error {
	snapshot, err := kv.Load(path)
	if err != nil {
		return err
	}
	return s.RestoreKV(snapshot)
}

// SaveKVToMemvid captures and writes the current retained KV state to memvid.
func (s *ModelSession) SaveKVToMemvid(ctx context.Context, store memvid.Writer, opts kv.MemvidOptions) (memvid.ChunkRef, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	captureOpts := kv.CaptureOptions{}
	if opts.KVEncoding == kv.EncodingNative {
		captureOpts.RawKVOnly = true
	}
	snapshot, err := s.CaptureKVWithOptions(captureOpts)
	if err != nil {
		return memvid.ChunkRef{}, err
	}
	return snapshot.SaveMemvid(ctx, store, opts)
}

// LoadKVFromMemvid restores retained session state from a memvid KV snapshot.
func (s *ModelSession) LoadKVFromMemvid(ctx context.Context, store memvid.Store, ref memvid.ChunkRef) error {
	if ctx == nil {
		ctx = context.Background()
	}
	snapshot, err := kv.LoadFromMemvid(ctx, store, ref)
	if err != nil {
		return err
	}
	return s.RestoreKV(snapshot)
}

// SaveKVBlocksToState captures retained KV state and writes per-block State
// chunks.
func (s *ModelSession) SaveKVBlocksToState(ctx context.Context, store memvid.Writer, opts kv.StateBlockOptions) (*kv.StateBlockBundle, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if s == nil || s.session == nil {
		return nil, core.NewError("mlx: model session is nil")
	}
	captureOpts := kv.CaptureOptions{}
	if opts.KVEncoding == kv.EncodingNative {
		captureOpts.RawKVOnly = true
	}
	blockSize := opts.BlockSize
	if blockSize <= 0 {
		blockSize = blockcache.DefaultBlockSize
	}
	return kv.SaveStateBlocksFromStream(ctx, store, opts, func(yield func(kv.Block) (bool, error)) error {
		return s.session.RangeKVBlocks(ctx, blockSize, toMetalKVSnapshotCaptureOptions(captureOpts), func(block metal.KVSnapshotBlock) (bool, error) {
			return yield(kv.Block{
				Index:      block.Index,
				TokenStart: block.TokenStart,
				TokenCount: block.TokenCount,
				Snapshot:   toRootKVSnapshot(block.Snapshot),
			})
		})
	})
}

// SaveKVBlocksToMemvid captures retained KV state and writes per-block KV
// chunks.
//
// Deprecated: use SaveKVBlocksToState.
func (s *ModelSession) SaveKVBlocksToMemvid(ctx context.Context, store memvid.Writer, opts kv.MemvidBlockOptions) (*kv.MemvidBlockBundle, error) {
	return s.SaveKVBlocksToState(ctx, store, opts)
}

// LoadKVBlocksFromState restores retained session state from per-block State
// chunks.
func (s *ModelSession) LoadKVBlocksFromState(ctx context.Context, store memvid.Store, bundle *kv.StateBlockBundle) error {
	return s.LoadKVPrefixBlocksFromState(ctx, store, bundle, 0)
}

// LoadKVBlocksFromMemvid restores retained session state from per-block KV
// chunks.
//
// Deprecated: use LoadKVBlocksFromState.
func (s *ModelSession) LoadKVBlocksFromMemvid(ctx context.Context, store memvid.Store, bundle *kv.MemvidBlockBundle) error {
	return s.LoadKVBlocksFromState(ctx, store, bundle)
}

// LoadKVPrefixBlocksFromState restores a retained session state from the
// State KV blocks needed to cover prefixTokens. Native sessions consume the
// blocks as a stream, avoiding a full CPU-side assembled snapshot.
func (s *ModelSession) LoadKVPrefixBlocksFromState(ctx context.Context, store memvid.Store, bundle *kv.StateBlockBundle, prefixTokens int) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if s == nil || s.session == nil {
		return core.NewError("mlx: model session is nil")
	}
	if bundle == nil {
		return core.NewError("mlx: State KV block bundle is nil")
	}
	if restorer, ok := s.session.(nativeSessionKVBlockRestorer); ok {
		source, err := metalKVSnapshotBlockSource(ctx, store, bundle, prefixTokens)
		if err != nil {
			return err
		}
		if err := restorer.RestoreKVBlocks(ctx, source); err != nil {
			return err
		}
		s.agentMemory = nil
		return nil
	}
	loadOpts := kv.LoadOptions{}
	if bundle.KVEncoding == kv.EncodingNative {
		loadOpts.RawKVOnly = true
	}
	snapshot, err := kv.LoadPrefixFromStateBlocksWithOptions(ctx, store, bundle, prefixTokens, loadOpts)
	if err != nil {
		return err
	}
	return s.RestoreKV(snapshot)
}

// LoadKVPrefixBlocksFromMemvid restores a retained session state from the
// memvid KV blocks needed to cover prefixTokens. Native sessions consume the
// blocks as a stream, avoiding a full CPU-side assembled snapshot.
//
// Deprecated: use LoadKVPrefixBlocksFromState.
func (s *ModelSession) LoadKVPrefixBlocksFromMemvid(ctx context.Context, store memvid.Store, bundle *kv.MemvidBlockBundle, prefixTokens int) error {
	return s.LoadKVPrefixBlocksFromState(ctx, store, bundle, prefixTokens)
}

// RestoreBundle restores the session from a state bundle.
func (s *ModelSession) RestoreBundle(b *bundle.Bundle) error {
	if b == nil {
		return core.NewError("mlx: state bundle is nil")
	}
	if err := bundle.CheckCompatibility(modelInfoToBundle(s.info), b); err != nil {
		return err
	}
	snapshot, err := b.Snapshot()
	if err != nil {
		return err
	}
	return s.RestoreKV(snapshot)
}

// RestoreBundleFromMemvid restores the session from a state bundle whose KV is
// held in memvid cold storage.
func (s *ModelSession) RestoreBundleFromMemvid(ctx context.Context, b *bundle.Bundle, store memvid.Store) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if b == nil {
		return core.NewError("mlx: state bundle is nil")
	}
	if err := bundle.CheckCompatibility(modelInfoToBundle(s.info), b); err != nil {
		return err
	}
	snapshot, err := b.SnapshotFromMemvid(ctx, store)
	if err != nil {
		return err
	}
	return s.RestoreKV(snapshot)
}

// LoadBundle reads a state bundle from path and restores it into the session.
func (s *ModelSession) LoadBundle(path string) error {
	b, err := bundle.Load(path)
	if err != nil {
		return err
	}
	return s.RestoreBundle(b)
}

// Fork creates an independent session that starts from the same retained state.
func (s *ModelSession) Fork() (*ModelSession, error) {
	if s == nil || s.session == nil {
		return nil, core.NewError("mlx: model session is nil")
	}
	forked, err := s.session.Fork(context.Background())
	if err != nil {
		return nil, err
	}
	if forked == nil {
		return nil, core.NewError("mlx: native model returned nil session fork")
	}
	return &ModelSession{session: forked, info: s.info, tok: s.tok, agentMemory: agent.CloneWakeReport(s.agentMemory)}, nil
}

// Reset releases retained state and leaves the session ready for another prefill.
func (s *ModelSession) Reset() {
	if s == nil || s.session == nil {
		return
	}
	s.session.Reset()
	s.agentMemory = nil
}

// Close releases retained session state.
func (s *ModelSession) Close() error {
	if s == nil || s.session == nil {
		return nil
	}
	err := s.session.Close()
	s.session = nil
	return err
}

// Err returns the last session error.
func (s *ModelSession) Err() error {
	if s == nil || s.session == nil {
		return nil
	}
	return s.session.Err()
}
