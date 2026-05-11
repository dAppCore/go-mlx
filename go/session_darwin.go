// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && !nomlx

package mlx

import (
	"context"

	core "dappco.re/go"
	memvid "dappco.re/go/inference/state"
	"dappco.re/go/mlx/internal/metal"
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

// ModelSession is a persistent model-state handle with retained KV cache.
type ModelSession struct {
	session     metal.SessionHandle
	info        ModelInfo
	agentMemory *AgentMemoryWakeReport
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
	return &ModelSession{session: session, info: m.Info()}, nil
}

// NewSessionFromKV creates a persistent session restored from a KV snapshot.
func (m *Model) NewSessionFromKV(snapshot *KVSnapshot) (*ModelSession, error) {
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
func (m *Model) NewSessionFromBundle(bundle *StateBundle) (*ModelSession, error) {
	if bundle == nil {
		return nil, core.NewError("mlx: state bundle is nil")
	}
	if err := CheckStateBundleCompatibility(m.Info(), bundle); err != nil {
		return nil, err
	}
	snapshot, err := bundle.Snapshot()
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

// AppendPrompt appends prompt tokens to the retained session KV state without
// replaying the existing prefix.
func (s *ModelSession) AppendPrompt(prompt string) error {
	if s == nil || s.session == nil {
		return core.NewError("mlx: model session is nil")
	}
	return s.session.AppendPrompt(context.Background(), prompt)
}

// Generate produces a buffered string from the retained session state.
func (s *ModelSession) Generate(opts ...GenerateOption) (string, error) {
	if s == nil || s.session == nil {
		return "", core.NewError("mlx: model session is nil")
	}
	builder := core.NewBuilder()
	for tok := range s.session.Generate(context.Background(), toMetalGenerateConfig(applyGenerateOptions(opts))) {
		builder.WriteString(tok.Text)
	}
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
		cfg := toMetalGenerateConfig(applyGenerateOptions(opts))
		for tok := range s.session.Generate(ctx, cfg) {
			if ctx.Err() != nil {
				return
			}
			select {
			case out <- toRootToken(tok):
			case <-ctx.Done():
				return
			}
		}
	}()
	return out
}

// CaptureKV copies the current retained KV cache tensors to CPU memory.
func (s *ModelSession) CaptureKV() (*KVSnapshot, error) {
	return s.CaptureKVWithOptions(KVSnapshotCaptureOptions{})
}

// CaptureKVWithOptions copies the current retained KV cache tensors to CPU
// memory with explicit capture options.
func (s *ModelSession) CaptureKVWithOptions(opts KVSnapshotCaptureOptions) (*KVSnapshot, error) {
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
		dropKVSnapshotFloat32(root)
	}
	return root, nil
}

// AnalyzeKV captures and analyses the current retained KV state.
func (s *ModelSession) AnalyzeKV() (*KVAnalysis, error) {
	snapshot, err := s.CaptureKV()
	if err != nil {
		return nil, err
	}
	return AnalyzeKV(snapshot), nil
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
func (s *ModelSession) RestoreKV(snapshot *KVSnapshot) error {
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
	snapshot, err := LoadKVSnapshot(path)
	if err != nil {
		return err
	}
	return s.RestoreKV(snapshot)
}

// SaveKVToMemvid captures and writes the current retained KV state to memvid.
func (s *ModelSession) SaveKVToMemvid(ctx context.Context, store memvid.Writer, opts KVSnapshotMemvidOptions) (memvid.ChunkRef, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	captureOpts := KVSnapshotCaptureOptions{}
	if opts.KVEncoding == KVSnapshotEncodingNative {
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
	snapshot, err := LoadKVSnapshotFromMemvid(ctx, store, ref)
	if err != nil {
		return err
	}
	return s.RestoreKV(snapshot)
}

// SaveKVBlocksToMemvid captures retained KV state and writes per-block KV chunks.
func (s *ModelSession) SaveKVBlocksToMemvid(ctx context.Context, store memvid.Writer, opts KVSnapshotMemvidBlockOptions) (*KVSnapshotMemvidBlockBundle, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if s == nil || s.session == nil {
		return nil, core.NewError("mlx: model session is nil")
	}
	captureOpts := KVSnapshotCaptureOptions{}
	if opts.KVEncoding == KVSnapshotEncodingNative {
		captureOpts.RawKVOnly = true
	}
	blockSize := opts.BlockSize
	if blockSize <= 0 {
		blockSize = DefaultCacheBlockSize
	}
	return SaveMemvidBlocksFromStream(ctx, store, opts, func(yield func(KVSnapshotBlock) (bool, error)) error {
		return s.session.RangeKVBlocks(ctx, blockSize, toMetalKVSnapshotCaptureOptions(captureOpts), func(block metal.KVSnapshotBlock) (bool, error) {
			return yield(KVSnapshotBlock{
				Index:      block.Index,
				TokenStart: block.TokenStart,
				TokenCount: block.TokenCount,
				Snapshot:   toRootKVSnapshot(block.Snapshot),
			})
		})
	})
}

// LoadKVBlocksFromMemvid restores retained session state from per-block KV chunks.
func (s *ModelSession) LoadKVBlocksFromMemvid(ctx context.Context, store memvid.Store, bundle *KVSnapshotMemvidBlockBundle) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if s == nil || s.session == nil {
		return core.NewError("mlx: model session is nil")
	}
	if bundle == nil {
		return core.NewError("mlx: memvid KV block bundle is nil")
	}
	if restorer, ok := s.session.(nativeSessionKVBlockRestorer); ok {
		source, err := metalKVSnapshotBlockSource(ctx, store, bundle, bundle.TokenCount)
		if err != nil {
			return err
		}
		if err := restorer.RestoreKVBlocks(ctx, source); err != nil {
			return err
		}
		s.agentMemory = nil
		return nil
	}
	snapshot, err := LoadKVSnapshotFromMemvidBlocks(ctx, store, bundle)
	if err != nil {
		return err
	}
	return s.RestoreKV(snapshot)
}

// RestoreBundle restores the session from a state bundle.
func (s *ModelSession) RestoreBundle(bundle *StateBundle) error {
	if bundle == nil {
		return core.NewError("mlx: state bundle is nil")
	}
	if err := CheckStateBundleCompatibility(s.info, bundle); err != nil {
		return err
	}
	snapshot, err := bundle.Snapshot()
	if err != nil {
		return err
	}
	return s.RestoreKV(snapshot)
}

// RestoreBundleFromMemvid restores the session from a state bundle whose KV is
// held in memvid cold storage.
func (s *ModelSession) RestoreBundleFromMemvid(ctx context.Context, bundle *StateBundle, store memvid.Store) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if bundle == nil {
		return core.NewError("mlx: state bundle is nil")
	}
	if err := CheckStateBundleCompatibility(s.info, bundle); err != nil {
		return err
	}
	snapshot, err := bundle.SnapshotFromMemvid(ctx, store)
	if err != nil {
		return err
	}
	return s.RestoreKV(snapshot)
}

// LoadBundle reads a state bundle from path and restores it into the session.
func (s *ModelSession) LoadBundle(path string) error {
	bundle, err := LoadStateBundle(path)
	if err != nil {
		return err
	}
	return s.RestoreBundle(bundle)
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
	return &ModelSession{session: forked, info: s.info, agentMemory: cloneAgentMemoryWakeReport(s.agentMemory)}, nil
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
