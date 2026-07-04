// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"iter"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/kv"
	"dappco.re/go/mlx/kvconv"
	"dappco.re/go/mlx/pkg/metal"
	"dappco.re/go/mlx/spine"
)

// metal_session_adapter.go: wrap a cgo metal.SessionHandle as the engine-
// neutral inference.SessionHandle the lifted session package
// (dappco.re/go/inference/state/session) consumes. It converts between metal's
// pkg/metal snapshot/config/token types and the inference/kv contracts
// INTERNALLY via kvconv — kvconv survives only here, on the metal lane, and
// dies with pkg/metal (docs/engine-merge.md). The native engine needs no such
// adapter: nativeTextSession speaks inference types directly.

var (
	errMetalSessionNilFork          = core.NewError("mlx: metal session fork returned nil handle")
	errMetalSessionNoChunkPrefill   = core.NewError("mlx: metal session does not support chunk prefill")
	errMetalSessionNoTokenPrefill   = core.NewError("mlx: metal session does not support token prefill")
	errMetalSessionNoChunkAppend    = core.NewError("mlx: metal session does not support chunk append")
	errMetalSessionNoTokenAppend    = core.NewError("mlx: metal session does not support token append")
	errMetalSessionNoKVRestore      = core.NewError("mlx: metal session does not support KV restore")
	errMetalSessionNoKVBlockRestore = core.NewError("mlx: metal session does not support KV block restore")
)

// metalSessionAdapter is the go-mlx-side wrapper that satisfies
// inference.SessionHandle over a pkg/metal handle.
type metalSessionAdapter struct {
	handle metal.SessionHandle
}

var _ inference.SessionHandle = (*metalSessionAdapter)(nil)

// newMetalSessionAdapter wraps handle, or returns nil for a nil handle so the
// root Model.NewSession nil-guard still fires.
func newMetalSessionAdapter(handle metal.SessionHandle) *metalSessionAdapter {
	if handle == nil {
		return nil
	}
	return &metalSessionAdapter{handle: handle}
}

// --- inference.SessionHandle core surface ---

func (a *metalSessionAdapter) Prefill(ctx context.Context, prompt string) error {
	return a.handle.Prefill(ctx, prompt)
}

func (a *metalSessionAdapter) AppendPrompt(ctx context.Context, prompt string) error {
	return a.handle.AppendPrompt(ctx, prompt)
}

func (a *metalSessionAdapter) Generate(ctx context.Context, cfg inference.GenerateConfig) iter.Seq[inference.Token] {
	metalCfg := inferenceGenerateConfigToMetal(cfg)
	// inferenceGenerateConfigToMetal (the shared serve converter) carries the
	// sampling knobs; the session lane additionally forwards the trace +
	// cache-hygiene knobs and bridges probe telemetry exactly as the retired
	// spine session path did (spine.ToMetalProbeSink wraps the neutral
	// probe.Sink for the metal engine; nil in, nil out).
	metalCfg.TraceTokenPhases = cfg.TraceTokenPhases
	metalCfg.TraceTokenText = cfg.TraceTokenText
	metalCfg.ClearCache = cfg.GenerationClearCache
	metalCfg.ClearCacheInterval = cfg.GenerationClearCacheInterval
	metalCfg.ProbeSink = spine.ToMetalProbeSink(cfg.ProbeSink)
	return func(yield func(inference.Token) bool) {
		for tok := range a.handle.Generate(ctx, metalCfg) {
			if !yield(inference.Token{ID: tok.ID, Text: tok.Text}) {
				return
			}
		}
	}
}

func (a *metalSessionAdapter) CaptureKV(ctx context.Context) (*kv.Snapshot, error) {
	snapshot, err := a.handle.CaptureKV(ctx)
	if err != nil {
		return nil, err
	}
	return kvconv.ToRootKVSnapshot(snapshot), nil
}

func (a *metalSessionAdapter) RangeKVBlocks(ctx context.Context, blockSize int, opts kv.CaptureOptions, yield func(kv.Block) (bool, error)) error {
	return a.handle.RangeKVBlocks(ctx, blockSize, kvconv.ToMetalKVSnapshotCaptureOptions(opts), func(block metal.KVSnapshotBlock) (bool, error) {
		return yield(kv.Block{
			Index:      block.Index,
			TokenStart: block.TokenStart,
			TokenCount: block.TokenCount,
			Snapshot:   kvconv.ToRootKVSnapshot(block.Snapshot),
		})
	})
}

func (a *metalSessionAdapter) Fork(ctx context.Context) (inference.SessionHandle, error) {
	forked, err := a.handle.Fork(ctx)
	if err != nil {
		return nil, err
	}
	if forked == nil {
		return nil, errMetalSessionNilFork
	}
	return newMetalSessionAdapter(forked), nil
}

func (a *metalSessionAdapter) Reset()       { a.handle.Reset() }
func (a *metalSessionAdapter) Close() error { return a.handle.Close() }
func (a *metalSessionAdapter) Err() error   { return a.handle.Err() }

// --- optional capability probes (metal.ModelSession implements all) ---

func (a *metalSessionAdapter) PrefillChunks(ctx context.Context, chunks iter.Seq[string]) error {
	prefiller, ok := a.handle.(interface {
		PrefillChunks(context.Context, iter.Seq[string]) error
	})
	if !ok {
		return errMetalSessionNoChunkPrefill
	}
	return prefiller.PrefillChunks(ctx, chunks)
}

func (a *metalSessionAdapter) PrefillTokens(ctx context.Context, tokens []int32) error {
	prefiller, ok := a.handle.(interface {
		PrefillTokens(context.Context, []int32) error
	})
	if !ok {
		return errMetalSessionNoTokenPrefill
	}
	return prefiller.PrefillTokens(ctx, tokens)
}

func (a *metalSessionAdapter) AppendPromptChunks(ctx context.Context, chunks iter.Seq[string]) error {
	appender, ok := a.handle.(interface {
		AppendPromptChunks(context.Context, iter.Seq[string]) error
	})
	if !ok {
		return errMetalSessionNoChunkAppend
	}
	return appender.AppendPromptChunks(ctx, chunks)
}

func (a *metalSessionAdapter) AppendTokens(ctx context.Context, tokens []int32) error {
	appender, ok := a.handle.(interface {
		AppendTokens(context.Context, []int32) error
	})
	if !ok {
		return errMetalSessionNoTokenAppend
	}
	return appender.AppendTokens(ctx, tokens)
}

func (a *metalSessionAdapter) CaptureKVWithOptions(ctx context.Context, opts kv.CaptureOptions) (*kv.Snapshot, error) {
	snapshotter, ok := a.handle.(interface {
		CaptureKVWithOptions(context.Context, metal.KVSnapshotCaptureOptions) (*metal.KVSnapshot, error)
	})
	if !ok {
		return a.CaptureKV(ctx)
	}
	snapshot, err := snapshotter.CaptureKVWithOptions(ctx, kvconv.ToMetalKVSnapshotCaptureOptions(opts))
	if err != nil {
		return nil, err
	}
	return kvconv.ToRootKVSnapshot(snapshot), nil
}

func (a *metalSessionAdapter) RestoreKV(ctx context.Context, snapshot *kv.Snapshot) error {
	restorer, ok := a.handle.(interface {
		RestoreKV(context.Context, *metal.KVSnapshot) error
	})
	if !ok {
		return errMetalSessionNoKVRestore
	}
	return restorer.RestoreKV(ctx, kvconv.ToMetalKVSnapshot(snapshot))
}

func (a *metalSessionAdapter) RestoreKVBlocks(ctx context.Context, source kv.BlockSource) error {
	restorer, ok := a.handle.(interface {
		RestoreKVBlocks(context.Context, metal.KVSnapshotBlockSource) error
	})
	if !ok {
		return errMetalSessionNoKVBlockRestore
	}
	return restorer.RestoreKVBlocks(ctx, metalKVBlockSourceFromRoot(source))
}

// metalKVBlockSourceFromRoot rewraps an engine-neutral kv.BlockSource as the
// metal-typed source metal.ModelSession.RestoreKVBlocks consumes, converting
// each streamed block via kvconv on demand.
func metalKVBlockSourceFromRoot(source kv.BlockSource) metal.KVSnapshotBlockSource {
	out := metal.KVSnapshotBlockSource{
		TokenCount:   source.TokenCount,
		PrefixTokens: source.PrefixTokens,
		BlockCount:   source.BlockCount,
	}
	if source.Load != nil {
		out.Load = func(ctx context.Context, index int) (metal.KVSnapshotBlock, error) {
			block, err := source.Load(ctx, index)
			if err != nil {
				return metal.KVSnapshotBlock{}, err
			}
			return metal.KVSnapshotBlock{
				Index:      block.Index,
				TokenStart: block.TokenStart,
				TokenCount: block.TokenCount,
				Snapshot:   kvconv.ToMetalKVSnapshot(block.Snapshot),
			}, nil
		}
	}
	return out
}
