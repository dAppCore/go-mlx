// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"iter"
	"unsafe"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/parser"
	"dappco.re/go/mlx/gguf"
	"dappco.re/go/mlx/kv"
	"dappco.re/go/mlx/kvconv"
	"dappco.re/go/mlx/lora"
	"dappco.re/go/mlx/pkg/metal"
	"dappco.re/go/mlx/probe"
)

// Compile-time layout guard for the metal.ProbeLogit / probe.Logit
// reinterpret cast in toRootProbeLogits. Both types carry int32 +
// float32 + float64 with the same Go field ordering; the assertions
// below break the build if either struct grows / shrinks / changes
// field order, forcing a manual review of the unsafe cast.
var _ [unsafe.Sizeof(metal.ProbeLogit{}) - unsafe.Sizeof(probe.Logit{})]byte
var _ [unsafe.Sizeof(probe.Logit{}) - unsafe.Sizeof(metal.ProbeLogit{})]byte
var _ [unsafe.Offsetof(metal.ProbeLogit{}.TokenID) - unsafe.Offsetof(probe.Logit{}.TokenID)]byte
var _ [unsafe.Offsetof(metal.ProbeLogit{}.Logit) - unsafe.Offsetof(probe.Logit{}.Logit)]byte
var _ [unsafe.Offsetof(metal.ProbeLogit{}.Probability) - unsafe.Offsetof(probe.Logit{}.Probability)]byte

// Compile-time layout guard for the inference.Message / metal.ChatMessage
// reinterpret cast in chatMessagesAsMetal. Both types are {Role string;
// Content string} with the same field order; the assertions below break
// the build if either struct ever changes.
var _ [unsafe.Sizeof(inference.Message{}) - unsafe.Sizeof(metal.ChatMessage{})]byte
var _ [unsafe.Sizeof(metal.ChatMessage{}) - unsafe.Sizeof(inference.Message{})]byte
var _ [unsafe.Offsetof(inference.Message{}.Role) - unsafe.Offsetof(metal.ChatMessage{}.Role)]byte
var _ [unsafe.Offsetof(inference.Message{}.Content) - unsafe.Offsetof(metal.ChatMessage{}.Content)]byte

// chatMessagesAsMetal reinterprets a []inference.Message as
// []metal.ChatMessage without copying. The compile-time guards above
// pin the layout match — both structs carry {Role string; Content
// string} with the same field order, so a pointer-cast yields a
// valid metal-side slice. The receiving Chat / ChatChunks paths only
// read from the slice (they format the messages into a prompt string
// and return), so the borrow lifetime is bounded by the call. The
// prior pattern allocated a fresh []metal.ChatMessage + per-message
// struct copy on every call — for long histories the slice + copy
// dominated the dispatch cost for Chat / ChatStream / ChatChunksStream.
func chatMessagesAsMetal(messages []inference.Message) []metal.ChatMessage {
	if len(messages) == 0 {
		return nil
	}
	return unsafe.Slice((*metal.ChatMessage)(unsafe.Pointer(&messages[0])), len(messages))
}

// Model is the RFC-style root-package model handle.
type Model struct {
	model       NativeModel
	cfg         LoadConfig
	tok         *Tokenizer
	gguf        *gguf.Info
	adapterInfo lora.AdapterInfo
	cleanup     func() error
	// cachedParserHint is the memoised parser.Hint dispatched into
	// parser.NewProcessor on every Generate / Chat / *Stream entry.
	// LoadModel pre-builds it; the 7 hot-path entries call hintForParser
	// which falls back to a one-time build when callers construct *Model
	// directly (test fixtures, sidecar adapters). Skips the per-call
	// m.model.Info() fan-out that otherwise clones the native
	// AdapterInfo.TargetKeys slice on every dispatch.
	cachedParserHint parser.Hint
	// parserHintBuilt gates the lazy build in hintForParser — set true
	// by refreshParserHint (LoadModel and LoRA mutation surfaces).
	parserHintBuilt bool
}

var loadNativeModel = func(modelPath string, cfg metal.LoadConfig) (NativeModel, error) {
	return metal.LoadAndInit(modelPath, cfg)
}

// Package-level sentinel for the "model is nil" guard that fires from
// every public Model method when the caller passes a zero-value or
// already-Close()d *Model. Sharing one *Err avoids an allocation per
// call on what is almost always a hot path during test fixtures and
// during defensive checks in adapter / sidecar code.
var (
	errMLXModelNil               = core.NewError("mlx: model is nil")
	errMLXKVPromptRestoreUnsupp  = core.NewError("mlx: native model does not support KV prompt cache restore")
	errMLXKVCaptureUnsupp        = core.NewError("mlx: native model does not support KV capture")
	errMLXPromptCacheWarmUnsupp  = core.NewError("mlx: native model does not support prompt cache warming")
	errMLXPromptCacheClearUnsupp = core.NewError("mlx: native model does not support prompt cache clearing")
	errMLXLoRALoadUnsupp         = core.NewError("mlx: native model does not support LoRA loading")
	errMLXLoRAUnloadUnsupp       = core.NewError("mlx: native model does not support LoRA unloading")
)

// closedTokenChan is the shared "no tokens, generation skipped" channel
// returned by every Stream entry when the receiver model is nil. Sharing
// one closed channel avoids both the per-call make(chan Token) and the
// goroutine launch that would otherwise just defer-close.
var closedTokenChan = func() chan Token {
	c := make(chan Token)
	close(c)
	return c
}()

// buildParserHint constructs the parser.Hint from the live native model
// info + cached adapter / gguf metadata. The Hint only needs Architecture
// + Adapter name; everything else m.Info() composes is dead weight on the
// parser path. Called once at LoadModel and again from the LoRA mutation
// surfaces (LoadLoRA / UnloadLoRA / NewLoRA) — the inference hot paths
// then read the cached value direct from m.parserHint without re-entering
// m.model.Info() (which itself clones the native AdapterInfo.TargetKeys
// slice via cloneMetalAdapterInfo).
func (m *Model) buildParserHint() parser.Hint {
	info := m.model.Info()
	architecture := info.Architecture
	if architecture == "" && m.gguf != nil {
		architecture = m.gguf.Architecture
	}
	adapterName := m.adapterInfo.Name
	if adapterName == "" {
		adapterName = info.Adapter.Name
	}
	return parser.Hint{
		Architecture: architecture,
		AdapterName:  adapterName,
	}
}

// refreshParserHint recomputes and stores the cached parser.Hint after a
// mutation that could change either the architecture (gguf reload) or the
// adapter name (LoRA load / unload / re-apply). The 7 Generate / Chat /
// *Stream entry points read the cached value with no further allocation,
// so the cost is paid once at the mutation point instead of per call.
// Safe to call only after m.model is wired (the m.model nil guard up top
// of every entry path runs first); refreshing in that state would panic,
// so callers in the LoRA / Load path are the only valid sites.
func (m *Model) refreshParserHint() {
	m.cachedParserHint = m.buildParserHint()
	m.parserHintBuilt = true
}

// hintForParser returns the cached parser.Hint, building it on first call
// when *Model was constructed directly (test fixtures, in-tree adapters
// bypassing LoadModel). The eager LoadModel path warms the cache so the
// hot-path read on production traffic is a single field load.
func (m *Model) hintForParser() parser.Hint {
	if !m.parserHintBuilt {
		m.refreshParserHint()
	}
	return m.cachedParserHint
}

var readGGUFInfo = gguf.ReadInfo

func appendCleanup(cleanup *func() error, next func() error) {
	if next == nil {
		return
	}
	if *cleanup == nil {
		*cleanup = next
		return
	}
	prev := *cleanup
	*cleanup = func() error {
		return core.ErrorJoin(prev(), next())
	}
}

// runCleanup invokes the optional cleanup closure, returning nil if cleanup
// itself is nil. Lets LoadModel keep a nil cleanup on the common no-Medium
// path without a no-op closure allocation.
func runCleanup(cleanup func() error) error {
	if cleanup == nil {
		return nil
	}
	return cleanup()
}

// LoadModel loads a model directly through go-mlx without going through go-inference.
func LoadModel(modelPath string, opts ...LoadOption) (*Model, error) {
	cfg, err := normalizeLoadConfig(applyLoadOptions(opts))
	if err != nil {
		return nil, err
	}

	resolvedPath := modelPath
	resolvedAdapterPath := cfg.AdapterPath
	var adapterInfo lora.AdapterInfo
	// cleanup stays nil on the common no-Medium path. runCleanup +
	// Close already short on nil, sparing a no-op closure allocation
	// per LoadModel call.
	var cleanup func() error
	if cfg.Medium != nil {
		resolvedPath, cleanup, err = stageModelFromMedium(cfg.Medium, modelPath)
		if err != nil {
			return nil, err
		}
		if cfg.AdapterPath != "" {
			var adapterCleanup func() error
			resolvedAdapterPath, adapterCleanup, err = stagePathFromMedium(cfg.Medium, cfg.AdapterPath)
			if err != nil {
				if cleanupErr := runCleanup(cleanup); cleanupErr != nil {
					return nil, core.ErrorJoin(err, cleanupErr)
				}
				return nil, err
			}
			appendCleanup(&cleanup, adapterCleanup)
		}
	}
	if slice, ok, sliceErr := inspectModelSliceIfPresent(resolvedPath); sliceErr != nil {
		if cleanupErr := runCleanup(cleanup); cleanupErr != nil {
			return nil, core.ErrorJoin(sliceErr, cleanupErr)
		}
		return nil, sliceErr
	} else if ok && slice.RequiresSplitPlacement {
		err := core.NewError("mlx: model slice requires split placement; use LoadSplitExecutor or lthn-mlx slice-smoke -split")
		if cleanupErr := runCleanup(cleanup); cleanupErr != nil {
			return nil, core.ErrorJoin(err, cleanupErr)
		}
		return nil, err
	}
	cfg = applyMemoryPlanToLoadConfig(resolvedPath, cfg)
	if resolvedAdapterPath != "" {
		adapterInfo, err = lora.Inspect(resolvedAdapterPath, cfg.AdapterPath)
		if err != nil {
			if cleanupErr := runCleanup(cleanup); cleanupErr != nil {
				return nil, core.ErrorJoin(err, cleanupErr)
			}
			return nil, err
		}
	}

	native, err := loadNativeModel(resolvedPath, metal.LoadConfig{
		ContextLen:            cfg.ContextLength,
		ParallelSlots:         cfg.ParallelSlots,
		DisablePromptCache:    !cfg.PromptCache,
		PromptCacheMinTokens:  cfg.PromptCacheMinTokens,
		AdapterPath:           resolvedAdapterPath,
		Device:                metal.DeviceType(cfg.Device),
		CachePolicy:           string(cfg.CachePolicy),
		KVCacheMode:           string(cfg.CacheMode),
		KVCacheStorageDType:   cfg.KVCacheStorageDType,
		PagedKVPageSize:       cfg.PagedKVPageSize,
		PagedKVPrealloc:       cfg.PagedKVPrealloc,
		FixedSlidingCacheSize: cfg.FixedSlidingCacheSize,
		BatchSize:             cfg.BatchSize,
		PrefillChunkSize:      cfg.PrefillChunkSize,
		ExpectedQuantization:  cfg.ExpectedQuantization,
		MemoryLimitBytes:      cfg.MemoryLimitBytes,
		CacheLimitBytes:       cfg.CacheLimitBytes,
		WiredLimitBytes:       cfg.WiredLimitBytes,
	})
	if err != nil {
		if cleanupErr := runCleanup(cleanup); cleanupErr != nil {
			return nil, core.ErrorJoin(err, cleanupErr)
		}
		return nil, err
	}

	info := native.Info()
	if !adapterInfo.IsEmpty() {
		adapterInfo = mergeLoadedAdapterInfo(adapterInfo, toRootAdapterInfo(info.Adapter))
	}
	var ggufInfo *gguf.Info
	if info.QuantBits == 0 || info.QuantGroup == 0 || info.Architecture == "" || info.NumLayers == 0 {
		if parsed, parsedErr := readGGUFInfo(resolvedPath); parsedErr == nil {
			ggufInfo = &parsed
		}
	}

	effectiveQuantBits := info.QuantBits
	if effectiveQuantBits == 0 && ggufInfo != nil {
		effectiveQuantBits = ggufInfo.QuantBits
	}
	if cfg.Quantization > 0 && effectiveQuantBits > 0 && effectiveQuantBits != cfg.Quantization {
		quantErr := core.NewError("mlx: loaded model quantization does not match requested bits")
		if closeErr := native.Close(); closeErr != nil {
			quantErr = core.ErrorJoin(quantErr, closeErr)
		}
		if cleanupErr := runCleanup(cleanup); cleanupErr != nil {
			quantErr = core.ErrorJoin(quantErr, cleanupErr)
		}
		return nil, quantErr
	}

	m := &Model{
		model:       native,
		cfg:         cfg,
		tok:         &Tokenizer{tok: native.Tokenizer()},
		gguf:        ggufInfo,
		adapterInfo: adapterInfo,
		cleanup:     cleanup,
	}
	// Pre-build the parser hint once now — the 7 Generate / Chat / *Stream
	// entry points then read m.parserHint directly without re-entering
	// m.model.Info() (which clones native AdapterInfo.TargetKeys) per call.
	m.refreshParserHint()
	return m, nil
}

// Err returns the last generation error, if any.
func (m *Model) Err() error {
	if m == nil || m.model == nil {
		return nil
	}
	return m.model.Err()
}

// Metrics returns performance counters from the last inference call.
func (m *Model) Metrics() Metrics {
	if m == nil || m.model == nil {
		return Metrics{}
	}
	metrics := toRootMetrics(m.model.LastMetrics())
	if metrics.Adapter.IsEmpty() {
		metrics.Adapter = m.adapterInfo
	}
	return metrics
}

// ModelType returns the internal architecture identifier.
func (m *Model) ModelType() string {
	if m == nil || m.model == nil {
		return ""
	}
	return m.model.ModelType()
}

// Info returns metadata about the loaded model.
func (m *Model) Info() ModelInfo {
	if m == nil || m.model == nil {
		return ModelInfo{}
	}
	info := m.model.Info()
	contextLength := info.ContextLength
	if m.cfg.ContextLength > 0 {
		contextLength = m.cfg.ContextLength
	}
	architecture := info.Architecture
	vocabSize := info.VocabSize
	numLayers := info.NumLayers
	numHeads := info.NumHeads
	hiddenSize := info.HiddenSize
	quantBits := info.QuantBits
	quantGroup := info.QuantGroup
	if m.gguf != nil {
		if architecture == "" {
			architecture = m.gguf.Architecture
		}
		if vocabSize == 0 {
			vocabSize = m.gguf.VocabSize
		}
		if numLayers == 0 {
			numLayers = m.gguf.NumLayers
		}
		if hiddenSize == 0 {
			hiddenSize = m.gguf.HiddenSize
		}
		if contextLength == 0 {
			contextLength = m.gguf.ContextLength
		}
		if quantBits == 0 {
			quantBits = m.gguf.QuantBits
		}
		if quantGroup == 0 {
			quantGroup = m.gguf.QuantGroup
		}
	}
	return ModelInfo{
		Architecture:          architecture,
		VocabSize:             vocabSize,
		NumLayers:             numLayers,
		NumHeads:              numHeads,
		HiddenSize:            hiddenSize,
		QuantBits:             quantBits,
		QuantGroup:            quantGroup,
		ContextLength:         contextLength,
		SlidingWindow:         info.SlidingWindow,
		ParallelSlots:         m.cfg.ParallelSlots,
		PromptCache:           m.cfg.PromptCache,
		PromptCacheMinTokens:  m.cfg.PromptCacheMinTokens,
		CachePolicy:           m.cfg.CachePolicy,
		CacheMode:             m.cfg.CacheMode,
		KVCacheStorageDType:   m.cfg.KVCacheStorageDType,
		PagedKVPageSize:       m.cfg.PagedKVPageSize,
		PagedKVPrealloc:       m.cfg.PagedKVPrealloc,
		FixedSlidingCacheSize: m.cfg.FixedSlidingCacheSize,
		BatchSize:             m.cfg.BatchSize,
		PrefillChunkSize:      m.cfg.PrefillChunkSize,
		ExpectedQuantization:  m.cfg.ExpectedQuantization,
		MemoryLimitBytes:      m.cfg.MemoryLimitBytes,
		CacheLimitBytes:       m.cfg.CacheLimitBytes,
		WiredLimitBytes:       m.cfg.WiredLimitBytes,
		// Reuse the info we already pulled from the native model — calling
		// m.Adapter() here would re-enter m.model.Info() when adapterInfo
		// is empty, doubling the native-side fetch.
		Adapter: m.adapterFromNativeInfo(info),
	}
}

// adapterFromNativeInfo mirrors m.Adapter() but reuses an already-loaded
// metal.ModelInfo, sparing the second m.model.Info() round-trip.
func (m *Model) adapterFromNativeInfo(info metal.ModelInfo) lora.AdapterInfo {
	if !m.adapterInfo.IsEmpty() {
		return m.adapterInfo
	}
	return toRootAdapterInfo(info.Adapter)
}

// Adapter returns the active LoRA inference adapter identity.
func (m *Model) Adapter() lora.AdapterInfo {
	if m == nil {
		return lora.AdapterInfo{}
	}
	if !m.adapterInfo.IsEmpty() {
		return m.adapterInfo
	}
	if m.model != nil {
		info := m.model.Info()
		return toRootAdapterInfo(info.Adapter)
	}
	return lora.AdapterInfo{}
}

// InspectAttention runs a single prefill pass and returns extracted K tensors.
func (m *Model) InspectAttention(prompt string) (*AttentionSnapshot, error) {
	if m == nil || m.model == nil {
		return nil, errMLXModelNil
	}
	result, err := m.model.InspectAttention(context.Background(), prompt)
	if err != nil {
		return nil, err
	}
	return toRootAttentionSnapshot(result), nil
}

// CaptureKV runs a single prefill pass and returns extracted K/V cache tensors.
func (m *Model) CaptureKV(prompt string) (*kv.Snapshot, error) {
	return m.CaptureKVWithOptions(prompt, kv.CaptureOptions{})
}

// CaptureKVWithOptions runs a single prefill pass and returns extracted K/V
// cache tensors with explicit capture options.
func (m *Model) CaptureKVWithOptions(prompt string, opts kv.CaptureOptions) (*kv.Snapshot, error) {
	if m == nil || m.model == nil {
		return nil, errMLXModelNil
	}
	if snapshotter, ok := m.model.(nativeKVSnapshotterWithOptions); ok {
		result, err := snapshotter.CaptureKVWithOptions(context.Background(), prompt, kvconv.ToMetalKVSnapshotCaptureOptions(opts))
		if err != nil {
			return nil, err
		}
		snapshot := kvconv.ToRootKVSnapshot(result)
		if opts.RawKVOnly {
			kv.DropFloat32(snapshot)
		}
		return snapshot, nil
	}
	snapshotter, ok := m.model.(nativeKVSnapshotter)
	if !ok {
		return nil, errMLXKVCaptureUnsupp
	}
	result, err := snapshotter.CaptureKV(context.Background(), prompt)
	if err != nil {
		return nil, err
	}
	snapshot := kvconv.ToRootKVSnapshot(result)
	if opts.RawKVOnly {
		kv.DropFloat32(snapshot)
	}
	return snapshot, nil
}

// CaptureKVChunks captures K/V state from streaming prompt chunks without one
// giant prompt-tokenization pass.
func (m *Model) CaptureKVChunks(ctx context.Context, chunks iter.Seq[string]) (*kv.Snapshot, error) {
	return m.CaptureKVChunksWithOptions(ctx, chunks, kv.CaptureOptions{})
}

// CaptureKVChunksWithOptions captures K/V state from streaming prompt chunks
// with explicit capture options.
func (m *Model) CaptureKVChunksWithOptions(ctx context.Context, chunks iter.Seq[string], opts kv.CaptureOptions) (*kv.Snapshot, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if m == nil || m.model == nil {
		return nil, errMLXModelNil
	}
	if snapshotter, ok := m.model.(nativeKVChunkSnapshotterWithOptions); ok {
		result, err := snapshotter.CaptureKVChunksWithOptions(ctx, chunks, kvconv.ToMetalKVSnapshotCaptureOptions(opts))
		if err != nil {
			return nil, err
		}
		snapshot := kvconv.ToRootKVSnapshot(result)
		if opts.RawKVOnly {
			kv.DropFloat32(snapshot)
		}
		return snapshot, nil
	}
	if snapshotter, ok := m.model.(nativeKVChunkSnapshotter); ok {
		result, err := snapshotter.CaptureKVChunks(ctx, chunks)
		if err != nil {
			return nil, err
		}
		snapshot := kvconv.ToRootKVSnapshot(result)
		if opts.RawKVOnly {
			kv.DropFloat32(snapshot)
		}
		return snapshot, nil
	}
	return m.CaptureKVWithOptions(promptChunksToString(chunks), opts)
}

func promptChunksToString(chunks iter.Seq[string]) string {
	if chunks == nil {
		return ""
	}
	builder := core.NewBuilder()
	for chunk := range chunks {
		builder.WriteString(chunk)
	}
	return builder.String()
}

// Tokenizer returns the model tokenizer.
func (m *Model) Tokenizer() *Tokenizer {
	if m == nil {
		return nil
	}
	return m.tok
}

// Close releases model resources.
func (m *Model) Close() error {
	if m == nil || m.model == nil {
		if m != nil && m.cleanup != nil {
			err := m.cleanup()
			m.cleanup = nil
			return err
		}
		return nil
	}
	native := m.model
	m.model = nil
	m.tok = nil
	err := native.Close()
	if m.cleanup != nil {
		err = core.ErrorJoin(err, m.cleanup())
		m.cleanup = nil
	}
	return err
}
