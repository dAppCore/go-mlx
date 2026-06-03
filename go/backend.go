// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"iter"
	"unsafe"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/parser"
	state "dappco.re/go/inference/state"
	"dappco.re/go/mlx/gguf"
	"dappco.re/go/mlx/kv"
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

type nativeModel interface {
	ApplyLoRA(metal.LoRAConfig) *metal.LoRAAdapter
	BatchGenerate(context.Context, []string, metal.GenerateConfig) ([]metal.BatchResult, error)
	Chat(context.Context, []metal.ChatMessage, metal.GenerateConfig) iter.Seq[metal.Token]
	Classify(context.Context, []string, metal.GenerateConfig, bool) ([]metal.ClassifyResult, error)
	Close() error
	Err() error
	Generate(context.Context, string, metal.GenerateConfig) iter.Seq[metal.Token]
	Info() metal.ModelInfo
	InspectAttention(context.Context, string) (*metal.AttentionResult, error)
	LastMetrics() metal.Metrics
	ModelType() string
	Tokenizer() *metal.Tokenizer
}

type nativePromptCacheWarmer interface {
	WarmPromptCache(context.Context, string) error
}

type nativePromptCacheChunkWarmer interface {
	WarmPromptCacheChunks(context.Context, iter.Seq[string]) error
}

type nativePromptCacheClearer interface {
	ClearPromptCache()
}

type nativePromptCacheKVRestorer interface {
	RestorePromptCacheFromKV(context.Context, *metal.KVSnapshot) error
}

type nativePromptCacheKVBlockRestorer interface {
	RestorePromptCacheFromKVBlocks(context.Context, metal.KVSnapshotBlockSource) error
}

type nativeKVSnapshotter interface {
	CaptureKV(context.Context, string) (*metal.KVSnapshot, error)
}

type nativeKVSnapshotterWithOptions interface {
	CaptureKVWithOptions(context.Context, string, metal.KVSnapshotCaptureOptions) (*metal.KVSnapshot, error)
}

type nativeKVChunkSnapshotter interface {
	CaptureKVChunks(context.Context, iter.Seq[string]) (*metal.KVSnapshot, error)
}

type nativeKVChunkSnapshotterWithOptions interface {
	CaptureKVChunksWithOptions(context.Context, iter.Seq[string], metal.KVSnapshotCaptureOptions) (*metal.KVSnapshot, error)
}

type nativeChunkGenerator interface {
	GenerateChunks(context.Context, iter.Seq[string], metal.GenerateConfig) iter.Seq[metal.Token]
}

type nativeChatChunkGenerator interface {
	ChatChunks(context.Context, []metal.ChatMessage, int, metal.GenerateConfig) iter.Seq[metal.Token]
}

type nativeLoRALoader interface {
	LoadLoRA(string) (*metal.LoRAAdapter, error)
}

type nativeLoRAUnloader interface {
	UnloadLoRA() error
}

// Model is the RFC-style root-package model handle.
type Model struct {
	model       nativeModel
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

var loadNativeModel = func(modelPath string, cfg metal.LoadConfig) (nativeModel, error) {
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
	// Per-block sentinels hit on the State KV block restore hot path —
	// metalKVSnapshotBlockSource.Load fires once per covering block during
	// every WarmPromptCacheFromStateBlocks call (large prefixes mean dozens
	// of invocations), so hoisting these to package-level drops a per-block
	// core.NewError alloc on every load.
	errMLXStateKVStoreNil          = core.NewError("mlx: state store is nil")
	errMLXStateKVPrefixExceeds     = core.NewError("mlx: State KV prefix exceeds bundle token count")
	errMLXStateKVPrefixNoCovering  = core.NewError("mlx: State KV prefix has no covering blocks")
	errMLXStateKVBlockOutOfRange   = core.NewError("mlx: State KV block index is out of range")
	errMLXStateKVBlockMetaMismatch = core.NewError("mlx: State KV block metadata mismatch")
	errMLXStateKVBlockSnapshotNil  = core.NewError("mlx: State KV block snapshot is nil")
	errMLXStateKVPrefixInvalidTrim = core.NewError("mlx: State KV prefix has invalid trim range")
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
		ContextLen:           cfg.ContextLength,
		Gemma4SlidingWindow:  cfg.Gemma4SlidingWindow,
		ParallelSlots:        cfg.ParallelSlots,
		DisablePromptCache:   !cfg.PromptCache,
		PromptCacheMinTokens: cfg.PromptCacheMinTokens,
		AdapterPath:          resolvedAdapterPath,
		Device:               metal.DeviceType(cfg.Device),
		CachePolicy:          string(cfg.CachePolicy),
		KVCacheMode:          string(cfg.CacheMode),
		BatchSize:            cfg.BatchSize,
		PrefillChunkSize:     cfg.PrefillChunkSize,
		ExpectedQuantization: cfg.ExpectedQuantization,
		MemoryLimitBytes:     cfg.MemoryLimitBytes,
		CacheLimitBytes:      cfg.CacheLimitBytes,
		WiredLimitBytes:      cfg.WiredLimitBytes,
	})
	if err != nil {
		if cleanupErr := runCleanup(cleanup); cleanupErr != nil {
			return nil, core.ErrorJoin(err, cleanupErr)
		}
		return nil, err
	}

	info := native.Info()
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

func toMetalGenerateConfig(cfg GenerateConfig) metal.GenerateConfig {
	return metal.GenerateConfig{
		MaxTokens:           cfg.MaxTokens,
		Temperature:         cfg.Temperature,
		TopK:                cfg.TopK,
		TopP:                cfg.TopP,
		MinP:                cfg.MinP,
		Seed:                cfg.Seed,
		SeedSet:             cfg.SeedSet,
		StopTokens:          cfg.StopTokens,
		SuppressTokens:      cfg.SuppressTokens,
		MinTokensBeforeStop: cfg.MinTokensBeforeStop,
		RepeatPenalty:       cfg.RepeatPenalty,
		ProbeSink:           toMetalProbeSink(cfg.ProbeSink),
		TraceTokenPhases:    cfg.TraceTokenPhases,
		TraceTokenText:      cfg.TraceTokenText,
	}
}

// metalProbeSinkAdapter forwards metal.ProbeEvent into a probe.Sink
// after the metal→root event conversion. Replaces the per-call closure
// allocation in toMetalProbeSink — the closure form below captured
// `sink` into a fresh func per Generate/Chat/Classify call (24 B + GC
// pressure on the per-call hot path even when ProbeSink was non-nil but
// emitted few events). The struct form is heap-allocated once per call
// but is two pointer-sized words and qualifies for stack allocation
// when the metal config doesn't escape.
type metalProbeSinkAdapter struct {
	sink probe.Sink
}

// EmitProbe converts metal.ProbeEvent to probe.Event and forwards to the
// wrapped root sink. Called per token during generation when the caller
// supplies a ProbeSink — the conversion still allocates per event but
// the dispatch site no longer allocates a closure per Generate call.
func (a metalProbeSinkAdapter) EmitProbe(event metal.ProbeEvent) {
	a.sink.EmitProbe(toRootProbeEvent(event))
}

func toMetalProbeSink(sink probe.Sink) metal.ProbeSink {
	if sink == nil {
		return nil
	}
	return metalProbeSinkAdapter{sink: sink}
}

func toRootProbeEvent(event metal.ProbeEvent) probe.Event {
	// Read sub-fields direct through the source pointer — the previous
	// `x := *event.X` dereference-copy form materialised the entire
	// substruct (ProbeLogits alone is ~130 B with three slice headers
	// + a map header) into a local before reading individual fields.
	// toRootProbeEvent fires per probe event, which under ProbeSink is
	// emitted PER TOKEN during generation — skipping the redundant
	// substruct copy compounds across long generations.
	out := probe.Event{
		Kind:  probe.Kind(event.Kind),
		Phase: probe.Phase(event.Phase),
		Step:  event.Step,
		Meta:  cloneMetalProbeMeta(event.Meta),
	}
	if event.Token != nil {
		token := event.Token
		out.Token = &probe.Token{
			ID:              token.ID,
			Text:            token.Text,
			PromptTokens:    token.PromptTokens,
			GeneratedTokens: token.GeneratedTokens,
		}
	}
	if event.Logits != nil {
		logits := event.Logits
		out.Logits = &probe.Logits{
			Shape:      core.SliceClone(logits.Shape),
			VocabSize:  logits.VocabSize,
			MaxTokenID: logits.MaxTokenID,
			MaxLogit:   logits.MaxLogit,
			MinTokenID: logits.MinTokenID,
			MinLogit:   logits.MinLogit,
			MeanLogit:  logits.MeanLogit,
			Top:        toRootProbeLogits(logits.Top),
			Values:     core.SliceClone(logits.Values),
			Meta:       cloneMetalProbeMeta(logits.Meta),
		}
	}
	if event.Entropy != nil {
		entropy := event.Entropy
		out.Entropy = &probe.Entropy{Value: entropy.Value, Unit: entropy.Unit}
	}
	if event.SelectedHeads != nil {
		heads := event.SelectedHeads
		out.SelectedHeads = &probe.HeadSelection{
			Layer:  heads.Layer,
			Heads:  core.SliceClone(heads.Heads),
			Scores: core.SliceClone(heads.Scores),
		}
	}
	if event.LayerCoherence != nil {
		coherence := event.LayerCoherence
		out.LayerCoherence = &probe.LayerCoherence{
			Layer:          coherence.Layer,
			KeyCoherence:   coherence.KeyCoherence,
			ValueCoherence: coherence.ValueCoherence,
			CrossAlignment: coherence.CrossAlignment,
			KVCoupling:     coherence.KVCoupling,
			HeadEntropy:    coherence.HeadEntropy,
			PhaseLock:      coherence.PhaseLock,
		}
	}
	if event.RouterDecision != nil {
		router := event.RouterDecision
		out.RouterDecision = &probe.RouterDecision{
			Layer:       router.Layer,
			TokenID:     router.TokenID,
			ExpertIDs:   core.SliceClone(router.ExpertIDs),
			Weights:     core.SliceClone(router.Weights),
			Temperature: router.Temperature,
		}
	}
	if event.Residual != nil {
		residual := event.Residual
		out.Residual = &probe.ResidualSummary{
			Layer:    residual.Layer,
			Mean:     residual.Mean,
			Variance: residual.Variance,
			RMS:      residual.RMS,
			L2Norm:   residual.L2Norm,
			MaxAbs:   residual.MaxAbs,
		}
	}
	if event.Cache != nil {
		cache := event.Cache
		out.Cache = &probe.CachePressure{
			PromptTokens:    cache.PromptTokens,
			GeneratedTokens: cache.GeneratedTokens,
			LayerCount:      cache.LayerCount,
			CacheTokens:     cache.CacheTokens,
			ProcessedTokens: cache.ProcessedTokens,
			MaxCacheTokens:  cache.MaxCacheTokens,
			Utilization:     cache.Utilization,
			Rotating:        cache.Rotating,
		}
	}
	if event.Memory != nil {
		memory := event.Memory
		out.Memory = &probe.MemoryPressure{
			ActiveBytes: memory.ActiveBytes,
			PeakBytes:   memory.PeakBytes,
			CacheBytes:  memory.CacheBytes,
		}
	}
	if event.Training != nil {
		training := event.Training
		out.Training = &probe.Training{
			Step:         training.Step,
			Epoch:        training.Epoch,
			Loss:         training.Loss,
			LearningRate: training.LearningRate,
			GradNorm:     training.GradNorm,
		}
	}
	return out
}

func toRootProbeLogits(logits []metal.ProbeLogit) []probe.Logit {
	if len(logits) == 0 {
		return nil
	}
	// W8-A2 unsafe reinterpret — metal.ProbeLogit and probe.Logit have
	// bit-identical layout (int32 TokenID + float32 Logit + float64
	// Probability, with the same field order). The compile-time guard
	// at the top of the file fires if either struct ever drifts. Cast
	// the source slice header in-place, then `copy` does one memcpy
	// instead of len(logits) per-field unpacks. Top-K is commonly
	// 50-100 entries per probe event, emitted per-token when ProbeSink
	// is enabled — every saved unpack compounds across the generation.
	src := unsafe.Slice((*probe.Logit)(unsafe.Pointer(&logits[0])), len(logits))
	out := make([]probe.Logit, len(logits))
	copy(out, src)
	return out
}

func cloneMetalProbeMeta(meta map[string]string) map[string]string {
	if len(meta) == 0 {
		return nil
	}
	return core.MapClone(meta)
}

func toRootMetrics(metrics metal.Metrics) Metrics {
	return Metrics{
		PromptTokens:               metrics.PromptTokens,
		GeneratedTokens:            metrics.GeneratedTokens,
		FirstTokenDuration:         metrics.FirstTokenDuration,
		PrefillDuration:            metrics.PrefillDuration,
		DecodeDuration:             metrics.DecodeDuration,
		TotalDuration:              metrics.TotalDuration,
		PrefillTokensPerSec:        metrics.PrefillTokensPerSec,
		DecodeTokensPerSec:         metrics.DecodeTokensPerSec,
		PeakMemoryBytes:            metrics.PeakMemoryBytes,
		ActiveMemoryBytes:          metrics.ActiveMemoryBytes,
		CacheMemoryBytes:           metrics.CacheMemoryBytes,
		ProcessVirtualMemoryBytes:  metrics.ProcessVirtualMemoryBytes,
		ProcessResidentMemoryBytes: metrics.ProcessResidentMemoryBytes,
		ProcessPeakResidentBytes:   metrics.ProcessPeakResidentBytes,
		PromptCacheHits:            metrics.PromptCacheHits,
		PromptCacheMisses:          metrics.PromptCacheMisses,
		PromptCacheHitTokens:       metrics.PromptCacheHitTokens,
		PromptCacheMissTokens:      metrics.PromptCacheMissTokens,
		PromptCacheRestoreDuration: metrics.PromptCacheRestoreDuration,
		CacheProfile:               toRootCacheProfile(metrics.CacheProfile),
		TurboQuantKVPayload:        toRootTurboQuantKVPayloadEstimate(metrics.TurboQuantKVPayload),
		TokenPhases:                toRootTokenPhaseTraces(metrics.TokenPhases),
		MTP:                        toRootMTPMetrics(metrics.MTP),
		Adapter:                    toRootAdapterInfo(metrics.Adapter),
	}
}

func toRootTurboQuantKVPayloadEstimate(estimate *metal.TurboQuantKVCachePayloadEstimate) *TurboQuantKVPayloadEstimate {
	if estimate == nil {
		return nil
	}
	return &TurboQuantKVPayloadEstimate{
		Pages:                     estimate.Pages,
		PageVectors:               estimate.PageVectors,
		PageElements:              estimate.PageElements,
		KeyCentroidBytes:          estimate.KeyCentroidBytes,
		KeyQJLSignBytes:           estimate.KeyQJLSignBytes,
		KeyNormBytes:              estimate.KeyNormBytes,
		KeyResidualNormBytes:      estimate.KeyResidualNormBytes,
		ValueCentroidBytes:        estimate.ValueCentroidBytes,
		ValueNormBytes:            estimate.ValueNormBytes,
		OutlierMaskBytes:          estimate.OutlierMaskBytes,
		PayloadBytes:              estimate.PayloadBytes,
		PaddedPayloadBytes:        estimate.PaddedPayloadBytes,
		AlignmentPaddingBytes:     estimate.AlignmentPaddingBytes,
		FP16BaselineBytes:         estimate.FP16BaselineBytes,
		PayloadToFP16Ratio:        estimate.PayloadToFP16Ratio,
		PaddedPayloadToFP16Ratio:  estimate.PaddedPayloadToFP16Ratio,
		PayloadSavingsRatio:       estimate.PayloadSavingsRatio,
		PaddedPayloadSavingsRatio: estimate.PaddedPayloadSavingsRatio,
	}
}

func toRootMTPMetrics(metrics *metal.MTPMetrics) *MTPMetrics {
	if metrics == nil {
		return nil
	}
	return &MTPMetrics{
		DraftTokenSchedule:     append([]int(nil), metrics.DraftTokenSchedule...),
		ProposedTokens:         metrics.ProposedTokens,
		AcceptedTokens:         metrics.AcceptedTokens,
		RejectedTokens:         metrics.RejectedTokens,
		TargetVerifyCalls:      metrics.TargetVerifyCalls,
		TargetCalls:            metrics.TargetCalls,
		DraftCalls:             metrics.DraftCalls,
		AcceptanceRate:         metrics.AcceptanceRate,
		VisibleTokensPerSec:    metrics.VisibleTokensPerSec,
		TargetTokensPerSec:     metrics.TargetTokensPerSec,
		WarmDecodeTokensPerSec: metrics.WarmDecodeTokensPerSec,
		WallDuration:           metrics.WallDuration,
		RestoreDuration:        metrics.RestoreDuration,
		TargetVerifyDuration:   metrics.TargetVerifyDuration,
		TargetDuration:         metrics.TargetDuration,
		DraftDuration:          metrics.DraftDuration,
		PeakMemoryBytes:        metrics.PeakMemoryBytes,
	}
}

func toRootCacheProfile(profile *metal.CacheProfile) *CacheProfile {
	if profile == nil {
		return nil
	}
	return &CacheProfile{
		Architecture:       profile.Architecture,
		TotalCaches:        profile.TotalCaches,
		LocalCaches:        profile.LocalCaches,
		GlobalCaches:       profile.GlobalCaches,
		SharedLayers:       profile.SharedLayers,
		CachelessLayers:    profile.CachelessLayers,
		LocalWindowTokens:  profile.LocalWindowTokens,
		MaxLocalTokens:     profile.MaxLocalTokens,
		MaxLocalCapacity:   profile.MaxLocalCapacity,
		MaxGlobalTokens:    profile.MaxGlobalTokens,
		MaxGlobalCapacity:  profile.MaxGlobalCapacity,
		MaxCacheTokens:     profile.MaxCacheTokens,
		MaxCacheCapacity:   profile.MaxCacheCapacity,
		MaxProcessedTokens: profile.MaxProcessedTokens,
		FullCaches:         profile.FullCaches,
		RotatingCaches:     profile.RotatingCaches,
		FixedCaches:        profile.FixedCaches,
		PagedCaches:        profile.PagedCaches,
		QuantizedCaches:    profile.QuantizedCaches,
		UnknownCaches:      profile.UnknownCaches,
		UnboundedCaches:    profile.UnboundedCaches,
		LocalWindowLeaked:  profile.LocalWindowLeaked,
	}
}

func toRootTokenPhaseTraces(phases []metal.TokenPhaseTrace) []TokenPhaseTrace {
	if len(phases) == 0 {
		return nil
	}
	out := make([]TokenPhaseTrace, len(phases))
	// Single arena allocation for the per-phase NativeEvents slices.
	// TraceTokenPhases-enabled metrics emit one TokenPhaseTrace per
	// decoded token, each with a NativeEvents fanout — collapsing the
	// per-phase make into one slab avoids len(phases) small allocs on
	// every Metrics() read with phase tracing enabled.
	totalNative := 0
	for i := range phases {
		totalNative += len(phases[i].NativeEvents)
	}
	var nativeSlab []NativePhaseTrace
	nativeOffset := 0
	if totalNative > 0 {
		nativeSlab = make([]NativePhaseTrace, totalNative)
	}
	// Index iteration — metal.TokenPhaseTrace is ~192 B (19 duration
	// + Step int + TokenID int32 + TokenText string + FinalToken bool
	// + NativeEvents slice header).
	// metal.NativePhaseTrace is small but contains strings and counters; avoid
	// copying it through a range variable on long traced generations.
	// TraceTokenPhases emits ONE phase trace per decoded token, so for
	// long generations the range form was copying many KB of struct
	// data into loop variables before re-emitting it via field rebuild.
	for i := range phases {
		phase := &phases[i]
		nativeSrc := phase.NativeEvents
		var phaseNative []NativePhaseTrace
		if n := len(nativeSrc); n > 0 {
			end := nativeOffset + n
			phaseNative = nativeSlab[nativeOffset:end:end]
			for j := range nativeSrc {
				event := &nativeSrc[j]
				phaseNative[j] = NativePhaseTrace{
					Name:     event.Name,
					Duration: event.Duration,
					Error:    event.Error,
					Pages:    event.Pages,
					Tokens:   event.Tokens,
				}
			}
			nativeOffset = end
		}
		out[i] = TokenPhaseTrace{
			Step:                   phase.Step,
			TokenID:                phase.TokenID,
			TokenText:              phase.TokenText,
			FinalToken:             phase.FinalToken,
			TotalDuration:          phase.TotalDuration,
			LogitsDuration:         phase.LogitsDuration,
			SampleDuration:         phase.SampleDuration,
			SampleEvalDuration:     phase.SampleEvalDuration,
			TokenReadDuration:      phase.TokenReadDuration,
			DecodeTextDuration:     phase.DecodeTextDuration,
			ProbeTokenDuration:     phase.ProbeTokenDuration,
			YieldDuration:          phase.YieldDuration,
			NextInputDuration:      phase.NextInputDuration,
			ForwardDuration:        phase.ForwardDuration,
			PrefetchDuration:       phase.PrefetchDuration,
			PrefetchLogitsDuration: phase.PrefetchLogitsDuration,
			PrefetchCacheDuration:  phase.PrefetchCacheDuration,
			MaterializeDuration:    phase.MaterializeDuration,
			DetachDuration:         phase.DetachDuration,
			CacheProbeDuration:     phase.CacheProbeDuration,
			OtherDuration:          phase.OtherDuration,
			NativeEvents:           phaseNative,
		}
	}
	return out
}

func toRootNativePhaseTraces(events []metal.NativePhaseTrace) []NativePhaseTrace {
	if len(events) == 0 {
		return nil
	}
	out := make([]NativePhaseTrace, len(events))
	// Index iteration — see toRootTokenPhaseTraces; NativePhaseTrace is
	// ~48 B and the range form copied each event into the loop variable
	// before re-emitting via field rebuild.
	for i := range events {
		event := &events[i]
		out[i] = NativePhaseTrace{
			Name:     event.Name,
			Duration: event.Duration,
			Error:    event.Error,
			Pages:    event.Pages,
			Tokens:   event.Tokens,
		}
	}
	return out
}

// toRootAdapterInfo shuffles an already-cloned metal AdapterInfo into the
// root-facing lora.AdapterInfo. All four callers pass slices that the
// metal side already cloned for caller isolation:
//
//   - toRootMetrics — metrics.Adapter comes from m.lastMetrics.Adapter
//     which is assigned via metal.(*Model).Adapter() (cloneMetalAdapterInfo).
//   - adapterFromNativeInfo + (*Model).Adapter — info.Adapter likewise
//     comes from m.Info() → m.Adapter() which clones.
//   - inference_contract.go — passes adapter.model.Adapter() directly.
//
// The previous core.SliceClone(info.TargetKeys) at this layer was a
// redundant second clone — drops a 64 B / 1 alloc per call by sharing
// the already-isolated slice with the root-side handle. Every Info() /
// Metrics() / Adapter() read on a LoRA-loaded model fires this site.
func toRootAdapterInfo(info metal.AdapterInfo) lora.AdapterInfo {
	return lora.AdapterInfo{
		Name:       info.Name,
		Path:       info.Path,
		Hash:       info.Hash,
		Rank:       info.Rank,
		Alpha:      info.Alpha,
		Scale:      info.Scale,
		TargetKeys: info.TargetKeys,
	}
}

func toRootToken(token metal.Token) Token {
	return Token{ID: token.ID, Value: token.Text, Text: token.Text}
}

func toRootClassifyResults(results []metal.ClassifyResult) []ClassifyResult {
	if len(results) == 0 {
		return nil
	}
	out := make([]ClassifyResult, len(results))
	// Single arena allocation for all per-result Logits slices. Classify
	// is called over multiple prompts at once and each result has a
	// vocab-sized logits vector — collapsing the per-result clone into
	// one slab cuts N allocs to 1 on the return path. Per-result nil vs
	// non-nil empty is preserved (matches the prior core.SliceClone
	// nil-in / empty-in semantics).
	totalLogits := 0
	for i := range results {
		totalLogits += len(results[i].Logits)
	}
	var logitsSlab []float32
	logitsOffset := 0
	if totalLogits > 0 {
		logitsSlab = make([]float32, totalLogits)
	}
	// Index iteration — metal.ClassifyResult carries a Token (3 fields)
	// + Logits slice header. Skip the per-iter struct copy.
	for i := range results {
		result := &results[i]
		var resultLogits []float32
		switch {
		case result.Logits == nil:
			// nil in -> nil out (matches slices.Clone(nil)).
		case len(result.Logits) == 0:
			resultLogits = []float32{}
		default:
			end := logitsOffset + len(result.Logits)
			resultLogits = logitsSlab[logitsOffset:end:end]
			copy(resultLogits, result.Logits)
			logitsOffset = end
		}
		out[i] = ClassifyResult{
			Token:  toRootToken(result.Token),
			Logits: resultLogits,
		}
	}
	return out
}

func toRootBatchResults(results []metal.BatchResult) []BatchResult {
	if len(results) == 0 {
		return nil
	}
	out := make([]BatchResult, len(results))
	// Single arena allocation for all per-result Tokens slices. Avoids
	// len(results) small allocations on BatchGenerate's return path.
	totalTokens := 0
	for i := range results {
		totalTokens += len(results[i].Tokens)
	}
	tokensSlab := make([]Token, totalTokens)
	tokensOffset := 0
	// Index iteration — metal.BatchResult is a Tokens slice header +
	// error interface. metal.Token is a small (ID int32 + Text string)
	// 24 B struct, but for long-generation batches the outer slice can
	// be hundreds long and the inner Tokens slices can be thousands.
	for i := range results {
		result := &results[i]
		tokensSrc := result.Tokens
		tokensEnd := tokensOffset + len(tokensSrc)
		resultTokens := tokensSlab[tokensOffset:tokensEnd:tokensEnd]
		for j := range tokensSrc {
			resultTokens[j] = toRootToken(tokensSrc[j])
		}
		out[i] = BatchResult{
			Tokens: resultTokens,
			Err:    result.Err,
		}
		tokensOffset = tokensEnd
	}
	return out
}

func toRootAttentionSnapshot(result *metal.AttentionResult) *AttentionSnapshot {
	if result == nil {
		return nil
	}
	return &AttentionSnapshot{
		NumLayers:     result.NumLayers,
		NumHeads:      result.NumHeads,
		SeqLen:        result.SeqLen,
		HeadDim:       result.HeadDim,
		NumQueryHeads: result.NumQueryHeads,
		Keys:          result.Keys,
		Queries:       result.Queries,
		Architecture:  result.Architecture,
	}
}

func toRootKVSnapshot(result *metal.KVSnapshot) *kv.Snapshot {
	if result == nil {
		return nil
	}
	resultLayers := result.Layers
	layers := make([]kv.LayerSnapshot, len(resultLayers))
	// Single arena allocation for all per-layer Heads slices. Avoids N
	// small allocations on a path that runs per KV capture / restore.
	totalHeads := 0
	totalKey := 0
	totalValue := 0
	totalKeyBytes := 0
	totalValueBytes := 0
	// totalInt32 covers per-layer KeyShape + ValueShape AND the top-level
	// Tokens + Generated + LogitShape slices — all share the same int32
	// element type and the same once-per-snapshot lifetime, so they share
	// one arena. Drops 3 + 2×layers small clones to 1 outer alloc.
	totalInt32 := len(result.Tokens) + len(result.Generated) + len(result.LogitShape)
	totalLogits := len(result.Logits)
	for i := range resultLayers {
		layer := &resultLayers[i]
		heads := layer.Heads
		totalHeads += len(heads)
		totalInt32 += len(layer.KeyShape) + len(layer.ValueShape)
		for j := range heads {
			head := &heads[j]
			totalKey += len(head.Key)
			totalValue += len(head.Value)
			totalKeyBytes += len(head.KeyBytes)
			totalValueBytes += len(head.ValueBytes)
		}
	}
	headsSlab := make([]kv.HeadSnapshot, totalHeads)
	// One float32 slab covers per-head Key + per-head Value + top-level
	// Logits — all are []float32 with once-per-snapshot lifetime. Previous
	// shape: 2 head-family slabs + 1 standalone Logits clone = 3 allocs;
	// unified: 1 alloc regardless of (layers × heads × Logits len).
	// keyOffset / valueOffset / logitsOffset partition the slab into the
	// three regions without ever overlapping (offsets are monotonic and
	// total exactly totalFloat32). 3-cap sub-slicing keeps each sub-region
	// safely append-bounded against neighbours.
	totalFloat32 := totalKey + totalValue + totalLogits
	var float32Slab []float32
	if totalFloat32 > 0 {
		float32Slab = make([]float32, totalFloat32)
	}
	// Same pattern for per-head KeyBytes + ValueBytes — both []byte, both
	// once-per-snapshot — one byteSlab instead of two outer allocs.
	totalBytes := totalKeyBytes + totalValueBytes
	var byteSlab []byte
	if totalBytes > 0 {
		byteSlab = make([]byte, totalBytes)
	}
	var int32Slab []int32
	if totalInt32 > 0 {
		int32Slab = make([]int32, totalInt32)
	}
	headsOffset := 0
	keyOffset := 0
	// value region begins where key region ends.
	valueOffset := totalKey
	// logits region begins where value region ends (we lay it down at the
	// end below).
	logitsOffset := totalKey + totalValue
	keyBytesOffset := 0
	// valueBytes region begins where keyBytes region ends.
	valueBytesOffset := totalKeyBytes
	int32Offset := 0
	// Index iteration on both loops — KVLayerSnapshot is ~136 B (4 slice
	// headers + 2 strings + 2 byte-slice headers) and KVHeadSnapshot is
	// ~160 B (6 slice headers + 2 dtype strings); for deep models (Gemma
	// 4 E4B = 30 layers × 16 heads = 480 head-copies per snapshot)
	// the range-and-copy intermediate variable was 100+ KB of redundant
	// stack copies per capture. Read fields direct from resultLayers[i].
	for i := range resultLayers {
		layer := &resultLayers[i]
		layerHeadsSrc := layer.Heads
		headsEnd := headsOffset + len(layerHeadsSrc)
		layerHeads := headsSlab[headsOffset:headsEnd:headsEnd]
		// Per-layer shape clones cut from the shared int32 arena.
		var keyShape, valueShape []int32
		switch {
		case layer.KeyShape == nil:
		case len(layer.KeyShape) == 0:
			keyShape = []int32{}
		default:
			end := int32Offset + len(layer.KeyShape)
			keyShape = int32Slab[int32Offset:end:end]
			copy(keyShape, layer.KeyShape)
			int32Offset = end
		}
		switch {
		case layer.ValueShape == nil:
		case len(layer.ValueShape) == 0:
			valueShape = []int32{}
		default:
			end := int32Offset + len(layer.ValueShape)
			valueShape = int32Slab[int32Offset:end:end]
			copy(valueShape, layer.ValueShape)
			int32Offset = end
		}
		layers[i] = kv.LayerSnapshot{
			Layer:              layer.Layer,
			CacheIndex:         layer.CacheIndex,
			CacheMode:          string(layer.CacheMode),
			TurboQuantPayloads: rootTurboQuantPayloads(layer.TurboQuantPayloads),
			KeyDType:           rootKVHeadDType(layer.KeyDType, layer.KeyBytes),
			KeyBytes:           layer.KeyBytes,
			KeyShape:           keyShape,
			ValueDType:         rootKVHeadDType(layer.ValueDType, layer.ValueBytes),
			ValueBytes:         layer.ValueBytes,
			ValueShape:         valueShape,
			Heads:              layerHeads,
		}
		for j := range layerHeadsSrc {
			head := &layerHeadsSrc[j]
			// Allocate per-head slices out of the pre-sized arenas. Each
			// branch preserves the prior nil-in -> nil-out / empty-in ->
			// empty-out semantics of core.SliceClone so downstream
			// callers see identical post-clone shape.
			var headKey []float32
			switch {
			case head.Key == nil:
				// nil in -> nil out
			case len(head.Key) == 0:
				headKey = []float32{}
			default:
				end := keyOffset + len(head.Key)
				headKey = float32Slab[keyOffset:end:end]
				copy(headKey, head.Key)
				keyOffset = end
			}
			var headValue []float32
			switch {
			case head.Value == nil:
			case len(head.Value) == 0:
				headValue = []float32{}
			default:
				end := valueOffset + len(head.Value)
				headValue = float32Slab[valueOffset:end:end]
				copy(headValue, head.Value)
				valueOffset = end
			}
			var headKeyBytes []byte
			switch {
			case head.KeyBytes == nil:
			case len(head.KeyBytes) == 0:
				headKeyBytes = []byte{}
			default:
				end := keyBytesOffset + len(head.KeyBytes)
				headKeyBytes = byteSlab[keyBytesOffset:end:end]
				copy(headKeyBytes, head.KeyBytes)
				keyBytesOffset = end
			}
			var headValueBytes []byte
			switch {
			case head.ValueBytes == nil:
			case len(head.ValueBytes) == 0:
				headValueBytes = []byte{}
			default:
				end := valueBytesOffset + len(head.ValueBytes)
				headValueBytes = byteSlab[valueBytesOffset:end:end]
				copy(headValueBytes, head.ValueBytes)
				valueBytesOffset = end
			}
			layerHeads[j] = kv.HeadSnapshot{
				Key:        headKey,
				KeyDType:   rootKVHeadDType(head.KeyDType, head.KeyBytes),
				KeyBytes:   headKeyBytes,
				Value:      headValue,
				ValueDType: rootKVHeadDType(head.ValueDType, head.ValueBytes),
				ValueBytes: headValueBytes,
			}
		}
		headsOffset = headsEnd
	}
	// Top-level int32 slices share the same arena as the per-layer shape
	// clones — preserves the same nil-in/empty-in/non-empty semantics
	// core.SliceClone provided so downstream callers see no change.
	var tokens, generated, logitShape []int32
	switch {
	case result.Tokens == nil:
	case len(result.Tokens) == 0:
		tokens = []int32{}
	default:
		end := int32Offset + len(result.Tokens)
		tokens = int32Slab[int32Offset:end:end]
		copy(tokens, result.Tokens)
		int32Offset = end
	}
	switch {
	case result.Generated == nil:
	case len(result.Generated) == 0:
		generated = []int32{}
	default:
		end := int32Offset + len(result.Generated)
		generated = int32Slab[int32Offset:end:end]
		copy(generated, result.Generated)
		int32Offset = end
	}
	switch {
	case result.LogitShape == nil:
	case len(result.LogitShape) == 0:
		logitShape = []int32{}
	default:
		end := int32Offset + len(result.LogitShape)
		logitShape = int32Slab[int32Offset:end:end]
		copy(logitShape, result.LogitShape)
		int32Offset = end
	}
	// Top-level Logits sits in the tail region of the shared float32 slab.
	var topLogits []float32
	switch {
	case result.Logits == nil:
	case len(result.Logits) == 0:
		topLogits = []float32{}
	default:
		end := logitsOffset + len(result.Logits)
		topLogits = float32Slab[logitsOffset:end:end]
		copy(topLogits, result.Logits)
		logitsOffset = end
	}
	return &kv.Snapshot{
		Version:       result.Version,
		Architecture:  result.Architecture,
		Tokens:        tokens,
		Generated:     generated,
		TokenOffset:   result.TokenOffset,
		NumLayers:     result.NumLayers,
		NumHeads:      result.NumHeads,
		SeqLen:        result.SeqLen,
		HeadDim:       result.HeadDim,
		NumQueryHeads: result.NumQueryHeads,
		LogitShape:    logitShape,
		Logits:        topLogits,
		Layers:        layers,
	}
}

// kvLayerHasNativeSlab reports whether a layer carries native K/V slab
// bytes. When true the metal restorer pins those bytes zero-copy and never
// reads the layer's per-head float32, so toMetalKVSnapshot can skip the
// per-head materialisation. Both K and V must be present — a half-native
// layer would still hit the heads decode path on the missing side.
//
//	kvLayerHasNativeSlab(&kv.LayerSnapshot{KeyBytes: b, ValueBytes: b}) // true
func kvLayerHasNativeSlab(layer *kv.LayerSnapshot) bool {
	return len(layer.KeyBytes) > 0 && len(layer.ValueBytes) > 0
}

func rootTurboQuantPayloads(payloads []metal.TurboQuantKVReferencePagePayload) [][]byte {
	if len(payloads) == 0 {
		return nil
	}
	out := make([][]byte, 0, len(payloads))
	for idx := range payloads {
		encoded := core.JSONMarshal(payloads[idx])
		if !encoded.OK {
			return nil
		}
		out = append(out, core.SliceClone(encoded.Value.([]byte)))
	}
	return out
}

func metalTurboQuantPayloads(payloads [][]byte) []metal.TurboQuantKVReferencePagePayload {
	if len(payloads) == 0 {
		return nil
	}
	out := make([]metal.TurboQuantKVReferencePagePayload, 0, len(payloads))
	for idx := range payloads {
		if len(payloads[idx]) == 0 {
			return nil
		}
		var payload metal.TurboQuantKVReferencePagePayload
		if result := core.JSONUnmarshal(payloads[idx], &payload); !result.OK {
			return nil
		}
		if err := payload.Layout.Validate(); err != nil {
			return nil
		}
		out = append(out, payload)
	}
	return out
}

func toMetalKVSnapshot(result *kv.Snapshot) *metal.KVSnapshot {
	if result == nil {
		return nil
	}
	resultLayers := result.Layers
	layers := make([]metal.KVLayerSnapshot, len(resultLayers))
	// Single arena allocations for the per-layer Heads slices and the
	// per-head Key + Value tensor copies. The inverse direction only
	// clones Key + Value (KeyBytes / ValueBytes pass through by reference
	// from the root side), so the per-head alloc budget is 2 instead of
	// toRootKVSnapshot's 4. Coalescing into single float32 slabs drops
	// 2×heads small allocations to 2 outer allocations regardless of
	// (layers × heads). Gemma 4 E4B (30 × 16 = 480 heads) goes from 960
	// to 2 per snapshot.
	totalHeads := 0
	totalKey := 0
	totalValue := 0
	// totalInt32 covers per-layer KeyShape + ValueShape AND the top-level
	// Tokens + Generated + LogitShape slices — all share the same int32
	// element type and the same once-per-snapshot lifetime, so they share
	// one arena. Drops 3 + 2×layers small clones to 1 outer alloc.
	totalInt32 := len(result.Tokens) + len(result.Generated) + len(result.LogitShape)
	totalLogits := len(result.Logits)
	for i := range resultLayers {
		layer := &resultLayers[i]
		heads := layer.Heads
		totalHeads += len(heads)
		totalInt32 += len(layer.KeyShape) + len(layer.ValueShape)
		// When a layer carries native K/V slab bytes the metal restorer
		// reads ONLY those bytes (kvLayerArrays takes the native-slab
		// branch and ignores per-head Key/Value); the decoded per-head
		// float32 are dead weight. A v4 snapshot loaded with the default
		// (non-RawKVOnly) options populates BOTH — copying the heads here
		// would materialise the entire prefix cache a second time alongside
		// the byte slab the restorer actually pins zero-copy. Skip them.
		if kvLayerHasNativeSlab(layer) {
			continue
		}
		for j := range heads {
			head := &heads[j]
			totalKey += len(head.Key)
			totalValue += len(head.Value)
		}
	}
	headsSlab := make([]metal.KVHeadSnapshot, totalHeads)
	// One float32 slab covers per-head Key + per-head Value + top-level
	// Logits — all []float32, all once-per-snapshot. Previous shape was
	// 2 head-family slabs + 1 standalone Logits clone = 3 outer allocs;
	// unified: 1 alloc regardless of (layers × heads × Logits len).
	totalFloat32 := totalKey + totalValue + totalLogits
	var float32Slab []float32
	if totalFloat32 > 0 {
		float32Slab = make([]float32, totalFloat32)
	}
	var int32Slab []int32
	if totalInt32 > 0 {
		int32Slab = make([]int32, totalInt32)
	}
	headsOffset := 0
	keyOffset := 0
	// value region begins where key region ends.
	valueOffset := totalKey
	// logits region begins where value region ends.
	logitsOffset := totalKey + totalValue
	int32Offset := 0
	// Index iteration — see toRootKVSnapshot for rationale; same N×layer
	// + N×head struct-copy elision on the inverse direction.
	for i := range resultLayers {
		layer := &resultLayers[i]
		layerHeadsSrc := layer.Heads
		headsEnd := headsOffset + len(layerHeadsSrc)
		layerHeads := headsSlab[headsOffset:headsEnd:headsEnd]
		// Per-layer shape clones cut from the shared arena.
		var keyShape, valueShape []int32
		switch {
		case layer.KeyShape == nil:
		case len(layer.KeyShape) == 0:
			keyShape = []int32{}
		default:
			end := int32Offset + len(layer.KeyShape)
			keyShape = int32Slab[int32Offset:end:end]
			copy(keyShape, layer.KeyShape)
			int32Offset = end
		}
		switch {
		case layer.ValueShape == nil:
		case len(layer.ValueShape) == 0:
			valueShape = []int32{}
		default:
			end := int32Offset + len(layer.ValueShape)
			valueShape = int32Slab[int32Offset:end:end]
			copy(valueShape, layer.ValueShape)
			int32Offset = end
		}
		layers[i] = metal.KVLayerSnapshot{
			Layer:              layer.Layer,
			CacheIndex:         layer.CacheIndex,
			CacheMode:          metal.KVCacheMode(layer.CacheMode),
			TurboQuantPayloads: metalTurboQuantPayloads(layer.TurboQuantPayloads),
			KeyDType:           metalKVHeadDType(layer.KeyDType, layer.KeyBytes),
			KeyBytes:           layer.KeyBytes,
			KeyShape:           keyShape,
			ValueDType:         metalKVHeadDType(layer.ValueDType, layer.ValueBytes),
			ValueBytes:         layer.ValueBytes,
			ValueShape:         valueShape,
			Heads:              layerHeads,
		}
		// Native-slab layers never have their per-head float32 read by the
		// restorer (see the sizing-loop note), so pass the source slices
		// through by reference — same ownership contract as KeyBytes above,
		// where the source snapshot already outlives the metal snapshot for
		// the duration of the restore call. Zero copy, zero slab footprint.
		layerNative := kvLayerHasNativeSlab(layer)
		for j := range layerHeadsSrc {
			head := &layerHeadsSrc[j]
			// Allocate per-head Key + Value out of the pre-sized arenas;
			// preserve the prior nil-in -> nil-out / empty-in -> empty-out
			// shape of core.SliceClone so downstream metal sees no
			// behavioural change.
			var headKey []float32
			switch {
			case layerNative:
				headKey = head.Key
			case head.Key == nil:
				// nil in -> nil out
			case len(head.Key) == 0:
				headKey = []float32{}
			default:
				end := keyOffset + len(head.Key)
				headKey = float32Slab[keyOffset:end:end]
				copy(headKey, head.Key)
				keyOffset = end
			}
			var headValue []float32
			switch {
			case layerNative:
				headValue = head.Value
			case head.Value == nil:
			case len(head.Value) == 0:
				headValue = []float32{}
			default:
				end := valueOffset + len(head.Value)
				headValue = float32Slab[valueOffset:end:end]
				copy(headValue, head.Value)
				valueOffset = end
			}
			layerHeads[j] = metal.KVHeadSnapshot{
				Key:        headKey,
				KeyDType:   metalKVHeadDType(head.KeyDType, head.KeyBytes),
				KeyBytes:   head.KeyBytes,
				Value:      headValue,
				ValueDType: metalKVHeadDType(head.ValueDType, head.ValueBytes),
				ValueBytes: head.ValueBytes,
			}
		}
		headsOffset = headsEnd
	}
	// Top-level int32 slices share the same arena as the per-layer shape
	// clones — preserves the same nil-in/empty-in/non-empty semantics
	// core.SliceClone provided so downstream callers see no change.
	var tokens, generated, logitShape []int32
	switch {
	case result.Tokens == nil:
	case len(result.Tokens) == 0:
		tokens = []int32{}
	default:
		end := int32Offset + len(result.Tokens)
		tokens = int32Slab[int32Offset:end:end]
		copy(tokens, result.Tokens)
		int32Offset = end
	}
	switch {
	case result.Generated == nil:
	case len(result.Generated) == 0:
		generated = []int32{}
	default:
		end := int32Offset + len(result.Generated)
		generated = int32Slab[int32Offset:end:end]
		copy(generated, result.Generated)
		int32Offset = end
	}
	switch {
	case result.LogitShape == nil:
	case len(result.LogitShape) == 0:
		logitShape = []int32{}
	default:
		end := int32Offset + len(result.LogitShape)
		logitShape = int32Slab[int32Offset:end:end]
		copy(logitShape, result.LogitShape)
		int32Offset = end
	}
	// Top-level Logits sits in the tail region of the shared float32 slab.
	var topLogits []float32
	switch {
	case result.Logits == nil:
	case len(result.Logits) == 0:
		topLogits = []float32{}
	default:
		end := logitsOffset + len(result.Logits)
		topLogits = float32Slab[logitsOffset:end:end]
		copy(topLogits, result.Logits)
		logitsOffset = end
	}
	return &metal.KVSnapshot{
		Version:       result.Version,
		Architecture:  result.Architecture,
		Tokens:        tokens,
		Generated:     generated,
		TokenOffset:   result.TokenOffset,
		NumLayers:     result.NumLayers,
		NumHeads:      result.NumHeads,
		SeqLen:        result.SeqLen,
		HeadDim:       result.HeadDim,
		NumQueryHeads: result.NumQueryHeads,
		LogitShape:    logitShape,
		Logits:        topLogits,
		Layers:        layers,
	}
}

func toMetalKVSnapshotCaptureOptions(opts kv.CaptureOptions) metal.KVSnapshotCaptureOptions {
	return metal.KVSnapshotCaptureOptions{RawKVOnly: opts.RawKVOnly}
}

func rootKVHeadDType(dtype metal.DType, raw []byte) string {
	if len(raw) == 0 {
		return ""
	}
	// Inline the three KV-supported dtype names to avoid the dtype.String()
	// map lookup. Called per-head inside the KV snapshot clone hot path —
	// thousands of invocations per snapshot.
	switch dtype {
	case metal.DTypeFloat32:
		return "float32"
	case metal.DTypeFloat16:
		return "float16"
	case metal.DTypeBFloat16:
		return "bfloat16"
	default:
		return ""
	}
}

func metalKVHeadDType(dtype string, raw []byte) metal.DType {
	if len(raw) == 0 {
		return 0
	}
	switch dtype {
	case "float32", "F32":
		return metal.DTypeFloat32
	case "float16", "F16":
		return metal.DTypeFloat16
	case "bfloat16", "BF16":
		return metal.DTypeBFloat16
	default:
		return 0
	}
}

// Generate produces a buffered string result.
func (m *Model) Generate(prompt string, opts ...GenerateOption) (string, error) {
	if m == nil || m.model == nil {
		return "", errMLXModelNil
	}
	cfg := applyGenerateOptions(opts)
	filter := parser.NewProcessor(cfg.Thinking, m.hintForParser())
	builder := core.NewBuilder()
	// Pre-grow for the expected output footprint — MaxTokens caps the
	// emitted token stream and 4 bytes/token is a conservative average
	// across ASCII + short BPE pieces, matching the FilterThinkingTokens
	// sizing heuristic in thinking.go. Grow(0) is a no-op when MaxTokens
	// is unset.
	builder.Grow(cfg.MaxTokens * 4)
	for tok := range m.model.Generate(context.Background(), prompt, toMetalGenerateConfig(cfg)) {
		builder.WriteString(filter.Process(tok.Text))
	}
	builder.WriteString(filter.Flush())
	if err := m.model.Err(); err != nil {
		return "", err
	}
	return builder.String(), nil
}

// Chat produces a buffered string result using the model's native chat template.
func (m *Model) Chat(messages []inference.Message, opts ...GenerateOption) (string, error) {
	if m == nil || m.model == nil {
		return "", errMLXModelNil
	}
	cfg := applyGenerateOptions(opts)
	filter := parser.NewProcessor(cfg.Thinking, m.hintForParser())
	// chatMessagesAsMetal is a layout-guarded reinterpret of the input
	// slice — inference.Message and metal.ChatMessage are bit-identical
	// ({Role string; Content string} same field order). The receiving
	// metal.Chat path only reads (it formats the slice into a prompt
	// string and returns); the borrow lifetime is bounded by this call,
	// so dropping the make+per-message copy is sound.
	metalMessages := chatMessagesAsMetal(messages)
	builder := core.NewBuilder()
	// Pre-grow for MaxTokens × 4-byte average — same heuristic as the
	// FilterThinkingTokens decoder and Model.Generate above.
	builder.Grow(cfg.MaxTokens * 4)
	for tok := range m.model.Chat(context.Background(), metalMessages, toMetalGenerateConfig(cfg)) {
		builder.WriteString(filter.Process(tok.Text))
	}
	builder.WriteString(filter.Flush())
	if err := m.model.Err(); err != nil {
		return "", err
	}
	return builder.String(), nil
}

// GenerateChunks produces a buffered string result from streaming prompt chunks.
// Chunked prompts avoid one giant tokenizer call while preserving one logical
// prompt token stream for cache matching and KV capture.
func (m *Model) GenerateChunks(ctx context.Context, chunks iter.Seq[string], opts ...GenerateOption) (string, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if m == nil || m.model == nil {
		return "", errMLXModelNil
	}
	if generator, ok := m.model.(nativeChunkGenerator); ok {
		cfg := applyGenerateOptions(opts)
		filter := parser.NewProcessor(cfg.Thinking, m.hintForParser())
		builder := core.NewBuilder()
		// Same MaxTokens × 4 pre-grow as Generate/Chat above — keeps the
		// chunked path on the same allocation budget as the giant-string
		// path it falls back to.
		builder.Grow(cfg.MaxTokens * 4)
		for tok := range generator.GenerateChunks(ctx, chunks, toMetalGenerateConfig(cfg)) {
			builder.WriteString(filter.Process(tok.Text))
		}
		builder.WriteString(filter.Flush())
		if err := m.model.Err(); err != nil {
			return "", err
		}
		return builder.String(), nil
	}
	return m.Generate(promptChunksToString(chunks), opts...)
}

// WarmPromptCache prefills the exact token-prefix cache for a stable prompt prefix.
func (m *Model) WarmPromptCache(prompt string) error {
	if m == nil || m.model == nil {
		return errMLXModelNil
	}
	warmer, ok := m.model.(nativePromptCacheWarmer)
	if !ok {
		return errMLXPromptCacheWarmUnsupp
	}
	return warmer.WarmPromptCache(context.Background(), prompt)
}

// WarmPromptCacheChunks prefills the exact token-prefix cache from streaming
// prompt chunks without building or tokenizing one giant prompt string.
func (m *Model) WarmPromptCacheChunks(ctx context.Context, chunks iter.Seq[string]) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if m == nil || m.model == nil {
		return errMLXModelNil
	}
	if warmer, ok := m.model.(nativePromptCacheChunkWarmer); ok {
		return warmer.WarmPromptCacheChunks(ctx, chunks)
	}
	return m.WarmPromptCache(promptChunksToString(chunks))
}

// ClearPromptCache drops the exact token-prefix KV cache without unloading the
// model. TRAD comparison runners use this to force a fresh prefill between
// turns while keeping the same loaded weights.
func (m *Model) ClearPromptCache() error {
	if m == nil || m.model == nil {
		return errMLXModelNil
	}
	clearer, ok := m.model.(nativePromptCacheClearer)
	if !ok {
		return errMLXPromptCacheClearUnsupp
	}
	clearer.ClearPromptCache()
	return nil
}

// WarmPromptCacheFromKV installs a captured K/V prefix directly as the model prompt cache.
func (m *Model) WarmPromptCacheFromKV(snapshot *kv.Snapshot) error {
	if m == nil || m.model == nil {
		return errMLXModelNil
	}
	restorer, ok := m.model.(nativePromptCacheKVRestorer)
	if !ok {
		return errMLXKVPromptRestoreUnsupp
	}
	return restorer.RestorePromptCacheFromKV(context.Background(), toMetalKVSnapshot(snapshot))
}

// WarmPromptCacheFromStateBlocks loads the requested State KV prefix blocks and
// installs them directly as the model prompt cache.
func (m *Model) WarmPromptCacheFromStateBlocks(ctx context.Context, store state.Store, bundle *kv.StateBlockBundle, prefixTokens int) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if m == nil || m.model == nil {
		return errMLXModelNil
	}
	if restorer, ok := m.model.(nativePromptCacheKVBlockRestorer); ok {
		source, err := metalKVSnapshotBlockSource(ctx, store, bundle, prefixTokens)
		if err != nil {
			return err
		}
		return restorer.RestorePromptCacheFromKVBlocks(ctx, source)
	}
	snapshot, err := kv.LoadPrefixFromStateBlocks(ctx, store, bundle, prefixTokens)
	if err != nil {
		return err
	}
	restorer, ok := m.model.(nativePromptCacheKVRestorer)
	if !ok {
		return errMLXKVPromptRestoreUnsupp
	}
	return restorer.RestorePromptCacheFromKV(ctx, toMetalKVSnapshot(snapshot))
}

// WarmPromptCacheFromMemvidBlocks loads the requested old memvid-named State
// KV prefix blocks and installs them directly as the model prompt cache.
//
// Deprecated: use WarmPromptCacheFromStateBlocks.
func (m *Model) WarmPromptCacheFromMemvidBlocks(ctx context.Context, store state.Store, bundle *kv.MemvidBlockBundle, prefixTokens int) error {
	return m.WarmPromptCacheFromStateBlocks(ctx, store, bundle, prefixTokens)
}

func metalKVSnapshotBlockSource(ctx context.Context, store state.Store, bundle *kv.StateBlockBundle, prefixTokens int) (metal.KVSnapshotBlockSource, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if store == nil {
		return metal.KVSnapshotBlockSource{}, errMLXStateKVStoreNil
	}
	if err := kv.ValidateStateBlockBundle(bundle); err != nil {
		return metal.KVSnapshotBlockSource{}, err
	}
	if prefixTokens <= 0 {
		prefixTokens = bundle.TokenCount
	}
	if prefixTokens > bundle.TokenCount {
		return metal.KVSnapshotBlockSource{}, errMLXStateKVPrefixExceeds
	}
	blocks := bundle.Blocks
	blockCount, err := metalKVSnapshotBlockSourceCoverage(blocks, prefixTokens)
	if err != nil {
		return metal.KVSnapshotBlockSource{}, err
	}
	source := metal.KVSnapshotBlockSource{
		TokenCount:   bundle.TokenCount,
		PrefixTokens: prefixTokens,
		BlockCount:   blockCount,
	}
	// Hoist invariants out of the per-block closure. KVEncoding is bundle-
	// scoped — checking it once at construction lets each Load call use
	// the captured loadOpts directly without re-branching on every block.
	loadOpts := kv.LoadOptions{}
	if bundle.KVEncoding == kv.EncodingNative {
		loadOpts.RawKVOnly = true
	}
	source.Load = func(loadCtx context.Context, index int) (metal.KVSnapshotBlock, error) {
		if loadCtx == nil {
			loadCtx = ctx
		}
		if index < 0 || index >= blockCount {
			return metal.KVSnapshotBlock{}, errMLXStateKVBlockOutOfRange
		}
		ref := &blocks[index]
		block, err := kv.LoadStateBlockWithOptions(loadCtx, store, *ref, loadOpts)
		if err != nil {
			return metal.KVSnapshotBlock{}, err
		}
		if block.TokenStart != ref.TokenStart || block.TokenCount != ref.TokenCount {
			return metal.KVSnapshotBlock{}, errMLXStateKVBlockMetaMismatch
		}
		snapshot := block.Snapshot
		if snapshot == nil {
			return metal.KVSnapshotBlock{}, errMLXStateKVBlockSnapshotNil
		}
		if block.TokenStart+block.TokenCount > prefixTokens {
			trimTokens := prefixTokens - block.TokenStart
			if trimTokens <= 0 {
				return metal.KVSnapshotBlock{}, errMLXStateKVPrefixInvalidTrim
			}
			baseOffset := max(kv.EffectiveTokenOffset(snapshot)-kv.EffectiveSeqLen(snapshot), 0)
			trimmed, trimErr := snapshot.SliceBlock(0, trimTokens, baseOffset, false)
			if trimErr != nil {
				return metal.KVSnapshotBlock{}, trimErr
			}
			snapshot = trimmed
			block.TokenCount = trimTokens
		}
		if block.TokenStart+block.TokenCount < bundle.TokenCount {
			kv.ClearTerminalState(snapshot)
		}
		return metal.KVSnapshotBlock{
			Index:      index,
			TokenStart: block.TokenStart,
			TokenCount: block.TokenCount,
			Snapshot:   toMetalKVSnapshot(snapshot),
		}, nil
	}
	return source, nil
}

func metalKVSnapshotBlockSourceCoverage(blocks []kv.StateBlockRef, prefixTokens int) (int, error) {
	if len(blocks) == 0 {
		return 0, errMLXStateKVPrefixNoCovering
	}
	nextStart := 0
	blockCount := 0
	for i := range blocks {
		ref := &blocks[i]
		if ref.TokenStart >= prefixTokens {
			break
		}
		if ref.Index != i || ref.TokenStart != nextStart || ref.TokenCount <= 0 {
			return 0, errMLXStateKVBlockMetaMismatch
		}
		nextStart += ref.TokenCount
		blockCount++
		if nextStart >= prefixTokens {
			break
		}
	}
	if blockCount == 0 || nextStart < prefixTokens {
		return 0, errMLXStateKVPrefixNoCovering
	}
	return blockCount, nil
}

// GenerateStream streams tokens through a channel until generation completes or ctx is cancelled.
func (m *Model) GenerateStream(ctx context.Context, prompt string, opts ...GenerateOption) <-chan Token {
	if m == nil || m.model == nil {
		return closedTokenChan
	}
	out := make(chan Token)
	go func() {
		defer close(out)
		if ctx == nil {
			ctx = context.Background()
		}
		cfg := applyGenerateOptions(opts)
		filter := parser.NewProcessor(cfg.Thinking, m.hintForParser())
		for tok := range m.model.Generate(ctx, prompt, toMetalGenerateConfig(cfg)) {
			text := filter.Process(tok.Text)
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

// GenerateChunksStream streams tokens from bounded prompt chunks without
// building or tokenizing one giant prompt string.
func (m *Model) GenerateChunksStream(ctx context.Context, chunks iter.Seq[string], opts ...GenerateOption) <-chan Token {
	if m == nil || m.model == nil {
		return closedTokenChan
	}
	out := make(chan Token)
	go func() {
		defer close(out)
		if ctx == nil {
			ctx = context.Background()
		}
		cfg := applyGenerateOptions(opts)
		filter := parser.NewProcessor(cfg.Thinking, m.hintForParser())
		if generator, ok := m.model.(nativeChunkGenerator); ok {
			for tok := range generator.GenerateChunks(ctx, chunks, toMetalGenerateConfig(cfg)) {
				text := filter.Process(tok.Text)
				if text == "" {
					continue
				}
				select {
				case out <- Token{ID: tok.ID, Value: text, Text: text}:
				case <-ctx.Done():
					return
				}
			}
		} else {
			for tok := range m.model.Generate(ctx, promptChunksToString(chunks), toMetalGenerateConfig(cfg)) {
				text := filter.Process(tok.Text)
				if text == "" {
					continue
				}
				select {
				case out <- Token{ID: tok.ID, Value: text, Text: text}:
				case <-ctx.Done():
					return
				}
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

// ChatChunksStream streams chat tokens through the native template while
// feeding long message content as bounded prompt chunks.
func (m *Model) ChatChunksStream(ctx context.Context, messages []inference.Message, chunkBytes int, opts ...GenerateOption) <-chan Token {
	if m == nil || m.model == nil {
		return closedTokenChan
	}
	out := make(chan Token)
	go func() {
		defer close(out)
		if ctx == nil {
			ctx = context.Background()
		}
		cfg := applyGenerateOptions(opts)
		filter := parser.NewProcessor(cfg.Thinking, m.hintForParser())
		// chatMessagesAsMetal reinterprets in place — see Model.Chat for
		// the layout-guard rationale. Borrow lifetime ends with this
		// call into the chat-chunk generator path.
		metalMessages := chatMessagesAsMetal(messages)
		if generator, ok := m.model.(nativeChatChunkGenerator); ok {
			for tok := range generator.ChatChunks(ctx, metalMessages, chunkBytes, toMetalGenerateConfig(cfg)) {
				text := filter.Process(tok.Text)
				if text == "" {
					continue
				}
				select {
				case out <- Token{ID: tok.ID, Value: text, Text: text}:
				case <-ctx.Done():
					return
				}
			}
		} else {
			for tok := range m.model.Chat(ctx, metalMessages, toMetalGenerateConfig(cfg)) {
				text := filter.Process(tok.Text)
				if text == "" {
					continue
				}
				select {
				case out <- Token{ID: tok.ID, Value: text, Text: text}:
				case <-ctx.Done():
					return
				}
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

// ChatStream streams chat tokens through a channel until generation completes or ctx is cancelled.
func (m *Model) ChatStream(ctx context.Context, messages []inference.Message, opts ...GenerateOption) <-chan Token {
	if m == nil || m.model == nil {
		return closedTokenChan
	}
	out := make(chan Token)
	go func() {
		defer close(out)
		if ctx == nil {
			ctx = context.Background()
		}
		cfg := applyGenerateOptions(opts)
		filter := parser.NewProcessor(cfg.Thinking, m.hintForParser())
		// chatMessagesAsMetal reinterprets in place — see Model.Chat for
		// the layout-guard rationale. Borrow lifetime ends with the
		// streaming m.model.Chat call drained below.
		metalMessages := chatMessagesAsMetal(messages)
		for tok := range m.model.Chat(ctx, metalMessages, toMetalGenerateConfig(cfg)) {
			text := filter.Process(tok.Text)
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

// Classify runs batched prefill-only inference over multiple prompts.
func (m *Model) Classify(prompts []string, opts ...GenerateOption) ([]ClassifyResult, error) {
	if m == nil || m.model == nil {
		return nil, errMLXModelNil
	}
	cfg := applyGenerateOptions(opts)
	results, err := m.model.Classify(context.Background(), prompts, toMetalGenerateConfig(cfg), cfg.ReturnLogits)
	if err != nil {
		return nil, err
	}
	return toRootClassifyResults(results), nil
}

// BatchGenerate runs autoregressive generation for multiple prompts at once.
func (m *Model) BatchGenerate(prompts []string, opts ...GenerateOption) ([]BatchResult, error) {
	if m == nil || m.model == nil {
		return nil, errMLXModelNil
	}
	results, err := m.model.BatchGenerate(context.Background(), prompts, toMetalGenerateConfig(applyGenerateOptions(opts)))
	if err != nil {
		return nil, err
	}
	return toRootBatchResults(results), nil
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
	gemma4SlidingWindow := info.Gemma4SlidingWindow
	if gemma4SlidingWindow == 0 && m.cfg.Gemma4SlidingWindow > 0 {
		gemma4SlidingWindow = m.cfg.Gemma4SlidingWindow
	}
	architecture := info.Architecture
	vocabSize := info.VocabSize
	numLayers := info.NumLayers
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
		Architecture:         architecture,
		VocabSize:            vocabSize,
		NumLayers:            numLayers,
		HiddenSize:           hiddenSize,
		QuantBits:            quantBits,
		QuantGroup:           quantGroup,
		ContextLength:        contextLength,
		Gemma4SlidingWindow:  gemma4SlidingWindow,
		ParallelSlots:        m.cfg.ParallelSlots,
		PromptCache:          m.cfg.PromptCache,
		PromptCacheMinTokens: m.cfg.PromptCacheMinTokens,
		CachePolicy:          m.cfg.CachePolicy,
		CacheMode:            m.cfg.CacheMode,
		BatchSize:            m.cfg.BatchSize,
		PrefillChunkSize:     m.cfg.PrefillChunkSize,
		ExpectedQuantization: m.cfg.ExpectedQuantization,
		MemoryLimitBytes:     m.cfg.MemoryLimitBytes,
		CacheLimitBytes:      m.cfg.CacheLimitBytes,
		WiredLimitBytes:      m.cfg.WiredLimitBytes,
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
		result, err := snapshotter.CaptureKVWithOptions(context.Background(), prompt, toMetalKVSnapshotCaptureOptions(opts))
		if err != nil {
			return nil, err
		}
		snapshot := toRootKVSnapshot(result)
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
	snapshot := toRootKVSnapshot(result)
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
		result, err := snapshotter.CaptureKVChunksWithOptions(ctx, chunks, toMetalKVSnapshotCaptureOptions(opts))
		if err != nil {
			return nil, err
		}
		snapshot := toRootKVSnapshot(result)
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
		snapshot := toRootKVSnapshot(result)
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

// NewLoRA applies a LoRA adapter to a loaded model.
func NewLoRA(model *Model, cfg *LoRAConfig) *LoRAAdapter {
	if model == nil || model.model == nil {
		return nil
	}
	mcfg := DefaultLoRAConfig()
	if cfg != nil {
		mcfg = *cfg
	}
	adapter := model.model.ApplyLoRA(toMetalLoRAConfig(mcfg))
	// ApplyLoRA mutates the native model's adapter identity — refresh the
	// cached parserHint so the next Generate / Chat picks up the new
	// adapter name in its parser dispatch without re-reading m.model.Info()
	// per call.
	model.refreshParserHint()
	return adapter
}

// LoadLoRA loads a saved adapter package into a loaded model and returns it.
func (m *Model) LoadLoRA(path string) (*LoRAAdapter, error) {
	if m == nil || m.model == nil {
		return nil, errMLXModelNil
	}
	info, err := lora.InspectAdapter(path)
	if err != nil {
		return nil, err
	}
	loader, ok := m.model.(nativeLoRALoader)
	if !ok {
		return nil, errMLXLoRALoadUnsupp
	}
	adapter, err := loader.LoadLoRA(path)
	if err != nil {
		return nil, err
	}
	m.adapterInfo = info
	m.cfg.AdapterPath = path
	// Adapter identity changed — refresh the cached parserHint so the next
	// Generate / Chat picks up the new adapter name without paying for an
	// m.model.Info() fan-out per call.
	m.refreshParserHint()
	return adapter, nil
}

// UnloadLoRA removes the active inference adapter when the backend supports it.
func (m *Model) UnloadLoRA() error {
	if m == nil || m.model == nil {
		return errMLXModelNil
	}
	if m.adapterInfo.IsEmpty() {
		return nil
	}
	unloader, ok := m.model.(nativeLoRAUnloader)
	if !ok {
		return errMLXLoRAUnloadUnsupp
	}
	if err := unloader.UnloadLoRA(); err != nil {
		return err
	}
	m.adapterInfo = lora.AdapterInfo{}
	m.cfg.AdapterPath = ""
	// Adapter cleared — refresh the cached parserHint so the next Generate
	// / Chat reads the post-unload adapter name (may fall back to the
	// native model's AdapterInfo.Name) without re-entering m.model.Info()
	// per call.
	m.refreshParserHint()
	return nil
}

// SwapLoRA replaces the active inference adapter with another adapter package.
func (m *Model) SwapLoRA(path string) (*LoRAAdapter, error) {
	if err := m.UnloadLoRA(); err != nil {
		return nil, err
	}
	return m.LoadLoRA(path)
}

// MergeLoRA returns the current model with the adapter applied in-place.
func (m *Model) MergeLoRA(adapter *LoRAAdapter) *Model {
	if adapter == nil {
		return m
	}
	adapter.Merge()
	return m
}

// MatMul returns the matrix product of a and b.
func MatMul(a, b *Array) *Array { return metal.Matmul(a, b) }

// Add returns element-wise a + b.
func Add(a, b *Array) *Array { return metal.Add(a, b) }

// Mul returns element-wise a * b.
func Mul(a, b *Array) *Array { return metal.Mul(a, b) }

// Softmax returns softmax along the last axis.
func Softmax(a *Array) *Array { return metal.Softmax(a) }

// Slice extracts a sub-array along a single axis.
func Slice(a *Array, start, end, axis any) *Array {
	return metal.SliceAxis(
		a,
		normalizeRootIntArg("axis", axis),
		normalizeRootInt32Arg("start", start),
		normalizeRootInt32Arg("end", end),
	)
}

// Reshape returns a view with the given shape.
func Reshape(a *Array, shape ...any) *Array {
	return metal.Reshape(a, normalizeRootShapeArgs(shape)...)
}

// VJP computes the vector-Jacobian product.
func VJP(fn func([]*Array) []*Array, primals []*Array, cotangents []*Array) (outputs []*Array, vjps []*Array, err error) {
	return metal.VJP(fn, primals, cotangents)
}

// JVP computes the Jacobian-vector product.
func JVP(fn func([]*Array) []*Array, primals []*Array, tangents []*Array) (outputs []*Array, jvps []*Array, err error) {
	return metal.JVP(fn, primals, tangents)
}
