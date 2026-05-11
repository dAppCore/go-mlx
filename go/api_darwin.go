// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && !nomlx

package mlx

import (
	"context"
	"iter"

	core "dappco.re/go"
	"dappco.re/go/mlx/gguf"
	"dappco.re/go/inference/parser"
	memvid "dappco.re/go/inference/state"
	"dappco.re/go/mlx/kv"
	"dappco.re/go/mlx/internal/metal"
	"dappco.re/go/mlx/lora"
)

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
}

var loadNativeModel = func(modelPath string, cfg metal.LoadConfig) (nativeModel, error) {
	return metal.LoadAndInit(modelPath, cfg)
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

// LoadModel loads a model directly through go-mlx without going through go-inference.
func LoadModel(modelPath string, opts ...LoadOption) (*Model, error) {
	cfg, err := normalizeLoadConfig(applyLoadOptions(opts))
	if err != nil {
		return nil, err
	}

	resolvedPath := modelPath
	resolvedAdapterPath := cfg.AdapterPath
	var adapterInfo lora.AdapterInfo
	cleanup := func() error { return nil }
	if cfg.Medium != nil {
		resolvedPath, cleanup, err = stageModelFromMedium(cfg.Medium, modelPath)
		if err != nil {
			return nil, err
		}
		if cfg.AdapterPath != "" {
			var adapterCleanup func() error
			resolvedAdapterPath, adapterCleanup, err = stagePathFromMedium(cfg.Medium, cfg.AdapterPath)
			if err != nil {
				if cleanupErr := cleanup(); cleanupErr != nil {
					return nil, core.ErrorJoin(err, cleanupErr)
				}
				return nil, err
			}
			appendCleanup(&cleanup, adapterCleanup)
		}
	}
	cfg = applyMemoryPlanToLoadConfig(resolvedPath, cfg)
	if resolvedAdapterPath != "" {
		adapterInfo, err = lora.Inspect(resolvedAdapterPath, cfg.AdapterPath)
		if err != nil {
			if cleanupErr := cleanup(); cleanupErr != nil {
				return nil, core.ErrorJoin(err, cleanupErr)
			}
			return nil, err
		}
	}

	native, err := loadNativeModel(resolvedPath, metal.LoadConfig{
		ContextLen:           cfg.ContextLength,
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
		if cleanupErr := cleanup(); cleanupErr != nil {
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
		if cleanupErr := cleanup(); cleanupErr != nil {
			quantErr = core.ErrorJoin(quantErr, cleanupErr)
		}
		return nil, quantErr
	}

	return &Model{
		model:       native,
		cfg:         cfg,
		tok:         &Tokenizer{tok: native.Tokenizer()},
		gguf:        ggufInfo,
		adapterInfo: adapterInfo,
		cleanup:     cleanup,
	}, nil
}

func toMetalGenerateConfig(cfg GenerateConfig) metal.GenerateConfig {
	return metal.GenerateConfig{
		MaxTokens:     cfg.MaxTokens,
		Temperature:   cfg.Temperature,
		TopK:          cfg.TopK,
		TopP:          cfg.TopP,
		MinP:          cfg.MinP,
		StopTokens:    cfg.StopTokens,
		RepeatPenalty: cfg.RepeatPenalty,
		ProbeSink:     toMetalProbeSink(cfg.ProbeSink),
	}
}

func toMetalProbeSink(sink ProbeSink) metal.ProbeSink {
	if sink == nil {
		return nil
	}
	return metal.ProbeSinkFunc(func(event metal.ProbeEvent) {
		sink.EmitProbe(toRootProbeEvent(event))
	})
}

func toRootProbeEvent(event metal.ProbeEvent) ProbeEvent {
	out := ProbeEvent{
		Kind:  ProbeEventKind(event.Kind),
		Phase: ProbePhase(event.Phase),
		Step:  event.Step,
		Meta:  cloneMetalProbeMeta(event.Meta),
	}
	if event.Token != nil {
		token := *event.Token
		out.Token = &ProbeToken{
			ID:              token.ID,
			Text:            token.Text,
			PromptTokens:    token.PromptTokens,
			GeneratedTokens: token.GeneratedTokens,
		}
	}
	if event.Logits != nil {
		logits := *event.Logits
		out.Logits = &ProbeLogits{
			Shape:      append([]int32(nil), logits.Shape...),
			VocabSize:  logits.VocabSize,
			MaxTokenID: logits.MaxTokenID,
			MaxLogit:   logits.MaxLogit,
			MinTokenID: logits.MinTokenID,
			MinLogit:   logits.MinLogit,
			MeanLogit:  logits.MeanLogit,
			Top:        toRootProbeLogits(logits.Top),
			Values:     append([]float32(nil), logits.Values...),
			Meta:       cloneMetalProbeMeta(logits.Meta),
		}
	}
	if event.Entropy != nil {
		entropy := *event.Entropy
		out.Entropy = &ProbeEntropy{Value: entropy.Value, Unit: entropy.Unit}
	}
	if event.SelectedHeads != nil {
		heads := *event.SelectedHeads
		out.SelectedHeads = &ProbeHeadSelection{
			Layer:  heads.Layer,
			Heads:  append([]int(nil), heads.Heads...),
			Scores: append([]float64(nil), heads.Scores...),
		}
	}
	if event.LayerCoherence != nil {
		coherence := *event.LayerCoherence
		out.LayerCoherence = &ProbeLayerCoherence{
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
		router := *event.RouterDecision
		out.RouterDecision = &ProbeRouterDecision{
			Layer:       router.Layer,
			TokenID:     router.TokenID,
			ExpertIDs:   append([]int(nil), router.ExpertIDs...),
			Weights:     append([]float32(nil), router.Weights...),
			Temperature: router.Temperature,
		}
	}
	if event.Residual != nil {
		residual := *event.Residual
		out.Residual = &ProbeResidualSummary{
			Layer:    residual.Layer,
			Mean:     residual.Mean,
			Variance: residual.Variance,
			RMS:      residual.RMS,
			L2Norm:   residual.L2Norm,
			MaxAbs:   residual.MaxAbs,
		}
	}
	if event.Cache != nil {
		cache := *event.Cache
		out.Cache = &ProbeCachePressure{
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
		memory := *event.Memory
		out.Memory = &ProbeMemoryPressure{
			ActiveBytes: memory.ActiveBytes,
			PeakBytes:   memory.PeakBytes,
			CacheBytes:  memory.CacheBytes,
		}
	}
	if event.Training != nil {
		training := *event.Training
		out.Training = &ProbeTraining{
			Step:         training.Step,
			Epoch:        training.Epoch,
			Loss:         training.Loss,
			LearningRate: training.LearningRate,
			GradNorm:     training.GradNorm,
		}
	}
	return out
}

func toRootProbeLogits(logits []metal.ProbeLogit) []ProbeLogit {
	if len(logits) == 0 {
		return nil
	}
	out := make([]ProbeLogit, len(logits))
	for i, logit := range logits {
		out[i] = ProbeLogit{
			TokenID:     logit.TokenID,
			Logit:       logit.Logit,
			Probability: logit.Probability,
		}
	}
	return out
}

func cloneMetalProbeMeta(meta map[string]string) map[string]string {
	if len(meta) == 0 {
		return nil
	}
	out := make(map[string]string, len(meta))
	for key, value := range meta {
		out[key] = value
	}
	return out
}

func toRootMetrics(metrics metal.Metrics) Metrics {
	return Metrics{
		PromptTokens:               metrics.PromptTokens,
		GeneratedTokens:            metrics.GeneratedTokens,
		PrefillDuration:            metrics.PrefillDuration,
		DecodeDuration:             metrics.DecodeDuration,
		TotalDuration:              metrics.TotalDuration,
		PrefillTokensPerSec:        metrics.PrefillTokensPerSec,
		DecodeTokensPerSec:         metrics.DecodeTokensPerSec,
		PeakMemoryBytes:            metrics.PeakMemoryBytes,
		ActiveMemoryBytes:          metrics.ActiveMemoryBytes,
		PromptCacheHits:            metrics.PromptCacheHits,
		PromptCacheMisses:          metrics.PromptCacheMisses,
		PromptCacheHitTokens:       metrics.PromptCacheHitTokens,
		PromptCacheMissTokens:      metrics.PromptCacheMissTokens,
		PromptCacheRestoreDuration: metrics.PromptCacheRestoreDuration,
		Adapter:                    toRootAdapterInfo(metrics.Adapter),
	}
}

func toRootAdapterInfo(info metal.AdapterInfo) lora.AdapterInfo {
	return lora.AdapterInfo{
		Name:       info.Name,
		Path:       info.Path,
		Hash:       info.Hash,
		Rank:       info.Rank,
		Alpha:      info.Alpha,
		Scale:      info.Scale,
		TargetKeys: append([]string(nil), info.TargetKeys...),
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
	for i, result := range results {
		out[i] = ClassifyResult{
			Token:  toRootToken(result.Token),
			Logits: append([]float32(nil), result.Logits...),
		}
	}
	return out
}

func toRootBatchResults(results []metal.BatchResult) []BatchResult {
	if len(results) == 0 {
		return nil
	}
	out := make([]BatchResult, len(results))
	for i, result := range results {
		tokens := make([]Token, len(result.Tokens))
		for j, token := range result.Tokens {
			tokens[j] = toRootToken(token)
		}
		out[i] = BatchResult{
			Tokens: tokens,
			Err:    result.Err,
		}
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
	layers := make([]kv.LayerSnapshot, len(result.Layers))
	for i, layer := range result.Layers {
		layers[i] = kv.LayerSnapshot{
			Layer:      layer.Layer,
			CacheIndex: layer.CacheIndex,
			Heads:      make([]kv.HeadSnapshot, len(layer.Heads)),
		}
		for j, head := range layer.Heads {
			layers[i].Heads[j] = kv.HeadSnapshot{
				Key:        append([]float32(nil), head.Key...),
				KeyDType:   rootKVHeadDType(head.KeyDType, head.KeyBytes),
				KeyBytes:   append([]byte(nil), head.KeyBytes...),
				Value:      append([]float32(nil), head.Value...),
				ValueDType: rootKVHeadDType(head.ValueDType, head.ValueBytes),
				ValueBytes: append([]byte(nil), head.ValueBytes...),
			}
		}
	}
	return &kv.Snapshot{
		Version:       result.Version,
		Architecture:  result.Architecture,
		Tokens:        append([]int32(nil), result.Tokens...),
		Generated:     append([]int32(nil), result.Generated...),
		TokenOffset:   result.TokenOffset,
		NumLayers:     result.NumLayers,
		NumHeads:      result.NumHeads,
		SeqLen:        result.SeqLen,
		HeadDim:       result.HeadDim,
		NumQueryHeads: result.NumQueryHeads,
		LogitShape:    append([]int32(nil), result.LogitShape...),
		Logits:        append([]float32(nil), result.Logits...),
		Layers:        layers,
	}
}

func toMetalKVSnapshot(result *kv.Snapshot) *metal.KVSnapshot {
	if result == nil {
		return nil
	}
	layers := make([]metal.KVLayerSnapshot, len(result.Layers))
	for i, layer := range result.Layers {
		layers[i] = metal.KVLayerSnapshot{
			Layer:      layer.Layer,
			CacheIndex: layer.CacheIndex,
			Heads:      make([]metal.KVHeadSnapshot, len(layer.Heads)),
		}
		for j, head := range layer.Heads {
			layers[i].Heads[j] = metal.KVHeadSnapshot{
				Key:        append([]float32(nil), head.Key...),
				KeyDType:   metalKVHeadDType(head.KeyDType, head.KeyBytes),
				KeyBytes:   append([]byte(nil), head.KeyBytes...),
				Value:      append([]float32(nil), head.Value...),
				ValueDType: metalKVHeadDType(head.ValueDType, head.ValueBytes),
				ValueBytes: append([]byte(nil), head.ValueBytes...),
			}
		}
	}
	return &metal.KVSnapshot{
		Version:       result.Version,
		Architecture:  result.Architecture,
		Tokens:        append([]int32(nil), result.Tokens...),
		Generated:     append([]int32(nil), result.Generated...),
		TokenOffset:   result.TokenOffset,
		NumLayers:     result.NumLayers,
		NumHeads:      result.NumHeads,
		SeqLen:        result.SeqLen,
		HeadDim:       result.HeadDim,
		NumQueryHeads: result.NumQueryHeads,
		LogitShape:    append([]int32(nil), result.LogitShape...),
		Logits:        append([]float32(nil), result.Logits...),
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
	switch dtype {
	case metal.DTypeFloat32, metal.DTypeFloat16, metal.DTypeBFloat16:
		return dtype.String()
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
		return "", core.NewError("mlx: model is nil")
	}
	cfg := applyGenerateOptions(opts)
	filter := parser.NewProcessor(cfg.Thinking, parserHint(m.Info()))
	builder := core.NewBuilder()
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
func (m *Model) Chat(messages []Message, opts ...GenerateOption) (string, error) {
	if m == nil || m.model == nil {
		return "", core.NewError("mlx: model is nil")
	}
	cfg := applyGenerateOptions(opts)
	filter := parser.NewProcessor(cfg.Thinking, parserHint(m.Info()))
	metalMessages := make([]metal.ChatMessage, len(messages))
	for i, msg := range messages {
		metalMessages[i] = metal.ChatMessage{Role: msg.Role, Content: msg.Content}
	}
	builder := core.NewBuilder()
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
		return "", core.NewError("mlx: model is nil")
	}
	if generator, ok := m.model.(nativeChunkGenerator); ok {
		cfg := applyGenerateOptions(opts)
		filter := parser.NewProcessor(cfg.Thinking, parserHint(m.Info()))
		builder := core.NewBuilder()
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
		return core.NewError("mlx: model is nil")
	}
	warmer, ok := m.model.(nativePromptCacheWarmer)
	if !ok {
		return core.NewError("mlx: native model does not support prompt cache warming")
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
		return core.NewError("mlx: model is nil")
	}
	if warmer, ok := m.model.(nativePromptCacheChunkWarmer); ok {
		return warmer.WarmPromptCacheChunks(ctx, chunks)
	}
	return m.WarmPromptCache(promptChunksToString(chunks))
}

// WarmPromptCacheFromKV installs a captured K/V prefix directly as the model prompt cache.
func (m *Model) WarmPromptCacheFromKV(snapshot *kv.Snapshot) error {
	if m == nil || m.model == nil {
		return core.NewError("mlx: model is nil")
	}
	restorer, ok := m.model.(nativePromptCacheKVRestorer)
	if !ok {
		return core.NewError("mlx: native model does not support KV prompt cache restore")
	}
	return restorer.RestorePromptCacheFromKV(context.Background(), toMetalKVSnapshot(snapshot))
}

// WarmPromptCacheFromMemvidBlocks loads the requested memvid KV prefix blocks and
// installs them directly as the model prompt cache.
func (m *Model) WarmPromptCacheFromMemvidBlocks(ctx context.Context, store memvid.Store, bundle *kv.MemvidBlockBundle, prefixTokens int) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if m == nil || m.model == nil {
		return core.NewError("mlx: model is nil")
	}
	if restorer, ok := m.model.(nativePromptCacheKVBlockRestorer); ok {
		source, err := metalKVSnapshotBlockSource(ctx, store, bundle, prefixTokens)
		if err != nil {
			return err
		}
		return restorer.RestorePromptCacheFromKVBlocks(ctx, source)
	}
	snapshot, err := kv.LoadPrefixFromMemvidBlocks(ctx, store, bundle, prefixTokens)
	if err != nil {
		return err
	}
	restorer, ok := m.model.(nativePromptCacheKVRestorer)
	if !ok {
		return core.NewError("mlx: native model does not support KV prompt cache restore")
	}
	return restorer.RestorePromptCacheFromKV(ctx, toMetalKVSnapshot(snapshot))
}

func metalKVSnapshotBlockSource(ctx context.Context, store memvid.Store, bundle *kv.MemvidBlockBundle, prefixTokens int) (metal.KVSnapshotBlockSource, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if store == nil {
		return metal.KVSnapshotBlockSource{}, core.NewError("mlx: memvid store is nil")
	}
	if err := kv.ValidateMemvidBlockBundle(bundle); err != nil {
		return metal.KVSnapshotBlockSource{}, err
	}
	if prefixTokens <= 0 {
		prefixTokens = bundle.TokenCount
	}
	if prefixTokens > bundle.TokenCount {
		return metal.KVSnapshotBlockSource{}, core.NewError("mlx: memvid KV prefix exceeds bundle token count")
	}
	refs := make([]kv.MemvidBlockRef, 0, len(bundle.Blocks))
	for _, ref := range bundle.Blocks {
		if ref.TokenStart >= prefixTokens {
			break
		}
		refs = append(refs, ref)
		if ref.TokenStart+ref.TokenCount >= prefixTokens {
			break
		}
	}
	if len(refs) == 0 {
		return metal.KVSnapshotBlockSource{}, core.NewError("mlx: memvid KV prefix has no covering blocks")
	}
	source := metal.KVSnapshotBlockSource{
		TokenCount:   bundle.TokenCount,
		PrefixTokens: prefixTokens,
		BlockCount:   len(refs),
	}
	source.Load = func(loadCtx context.Context, index int) (metal.KVSnapshotBlock, error) {
		if loadCtx == nil {
			loadCtx = ctx
		}
		if index < 0 || index >= len(refs) {
			return metal.KVSnapshotBlock{}, core.NewError("mlx: memvid KV block index is out of range")
		}
		ref := refs[index]
		loadOpts := kv.LoadOptions{}
		if bundle.KVEncoding == kv.EncodingNative {
			loadOpts.RawKVOnly = true
		}
		block, err := kv.LoadMemvidBlockWithOptions(loadCtx, store, ref, loadOpts)
		if err != nil {
			return metal.KVSnapshotBlock{}, err
		}
		if block.TokenStart != ref.TokenStart || block.TokenCount != ref.TokenCount {
			return metal.KVSnapshotBlock{}, core.NewError("mlx: memvid KV block metadata mismatch")
		}
		snapshot := block.Snapshot
		if snapshot == nil {
			return metal.KVSnapshotBlock{}, core.NewError("mlx: memvid KV block snapshot is nil")
		}
		if block.TokenStart+block.TokenCount > prefixTokens {
			trimTokens := prefixTokens - block.TokenStart
			if trimTokens <= 0 {
				return metal.KVSnapshotBlock{}, core.NewError("mlx: memvid KV prefix has invalid trim range")
			}
			baseOffset := kv.EffectiveTokenOffset(snapshot) - kv.EffectiveSeqLen(snapshot)
			if baseOffset < 0 {
				baseOffset = 0
			}
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

// GenerateStream streams tokens through a channel until generation completes or ctx is cancelled.
func (m *Model) GenerateStream(ctx context.Context, prompt string, opts ...GenerateOption) <-chan Token {
	out := make(chan Token)
	go func() {
		defer close(out)
		if m == nil || m.model == nil {
			return
		}
		if ctx == nil {
			ctx = context.Background()
		}
		cfg := applyGenerateOptions(opts)
		filter := parser.NewProcessor(cfg.Thinking, parserHint(m.Info()))
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

// ChatStream streams chat tokens through a channel until generation completes or ctx is cancelled.
func (m *Model) ChatStream(ctx context.Context, messages []Message, opts ...GenerateOption) <-chan Token {
	out := make(chan Token)
	go func() {
		defer close(out)
		if m == nil || m.model == nil {
			return
		}
		if ctx == nil {
			ctx = context.Background()
		}
		cfg := applyGenerateOptions(opts)
		filter := parser.NewProcessor(cfg.Thinking, parserHint(m.Info()))
		metalMessages := make([]metal.ChatMessage, len(messages))
		for i, msg := range messages {
			metalMessages[i] = metal.ChatMessage{Role: msg.Role, Content: msg.Content}
		}
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
		return nil, core.NewError("mlx: model is nil")
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
		return nil, core.NewError("mlx: model is nil")
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
		Architecture:  architecture,
		VocabSize:     vocabSize,
		NumLayers:     numLayers,
		HiddenSize:    hiddenSize,
		QuantBits:     quantBits,
		QuantGroup:    quantGroup,
		ContextLength: contextLength,
		Adapter:       m.Adapter(),
	}
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
		return nil, core.NewError("mlx: model is nil")
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
		return nil, core.NewError("mlx: model is nil")
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
		return nil, core.NewError("mlx: native model does not support KV capture")
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
		return nil, core.NewError("mlx: model is nil")
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
	builder := core.NewBuilder()
	if chunks == nil {
		return ""
	}
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
	return model.model.ApplyLoRA(toMetalLoRAConfig(mcfg))
}

// LoadLoRA loads a saved adapter package into a loaded model and returns it.
func (m *Model) LoadLoRA(path string) (*LoRAAdapter, error) {
	if m == nil || m.model == nil {
		return nil, core.NewError("mlx: model is nil")
	}
	info, err := lora.InspectAdapter(path)
	if err != nil {
		return nil, err
	}
	loader, ok := m.model.(nativeLoRALoader)
	if !ok {
		return nil, core.NewError("mlx: native model does not support LoRA loading")
	}
	adapter, err := loader.LoadLoRA(path)
	if err != nil {
		return nil, err
	}
	m.adapterInfo = info
	m.cfg.AdapterPath = path
	return adapter, nil
}

// UnloadLoRA removes the active inference adapter when the backend supports it.
func (m *Model) UnloadLoRA() error {
	if m == nil || m.model == nil {
		return core.NewError("mlx: model is nil")
	}
	if m.adapterInfo.IsEmpty() {
		return nil
	}
	unloader, ok := m.model.(nativeLoRAUnloader)
	if !ok {
		return core.NewError("mlx: native model does not support LoRA unloading")
	}
	if err := unloader.UnloadLoRA(); err != nil {
		return err
	}
	m.adapterInfo = lora.AdapterInfo{}
	m.cfg.AdapterPath = ""
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
