// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"dappco.re/go/mlx/blockcache"
	"iter"
	"sync"

	"dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/scheduler"
	"dappco.re/go/mlx/pkg/metal"
)

func init() {
	inference.Register(&metalbackend{})
}

// MetalAvailable reports whether native Metal inference is available.
//
//	if mlx.MetalAvailable() { /* run on GPU */ }
func MetalAvailable() bool { return metal.MetalAvailable() }

// Available reports whether native MLX support is available in this build.
func Available() bool { return MetalAvailable() }

// SetCacheLimit sets the Metal memory cache limit. Returns the previous value.
//
//	mlx.SetCacheLimit(4 << 30) // 4 GB cache limit
func SetCacheLimit(limit uint64) uint64 { return metal.SetCacheLimit(limit) }

// SetMemoryLimit sets the Metal memory hard limit. Returns the previous value.
//
//	mlx.SetMemoryLimit(32 << 30) // 32 GB hard limit
func SetMemoryLimit(limit uint64) uint64 { return metal.SetMemoryLimit(limit) }

// GetActiveMemory returns the current active Metal memory usage in bytes.
//
//	fmt.Printf("active: %d MB\n", mlx.GetActiveMemory()/1024/1024)
func GetActiveMemory() uint64 { return metal.GetActiveMemory() }

// GetPeakMemory returns the peak Metal memory usage in bytes.
//
//	fmt.Printf("peak: %d MB\n", mlx.GetPeakMemory()/1024/1024)
func GetPeakMemory() uint64 { return metal.GetPeakMemory() }

// ClearCache releases Metal memory held in the allocator cache.
//
//	mlx.ClearCache() // reclaim prompt-cache memory between chat turns
func ClearCache() { metal.ClearCache() }

// GetCacheMemory returns the current Metal cache memory in bytes.
//
//	fmt.Printf("cache: %d MB\n", mlx.GetCacheMemory()/1024/1024)
func GetCacheMemory() uint64 { return metal.GetCacheMemory() }

// ResetPeakMemory resets the peak memory high-water mark to zero.
//
//	mlx.ResetPeakMemory()
func ResetPeakMemory() { metal.ResetPeakMemory() }

// SetWiredLimit sets the Metal wired memory limit. Returns the previous value.
//
//	mlx.SetWiredLimit(8 << 30) // 8 GB wired limit
func SetWiredLimit(limit uint64) uint64 { return metal.SetWiredLimit(limit) }

// DeviceInfo holds Metal GPU hardware information.
type DeviceInfo = metal.DeviceInfo

// GetDeviceInfo returns Metal GPU hardware information.
//
//	info := mlx.GetDeviceInfo()
//	fmt.Printf("%s %d MB\n", info.Architecture, info.MemorySize/1024/1024)
func GetDeviceInfo() DeviceInfo { return metal.GetDeviceInfo() }

type metalbackend struct{}

func (backend *metalbackend) Name() string    { return "metal" }
func (backend *metalbackend) Available() bool { return MetalAvailable() }

var loadBackendModel = func(modelPath string, cfg metal.LoadConfig) (*metal.Model, error) {
	return metal.LoadAndInit(modelPath, cfg)
}

// LoadModelAsTextModel loads modelPath with the rich mlx.LoadOption
// surface and returns it as an inference.TextModel ready for the
// openai/anthropic/ollama compat handlers (drop into an
// openaicompat.ResolverFunc).
//
// Bridge for cmd/mlx's `serve` command: the standard inference.LoadOption
// boundary (via inference.LoadModel + metalbackend.LoadModel) only flows
// ContextLength + ParallelSlots + AdapterPath + GPULayers from caller
// inputs; tuned profiles' CacheMode, CachePolicy, BatchSize, PromptCache,
// memory caps, etc. get filled in by PlanMemory() defaults and the
// caller-supplied values get silently dropped. This bridge skips that
// narrowing by translating mlx.LoadOption → metal.LoadConfig directly,
// preserving all 13 candidate fields the auto-tune profile encodes.
//
//	model, err := mlx.LoadModelAsTextModel(modelPath,
//	    mlx.WithContextLength(8192),
//	    mlx.WithKVCacheMode(memory.KVCacheModeStreaming),
//	    mlx.WithBatchSize(64),
//	)
func LoadModelAsTextModel(modelPath string, opts ...LoadOption) (inference.TextModel, error) {
	cfg, err := normalizeLoadConfig(applyLoadOptions(opts))
	if err != nil {
		return nil, err
	}
	cfg = applyMemoryPlanToLoadConfig(modelPath, cfg)
	metalCfg := metal.LoadConfig{
		ContextLen:            cfg.ContextLength,
		ParallelSlots:         cfg.ParallelSlots,
		DisablePromptCache:    !cfg.PromptCache,
		PromptCacheMinTokens:  cfg.PromptCacheMinTokens,
		AdapterPath:           cfg.AdapterPath,
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
	}
	nativeModel, err := loadBackendModel(modelPath, metalCfg)
	if err != nil {
		return nil, err
	}
	return &metaladapter{model: nativeModel, schedulerMaxConcurrent: cfg.ParallelSlots}, nil
}

func (backend *metalbackend) LoadModel(modelPath string, opts ...inference.LoadOption) core.Result {
	loadOptions := inference.ApplyLoadOpts(opts)
	deviceName, partialOffloadUnsupported := backendDeviceForGPULayers(loadOptions.GPULayers)
	if partialOffloadUnsupported {
		core.Warn("mlx: partial GPULayers unsupported — using full GPU offload", "gpu_layers", loadOptions.GPULayers)
	}
	plan := PlanMemory(MemoryPlanInput{Device: memoryPlannerDeviceInfo()})
	contextLen := loadOptions.ContextLen
	if contextLen == 0 {
		contextLen = plan.ContextLength
	}
	parallelSlots := loadOptions.ParallelSlots
	if parallelSlots == 0 {
		parallelSlots = plan.ParallelSlots
	}
	model, err := loadBackendModel(modelPath, metal.LoadConfig{
		ContextLen:           contextLen,
		ParallelSlots:        parallelSlots,
		DisablePromptCache:   !plan.PromptCache,
		PromptCacheMinTokens: plan.PromptCacheMinTokens,
		AdapterPath:          loadOptions.AdapterPath,
		Device:               metal.DeviceType(deviceName),
		CachePolicy:          string(plan.CachePolicy),
		KVCacheMode:          string(plan.CacheMode),
		BatchSize:            plan.BatchSize,
		PrefillChunkSize:     plan.PrefillChunkSize,
		ExpectedQuantization: plan.ModelQuantization,
		MemoryLimitBytes:     plan.MemoryLimitBytes,
		CacheLimitBytes:      plan.CacheLimitBytes,
		WiredLimitBytes:      plan.WiredLimitBytes,
	})
	if err != nil {
		return core.Fail(err)
	}
	return core.Ok(&metaladapter{model: model, schedulerMaxConcurrent: parallelSlots})
}

type metaladapter struct {
	model                  *metal.Model
	probeSink              inference.ProbeSink
	schedulerMu            sync.Mutex
	scheduler              *scheduler.Model
	schedulerMaxConcurrent int
	cacheMu                sync.Mutex
	cacheService           *blockcache.Service
	// continuity, when set via EnableConversationContinuity, routes Chat
	// through the no-prompt-replay conversation loop; declined requests fall
	// through to the stateless path.
	continuity *ConversationContinuity
}

func (adapter *metaladapter) Generate(ctx context.Context, prompt string, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	metalOptions := adapter.generateConfig(opts...)
	return func(yield func(inference.Token) bool) {
		for token := range adapter.model.Generate(ctx, prompt, metalOptions) {
			if !yield(inference.Token{ID: token.ID, Text: token.Text}) {
				return
			}
		}
	}
}

func (adapter *metaladapter) Chat(ctx context.Context, messages []inference.Message, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	if adapter.continuity != nil {
		if seq, ok := adapter.continuity.Chat(ctx, messages, opts...); ok {
			return seq
		}
	}
	metalOptions := adapter.generateConfig(opts...)
	metalMessages := make([]metal.ChatMessage, len(messages))
	for i, msg := range messages {
		metalMessages[i] = metal.ChatMessage{Role: msg.Role, Content: msg.Content, Images: msg.Images}
	}
	return func(yield func(inference.Token) bool) {
		for token := range adapter.model.Chat(ctx, metalMessages, metalOptions) {
			if !yield(inference.Token{ID: token.ID, Text: token.Text}) {
				return
			}
		}
	}
}

// AcceptsImages reports whether the loaded checkpoint serves image chat
// turns — the inference.VisionModel capability probe.
func (adapter *metaladapter) AcceptsImages() bool {
	return adapter.model.AcceptsImages()
}

func inferenceMessagesCarryImages(messages []inference.Message) bool {
	for i := range messages {
		if len(messages[i].Images) > 0 {
			return true
		}
	}
	return false
}

func (adapter *metaladapter) Classify(ctx context.Context, prompts []string, opts ...inference.GenerateOption) core.Result {
	generateOptions := inference.ApplyGenerateOpts(opts)
	metalOptions := adapter.generateConfig(opts...)
	results, err := adapter.model.Classify(ctx, prompts, metalOptions, generateOptions.ReturnLogits)
	if err != nil {
		return core.Fail(err)
	}
	classifications := make([]inference.ClassifyResult, len(results))
	for index, result := range results {
		classifications[index] = inference.ClassifyResult{
			Token:  inference.Token{ID: result.Token.ID, Text: result.Token.Text},
			Logits: result.Logits,
		}
	}
	return core.Ok(classifications)
}

func (adapter *metaladapter) BatchGenerate(ctx context.Context, prompts []string, opts ...inference.GenerateOption) core.Result {
	metalOptions := adapter.generateConfig(opts...)
	results, err := adapter.model.BatchGenerate(ctx, prompts, metalOptions)
	if err != nil {
		return core.Fail(err)
	}
	batchResults := make([]inference.BatchResult, len(results))
	for index, result := range results {
		tokens := make([]inference.Token, len(result.Tokens))
		for tokenIndex, token := range result.Tokens {
			tokens[tokenIndex] = inference.Token{ID: token.ID, Text: token.Text}
		}
		batchResults[index] = inference.BatchResult{Tokens: tokens, Err: result.Err}
	}
	return core.Ok(batchResults)
}

func (adapter *metaladapter) Metrics() inference.GenerateMetrics {
	metrics := adapter.model.LastMetrics()
	return inference.GenerateMetrics{
		PromptTokens:        metrics.PromptTokens,
		GeneratedTokens:     metrics.GeneratedTokens,
		PrefillDuration:     metrics.PrefillDuration,
		DecodeDuration:      metrics.DecodeDuration,
		TotalDuration:       metrics.TotalDuration,
		PrefillTokensPerSec: metrics.PrefillTokensPerSec,
		DecodeTokensPerSec:  metrics.DecodeTokensPerSec,
		PeakMemoryBytes:     metrics.PeakMemoryBytes,
		ActiveMemoryBytes:   metrics.ActiveMemoryBytes,
	}
}

func (adapter *metaladapter) ModelType() string { return adapter.model.ModelType() }
func (adapter *metaladapter) Info() inference.ModelInfo {
	modelInfo := adapter.model.Info()
	return inference.ModelInfo{
		Architecture: modelInfo.Architecture,
		VocabSize:    modelInfo.VocabSize,
		NumLayers:    modelInfo.NumLayers,
		HiddenSize:   modelInfo.HiddenSize,
		QuantBits:    modelInfo.QuantBits,
		QuantGroup:   modelInfo.QuantGroup,
	}
}
func (adapter *metaladapter) InspectAttention(ctx context.Context, prompt string, opts ...inference.GenerateOption) (*inference.AttentionSnapshot, error) {
	attention, err := adapter.model.InspectAttention(ctx, prompt)
	if err != nil {
		return nil, err
	}
	return &inference.AttentionSnapshot{
		NumLayers:     attention.NumLayers,
		NumHeads:      attention.NumHeads,
		SeqLen:        attention.SeqLen,
		HeadDim:       attention.HeadDim,
		NumQueryHeads: attention.NumQueryHeads,
		Keys:          attention.Keys,
		Queries:       attention.Queries,
		Architecture:  attention.Architecture,
	}, nil
}

func (adapter *metaladapter) Err() core.Result   { return core.ResultOf(nil, adapter.model.Err()) }
func (adapter *metaladapter) Close() core.Result { return core.ResultOf(nil, adapter.model.Close()) }
