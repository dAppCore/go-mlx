//go:build darwin && arm64

package mlx

import (
	"context"
	"iter"
	"log/slog"

	"forge.lthn.ai/core/go-inference"
	"forge.lthn.ai/core/go-mlx/internal/metal"
)

func init() {
	inference.Register(&metalBackend{})
}

// MetalAvailable reports whether native Metal inference is available.
//
//	if mlx.MetalAvailable() { /* run on GPU */ }
func MetalAvailable() bool { return true }

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

type metalBackend struct{}

func (backend *metalBackend) Name() string    { return "metal" }
func (backend *metalBackend) Available() bool { return true }

func (backend *metalBackend) LoadModel(modelPath string, opts ...inference.LoadOption) (inference.TextModel, error) {
	loadOptions := inference.ApplyLoadOpts(opts)
	if loadOptions.GPULayers == 0 {
		slog.Warn("mlx: GPULayers=0 ignored — Metal always uses full GPU offload")
	}
	model, err := metal.LoadAndInit(modelPath, metal.LoadConfig{
		ContextLen:  loadOptions.ContextLen,
		AdapterPath: loadOptions.AdapterPath,
	})
	if err != nil {
		return nil, err
	}
	return &metalAdapter{model: model}, nil
}

type metalAdapter struct {
	model *metal.Model
}

func (adapter *metalAdapter) Generate(ctx context.Context, prompt string, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	generateOptions := inference.ApplyGenerateOpts(opts)
	metalOptions := metal.GenerateConfig{
		MaxTokens:     generateOptions.MaxTokens,
		Temperature:   generateOptions.Temperature,
		TopK:          generateOptions.TopK,
		TopP:          generateOptions.TopP,
		StopTokens:    generateOptions.StopTokens,
		RepeatPenalty: generateOptions.RepeatPenalty,
	}
	return func(yield func(inference.Token) bool) {
		for token := range adapter.model.Generate(ctx, prompt, metalOptions) {
			if !yield(inference.Token{ID: token.ID, Text: token.Text}) {
				return
			}
		}
	}
}

func (adapter *metalAdapter) Chat(ctx context.Context, messages []inference.Message, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	generateOptions := inference.ApplyGenerateOpts(opts)
	metalOptions := metal.GenerateConfig{
		MaxTokens:     generateOptions.MaxTokens,
		Temperature:   generateOptions.Temperature,
		TopK:          generateOptions.TopK,
		TopP:          generateOptions.TopP,
		StopTokens:    generateOptions.StopTokens,
		RepeatPenalty: generateOptions.RepeatPenalty,
	}
	metalMessages := make([]metal.ChatMessage, len(messages))
	for i, msg := range messages {
		metalMessages[i] = metal.ChatMessage{Role: msg.Role, Content: msg.Content}
	}
	return func(yield func(inference.Token) bool) {
		for token := range adapter.model.Chat(ctx, metalMessages, metalOptions) {
			if !yield(inference.Token{ID: token.ID, Text: token.Text}) {
				return
			}
		}
	}
}

func (adapter *metalAdapter) Classify(ctx context.Context, prompts []string, opts ...inference.GenerateOption) ([]inference.ClassifyResult, error) {
	generateOptions := inference.ApplyGenerateOpts(opts)
	metalOptions := metal.GenerateConfig{
		Temperature: generateOptions.Temperature,
		TopK:        generateOptions.TopK,
	}
	results, err := adapter.model.Classify(ctx, prompts, metalOptions, generateOptions.ReturnLogits)
	if err != nil {
		return nil, err
	}
	classifications := make([]inference.ClassifyResult, len(results))
	for index, result := range results {
		classifications[index] = inference.ClassifyResult{
			Token:  inference.Token{ID: result.Token.ID, Text: result.Token.Text},
			Logits: result.Logits,
		}
	}
	return classifications, nil
}

func (adapter *metalAdapter) BatchGenerate(ctx context.Context, prompts []string, opts ...inference.GenerateOption) ([]inference.BatchResult, error) {
	generateOptions := inference.ApplyGenerateOpts(opts)
	metalOptions := metal.GenerateConfig{
		MaxTokens:     generateOptions.MaxTokens,
		Temperature:   generateOptions.Temperature,
		TopK:          generateOptions.TopK,
		TopP:          generateOptions.TopP,
		StopTokens:    generateOptions.StopTokens,
		RepeatPenalty: generateOptions.RepeatPenalty,
	}
	results, err := adapter.model.BatchGenerate(ctx, prompts, metalOptions)
	if err != nil {
		return nil, err
	}
	batchResults := make([]inference.BatchResult, len(results))
	for index, result := range results {
		tokens := make([]inference.Token, len(result.Tokens))
		for tokenIndex, token := range result.Tokens {
			tokens[tokenIndex] = inference.Token{ID: token.ID, Text: token.Text}
		}
		batchResults[index] = inference.BatchResult{Tokens: tokens, Err: result.Err}
	}
	return batchResults, nil
}

func (adapter *metalAdapter) Metrics() inference.GenerateMetrics {
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

func (adapter *metalAdapter) ModelType() string { return adapter.model.ModelType() }
func (adapter *metalAdapter) Info() inference.ModelInfo {
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
func (adapter *metalAdapter) InspectAttention(ctx context.Context, prompt string, opts ...inference.GenerateOption) (*inference.AttentionSnapshot, error) {
	attention, err := adapter.model.InspectAttention(ctx, prompt)
	if err != nil {
		return nil, err
	}
	return &inference.AttentionSnapshot{
		NumLayers:    attention.NumLayers,
		NumHeads:     attention.NumHeads,
		SeqLen:       attention.SeqLen,
		HeadDim:      attention.HeadDim,
		Keys:         attention.Keys,
		Architecture: attention.Architecture,
	}, nil
}

func (adapter *metalAdapter) Err() error   { return adapter.model.Err() }
func (adapter *metalAdapter) Close() error { return adapter.model.Close() }
