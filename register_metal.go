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
// Call between chat turns to reclaim prompt cache memory promptly.
//
//	mlx.ClearCache()
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

func (b *metalBackend) Name() string   { return "metal" }
func (b *metalBackend) Available() bool { return true }

func (b *metalBackend) LoadModel(path string, opts ...inference.LoadOption) (inference.TextModel, error) {
	cfg := inference.ApplyLoadOpts(opts)
	if cfg.GPULayers == 0 {
		slog.Warn("mlx: GPULayers=0 ignored — Metal always uses full GPU offload")
	}
	m, err := metal.LoadAndInit(path, metal.LoadConfig{
		ContextLen:  cfg.ContextLen,
		AdapterPath: cfg.AdapterPath,
	})
	if err != nil {
		return nil, err
	}
	return &metalAdapter{m: m}, nil
}

type metalAdapter struct {
	m *metal.Model
}

func (a *metalAdapter) Generate(ctx context.Context, prompt string, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	cfg := inference.ApplyGenerateOpts(opts)
	mcfg := metal.GenerateConfig{
		MaxTokens:     cfg.MaxTokens,
		Temperature:   cfg.Temperature,
		TopK:          cfg.TopK,
		TopP:          cfg.TopP,
		StopTokens:    cfg.StopTokens,
		RepeatPenalty: cfg.RepeatPenalty,
	}
	return func(yield func(inference.Token) bool) {
		for tok := range a.m.Generate(ctx, prompt, mcfg) {
			if !yield(inference.Token{ID: tok.ID, Text: tok.Text}) {
				return
			}
		}
	}
}

func (a *metalAdapter) Chat(ctx context.Context, messages []inference.Message, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	cfg := inference.ApplyGenerateOpts(opts)
	mcfg := metal.GenerateConfig{
		MaxTokens:     cfg.MaxTokens,
		Temperature:   cfg.Temperature,
		TopK:          cfg.TopK,
		TopP:          cfg.TopP,
		StopTokens:    cfg.StopTokens,
		RepeatPenalty: cfg.RepeatPenalty,
	}
	mmsgs := make([]metal.ChatMessage, len(messages))
	for i, msg := range messages {
		mmsgs[i] = metal.ChatMessage{Role: msg.Role, Content: msg.Content}
	}
	return func(yield func(inference.Token) bool) {
		for tok := range a.m.Chat(ctx, mmsgs, mcfg) {
			if !yield(inference.Token{ID: tok.ID, Text: tok.Text}) {
				return
			}
		}
	}
}

func (a *metalAdapter) Classify(ctx context.Context, prompts []string, opts ...inference.GenerateOption) ([]inference.ClassifyResult, error) {
	cfg := inference.ApplyGenerateOpts(opts)
	mcfg := metal.GenerateConfig{
		Temperature: cfg.Temperature,
		TopK:        cfg.TopK,
	}
	results, err := a.m.Classify(ctx, prompts, mcfg, cfg.ReturnLogits)
	if err != nil {
		return nil, err
	}
	out := make([]inference.ClassifyResult, len(results))
	for i, r := range results {
		out[i] = inference.ClassifyResult{
			Token:  inference.Token{ID: r.Token.ID, Text: r.Token.Text},
			Logits: r.Logits,
		}
	}
	return out, nil
}

func (a *metalAdapter) BatchGenerate(ctx context.Context, prompts []string, opts ...inference.GenerateOption) ([]inference.BatchResult, error) {
	cfg := inference.ApplyGenerateOpts(opts)
	mcfg := metal.GenerateConfig{
		MaxTokens:     cfg.MaxTokens,
		Temperature:   cfg.Temperature,
		TopK:          cfg.TopK,
		TopP:          cfg.TopP,
		StopTokens:    cfg.StopTokens,
		RepeatPenalty: cfg.RepeatPenalty,
	}
	results, err := a.m.BatchGenerate(ctx, prompts, mcfg)
	if err != nil {
		return nil, err
	}
	out := make([]inference.BatchResult, len(results))
	for i, r := range results {
		tokens := make([]inference.Token, len(r.Tokens))
		for j, t := range r.Tokens {
			tokens[j] = inference.Token{ID: t.ID, Text: t.Text}
		}
		out[i] = inference.BatchResult{Tokens: tokens, Err: r.Err}
	}
	return out, nil
}

func (a *metalAdapter) Metrics() inference.GenerateMetrics {
	m := a.m.LastMetrics()
	return inference.GenerateMetrics{
		PromptTokens:        m.PromptTokens,
		GeneratedTokens:     m.GeneratedTokens,
		PrefillDuration:     m.PrefillDuration,
		DecodeDuration:      m.DecodeDuration,
		TotalDuration:       m.TotalDuration,
		PrefillTokensPerSec: m.PrefillTokensPerSec,
		DecodeTokensPerSec:  m.DecodeTokensPerSec,
		PeakMemoryBytes:     m.PeakMemoryBytes,
		ActiveMemoryBytes:   m.ActiveMemoryBytes,
	}
}

func (a *metalAdapter) ModelType() string { return a.m.ModelType() }
func (a *metalAdapter) Info() inference.ModelInfo {
	i := a.m.Info()
	return inference.ModelInfo{
		Architecture: i.Architecture,
		VocabSize:    i.VocabSize,
		NumLayers:    i.NumLayers,
		HiddenSize:   i.HiddenSize,
		QuantBits:    i.QuantBits,
		QuantGroup:   i.QuantGroup,
	}
}
func (a *metalAdapter) InspectAttention(ctx context.Context, prompt string, opts ...inference.GenerateOption) (*inference.AttentionSnapshot, error) {
	result, err := a.m.InspectAttention(ctx, prompt)
	if err != nil {
		return nil, err
	}
	return &inference.AttentionSnapshot{
		NumLayers:    result.NumLayers,
		NumHeads:     result.NumHeads,
		SeqLen:       result.SeqLen,
		HeadDim:      result.HeadDim,
		Keys:         result.Keys,
		Architecture: result.Architecture,
	}, nil
}

func (a *metalAdapter) Err() error   { return a.m.Err() }
func (a *metalAdapter) Close() error { return a.m.Close() }
