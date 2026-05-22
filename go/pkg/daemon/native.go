// SPDX-Licence-Identifier: EUPL-1.2

package daemon

import (
	"context"
	"sync"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
	mlx "dappco.re/go/mlx"
)

const defaultNativeModelName = "default"

var (
	errRunnerNil            = core.NewError("native generate runner is nil")
	errPromptRequired       = core.NewError("generate prompt is required")
	errNoModelsConfigured   = core.NewError("no native models configured")
	errGenerateModelMissing = core.NewError("generate model is required")
)

type nativeGenerateModel interface {
	GenerateStream(context.Context, string, ...mlx.GenerateOption) <-chan mlx.Token
	ChatStream(context.Context, []inference.Message, ...mlx.GenerateOption) <-chan mlx.Token
	WarmPromptCache(string) error
	Metrics() mlx.Metrics
	Err() error
	Close() error
}

// NativeGenerateConfig configures the Violet non-HTTP native generate route.
type NativeGenerateConfig struct {
	ModelPaths       map[string]string
	DefaultModelName string
	DefaultMaxTokens int
	LoadOptions      []mlx.LoadOption
}

// NativeGenerateRunner loads go-mlx models once and serves generate requests.
type NativeGenerateRunner struct {
	mu              sync.Mutex
	modelPaths      map[string]string
	defaultModel    string
	defaultMaxToken int
	loadOptions     []mlx.LoadOption
	loadModel       func(string, ...mlx.LoadOption) (nativeGenerateModel, error)
	models          map[string]nativeGenerateModel
}

// NewNativeGenerateRunner builds a native go-mlx generate backend.
func NewNativeGenerateRunner(cfg NativeGenerateConfig) *NativeGenerateRunner {
	defaultModel := core.Trim(cfg.DefaultModelName)
	if defaultModel == "" {
		defaultModel = defaultNativeModelName
	}
	return &NativeGenerateRunner{
		modelPaths:      copyStringMap(cfg.ModelPaths),
		defaultModel:    defaultModel,
		defaultMaxToken: cfg.DefaultMaxTokens,
		loadOptions:     append([]mlx.LoadOption(nil), cfg.LoadOptions...),
		loadModel: func(path string, opts ...mlx.LoadOption) (nativeGenerateModel, error) {
			return mlx.LoadModel(path, opts...)
		},
		models: make(map[string]nativeGenerateModel),
	}
}

// Generate runs a prompt or chat request through a cached native go-mlx model.
func (runner *NativeGenerateRunner) Generate(ctx context.Context, req GenerateRequest) (GenerateResult, error) {
	if runner == nil {
		return GenerateResult{}, errRunnerNil
	}
	if ctx == nil {
		ctx = context.Background()
	}

	modelName, modelPath, err := runner.resolveModel(req.Model)
	if err != nil {
		return GenerateResult{}, err
	}
	model, err := runner.modelFor(modelName, modelPath)
	if err != nil {
		return GenerateResult{}, err
	}

	opts := runner.generateOptions(req)
	builder := core.NewBuilder()
	// Pre-grow the response buffer. Tokens average ~4 bytes; sizing
	// once up front avoids the strings.Builder growth ladder
	// (8 -> 16 -> 32 -> ...) during the per-token write loop.
	if hint := estimateGenerateBytes(req, runner.defaultMaxToken); hint > 0 {
		builder.Grow(hint)
	}
	if len(req.Messages) > 0 {
		for token := range model.ChatStream(ctx, toMLXMessages(req.Messages), opts...) {
			builder.WriteString(token.Text)
		}
	} else {
		if core.Trim(req.Prompt) == "" {
			return GenerateResult{}, errPromptRequired
		}
		for token := range model.GenerateStream(ctx, req.Prompt, opts...) {
			builder.WriteString(token.Text)
		}
	}
	if err := model.Err(); err != nil {
		return GenerateResult{Text: builder.String(), Model: modelName}, err
	}

	return GenerateResult{
		Text:    builder.String(),
		Model:   modelName,
		Metrics: toGenerateMetrics(model.Metrics()),
	}, nil
}

// Close releases all loaded native models.
func (runner *NativeGenerateRunner) Close() error {
	if runner == nil {
		return nil
	}
	runner.mu.Lock()
	models := runner.models
	runner.models = make(map[string]nativeGenerateModel)
	runner.mu.Unlock()

	var closeErr error
	for _, model := range models {
		if model == nil {
			continue
		}
		closeErr = core.ErrorJoin(closeErr, model.Close())
	}
	return closeErr
}

func (runner *NativeGenerateRunner) resolveModel(requested string) (string, string, error) {
	if len(runner.modelPaths) == 0 {
		return "", "", errNoModelsConfigured
	}
	modelName := core.Trim(requested)
	if modelName != "" {
		path := runner.modelPaths[modelName]
		if path == "" {
			return "", "", core.Errorf("unknown model %q", modelName)
		}
		return modelName, path, nil
	}
	if runner.defaultModel != "" {
		if path := runner.modelPaths[runner.defaultModel]; path != "" {
			return runner.defaultModel, path, nil
		}
	}
	if path := runner.modelPaths["generate"]; path != "" {
		return "generate", path, nil
	}
	if len(runner.modelPaths) == 1 {
		for name, path := range runner.modelPaths {
			return name, path, nil
		}
	}
	return "", "", errGenerateModelMissing
}

func (runner *NativeGenerateRunner) modelFor(name, path string) (nativeGenerateModel, error) {
	runner.mu.Lock()
	defer runner.mu.Unlock()

	if model := runner.models[name]; model != nil {
		return model, nil
	}
	model, err := runner.loadModel(path, runner.loadOptions...)
	if err != nil {
		return nil, core.Errorf("load native model %q: %w", name, err)
	}
	runner.models[name] = model
	return model, nil
}

func (runner *NativeGenerateRunner) generateOptions(req GenerateRequest) []mlx.GenerateOption {
	var opts []mlx.GenerateOption
	maxTokens := req.MaxTokens
	if maxTokens == 0 {
		maxTokens = runner.defaultMaxToken
	}
	if maxTokens > 0 {
		opts = append(opts, mlx.WithMaxTokens(maxTokens))
	}
	if req.Temperature != 0 {
		opts = append(opts, mlx.WithTemperature(float32(req.Temperature)))
	}
	return opts
}

func toMLXMessages(messages []Message) []inference.Message {
	out := make([]inference.Message, len(messages))
	for i, message := range messages {
		out[i] = inference.Message{Role: message.Role, Content: message.Content}
	}
	return out
}

func toGenerateMetrics(metrics mlx.Metrics) GenerateMetrics {
	return GenerateMetrics{
		PromptTokens:             metrics.PromptTokens,
		GeneratedTokens:          metrics.GeneratedTokens,
		PrefillSeconds:           metrics.PrefillDuration.Seconds(),
		DecodeSeconds:            metrics.DecodeDuration.Seconds(),
		TotalSeconds:             metrics.TotalDuration.Seconds(),
		PrefillTokensPerSec:      metrics.PrefillTokensPerSec,
		DecodeTokensPerSec:       metrics.DecodeTokensPerSec,
		PeakMemoryBytes:          metrics.PeakMemoryBytes,
		ActiveMemoryBytes:        metrics.ActiveMemoryBytes,
		PromptCacheHits:          metrics.PromptCacheHits,
		PromptCacheMisses:        metrics.PromptCacheMisses,
		PromptCacheHitTokens:     metrics.PromptCacheHitTokens,
		PromptCacheMissTokens:    metrics.PromptCacheMissTokens,
		PromptCacheRestoreMillis: float64(metrics.PromptCacheRestoreDuration) / float64(time.Millisecond),
	}
}

// estimateGenerateBytes returns a strings.Builder pre-grow hint for
// the generated response. The byte-per-token coefficient is a
// conservative average across typical chat tokens.
func estimateGenerateBytes(req GenerateRequest, fallbackMaxTokens int) int {
	const bytesPerToken = 4
	maxTokens := req.MaxTokens
	if maxTokens == 0 {
		maxTokens = fallbackMaxTokens
	}
	if maxTokens <= 0 {
		return 0
	}
	return maxTokens * bytesPerToken
}

func copyStringMap(in map[string]string) map[string]string {
	if len(in) == 0 {
		return nil
	}
	out := make(map[string]string, len(in))
	for key, value := range in {
		out[key] = value
	}
	return out
}
