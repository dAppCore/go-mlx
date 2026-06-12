// SPDX-Licence-Identifier: EUPL-1.2

// Package speculative provides profile and report helpers for target/draft
// decoding without pulling benchmark business logic into command entry points.
package specprofile

import (
	"context"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/decode"
	mlx "dappco.re/go/mlx"
	"dappco.re/go/mlx/chat"
)

// Pair is the minimal target/draft runtime surface needed to profile one run.
type Pair interface {
	Generate(context.Context, string, mlx.SpeculativeDecodeConfig) (mlx.SpeculativeDecodeResult, error)
	Metrics() mlx.Metrics
	Err() error
}

// ProfileConfig configures one profiled target/draft generation.
type ProfileConfig struct {
	Prompt         string
	MaxTokens      int
	DraftTokens    int
	IncludeOutput  bool
	Chat           bool
	Architecture   string
	GenerateConfig mlx.GenerateConfig
}

// ProfileRun records one target/draft generation with sampled token IDs,
// optional token text, and the target model's native metrics.
type ProfileRun struct {
	Duration          time.Duration
	VisibleTokens     int
	SampledTokenIDs   []int32
	SampledTokenTexts []string
	Output            string
	Result            mlx.SpeculativeDecodeResult
	Metrics           mlx.Metrics
}

var errSpeculativeProfilePairNil = core.NewError("mlx/speculative: pair is nil")

var _ Pair = (*mlx.SpeculativePair)(nil)

// RunPairProfile runs an already-loaded target/draft pair once and returns a
// profile-friendly result. CLI commands own loading, report shaping, and
// safety-policy decisions; this package function owns the decode call and
// native metrics capture.
func RunPairProfile(ctx context.Context, pair Pair, cfg ProfileConfig) (ProfileRun, error) {
	start := time.Now()
	if ctx == nil {
		ctx = context.Background()
	}
	if pair == nil {
		return ProfileRun{Duration: nonZeroProfileDuration(time.Since(start))}, errSpeculativeProfilePairNil
	}
	if cfg.MaxTokens <= 0 {
		cfg.MaxTokens = 1
	}
	if cfg.DraftTokens <= 0 {
		cfg.DraftTokens = mlx.MTPDefaultDraftTokens
	}
	prompt := cfg.Prompt
	if cfg.Chat {
		prompt = chat.Format([]inference.Message{{Role: "user", Content: cfg.Prompt}}, chat.Config{
			Architecture: cfg.Architecture,
		})
	}
	generateConfig := cfg.GenerateConfig
	generateConfig.MaxTokens = cfg.MaxTokens
	generateConfig.Temperature = 0
	result, err := pair.Generate(ctx, prompt, mlx.SpeculativeDecodeConfig{
		MaxTokens:      cfg.MaxTokens,
		DraftTokens:    cfg.DraftTokens,
		GenerateConfig: generateConfig,
	})
	run := ProfileRun{
		Duration:          nonZeroProfileDuration(time.Since(start)),
		VisibleTokens:     len(result.Tokens),
		SampledTokenIDs:   sampledTokenIDs(result.Tokens, 32),
		SampledTokenTexts: sampledTokenTexts(result.Tokens, 32, cfg.IncludeOutput),
		Result:            result,
		Metrics:           pair.Metrics(),
	}
	if cfg.IncludeOutput {
		run.Output = result.Text
	}
	if err != nil {
		return run, err
	}
	if err := pair.Err(); err != nil {
		return run, err
	}
	if err := ctx.Err(); err != nil {
		return run, err
	}
	return run, nil
}

func sampledTokenIDs(tokens []decode.Token, limit int) []int32 {
	if len(tokens) == 0 || limit <= 0 {
		return nil
	}
	if len(tokens) < limit {
		limit = len(tokens)
	}
	out := make([]int32, limit)
	for i := 0; i < limit; i++ {
		out[i] = tokens[i].ID
	}
	return out
}

func sampledTokenTexts(tokens []decode.Token, limit int, include bool) []string {
	if !include || len(tokens) == 0 || limit <= 0 {
		return nil
	}
	if len(tokens) < limit {
		limit = len(tokens)
	}
	out := make([]string, limit)
	for i := 0; i < limit; i++ {
		out[i] = tokens[i].Text
	}
	return out
}

func nonZeroProfileDuration(duration time.Duration) time.Duration {
	if duration <= 0 {
		return time.Nanosecond
	}
	return duration
}
