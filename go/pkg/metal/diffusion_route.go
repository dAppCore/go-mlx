// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"context"
	"iter"
	"time"
)

// Block-diffusion generation — the neutral capability seam. A family model
// that decodes by canvas denoising (DiffusionGemma) implements
// BlockDiffusionModel; Model.Generate routes there ahead of the
// autoregressive session lane, so every consumer of the token-stream
// surface (the serve adapter, the CLI, lib callers) gets diffusion decoding
// transparently. The family owns the algorithm; this file owns the contract
// (AX-8: the engine never imports the family).

// BlockDiffusionOptions carries the request-level knobs the neutral surface
// can express. The family applies its own reference defaults for anything
// zero-valued.
type BlockDiffusionOptions struct {
	// MaxTokens bounds the emitted response (canvases are derived from it).
	MaxTokens int
	// Temperature scales the family's annealing temperature range — 0 or 1
	// keeps the reference schedule.
	Temperature float32
	// Seed roots the PRNG key chain; SeedSet false derives one from time.
	Seed    uint64
	SeedSet bool
	// StopTokens extend the checkpoint's own end-of-sequence set.
	StopTokens []int32
}

// BlockDiffusionModel is the family-side implementation: stream tokens for a
// formatted prompt, canvas by canvas. The iterator's final error is exposed
// via Err on the model after the sequence completes (the Token stream
// contract matches the session lanes).
type BlockDiffusionModel interface {
	GenerateBlockDiffusion(ctx context.Context, prompt string, opts BlockDiffusionOptions) iter.Seq[Token]
	BlockDiffusionErr() error
	BlockDiffusionMetrics() BlockDiffusionMetrics
}

// BlockDiffusionMetrics is the neutral readback of one diffusion run.
type BlockDiffusionMetrics struct {
	PromptTokens  int
	EmittedTokens int
	Canvases      int
	TotalSteps    int
	PrefillDur    time.Duration
	DenoiseDur    time.Duration
	CommitDur     time.Duration
	TotalDur      time.Duration
}

// generateViaBlockDiffusion adapts the family capability to the Model
// surface: seeds the options from GenerateConfig, streams the canvases, and
// publishes Metrics/lastErr exactly like the session route.
func (m *Model) generateViaBlockDiffusion(ctx context.Context, bd BlockDiffusionModel, prompt string, cfg GenerateConfig) iter.Seq[Token] {
	return func(yield func(Token) bool) {
		opts := BlockDiffusionOptions{
			MaxTokens:   cfg.MaxTokens,
			Temperature: cfg.Temperature,
			Seed:        uint64(cfg.Seed),
			SeedSet:     cfg.SeedSet,
			StopTokens:  cfg.StopTokens,
		}
		start := time.Now()
		for tok := range bd.GenerateBlockDiffusion(ctx, prompt, opts) {
			if !yield(tok) {
				break
			}
		}
		if err := bd.BlockDiffusionErr(); err != nil {
			m.lastErr = err
		}
		bm := bd.BlockDiffusionMetrics()
		metrics := Metrics{
			PromptTokens:     bm.PromptTokens,
			GeneratedTokens:  bm.EmittedTokens,
			PrefillDuration:  bm.PrefillDur,
			DecodeDuration:   bm.DenoiseDur + bm.CommitDur,
			TotalDuration:    time.Since(start),
			DecodeLane:       "block-diffusion",
			DecodeLaneReason: "",
		}
		if bm.PrefillDur > 0 && bm.PromptTokens > 0 {
			metrics.PrefillTokensPerSec = float64(bm.PromptTokens) / bm.PrefillDur.Seconds()
		}
		if d := bm.DenoiseDur + bm.CommitDur; d > 0 && bm.EmittedTokens > 0 {
			metrics.DecodeTokensPerSec = float64(bm.EmittedTokens) / d.Seconds()
			metrics.WarmDecodeTokensPerSec = metrics.DecodeTokensPerSec
		}
		m.lastMetrics = metrics
	}
}
