// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"context"
	"iter"
	"sync"
	"time"

	"dappco.re/go/mlx/pkg/metal"
)

// The serve bridge (docs/RFC.diffusion-gemma.md, Unit D): DiffusionGemma
// implements metal.BlockDiffusionModel, so Model.Generate — and with it the
// serve adapter, the CLI plain lane, and every lib caller — streams canvases
// instead of running the autoregressive loop. Tokens yield one canvas at a
// time (the natural grain of block diffusion); the serve's thinking
// extractor splits the thought channel downstream exactly as it does for
// the autoregressive family.

// diffusionRunState carries the last run's error and metrics for the
// neutral readbacks. Single-flight per model (the serve serialises requests
// per loaded model the same way the session lanes do).
type diffusionRunState struct {
	mu      sync.Mutex
	err     error
	metrics metal.BlockDiffusionMetrics
}

// GenerateBlockDiffusion streams a block-diffusion generation, one canvas at
// a time. The prompt arrives formatted (the chat layer is upstream).
func (m *DiffusionGemmaModel) GenerateBlockDiffusion(ctx context.Context, prompt string, opts metal.BlockDiffusionOptions) iter.Seq[metal.Token] {
	return func(yield func(metal.Token) bool) {
		if ctx == nil {
			ctx = context.Background()
		}
		m.runState().set(nil, metal.BlockDiffusionMetrics{})

		// The serve runs the tuned decode profile (DefaultCanvasLength /
		// DefaultMaxSteps via zero-values), not the checkpoint's declared
		// 256-canvas — 2x the rate, same text, and canvas-grain streaming
		// deltas land 4x as often.
		canvasLen := int32(DefaultCanvasLength)
		maxTokens := opts.MaxTokens
		if maxTokens <= 0 {
			maxTokens = int(canvasLen)
		}
		maxCanvases := (maxTokens + int(canvasLen) - 1) / int(canvasLen)

		cfg := DiffusionGenerateConfig{
			Step:         DefaultDiffusionStepConfig(m.Cfg.VocabSize),
			CanvasLength: canvasLen,
			MaxCanvases:  maxCanvases,
			StopTokens:   append(append([]int32(nil), m.EOSTokens...), opts.StopTokens...),
		}
		if opts.SeedSet {
			cfg.Step.Seed = opts.Seed
		} else {
			cfg.Step.Seed = uint64(time.Now().UnixNano())
		}
		// Temperature scales the annealing range; 0 and 1 keep the
		// reference schedule (0.8 -> 0.4).
		if opts.Temperature > 0 && opts.Temperature != 1.0 {
			cfg.Step.MaxTemperature *= opts.Temperature
			cfg.Step.MinTemperature *= opts.Temperature
		}

		// Request-scoped caches: full-size fixed (the diffusion masks carry
		// the window semantics, and TruncateTo needs pre-cap headroom for
		// every canvas).
		promptTokens := m.Tok.Encode(prompt)
		capacity := len(promptTokens) + (int(canvasLen)+8)*maxCanvases + 64
		caches := make([]metal.Cache, len(m.Layers))
		for i := range caches {
			caches[i] = metal.NewFixedKVCache(capacity)
		}
		defer metal.FreeCaches(caches)

		// Canvas-at-a-time streaming: each committed canvas decodes as one
		// text delta (the serve treats Token.Text as a delta; per-token
		// granularity is a later polish). A consumer stop cancels the loop.
		genCtx, cancel := context.WithCancel(ctx)
		defer cancel()
		emitted := 0
		cfg.OnCanvas = func(_ int, kept []int32, _ int, _ time.Duration) {
			if len(kept) == 0 {
				return
			}
			if emitted+len(kept) > maxTokens {
				kept = kept[:maxTokens-emitted]
			}
			if len(kept) == 0 {
				cancel()
				return
			}
			emitted += len(kept)
			text := m.Tok.Decode(kept)
			if !yield(metal.Token{ID: kept[len(kept)-1], Text: text}) {
				cancel()
			}
		}

		_, dm, err := m.GenerateDiffusion(genCtx, prompt, caches, cfg)
		if err != nil && genCtx.Err() == nil {
			m.runState().set(err, metal.BlockDiffusionMetrics{})
			return
		}
		m.runState().set(nil, metal.BlockDiffusionMetrics{
			PromptTokens:  dm.PrefillTokens,
			EmittedTokens: dm.EmittedTokens,
			Canvases:      dm.Canvases,
			TotalSteps:    dm.TotalSteps,
			PrefillDur:    dm.PrefillDur,
			DenoiseDur:    dm.DenoiseDur,
			CommitDur:     dm.CommitDur,
			TotalDur:      dm.TotalDur,
		})
	}
}

// BlockDiffusionErr reports the last run's terminal error.
func (m *DiffusionGemmaModel) BlockDiffusionErr() error {
	s := m.runState()
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.err
}

// BlockDiffusionMetrics reports the last run's counters.
func (m *DiffusionGemmaModel) BlockDiffusionMetrics() metal.BlockDiffusionMetrics {
	s := m.runState()
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.metrics
}

func (m *DiffusionGemmaModel) runState() *diffusionRunState {
	m.runOnce.Do(func() { m.run = &diffusionRunState{} })
	return m.run
}

func (s *diffusionRunState) set(err error, metrics metal.BlockDiffusionMetrics) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.err = err
	s.metrics = metrics
}

// Compile-time proof the family satisfies the neutral capability.
var _ metal.BlockDiffusionModel = (*DiffusionGemmaModel)(nil)
