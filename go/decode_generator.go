// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"sync"

	core "dappco.re/go"
	"dappco.re/go/inference/decode"
	"dappco.re/go/mlx/spine"
)

// errModelDecodeNil is returned by modelDecodeGenerator.Generate when the
// pooled generator has no live model attached.
var errModelDecodeNil = core.NewError("mlx: decode generator has nil model")

// modelDecodeGenerator is the pooled-struct shape that implements
// decode.Generator on a pointer receiver. Two fields, both pointers
// (model + base) — the per-call closure is gone, so the only allocation
// that remains for the decode hot path is the one decode.Speculative /
// decode.PromptLookup pays inside its own acceptance machinery.
//
// Concurrency: decode.Speculative invokes draft then target sequentially
// (single goroutine, draft Generate returns before target Generate is
// dispatched). decode.PromptLookup is single-Generate. So a generator
// instance is never invoked from two goroutines at once on any current
// decode path. If a future decode driver fan-outs Generate calls
// concurrently, each goroutine MUST acquire its own pool entry — base is
// shared by pointer so callers must treat it as read-only post-acquire
// (the Generate body dereferences `*g.base` into a local copy before
// mutating).
type modelDecodeGenerator struct {
	model *Model
	base  *GenerateConfig
}

// modelDecodeGeneratorPool recycles *modelDecodeGenerator across decode
// dispatches. Steady-state allocation count drops from "one closure per
// call" to "zero after the pool warms" because the struct itself is
// reused.
var modelDecodeGeneratorPool = sync.Pool{
	New: func() any { return &modelDecodeGenerator{} },
}

// acquireModelDecodeGenerator rents a generator from the pool and parks
// the (model, base) pair on it. Returning the struct pointer directly
// (rather than a release closure) is the load-bearing detail: any closure
// returned here would heap-allocate per call and drown the pooled-struct
// win. Callers pair this with a defer releaseModelDecodeGenerator(g).
func acquireModelDecodeGenerator(model *Model, base *GenerateConfig) *modelDecodeGenerator {
	g := modelDecodeGeneratorPool.Get().(*modelDecodeGenerator)
	g.model = model
	g.base = base
	return g
}

// releaseModelDecodeGenerator zeros the captured fields (so a stale model
// pointer does not keep a closed Model alive past its lifetime) and puts
// the struct back in the pool. Callers must not touch g after release.
func releaseModelDecodeGenerator(g *modelDecodeGenerator) {
	if g == nil {
		return
	}
	g.model = nil
	g.base = nil
	modelDecodeGeneratorPool.Put(g)
}

// Generate satisfies decode.Generator. Pointer receiver so the pool can
// hand back stored *modelDecodeGenerator values without per-call boxing.
func (g *modelDecodeGenerator) Generate(ctx context.Context, prompt string, cfg decode.GenerateConfig) (decode.Generation, error) {
	if g.model == nil || g.model.model == nil {
		return decode.Generation{}, errModelDecodeNil
	}
	generateCfg := *g.base
	if cfg.MaxTokens > 0 {
		generateCfg.MaxTokens = cfg.MaxTokens
	}
	// Pre-size tokens to MaxTokens — speculative/prompt-lookup decode
	// caps emitted tokens at MaxTokens, so a single make() avoids the
	// per-token append-grow doubling on every decoded step.
	tokens := make([]decode.Token, 0, generateCfg.MaxTokens)
	for token := range g.model.model.Generate(ctx, prompt, spine.ToMetalGenerateConfig(generateCfg)) {
		tokens = append(tokens, decode.Token{
			ID:   token.ID,
			Text: token.Text,
		})
	}
	if err := g.model.model.Err(); err != nil {
		return decode.Generation{}, err
	}
	return decode.Generation{Tokens: tokens, Text: decode.TokensText(tokens)}, nil
}
