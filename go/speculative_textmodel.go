// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"iter"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/mlx/chat"
	"dappco.re/go/mlx/pkg/metal"
)

// speculativeTextModel serves a Gemma-4 target through its native MTP assistant
// drafter — the block-draft + batched-verify speculative lane in
// gemma4.Gemma4AssistantPair. It embeds the target *metaladapter (inheriting
// every inference.TextModel method) and overrides only the two generation
// entries to route decode through the assistant pair.
//
// Correctness: speculative decode emits exactly the tokens the target would
// have produced under plain decoding — acceptance is TARGET-SAMPLED prefix
// verification, so the target's own sampler decides every committed token
// (greedy: batched argmax; temperature: the same keyed sampler chain plain
// decode draws from, one draw per committed token). A seeded sampled request
// commits byte-identically to plain AR decode, just in fewer target passes.
//
// Streaming: each verified token is yielded as its verify round lands
// (GenerateWithSink) — a streaming client sees tokens incrementally, paced by
// accepted blocks rather than single tokens.
//
// Fallbacks: repetition-penalty and probe-sink requests take plain target
// decode (still correct, no speculative speedup) so the served model never
// errors on them.
type speculativeTextModel struct {
	*metaladapter
	pair        *SpeculativePair
	draftTokens int
}

// LoadSpeculativePairAsTextModel loads a Gemma-4 target beside its native MTP
// assistant drafter and returns it as an inference.TextModel whose Generate and
// Chat run the native speculative lane. draftPath MUST resolve to a
// gemma4_assistant pack: generic target/draft speculative decode
// (decode.Speculative) is a deterministic acceptance-metrics reference that runs
// both models to completion — it is strictly slower than plain decode and is
// rejected here, never served.
//
//	tm, err := mlx.LoadSpeculativePairAsTextModel(
//	    "~/models/gemma-4-e2b-it-6bit",
//	    "~/models/gemma-4-E2B-it-assistant-bf16",
//	    mlx.WithContextLength(8192),
//	)
func LoadSpeculativePairAsTextModel(targetPath, draftPath string, opts ...LoadOption) (inference.TextModel, error) {
	return LoadSpeculativePairAsTextModelBlock(targetPath, draftPath, 0, opts...)
}

// LoadSpeculativePairAsTextModelBlock is LoadSpeculativePairAsTextModel with
// an explicit draft BLOCK (MTPLX semantics: the verify forward covers the
// carried lead plus block-1 assistant proposals). draftBlock <= 0 uses
// MTPDefaultDraftBlock; the serve wires its --draft-block flag (or a tuned
// profile) through here.
//
//	tm, err := mlx.LoadSpeculativePairAsTextModelBlock(target, draft, 5)
func LoadSpeculativePairAsTextModelBlock(targetPath, draftPath string, draftBlock int, opts ...LoadOption) (inference.TextModel, error) {
	pair, err := LoadSpeculativePair(targetPath, draftPath, SpeculativePairConfig{TargetOptions: opts})
	if err != nil {
		return nil, err
	}
	if pair.Gemma4Assistant == nil {
		closeErr := pair.Close()
		return nil, core.ErrorJoin(core.NewError("mlx: speculative serve requires a gemma4_assistant drafter — generic target/draft speculative decode is a metrics reference, not a serve speedup"), closeErr)
	}
	targetMetal, ok := pair.Target.model.(*metal.Model)
	if !ok {
		closeErr := pair.Close()
		return nil, core.ErrorJoin(core.NewError("mlx: speculative serve target is not a native metal model"), closeErr)
	}
	if draftBlock <= 0 {
		draftBlock = MTPDefaultDraftBlock
	}
	return &speculativeTextModel{
		metaladapter: &metaladapter{model: targetMetal},
		pair:         pair,
		draftTokens:  max(1, draftBlock-1),
	}, nil
}

// Generate streams the target's tokens for a raw prompt via the MTP lane.
func (s *speculativeTextModel) Generate(ctx context.Context, prompt string, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	return s.speculativeStream(ctx, prompt, opts...)
}

// Chat formats the conversation with the model's native template, then streams
// via the MTP lane. The template is applied here because the MTP loop tokenises
// its prompt as-is (it has no chat-template step of its own) — and it must
// honour the request's thinking override: templating with thinking ON for a
// no-think request sends the whole budget into the thought channel (caught
// live: 801 thought tokens, empty content, finish=length).
func (s *speculativeTextModel) Chat(ctx context.Context, messages []inference.Message, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	if inferenceMessagesCarryImages(messages) {
		// The MTP loop tokenises a flat prompt — image features can't ride
		// it. The embedded adapter's chat lane serves the vision path.
		return s.metaladapter.Chat(ctx, messages, opts...)
	}
	cfg := inference.ApplyGenerateOpts(opts)
	tplCfg := metalAdapterChatConfig(s.model.Info(), s.model.ModelType())
	if cfg.EnableThinking != nil {
		tplCfg.EnableThinking = *cfg.EnableThinking
	}
	return s.speculativeStream(ctx, chat.Format(messages, tplCfg), opts...)
}

// speculativeStream runs the native MTP assistant decode for greedy requests and
// yields the verified tokens; sampled / probe-sink requests fall back to plain
// target decode so the served model never rejects a valid generation request.
func (s *speculativeTextModel) speculativeStream(ctx context.Context, prompt string, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	metalCfg := s.generateConfig(opts...)
	if !s.mtpEligible(metalCfg) {
		return func(yield func(inference.Token) bool) {
			for token := range s.model.Generate(ctx, prompt, metalCfg) {
				if !yield(inference.Token{ID: token.ID, Text: token.Text}) {
					return
				}
			}
		}
	}
	return func(yield func(inference.Token) bool) {
		// Per-verify-block streaming: the MTP loop hands each verified token
		// to the sink as its round lands; the channel bridges the loop's
		// goroutine into this iterator. yield=false (client gone) cancels the
		// loop via genCtx and drains the channel so the sender never blocks.
		genCtx, cancel := context.WithCancel(ctx)
		defer cancel()
		tokens := make(chan inference.Token, 64)
		go func() {
			defer close(tokens)
			_, err := s.pair.Gemma4Assistant.GenerateWithSink(genCtx, s.model, prompt, metalCfg, s.draftTokens,
				func(token metal.Token) bool {
					select {
					case tokens <- inference.Token{ID: token.ID, Text: token.Text}:
						return true
					case <-genCtx.Done():
						return false
					}
				})
			// GenerateWithSink records errors on the model; metaladapter.Err()
			// surfaces them after the iterator stops.
			_ = err
		}()
		for token := range tokens {
			if !yield(token) {
				cancel()
				for range tokens {
				}
				return
			}
		}
	}
}

// mtpEligible reports whether this request can run through the native MTP
// lane. Greedy AND sampled requests are eligible: acceptance is TARGET-SAMPLED
// prefix verification (the target's own sampler judges every drafted token —
// no p/q residual maths, no drafter-distribution readbacks), and the verify
// runs on the live caches with journaled exact rollback instead of cloning
// every cache per round. The two costs that made the earlier sampled lane
// SLOWER than plain decode (p/q distribution materialisation per row + a full
// KV clone per verify round) are gone. Repetition penalty (logit rewrites
// from emitted history the verify rows do not carry) and probe sinks still
// take the plain lane — mirrors validateGemma4AssistantGenerateConfig.
func (s *speculativeTextModel) mtpEligible(cfg metal.GenerateConfig) bool {
	return cfg.RepeatPenalty <= 1 && cfg.ProbeSink == nil
}

// MTPMetrics returns the MTP counters from the last generation, or nil when
// it took a fallback lane. The generate bench prints the acceptance line off
// this — inference.GenerateMetrics carries no speculative detail.
//
//	if p, ok := tm.(interface{ MTPMetrics() *metal.MTPMetrics }); ok { … }
func (s *speculativeTextModel) MTPMetrics() *metal.MTPMetrics {
	return s.model.LastMetrics().MTP
}

// Close releases both the target and the attached assistant drafter.
func (s *speculativeTextModel) Close() error {
	return s.pair.Close()
}

// IsSpeculativeTextModel reports whether tm is the MTP pair lane — callers
// (the serve's continuity wiring) phrase operator notices truthfully for
// the pair's designed stateless posture (#92).
//
//	if mlx.IsSpeculativeTextModel(tm) { /* pair lane */ }
func IsSpeculativeTextModel(tm inference.TextModel) bool {
	switch tm.(type) {
	case *speculativeTextModel, *nativeSpeculativeTextModel:
		return true
	default:
		return false
	}
}
