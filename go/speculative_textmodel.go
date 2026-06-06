// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"iter"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/mlx/pkg/metal"
)

// speculativeTextModel serves a Gemma-4 target through its native MTP assistant
// drafter — the block-draft + batched-verify speculative lane in
// gemma4.Gemma4AssistantPair. It embeds the target *metaladapter (inheriting
// every inference.TextModel method) and overrides only the two generation
// entries to route decode through the assistant pair.
//
// Correctness: speculative decode emits exactly the tokens the target would
// have produced under greedy decoding — the draft only proposes candidates the
// target then verifies, so accepted output is identical to plain target decode,
// just produced in fewer target forward passes.
//
// Buffering: the native MTP loop (Gemma4AssistantPair.Generate) is one-shot, so
// this adapter collects the whole result then yields its tokens. Visible
// per-accepted-block streaming is a deliberate later refinement; for now a
// streaming client receives the full result promptly rather than incrementally.
//
// Sampling: the native MTP path is greedy-only. Sampled / probe-sink requests
// fall back to plain target decode (still correct, no speculative speedup) so
// the served model never errors on a temperature>0 request.
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
	return &speculativeTextModel{
		metaladapter: &metaladapter{model: targetMetal},
		pair:         pair,
		draftTokens:  MTPDefaultDraftTokens,
	}, nil
}

// Generate streams the target's tokens for a raw prompt via the MTP lane.
func (s *speculativeTextModel) Generate(ctx context.Context, prompt string, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	return s.speculativeStream(ctx, prompt, opts...)
}

// Chat formats the conversation with the model's native template, then streams
// via the MTP lane. The template is applied here because the MTP loop tokenises
// its prompt as-is (it has no chat-template step of its own).
func (s *speculativeTextModel) Chat(ctx context.Context, messages []inference.Message, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	prompt, err := s.ApplyChatTemplate(messages)
	if err != nil {
		s.model.SetLastErr(err)
		return func(func(inference.Token) bool) {}
	}
	return s.speculativeStream(ctx, prompt, opts...)
}

// speculativeStream runs the native MTP assistant decode for greedy requests and
// yields the verified tokens; sampled / probe-sink requests fall back to plain
// target decode so the served model never rejects a valid generation request.
func (s *speculativeTextModel) speculativeStream(ctx context.Context, prompt string, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	metalCfg := s.generateConfig(opts...)
	if !mtpGreedyCompatible(metalCfg) {
		return func(yield func(inference.Token) bool) {
			for token := range s.model.Generate(ctx, prompt, metalCfg) {
				if !yield(inference.Token{ID: token.ID, Text: token.Text}) {
					return
				}
			}
		}
	}
	return func(yield func(inference.Token) bool) {
		result, err := s.pair.Gemma4Assistant.Generate(ctx, s.model, prompt, metalCfg, s.draftTokens)
		if err != nil {
			// Gemma4AssistantPair.Generate records the error on the model;
			// metaladapter.Err() surfaces it after the iterator stops.
			return
		}
		for _, token := range result.Tokens {
			if !yield(inference.Token{ID: token.ID, Text: token.Text}) {
				return
			}
		}
	}
}

// mtpGreedyCompatible reports whether a request can use the native MTP assistant
// lane, which supports only greedy decoding and no probe sink. Mirrors gemma4's
// validateGemma4AssistantGenerateConfig so the fallback decision matches what
// the native path would accept.
func mtpGreedyCompatible(cfg metal.GenerateConfig) bool {
	return cfg.Temperature == 0 && cfg.TopK == 0 && cfg.TopP == 0 && cfg.MinP == 0 && cfg.RepeatPenalty <= 1 && cfg.ProbeSink == nil
}

// Close releases both the target and the attached assistant drafter.
func (s *speculativeTextModel) Close() error {
	return s.pair.Close()
}
