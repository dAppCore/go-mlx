// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"iter"
	"sync"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/decode"
	"dappco.re/go/mlx/chat"
	"dappco.re/go/mlx/pkg/metal"
	"dappco.re/go/mlx/pkg/native"
	"dappco.re/go/mlx/pkg/tokenizer"
)

type nativeSpeculativeTextModel struct {
	*nativeTextModel
	draft           *nativeTextModel
	nativeAssistant *native.Gemma4AssistantPair
	draftTokens     int

	mtpMu      sync.Mutex
	mtpMetrics *metal.MTPMetrics
}

// LoadNativeSpeculativePairAsTextModel loads two no-cgo native text models and
// serves target generation through pkg/native's MTP target/draft verifier.
func LoadNativeSpeculativePairAsTextModel(targetPath, draftPath string, opts ...LoadOption) (inference.TextModel, error) {
	return LoadNativeSpeculativePairAsTextModelBlock(targetPath, draftPath, 0, opts...)
}

// LoadNativeSpeculativePairAsTextModelBlock is the native no-cgo sibling of
// LoadSpeculativePairAsTextModelBlock. Official Gemma 4 assistant-only drafter
// packs attach through pkg/native's assistant session verifier; full target/draft
// checkpoints use the generic native MTP path.
func LoadNativeSpeculativePairAsTextModelBlock(targetPath, draftPath string, draftBlock int, opts ...LoadOption) (inference.TextModel, error) {
	targetTM, err := LoadNativeTextModel(targetPath, opts...)
	if err != nil {
		return nil, err
	}
	target, ok := targetTM.(*nativeTextModel)
	if !ok {
		closeErr := targetTM.Close()
		return nil, core.ErrorJoin(core.NewError("mlx: native speculative target is not a native text model"), closeErr)
	}
	if draftBlock <= 0 {
		draftBlock = MTPDefaultDraftBlock
	}
	if isGemma4AssistantDraft(draftPath) {
		if err := validateNativeSpeculativeAssistantTokenizer(target.tok, draftPath); err != nil {
			closeErr := target.Close()
			return nil, core.ErrorJoin(err, closeErr)
		}
		assistant, err := native.LoadGemma4AssistantPairDirs(targetPath, draftPath)
		if err != nil {
			closeErr := target.Close()
			return nil, core.ErrorJoin(err, closeErr)
		}
		return &nativeSpeculativeTextModel{
			nativeTextModel: target,
			nativeAssistant: assistant,
			draftTokens:     max(1, draftBlock-1),
		}, nil
	}
	draftTM, err := LoadNativeTextModel(draftPath, opts...)
	if err != nil {
		closeErr := target.Close()
		return nil, core.ErrorJoin(err, closeErr)
	}
	draft, ok := draftTM.(*nativeTextModel)
	if !ok {
		closeErr := core.ErrorJoin(target.Close(), draftTM.Close())
		return nil, core.ErrorJoin(core.NewError("mlx: native speculative draft is not a native text model"), closeErr)
	}
	return &nativeSpeculativeTextModel{
		nativeTextModel: target,
		draft:           draft,
		draftTokens:     max(1, draftBlock-1),
	}, nil
}

func (s *nativeSpeculativeTextModel) Generate(ctx context.Context, prompt string, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	cfg := inference.ApplyGenerateOpts(opts)
	return s.nativeSpeculativeStream(ctx, prompt, cfg)
}

func (s *nativeSpeculativeTextModel) Chat(ctx context.Context, messages []inference.Message, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	if s == nil || s.nativeTextModel == nil {
		return func(func(inference.Token) bool) {}
	}
	if inferenceMessagesCarryImages(messages) {
		return s.nativeTextModel.Chat(ctx, messages, opts...)
	}
	cfg := inference.ApplyGenerateOpts(opts)
	return s.nativeSpeculativeStream(ctx, chat.Format(messages, s.chatConfig(cfg)), cfg)
}

func (s *nativeSpeculativeTextModel) nativeSpeculativeStream(ctx context.Context, prompt string, cfg inference.GenerateConfig) iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {
		if s == nil || s.nativeTextModel == nil || (s.draft == nil && s.nativeAssistant == nil) {
			return
		}
		start := time.Now()
		rootCfg := nativeSpeculativeRootGenerateConfig(cfg)
		emit := func(token decode.Token) bool {
			return yield == nil || yield(inference.Token{ID: token.ID, Text: token.Text})
		}
		var (
			result  SpeculativeDecodeResult
			handled bool
			err     error
		)
		if s.nativeAssistant != nil {
			if !nativeGemma4AssistantEligible(rootCfg, s.nativeTextModel) {
				for token := range s.nativeTextModel.stream(ctx, s.tok.Encode(prompt), cfg) {
					if yield != nil && !yield(token) {
						return
					}
				}
				s.setNativeMTPMetrics(nil)
				return
			}
			result, handled, err = s.nativeTextModel.GenerateNativeGemma4AssistantEach(ctx, s.nativeAssistant, prompt, rootCfg, s.draftTokens, emit)
		} else {
			result, handled, err = s.nativeTextModel.GenerateNativeSpeculativeEach(ctx, s.draft, prompt, SpeculativeDecodeConfig{
				MaxTokens:      cfg.MaxTokens,
				DraftTokens:    s.draftTokens,
				GenerateConfig: rootCfg,
			}, emit)
		}
		if err != nil {
			s.nativeTextModel.setErr(err)
			return
		}
		if !handled {
			for token := range s.nativeTextModel.stream(ctx, s.tok.Encode(prompt), cfg) {
				if yield != nil && !yield(token) {
					return
				}
			}
			s.setNativeMTPMetrics(nil)
			return
		}
		if result.Metrics.Duration == 0 {
			result.Metrics.Duration = time.Since(start)
		}
		s.nativeTextModel.setMetrics(len(s.tok.Encode(prompt)), len(result.Tokens), result.Metrics.Duration)
		s.setNativeMTPMetrics(nativeMTPMetricsFromDecode(result.Metrics))
	}
}

func nativeSpeculativeRootGenerateConfig(cfg inference.GenerateConfig) GenerateConfig {
	return GenerateConfig{
		MaxTokens:     cfg.MaxTokens,
		Temperature:   cfg.Temperature,
		TopK:          cfg.TopK,
		TopP:          cfg.TopP,
		ReturnLogits:  cfg.ReturnLogits,
		StopTokens:    cfg.StopTokens,
		RepeatPenalty: cfg.RepeatPenalty,
	}
}

func nativeGemma4AssistantEligible(cfg GenerateConfig, target *nativeTextModel) bool {
	return cfg.RepeatPenalty <= 1 && nativeTextModelProbeSink(target) == nil
}

func nativeTextModelProbeSink(target *nativeTextModel) inference.ProbeSink {
	if target == nil {
		return nil
	}
	target.schedulerMu.Lock()
	defer target.schedulerMu.Unlock()
	return target.probeSink
}

func validateNativeSpeculativeAssistantTokenizer(target *tokenizer.Tokenizer, assistantPath string) error {
	if target == nil {
		return errMLXSpeculativeTokenizersRequired
	}
	assistant, err := tokenizer.LoadTokenizer(core.PathJoin(assistantPath, "tokenizer.json"))
	if err != nil {
		return core.E("mlx.LoadNativeSpeculativePairAsTextModel", "load assistant tokenizer", err)
	}
	for _, probe := range speculativeTokenizerProbes(nil) {
		if !int32SlicesEqual(target.Encode(probe), assistant.Encode(probe)) {
			return errMLXSpeculativeTokenizersDiffer
		}
	}
	return nil
}

func nativeMTPMetricsFromDecode(metrics SpeculativeDecodeMetrics) *metal.MTPMetrics {
	return &metal.MTPMetrics{
		ProposedTokens:       metrics.DraftTokens,
		AcceptedTokens:       metrics.AcceptedTokens,
		RejectedTokens:       metrics.RejectedTokens,
		TargetVerifyCalls:    metrics.TargetCalls,
		TargetCalls:          metrics.TargetCalls,
		DraftCalls:           metrics.DraftCalls,
		AcceptanceRate:       metrics.AcceptanceRate,
		VisibleTokensPerSec:  tokensPerSecond(metrics.EmittedTokens, metrics.Duration),
		TargetTokensPerSec:   tokensPerSecond(metrics.TargetTokens, metrics.TargetDuration),
		WallDuration:         metrics.Duration,
		TargetVerifyDuration: metrics.TargetDuration,
		TargetDuration:       metrics.TargetDuration,
		DraftDuration:        metrics.DraftDuration,
	}
}

func tokensPerSecond(tokens int, d time.Duration) float64 {
	if tokens <= 0 || d <= 0 {
		return 0
	}
	return float64(tokens) / d.Seconds()
}

func (s *nativeSpeculativeTextModel) setNativeMTPMetrics(metrics *metal.MTPMetrics) {
	s.mtpMu.Lock()
	s.mtpMetrics = metrics
	s.mtpMu.Unlock()
}

func (s *nativeSpeculativeTextModel) MTPMetrics() *metal.MTPMetrics {
	if s == nil {
		return nil
	}
	s.mtpMu.Lock()
	defer s.mtpMu.Unlock()
	if s.mtpMetrics == nil {
		return nil
	}
	clone := *s.mtpMetrics
	if len(clone.DraftTokenSchedule) > 0 {
		clone.DraftTokenSchedule = append([]int(nil), clone.DraftTokenSchedule...)
	}
	return &clone
}

func (s *nativeSpeculativeTextModel) Close() error {
	if s == nil {
		return nil
	}
	var err error
	if s.nativeTextModel != nil {
		err = core.ErrorJoin(err, s.nativeTextModel.Close())
	}
	if s.draft != nil && s.draft != s.nativeTextModel {
		err = core.ErrorJoin(err, s.draft.Close())
	}
	if s.nativeAssistant != nil {
		err = core.ErrorJoin(err, s.nativeAssistant.Close())
	}
	return err
}
