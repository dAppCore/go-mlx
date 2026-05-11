// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"time"

	core "dappco.re/go"
)

// DecodeGenerateFunc is the small generation hook used by optional decode
// optimisation experiments. It returns tokens so the harness can measure
// accepted and rejected candidates without depending on a concrete runtime.
type DecodeGenerateFunc func(context.Context, string, GenerateConfig) (DecodeGeneration, error)

// DecodeGeneration is a tokenised generation result used by speculative and
// prompt-lookup decode experiments.
type DecodeGeneration struct {
	Tokens  []Token `json:"tokens,omitempty"`
	Text    string  `json:"text,omitempty"`
	Metrics Metrics `json:"metrics,omitempty"`
}

// SpeculativeDecodeConfig configures the package-first speculative decode
// reference path. It is opt-in and benchmark-facing; native batch verification
// can replace the generate hooks later without changing the report shape.
type SpeculativeDecodeConfig struct {
	Prompt         string             `json:"prompt,omitempty"`
	MaxTokens      int                `json:"max_tokens,omitempty"`
	DraftTokens    int                `json:"draft_tokens,omitempty"`
	GenerateConfig GenerateConfig     `json:"generate_config,omitempty"`
	TargetGenerate DecodeGenerateFunc `json:"-"`
	DraftGenerate  DecodeGenerateFunc `json:"-"`
}

// PromptLookupDecodeConfig configures prompt lookup decoding over a known token
// sequence from repeated context. It is deliberately explicit: callers provide
// lookup tokens from their tokenizer/cache layer instead of relying on ad-hoc
// string splitting.
type PromptLookupDecodeConfig struct {
	Prompt         string             `json:"prompt,omitempty"`
	MaxTokens      int                `json:"max_tokens,omitempty"`
	GenerateConfig GenerateConfig     `json:"generate_config,omitempty"`
	TargetGenerate DecodeGenerateFunc `json:"-"`
	LookupTokens   []Token            `json:"lookup_tokens,omitempty"`
}

// DecodeOptimisationResult is the common report for speculative and
// prompt-lookup decode experiments.
type DecodeOptimisationResult struct {
	Mode    string                    `json:"mode"`
	Prompt  string                    `json:"prompt,omitempty"`
	Text    string                    `json:"text,omitempty"`
	Tokens  []Token                   `json:"tokens,omitempty"`
	Metrics DecodeOptimisationMetrics `json:"metrics"`
}

// DecodeOptimisationMetrics records candidate acceptance and call-level timing.
type DecodeOptimisationMetrics struct {
	TargetTokens   int           `json:"target_tokens,omitempty"`
	DraftTokens    int           `json:"draft_tokens,omitempty"`
	LookupTokens   int           `json:"lookup_tokens,omitempty"`
	AcceptedTokens int           `json:"accepted_tokens,omitempty"`
	RejectedTokens int           `json:"rejected_tokens,omitempty"`
	EmittedTokens  int           `json:"emitted_tokens,omitempty"`
	AcceptanceRate float64       `json:"acceptance_rate,omitempty"`
	TargetCalls    int           `json:"target_calls,omitempty"`
	DraftCalls     int           `json:"draft_calls,omitempty"`
	Duration       time.Duration `json:"duration,omitempty"`
	TargetDuration time.Duration `json:"target_duration,omitempty"`
	DraftDuration  time.Duration `json:"draft_duration,omitempty"`
}

const (
	DecodeModeSpeculative  = "speculative"
	DecodeModePromptLookup = "prompt_lookup"
)

// RunSpeculativeDecode compares draft-model candidates against target-model
// tokens and reports deterministic acceptance metrics. This is the safe
// reference API; it does not claim a speedup until a backend provides native
// verification that the benchmark can measure.
func RunSpeculativeDecode(ctx context.Context, cfg SpeculativeDecodeConfig) (DecodeOptimisationResult, error) {
	if cfg.TargetGenerate == nil {
		return DecodeOptimisationResult{}, core.NewError("mlx: speculative decode requires target generator")
	}
	if cfg.DraftGenerate == nil {
		return DecodeOptimisationResult{}, core.NewError("mlx: speculative decode requires draft generator")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	maxTokens := normaliseDecodeMaxTokens(cfg.MaxTokens, cfg.GenerateConfig.MaxTokens)
	targetCfg := cfg.GenerateConfig
	targetCfg.MaxTokens = maxTokens
	draftCfg := cfg.GenerateConfig
	draftCfg.MaxTokens = cfg.DraftTokens
	if draftCfg.MaxTokens <= 0 || draftCfg.MaxTokens > maxTokens {
		draftCfg.MaxTokens = maxTokens
	}

	start := time.Now()
	draftStart := time.Now()
	draft, err := cfg.DraftGenerate(ctx, cfg.Prompt, draftCfg)
	draftDuration := nonZeroDuration(time.Since(draftStart))
	if err != nil {
		return DecodeOptimisationResult{}, err
	}
	targetStart := time.Now()
	target, err := cfg.TargetGenerate(ctx, cfg.Prompt, targetCfg)
	targetDuration := nonZeroDuration(time.Since(targetStart))
	if err != nil {
		return DecodeOptimisationResult{}, err
	}
	result := buildDecodeAcceptanceResult(DecodeModeSpeculative, cfg.Prompt, target.Tokens, draft.Tokens, maxTokens)
	result.Metrics.TargetTokens = len(target.Tokens)
	result.Metrics.DraftTokens = len(draft.Tokens)
	result.Metrics.TargetCalls = 1
	result.Metrics.DraftCalls = 1
	result.Metrics.Duration = nonZeroDuration(time.Since(start))
	result.Metrics.TargetDuration = targetDuration
	result.Metrics.DraftDuration = draftDuration
	return result, nil
}

// RunPromptLookupDecode compares prompt-derived lookup candidates against the
// target stream and reports how often repeated-context tokens were reusable.
func RunPromptLookupDecode(ctx context.Context, cfg PromptLookupDecodeConfig) (DecodeOptimisationResult, error) {
	if cfg.TargetGenerate == nil {
		return DecodeOptimisationResult{}, core.NewError("mlx: prompt lookup decode requires target generator")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	maxTokens := normaliseDecodeMaxTokens(cfg.MaxTokens, cfg.GenerateConfig.MaxTokens)
	targetCfg := cfg.GenerateConfig
	targetCfg.MaxTokens = maxTokens
	start := time.Now()
	targetStart := time.Now()
	target, err := cfg.TargetGenerate(ctx, cfg.Prompt, targetCfg)
	targetDuration := nonZeroDuration(time.Since(targetStart))
	if err != nil {
		return DecodeOptimisationResult{}, err
	}
	result := buildDecodeAcceptanceResult(DecodeModePromptLookup, cfg.Prompt, target.Tokens, cfg.LookupTokens, maxTokens)
	result.Metrics.TargetTokens = len(target.Tokens)
	result.Metrics.LookupTokens = len(cfg.LookupTokens)
	result.Metrics.TargetCalls = 1
	result.Metrics.Duration = nonZeroDuration(time.Since(start))
	result.Metrics.TargetDuration = targetDuration
	return result, nil
}

func buildDecodeAcceptanceResult(mode, prompt string, target, candidates []Token, maxTokens int) DecodeOptimisationResult {
	limit := len(target)
	if maxTokens > 0 && maxTokens < limit {
		limit = maxTokens
	}
	out := make([]Token, 0, limit)
	var accepted, rejected int
	for i := 0; i < limit; i++ {
		targetToken := target[i]
		if i < len(candidates) {
			if decodeTokenEqual(candidates[i], targetToken) {
				out = append(out, cloneDecodeToken(candidates[i]))
				accepted++
				continue
			}
			rejected++
		}
		out = append(out, cloneDecodeToken(targetToken))
	}
	attempted := accepted + rejected
	metrics := DecodeOptimisationMetrics{
		AcceptedTokens: accepted,
		RejectedTokens: rejected,
		EmittedTokens:  len(out),
	}
	if attempted > 0 {
		metrics.AcceptanceRate = float64(accepted) / float64(attempted)
	}
	return DecodeOptimisationResult{
		Mode:    mode,
		Prompt:  prompt,
		Text:    decodeTokensText(out),
		Tokens:  out,
		Metrics: metrics,
	}
}

func normaliseDecodeMaxTokens(values ...int) int {
	for _, value := range values {
		if value > 0 {
			return value
		}
	}
	return DefaultGenerateConfig().MaxTokens
}

func decodeTokensText(tokens []Token) string {
	builder := core.NewBuilder()
	for _, token := range tokens {
		builder.WriteString(firstNonEmpty(token.Text, token.Value))
	}
	return builder.String()
}

func cloneDecodeTokens(tokens []Token) []Token {
	out := make([]Token, len(tokens))
	copy(out, tokens)
	return out
}

func cloneDecodeToken(token Token) Token {
	return Token{ID: token.ID, Value: token.Value, Text: token.Text}
}

func decodeTokenEqual(a, b Token) bool {
	if a.ID != b.ID {
		return false
	}
	aText := firstNonEmpty(a.Text, a.Value)
	bText := firstNonEmpty(b.Text, b.Value)
	if aText == "" || bText == "" {
		return true
	}
	return aText == bText
}
