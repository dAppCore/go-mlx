// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	"slices"

	core "dappco.re/go"
	"dappco.re/go/inference/decode"
	"dappco.re/go/mlx/internal/metal"
	modelinspect "dappco.re/go/mlx/model"
)

// SpeculativeDecodeResult is the target/draft accept-reject report shared with
// the portable go-inference decode harness.
type SpeculativeDecodeResult = decode.Result

// SpeculativeDecodeMetrics records proposed, accepted, rejected, and timing
// counters for a target/draft decode attempt.
type SpeculativeDecodeMetrics = decode.Metrics

// SpeculativeDecodeConfig configures the package-first target/draft reference
// path. Native block verification is intentionally separate from this API.
type SpeculativeDecodeConfig struct {
	MaxTokens      int
	DraftTokens    int
	GenerateConfig GenerateConfig
}

// SpeculativePairConfig configures loading a target model beside a drafter.
type SpeculativePairConfig struct {
	TargetOptions  []LoadOption
	DraftOptions   []LoadOption
	TokenizerProbe []string
}

// SpeculativePairReport records the compatibility checks for a loaded pair.
type SpeculativePairReport struct {
	Target         ModelInfo `json:"target"`
	Draft          ModelInfo `json:"draft"`
	TokenizerProbe []string  `json:"tokenizer_probe,omitempty"`
}

// SpeculativePair owns a target model and an assistant/draft model.
type SpeculativePair struct {
	Target          *Model
	Draft           *Model
	Gemma4Assistant *metal.Gemma4AssistantPair
	Report          SpeculativePairReport
}

type nativeGemma4AssistantAttacher interface {
	AttachGemma4Assistant(string) (*metal.Gemma4AssistantPair, error)
}

type nativeGemma4AssistantGenerator interface {
	GenerateGemma4Assistant(context.Context, *metal.Gemma4AssistantPair, string, metal.GenerateConfig, int) (metal.Gemma4AssistantGenerateResult, error)
}

var (
	inspectSpeculativeDraftModelPack = modelinspect.Inspect
	attachGemma4AssistantDraft       = attachGemma4AssistantDraftToTarget
)

// Per-request hot-path error sentinels — these fire from the public
// SpeculativePair.Generate / Model.GenerateSpeculative entries on every
// invocation that misses a precondition. Hoisting to package level drops
// the per-call core.NewError alloc on the (target nil / draft nil / pair
// nil / target runtime missing) paths.
var (
	errMLXSpeculativeTargetNil          = core.NewError("mlx: target model is nil")
	errMLXSpeculativeDraftNil           = core.NewError("mlx: draft model is nil")
	errMLXSpeculativeMaxNeg             = core.NewError("mlx: speculative max tokens must be >= 0")
	errMLXSpeculativeDraftTokensNeg     = core.NewError("mlx: speculative draft tokens must be >= 0")
	errMLXSpeculativePairNil            = core.NewError("mlx: speculative pair is nil")
	errMLXSpeculativeGemma4Unsupp       = core.NewError("mlx: target runtime cannot run Gemma 4 assistant generation")
	errMLXSpeculativeGemma4Attach       = core.NewError("mlx: target runtime cannot attach Gemma 4 assistant")
	errMLXSpeculativeTargetPathRequired = core.NewError("mlx: speculative target path is required")
	errMLXSpeculativeDraftPathRequired  = core.NewError("mlx: speculative draft path is required")
	errMLXSpeculativeValidateTargetNil  = core.NewError("mlx: speculative target model is nil")
	errMLXSpeculativeValidateDraftNil   = core.NewError("mlx: speculative draft model is nil")
	errMLXSpeculativeVocabMismatch      = core.NewError("mlx: speculative target and draft vocab sizes differ")
	errMLXSpeculativeTokenizersRequired = core.NewError("mlx: speculative target and draft tokenizers are required")
	errMLXSpeculativeTokenizersDiffer   = core.NewError("mlx: speculative target and draft tokenizers differ")
	errMLXSpeculativeAssistantNil       = core.NewError("mlx: speculative Gemma 4 assistant is nil")
	errMLXSpeculativeTokenizerNil       = core.NewError("mlx: speculative tokenizer is nil")
	errMLXSpeculativeTokenizerProbeFail = core.NewError("mlx: speculative tokenizer probe failed")
)

// GenerateSpeculative runs the portable target/draft speculative decode
// reference path and returns acceptance metrics. It does not yet claim a native
// MTP speedup; production visible-throughput work still needs backend block
// verification.
func (m *Model) GenerateSpeculative(ctx context.Context, draft *Model, prompt string, cfg SpeculativeDecodeConfig) (SpeculativeDecodeResult, error) {
	if m == nil || m.model == nil {
		return SpeculativeDecodeResult{}, errMLXSpeculativeTargetNil
	}
	if draft == nil || draft.model == nil {
		return SpeculativeDecodeResult{}, errMLXSpeculativeDraftNil
	}
	if cfg.MaxTokens < 0 {
		return SpeculativeDecodeResult{}, errMLXSpeculativeMaxNeg
	}
	if cfg.DraftTokens < 0 {
		return SpeculativeDecodeResult{}, errMLXSpeculativeDraftTokensNeg
	}
	if ctx == nil {
		ctx = context.Background()
	}
	generateCfg := cfg.GenerateConfig
	if generateCfg.MaxTokens == 0 {
		generateCfg = DefaultGenerateConfig()
	}
	maxTokens := cfg.MaxTokens
	if maxTokens == 0 {
		maxTokens = generateCfg.MaxTokens
	}
	// Share generateCfg by pointer across both pooled generators — both
	// target + draft acquire from modelDecodeGeneratorPool and point at
	// the same heap-resident GenerateConfig. Direct acquire/release
	// (defer) — a release-closure would re-allocate per call and undo
	// the structurally-pooled win this lane lands.
	target := acquireModelDecodeGenerator(m, &generateCfg)
	defer releaseModelDecodeGenerator(target)
	draftGen := acquireModelDecodeGenerator(draft, &generateCfg)
	defer releaseModelDecodeGenerator(draftGen)
	return decode.Speculative(ctx, decode.SpeculativeConfig{
		Prompt:         prompt,
		MaxTokens:      maxTokens,
		DraftTokens:    cfg.DraftTokens,
		GenerateConfig: decode.GenerateConfig{MaxTokens: maxTokens},
		TargetGenerate: target,
		DraftGenerate:  draftGen,
	})
}

// LoadSpeculativePair loads a target model and its assistant/drafter, then
// validates the shared tokenizer surface required by speculative decoding.
func LoadSpeculativePair(targetPath, draftPath string, cfg SpeculativePairConfig) (*SpeculativePair, error) {
	if core.Trim(targetPath) == "" {
		return nil, errMLXSpeculativeTargetPathRequired
	}
	if core.Trim(draftPath) == "" {
		return nil, errMLXSpeculativeDraftPathRequired
	}
	target, err := LoadModel(targetPath, cfg.TargetOptions...)
	if err != nil {
		return nil, err
	}
	if isGemma4AssistantDraft(draftPath) {
		assistant, err := attachGemma4AssistantDraft(target.model, draftPath)
		if err != nil {
			if closeErr := target.Close(); closeErr != nil {
				err = core.ErrorJoin(err, closeErr)
			}
			return nil, err
		}
		pair := &SpeculativePair{Target: target, Gemma4Assistant: assistant}
		report, err := validateSpeculativeGemma4AssistantPair(target, assistant, cfg.TokenizerProbe)
		if err != nil {
			if closeErr := pair.Close(); closeErr != nil {
				err = core.ErrorJoin(err, closeErr)
			}
			return nil, err
		}
		pair.Report = report
		return pair, nil
	}
	draft, err := LoadModel(draftPath, cfg.DraftOptions...)
	if err != nil {
		if closeErr := target.Close(); closeErr != nil {
			err = core.ErrorJoin(err, closeErr)
		}
		return nil, err
	}
	pair := &SpeculativePair{Target: target, Draft: draft}
	report, err := validateSpeculativePair(target, draft, cfg.TokenizerProbe)
	if err != nil {
		if closeErr := pair.Close(); closeErr != nil {
			err = core.ErrorJoin(err, closeErr)
		}
		return nil, err
	}
	pair.Report = report
	return pair, nil
}

// Generate runs the pair through the package-first speculative reference path.
func (pair *SpeculativePair) Generate(ctx context.Context, prompt string, cfg SpeculativeDecodeConfig) (SpeculativeDecodeResult, error) {
	if pair == nil {
		return SpeculativeDecodeResult{}, errMLXSpeculativePairNil
	}
	if pair.Gemma4Assistant != nil {
		generator, ok := pair.Target.model.(nativeGemma4AssistantGenerator)
		if !ok {
			return SpeculativeDecodeResult{}, errMLXSpeculativeGemma4Unsupp
		}
		generateCfg := cfg.GenerateConfig
		if generateCfg.MaxTokens == 0 {
			generateCfg = DefaultGenerateConfig()
		}
		maxTokens := cfg.MaxTokens
		if maxTokens <= 0 {
			maxTokens = generateCfg.MaxTokens
		}
		generateCfg.MaxTokens = maxTokens
		draftTokens := cfg.DraftTokens
		if draftTokens <= 0 {
			draftTokens = 1
		}
		result, err := generator.GenerateGemma4Assistant(ctx, pair.Gemma4Assistant, prompt, toMetalGenerateConfig(generateCfg), draftTokens)
		if err != nil {
			return SpeculativeDecodeResult{}, err
		}
		return gemma4AssistantGenerateResultToDecode(prompt, result), nil
	}
	return pair.Target.GenerateSpeculative(ctx, pair.Draft, prompt, cfg)
}

// Metrics returns the target model's latest counters for a target/draft pair.
func (pair *SpeculativePair) Metrics() Metrics {
	if pair == nil || pair.Target == nil {
		return Metrics{}
	}
	return pair.Target.Metrics()
}

// Err returns the target model's latest generation error for a target/draft pair.
func (pair *SpeculativePair) Err() error {
	if pair == nil || pair.Target == nil {
		return nil
	}
	return pair.Target.Err()
}

// Close releases both models owned by the pair.
func (pair *SpeculativePair) Close() error {
	if pair == nil {
		return nil
	}
	var err error
	if pair.Target != nil {
		err = core.ErrorJoin(err, pair.Target.Close())
	}
	if pair.Draft != nil && pair.Draft != pair.Target {
		err = core.ErrorJoin(err, pair.Draft.Close())
	}
	if pair.Gemma4Assistant != nil {
		err = core.ErrorJoin(err, pair.Gemma4Assistant.Close())
	}
	return err
}

func isGemma4AssistantDraft(draftPath string) bool {
	pack, err := inspectSpeculativeDraftModelPack(draftPath)
	if err != nil {
		return false
	}
	return pack.Architecture == "gemma4_assistant"
}

func attachGemma4AssistantDraftToTarget(target nativeModel, draftPath string) (*metal.Gemma4AssistantPair, error) {
	attacher, ok := target.(nativeGemma4AssistantAttacher)
	if !ok {
		return nil, errMLXSpeculativeGemma4Attach
	}
	return attacher.AttachGemma4Assistant(draftPath)
}

func gemma4AssistantGenerateResultToDecode(prompt string, result metal.Gemma4AssistantGenerateResult) decode.Result {
	emitted := len(result.Tokens)
	tokens := make([]decode.Token, emitted)
	// Per-field assignment — the prior `decode.Token{ID, Text}` literal
	// emitted redundant zero writes to the Value field (the struct
	// literal zeroes every field then overwrites named ones), then a
	// runtime.wbZero call for the string header before the write-barrier
	// copy. makeslice already zeroes the destination, so writing only
	// ID + Text directly skips the zero work on long generations.
	src := result.Tokens
	for i := range src {
		tokens[i].ID = src[i].ID
		tokens[i].Text = src[i].Text
	}
	var acceptanceRate float64
	if result.DraftTokens > 0 {
		acceptanceRate = float64(result.AcceptedTokens) / float64(result.DraftTokens)
	}
	return decode.Result{
		Mode:   decode.ModeSpeculative,
		Prompt: prompt,
		Text:   result.Text,
		Tokens: tokens,
		Metrics: decode.Metrics{
			TargetTokens:   result.TargetTokens,
			DraftTokens:    result.DraftTokens,
			AcceptedTokens: result.AcceptedTokens,
			RejectedTokens: result.RejectedTokens,
			EmittedTokens:  emitted,
			AcceptanceRate: acceptanceRate,
			TargetCalls:    result.TargetCalls,
			DraftCalls:     result.DraftCalls,
			Duration:       result.Duration,
			TargetDuration: result.TargetDuration,
			DraftDuration:  result.DraftDuration,
		},
	}
}

func validateSpeculativePair(target, draft *Model, probes []string) (SpeculativePairReport, error) {
	if target == nil || target.model == nil {
		return SpeculativePairReport{}, errMLXSpeculativeValidateTargetNil
	}
	if draft == nil || draft.model == nil {
		return SpeculativePairReport{}, errMLXSpeculativeValidateDraftNil
	}
	report := SpeculativePairReport{
		Target: target.Info(),
		Draft:  draft.Info(),
	}
	if report.Target.VocabSize > 0 && report.Draft.VocabSize > 0 && report.Target.VocabSize != report.Draft.VocabSize {
		return report, errMLXSpeculativeVocabMismatch
	}
	targetTokenizer := target.Tokenizer()
	draftTokenizer := draft.Tokenizer()
	if targetTokenizer == nil || targetTokenizer.tok == nil || draftTokenizer == nil || draftTokenizer.tok == nil {
		return report, errMLXSpeculativeTokenizersRequired
	}
	report.TokenizerProbe = speculativeTokenizerProbes(probes)
	for _, probe := range report.TokenizerProbe {
		targetTokens, err := encodeSpeculativeProbe(targetTokenizer, probe)
		if err != nil {
			return report, err
		}
		draftTokens, err := encodeSpeculativeProbe(draftTokenizer, probe)
		if err != nil {
			return report, err
		}
		if !int32SlicesEqual(targetTokens, draftTokens) {
			return report, errMLXSpeculativeTokenizersDiffer
		}
	}
	return report, nil
}

func validateSpeculativeGemma4AssistantPair(target *Model, assistant *metal.Gemma4AssistantPair, probes []string) (SpeculativePairReport, error) {
	if target == nil || target.model == nil {
		return SpeculativePairReport{}, errMLXSpeculativeValidateTargetNil
	}
	if assistant == nil || assistant.Assistant == nil {
		return SpeculativePairReport{}, errMLXSpeculativeAssistantNil
	}
	report := SpeculativePairReport{
		Target: target.Info(),
		Draft:  gemma4AssistantModelInfo(assistant.Assistant),
	}
	if report.Target.VocabSize > 0 && report.Draft.VocabSize > 0 && report.Target.VocabSize != report.Draft.VocabSize {
		return report, errMLXSpeculativeVocabMismatch
	}
	targetTokenizer := target.Tokenizer()
	draftTokenizer := &Tokenizer{tok: assistant.Assistant.Tokenizer()}
	if targetTokenizer == nil || targetTokenizer.tok == nil || draftTokenizer.tok == nil {
		return report, errMLXSpeculativeTokenizersRequired
	}
	report.TokenizerProbe = speculativeTokenizerProbes(probes)
	for _, probe := range report.TokenizerProbe {
		targetTokens, err := encodeSpeculativeProbe(targetTokenizer, probe)
		if err != nil {
			return report, err
		}
		draftTokens, err := encodeSpeculativeProbe(draftTokenizer, probe)
		if err != nil {
			return report, err
		}
		if !int32SlicesEqual(targetTokens, draftTokens) {
			return report, errMLXSpeculativeTokenizersDiffer
		}
	}
	return report, nil
}

func gemma4AssistantModelInfo(assistant *metal.Gemma4AssistantModel) ModelInfo {
	info := ModelInfo{Architecture: "gemma4_assistant"}
	if assistant == nil || assistant.Cfg == nil {
		return info
	}
	info.VocabSize = int(assistant.Cfg.VocabSize)
	info.NumLayers = assistant.NumLayers()
	info.HiddenSize = int(assistant.Cfg.HiddenSize)
	info.ContextLength = int(assistant.Cfg.MaxPositionEmbeddings)
	if assistant.Cfg.Quantization != nil {
		info.QuantBits = assistant.Cfg.Quantization.Bits
		info.QuantGroup = assistant.Cfg.Quantization.GroupSize
	}
	return info
}

func encodeSpeculativeProbe(tok *Tokenizer, probe string) (tokens []int32, err error) {
	if tok == nil || tok.tok == nil {
		return nil, errMLXSpeculativeTokenizerNil
	}
	defer func() {
		if r := recover(); r != nil {
			err = errMLXSpeculativeTokenizerProbeFail
			tokens = nil
		}
	}()
	return tok.Encode(probe)
}

// defaultSpeculativeTokenizerProbes is the shared default probe set
// returned by speculativeTokenizerProbes when the caller passes nil/
// empty probes. Hoisted to package level so each LoadSpeculativePair
// call returns the same slice header instead of rebuilding a 3-string
// literal on every invocation.
var defaultSpeculativeTokenizerProbes = []string{"hello", "The quick brown fox", "Answer in one short sentence."}

func speculativeTokenizerProbes(probes []string) []string {
	if len(probes) == 0 {
		return defaultSpeculativeTokenizerProbes
	}
	out := make([]string, len(probes))
	copy(out, probes)
	return out
}

func int32SlicesEqual(a, b []int32) bool {
	return slices.Equal(a, b)
}
