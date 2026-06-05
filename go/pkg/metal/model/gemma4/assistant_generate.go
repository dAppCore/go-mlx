// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"context"
	"slices"
	"time"

	core "dappco.re/go"
	"dappco.re/go/mlx/pkg/metal"
)

// gemma4AssistantDefaultDraftTokens mirrors the production MTP default without
// making pkg/metal depend on its parent package.
const gemma4AssistantDefaultDraftTokens = 2

// Gemma4AssistantGenerateResult records one metal.Greedy MTP generation run.
type Gemma4AssistantGenerateResult struct {
	Tokens               []metal.Token
	Text                 string
	PromptTokens         int
	TargetTokens         int
	DraftTokens          int
	AcceptedTokens       int
	RejectedTokens       int
	TargetVerifyCalls    int
	TargetCalls          int
	DraftCalls           int
	DraftTokenSchedule   []int
	Duration             time.Duration
	PrefillDuration      time.Duration
	FirstTokenDuration   time.Duration
	TargetVerifyDuration time.Duration
	TargetDuration       time.Duration
	DraftDuration        time.Duration
}

// GenerateGemma4Assistant runs a conservative metal.Greedy MTP generation loop over
// an attached Gemma 4 assistant pair. Sampling-aware verification is kept out
// until the metal.Greedy accept/reject path is benchmarked.
// Generate runs a conservative greedy MTP generation loop over this attached
// Gemma 4 assistant pair, driving the supplied target runtime m. Sampling-aware
// verification is kept out until the greedy accept/reject path is benchmarked.
func (pair *Gemma4AssistantPair) Generate(ctx context.Context, m *metal.Model, prompt string, cfg metal.GenerateConfig, draftTokens int) (Gemma4AssistantGenerateResult, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if cfg.MaxTokens <= 0 {
		// Model-declared default_output_length, else context length — never the
		// 256 literal codex used over the model's own declared value.
		info := m.Info()
		if cfg.MaxTokens = info.DefaultOutputLength; cfg.MaxTokens <= 0 {
			cfg.MaxTokens = info.ContextLength
		}
	}
	draftTokens = gemma4AssistantResolveDraftTokens(draftTokens)
	if err := validateGemma4AssistantGenerateConfig(cfg); err != nil {
		return Gemma4AssistantGenerateResult{}, err
	}
	if err := m.RequireTextRuntime("Model.GenerateGemma4Assistant"); err != nil {
		return Gemma4AssistantGenerateResult{}, err
	}
	if pair == nil || pair.Target == nil || pair.Assistant == nil {
		return Gemma4AssistantGenerateResult{}, core.NewError("gemma4.assistant generation requires an attached pair")
	}
	target, ok := m.UnderlyingModel().(*Gemma4Model)
	if !ok || target != pair.Target {
		return Gemma4AssistantGenerateResult{}, core.NewError("gemma4.assistant generation pair does not match target runtime")
	}

	m.SetLastErr(nil)
	m.SetLastMetrics(metal.Metrics{})
	release, err := m.AcquireSlot(ctx)
	if err != nil {
		m.SetLastErr(err)
		return Gemma4AssistantGenerateResult{}, err
	}
	defer release()
	releasePromptCache := m.AcquirePromptCache()
	defer releasePromptCache()

	var result Gemma4AssistantGenerateResult
	if deviceErr := m.WithDevice(func() {
		result, err = generateGemma4Assistant(ctx, m, pair, prompt, cfg, draftTokens)
	}); deviceErr != nil {
		err = deviceErr
	}
	if err != nil {
		m.SetLastErr(err)
	}
	return result, err
}

func gemma4AssistantResolveDraftTokens(draftTokens int) int {
	if draftTokens <= 0 {
		return gemma4AssistantDefaultDraftTokens
	}
	return draftTokens
}

func validateGemma4AssistantGenerateConfig(cfg metal.GenerateConfig) error {
	if cfg.Temperature != 0 || cfg.TopK != 0 || cfg.TopP != 0 || cfg.MinP != 0 || cfg.RepeatPenalty > 1 {
		return core.NewError("gemma4.assistant generation currently supports metal.Greedy decoding only")
	}
	if cfg.ProbeSink != nil {
		return core.NewError("gemma4.assistant generation does not support probe sinks yet")
	}
	return nil
}

func generateGemma4Assistant(ctx context.Context, m *metal.Model, pair *Gemma4AssistantPair, prompt string, cfg metal.GenerateConfig, draftTokens int) (Gemma4AssistantGenerateResult, error) {
	start := time.Now()
	metal.ResetPeakMemory()
	promptTokens := m.RuntimeTokenizer().Encode(prompt)
	if len(promptTokens) == 0 {
		return Gemma4AssistantGenerateResult{}, core.NewError("Model.GenerateGemma4Assistant: empty prompt after tokenisation")
	}
	prepared, err := prepareGemma4AssistantPrompt(ctx, m, pair, promptTokens, cfg)
	if err != nil {
		return Gemma4AssistantGenerateResult{}, err
	}
	caches := prepared.Caches
	logits := prepared.Logits
	hidden := prepared.Hidden
	defer func() { metal.FreeCaches(caches) }()
	defer metal.Free(logits, hidden)

	result := Gemma4AssistantGenerateResult{
		PromptTokens:    len(promptTokens),
		PrefillDuration: prepared.Duration,
	}
	if draftTokens > 0 {
		result.DraftTokenSchedule = make([]int, 0, (cfg.MaxTokens+draftTokens-1)/draftTokens)
	}
	lastToken := promptTokens[len(promptTokens)-1]
	stopped := false
	for len(result.Tokens) < cfg.MaxTokens && !stopped {
		select {
		case <-ctx.Done():
			return result, ctx.Err()
		default:
		}

		remaining := cfg.MaxTokens - len(result.Tokens)
		blockSize := min(draftTokens, remaining)
		draftStart := time.Now()
		draft, err := pair.DraftBlockWithSuppression(lastToken, hidden, caches, blockSize, cfg.SuppressTokens)
		result.DraftDuration += time.Since(draftStart)
		result.DraftCalls++
		if err != nil {
			return result, err
		}
		result.DraftTokens += len(draft.Tokens)
		result.DraftTokenSchedule = append(result.DraftTokenSchedule, blockSize)

		targetStart := time.Now()
		verify, err := pair.VerifyDraftBlockWithSuppression(logits, draft.Tokens, caches, cfg.SuppressTokens)
		verifyDuration := time.Since(targetStart)
		result.TargetVerifyDuration += verifyDuration
		result.TargetDuration += verifyDuration
		result.TargetVerifyCalls++
		result.TargetCalls++
		draft.Close()
		if err != nil {
			return result, err
		}

		for _, id := range verify.AcceptedTokens {
			stops := appendGemma4AssistantToken(m, &result, id, cfg)
			recordGemma4AssistantFirstToken(&result, start)
			if stops {
				stopped = true
				break
			}
			lastToken = id
		}
		result.AcceptedTokens += verify.AcceptedCount
		result.RejectedTokens += verify.RejectedCount
		result.TargetTokens += verify.AcceptedCount

		if stopped {
			verify.Close()
			break
		}

		nextCaches := verify.Caches
		nextLogits := verify.Logits
		nextHidden := verify.Hidden
		verify.Caches = nil
		verify.Logits = nil
		verify.Hidden = nil

		metal.FreeCaches(caches)
		caches = nextCaches
		metal.Free(logits, hidden)
		logits = nextLogits
		hidden = nextHidden

		if !verify.AllAccepted {
			replacement := verify.ReplacementToken
			stops := appendGemma4AssistantToken(m, &result, replacement, cfg)
			recordGemma4AssistantFirstToken(&result, start)
			if stops {
				lastToken = replacement
				stopped = true
				verify.Close()
				break
			}
			lastToken = replacement
			result.TargetTokens++

			targetStart = time.Now()
			nextLogits, nextHidden, err := pair.forwardGemma4AssistantAcceptedToken(replacement, caches)
			result.TargetDuration += time.Since(targetStart)
			result.TargetCalls++
			if err != nil {
				verify.Close()
				return result, err
			}
			metal.Free(logits, hidden)
			logits = nextLogits
			hidden = nextHidden
		}
		verify.Close()
	}

	result.Duration = time.Since(start)
	if result.Duration <= 0 {
		result.Duration = time.Nanosecond
	}
	decodeDuration := result.Duration - result.PrefillDuration
	if decodeDuration <= 0 {
		decodeDuration = time.Nanosecond
	}
	processMemory := metal.GetProcessMemory()
	metrics := metal.Metrics{
		PromptTokens:               result.PromptTokens,
		GeneratedTokens:            len(result.Tokens),
		PrefillDuration:            result.PrefillDuration,
		FirstTokenDuration:         result.FirstTokenDuration,
		DecodeDuration:             decodeDuration,
		TotalDuration:              result.Duration,
		PeakMemoryBytes:            metal.GetPeakMemory(),
		ActiveMemoryBytes:          metal.GetActiveMemory(),
		CacheMemoryBytes:           metal.GetCacheMemory(),
		ProcessVirtualMemoryBytes:  processMemory.VirtualMemoryBytes,
		ProcessResidentMemoryBytes: processMemory.ResidentMemoryBytes,
		ProcessPeakResidentBytes:   processMemory.PeakResidentMemoryBytes,
		Adapter:                    m.Adapter(),
		PromptCacheHitTokens:       prepared.CacheHitTokens,
		PromptCacheMissTokens:      prepared.CacheMissTokens,
		PromptCacheRestoreDuration: prepared.RestoreDuration,
	}
	if prepared.CacheHit {
		metrics.PromptCacheHits = 1
	} else {
		metrics.PromptCacheMisses = 1
	}
	if result.PrefillDuration > 0 {
		metrics.PrefillTokensPerSec = float64(len(promptTokens)) / result.PrefillDuration.Seconds()
	}
	if decodeDuration > 0 {
		metrics.DecodeTokensPerSec = float64(len(result.Tokens)) / decodeDuration.Seconds()
	}
	if result.DraftCalls > 0 || result.DraftTokens > 0 {
		var acceptanceRate float64
		if result.DraftTokens > 0 {
			acceptanceRate = float64(result.AcceptedTokens) / float64(result.DraftTokens)
		}
		var visibleTokensPerSec float64
		if result.Duration > 0 {
			visibleTokensPerSec = float64(len(result.Tokens)) / result.Duration.Seconds()
		}
		var targetTokensPerSec float64
		if result.TargetDuration > 0 {
			targetTokensPerSec = float64(result.TargetTokens) / result.TargetDuration.Seconds()
		}
		metrics.MTP = &metal.MTPMetrics{
			DraftTokenSchedule:     slices.Clone(result.DraftTokenSchedule),
			ProposedTokens:         result.DraftTokens,
			AcceptedTokens:         result.AcceptedTokens,
			RejectedTokens:         result.RejectedTokens,
			TargetVerifyCalls:      result.TargetVerifyCalls,
			TargetCalls:            result.TargetCalls,
			DraftCalls:             result.DraftCalls,
			AcceptanceRate:         acceptanceRate,
			VisibleTokensPerSec:    visibleTokensPerSec,
			TargetTokensPerSec:     targetTokensPerSec,
			WarmDecodeTokensPerSec: metrics.DecodeTokensPerSec,
			WallDuration:           result.Duration,
			RestoreDuration:        prepared.RestoreDuration,
			TargetVerifyDuration:   result.TargetVerifyDuration,
			TargetDuration:         result.TargetDuration,
			DraftDuration:          result.DraftDuration,
			PeakMemoryBytes:        metrics.PeakMemoryBytes,
		}
	}
	m.SetLastMetrics(metrics)
	return result, nil
}

func prefillGemma4AssistantPrompt(ctx context.Context, m *metal.Model, pair *Gemma4AssistantPair, tokens []int32, caches []metal.Cache) (*metal.Array, *metal.Array, error) {
	if len(tokens) == 0 {
		return nil, nil, core.NewError("Model.GenerateGemma4Assistant: empty prompt after tokenisation")
	}
	chunkSize := m.PrefillChunkSize()
	if chunkSize > 0 && len(tokens) > chunkSize {
		var logits, hidden *metal.Array
		for start := 0; start < len(tokens); start += chunkSize {
			end := min(start+chunkSize, len(tokens))
			nextLogits, nextHidden, err := prefillGemma4AssistantPromptOnce(ctx, pair, tokens[start:end], caches)
			if err != nil {
				metal.Free(logits, hidden)
				return nil, nil, core.E("Model.GenerateGemma4Assistant", core.Sprintf("prefill chunk %d:%d", start, end), err)
			}
			metal.Free(logits, hidden)
			logits = nextLogits
			hidden = nextHidden
		}
		return logits, hidden, nil
	}
	return prefillGemma4AssistantPromptOnce(ctx, pair, tokens, caches)
}

func prefillGemma4AssistantPromptOnce(ctx context.Context, pair *Gemma4AssistantPair, tokens []int32, caches []metal.Cache) (*metal.Array, *metal.Array, error) {
	select {
	case <-ctx.Done():
		return nil, nil, ctx.Err()
	default:
	}
	vInput := metal.FromValues(tokens, len(tokens))
	input := metal.Reshape2(vInput, 1, int32(len(tokens)))
	metal.Free(vInput)
	logits, hidden := pair.Target.ForwardLastTokenLogitsAndHidden(input, nil, caches)
	metal.Free(input)
	if logits == nil || hidden == nil || !logits.Valid() || !hidden.Valid() {
		metal.Free(logits, hidden)
		return nil, nil, core.NewError("Model.GenerateGemma4Assistant: target prefill returned invalid state")
	}
	if err := metal.Eval(logits, hidden); err != nil {
		metal.Free(logits, hidden)
		return nil, nil, core.E("Model.GenerateGemma4Assistant", "prefill", err)
	}
	metal.DetachCaches(caches)
	return logits, hidden, nil
}

func prepareGemma4AssistantPrompt(ctx context.Context, m *metal.Model, pair *Gemma4AssistantPair, tokens []int32, cfg metal.GenerateConfig) (metal.PromptPreparation, error) {
	start := time.Now()
	requestFixedSize := m.GenerationFixedGemma4CacheSize(len(tokens), cfg.MaxTokens)
	if entry, prefixLen := m.PromptCacheMatchWithHidden(tokens); entry != nil {
		restoreStart := time.Now()
		caches, logits, hidden, err := prefillGemma4AssistantFromPromptCache(ctx, pair, entry, tokens, prefixLen, requestFixedSize)
		restoreDuration := time.Since(restoreStart)
		return metal.PromptPreparation{
			Caches:          caches,
			Logits:          logits,
			Hidden:          hidden,
			Duration:        time.Since(start),
			CacheHit:        err == nil,
			CacheHitTokens:  prefixLen,
			CacheMissTokens: max(0, len(tokens)-prefixLen),
			RestoreDuration: restoreDuration,
		}, err
	}

	caches := m.NewCachesWithRequestFixedSize(requestFixedSize)
	logits, hidden, err := prefillGemma4AssistantPrompt(ctx, m, pair, tokens, caches)
	if err != nil {
		metal.FreeCaches(caches)
		return metal.PromptPreparation{}, err
	}
	if m.RuntimeCachesSnapshotSafe() {
		if err := storeGemma4AssistantPromptCache(m, tokens, caches, logits, hidden); err != nil {
			metal.Free(logits, hidden)
			metal.FreeCaches(caches)
			return metal.PromptPreparation{}, err
		}
	}
	return metal.PromptPreparation{
		Caches:          caches,
		Logits:          logits,
		Hidden:          hidden,
		Duration:        time.Since(start),
		CacheMissTokens: len(tokens),
	}, nil
}

func prefillGemma4AssistantFromPromptCache(ctx context.Context, pair *Gemma4AssistantPair, entry *metal.PromptCacheEntry, tokens []int32, prefixLen, requestFixedSize int) ([]metal.Cache, *metal.Array, *metal.Array, error) {
	caches, err := entry.RestoreCaches(prefixLen, requestFixedSize)
	if err != nil {
		return nil, nil, nil, err
	}
	if entryLogits, entryHidden := entry.Logits(), entry.Hidden(); prefixLen == len(tokens) && entryLogits != nil && entryLogits.Valid() && entryHidden != nil && entryHidden.Valid() {
		logits := metal.Copy(entryLogits)
		hidden := metal.Copy(entryHidden)
		if err := metal.Eval(logits, hidden); err != nil {
			metal.Free(logits, hidden)
			metal.FreeCaches(caches)
			return nil, nil, nil, core.E("Model.GenerateGemma4Assistant", "restore prompt state", err)
		}
		metal.Detach(logits, hidden)
		return caches, logits, hidden, nil
	}

	var logits, hidden *metal.Array
	for _, id := range tokens[prefixLen:] {
		select {
		case <-ctx.Done():
			metal.Free(logits, hidden)
			metal.FreeCaches(caches)
			return nil, nil, nil, ctx.Err()
		default:
		}

		nextLogits, nextHidden, err := pair.forwardGemma4AssistantAcceptedToken(id, caches)
		if err != nil {
			metal.Free(logits, hidden)
			metal.FreeCaches(caches)
			return nil, nil, nil, core.E("Model.GenerateGemma4Assistant", "prompt cache suffix", err)
		}
		metal.Free(logits, hidden)
		logits = nextLogits
		hidden = nextHidden
	}
	if logits == nil || hidden == nil {
		metal.FreeCaches(caches)
		return nil, nil, nil, core.NewError("Model.GenerateGemma4Assistant: prompt cache hit had no suffix state")
	}
	return caches, logits, hidden, nil
}

func storeGemma4AssistantPromptCache(m *metal.Model, tokens []int32, caches []metal.Cache, logits, hidden *metal.Array) error {
	if m == nil || !m.PromptCacheEnabled() || len(tokens) < m.PromptCacheMinimum() {
		return nil
	}
	entry, err := metal.NewPromptCacheEntryWithHidden(tokens, caches, logits, hidden)
	if err != nil {
		return err
	}
	if entry == nil {
		return nil
	}
	m.StorePromptCacheEntry(entry)
	return nil
}

func (pair *Gemma4AssistantPair) forwardGemma4AssistantAcceptedToken(token int32, caches []metal.Cache) (*metal.Array, *metal.Array, error) {
	input := metal.FromSingleInt32Matrix(token)
	logits, hidden := pair.Target.ForwardLastTokenLogitsAndHidden(input, nil, caches)
	metal.Free(input)
	if logits == nil || hidden == nil || !logits.Valid() || !hidden.Valid() {
		metal.Free(logits, hidden)
		return nil, nil, core.NewError("gemma4.assistant generation target forward returned invalid state")
	}
	if err := metal.Eval(logits, hidden); err != nil {
		metal.Free(logits, hidden)
		return nil, nil, core.E("gemma4.assistant generation", "target accepted token", err)
	}
	metal.DetachCaches(caches)
	return logits, hidden, nil
}

func appendGemma4AssistantToken(m *metal.Model, result *Gemma4AssistantGenerateResult, id int32, cfg metal.GenerateConfig) bool {
	tok := m.RuntimeTokenizer()
	if tok.HasEOSToken() && id == tok.EOSToken() {
		return true
	}
	if slices.Contains(cfg.StopTokens, id) {
		return true
	}
	text := tok.DecodeToken(id)
	result.Tokens = append(result.Tokens, metal.Token{ID: id, Text: text})
	result.Text += text
	return false
}

func recordGemma4AssistantFirstToken(result *Gemma4AssistantGenerateResult, start time.Time) {
	if result == nil || result.FirstTokenDuration > 0 || len(result.Tokens) == 0 {
		return
	}
	result.FirstTokenDuration = time.Since(start)
	if result.FirstTokenDuration <= 0 {
		result.FirstTokenDuration = time.Nanosecond
	}
}
