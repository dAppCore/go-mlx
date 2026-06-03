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
func (m *metal.Model) GenerateGemma4Assistant(ctx context.Context, pair *Gemma4AssistantPair, prompt string, cfg metal.GenerateConfig, draftTokens int) (Gemma4AssistantGenerateResult, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if cfg.MaxTokens <= 0 {
		cfg.MaxTokens = 256
	}
	draftTokens = gemma4AssistantResolveDraftTokens(draftTokens)
	if err := validateGemma4AssistantGenerateConfig(cfg); err != nil {
		return Gemma4AssistantGenerateResult{}, err
	}
	if err := m.requireTextRuntime("Model.GenerateGemma4Assistant"); err != nil {
		return Gemma4AssistantGenerateResult{}, err
	}
	if pair == nil || pair.Target == nil || pair.Assistant == nil {
		return Gemma4AssistantGenerateResult{}, core.NewError("gemma4.assistant generation requires an attached pair")
	}
	target, ok := m.model.(*Gemma4Model)
	if !ok || target != pair.Target {
		return Gemma4AssistantGenerateResult{}, core.NewError("gemma4.assistant generation pair does not match target runtime")
	}

	m.lastErr = nil
	m.lastMetrics = metal.Metrics{}
	release, err := m.acquireSlot(ctx)
	if err != nil {
		m.lastErr = err
		return Gemma4AssistantGenerateResult{}, err
	}
	defer release()
	releasePromptCache := m.acquirePromptCache()
	defer releasePromptCache()

	var result Gemma4AssistantGenerateResult
	if deviceErr := m.withDevice(func() {
		result, err = m.generateGemma4Assistant(ctx, pair, prompt, cfg, draftTokens)
	}); deviceErr != nil {
		err = deviceErr
	}
	if err != nil {
		m.lastErr = err
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

func (m *metal.Model) generateGemma4Assistant(ctx context.Context, pair *Gemma4AssistantPair, prompt string, cfg metal.GenerateConfig, draftTokens int) (Gemma4AssistantGenerateResult, error) {
	start := time.Now()
	metal.ResetPeakMemory()
	promptTokens := m.tokenizer.Encode(prompt)
	if len(promptTokens) == 0 {
		return Gemma4AssistantGenerateResult{}, core.NewError("Model.GenerateGemma4Assistant: empty prompt after tokenisation")
	}
	prepared, err := m.prepareGemma4AssistantPrompt(ctx, pair, promptTokens, cfg)
	if err != nil {
		return Gemma4AssistantGenerateResult{}, err
	}
	caches := prepared.caches
	logits := prepared.logits
	hidden := prepared.hidden
	defer func() { metal.FreeCaches(caches) }()
	defer metal.Free(logits, hidden)

	result := Gemma4AssistantGenerateResult{
		PromptTokens:    len(promptTokens),
		PrefillDuration: prepared.duration,
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
			stops := m.appendGemma4AssistantToken(&result, id, cfg)
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
			stops := m.appendGemma4AssistantToken(&result, replacement, cfg)
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
	m.lastMetrics = metal.Metrics{
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
		PromptCacheHitTokens:       prepared.cacheHitTokens,
		PromptCacheMissTokens:      prepared.cacheMissTokens,
		PromptCacheRestoreDuration: prepared.restoreDuration,
	}
	if prepared.cacheHit {
		m.lastMetrics.PromptCacheHits = 1
	} else {
		m.lastMetrics.PromptCacheMisses = 1
	}
	if result.PrefillDuration > 0 {
		m.lastMetrics.PrefillTokensPerSec = float64(len(promptTokens)) / result.PrefillDuration.Seconds()
	}
	if decodeDuration > 0 {
		m.lastMetrics.DecodeTokensPerSec = float64(len(result.Tokens)) / decodeDuration.Seconds()
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
		m.lastMetrics.MTP = &metal.MTPMetrics{
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
			WarmDecodeTokensPerSec: m.lastMetrics.DecodeTokensPerSec,
			WallDuration:           result.Duration,
			RestoreDuration:        prepared.restoreDuration,
			TargetVerifyDuration:   result.TargetVerifyDuration,
			TargetDuration:         result.TargetDuration,
			DraftDuration:          result.DraftDuration,
			PeakMemoryBytes:        m.lastMetrics.PeakMemoryBytes,
		}
	}
	return result, nil
}

func (m *metal.Model) prefillGemma4AssistantPrompt(ctx context.Context, pair *Gemma4AssistantPair, tokens []int32, caches []metal.Cache) (*metal.Array, *metal.Array, error) {
	if len(tokens) == 0 {
		return nil, nil, core.NewError("Model.GenerateGemma4Assistant: empty prompt after tokenisation")
	}
	chunkSize := m.prefillChunkSize
	if chunkSize > 0 && len(tokens) > chunkSize {
		var logits, hidden *metal.Array
		for start := 0; start < len(tokens); start += chunkSize {
			end := min(start+chunkSize, len(tokens))
			nextLogits, nextHidden, err := m.prefillGemma4AssistantPromptOnce(ctx, pair, tokens[start:end], caches)
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
	return m.prefillGemma4AssistantPromptOnce(ctx, pair, tokens, caches)
}

func (m *metal.Model) prefillGemma4AssistantPromptOnce(ctx context.Context, pair *Gemma4AssistantPair, tokens []int32, caches []metal.Cache) (*metal.Array, *metal.Array, error) {
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

func (m *metal.Model) prepareGemma4AssistantPrompt(ctx context.Context, pair *Gemma4AssistantPair, tokens []int32, cfg metal.GenerateConfig) (metal.PromptPreparation, error) {
	start := time.Now()
	requestFixedSize := m.generationFixedGemma4CacheSize(len(tokens), cfg.MaxTokens)
	if entry, prefixLen := m.promptCacheMatchWithHidden(tokens); entry != nil {
		restoreStart := time.Now()
		caches, logits, hidden, err := m.prefillGemma4AssistantFromPromptCache(ctx, pair, entry, tokens, prefixLen, requestFixedSize)
		restoreDuration := time.Since(restoreStart)
		return metal.PromptPreparation{
			caches:          caches,
			logits:          logits,
			hidden:          hidden,
			duration:        time.Since(start),
			cacheHit:        err == nil,
			cacheHitTokens:  prefixLen,
			cacheMissTokens: max(0, len(tokens)-prefixLen),
			restoreDuration: restoreDuration,
		}, err
	}

	caches := m.newCachesWithRequestFixedSize(requestFixedSize)
	logits, hidden, err := m.prefillGemma4AssistantPrompt(ctx, pair, tokens, caches)
	if err != nil {
		metal.FreeCaches(caches)
		return metal.PromptPreparation{}, err
	}
	if m.runtimeCachesSnapshotSafe() {
		if err := m.storeGemma4AssistantPromptCache(tokens, caches, logits, hidden); err != nil {
			metal.Free(logits, hidden)
			metal.FreeCaches(caches)
			return metal.PromptPreparation{}, err
		}
	}
	return metal.PromptPreparation{
		caches:          caches,
		logits:          logits,
		hidden:          hidden,
		duration:        time.Since(start),
		cacheMissTokens: len(tokens),
	}, nil
}

func (m *metal.Model) prefillGemma4AssistantFromPromptCache(ctx context.Context, pair *Gemma4AssistantPair, entry *metal.PromptCacheEntry, tokens []int32, prefixLen, requestFixedSize int) ([]metal.Cache, *metal.Array, *metal.Array, error) {
	caches, err := metal.RestorePromptCachesWithRequestFixedSize(entry.caches, prefixLen, requestFixedSize)
	if err != nil {
		return nil, nil, nil, err
	}
	if prefixLen == len(tokens) && entry.logits != nil && entry.logits.Valid() && entry.hidden != nil && entry.hidden.Valid() {
		logits := metal.Copy(entry.logits)
		hidden := metal.Copy(entry.hidden)
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

func (m *metal.Model) storeGemma4AssistantPromptCache(tokens []int32, caches []metal.Cache, logits, hidden *metal.Array) error {
	if m == nil || !m.promptCacheEnabled || len(tokens) < m.promptCacheMinimum() {
		return nil
	}
	entry, err := metal.NewPromptCacheEntryWithHidden(tokens, caches, logits, hidden)
	if err != nil {
		return err
	}
	if entry == nil {
		return nil
	}
	entry.adapterHash = m.adapterCacheKey()
	m.clearPromptCache()
	m.promptCache = entry
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

func (m *metal.Model) appendGemma4AssistantToken(result *Gemma4AssistantGenerateResult, id int32, cfg metal.GenerateConfig) bool {
	if m.tokenizer.HasEOSToken() && id == m.tokenizer.EOSToken() {
		return true
	}
	if slices.Contains(cfg.StopTokens, id) {
		return true
	}
	text := m.tokenizer.DecodeToken(id)
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
