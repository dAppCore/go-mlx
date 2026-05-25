// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"context"
	"slices"
	"time"

	core "dappco.re/go"
)

// Gemma4AssistantGenerateResult records one greedy MTP generation run.
type Gemma4AssistantGenerateResult struct {
	Tokens          []Token
	Text            string
	PromptTokens    int
	TargetTokens    int
	DraftTokens     int
	AcceptedTokens  int
	RejectedTokens  int
	TargetCalls     int
	DraftCalls      int
	Duration        time.Duration
	PrefillDuration time.Duration
	TargetDuration  time.Duration
	DraftDuration   time.Duration
}

// GenerateGemma4Assistant runs a conservative greedy MTP generation loop over
// an attached Gemma 4 assistant pair. Sampling-aware verification is kept out
// until the greedy accept/reject path is benchmarked.
func (m *Model) GenerateGemma4Assistant(ctx context.Context, pair *Gemma4AssistantPair, prompt string, cfg GenerateConfig, draftTokens int) (Gemma4AssistantGenerateResult, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if cfg.MaxTokens <= 0 {
		cfg.MaxTokens = 256
	}
	if draftTokens <= 0 {
		draftTokens = 1
	}
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
	m.lastMetrics = Metrics{}
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

func validateGemma4AssistantGenerateConfig(cfg GenerateConfig) error {
	if cfg.Temperature != 0 || cfg.TopK != 0 || cfg.TopP != 0 || cfg.MinP != 0 || cfg.RepeatPenalty > 1 {
		return core.NewError("gemma4.assistant generation currently supports greedy decoding only")
	}
	if cfg.ProbeSink != nil {
		return core.NewError("gemma4.assistant generation does not support probe sinks yet")
	}
	return nil
}

func (m *Model) generateGemma4Assistant(ctx context.Context, pair *Gemma4AssistantPair, prompt string, cfg GenerateConfig, draftTokens int) (Gemma4AssistantGenerateResult, error) {
	start := time.Now()
	ResetPeakMemory()
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
	defer func() { freeCaches(caches) }()
	defer Free(logits, hidden)

	result := Gemma4AssistantGenerateResult{
		PromptTokens:    len(promptTokens),
		PrefillDuration: prepared.duration,
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
		draft, err := pair.DraftBlock(lastToken, hidden, caches, blockSize)
		result.DraftDuration += time.Since(draftStart)
		result.DraftCalls++
		if err != nil {
			return result, err
		}
		result.DraftTokens += len(draft.Tokens)

		targetStart := time.Now()
		verify, err := pair.VerifyDraftBlock(logits, draft.Tokens, caches)
		result.TargetDuration += time.Since(targetStart)
		result.TargetCalls++
		draft.Close()
		if err != nil {
			return result, err
		}

		for _, id := range verify.AcceptedTokens {
			if m.appendGemma4AssistantToken(&result, id, cfg) {
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

		freeCaches(caches)
		caches = nextCaches
		Free(logits, hidden)
		logits = nextLogits
		hidden = nextHidden

		if !verify.AllAccepted {
			replacement := verify.ReplacementToken
			if m.appendGemma4AssistantToken(&result, replacement, cfg) {
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
			Free(logits, hidden)
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
	processMemory := GetProcessMemory()
	m.lastMetrics = Metrics{
		PromptTokens:               result.PromptTokens,
		GeneratedTokens:            len(result.Tokens),
		PrefillDuration:            result.PrefillDuration,
		DecodeDuration:             decodeDuration,
		TotalDuration:              result.Duration,
		PeakMemoryBytes:            GetPeakMemory(),
		ActiveMemoryBytes:          GetActiveMemory(),
		CacheMemoryBytes:           GetCacheMemory(),
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
	return result, nil
}

func (m *Model) prefillGemma4AssistantPrompt(ctx context.Context, pair *Gemma4AssistantPair, tokens []int32, caches []Cache) (*Array, *Array, error) {
	if len(tokens) == 0 {
		return nil, nil, core.NewError("Model.GenerateGemma4Assistant: empty prompt after tokenisation")
	}
	chunkSize := m.prefillChunkSize
	if chunkSize > 0 && len(tokens) > chunkSize {
		var logits, hidden *Array
		for start := 0; start < len(tokens); start += chunkSize {
			end := start + chunkSize
			if end > len(tokens) {
				end = len(tokens)
			}
			nextLogits, nextHidden, err := m.prefillGemma4AssistantPromptOnce(ctx, pair, tokens[start:end], caches)
			if err != nil {
				Free(logits, hidden)
				return nil, nil, core.E("Model.GenerateGemma4Assistant", core.Sprintf("prefill chunk %d:%d", start, end), err)
			}
			Free(logits, hidden)
			logits = nextLogits
			hidden = nextHidden
		}
		return logits, hidden, nil
	}
	return m.prefillGemma4AssistantPromptOnce(ctx, pair, tokens, caches)
}

func (m *Model) prefillGemma4AssistantPromptOnce(ctx context.Context, pair *Gemma4AssistantPair, tokens []int32, caches []Cache) (*Array, *Array, error) {
	select {
	case <-ctx.Done():
		return nil, nil, ctx.Err()
	default:
	}
	vInput := FromValues(tokens, len(tokens))
	input := Reshape2(vInput, 1, int32(len(tokens)))
	Free(vInput)
	logits, hidden := pair.Target.ForwardLastTokenLogitsAndHidden(input, nil, caches)
	Free(input)
	if logits == nil || hidden == nil || !logits.Valid() || !hidden.Valid() {
		Free(logits, hidden)
		return nil, nil, core.NewError("Model.GenerateGemma4Assistant: target prefill returned invalid state")
	}
	if err := Eval(logits, hidden); err != nil {
		Free(logits, hidden)
		return nil, nil, core.E("Model.GenerateGemma4Assistant", "prefill", err)
	}
	detachCaches(caches)
	return logits, hidden, nil
}

func (m *Model) prepareGemma4AssistantPrompt(ctx context.Context, pair *Gemma4AssistantPair, tokens []int32, cfg GenerateConfig) (promptPreparation, error) {
	start := time.Now()
	requestFixedSize := m.generationFixedGemma4CacheSize(len(tokens), cfg.MaxTokens)
	if entry, prefixLen := m.promptCacheMatchWithHidden(tokens); entry != nil {
		restoreStart := time.Now()
		caches, logits, hidden, err := m.prefillGemma4AssistantFromPromptCache(ctx, pair, entry, tokens, prefixLen, requestFixedSize)
		restoreDuration := time.Since(restoreStart)
		return promptPreparation{
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
		freeCaches(caches)
		return promptPreparation{}, err
	}
	if m.runtimeCachesSnapshotSafe() {
		if err := m.storeGemma4AssistantPromptCache(tokens, caches, logits, hidden); err != nil {
			Free(logits, hidden)
			freeCaches(caches)
			return promptPreparation{}, err
		}
	}
	return promptPreparation{
		caches:          caches,
		logits:          logits,
		hidden:          hidden,
		duration:        time.Since(start),
		cacheMissTokens: len(tokens),
	}, nil
}

func (m *Model) prefillGemma4AssistantFromPromptCache(ctx context.Context, pair *Gemma4AssistantPair, entry *promptCacheEntry, tokens []int32, prefixLen, requestFixedSize int) ([]Cache, *Array, *Array, error) {
	caches, err := restorePromptCachesWithRequestFixedSize(entry.caches, prefixLen, requestFixedSize)
	if err != nil {
		return nil, nil, nil, err
	}
	if prefixLen == len(tokens) && entry.logits != nil && entry.logits.Valid() && entry.hidden != nil && entry.hidden.Valid() {
		logits := Copy(entry.logits)
		hidden := Copy(entry.hidden)
		if err := Eval(logits, hidden); err != nil {
			Free(logits, hidden)
			freeCaches(caches)
			return nil, nil, nil, core.E("Model.GenerateGemma4Assistant", "restore prompt state", err)
		}
		Detach(logits, hidden)
		return caches, logits, hidden, nil
	}

	var logits, hidden *Array
	for _, id := range tokens[prefixLen:] {
		select {
		case <-ctx.Done():
			Free(logits, hidden)
			freeCaches(caches)
			return nil, nil, nil, ctx.Err()
		default:
		}

		nextLogits, nextHidden, err := pair.forwardGemma4AssistantAcceptedToken(id, caches)
		if err != nil {
			Free(logits, hidden)
			freeCaches(caches)
			return nil, nil, nil, core.E("Model.GenerateGemma4Assistant", "prompt cache suffix", err)
		}
		Free(logits, hidden)
		logits = nextLogits
		hidden = nextHidden
	}
	if logits == nil || hidden == nil {
		freeCaches(caches)
		return nil, nil, nil, core.NewError("Model.GenerateGemma4Assistant: prompt cache hit had no suffix state")
	}
	return caches, logits, hidden, nil
}

func (m *Model) storeGemma4AssistantPromptCache(tokens []int32, caches []Cache, logits, hidden *Array) error {
	if m == nil || !m.promptCacheEnabled || len(tokens) < m.promptCacheMinimum() {
		return nil
	}
	entry, err := newPromptCacheEntryWithHidden(tokens, caches, logits, hidden)
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

func (pair *Gemma4AssistantPair) forwardGemma4AssistantAcceptedToken(token int32, caches []Cache) (*Array, *Array, error) {
	vInput := fromSingleInt32(token)
	input := Reshape2(vInput, 1, 1)
	Free(vInput)
	logits, hidden := pair.Target.ForwardLastTokenLogitsAndHidden(input, nil, caches)
	Free(input)
	if logits == nil || hidden == nil || !logits.Valid() || !hidden.Valid() {
		Free(logits, hidden)
		return nil, nil, core.NewError("gemma4.assistant generation target forward returned invalid state")
	}
	if err := Eval(logits, hidden); err != nil {
		Free(logits, hidden)
		return nil, nil, core.E("gemma4.assistant generation", "target accepted token", err)
	}
	detachCaches(caches)
	return logits, hidden, nil
}

func (m *Model) appendGemma4AssistantToken(result *Gemma4AssistantGenerateResult, id int32, cfg GenerateConfig) bool {
	text := m.tokenizer.DecodeToken(id)
	result.Tokens = append(result.Tokens, Token{ID: id, Text: text})
	result.Text += text
	if m.tokenizer.HasEOSToken() && id == m.tokenizer.EOSToken() {
		return true
	}
	return slices.Contains(cfg.StopTokens, id)
}
