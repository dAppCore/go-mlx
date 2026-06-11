// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package gemma4

import (
	"context"
	"math/rand"
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
	return pair.GenerateWithSink(ctx, m, prompt, cfg, draftTokens, nil)
}

// Gemma4AssistantTokenSink receives each verified token as the MTP loop emits
// it — the serve streaming hook. Returning false stops generation (the client
// went away); the loop returns what it has, no error.
type Gemma4AssistantTokenSink func(metal.Token) bool

// GenerateWithSink is Generate with per-token streaming: every verified token
// is handed to sink as soon as its verify round lands, instead of only
// arriving in the collected result. A nil sink is exactly Generate.
func (pair *Gemma4AssistantPair) GenerateWithSink(ctx context.Context, m *metal.Model, prompt string, cfg metal.GenerateConfig, draftTokens int, sink Gemma4AssistantTokenSink) (Gemma4AssistantGenerateResult, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if cfg.MaxTokens <= 0 {
		// No caller cap → the model's context length; generation stops on EOS.
		// The model declares no text output-length knob.
		cfg.MaxTokens = m.Info().ContextLength
	}
	draftTokens = gemma4AssistantResolveDraftTokens(draftTokens)
	if err := validateGemma4AssistantGenerateConfig(cfg); err != nil {
		return Gemma4AssistantGenerateResult{}, err
	}
	// Speculative sampling needs the drafter's distribution q. Both drafter
	// kinds provide it now (dense natively; ordered-embedding via the
	// sparse->dense scatter), so temperature>0 only needs an assistant present.
	if cfg.Temperature > 0 && pair.Assistant == nil {
		return Gemma4AssistantGenerateResult{}, core.NewError("gemma4.assistant: temperature>0 requires an assistant drafter")
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
		result, err = generateGemma4Assistant(ctx, m, pair, prompt, cfg, draftTokens, sink)
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
	// temperature / top-k / top-p / min-p are all supported now: greedy
	// (temp==0) on the argmax fast path, temperature>0 via speculative SAMPLING.
	// Repetition penalty and probe sinks are not yet modelled by the sampled
	// accept maths, so they still fall back to plain decode.
	if cfg.RepeatPenalty > 1 {
		return core.NewError("gemma4.assistant generation does not support repetition penalty")
	}
	if cfg.ProbeSink != nil {
		return core.NewError("gemma4.assistant generation does not support probe sinks yet")
	}
	return nil
}

func generateGemma4Assistant(ctx context.Context, m *metal.Model, pair *Gemma4AssistantPair, prompt string, cfg metal.GenerateConfig, draftTokens int, sink Gemma4AssistantTokenSink) (Gemma4AssistantGenerateResult, error) {
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
	// carryLead is the previous round's bonus token: emitted, but NOT yet in the
	// cache. It is prepended to the next draft block so the target re-sees it for
	// free inside the one batched verify forward — eliminating the per-reject
	// replacement forward that made speculative decode no faster than plain.
	carryLead := int32(-1)
	stopped := false
	// Speculative SAMPLING path (temperature>0 with a logits-exposing drafter):
	// same block-draft + carryLead structure as greedy below, but accepts each
	// token with prob min(1, p/q) and samples rejects/bonus from the target
	// distribution p, so output is distributed exactly as plain temperature-1
	// sampling — just faster. Greedy (temp==0) and ordered-embedding drafters
	// (which expose no drafter distribution q) fall through to the greedy loop,
	// gated by !sampling.
	sampling := cfg.Temperature > 0 && pair.Assistant != nil
	if sampling {
		var rng *rand.Rand
		if cfg.SeedSet {
			rng = rand.New(rand.NewSource(int64(cfg.Seed)))
		} else {
			rng = rand.New(rand.NewSource(time.Now().UnixNano()))
		}
		uniform := func() float32 { return rng.Float32() }
		for len(result.Tokens) < cfg.MaxTokens && !stopped {
			select {
			case <-ctx.Done():
				return result, ctx.Err()
			default:
			}
			remaining := cfg.MaxTokens - len(result.Tokens)
			blockSize := min(draftTokens, remaining)
			draftStart := time.Now()
			draft, err := pair.draftBlockSampled(lastToken, hidden, caches, blockSize, cfg.SuppressTokens, cfg, uniform)
			result.DraftDuration += time.Since(draftStart)
			result.DraftCalls++
			if err != nil {
				return result, err
			}
			result.DraftTokens += len(draft.Tokens)
			result.DraftTokenSchedule = append(result.DraftTokenSchedule, blockSize)

			block := draft.Tokens
			blockDraftLogits := draft.Logits
			if carryLead >= 0 {
				block = append([]int32{carryLead}, draft.Tokens...)
				blockDraftLogits = append([]*metal.Array{nil}, draft.Logits...)
			}

			targetStart := time.Now()
			verify, err := pair.verifyDraftBlockSampled(logits, block, blockDraftLogits, caches, cfg, uniform, cfg.SuppressTokens)
			verifyDur := time.Since(targetStart)
			result.TargetVerifyDuration += verifyDur
			result.TargetDuration += verifyDur
			result.TargetVerifyCalls++
			result.TargetCalls++
			metal.Free(draft.Logits...) // retained drafter logits, done after verify
			draft.Close()               // frees draft.Hidden
			if err != nil {
				return result, err
			}

			// carryLead (block[0]) was emitted last round — skip re-emitting it.
			emitStart := 0
			if carryLead >= 0 && len(verify.AcceptedTokens) > 0 && verify.AcceptedTokens[0] == carryLead {
				emitStart = 1
			}
			newAccepted := 0
			for _, id := range verify.AcceptedTokens[emitStart:] {
				stops := appendGemma4AssistantToken(m, &result, id, cfg, sink)
				recordGemma4AssistantFirstToken(&result, start)
				if stops {
					stopped = true
					break
				}
				lastToken = id
				newAccepted++
			}
			result.AcceptedTokens += newAccepted
			result.TargetTokens += newAccepted
			result.RejectedTokens += verify.RejectedCount

			metal.FreeCaches(caches)
			caches = verify.Caches
			verify.Caches = nil
			if verify.Hidden != nil {
				metal.Free(hidden)
				hidden = verify.Hidden
				verify.Hidden = nil
			}
			metal.Free(logits)
			logits = verify.Logits
			verify.Logits = nil

			if stopped {
				verify.Close()
				break
			}

			// Emit the next token (residual replacement, or the sampled bonus on a
			// full accept) and carry it as the next round's lead.
			next := verify.ReplacementToken
			stops := appendGemma4AssistantToken(m, &result, next, cfg, sink)
			recordGemma4AssistantFirstToken(&result, start)
			result.TargetTokens++
			lastToken = next
			carryLead = next
			if stops {
				stopped = true
			}
			verify.Close()
		}
	}
	for !sampling && len(result.Tokens) < cfg.MaxTokens && !stopped {
		select {
		case <-ctx.Done():
			return result, ctx.Err()
		default:
		}

		remaining := cfg.MaxTokens - len(result.Tokens)
		blockSize := min(draftTokens, remaining)
		if core.Getenv("GO_MLX_MTP_DIAG") != "" && result.DraftCalls < 6 {
			gemma4LogMTPStepDiag(pair, lastToken, hidden, caches, logits)
		}
		draftStart := time.Now()
		draft, err := pair.DraftBlockWithSuppression(lastToken, hidden, caches, blockSize, cfg.SuppressTokens)
		result.DraftDuration += time.Since(draftStart)
		result.DraftCalls++
		if err != nil {
			return result, err
		}
		result.DraftTokens += len(draft.Tokens)
		result.DraftTokenSchedule = append(result.DraftTokenSchedule, blockSize)

		block := draft.Tokens
		if carryLead >= 0 {
			block = append([]int32{carryLead}, draft.Tokens...)
		}

		targetStart := time.Now()
		verify, err := pair.VerifyDraftBlockWithSuppression(logits, block, caches, cfg.SuppressTokens)
		verifyDuration := time.Since(targetStart)
		result.TargetVerifyDuration += verifyDuration
		result.TargetDuration += verifyDuration
		result.TargetVerifyCalls++
		result.TargetCalls++
		draft.Close()
		if err != nil {
			return result, err
		}

		// carryLead (block[0]) is always re-accepted (it is argmax of the carried
		// logits); skip it when emitting since it was emitted last round.
		emitStart := 0
		if carryLead >= 0 && len(verify.AcceptedTokens) > 0 && verify.AcceptedTokens[0] == carryLead {
			emitStart = 1
		}
		newDrafts := 0
		for _, id := range verify.AcceptedTokens[emitStart:] {
			stops := appendGemma4AssistantToken(m, &result, id, cfg, sink)
			recordGemma4AssistantFirstToken(&result, start)
			if stops {
				stopped = true
				break
			}
			lastToken = id
			newDrafts++
		}
		result.AcceptedTokens += newDrafts
		result.RejectedTokens += verify.RejectedCount
		result.TargetTokens += newDrafts

		metal.FreeCaches(caches)
		caches = verify.Caches
		verify.Caches = nil
		if verify.Hidden != nil {
			metal.Free(hidden)
			hidden = verify.Hidden
			verify.Hidden = nil
		}
		metal.Free(logits)
		logits = verify.Logits
		verify.Logits = nil

		if stopped {
			verify.Close()
			break
		}

		if verify.AllAccepted {
			// Whole block accepted — the next token is judged by the carried
			// logits; nothing is outstanding to prepend next round.
			carryLead = -1
		} else {
			// Emit the bonus and carry it as the next round's lead (NOT forwarded
			// here — it rides the next batched verify).
			replacement := verify.ReplacementToken
			stops := appendGemma4AssistantToken(m, &result, replacement, cfg, sink)
			recordGemma4AssistantFirstToken(&result, start)
			result.TargetTokens++
			lastToken = replacement
			carryLead = replacement
			if stops {
				stopped = true
				verify.Close()
				break
			}
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
	requestFixedSize := m.GenerationFixedSlidingCacheSize(len(tokens), cfg.MaxTokens)
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

func appendGemma4AssistantToken(m *metal.Model, result *Gemma4AssistantGenerateResult, id int32, cfg metal.GenerateConfig, sink Gemma4AssistantTokenSink) bool {
	tok := m.RuntimeTokenizer()
	if tok.HasEOSToken() && id == tok.EOSToken() {
		return true
	}
	if slices.Contains(cfg.StopTokens, id) {
		return true
	}
	text := tok.DecodeToken(id)
	token := metal.Token{ID: id, Text: text}
	result.Tokens = append(result.Tokens, token)
	result.Text += text
	if sink != nil && !sink(token) {
		return true
	}
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
