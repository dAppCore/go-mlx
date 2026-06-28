// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package mlx

import (
	"context"
	"iter"
	"math"
	"sync"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/mlx/pkg/metal"
	"dappco.re/go/mlx/pkg/model"
	// Register the native model loaders the reactive LoadTokenModelDir dispatches to — the deleted
	// per-arch loaders used to pull these in transitively; the serve layer now imports them explicitly
	// (pkg/native itself stays arch-free).
	_ "dappco.re/go/mlx/pkg/model/gemma4"
	_ "dappco.re/go/mlx/pkg/model/mistral"
	"dappco.re/go/mlx/pkg/native"
	"dappco.re/go/mlx/pkg/tokenizer"
)

// nativeTextModel exposes the no-cgo native token-loop contract (a model.TokenModel
// + tokenizer) as an inference.TextModel — the sibling of metaladapter (which wraps
// the cgo metal.Model). The OpenAI/Anthropic/Ollama serve handlers drive it with
// ZERO cgo: model.Generate over the contract under the hood (incrementally, since
// NativeTokenModel is a SessionModel — each call opens a persistent-cache session
// and Close frees it). The straight contract path: no prompt cache / MTP / batching
// (those are pkg/metal engine features), so it is the simplest correct serve, the
// proof that the no-cgo stack serves real tokens through the unified contract.
//
// Greedy requests stream from the native session as each token is decoded; sampling
// uses the native sampled session path when available, falling back to the shared
// model sampler loop. Close is a no-op: the resident weights live for the process
// (a single served model), matching the load-once serve shape.
type nativeTextModel struct {
	tm        model.TokenModel
	tok       *tokenizer.Tokenizer
	modelType string
	info      inference.ModelInfo
	maxLen    int

	mu          sync.Mutex
	lastErr     error
	lastMetrics inference.GenerateMetrics
	cacheSess   nativeTextPromptCacheSession
}

var _ inference.TextModel = (*nativeTextModel)(nil)

type nativeTextPromptCacheSession interface {
	model.DecodeStepper
	WarmPromptCache([]int32) error
	GenerateCached([]int32, int, int) ([]int32, error)
	ClearPromptCache()
}

type nativeTextPromptCacheSizer interface {
	CachedPrefixLen() int
}

type nativeTextGreedyStreamSession interface {
	GenerateEach([]int32, int, int, func(int32) bool) ([]int32, error)
}

type nativeTextGreedyTransformStreamSession interface {
	GenerateEachTransformed([]int32, int, int, native.TokenTransform, func(int32) bool) ([]int32, error)
}

type nativeTextGreedySuppressStreamSession interface {
	GenerateEachWithSuppression([]int32, int, int, []int32, func(int32) bool) ([]int32, error)
}

type nativeTextGreedySuppressTransformStreamSession interface {
	GenerateEachWithSuppressionAndTransform([]int32, int, int, []int32, native.TokenTransform, func(int32) bool) ([]int32, error)
}

type nativeTextSampledStreamSession interface {
	GenerateSampledEach([]int32, int, []int32, *model.Sampler, model.SampleParams, model.TokenTransform, func(int32) bool) ([]int32, error)
}

type nativeTextSampledOneShotStreamSession interface {
	GenerateSampledOneShotEach([]int32, int, []int32, *model.Sampler, model.SampleParams, model.TokenTransform, func(int32) bool) ([]int32, error)
}

type nativeTextPromptCacheGreedyStreamSession interface {
	GenerateCachedEach([]int32, int, int, func(int32) bool) ([]int32, error)
}

type nativeTextPromptCacheGreedyTransformStreamSession interface {
	GenerateCachedEachTransformed([]int32, int, int, native.TokenTransform, func(int32) bool) ([]int32, error)
}

type nativeTextPromptCacheGreedySuppressStreamSession interface {
	GenerateCachedEachWithSuppression([]int32, int, int, []int32, func(int32) bool) ([]int32, error)
}

type nativeTextPromptCacheGreedySuppressTransformStreamSession interface {
	GenerateCachedEachWithSuppressionAndTransform([]int32, int, int, []int32, native.TokenTransform, func(int32) bool) ([]int32, error)
}

type nativeTextPromptCacheSampledStreamSession interface {
	GenerateCachedSampledEach([]int32, int, []int32, *model.Sampler, model.SampleParams, model.TokenTransform, func(int32) bool) ([]int32, error)
}

type nativeTextRootSession interface {
	PrefillTokens([]int32) error
	AppendTokens([]int32) error
	BoundaryLogits() ([]byte, error)
	GenerateFromCacheEach(int, int, func(int32) bool) ([]int32, error)
	GenerateSampledFromCacheEach(int, []int32, *model.Sampler, model.SampleParams, model.TokenTransform, func(int32) bool) ([]int32, error)
	StateBlockSource(int) (native.SessionStateBlockSource, error)
	StateBlockSourceFrom(int, int) (native.SessionStateBlockSource, error)
	RestoreStateBlocks(native.SessionStateBlockSource) error
	Pos() int
	Close() error
}

type nativeTextSession struct {
	mu        sync.Mutex
	model     *nativeTextModel
	sess      nativeTextRootSession
	tokens    []int32
	generated []int32
	logits    []byte
	err       error
	closed    bool
}

var _ metal.SessionHandle = (*nativeTextSession)(nil)

// LoadNativeTextModel loads a gemma4 checkpoint directory as an inference.TextModel
// served entirely without cgo: the no-cgo native contract stack
// (native.LoadTokenModelDir — the reactive registry: dense / MoE / E2B-E4B PLE, 4-bit or bf16) plus
// the tokenizer, behind the standard serve handlers. WithContextLength sizes the KV
// cache (default 4096). The metallib loads at runtime (MLX_METALLIB_PATH or the
// embedded metallib), so the standard lthn-mlx binary serves it — no cgo, no Python.
func LoadNativeTextModel(modelPath string, opts ...LoadOption) (inference.TextModel, error) {
	loadCfg := applyLoadOptions(opts)
	maxLen := loadCfg.ContextLength
	if maxLen <= 0 {
		maxLen = 4096
	}
	tm, err := native.LoadTokenModelDir(modelPath, maxLen)
	if err != nil {
		return nil, err
	}
	tok, err := tokenizer.LoadTokenizer(core.PathJoin(modelPath, "tokenizer.json"))
	if err != nil {
		return nil, core.E("mlx.LoadNativeTextModel", "load tokenizer", err)
	}
	return &nativeTextModel{
		tm: tm, tok: tok, maxLen: maxLen, modelType: "gemma4",
		info: inference.ModelInfo{Architecture: "gemma4", VocabSize: tm.Vocab()},
	}, nil
}

// Generate streams tokens for a raw prompt (no chat template — Chat applies that).
func (m *nativeTextModel) Generate(ctx context.Context, prompt string, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	return m.stream(ctx, m.tok.Encode(prompt), inference.ApplyGenerateOpts(opts))
}

// Chat streams tokens from a multi-turn conversation rendered with the gemma turn
// template (user/model turns, a trailing model turn to complete).
func (m *nativeTextModel) Chat(ctx context.Context, messages []inference.Message, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	return m.stream(ctx, m.tok.Encode(formatGemmaChat(messages)), inference.ApplyGenerateOpts(opts))
}

// GenerateChunks streams tokens for a prompt supplied as bounded text chunks.
// Each chunk is tokenized independently and appended into one logical prompt
// stream, matching the metal chunk path without first joining the full prompt.
func (m *nativeTextModel) GenerateChunks(ctx context.Context, chunks iter.Seq[string], opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	cfg := inference.ApplyGenerateOpts(opts)
	return func(yield func(inference.Token) bool) {
		if ctx == nil {
			ctx = context.Background()
		}
		ids, err := m.encodePromptChunks(ctx, chunks, "mlx.nativeTextModel.GenerateChunks")
		if err != nil {
			m.setErr(err)
			return
		}
		for tok := range m.stream(ctx, ids, cfg) {
			if !yield(tok) {
				return
			}
		}
	}
}

// ChatChunks formats a chat prompt and feeds it through the chunked native
// generation path, preserving the same concatenated prompt as Chat.
func (m *nativeTextModel) ChatChunks(ctx context.Context, messages []inference.Message, chunkBytes int, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	return m.GenerateChunks(ctx, nativeGemmaChatChunks(messages, chunkBytes), opts...)
}

// formatGemmaChat renders messages in the gemma turn format. gemma has no system
// role, so system/user fold to "user" and assistant to "model"; a trailing model
// turn opens the completion.
func formatGemmaChat(messages []inference.Message) string {
	out := "<bos>"
	for _, msg := range messages {
		role := "user"
		if msg.Role == "assistant" {
			role = "model"
		}
		out += core.Sprintf("<start_of_turn>%s\n%s<end_of_turn>\n", role, msg.Content)
	}
	return out + "<start_of_turn>model\n"
}

func nativeGemmaChatChunks(messages []inference.Message, chunkBytes int) iter.Seq[string] {
	prompt := formatGemmaChat(messages)
	return func(yield func(string) bool) {
		if chunkBytes <= 0 {
			yield(prompt)
			return
		}
		for i := 0; i < len(prompt); i += chunkBytes {
			end := i + chunkBytes
			if end > len(prompt) {
				end = len(prompt)
			}
			if !yield(prompt[i:end]) {
				return
			}
		}
	}
}

func (m *nativeTextModel) stream(ctx context.Context, ids []int32, cfg inference.GenerateConfig) iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {
		start := time.Now()
		maxNew := cfg.MaxTokens
		if maxNew <= 0 || len(ids)+maxNew > m.maxLen {
			maxNew = m.maxLen - len(ids)
		}
		if maxNew <= 0 {
			m.setErr(core.NewError("mlx.nativeTextModel: prompt fills the context window, no room to generate"))
			return
		}
		stopTokens := m.stopTokens(cfg)
		eos := singleStopToken(stopTokens)
		suppressTokens := cfg.SuppressTokens
		needsSuppress := len(suppressTokens) > 0
		budgetTracker := newNativeThinkingBudgetTracker(m.tok, cfg)
		var nativeTransform native.TokenTransform
		var modelTransform model.TokenTransform
		if budgetTracker != nil {
			nativeTransform = native.TokenTransform(budgetTracker.observe)
			modelTransform = model.TokenTransform(budgetTracker.observe)
		}
		budgetForced := func() bool {
			return budgetTracker != nil && budgetTracker.forcedClose()
		}
		emit := func(id int32) bool {
			if ctx.Err() != nil {
				return false
			}
			if !yield(inference.Token{ID: id, Text: m.tok.DecodeToken(id)}) {
				return false
			}
			return !tokenInSet(id, stopTokens)
		}
		var (
			out      []int32
			err      error
			streamed bool
		)
		repeatPenaltyActive := cfg.RepeatPenalty > 1
		minStopActive := cfg.MinTokensBeforeStop > 0
		if cfg.Temperature <= 0 && !repeatPenaltyActive && !minStopActive {
			m.mu.Lock()
			cacheSess := m.cacheSess
			if cacheSess != nil {
				if needsSuppress || nativeTransform != nil {
					if streamSess, ok := cacheSess.(nativeTextPromptCacheGreedySuppressTransformStreamSession); ok {
						streamed = true
						out, err = streamSess.GenerateCachedEachWithSuppressionAndTransform(ids, maxNew, eos, suppressTokens, nativeTransform, emit)
					} else if needsSuppress && nativeTransform == nil {
						if streamSess, ok := cacheSess.(nativeTextPromptCacheGreedySuppressStreamSession); ok {
							streamed = true
							out, err = streamSess.GenerateCachedEachWithSuppression(ids, maxNew, eos, suppressTokens, emit)
						} else {
							cacheSess = nil
						}
					} else if streamSess, ok := cacheSess.(nativeTextPromptCacheGreedyTransformStreamSession); ok {
						streamed = true
						out, err = streamSess.GenerateCachedEachTransformed(ids, maxNew, eos, nativeTransform, emit)
					} else {
						cacheSess = nil
					}
				} else {
					if streamSess, ok := cacheSess.(nativeTextPromptCacheGreedyStreamSession); ok {
						streamed = true
						out, err = streamSess.GenerateCachedEach(ids, maxNew, eos, emit)
					} else {
						out, err = cacheSess.GenerateCached(ids, maxNew, eos)
					}
				}
			}
			m.mu.Unlock()
			if cacheSess != nil {
				if err != nil {
					m.setErr(err)
					return
				}
				if !streamed {
					emitted := 0
					for _, id := range out {
						emitted++
						if !emit(id) {
							m.setMetricsWithThinkingBudget(len(ids), emitted, time.Since(start), budgetForced())
							return
						}
					}
				}
				m.setMetricsWithThinkingBudget(len(ids), len(out), time.Since(start), budgetForced())
				return
			}
		}
		if cfg.Temperature > 0 || repeatPenaltyActive || minStopActive { // stochastic, or greedy with logits-side policy
			sampler := model.NewSampler(nativeSamplerSeed(cfg))
			sampleParams := model.SampleParams{Temperature: cfg.Temperature, TopK: cfg.TopK, TopP: cfg.TopP, MinP: cfg.MinP, SuppressTokens: suppressTokens, MinTokensBeforeStop: cfg.MinTokensBeforeStop, RepeatPenalty: cfg.RepeatPenalty}
			m.mu.Lock()
			cacheSess := m.cacheSess
			if cacheSess != nil {
				if streamSess, ok := cacheSess.(nativeTextPromptCacheSampledStreamSession); ok {
					streamed = true
					out, err = streamSess.GenerateCachedSampledEach(ids, maxNew, stopTokens, sampler, sampleParams, modelTransform, emit)
				} else {
					cacheSess = nil
				}
			}
			m.mu.Unlock()
			if cacheSess != nil {
				if err != nil {
					m.setErr(err)
					return
				}
				m.setMetricsWithThinkingBudget(len(ids), len(out), time.Since(start), budgetForced())
				return
			}
			if out, err, streamed = m.streamSampledSession(ids, maxNew, stopTokens, sampler, sampleParams, modelTransform, emit); streamed {
				if err != nil {
					m.setErr(err)
					return
				}
				m.setMetricsWithThinkingBudget(len(ids), len(out), time.Since(start), budgetForced())
				return
			}
			streamed = true // the sampled path streams per token (parity with greedy + pkg/metal)
			out, err = model.GenerateSampledWithStopTokensTransformEach(m.tm, sampler, sampleParams, ids, maxNew, stopTokens, modelTransform, emit)
		} else if out, err, streamed = m.streamGreedySession(ids, maxNew, eos, suppressTokens, nativeTransform, emit); streamed {
			if err != nil {
				m.setErr(err)
				return
			}
			m.setMetricsWithThinkingBudget(len(ids), len(out), time.Since(start), budgetForced())
			return
		} else if needsSuppress || modelTransform != nil {
			sampler := model.NewSampler(nativeSamplerSeed(cfg))
			streamed = true // greedy-with-suppression/transform also streams per token
			out, err = model.GenerateSampledWithStopTokensTransformEach(m.tm, sampler, model.SampleParams{MinP: cfg.MinP, SuppressTokens: suppressTokens, MinTokensBeforeStop: cfg.MinTokensBeforeStop}, ids, maxNew, stopTokens, modelTransform, emit)
		} else {
			out, err = model.Generate(m.tm, ids, maxNew, eos)
		}
		if err != nil {
			m.setErr(err)
			return
		}
		if !streamed { // batch paths emit here; the streaming paths already emitted via emit
			emitted := 0
			for _, id := range out {
				emitted++
				if !emit(id) {
					m.setMetricsWithThinkingBudget(len(ids), emitted, time.Since(start), budgetForced())
					return
				}
			}
		}
		m.setMetricsWithThinkingBudget(len(ids), len(out), time.Since(start), budgetForced())
	}
}

func (m *nativeTextModel) stopTokens(cfg inference.GenerateConfig) []int32 {
	if len(cfg.StopTokens) > 0 {
		return cfg.StopTokens
	}
	if m != nil && m.tok != nil && m.tok.HasEOSToken() {
		return []int32{m.tok.EOSToken()}
	}
	return nil
}

func singleStopToken(tokens []int32) int {
	if len(tokens) == 1 {
		return int(tokens[0])
	}
	return -1
}

func tokenInSet(id int32, tokens []int32) bool {
	for _, token := range tokens {
		if id == token {
			return true
		}
	}
	return false
}

func nativeSamplerSeed(cfg inference.GenerateConfig) uint64 {
	if cfg.SeedSet {
		return cfg.Seed
	}
	return uint64(time.Now().UnixNano())
}

func nativeMetalSamplerSeed(cfg metal.GenerateConfig) uint64 {
	if cfg.SeedSet {
		return cfg.Seed
	}
	return uint64(time.Now().UnixNano())
}

func (m *nativeTextModel) streamGreedySession(ids []int32, maxNew, eos int, suppress []int32, transform native.TokenTransform, yield func(int32) bool) ([]int32, error, bool) {
	sm, ok := m.tm.(model.SessionModel)
	if !ok {
		return nil, nil, false
	}
	sess, err := sm.OpenSession()
	if err != nil {
		return nil, err, true
	}
	if c, ok := sess.(interface{ Close() error }); ok {
		defer func() { _ = c.Close() }()
	}
	needsSuppress := len(suppress) > 0
	if needsSuppress || transform != nil {
		if streamSess, ok := sess.(nativeTextGreedySuppressTransformStreamSession); ok {
			out, err := streamSess.GenerateEachWithSuppressionAndTransform(ids, maxNew, eos, suppress, transform, yield)
			return out, err, true
		}
		if needsSuppress && transform == nil {
			if streamSess, ok := sess.(nativeTextGreedySuppressStreamSession); ok {
				out, err := streamSess.GenerateEachWithSuppression(ids, maxNew, eos, suppress, yield)
				return out, err, true
			}
			return nil, nil, false
		}
		if streamSess, ok := sess.(nativeTextGreedyTransformStreamSession); ok {
			out, err := streamSess.GenerateEachTransformed(ids, maxNew, eos, transform, yield)
			return out, err, true
		}
		return nil, nil, false
	}
	streamSess, ok := sess.(nativeTextGreedyStreamSession)
	if !ok {
		return nil, nil, false
	}
	out, err := streamSess.GenerateEach(ids, maxNew, eos, yield)
	return out, err, true
}

func (m *nativeTextModel) streamSampledSession(ids []int32, maxNew int, stopTokens []int32, sampler *model.Sampler, params model.SampleParams, transform model.TokenTransform, yield func(int32) bool) ([]int32, error, bool) {
	sm, ok := m.tm.(model.SessionModel)
	if !ok {
		return nil, nil, false
	}
	sess, err := sm.OpenSession()
	if err != nil {
		return nil, err, true
	}
	if c, ok := sess.(interface{ Close() error }); ok {
		defer func() { _ = c.Close() }()
	}
	if streamSess, ok := sess.(nativeTextSampledOneShotStreamSession); ok {
		out, err := streamSess.GenerateSampledOneShotEach(ids, maxNew, stopTokens, sampler, params, transform, yield)
		return out, err, true
	}
	streamSess, ok := sess.(nativeTextSampledStreamSession)
	if !ok {
		return nil, nil, false
	}
	out, err := streamSess.GenerateSampledEach(ids, maxNew, stopTokens, sampler, params, transform, yield)
	return out, err, true
}

func (m *nativeTextModel) WarmPromptCache(ctx context.Context, prompt string) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	ids := m.tok.Encode(prompt)
	m.mu.Lock()
	defer m.mu.Unlock()
	cacheSess, err := m.promptCacheSessionLocked()
	if err != nil {
		m.lastErr = err
		return err
	}
	if err := cacheSess.WarmPromptCache(ids); err != nil {
		m.lastErr = err
		return err
	}
	m.lastErr = nil
	return ctx.Err()
}

func (m *nativeTextModel) CacheStats(ctx context.Context) (inference.CacheStats, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return inference.CacheStats{}, err
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.cacheStatsLocked(nil), ctx.Err()
}

func (m *nativeTextModel) WarmCache(ctx context.Context, req inference.CacheWarmRequest) (inference.CacheWarmResult, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return inference.CacheWarmResult{}, err
	}
	ids := append([]int32(nil), req.Tokens...)
	if len(ids) == 0 && req.Prompt != "" {
		ids = m.tok.Encode(req.Prompt)
	}
	labels := nativeCacheLabels(req.Labels)

	m.mu.Lock()
	defer m.mu.Unlock()
	if len(ids) > 0 {
		cacheSess, err := m.promptCacheSessionLocked()
		if err != nil {
			m.lastErr = err
			return inference.CacheWarmResult{}, err
		}
		if err := cacheSess.WarmPromptCache(ids); err != nil {
			m.lastErr = err
			return inference.CacheWarmResult{}, err
		}
	}
	m.lastErr = nil
	stats := m.cacheStatsLocked(labels)
	result := inference.CacheWarmResult{
		Stats:  stats,
		Labels: labels,
	}
	if len(ids) > 0 {
		result.Blocks = []inference.CacheBlockRef{{
			ID:         "native-prompt",
			Kind:       "prompt",
			TokenStart: 0,
			TokenCount: len(ids),
			Labels:     nativeCacheLabels(labels),
		}}
	}
	return result, ctx.Err()
}

func (m *nativeTextModel) WarmPromptCacheChunks(ctx context.Context, chunks iter.Seq[string]) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	ids, err := m.encodePromptChunks(ctx, chunks, "mlx.nativeTextModel.WarmPromptCacheChunks")
	if err != nil {
		m.setErr(err)
		return err
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	cacheSess, err := m.promptCacheSessionLocked()
	if err != nil {
		m.lastErr = err
		return err
	}
	if err := cacheSess.WarmPromptCache(ids); err != nil {
		m.lastErr = err
		return err
	}
	m.lastErr = nil
	return ctx.Err()
}

func (m *nativeTextModel) ClearPromptCache() {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.cacheSess != nil {
		m.cacheSess.ClearPromptCache()
	}
}

func (m *nativeTextModel) ClearCache(ctx context.Context, labels map[string]string) (inference.CacheStats, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return inference.CacheStats{}, err
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.cacheSess != nil {
		m.cacheSess.ClearPromptCache()
	}
	return m.cacheStatsLocked(labels), ctx.Err()
}

func (m *nativeTextModel) cacheStatsLocked(labels map[string]string) inference.CacheStats {
	stats := inference.CacheStats{
		CacheMode: "native-prompt",
		Labels:    nativeCacheLabels(labels),
	}
	if m.cacheSess == nil {
		return stats
	}
	if sizer, ok := m.cacheSess.(nativeTextPromptCacheSizer); ok {
		if sizer.CachedPrefixLen() > 0 {
			stats.Blocks = 1
		}
		return stats
	}
	stats.Blocks = 1
	return stats
}

func nativeCacheLabels(labels map[string]string) map[string]string {
	if len(labels) == 0 {
		return nil
	}
	out := make(map[string]string, len(labels))
	for k, v := range labels {
		out[k] = v
	}
	return out
}

func (m *nativeTextModel) encodePromptChunks(ctx context.Context, chunks iter.Seq[string], scope string) ([]int32, error) {
	if m == nil || m.tok == nil {
		return nil, core.NewError("mlx.nativeTextModel: tokenizer is nil")
	}
	if chunks == nil {
		return nil, core.NewError("mlx.nativeTextModel: prompt chunks are nil")
	}
	if scope == "" {
		scope = "mlx.nativeTextModel.GenerateChunks"
	}
	tokens := make([]int32, 0, 256)
	seenContent := false
	for chunk := range chunks {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if chunk == "" {
			continue
		}
		ids := m.tok.Encode(chunk)
		if seenContent {
			ids = stripNativeImplicitChunkBOS(m.tok, ids)
		}
		if len(ids) == 0 {
			continue
		}
		tokens = append(tokens, ids...)
		seenContent = true
	}
	if len(tokens) == 0 {
		return nil, core.NewError(scope + ": empty prompt after tokenisation")
	}
	return tokens, nil
}

func stripNativeImplicitChunkBOS(tok *tokenizer.Tokenizer, ids []int32) []int32 {
	if tok == nil || !tok.HasBOSToken() || len(ids) == 0 {
		return ids
	}
	if ids[0] != tok.BOSToken() {
		return ids
	}
	return ids[1:]
}

func (m *nativeTextModel) promptCacheSessionLocked() (nativeTextPromptCacheSession, error) {
	if m.cacheSess != nil {
		return m.cacheSess, nil
	}
	sm, ok := m.tm.(model.SessionModel)
	if !ok {
		return nil, core.NewError("mlx.nativeTextModel: prompt cache requires a session model")
	}
	sess, err := sm.OpenSession()
	if err != nil {
		return nil, err
	}
	cacheSess, ok := sess.(nativeTextPromptCacheSession)
	if !ok {
		if c, closeOK := sess.(interface{ Close() error }); closeOK {
			_ = c.Close()
		}
		return nil, core.NewError("mlx.nativeTextModel: native session does not support prompt cache")
	}
	m.cacheSess = cacheSess
	return cacheSess, nil
}

func (m *nativeTextModel) NewSession() metal.SessionHandle {
	sess, err := m.openNativeRootSession()
	if err != nil {
		return nil
	}
	return &nativeTextSession{model: m, sess: sess}
}

func (m *nativeTextModel) openNativeRootSession() (nativeTextRootSession, error) {
	if m == nil || m.tm == nil {
		return nil, core.NewError("mlx.nativeTextModel.NewSession: model is not initialised")
	}
	sm, ok := m.tm.(model.SessionModel)
	if !ok {
		return nil, core.NewError("mlx.nativeTextModel.NewSession: token model does not support sessions")
	}
	stepper, err := sm.OpenSession()
	if err != nil {
		return nil, err
	}
	sess, ok := stepper.(nativeTextRootSession)
	if !ok {
		if c, closeOK := stepper.(interface{ Close() error }); closeOK {
			_ = c.Close()
		}
		return nil, core.NewError("mlx.nativeTextModel.NewSession: native session does not support state blocks")
	}
	return sess, nil
}

func (s *nativeTextSession) Prefill(ctx context.Context, prompt string) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.readyLocked("mlx.nativeTextSession.Prefill"); err != nil {
		s.err = err
		return err
	}
	if s.model.tok == nil {
		err := core.NewError("mlx.nativeTextSession.Prefill: tokenizer is nil")
		s.err = err
		return err
	}
	ids := s.model.tok.Encode(prompt)
	return s.prefillTokensLocked(ctx, ids)
}

func (s *nativeTextSession) PrefillTokens(ctx context.Context, tokens []int32) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.readyLocked("mlx.nativeTextSession.PrefillTokens"); err != nil {
		s.err = err
		return err
	}
	return s.prefillTokensLocked(ctx, tokens)
}

func (s *nativeTextSession) prefillTokensLocked(ctx context.Context, tokens []int32) error {
	if len(tokens) == 0 {
		err := core.NewError("mlx.nativeTextSession.PrefillTokens: empty prompt tokens")
		s.err = err
		return err
	}
	if err := ctx.Err(); err != nil {
		s.err = err
		return err
	}
	ids := append([]int32(nil), tokens...)
	if err := s.sess.PrefillTokens(ids); err != nil {
		s.err = err
		return err
	}
	logits, err := s.sess.BoundaryLogits()
	if err != nil {
		s.err = err
		return err
	}
	s.tokens = ids
	s.generated = nil
	s.logits = append(s.logits[:0], logits...)
	s.err = nil
	return ctx.Err()
}

func (s *nativeTextSession) AppendPrompt(ctx context.Context, prompt string) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.readyLocked("mlx.nativeTextSession.AppendPrompt"); err != nil {
		s.err = err
		return err
	}
	if len(s.tokens) == 0 {
		err := core.NewError("mlx.nativeTextSession.AppendPrompt: no retained prefix")
		s.err = err
		return err
	}
	if s.model.tok == nil {
		err := core.NewError("mlx.nativeTextSession.AppendPrompt: tokenizer is nil")
		s.err = err
		return err
	}
	ids := stripNativeImplicitChunkBOS(s.model.tok, s.model.tok.Encode(prompt))
	return s.appendTokensLocked(ctx, ids)
}

func (s *nativeTextSession) AppendTokens(ctx context.Context, tokens []int32) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.readyLocked("mlx.nativeTextSession.AppendTokens"); err != nil {
		s.err = err
		return err
	}
	if len(s.tokens) == 0 {
		err := core.NewError("mlx.nativeTextSession.AppendTokens: no retained prefix")
		s.err = err
		return err
	}
	return s.appendTokensLocked(ctx, stripNativeImplicitChunkBOS(s.model.tok, tokens))
}

func (s *nativeTextSession) appendTokensLocked(ctx context.Context, tokens []int32) error {
	if len(tokens) == 0 {
		err := core.NewError("mlx.nativeTextSession.AppendTokens: empty prompt tokens")
		s.err = err
		return err
	}
	if err := ctx.Err(); err != nil {
		s.err = err
		return err
	}
	ids := append([]int32(nil), tokens...)
	if err := s.sess.AppendTokens(ids); err != nil {
		s.err = err
		return err
	}
	logits, err := s.sess.BoundaryLogits()
	if err != nil {
		s.err = err
		return err
	}
	s.tokens = append(s.tokens, ids...)
	s.generated = nil
	s.logits = append(s.logits[:0], logits...)
	s.err = nil
	return ctx.Err()
}

func (s *nativeTextSession) Generate(ctx context.Context, cfg metal.GenerateConfig) iter.Seq[metal.Token] {
	return func(yield func(metal.Token) bool) {
		if ctx == nil {
			ctx = context.Background()
		}
		s.mu.Lock()
		defer s.mu.Unlock()
		if err := s.readyForGenerateLocked("mlx.nativeTextSession.Generate"); err != nil {
			s.err = err
			return
		}
		maxNew := cfg.MaxTokens
		if maxNew <= 0 || s.sess.Pos()+maxNew > s.model.maxLen {
			maxNew = s.model.maxLen - s.sess.Pos()
		}
		if maxNew <= 0 {
			s.err = core.NewError("mlx.nativeTextSession.Generate: no room to generate")
			return
		}
		stopTokens := s.stopTokens(cfg)
		eos := singleStopToken(stopTokens)
		emit := func(id int32) bool {
			if err := ctx.Err(); err != nil {
				s.err = err
				return false
			}
			if yield != nil && !yield(metal.Token{ID: id, Text: s.decodeToken(id)}) {
				return false
			}
			return !tokenInSet(id, stopTokens)
		}
		var (
			out []int32
			err error
		)
		sampled := cfg.Temperature > 0 || cfg.MinP > 0 || cfg.RepeatPenalty > 1 || cfg.MinTokensBeforeStop > 0 || len(cfg.SuppressTokens) > 0
		if sampled {
			params := model.SampleParams{
				Temperature:         cfg.Temperature,
				TopK:                cfg.TopK,
				TopP:                cfg.TopP,
				MinP:                cfg.MinP,
				SuppressTokens:      cfg.SuppressTokens,
				MinTokensBeforeStop: cfg.MinTokensBeforeStop,
				RepeatPenalty:       cfg.RepeatPenalty,
			}
			out, err = s.sess.GenerateSampledFromCacheEach(maxNew, stopTokens, model.NewSampler(nativeMetalSamplerSeed(cfg)), params, nil, emit)
		} else {
			out, err = s.sess.GenerateFromCacheEach(maxNew, eos, emit)
		}
		if err != nil {
			s.err = err
			return
		}
		s.tokens = append(s.tokens, out...)
		s.generated = append(s.generated, out...)
		logits, err := s.sess.BoundaryLogits()
		if err != nil {
			s.logits = nil
			s.err = err
			return
		}
		s.logits = append(s.logits[:0], logits...)
		s.err = ctx.Err()
	}
}

func (s *nativeTextSession) CaptureKV(ctx context.Context) (*metal.KVSnapshot, error) {
	return s.CaptureKVWithOptions(ctx, metal.KVSnapshotCaptureOptions{})
}

func (s *nativeTextSession) CaptureKVWithOptions(ctx context.Context, opts metal.KVSnapshotCaptureOptions) (*metal.KVSnapshot, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.readyForGenerateLocked("mlx.nativeTextSession.CaptureKV"); err != nil {
		s.err = err
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		s.err = err
		return nil, err
	}
	source, err := s.sess.StateBlockSource(max(1, s.sess.Pos()))
	if err != nil {
		s.err = err
		return nil, err
	}
	if source.BlockCount != 1 {
		err := core.NewError("mlx.nativeTextSession.CaptureKV: unexpected state block count")
		s.err = err
		return nil, err
	}
	block, err := source.Load(0)
	if err != nil {
		s.err = err
		return nil, err
	}
	snapshot := s.snapshotFromNativeBlock(source, block, true, true)
	if opts.RawKVOnly {
		nativeTextDropFloat32KV(snapshot)
	}
	s.err = nil
	return snapshot, ctx.Err()
}

func (s *nativeTextSession) RangeKVBlocks(ctx context.Context, blockSize int, opts metal.KVSnapshotCaptureOptions, yield func(metal.KVSnapshotBlock) (bool, error)) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if yield == nil {
		return core.NewError("mlx.nativeTextSession.RangeKVBlocks: nil yield")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.readyForGenerateLocked("mlx.nativeTextSession.RangeKVBlocks"); err != nil {
		s.err = err
		return err
	}
	source, err := s.sess.StateBlockSourceFrom(opts.BlockStartToken, blockSize)
	if err != nil {
		s.err = err
		return err
	}
	for i := 0; i < source.BlockCount; i++ {
		if err := ctx.Err(); err != nil {
			s.err = err
			return err
		}
		block, err := source.Load(i)
		if err != nil {
			s.err = err
			return err
		}
		final := block.TokenStart+block.TokenCount == source.Position
		snapshot := s.snapshotFromNativeBlock(source, block, final, false)
		if opts.RawKVOnly {
			nativeTextDropFloat32KV(snapshot)
		}
		ok, err := yield(metal.KVSnapshotBlock{
			Index:      block.Index,
			TokenStart: block.TokenStart,
			TokenCount: block.TokenCount,
			Snapshot:   snapshot,
		})
		if err != nil || !ok {
			s.err = err
			return err
		}
	}
	s.err = nil
	return nil
}

func (s *nativeTextSession) RestoreKV(ctx context.Context, snapshot *metal.KVSnapshot) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if snapshot == nil {
		return core.NewError("mlx.nativeTextSession.RestoreKV: nil snapshot")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.readyLocked("mlx.nativeTextSession.RestoreKV"); err != nil {
		s.err = err
		return err
	}
	if err := ctx.Err(); err != nil {
		s.err = err
		return err
	}
	source, tokens, generated, logits, err := nativeTextStateSourceFromSnapshot(snapshot)
	if err != nil {
		s.err = err
		return err
	}
	if err := s.sess.RestoreStateBlocks(source); err != nil {
		s.err = err
		return err
	}
	s.tokens = tokens
	s.generated = generated
	s.logits = logits
	s.err = nil
	return ctx.Err()
}

func (s *nativeTextSession) RestoreKVBlocks(ctx context.Context, source metal.KVSnapshotBlockSource) error {
	if ctx == nil {
		ctx = context.Background()
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.readyLocked("mlx.nativeTextSession.RestoreKVBlocks"); err != nil {
		s.err = err
		return err
	}
	restored, tokens, logits, err := nativeTextStateSourceFromBlockSource(ctx, source, s.tokens)
	if err != nil {
		s.err = err
		return err
	}
	if err := s.sess.RestoreStateBlocks(restored); err != nil {
		s.err = err
		return err
	}
	s.tokens = tokens
	s.generated = nil
	s.logits = logits
	s.err = nil
	return ctx.Err()
}

func (s *nativeTextSession) Fork(ctx context.Context) (metal.SessionHandle, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	snapshot, err := s.CaptureKV(ctx)
	if err != nil {
		return nil, err
	}
	fork := s.model.NewSession()
	if fork == nil {
		return nil, core.NewError("mlx.nativeTextSession.Fork: native model returned nil session")
	}
	restorer, ok := fork.(interface {
		RestoreKV(context.Context, *metal.KVSnapshot) error
	})
	if !ok {
		_ = fork.Close()
		return nil, core.NewError("mlx.nativeTextSession.Fork: forked session cannot restore KV")
	}
	if err := restorer.RestoreKV(ctx, snapshot); err != nil {
		_ = fork.Close()
		return nil, err
	}
	return fork, nil
}

func (s *nativeTextSession) Reset() {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.model == nil || s.closed {
		s.tokens = nil
		s.generated = nil
		s.logits = nil
		return
	}
	next, err := s.model.openNativeRootSession()
	if err != nil {
		s.err = err
		return
	}
	old := s.sess
	s.sess = next
	s.tokens = nil
	s.generated = nil
	s.logits = nil
	s.err = nil
	if old != nil {
		_ = old.Close()
	}
}

func (s *nativeTextSession) Close() error {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return s.err
	}
	s.closed = true
	s.tokens = nil
	s.generated = nil
	s.logits = nil
	if s.sess == nil {
		return s.err
	}
	err := s.sess.Close()
	if err != nil {
		s.err = err
	}
	s.sess = nil
	return err
}

func (s *nativeTextSession) Err() error {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.err
}

func (s *nativeTextSession) readyLocked(scope string) error {
	if s == nil || s.model == nil || s.sess == nil {
		return core.NewError(scope + ": nil session")
	}
	if s.closed {
		return core.NewError(scope + ": session is closed")
	}
	return nil
}

func (s *nativeTextSession) readyForGenerateLocked(scope string) error {
	if err := s.readyLocked(scope); err != nil {
		return err
	}
	if len(s.tokens) == 0 || s.sess.Pos() <= 0 {
		return core.NewError(scope + ": no retained prompt state")
	}
	return nil
}

func (s *nativeTextSession) stopTokens(cfg metal.GenerateConfig) []int32 {
	if len(cfg.StopTokens) > 0 {
		return cfg.StopTokens
	}
	if s != nil && s.model != nil && s.model.tok != nil && s.model.tok.HasEOSToken() {
		return []int32{s.model.tok.EOSToken()}
	}
	return nil
}

func (s *nativeTextSession) decodeToken(id int32) string {
	if s == nil || s.model == nil || s.model.tok == nil {
		return ""
	}
	return s.model.tok.DecodeToken(id)
}

func (s *nativeTextSession) snapshotFromNativeBlock(source native.SessionStateBlockSource, block native.SessionStateBlock, final, includeGenerated bool) *metal.KVSnapshot {
	layers := make([]metal.KVLayerSnapshot, len(block.Layers))
	numHeads, headDim := 0, 0
	for i, layer := range block.Layers {
		if numHeads == 0 {
			numHeads = layer.KVHeads
		}
		if headDim == 0 {
			headDim = layer.HeadDim
		}
		cacheMode := metal.KVCacheModeFixed
		if layer.CacheMode != "" {
			cacheMode = metal.KVCacheMode(layer.CacheMode)
		}
		layers[i] = metal.KVLayerSnapshot{
			Layer:      layer.Layer,
			CacheIndex: layer.CacheIndex,
			CacheMode:  cacheMode,
			MaxSize:    layer.MaxSize,
			KeyDType:   metal.DTypeBFloat16,
			KeyBytes:   append([]byte(nil), layer.KeyBytes...),
			KeyShape:   []int32{int32(block.TokenCount), int32(layer.KVHeads), int32(layer.HeadDim)},
			ValueDType: metal.DTypeBFloat16,
			ValueBytes: append([]byte(nil), layer.ValueBytes...),
			ValueShape: []int32{int32(block.TokenCount), int32(layer.KVHeads), int32(layer.HeadDim)},
		}
	}
	tokens := s.snapshotTokens(source, block.TokenStart, block.TokenCount)
	var generated []int32
	if includeGenerated {
		generated = append([]int32(nil), s.generated...)
	}
	var (
		logitShape []int32
		logits     []float32
	)
	if final && len(s.logits) > 0 {
		logitShape = []int32{1, 1, int32(s.modelVocab())}
		logits = nativeTextBF16ToF32(s.logits)
	}
	return &metal.KVSnapshot{
		Version:       metal.KVSnapshotVersion,
		Architecture:  s.modelArchitecture(),
		Tokens:        tokens,
		Generated:     generated,
		TokenOffset:   block.TokenStart + block.TokenCount,
		NumLayers:     nativeTextSnapshotLayerCount(layers, s.modelNumLayers()),
		NumHeads:      numHeads,
		SeqLen:        block.TokenCount,
		HeadDim:       headDim,
		NumQueryHeads: numHeads,
		LogitShape:    logitShape,
		Logits:        logits,
		Layers:        layers,
	}
}

func (s *nativeTextSession) snapshotTokens(source native.SessionStateBlockSource, start, count int) []int32 {
	tokens := s.tokens
	if len(source.CachedIDs) >= source.Position {
		tokens = source.CachedIDs
	}
	if start < 0 || count <= 0 || start+count > len(tokens) {
		return nil
	}
	return append([]int32(nil), tokens[start:start+count]...)
}

func (s *nativeTextSession) modelArchitecture() string {
	if s == nil || s.model == nil {
		return ""
	}
	if s.model.info.Architecture != "" {
		return s.model.info.Architecture
	}
	return s.model.modelType
}

func (s *nativeTextSession) modelNumLayers() int {
	if s == nil || s.model == nil {
		return 0
	}
	return s.model.info.NumLayers
}

func (s *nativeTextSession) modelVocab() int {
	if s == nil || s.model == nil {
		return 0
	}
	if s.model.info.VocabSize > 0 {
		return s.model.info.VocabSize
	}
	if s.model.tm != nil {
		return s.model.tm.Vocab()
	}
	return 0
}

func nativeTextSnapshotLayerCount(layers []metal.KVLayerSnapshot, fallback int) int {
	if fallback > 0 {
		return fallback
	}
	n := 0
	for _, layer := range layers {
		if layer.Layer >= n {
			n = layer.Layer + 1
		}
	}
	return n
}

func nativeTextStateSourceFromSnapshot(snapshot *metal.KVSnapshot) (native.SessionStateBlockSource, []int32, []int32, []byte, error) {
	if snapshot == nil {
		return native.SessionStateBlockSource{}, nil, nil, nil, core.NewError("mlx.nativeTextSession.RestoreKV: nil snapshot")
	}
	position := snapshot.SeqLen
	if position <= 0 {
		position = len(snapshot.Tokens)
	}
	if position <= 0 {
		return native.SessionStateBlockSource{}, nil, nil, nil, core.NewError("mlx.nativeTextSession.RestoreKV: empty snapshot")
	}
	if len(snapshot.Tokens) > position {
		return native.SessionStateBlockSource{}, nil, nil, nil, core.NewError("mlx.nativeTextSession.RestoreKV: token state exceeds sequence length")
	}
	layers, err := nativeTextStateLayersFromSnapshot(snapshot, position)
	if err != nil {
		return native.SessionStateBlockSource{}, nil, nil, nil, err
	}
	if len(layers) == 0 {
		return native.SessionStateBlockSource{}, nil, nil, nil, core.NewError("mlx.nativeTextSession.RestoreKV: snapshot has no native KV slabs")
	}
	tokens := append([]int32(nil), snapshot.Tokens...)
	logits := nativeTextF32ToBF16(snapshot.Logits)
	source := native.SessionStateBlockSource{
		Position:           position,
		CachedIDs:          append([]int32(nil), tokens...),
		CachedPromptIDs:    append([]int32(nil), tokens...),
		CachedPromptLogits: append([]byte(nil), logits...),
		RetainedLogits:     append([]byte(nil), logits...),
		BlockCount:         1,
	}
	source.Load = func(index int) (native.SessionStateBlock, error) {
		if index != 0 {
			return native.SessionStateBlock{}, core.NewError("mlx.nativeTextSession.RestoreKV: block index out of range")
		}
		return native.SessionStateBlock{Index: 0, TokenStart: 0, TokenCount: position, Layers: layers}, nil
	}
	return source, tokens, append([]int32(nil), snapshot.Generated...), logits, nil
}

func nativeTextStateSourceFromBlockSource(ctx context.Context, source metal.KVSnapshotBlockSource, residentTokens []int32) (native.SessionStateBlockSource, []int32, []byte, error) {
	if source.BlockCount <= 0 || source.Load == nil {
		return native.SessionStateBlockSource{}, nil, nil, core.NewError("mlx.nativeTextSession.RestoreKVBlocks: invalid block source")
	}
	position := source.PrefixTokens
	if position <= 0 || position > source.TokenCount {
		position = source.TokenCount
	}
	if position <= 0 {
		return native.SessionStateBlockSource{}, nil, nil, core.NewError("mlx.nativeTextSession.RestoreKVBlocks: empty block source")
	}
	blocks := make([]native.SessionStateBlock, 0, source.BlockCount)
	tokens := make([]int32, position)
	filledUntil := 0
	trustedFirstBlock, trustedBlockSize := 0, 0
	var logits []byte
	for i := 0; i < source.BlockCount && filledUntil < position; i++ {
		if err := ctx.Err(); err != nil {
			return native.SessionStateBlockSource{}, nil, nil, err
		}
		block, err := source.Load(ctx, i)
		if err != nil {
			return native.SessionStateBlockSource{}, nil, nil, err
		}
		if block.Snapshot == nil {
			return native.SessionStateBlockSource{}, nil, nil, core.NewError("mlx.nativeTextSession.RestoreKVBlocks: nil block snapshot")
		}
		if block.TokenStart < 0 || block.TokenCount <= 0 {
			continue
		}
		take := block.TokenCount
		if block.TokenStart+take > position {
			take = position - block.TokenStart
		}
		if take <= 0 {
			continue
		}
		if block.TokenStart > filledUntil {
			if filledUntil != 0 || block.TokenStart > len(residentTokens) {
				return native.SessionStateBlockSource{}, nil, nil, core.NewError("mlx.nativeTextSession.RestoreKVBlocks: block tokens do not cover position")
			}
			copy(tokens[:block.TokenStart], residentTokens[:block.TokenStart])
			filledUntil = block.TokenStart
			if block.Index > 0 && block.TokenStart%block.Index == 0 {
				trustedFirstBlock = block.Index
				trustedBlockSize = block.TokenStart / block.Index
			}
		}
		if block.TokenStart != filledUntil || len(block.Snapshot.Tokens) < take {
			return native.SessionStateBlockSource{}, nil, nil, core.NewError("mlx.nativeTextSession.RestoreKVBlocks: block tokens do not cover position")
		}
		layers, err := nativeTextStateLayersFromBlock(block.Snapshot, take)
		if err != nil {
			return native.SessionStateBlockSource{}, nil, nil, err
		}
		blocks = append(blocks, native.SessionStateBlock{
			Index:      block.Index,
			TokenStart: block.TokenStart,
			TokenCount: take,
			Layers:     layers,
		})
		copy(tokens[block.TokenStart:block.TokenStart+take], block.Snapshot.Tokens[:take])
		filledUntil += take
		if block.TokenStart+take == position && len(block.Snapshot.Logits) > 0 {
			logits = nativeTextF32ToBF16(block.Snapshot.Logits)
		}
	}
	if filledUntil != position {
		return native.SessionStateBlockSource{}, nil, nil, core.NewError("mlx.nativeTextSession.RestoreKVBlocks: block tokens do not cover position")
	}
	restored := native.SessionStateBlockSource{
		Position:           position,
		CachedIDs:          append([]int32(nil), tokens...),
		CachedPromptIDs:    append([]int32(nil), tokens...),
		CachedPromptLogits: append([]byte(nil), logits...),
		RetainedLogits:     append([]byte(nil), logits...),
		BlockCount:         len(blocks),
	}
	restored.Load = func(index int) (native.SessionStateBlock, error) {
		if index < 0 || index >= len(blocks) {
			return native.SessionStateBlock{}, core.NewError("mlx.nativeTextSession.RestoreKVBlocks: block index out of range")
		}
		return blocks[index], nil
	}
	if trustedFirstBlock > 0 && trustedBlockSize > 0 {
		if err := restored.TrustPrefixBlocks(trustedBlockSize, trustedFirstBlock); err != nil {
			return native.SessionStateBlockSource{}, nil, nil, err
		}
	}
	return restored, tokens, logits, nil
}

func nativeTextStateLayersFromSnapshot(snapshot *metal.KVSnapshot, tokenCount int) ([]native.SessionStateLayerBlock, error) {
	layers := make([]native.SessionStateLayerBlock, 0, len(snapshot.Layers))
	for _, layer := range snapshot.Layers {
		stateLayer, err := nativeTextStateLayerFromMetal(layer, tokenCount)
		if err != nil {
			return nil, err
		}
		if len(stateLayer.KeyBytes) == 0 && len(stateLayer.ValueBytes) == 0 {
			continue
		}
		layers = append(layers, stateLayer)
	}
	return layers, nil
}

func nativeTextStateLayersFromBlock(snapshot *metal.KVSnapshot, tokenCount int) ([]native.SessionStateLayerBlock, error) {
	layers := make([]native.SessionStateLayerBlock, 0, len(snapshot.Layers))
	for _, layer := range snapshot.Layers {
		stateLayer, err := nativeTextStateLayerFromMetal(layer, snapshot.SeqLen)
		if err != nil {
			return nil, err
		}
		if len(stateLayer.KeyBytes) == 0 && len(stateLayer.ValueBytes) == 0 {
			continue
		}
		n := tokenCount * stateLayer.RowBytes
		if n > len(stateLayer.KeyBytes) || n > len(stateLayer.ValueBytes) {
			return nil, core.NewError("mlx.nativeTextSession.RestoreKVBlocks: block payload shorter than token count")
		}
		stateLayer.KeyBytes = stateLayer.KeyBytes[:n]
		stateLayer.ValueBytes = stateLayer.ValueBytes[:n]
		layers = append(layers, stateLayer)
	}
	return layers, nil
}

func nativeTextStateLayerFromMetal(layer metal.KVLayerSnapshot, tokenCount int) (native.SessionStateLayerBlock, error) {
	if len(layer.KeyBytes) == 0 && len(layer.ValueBytes) == 0 {
		return native.SessionStateLayerBlock{}, nil
	}
	if layer.KeyDType != 0 && layer.KeyDType != metal.DTypeBFloat16 {
		return native.SessionStateLayerBlock{}, core.NewError("mlx.nativeTextSession.RestoreKV: key dtype must be bfloat16")
	}
	if layer.ValueDType != 0 && layer.ValueDType != metal.DTypeBFloat16 {
		return native.SessionStateLayerBlock{}, core.NewError("mlx.nativeTextSession.RestoreKV: value dtype must be bfloat16")
	}
	kvHeads, headDim, rowBytes, err := nativeTextLayerGeometry(layer, tokenCount)
	if err != nil {
		return native.SessionStateLayerBlock{}, err
	}
	n := tokenCount * rowBytes
	if len(layer.KeyBytes) != n || len(layer.ValueBytes) != n {
		return native.SessionStateLayerBlock{}, core.NewError("mlx.nativeTextSession.RestoreKV: native KV slab size mismatch")
	}
	return native.SessionStateLayerBlock{
		Layer:      layer.Layer,
		CacheIndex: layer.CacheIndex,
		CacheMode:  string(layer.CacheMode),
		MaxSize:    layer.MaxSize,
		KVHeads:    kvHeads,
		HeadDim:    headDim,
		RowBytes:   rowBytes,
		KeyBytes:   layer.KeyBytes,
		ValueBytes: layer.ValueBytes,
	}, nil
}

func nativeTextLayerGeometry(layer metal.KVLayerSnapshot, tokenCount int) (int, int, int, error) {
	kvHeads, headDim := 0, 0
	if len(layer.KeyShape) == 3 && int(layer.KeyShape[0]) == tokenCount {
		kvHeads, headDim = int(layer.KeyShape[1]), int(layer.KeyShape[2])
	} else if len(layer.KeyShape) == 4 && int(layer.KeyShape[2]) == tokenCount {
		kvHeads, headDim = int(layer.KeyShape[1]), int(layer.KeyShape[3])
	}
	if kvHeads <= 0 || headDim <= 0 {
		return 0, 0, 0, core.NewError("mlx.nativeTextSession.RestoreKV: missing native KV geometry")
	}
	rowBytes := kvHeads * headDim * 2
	return kvHeads, headDim, rowBytes, nil
}

func nativeTextBF16ToF32(raw []byte) []float32 {
	if len(raw) == 0 {
		return nil
	}
	out := make([]float32, len(raw)/2)
	for i := range out {
		off := i * 2
		out[i] = math.Float32frombits(uint32(uint16(raw[off])|uint16(raw[off+1])<<8) << 16)
	}
	return out
}

func nativeTextF32ToBF16(values []float32) []byte {
	if len(values) == 0 {
		return nil
	}
	out := make([]byte, len(values)*2)
	for i, v := range values {
		bits := math.Float32bits(v)
		h := uint16(bits >> 16)
		out[i*2] = byte(h)
		out[i*2+1] = byte(h >> 8)
	}
	return out
}

func nativeTextDropFloat32KV(snapshot *metal.KVSnapshot) {
	if snapshot == nil {
		return
	}
	for i := range snapshot.Layers {
		for j := range snapshot.Layers[i].Heads {
			snapshot.Layers[i].Heads[j].Key = nil
			snapshot.Layers[i].Heads[j].Value = nil
		}
	}
}

func (m *nativeTextModel) setErr(err error) {
	m.mu.Lock()
	m.lastErr = err
	m.mu.Unlock()
}

func (m *nativeTextModel) setMetrics(promptTokens, genTokens int, total time.Duration) {
	m.setMetricsWithThinkingBudget(promptTokens, genTokens, total, false)
}

func (m *nativeTextModel) setMetricsWithThinkingBudget(promptTokens, genTokens int, total time.Duration, thinkingBudgetForced bool) {
	tps := 0.0
	if total > 0 {
		tps = float64(genTokens) / total.Seconds()
	}
	m.mu.Lock()
	m.lastErr = nil
	m.lastMetrics = inference.GenerateMetrics{
		PromptTokens:         promptTokens,
		GeneratedTokens:      genTokens,
		TotalDuration:        total,
		DecodeDuration:       total,
		DecodeTokensPerSec:   tps,
		ThinkingBudgetForced: thinkingBudgetForced,
	}
	m.mu.Unlock()
}

// Classify samples one token per prompt (greedy) — the prefill-only fast path
// approximated over the contract (the contract has no batched prefill; one short
// Generate per prompt).
func (m *nativeTextModel) Classify(ctx context.Context, prompts []string, opts ...inference.GenerateOption) ([]inference.ClassifyResult, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if len(prompts) == 0 {
		return []inference.ClassifyResult{}, nil
	}
	if m == nil || m.tm == nil || m.tok == nil {
		return nil, core.NewError("mlx.nativeTextModel.Classify: model is not initialised")
	}
	start := time.Now()
	cfg := inference.ApplyGenerateOpts(opts)
	encoded := make([][]int32, len(prompts))
	totalPromptTokens := 0
	for i, p := range prompts {
		encoded[i] = m.tok.Encode(p)
		totalPromptTokens += len(encoded[i])
	}
	results := make([]inference.ClassifyResult, len(prompts))
	sampler := model.NewSampler(nativeSamplerSeed(cfg))
	for i := range encoded {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		logits, err := nativePromptLogits(m.tm, encoded[i])
		if err != nil {
			return nil, err
		}
		id, err := sampler.Sample(logits, m.tm.Vocab(), model.SampleParams{Temperature: cfg.Temperature, TopK: cfg.TopK, TopP: cfg.TopP, MinP: cfg.MinP, SuppressTokens: cfg.SuppressTokens})
		if err != nil {
			return nil, err
		}
		results[i] = inference.ClassifyResult{Token: inference.Token{ID: id, Text: m.tok.DecodeToken(id)}}
		if cfg.ReturnLogits {
			results[i].Logits, err = nativeBF16LogitsToF32(logits, m.tm.Vocab())
			if err != nil {
				return nil, err
			}
		}
	}
	m.setClassifyMetrics(totalPromptTokens, len(results), time.Since(start))
	return results, nil
}

func nativePromptLogits(tm model.TokenModel, ids []int32) ([]byte, error) {
	if tm == nil {
		return nil, core.NewError("mlx.nativePromptLogits: nil model")
	}
	if len(ids) == 0 {
		return nil, core.NewError("mlx.nativePromptLogits: empty prompt")
	}
	if sm, ok := tm.(model.SessionModel); ok {
		return nativePromptLogitsStepwise(sm, ids)
	}
	return nativePromptLogitsWholeSeq(tm, ids)
}

func nativePromptLogitsStepwise(tm model.SessionModel, ids []int32) ([]byte, error) {
	sess, err := tm.OpenSession()
	if err != nil {
		return nil, err
	}
	if c, ok := sess.(interface{ Close() error }); ok {
		defer func() { _ = c.Close() }()
	}
	stepID, idAware := sess.(interface {
		StepWithID(id int32, emb []byte) ([]byte, error)
	})
	var hidden []byte
	for _, id := range ids {
		emb, err := tm.Embed(id)
		if err != nil {
			return nil, err
		}
		if idAware {
			hidden, err = stepID.StepWithID(id, emb)
		} else {
			hidden, err = sess.Step(emb)
		}
		if err != nil {
			return nil, err
		}
	}
	if len(hidden) == 0 {
		return nil, core.NewError("mlx.nativePromptLogits: session returned no hidden state")
	}
	return tm.Head(hidden)
}

func nativePromptLogitsWholeSeq(tm model.TokenModel, ids []int32) ([]byte, error) {
	seq := make([][]byte, 0, len(ids))
	for _, id := range ids {
		emb, err := tm.Embed(id)
		if err != nil {
			return nil, err
		}
		seq = append(seq, emb)
	}
	hidden, err := tm.DecodeForward(seq)
	if err != nil {
		return nil, err
	}
	if len(hidden) == 0 {
		return nil, core.NewError("mlx.nativePromptLogits: backend returned no hidden states")
	}
	return tm.Head(hidden[len(hidden)-1])
}

func nativeBF16LogitsToF32(logits []byte, vocab int) ([]float32, error) {
	if len(logits) != vocab*2 {
		return nil, core.NewError("mlx.nativeBF16LogitsToF32: logits must be vocab bf16 bytes")
	}
	out := make([]float32, vocab)
	for i := 0; i < vocab; i++ {
		off := i * 2
		out[i] = math.Float32frombits(uint32(uint16(logits[off])|uint16(logits[off+1])<<8) << 16)
	}
	return out, nil
}

func (m *nativeTextModel) setClassifyMetrics(promptTokens, generatedTokens int, total time.Duration) {
	tps := 0.0
	if total > 0 {
		tps = float64(promptTokens) / total.Seconds()
	}
	m.mu.Lock()
	m.lastErr = nil
	m.lastMetrics = inference.GenerateMetrics{
		PromptTokens:        promptTokens,
		GeneratedTokens:     generatedTokens,
		PrefillDuration:     total,
		TotalDuration:       total,
		PrefillTokensPerSec: tps,
	}
	m.mu.Unlock()
}

func (m *nativeTextModel) setBatchGenerateMetrics(promptTokens, generatedTokens int, total time.Duration, err error) {
	tps := 0.0
	if total > 0 {
		tps = float64(generatedTokens) / total.Seconds()
	}
	m.mu.Lock()
	m.lastErr = err
	m.lastMetrics = inference.GenerateMetrics{
		PromptTokens:       promptTokens,
		GeneratedTokens:    generatedTokens,
		DecodeDuration:     total,
		TotalDuration:      total,
		DecodeTokensPerSec: tps,
	}
	m.mu.Unlock()
}

// BatchGenerate runs one Generate per prompt (the contract is single-sequence; no
// true batching — that is a pkg/metal scheduler feature).
func (m *nativeTextModel) BatchGenerate(ctx context.Context, prompts []string, opts ...inference.GenerateOption) ([]inference.BatchResult, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	cfg := inference.ApplyGenerateOpts(opts)
	results := make([]inference.BatchResult, len(prompts))
	totalStart := time.Now()
	totalPromptTokens := 0
	totalGenerated := 0
	var batchErr error
	for i, p := range prompts {
		ids := m.tok.Encode(p)
		totalPromptTokens += len(ids)
		var toks []inference.Token
		for tok := range m.stream(ctx, ids, cfg) {
			toks = append(toks, tok)
		}
		totalGenerated += len(toks)
		err := m.Err()
		if batchErr == nil {
			batchErr = err
		}
		results[i] = inference.BatchResult{Tokens: toks, Err: err}
	}
	if len(prompts) > 0 {
		m.setBatchGenerateMetrics(totalPromptTokens, totalGenerated, time.Since(totalStart), batchErr)
	}
	return results, nil
}

func (m *nativeTextModel) ModelType() string { return m.modelType }

func (m *nativeTextModel) Info() inference.ModelInfo { return m.info }

func (m *nativeTextModel) Metrics() inference.GenerateMetrics {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.lastMetrics
}

func (m *nativeTextModel) Err() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.lastErr
}

// Close releases any retained prompt-cache session. The resident weights live
// for the process in the serve shape; a warmed cache session is explicit mutable
// state and should be dropped on teardown or hot-swap.
func (m *nativeTextModel) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.cacheSess == nil {
		return nil
	}
	defer func() { m.cacheSess = nil }()
	if c, ok := m.cacheSess.(interface{ Close() error }); ok {
		return c.Close()
	}
	return nil
}
