// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package mlx

import (
	"context"
	"iter"
	"sync"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
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
// v1 generates the whole completion via model.Generate, then yields the tokens —
// correct for non-streaming requests; per-token streaming-as-decoded is a follow-up
// (a model.GenerateStream callback). Close is a no-op: the resident weights live for
// the process (a single served model), matching the load-once serve shape.
type nativeTextModel struct {
	tm        model.TokenModel
	tok       *tokenizer.Tokenizer
	modelType string
	info      inference.ModelInfo
	maxLen    int

	mu          sync.Mutex
	lastErr     error
	lastMetrics inference.GenerateMetrics
}

var _ inference.TextModel = (*nativeTextModel)(nil)

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
		eos := -1
		if m.tok.HasEOSToken() {
			eos = int(m.tok.EOSToken())
		}
		var (
			out []int32
			err error
		)
		if cfg.Temperature > 0 { // stochastic; greedy otherwise (deterministic)
			sampler := model.NewSampler(uint64(time.Now().UnixNano()))
			out, err = model.GenerateSampled(m.tm, sampler, model.SampleParams{Temperature: cfg.Temperature, TopK: cfg.TopK, TopP: cfg.TopP}, ids, maxNew, eos)
		} else {
			out, err = model.Generate(m.tm, ids, maxNew, eos)
		}
		if err != nil {
			m.setErr(err)
			return
		}
		for _, id := range out {
			if ctx.Err() != nil {
				return
			}
			if !yield(inference.Token{ID: id, Text: m.tok.DecodeToken(id)}) {
				return
			}
		}
		m.setMetrics(len(ids), len(out), time.Since(start))
	}
}

func (m *nativeTextModel) setErr(err error) {
	m.mu.Lock()
	m.lastErr = err
	m.mu.Unlock()
}

func (m *nativeTextModel) setMetrics(promptTokens, genTokens int, total time.Duration) {
	tps := 0.0
	if total > 0 {
		tps = float64(genTokens) / total.Seconds()
	}
	m.mu.Lock()
	m.lastErr = nil
	m.lastMetrics = inference.GenerateMetrics{
		PromptTokens:       promptTokens,
		GeneratedTokens:    genTokens,
		TotalDuration:      total,
		DecodeDuration:     total,
		DecodeTokensPerSec: tps,
	}
	m.mu.Unlock()
}

// Classify samples one token per prompt (greedy) — the prefill-only fast path
// approximated over the contract (the contract has no batched prefill; one short
// Generate per prompt).
func (m *nativeTextModel) Classify(ctx context.Context, prompts []string, opts ...inference.GenerateOption) ([]inference.ClassifyResult, error) {
	results := make([]inference.ClassifyResult, len(prompts))
	for i, p := range prompts {
		out, err := model.Generate(m.tm, m.tok.Encode(p), 1, -1)
		if err != nil {
			return nil, err
		}
		if len(out) > 0 {
			results[i] = inference.ClassifyResult{Token: inference.Token{ID: out[0], Text: m.tok.DecodeToken(out[0])}}
		}
	}
	return results, nil
}

// BatchGenerate runs one Generate per prompt (the contract is single-sequence; no
// true batching — that is a pkg/metal scheduler feature).
func (m *nativeTextModel) BatchGenerate(ctx context.Context, prompts []string, opts ...inference.GenerateOption) ([]inference.BatchResult, error) {
	cfg := inference.ApplyGenerateOpts(opts)
	results := make([]inference.BatchResult, len(prompts))
	for i, p := range prompts {
		var toks []inference.Token
		for tok := range m.stream(ctx, m.tok.Encode(p), cfg) {
			toks = append(toks, tok)
		}
		results[i] = inference.BatchResult{Tokens: toks, Err: m.Err()}
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

// Close is a no-op: the contract opens + frees a session per Generate (the Close
// hook), and the resident weights live for the process (a single served model).
func (m *nativeTextModel) Close() error { return nil }
