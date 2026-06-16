// SPDX-Licence-Identifier: EUPL-1.2

package openai

import (
	"context"
	"iter"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	anthropiccompat "dappco.re/go/inference/anthropic"
	ollamacompat "dappco.re/go/inference/ollama"
	openaicompat "dappco.re/go/inference/openai"
)

func TestOpenai_NewResolver_Good(t *testing.T) {
	resolver := NewResolver("/models/qwen3")
	if resolver == nil {
		t.Fatal("NewResolver() returned nil")
	}
	if resolver.BackendName != "metal" {
		t.Fatalf("BackendName = %q, want metal", resolver.BackendName)
	}
	if resolver.ModelPath != "/models/qwen3" {
		t.Fatalf("ModelPath = %q", resolver.ModelPath)
	}
}

func TestOpenai_NewHandler_Good(t *testing.T) {
	handler := NewHandler("/models/qwen3")
	if handler == nil {
		t.Fatal("NewHandler() returned nil")
	}
}

type openAIMockModel struct {
	tokens       []inference.Token
	metrics      inference.GenerateMetrics
	cancelled    string
	warmed       inference.CacheWarmRequest
	cacheEntries []inference.CacheBlockRef
	arch         string
	quantBits    int
	err          error
}

func (m *openAIMockModel) Generate(context.Context, string, ...inference.GenerateOption) iter.Seq[inference.Token] {
	return m.seq()
}

func (m *openAIMockModel) Chat(context.Context, []inference.Message, ...inference.GenerateOption) iter.Seq[inference.Token] {
	return m.seq()
}

func (m *openAIMockModel) Classify(context.Context, []string, ...inference.GenerateOption) ([]inference.ClassifyResult, error) {
	return nil, nil
}

func (m *openAIMockModel) BatchGenerate(context.Context, []string, ...inference.GenerateOption) ([]inference.BatchResult, error) {
	return nil, nil
}

func (m *openAIMockModel) ModelType() string { return "mock" }
func (m *openAIMockModel) Info() inference.ModelInfo {
	arch := m.arch
	if arch == "" {
		arch = "qwen3"
	}
	return inference.ModelInfo{Architecture: arch, QuantBits: m.quantBits}
}
func (m *openAIMockModel) Metrics() inference.GenerateMetrics { return m.metrics }
func (m *openAIMockModel) Err() error                         { return m.err }
func (m *openAIMockModel) Close() error                       { return nil }

func (m *openAIMockModel) Embed(_ context.Context, req inference.EmbeddingRequest) (*inference.EmbeddingResult, error) {
	return &inference.EmbeddingResult{
		Vectors: [][]float32{{float32(len(req.Input)), 1}},
		Usage:   inference.EmbeddingUsage{PromptTokens: len(req.Input), TotalTokens: len(req.Input)},
	}, nil
}

func (m *openAIMockModel) Rerank(_ context.Context, req inference.RerankRequest) (*inference.RerankResult, error) {
	return &inference.RerankResult{Results: []inference.RerankScore{{Index: 0, Score: 0.75, Text: req.Documents[0]}}}, nil
}

func (m *openAIMockModel) CacheStats(context.Context) (inference.CacheStats, error) {
	return inference.CacheStats{Blocks: 2, Hits: 3, Misses: 1, HitRate: 0.75, CacheMode: "block-q8"}, nil
}

func (m *openAIMockModel) WarmCache(_ context.Context, req inference.CacheWarmRequest) (inference.CacheWarmResult, error) {
	m.warmed = req
	return inference.CacheWarmResult{Blocks: []inference.CacheBlockRef{{ID: "blk", TokenCount: len(req.Tokens)}}}, nil
}

func (m *openAIMockModel) ClearCache(context.Context, map[string]string) (inference.CacheStats, error) {
	return inference.CacheStats{CacheMode: "block-q8"}, nil
}

func (m *openAIMockModel) CacheEntries(context.Context, map[string]string) ([]inference.CacheBlockRef, error) {
	return append([]inference.CacheBlockRef(nil), m.cacheEntries...), nil
}

func (m *openAIMockModel) CancelRequest(_ context.Context, id string) (inference.RequestCancelResult, error) {
	m.cancelled = id
	return inference.RequestCancelResult{ID: id, Cancelled: id != ""}, nil
}

func (m *openAIMockModel) seq() iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {
		for _, token := range m.tokens {
			if !yield(token) {
				return
			}
		}
	}
}

type openAISchedulerModel struct {
	openAIMockModel
}

func (m *openAISchedulerModel) Schedule(_ context.Context, req inference.ScheduledRequest) (inference.RequestHandle, <-chan inference.ScheduledToken, error) {
	ch := make(chan inference.ScheduledToken, 1)
	ch <- inference.ScheduledToken{RequestID: req.ID, Token: inference.Token{Text: "scheduled"}}
	close(ch)
	return inference.RequestHandle{ID: req.ID}, ch, nil
}

func TestOpenai_NewMux_Good(t *testing.T) {
	model := &openAIMockModel{
		tokens:  []inference.Token{{Text: "<think>plan</think>Answer"}},
		metrics: inference.GenerateMetrics{PromptTokens: 2, GeneratedTokens: 3},
	}
	resolver := openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": model})
	handler := NewMux(resolver)
	if handler == nil {
		t.Fatal("NewMux() returned nil")
	}

	cases := []struct {
		name   string
		method string
		path   string
		body   string
		want   string
	}{
		{
			name:   "chat",
			method: http.MethodPost,
			path:   openaicompat.DefaultChatCompletionsPath,
			body:   `{"model":"qwen","messages":[{"role":"user","content":"hi"}]}`,
			want:   `"content":"Answer"`,
		},
		{
			name:   "responses",
			method: http.MethodPost,
			path:   openaicompat.DefaultResponsesPath,
			body:   `{"model":"qwen","input":[{"role":"user","content":"hi"}]}`,
			want:   `"text":"Answer"`,
		},
		{
			name:   "embeddings",
			method: http.MethodPost,
			path:   openaicompat.DefaultEmbeddingsPath,
			body:   `{"model":"qwen","input":["alpha","beta"]}`,
			want:   `"embedding":[2,1]`,
		},
		{
			name:   "rerank",
			method: http.MethodPost,
			path:   openaicompat.DefaultRerankPath,
			body:   `{"model":"qwen","query":"core","documents":["doc"]}`,
			want:   `"score":0.75`,
		},
		{
			name:   "cache stats",
			method: http.MethodGet,
			path:   openaicompat.DefaultCacheStatsPath + "?model=qwen",
			want:   `"hit_rate":0.75`,
		},
		{
			name:   "cache warm",
			method: http.MethodPost,
			path:   openaicompat.DefaultCacheWarmPath,
			body:   `{"model":"qwen","tokens":[1,2,3]}`,
			want:   `"token_count":3`,
		},
		{
			name:   "cancel",
			method: http.MethodPost,
			path:   openaicompat.DefaultCancelPath,
			body:   `{"model":"qwen","id":"req_1"}`,
			want:   `"cancelled":true`,
		},
		{
			name:   "capabilities",
			method: http.MethodGet,
			path:   openaicompat.DefaultCapabilitiesPath + "?model=qwen",
			want:   `"embeddings"`,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req := httptest.NewRequest(tc.method, tc.path, strings.NewReader(tc.body))
			rec := httptest.NewRecorder()

			handler.ServeHTTP(rec, req)

			if rec.Code != http.StatusOK {
				t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
			}
			if !strings.Contains(rec.Body.String(), tc.want) {
				t.Fatalf("body = %s, want %s", rec.Body.String(), tc.want)
			}
		})
	}
	if model.cancelled != "req_1" {
		t.Fatalf("cancelled = %q, want req_1", model.cancelled)
	}
	if model.warmed.Model.ID != "qwen" || len(model.warmed.Tokens) != 3 {
		t.Fatalf("warmed = %+v", model.warmed)
	}
}

func TestOpenai_NewMux_Good_MountsAnthropicAndOllama(t *testing.T) {
	model := &openAIMockModel{
		tokens:  []inference.Token{{Text: "<think>plan</think>Answer"}},
		metrics: inference.GenerateMetrics{PromptTokens: 2, GeneratedTokens: 3},
	}
	resolver := openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": model})
	handler := NewMux(resolver)

	cases := []struct {
		name   string
		method string
		path   string
		body   string
		want   string
	}{
		{
			name:   "anthropic messages",
			method: http.MethodPost,
			path:   anthropiccompat.DefaultMessagesPath,
			body:   `{"model":"qwen","system":"be terse","messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}],"max_tokens":32}`,
			want:   `"text":"Answer"`,
		},
		{
			name:   "ollama chat",
			method: http.MethodPost,
			path:   ollamacompat.DefaultChatPath,
			body:   `{"model":"qwen","messages":[{"role":"user","content":"hi"}],"options":{"num_predict":32}}`,
			want:   `"content":"Answer"`,
		},
		{
			name:   "ollama generate",
			method: http.MethodPost,
			path:   ollamacompat.DefaultGeneratePath,
			body:   `{"model":"qwen","prompt":"hi","options":{"num_predict":32}}`,
			want:   `"response":"Answer"`,
		},
		{
			name:   "ollama show",
			method: http.MethodPost,
			path:   ollamacompat.DefaultShowPath,
			body:   `{"model":"qwen"}`,
			want:   `"architecture":"qwen3"`,
		},
		{
			name:   "ollama tags",
			method: http.MethodGet,
			path:   ollamacompat.DefaultTagsPath,
			want:   `"models"`,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req := httptest.NewRequest(tc.method, tc.path, strings.NewReader(tc.body))
			rec := httptest.NewRecorder()

			handler.ServeHTTP(rec, req)

			if rec.Code != http.StatusOK {
				t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
			}
			if !strings.Contains(rec.Body.String(), tc.want) {
				t.Fatalf("body = %s, want %s", rec.Body.String(), tc.want)
			}
		})
	}
}

func TestOpenAI_AnthropicMessages_Good_AppliesStopSequences(t *testing.T) {
	model := &openAIMockModel{
		tokens:  []inference.Token{{Text: "Answer STOP hidden"}},
		metrics: inference.GenerateMetrics{PromptTokens: 2, GeneratedTokens: 3},
	}
	resolver := openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": model})
	handler := NewMux(resolver)

	req := httptest.NewRequest(http.MethodPost, anthropiccompat.DefaultMessagesPath, strings.NewReader(`{"model":"qwen","messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}],"stop_sequences":[" STOP"]}`))
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	body := rec.Body.String()
	if !strings.Contains(body, `"text":"Answer"`) {
		t.Fatalf("body = %s, want stopped answer", body)
	}
	if strings.Contains(body, "hidden") {
		t.Fatalf("body = %s, stop sequence was not applied", body)
	}
}

// TestOpenAI_AnthropicMessages_StopSequenceAcrossTokens locks the
// cumulative-accumulation path serveAnthropicMessageStream walks when a stop
// sequence is only completed by joining successive tokens. The single-token
// case above never exercises the cross-boundary scan (emitted+delta), so this
// pins the exact streamed deltas: "STOP" first appears once "Answer S" + "TOP …"
// are joined, the cut lands inside the second token, and everything from the
// cut on (including the already-buffered residue past it) must be withheld.
func TestOpenAI_AnthropicMessages_StopSequenceAcrossTokens(t *testing.T) {
	model := &openAIMockModel{
		tokens:  []inference.Token{{Text: "Answer "}, {Text: "S"}, {Text: "TOP hidden"}},
		metrics: inference.GenerateMetrics{PromptTokens: 1, GeneratedTokens: 3},
	}
	resolver := openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": model})
	handler := NewMux(resolver)

	req := httptest.NewRequest(http.MethodPost, anthropiccompat.DefaultMessagesPath, strings.NewReader(`{"model":"qwen","stream":true,"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}],"stop_sequences":["STOP"]}`))
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	body := rec.Body.String()
	// "Answer " and "S" stream before the boundary completes "STOP"; the cut
	// then withholds the third token entirely (its delta resolves to empty).
	for _, want := range []string{`"text":"Answer "`, `"text":"S"`, `"stop_reason":"stop_sequence"`} {
		if !strings.Contains(body, want) {
			t.Fatalf("body = %s, want %s", body, want)
		}
	}
	if strings.Contains(body, "TOP") || strings.Contains(body, "hidden") {
		t.Fatalf("body = %s, stop sequence residue leaked", body)
	}
}

func TestOpenAI_OllamaGenerate_Good_StreamsJSONLines(t *testing.T) {
	model := &openAIMockModel{
		tokens:  []inference.Token{{Text: "An"}, {Text: "swer"}},
		metrics: inference.GenerateMetrics{PromptTokens: 1, GeneratedTokens: 2},
	}
	resolver := openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": model})
	handler := NewMux(resolver)

	req := httptest.NewRequest(http.MethodPost, ollamacompat.DefaultGeneratePath, strings.NewReader(`{"model":"qwen","prompt":"hi","stream":true}`))
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	body := rec.Body.String()
	if !strings.Contains(body, `"response":"An"`) || !strings.Contains(body, `"response":"swer"`) || !strings.Contains(body, `"done":true`) {
		t.Fatalf("body = %s, want streamed deltas and final done", body)
	}
}

func TestOpenAI_Responses_Good_StreamsServerSentEvents(t *testing.T) {
	model := &openAIMockModel{
		tokens:  []inference.Token{{Text: "An"}, {Text: "swer"}},
		metrics: inference.GenerateMetrics{PromptTokens: 1, GeneratedTokens: 2},
	}
	resolver := openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": model})
	handler := NewMux(resolver)

	req := httptest.NewRequest(http.MethodPost, openaicompat.DefaultResponsesPath, strings.NewReader(`{"model":"qwen","stream":true,"input":[{"role":"user","content":"hi"}]}`))
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	body := rec.Body.String()
	for _, want := range []string{"response.created", "response.output_text.delta", `"delta":"An"`, `"delta":"swer"`, "response.completed", "data: [DONE]"} {
		if !strings.Contains(body, want) {
			t.Fatalf("body = %s, want %s", body, want)
		}
	}
}

func TestOpenAI_AnthropicMessages_Good_StreamsEvents(t *testing.T) {
	model := &openAIMockModel{
		tokens:  []inference.Token{{Text: "An"}, {Text: "swer"}},
		metrics: inference.GenerateMetrics{PromptTokens: 1, GeneratedTokens: 2},
	}
	resolver := openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": model})
	handler := NewMux(resolver)

	req := httptest.NewRequest(http.MethodPost, anthropiccompat.DefaultMessagesPath, strings.NewReader(`{"model":"qwen","stream":true,"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`))
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	body := rec.Body.String()
	for _, want := range []string{"event: message_start", "event: content_block_delta", `"text":"An"`, `"text":"swer"`, "event: message_stop"} {
		if !strings.Contains(body, want) {
			t.Fatalf("body = %s, want %s", body, want)
		}
	}
}

func TestOpenAI_OllamaChat_Good_StreamsJSONLines(t *testing.T) {
	model := &openAIMockModel{
		tokens:  []inference.Token{{Text: "An"}, {Text: "swer"}},
		metrics: inference.GenerateMetrics{PromptTokens: 1, GeneratedTokens: 2},
	}
	resolver := openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": model})
	handler := NewMux(resolver)

	req := httptest.NewRequest(http.MethodPost, ollamacompat.DefaultChatPath, strings.NewReader(`{"model":"qwen","stream":true,"messages":[{"role":"user","content":"hi"}]}`))
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	body := rec.Body.String()
	if !strings.Contains(body, `"content":"An"`) || !strings.Contains(body, `"content":"swer"`) || !strings.Contains(body, `"done":true`) {
		t.Fatalf("body = %s, want streamed chat deltas and final done", body)
	}
}

func TestOpenai_NewMuxWithAdmin_Good(t *testing.T) {
	model := &openAIMockModel{
		cacheEntries: []inference.CacheBlockRef{{
			ID:         "blk-a",
			Kind:       "prefix",
			TokenCount: 16,
			Labels:     map[string]string{"tenant": "local"},
		}},
	}
	resolver := openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": model})
	var woke, slept bool
	handler := NewMuxWithAdmin(resolver, AdminConfig{
		Wake: func(context.Context) error {
			woke = true
			return nil
		},
		Sleep: func(context.Context) error {
			slept = true
			return nil
		},
	})

	cases := []struct {
		name   string
		method string
		path   string
		want   string
	}{
		{name: "health", method: http.MethodGet, path: DefaultHealthPath, want: `"status":"ok"`},
		{name: "wake", method: http.MethodPost, path: DefaultAdminWakePath, want: `"action":"wake"`},
		{name: "sleep", method: http.MethodPost, path: DefaultAdminSleepPath, want: `"action":"sleep"`},
		{name: "cache entries", method: http.MethodGet, path: DefaultAdminCacheEntriesPath + "?model=qwen&tenant=local", want: `"id":"blk-a"`},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req := httptest.NewRequest(tc.method, tc.path, nil)
			rec := httptest.NewRecorder()

			handler.ServeHTTP(rec, req)

			if rec.Code != http.StatusOK {
				t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
			}
			if !strings.Contains(rec.Body.String(), tc.want) {
				t.Fatalf("body = %s, want %s", rec.Body.String(), tc.want)
			}
		})
	}
	if !woke || !slept {
		t.Fatalf("woke=%v slept=%v, want callbacks invoked", woke, slept)
	}
}

func TestOpenAI_AdminCacheEntries_Bad_RequiresEntryLister(t *testing.T) {
	model := &openAITextOnlyModel{}
	resolver := openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": model})
	handler := NewMuxWithAdmin(resolver, AdminConfig{})

	req := httptest.NewRequest(http.MethodGet, DefaultAdminCacheEntriesPath+"?model=qwen", nil)
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusNotImplemented {
		t.Fatalf("status = %d body=%s, want 501", rec.Code, rec.Body.String())
	}
}

type openAITextOnlyModel struct{}

func (m *openAITextOnlyModel) Generate(context.Context, string, ...inference.GenerateOption) iter.Seq[inference.Token] {
	return func(func(inference.Token) bool) {}
}

func (m *openAITextOnlyModel) Chat(context.Context, []inference.Message, ...inference.GenerateOption) iter.Seq[inference.Token] {
	return func(func(inference.Token) bool) {}
}

func (m *openAITextOnlyModel) Classify(context.Context, []string, ...inference.GenerateOption) ([]inference.ClassifyResult, error) {
	return nil, nil
}

func (m *openAITextOnlyModel) BatchGenerate(context.Context, []string, ...inference.GenerateOption) ([]inference.BatchResult, error) {
	return nil, nil
}

func (m *openAITextOnlyModel) ModelType() string { return "text-only" }
func (m *openAITextOnlyModel) Info() inference.ModelInfo {
	return inference.ModelInfo{Architecture: "qwen3"}
}
func (m *openAITextOnlyModel) Metrics() inference.GenerateMetrics { return inference.GenerateMetrics{} }
func (m *openAITextOnlyModel) Err() error                         { return nil }
func (m *openAITextOnlyModel) Close() error                       { return nil }

func TestOpenAI_Responses_Good_UsesSchedulerModel(t *testing.T) {
	model := &openAISchedulerModel{openAIMockModel: openAIMockModel{
		tokens: []inference.Token{{Text: "direct"}},
	}}
	resolver := openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": model})
	handler := NewMux(resolver)

	req := httptest.NewRequest(http.MethodPost, openaicompat.DefaultResponsesPath, strings.NewReader(`{"model":"qwen","input":[{"role":"user","content":"hi"}]}`))
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), `"text":"scheduled"`) {
		t.Fatalf("body = %s, want scheduled text", rec.Body.String())
	}
	if strings.Contains(rec.Body.String(), `"text":"direct"`) {
		t.Fatalf("body = %s, bypassed scheduler", rec.Body.String())
	}
}

func TestOpenAI_Responses_Good_UsesModelParserRegistry(t *testing.T) {
	model := &openAIMockModel{
		arch:   "gpt_oss",
		tokens: []inference.Token{{Text: "<|channel>analysis\nplan<|channel>final\nAnswer"}},
	}
	resolver := openaicompat.NewStaticResolver(map[string]inference.TextModel{"gpt-oss": model})
	handler := NewMux(resolver)

	req := httptest.NewRequest(http.MethodPost, openaicompat.DefaultResponsesPath, strings.NewReader(`{"model":"gpt-oss","input":[{"role":"user","content":"hi"}]}`))
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	body := rec.Body.String()
	if !strings.Contains(body, `"text":"Answer"`) {
		t.Fatalf("body = %s, want parsed visible answer", body)
	}
	if !strings.Contains(body, `"thought":"plan"`) {
		t.Fatalf("body = %s, want parsed thought", body)
	}
}

func TestOpenai_NewModelMux_Good(t *testing.T) {
	handler := NewModelMux("/models/qwen3")
	if handler == nil {
		t.Fatal("NewModelMux() returned nil")
	}
}

func TestOpenAI_Responses_Bad_ReportsRequestAndModelErrors(t *testing.T) {
	rec := httptest.NewRecorder()
	(&openAIResponsesHandler{}).ServeHTTP(rec, httptest.NewRequest(http.MethodPost, openaicompat.DefaultResponsesPath, strings.NewReader(`{}`)))
	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("unconfigured status = %d body=%s", rec.Code, rec.Body.String())
	}
	rec = httptest.NewRecorder()
	newOpenAIResponsesHandler(openaicompat.NewStaticResolver(nil)).ServeHTTP(rec, nil)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("nil request status = %d body=%s", rec.Code, rec.Body.String())
	}
	rec = httptest.NewRecorder()
	newOpenAIResponsesHandler(openaicompat.NewStaticResolver(nil)).ServeHTTP(rec, httptest.NewRequest(http.MethodGet, openaicompat.DefaultResponsesPath, nil))
	if rec.Code != http.StatusMethodNotAllowed || rec.Header().Get("Allow") != http.MethodPost {
		t.Fatalf("method status/header = %d/%q", rec.Code, rec.Header().Get("Allow"))
	}
	rec = httptest.NewRecorder()
	newOpenAIResponsesHandler(openaicompat.NewStaticResolver(nil)).ServeHTTP(rec, httptest.NewRequest(http.MethodPost, openaicompat.DefaultResponsesPath, strings.NewReader(`{`)))
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("bad JSON status = %d body=%s", rec.Code, rec.Body.String())
	}
	rec = httptest.NewRecorder()
	newOpenAIResponsesHandler(openaicompat.NewStaticResolver(nil)).ServeHTTP(rec, httptest.NewRequest(http.MethodPost, openaicompat.DefaultResponsesPath, strings.NewReader(`{"input":"hi"}`)))
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("missing model status = %d body=%s", rec.Code, rec.Body.String())
	}
	rec = httptest.NewRecorder()
	newOpenAIResponsesHandler(openaicompat.NewStaticResolver(nil)).ServeHTTP(rec, httptest.NewRequest(http.MethodPost, openaicompat.DefaultResponsesPath, strings.NewReader(`{"model":"missing","input":[{"role":"user","content":"hi"}]}`)))
	if rec.Code != http.StatusNotFound {
		t.Fatalf("missing resolver model status = %d body=%s", rec.Code, rec.Body.String())
	}
	model := &openAIMockModel{tokens: []inference.Token{{Text: "Answer"}}, err: core.NewError("model failed")}
	rec = httptest.NewRecorder()
	newOpenAIResponsesHandler(openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": model})).ServeHTTP(rec, httptest.NewRequest(http.MethodPost, openaicompat.DefaultResponsesPath, strings.NewReader(`{"model":"qwen","input":[{"role":"user","content":"hi"}]}`)))
	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("model error status = %d body=%s", rec.Code, rec.Body.String())
	}
}

func TestOpenAI_AnthropicAndOllama_Bad_ReportsRequestErrors(t *testing.T) {
	rec := httptest.NewRecorder()
	(&anthropicMessagesHandler{}).ServeHTTP(rec, httptest.NewRequest(http.MethodPost, anthropiccompat.DefaultMessagesPath, strings.NewReader(`{}`)))
	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("anthropic unconfigured status = %d body=%s", rec.Code, rec.Body.String())
	}
	rec = httptest.NewRecorder()
	newAnthropicMessagesHandler(openaicompat.NewStaticResolver(nil)).ServeHTTP(rec, httptest.NewRequest(http.MethodGet, anthropiccompat.DefaultMessagesPath, nil))
	if rec.Code != http.StatusMethodNotAllowed || rec.Header().Get("Allow") != http.MethodPost {
		t.Fatalf("anthropic method status/header = %d/%q", rec.Code, rec.Header().Get("Allow"))
	}
	rec = httptest.NewRecorder()
	newAnthropicMessagesHandler(openaicompat.NewStaticResolver(nil)).ServeHTTP(rec, httptest.NewRequest(http.MethodPost, anthropiccompat.DefaultMessagesPath, strings.NewReader(`{"model":"qwen","messages":[],"stop_sequences":[""]}`)))
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("anthropic stop status = %d body=%s", rec.Code, rec.Body.String())
	}
	rec = httptest.NewRecorder()
	(&ollamaChatHandler{}).ServeHTTP(rec, httptest.NewRequest(http.MethodGet, ollamacompat.DefaultChatPath, nil))
	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("ollama method status = %d body=%s", rec.Code, rec.Body.String())
	}
	rec = httptest.NewRecorder()
	(&ollamaShowHandler{}).ServeHTTP(rec, httptest.NewRequest(http.MethodPost, ollamacompat.DefaultShowPath, strings.NewReader(`{"model":"qwen"}`)))
	if rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("ollama nil resolver status = %d body=%s", rec.Code, rec.Body.String())
	}
	rec = httptest.NewRecorder()
	newOllamaGenerateHandler(openaicompat.NewStaticResolver(nil)).ServeHTTP(rec, httptest.NewRequest(http.MethodPost, ollamacompat.DefaultGeneratePath, strings.NewReader(`{`)))
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("ollama bad JSON status = %d body=%s", rec.Code, rec.Body.String())
	}
}

type openAINameResolver struct{}

func (openAINameResolver) ResolveModel(context.Context, string) (inference.TextModel, error) {
	return nil, core.NewError("not found")
}

func (openAINameResolver) ModelNames() []string {
	return []string{"listed"}
}

func TestOpenAICompatHelpers_Good(t *testing.T) {
	if _, err := decodeOpenAIResponseRequest(strings.NewReader(`{"model":"qwen","input":[{"role":"user","content":"hi"}]}`), -1); err != nil {
		t.Fatalf("decodeOpenAIResponseRequest(valid) error = %v", err)
	}
	var payload map[string]string
	if err := decodeWireJSON(nil, &payload, "test"); err == nil {
		t.Fatal("decodeWireJSON(nil body) error = nil")
	}
	if err := decodeWireJSON(strings.NewReader(`{"a":"b"}`), &payload, "test"); err != nil || payload["a"] != "b" {
		t.Fatalf("decodeWireJSON(valid) = %+v/%v, want map", payload, err)
	}
	rec := httptest.NewRecorder()
	if requireCompatMethod(rec, nil, http.MethodPost) {
		t.Fatal("requireCompatMethod(nil request) = true")
	}
	rec = httptest.NewRecorder()
	if _, ok := resolveCompatModel(rec, context.Background(), nil, "qwen"); ok || rec.Code != http.StatusServiceUnavailable {
		t.Fatalf("resolve nil resolver = ok:%v status:%d", ok, rec.Code)
	}
	rec = httptest.NewRecorder()
	if _, ok := resolveCompatModel(rec, context.Background(), openaicompat.NewStaticResolver(nil), " "); ok || rec.Code != http.StatusBadRequest {
		t.Fatalf("resolve blank model = ok:%v status:%d", ok, rec.Code)
	}
	if names := resolverModelNames(openAINameResolver{}); len(names) != 1 || names[0] != "listed" {
		t.Fatalf("resolver names = %v, want listed", names)
	}
	if names := resolverModelNames(NewResolver("/models/qwen3")); len(names) != 1 || names[0] != "qwen3" {
		t.Fatalf("backend resolver names = %v, want qwen3", names)
	}
	if cut, ok := firstStopSequenceCut("alpha STOP beta END", []string{"END", " STOP"}); !ok || cut != len("alpha") {
		t.Fatalf("firstStopSequenceCut() = %d/%v, want earliest stop after alpha", cut, ok)
	}
	if stops, err := normalizeAnthropicStopSequences([]string{"END"}); err != nil || len(stops) != 1 || stops[0] != "END" {
		t.Fatalf("normalize stops = %v/%v", stops, err)
	}
	if got := openAITokensText([]inference.Token{{Text: "A"}, {Text: "B"}}); got != "AB" {
		t.Fatalf("openAITokensText() = %q, want AB", got)
	}
	if got := reasoningText([]inference.ReasoningSegment{{Text: "plan"}, {Text: " done"}}); got != "plan done" {
		t.Fatalf("reasoningText() = %q, want plan done", got)
	}
}

// errParserModel is a TextModel that satisfies inference.ReasoningParser and
// returns a parse error, so parseOpenAIModelOutput takes its error branch
// (which falls back to cleanChannelMarkers over the raw text).
type errParserModel struct {
	openAIMockModel
	result inference.ReasoningParseResult
	parErr error
}

func (m *errParserModel) ParseReasoning([]inference.Token, string) (inference.ReasoningParseResult, error) {
	return m.result, m.parErr
}

// TestOpenAIOpenAI_parseOpenAIModelOutput_GoodBadUgly exercises the three input
// classes of parseOpenAIModelOutput directly: a nil model (no parser, no info —
// the unreachable-via-handler branch), a ReasoningParser whose ParseReasoning
// errors (fall back to cleanChannelMarkers over the raw text), and a parser that
// returns empty visible text for non-empty input (the Gemma-4 unterminated-channel
// fallback that displays the full text rather than dropping the answer).
func TestOpenAIOpenAI_parseOpenAIModelOutput_GoodBadUgly(t *testing.T) {
	// Good: a parser that returns a clean split keeps the visible text and
	// joins the captured reasoning segments.
	good := &errParserModel{result: inference.ReasoningParseResult{
		VisibleText: "Answer",
		Reasoning:   []inference.ReasoningSegment{{Text: "plan"}},
	}}
	if visible, thought := parseOpenAIModelOutput(good, nil, "ignored"); visible != "Answer" || thought != "plan" {
		t.Fatalf("good parse = (%q,%q), want (Answer,plan)", visible, thought)
	}

	// Bad: the parser errors, so the helper falls back to cleaning channel
	// markers off the raw text and returns no thought.
	bad := &errParserModel{parErr: core.NewError("parse boom")}
	if visible, thought := parseOpenAIModelOutput(bad, nil, "<|channel>final\nAnswer"); visible != "Answer" || thought != "" {
		t.Fatalf("bad parse fallback = (%q,%q), want (Answer,)", visible, thought)
	}

	// Ugly: a nil model takes the parser.ForHint(Hint{}) path; a parser that
	// classifies everything as reasoning (empty visible) for non-empty text
	// must still surface the text rather than return an empty reply.
	if visible, _ := parseOpenAIModelOutput(nil, nil, "<|channel>thought\nfallback"); visible != "fallback" {
		t.Fatalf("nil model parse = %q, want fallback (markers cleaned)", visible)
	}
	empty := &errParserModel{result: inference.ReasoningParseResult{VisibleText: ""}}
	if visible, _ := parseOpenAIModelOutput(empty, nil, "<|channel>thought\nrescued"); visible != "rescued" {
		t.Fatalf("empty-visible fallback = %q, want rescued", visible)
	}
}

// TestOpenAIOpenAI_indexString_GoodBadUgly covers the substring locator's
// edge branches that firstStopSequenceCut's happy path never reaches: an
// empty needle (returns 0 by convention), a needle longer than the haystack
// (returns -1 before scanning), and a normal hit/miss.
func TestOpenAIOpenAI_indexString_GoodBadUgly(t *testing.T) {
	if got := indexString("alpha beta", "beta"); got != 6 {
		t.Fatalf("indexString(hit) = %d, want 6", got)
	}
	if got := indexString("alpha", "zz"); got != -1 {
		t.Fatalf("indexString(miss) = %d, want -1", got)
	}
	if got := indexString("alpha", ""); got != 0 {
		t.Fatalf("indexString(empty needle) = %d, want 0", got)
	}
	if got := indexString("ab", "abcdef"); got != -1 {
		t.Fatalf("indexString(needle longer than haystack) = %d, want -1", got)
	}
}

// TestOpenAIOpenAI_cleanChannelMarkers_GoodBadUgly pins the channel-marker
// stripper: text the parser already cleaned is returned trimmed (no-op),
// headered reasoning channels are removed, and bare residue markers are
// stripped without eating the surrounding answer.
func TestOpenAIOpenAI_cleanChannelMarkers_GoodBadUgly(t *testing.T) {
	if got := cleanChannelMarkers("  plain answer  "); got != "plain answer" {
		t.Fatalf("clean(plain) = %q, want trimmed plain answer", got)
	}
	if got := cleanChannelMarkers("<|channel>analysis\nthink<|channel>final\nAnswer"); got != "thinkAnswer" {
		t.Fatalf("clean(headered) = %q, want thinkAnswer", got)
	}
	if got := cleanChannelMarkers("<|channel>Answer<channel|>"); got != "Answer" {
		t.Fatalf("clean(bare residue) = %q, want Answer", got)
	}
}

// TestOpenAIAdmin_adminCacheEntryLabels_GoodBadUgly covers the request-level
// label extractor directly. It is production-equivalent to cacheEntryLabelsFrom
// but reads r.URL.Query() itself; the cache-entries handler now calls the split
// form, leaving this the bench's caller. Good: real filters drop the model key
// and trim values. Bad/Ugly: nil request and nil URL both yield an empty map
// rather than panicking.
func TestOpenAIAdmin_adminCacheEntryLabels_GoodBadUgly(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, DefaultAdminCacheEntriesPath+"?model=qwen&tenant=local&adapter=%20probe%20", nil)
	labels := adminCacheEntryLabels(req)
	if _, ok := labels["model"]; ok {
		t.Fatalf("labels still carry model key: %v", labels)
	}
	if labels["tenant"] != "local" || labels["adapter"] != "probe" {
		t.Fatalf("labels = %v, want tenant=local adapter=probe (trimmed)", labels)
	}
	if got := adminCacheEntryLabels(nil); len(got) != 0 {
		t.Fatalf("adminCacheEntryLabels(nil request) = %v, want empty", got)
	}
	if got := adminCacheEntryLabels(&http.Request{}); len(got) != 0 {
		t.Fatalf("adminCacheEntryLabels(nil URL) = %v, want empty", got)
	}
}

// TestOpenAIAdmin_HealthHandler_CustomCallback drives the health endpoint
// through NewMuxWithAdmin with a host-supplied Health closure, hitting the
// custom-callback branch of adminHealthHandler.ServeHTTP (25% baseline). The
// closure returns a partial Health to exercise the Status/Runtime/Time
// post-fill defaulting the handler applies on top of the host's payload.
func TestOpenAIAdmin_HealthHandler_CustomCallback(t *testing.T) {
	resolver := openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": &openAIMockModel{}})
	handler := NewMuxWithAdmin(resolver, AdminConfig{
		Health: func(context.Context) (Health, error) {
			// Status/Runtime/Time left zero so the handler must default them.
			return Health{Models: []string{"qwen3"}}, nil
		},
	})
	req := httptest.NewRequest(http.MethodGet, DefaultHealthPath, nil)
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	body := rec.Body.String()
	for _, want := range []string{`"status":"ok"`, `"runtime":"go-mlx"`, `"qwen3"`} {
		if !strings.Contains(body, want) {
			t.Fatalf("body = %s, want %s (defaulted field)", body, want)
		}
	}
	if !strings.Contains(body, `"time":`) {
		t.Fatalf("body = %s, want defaulted time", body)
	}
}

// TestOpenAIAdmin_HealthHandler_Bad_CallbackError covers the error branch of
// the custom Health callback: a non-nil error must surface as a 500 envelope.
func TestOpenAIAdmin_HealthHandler_Bad_CallbackError(t *testing.T) {
	resolver := openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": &openAIMockModel{}})
	handler := NewMuxWithAdmin(resolver, AdminConfig{
		Health: func(context.Context) (Health, error) {
			return Health{}, core.NewError("health probe failed")
		},
	})
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, DefaultHealthPath, nil))
	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("status = %d body=%s, want 500", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "health probe failed") {
		t.Fatalf("body = %s, want callback error message", rec.Body.String())
	}
}

// TestOpenAIAdmin_HealthHandler_Bad_WrongMethod confirms the health endpoint
// rejects a non-GET with 405 + an Allow: GET header.
func TestOpenAIAdmin_HealthHandler_Bad_WrongMethod(t *testing.T) {
	handler := NewMuxWithAdmin(openaicompat.NewStaticResolver(nil), AdminConfig{})
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultHealthPath, nil))
	if rec.Code != http.StatusMethodNotAllowed || rec.Header().Get("Allow") != http.MethodGet {
		t.Fatalf("status/allow = %d/%q, want 405/GET", rec.Code, rec.Header().Get("Allow"))
	}
}

// TestOpenAIAdmin_ActionHandler_Bad_CallbackError covers the wake/sleep
// callback error branch of adminActionHandler.ServeHTTP (70% baseline): a
// failing callback yields a 500 carrying the action name as the param.
func TestOpenAIAdmin_ActionHandler_Bad_CallbackError(t *testing.T) {
	handler := NewMuxWithAdmin(openaicompat.NewStaticResolver(nil), AdminConfig{
		Wake: func(context.Context) error { return core.NewError("wake failed") },
	})
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultAdminWakePath, nil))
	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("status = %d body=%s, want 500", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "wake failed") {
		t.Fatalf("body = %s, want wake error", rec.Body.String())
	}
}

// TestOpenAIAdmin_ActionHandler_Bad_WrongMethod confirms wake rejects GET.
func TestOpenAIAdmin_ActionHandler_Bad_WrongMethod(t *testing.T) {
	handler := NewMuxWithAdmin(openaicompat.NewStaticResolver(nil), AdminConfig{})
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, DefaultAdminSleepPath, nil))
	if rec.Code != http.StatusMethodNotAllowed || rec.Header().Get("Allow") != http.MethodPost {
		t.Fatalf("status/allow = %d/%q, want 405/POST", rec.Code, rec.Header().Get("Allow"))
	}
}

// cacheStatsErrModel is a CacheEntryLister whose CacheStats errors, so the
// cache-entries handler takes its CacheService-stats error branch after a
// successful entry list.
type cacheStatsErrModel struct {
	openAIMockModel
}

func (m *cacheStatsErrModel) CacheStats(context.Context) (inference.CacheStats, error) {
	return inference.CacheStats{}, core.NewError("stats unavailable")
}

// TestOpenAIAdmin_CacheEntries_Good_IncludesStats covers the happy path where
// the model is both a CacheEntryLister and a CacheService: the response embeds
// the cache stats block alongside the entries.
func TestOpenAIAdmin_CacheEntries_Good_IncludesStats(t *testing.T) {
	model := &openAIMockModel{cacheEntries: []inference.CacheBlockRef{{ID: "blk-a", TokenCount: 8}}}
	resolver := openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": model})
	handler := NewMuxWithAdmin(resolver, AdminConfig{})
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, DefaultAdminCacheEntriesPath+"?model=qwen", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	body := rec.Body.String()
	if !strings.Contains(body, `"id":"blk-a"`) || !strings.Contains(body, `"hit_rate":0.75`) {
		t.Fatalf("body = %s, want entries + embedded stats", body)
	}
}

// TestOpenAIAdmin_CacheEntries_Bad_StatsError covers the branch where entry
// listing succeeds but the subsequent CacheStats call errors — the handler
// must surface a 500 rather than a partial body.
func TestOpenAIAdmin_CacheEntries_Bad_StatsError(t *testing.T) {
	model := &cacheStatsErrModel{openAIMockModel: openAIMockModel{cacheEntries: []inference.CacheBlockRef{{ID: "blk-a"}}}}
	resolver := openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": model})
	handler := NewMuxWithAdmin(resolver, AdminConfig{})
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, DefaultAdminCacheEntriesPath+"?model=qwen", nil))
	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("status = %d body=%s, want 500", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "stats unavailable") {
		t.Fatalf("body = %s, want stats error", rec.Body.String())
	}
}

// TestOpenAIAdmin_CacheEntries_Bad_WrongMethod confirms the cache-entries
// endpoint rejects a non-GET request.
func TestOpenAIAdmin_CacheEntries_Bad_WrongMethod(t *testing.T) {
	handler := NewMuxWithAdmin(openaicompat.NewStaticResolver(nil), AdminConfig{})
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultAdminCacheEntriesPath, nil))
	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("status = %d, want 405", rec.Code)
	}
}

// TestOpenAI_AnthropicMessages_Bad_ModelError covers the model.Err() branch of
// anthropicMessagesHandler.ServeHTTP: a model that streams a token then reports
// an error post-collection must surface a 500.
func TestOpenAI_AnthropicMessages_Bad_ModelError(t *testing.T) {
	model := &openAIMockModel{tokens: []inference.Token{{Text: "partial"}}, err: core.NewError("anthropic model failed")}
	resolver := openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": model})
	handler := NewMux(resolver)
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, anthropiccompat.DefaultMessagesPath,
		strings.NewReader(`{"model":"qwen","messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`)))
	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("status = %d body=%s, want 500", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "anthropic model failed") {
		t.Fatalf("body = %s, want model error", rec.Body.String())
	}
}

// TestOpenAI_OllamaChat_Bad_ModelError covers the model.Err() branch of
// ollamaChatHandler.ServeHTTP (non-streaming path).
func TestOpenAI_OllamaChat_Bad_ModelError(t *testing.T) {
	model := &openAIMockModel{tokens: []inference.Token{{Text: "partial"}}, err: core.NewError("ollama chat failed")}
	resolver := openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": model})
	handler := NewMux(resolver)
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, ollamacompat.DefaultChatPath,
		strings.NewReader(`{"model":"qwen","messages":[{"role":"user","content":"hi"}]}`)))
	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("status = %d body=%s, want 500", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "ollama chat failed") {
		t.Fatalf("body = %s, want model error", rec.Body.String())
	}
}

// TestOpenAI_OllamaGenerate_Bad_ModelError covers the model.Err() branch of
// ollamaGenerateHandler.ServeHTTP (non-streaming generate path).
func TestOpenAI_OllamaGenerate_Bad_ModelError(t *testing.T) {
	model := &openAIMockModel{tokens: []inference.Token{{Text: "partial"}}, err: core.NewError("ollama generate failed")}
	resolver := openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": model})
	handler := NewMux(resolver)
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, ollamacompat.DefaultGeneratePath,
		strings.NewReader(`{"model":"qwen","prompt":"hi"}`)))
	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("status = %d body=%s, want 500", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "ollama generate failed") {
		t.Fatalf("body = %s, want model error", rec.Body.String())
	}
}

// TestOpenAI_OllamaTags_Bad_WrongMethod covers the bad-method branch of
// ollamaTagsHandler.ServeHTTP (a GET-only endpoint reached with POST).
func TestOpenAI_OllamaTags_Bad_WrongMethod(t *testing.T) {
	handler := NewMux(openaicompat.NewStaticResolver(nil))
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, ollamacompat.DefaultTagsPath, nil))
	if rec.Code != http.StatusMethodNotAllowed || rec.Header().Get("Allow") != http.MethodGet {
		t.Fatalf("status/allow = %d/%q, want 405/GET", rec.Code, rec.Header().Get("Allow"))
	}
}

// TestOpenAI_OllamaShow_Good_IncludesQuantisation covers the QuantBits>0
// detail branch of ollamaShowHandler.ServeHTTP — a quantised model must report
// its quantisation level in the show details.
func TestOpenAI_OllamaShow_Good_IncludesQuantisation(t *testing.T) {
	model := &openAIMockModel{arch: "gemma3", quantBits: 4}
	resolver := openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": model})
	handler := NewMux(resolver)
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, ollamacompat.DefaultShowPath, strings.NewReader(`{"model":"qwen"}`)))
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	body := rec.Body.String()
	if !strings.Contains(body, `"architecture":"gemma3"`) || !strings.Contains(body, `"quantization":"q4"`) {
		t.Fatalf("body = %s, want architecture + q4 quantisation", body)
	}
}

// TestOpenAI_OllamaShow_Bad_WrongMethod covers the bad-method branch of
// ollamaShowHandler.ServeHTTP.
func TestOpenAI_OllamaShow_Bad_WrongMethod(t *testing.T) {
	handler := NewMux(openaicompat.NewStaticResolver(nil))
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, ollamacompat.DefaultShowPath, nil))
	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("status = %d, want 405", rec.Code)
	}
}

// TestOpenAIOpenAI_readBodySized_GoodBadUgly exercises readBodySized across its
// branches: a known size (single right-sized allocation), an understated
// Content-Length (the grow-and-keep-reading path that still matches io.ReadAll),
// and an out-of-range hint (<=0 and >cap fall back to io.ReadAll). All three
// must return the body's true bytes regardless of the hint.
func TestOpenAIOpenAI_readBodySized_GoodBadUgly(t *testing.T) {
	const body = "the quick brown fox"
	// Good: exact size hint.
	got, err := readBodySized(strings.NewReader(body), int64(len(body)))
	if err != nil || string(got) != body {
		t.Fatalf("exact hint = %q/%v, want %q", got, err, body)
	}
	// Bad: understated Content-Length — the reader runs past the seeded cap.
	got, err = readBodySized(strings.NewReader(body), 4)
	if err != nil || string(got) != body {
		t.Fatalf("understated hint = %q/%v, want full body", got, err)
	}
	// Ugly: non-positive and oversized hints fall back to io.ReadAll.
	got, err = readBodySized(strings.NewReader(body), 0)
	if err != nil || string(got) != body {
		t.Fatalf("zero hint = %q/%v, want full body", got, err)
	}
	got, err = readBodySized(strings.NewReader(body), maxPresizedBody+1)
	if err != nil || string(got) != body {
		t.Fatalf("oversized hint = %q/%v, want full body", got, err)
	}
}

// TestOpenAIOpenAI_decodeWireJSONSized_Bad covers the error envelopes of the
// shared decode path: a nil body and a malformed JSON body must both return a
// scoped core error rather than partially populating the target.
func TestOpenAIOpenAI_decodeWireJSONSized_Bad(t *testing.T) {
	var into map[string]string
	if err := decodeWireJSONSized(nil, -1, &into, "test.scope"); err == nil {
		t.Fatal("decodeWireJSONSized(nil body) error = nil, want scoped error")
	}
	if err := decodeWireJSONSized(strings.NewReader(`{not json`), -1, &into, "test.scope"); err == nil {
		t.Fatal("decodeWireJSONSized(bad json) error = nil, want decode error")
	}
}

// scheduleErrModel is a SchedulerModel whose Schedule call fails, so the
// forEachCompatToken scheduler branch returns an error — the collect-error
// path (distinct from model.Err()) that the non-streaming handlers guard.
type scheduleErrModel struct {
	openAIMockModel
}

func (m *scheduleErrModel) Schedule(context.Context, inference.ScheduledRequest) (inference.RequestHandle, <-chan inference.ScheduledToken, error) {
	return inference.RequestHandle{}, nil, core.NewError("schedule rejected")
}

// TestOpenAI_NonStreaming_Bad_CollectTokensError drives the three non-streaming
// wire handlers (Anthropic messages, Ollama chat, Ollama generate) with a model
// whose scheduler rejects the request, so collectCompatTokens returns an error
// before any token arrives. Each handler must surface a 500 carrying the
// scheduler's message — the collect-error branch, separate from the post-collect
// model.Err() branch covered above.
func TestOpenAI_NonStreaming_Bad_CollectTokensError(t *testing.T) {
	model := &scheduleErrModel{}
	resolver := openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": model})
	handler := NewMux(resolver)

	cases := []struct {
		name string
		path string
		body string
	}{
		{"anthropic", anthropiccompat.DefaultMessagesPath, `{"model":"qwen","messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`},
		{"ollama chat", ollamacompat.DefaultChatPath, `{"model":"qwen","messages":[{"role":"user","content":"hi"}]}`},
		{"ollama generate", ollamacompat.DefaultGeneratePath, `{"model":"qwen","prompt":"hi"}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rec := httptest.NewRecorder()
			handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, tc.path, strings.NewReader(tc.body)))
			if rec.Code != http.StatusInternalServerError {
				t.Fatalf("status = %d body=%s, want 500", rec.Code, rec.Body.String())
			}
			if !strings.Contains(rec.Body.String(), "schedule rejected") {
				t.Fatalf("body = %s, want scheduler error", rec.Body.String())
			}
		})
	}
}

// TestOpenAI_WireHandlers_Bad_RequestEnvelopes sweeps the shared front-of-handler
// rejection branches reached through NewMux: a malformed JSON body and a model
// that the resolver cannot find. Both must produce the documented status codes
// (400 for body, 404 for unknown model) across the Anthropic + Ollama routes.
func TestOpenAI_WireHandlers_Bad_RequestEnvelopes(t *testing.T) {
	// Empty resolver: every model lookup misses with 404.
	handler := NewMux(openaicompat.NewStaticResolver(nil))

	badJSON := []struct {
		name string
		path string
	}{
		{"anthropic", anthropiccompat.DefaultMessagesPath},
		{"ollama chat", ollamacompat.DefaultChatPath},
		{"ollama show", ollamacompat.DefaultShowPath},
	}
	for _, tc := range badJSON {
		t.Run("badjson/"+tc.name, func(t *testing.T) {
			rec := httptest.NewRecorder()
			handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, tc.path, strings.NewReader(`{bad`)))
			if rec.Code != http.StatusBadRequest {
				t.Fatalf("status = %d body=%s, want 400", rec.Code, rec.Body.String())
			}
		})
	}

	missingModel := []struct {
		name string
		path string
		body string
	}{
		{"anthropic", anthropiccompat.DefaultMessagesPath, `{"model":"ghost","messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`},
		{"ollama chat", ollamacompat.DefaultChatPath, `{"model":"ghost","messages":[{"role":"user","content":"hi"}]}`},
		{"ollama generate", ollamacompat.DefaultGeneratePath, `{"model":"ghost","prompt":"hi"}`},
		{"ollama show", ollamacompat.DefaultShowPath, `{"model":"ghost"}`},
	}
	for _, tc := range missingModel {
		t.Run("missing/"+tc.name, func(t *testing.T) {
			rec := httptest.NewRecorder()
			handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, tc.path, strings.NewReader(tc.body)))
			if rec.Code != http.StatusNotFound {
				t.Fatalf("status = %d body=%s, want 404", rec.Code, rec.Body.String())
			}
		})
	}
}

// TestOpenAI_AnthropicMessages_Bad_NilRequest covers the r == nil guard of
// anthropicMessagesHandler.ServeHTTP (unreachable through net/http, so the
// handler is constructed directly — mirroring the existing responses test).
func TestOpenAI_AnthropicMessages_Bad_NilRequest(t *testing.T) {
	rec := httptest.NewRecorder()
	newAnthropicMessagesHandler(openaicompat.NewStaticResolver(nil)).ServeHTTP(rec, nil)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400 for nil request", rec.Code)
	}
}

// TestOpenAI_OllamaGenerate_Bad_WrongMethod covers the bad-method branch of
// ollamaGenerateHandler.ServeHTTP.
func TestOpenAI_OllamaGenerate_Bad_WrongMethod(t *testing.T) {
	handler := NewMux(openaicompat.NewStaticResolver(nil))
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, ollamacompat.DefaultGeneratePath, nil))
	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("status = %d, want 405", rec.Code)
	}
}

// TestOpenAI_OllamaTags_Good_ListsResolverModels covers the tag-assembly loop of
// ollamaTagsHandler.ServeHTTP: a resolver that lists model names must echo each
// one as a tag entry.
func TestOpenAI_OllamaTags_Good_ListsResolverModels(t *testing.T) {
	handler := NewMux(openAINameResolver{})
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, ollamacompat.DefaultTagsPath, nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), `"name":"listed"`) {
		t.Fatalf("body = %s, want listed model tag", rec.Body.String())
	}
}

// TestOpenAIAdmin_CacheEntries_Bad_MissingModel covers the resolveCompatModel
// miss inside adminCacheEntriesHandler.ServeHTTP — an unknown model yields 404.
func TestOpenAIAdmin_CacheEntries_Bad_MissingModel(t *testing.T) {
	handler := NewMuxWithAdmin(openaicompat.NewStaticResolver(nil), AdminConfig{})
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, DefaultAdminCacheEntriesPath+"?model=ghost", nil))
	if rec.Code != http.StatusNotFound {
		t.Fatalf("status = %d body=%s, want 404", rec.Code, rec.Body.String())
	}
}

// cacheEntriesErrModel lists entries with an error, hitting the entry-listing
// error branch of the cache-entries handler.
type cacheEntriesErrModel struct {
	openAIMockModel
}

func (m *cacheEntriesErrModel) CacheEntries(context.Context, map[string]string) ([]inference.CacheBlockRef, error) {
	return nil, core.NewError("entries unavailable")
}

// TestOpenAIAdmin_CacheEntries_Bad_ListError covers the branch where the model
// is a CacheEntryLister but listing fails — the handler surfaces a 500.
func TestOpenAIAdmin_CacheEntries_Bad_ListError(t *testing.T) {
	model := &cacheEntriesErrModel{}
	resolver := openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": model})
	handler := NewMuxWithAdmin(resolver, AdminConfig{})
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, DefaultAdminCacheEntriesPath+"?model=qwen", nil))
	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("status = %d body=%s, want 500", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "entries unavailable") {
		t.Fatalf("body = %s, want list error", rec.Body.String())
	}
}

// TestOpenAIAdmin_mountAdminHandlers_Bad_NilMux covers the nil-mux guard of
// mountAdminHandlers — a nil ServeMux must be a no-op rather than a panic.
func TestOpenAIAdmin_mountAdminHandlers_Bad_NilMux(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("mountAdminHandlers(nil mux) panicked: %v", r)
		}
	}()
	mountAdminHandlers(nil, openaicompat.NewStaticResolver(nil), AdminConfig{})
}

// TestOpenAI_Responses_Bad_ValidationEnvelopes sweeps the per-field rejection
// branches of openAIResponsesHandler.ServeHTTP that the existing bad-request
// test does not reach: a request with valid input but no model (400 model),
// an out-of-range temperature (400 from ResponseGenerateOptions), and an empty
// stop sequence (400 from NormalizeStopSequences).
func TestOpenAI_Responses_Bad_ValidationEnvelopes(t *testing.T) {
	resolver := openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": &openAIMockModel{}})
	handler := NewMux(resolver)
	cases := []struct {
		name string
		body string
	}{
		{"missing model", `{"input":[{"role":"user","content":"hi"}]}`},
		{"bad temperature", `{"model":"qwen","input":[{"role":"user","content":"hi"}],"temperature":5}`},
		{"empty stop", `{"model":"qwen","input":[{"role":"user","content":"hi"}],"stop":[""]}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rec := httptest.NewRecorder()
			handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, openaicompat.DefaultResponsesPath, strings.NewReader(tc.body)))
			if rec.Code != http.StatusBadRequest {
				t.Fatalf("status = %d body=%s, want 400", rec.Code, rec.Body.String())
			}
		})
	}
}

// TestOpenAI_Responses_Bad_CollectTokensError covers the collect-error branch of
// serveOpenAIResponse (the non-streaming /v1/responses path) using a model whose
// scheduler rejects the request.
func TestOpenAI_Responses_Bad_CollectTokensError(t *testing.T) {
	resolver := openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": &scheduleErrModel{}})
	handler := NewMux(resolver)
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, openaicompat.DefaultResponsesPath,
		strings.NewReader(`{"model":"qwen","input":[{"role":"user","content":"hi"}]}`)))
	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("status = %d body=%s, want 500", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "schedule rejected") {
		t.Fatalf("body = %s, want scheduler error", rec.Body.String())
	}
}

// TestOpenAI_AnthropicMessages_Bad_MissingModel covers the req.Model == ""
// branch of anthropicMessagesHandler.ServeHTTP (a decoded request whose model
// field is empty), reached through NewMux.
func TestOpenAI_AnthropicMessages_Bad_MissingModel(t *testing.T) {
	handler := NewMux(openaicompat.NewStaticResolver(nil))
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, anthropiccompat.DefaultMessagesPath,
		strings.NewReader(`{"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`)))
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d body=%s, want 400 for missing model", rec.Code, rec.Body.String())
	}
}

// TestOpenAIOpenAI_firstStopSequenceCut_GoodBadUgly covers the scanner's guard
// branches directly: empty content (no scan), an all-empty stop list (skipped),
// a no-match list (best stays -1), and the earliest-of-several hit.
func TestOpenAIOpenAI_firstStopSequenceCut_GoodBadUgly(t *testing.T) {
	if _, ok := firstStopSequenceCut("", []string{"x"}); ok {
		t.Fatal("firstStopSequenceCut(empty content) = ok, want false")
	}
	if _, ok := firstStopSequenceCut("alpha", []string{""}); ok {
		t.Fatal("firstStopSequenceCut(empty stop) = ok, want false (empty stops skipped)")
	}
	if _, ok := firstStopSequenceCut("alpha beta", []string{"zzz", "qqq"}); ok {
		t.Fatal("firstStopSequenceCut(no match) = ok, want false")
	}
	if cut, ok := firstStopSequenceCut("alpha END beta STOP", []string{"STOP", " END"}); !ok || cut != len("alpha") {
		t.Fatalf("firstStopSequenceCut = %d/%v, want earliest stop after alpha", cut, ok)
	}
}

// errReader is an io.Reader that fails mid-read so readBodySized takes its
// non-EOF error branch.
type errReader struct{}

func (errReader) Read([]byte) (int, error) { return 0, core.NewError("read fault") }

// TestOpenAIOpenAI_readBodySized_Bad_ReadError covers the non-EOF error branch
// of readBodySized: a reader that faults must return the error rather than
// silently treating it as end-of-stream. A positive in-range size hint keeps
// the seeded-buffer loop (not the io.ReadAll fallback) on the stack.
func TestOpenAIOpenAI_readBodySized_Bad_ReadError(t *testing.T) {
	if _, err := readBodySized(errReader{}, 16); err == nil {
		t.Fatal("readBodySized(faulting reader) error = nil, want read fault")
	}
}

// TestOpenAIOpenAI_decodeWireJSONSized_Bad_ReadError covers the read-error
// branch of decodeWireJSONSized (distinct from the nil-body and bad-JSON
// branches): a faulting body reader must surface a scoped read error.
func TestOpenAIOpenAI_decodeWireJSONSized_Bad_ReadError(t *testing.T) {
	var into map[string]string
	if err := decodeWireJSONSized(errReader{}, 16, &into, "test.scope"); err == nil {
		t.Fatal("decodeWireJSONSized(faulting reader) error = nil, want read error")
	}
}

// schedulerStreamModel is a SchedulerModel that emits a configurable number of
// scheduled tokens, so the forEachCompatToken scheduler loop can be driven to
// its early-cancel branch (yield returns false → CancelRequest is called).
type schedulerStreamModel struct {
	openAIMockModel
	emit int
}

func (m *schedulerStreamModel) Schedule(_ context.Context, req inference.ScheduledRequest) (inference.RequestHandle, <-chan inference.ScheduledToken, error) {
	ch := make(chan inference.ScheduledToken, m.emit)
	for i := 0; i < m.emit; i++ {
		ch <- inference.ScheduledToken{RequestID: req.ID, Token: inference.Token{Text: "tok"}}
	}
	close(ch)
	return inference.RequestHandle{ID: req.ID}, ch, nil
}

// TestOpenAIOpenAI_forEachCompatToken_Cancel covers the scheduler early-stop
// branch of forEachCompatToken: when the yield returns false mid-stream the
// helper must cancel the in-flight request (the model is a CancellableModel)
// and stop without error. openAIMockModel.CancelRequest records the cancelled
// handle ID so we can prove the cancel fired.
func TestOpenAIOpenAI_forEachCompatToken_Cancel(t *testing.T) {
	model := &schedulerStreamModel{emit: 3}
	var seen int
	err := forEachCompatToken(context.Background(), model, "req_cancel", "qwen", "", nil, nil, func(inference.Token) bool {
		seen++
		return false // stop on the first token
	})
	if err != nil {
		t.Fatalf("forEachCompatToken(early stop) error = %v, want nil", err)
	}
	if seen != 1 {
		t.Fatalf("yield called %d times, want 1 (stopped on first)", seen)
	}
	if model.cancelled != "req_cancel" {
		t.Fatalf("cancelled handle = %q, want req_cancel", model.cancelled)
	}
}

// TestOpenAI_Responses_Bad_StreamCollectError covers the streaming error branch
// of serveOpenAIResponseStream: when token generation fails the stream must emit
// a response.error event followed by the [DONE] terminator (not a
// response.completed). Driven via a scheduler-error model on the streaming
// /v1/responses path.
func TestOpenAI_Responses_Bad_StreamCollectError(t *testing.T) {
	resolver := openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": &scheduleErrModel{}})
	handler := NewMux(resolver)
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, openaicompat.DefaultResponsesPath,
		strings.NewReader(`{"model":"qwen","stream":true,"input":[{"role":"user","content":"hi"}]}`)))
	// The stream opens 200 OK then reports the failure as an SSE error event.
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	body := rec.Body.String()
	if !strings.Contains(body, "response.error") || !strings.Contains(body, "schedule rejected") {
		t.Fatalf("body = %s, want response.error with scheduler message", body)
	}
	if !strings.Contains(body, "data: [DONE]") {
		t.Fatalf("body = %s, want [DONE] terminator after error", body)
	}
	if strings.Contains(body, "response.completed") {
		t.Fatalf("body = %s, must not complete after an error", body)
	}
}

// TestOpenAI_Responses_Good_StreamsReasoningChannel covers the reasoning-aware
// branches of serveOpenAIResponseStream: a gpt-oss-style model that opens a
// thinking channel produces empty content deltas for the analysis tokens (the
// processor.Process == "" skip), so only the visible answer streams as a delta,
// and the captured reasoning surfaces as the completed response's thought (the
// processor.Reasoning() fallback). This is the streaming counterpart of the
// non-streaming UsesModelParserRegistry test.
func TestOpenAI_Responses_Good_StreamsReasoningChannel(t *testing.T) {
	model := &openAIMockModel{
		arch: "gpt_oss",
		tokens: []inference.Token{
			{Text: "<|channel>analysis\n"},
			{Text: "thinking"},
			{Text: "<|channel>final\n"},
			{Text: "Answer"},
		},
	}
	resolver := openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": model})
	handler := NewMux(resolver)
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, openaicompat.DefaultResponsesPath,
		strings.NewReader(`{"model":"qwen","stream":true,"input":[{"role":"user","content":"hi"}]}`)))
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	body := rec.Body.String()
	// Only the final-channel answer is emitted as a visible delta; the
	// analysis tokens are suppressed (empty deltas, skipped).
	if !strings.Contains(body, `"delta":"Answer"`) {
		t.Fatalf("body = %s, want Answer delta", body)
	}
	if strings.Contains(body, `"delta":"thinking"`) {
		t.Fatalf("body = %s, analysis token leaked as a visible delta", body)
	}
	// The captured reasoning surfaces on the completed event's thought.
	if !strings.Contains(body, `"thought":"thinking"`) {
		t.Fatalf("body = %s, want captured thought on completion", body)
	}
}

// TestOpenAI_OllamaChat_Good_SuppressesReasoningDeltas covers the
// processor.Process == "" skip in serveOllamaStream: streaming a model that
// opens a thinking channel must not emit NDJSON lines for the suppressed
// analysis tokens, only for the visible answer.
func TestOpenAI_OllamaChat_Good_SuppressesReasoningDeltas(t *testing.T) {
	model := &openAIMockModel{
		arch: "gpt_oss",
		tokens: []inference.Token{
			{Text: "<|channel>analysis\n"},
			{Text: "thinking"},
			{Text: "<|channel>final\n"},
			{Text: "Answer"},
		},
	}
	resolver := openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": model})
	handler := NewMux(resolver)
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, ollamacompat.DefaultChatPath,
		strings.NewReader(`{"model":"qwen","stream":true,"messages":[{"role":"user","content":"hi"}]}`)))
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	body := rec.Body.String()
	if !strings.Contains(body, `"content":"Answer"`) {
		t.Fatalf("body = %s, want Answer content line", body)
	}
	if strings.Contains(body, `"content":"thinking"`) {
		t.Fatalf("body = %s, analysis token leaked as a chat delta", body)
	}
	if !strings.Contains(body, `"done":true`) {
		t.Fatalf("body = %s, want final done line", body)
	}
}

// TestOpenAI_Responses_Good_StreamsFlushTail covers the processor.Flush() !=
// "" branch in serveOpenAIResponseStream (the post-loop tail emission). A
// single token "Visible <thi" streams "Visible " as a normal delta while the
// processor holds the "<thi" partial-marker prefix as pending; nothing closes
// it, so the flush after the generation loop drains "<thi" as a final visible
// delta. The standard streaming tests never leave a pending tail, so this is
// the only path that exercises the flush emission.
func TestOpenAI_Responses_Good_StreamsFlushTail(t *testing.T) {
	model := &openAIMockModel{
		tokens:  []inference.Token{{Text: "Visible <thi"}},
		metrics: inference.GenerateMetrics{PromptTokens: 1, GeneratedTokens: 1},
	}
	resolver := openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": model})
	handler := NewMux(resolver)

	req := httptest.NewRequest(http.MethodPost, openaicompat.DefaultResponsesPath, strings.NewReader(`{"model":"qwen","stream":true,"input":[{"role":"user","content":"hi"}]}`))
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	body := rec.Body.String()
	// "Visible " streams during the loop; "<thi" arrives only via the flush.
	// The JSON encoder HTML-escapes the literal "<", so the tail lands on the
	// wire as the escaped sequence whose ASCII suffix is "3cthi".
	for _, want := range []string{`"delta":"Visible "`, `3cthi`, "response.completed"} {
		if !strings.Contains(body, want) {
			t.Fatalf("body = %s, want flushed tail %s", body, want)
		}
	}
}

// TestOpenAI_AnthropicMessages_Good_StreamsFlushTail covers the
// processor.Flush() != "" branch in serveAnthropicMessageStream. As with the
// responses handler, "Visible <thi" streams "Visible " then leaves "<thi"
// pending; the post-loop flush writes it as a trailing text delta. No stop
// sequences here so the handler takes the fast path and the flush is the only
// source of the tail.
func TestOpenAI_AnthropicMessages_Good_StreamsFlushTail(t *testing.T) {
	model := &openAIMockModel{
		tokens:  []inference.Token{{Text: "Visible <thi"}},
		metrics: inference.GenerateMetrics{PromptTokens: 1, GeneratedTokens: 1},
	}
	resolver := openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": model})
	handler := NewMux(resolver)

	req := httptest.NewRequest(http.MethodPost, anthropiccompat.DefaultMessagesPath, strings.NewReader(`{"model":"qwen","stream":true,"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`))
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	body := rec.Body.String()
	// The Anthropic SSE builder emits text deltas verbatim (no HTML escaping,
	// unlike the OpenAI/Ollama JSON-marshal path), so the flushed tail appears
	// as the literal "<thi".
	for _, want := range []string{`"text":"Visible "`, `"text":"<thi"`, "event: message_stop"} {
		if !strings.Contains(body, want) {
			t.Fatalf("body = %s, want flushed tail %s", body, want)
		}
	}
}

// TestOpenAI_AnthropicMessages_Good_StopSequenceMidDelta covers the
// stop-cut else branch in serveAnthropicMessageStream (delta =
// candidate[prevLen:stopCut] when the cut lands past the already-emitted
// length). A single streamed token "Answer STOP hidden" with stop "STOP"
// produces prevLen == 0 on the first (only) delta, the cut at index 7 is
// greater than prevLen, so the emitted delta is exactly "Answer " and the
// stop reason flips. The AcrossTokens test only reaches the prevLen-clamped
// (delta == "") branch; the non-streaming AppliesStopSequences test never
// enters this streaming handler at all.
func TestOpenAI_AnthropicMessages_Good_StopSequenceMidDelta(t *testing.T) {
	model := &openAIMockModel{
		tokens:  []inference.Token{{Text: "Answer STOP hidden"}},
		metrics: inference.GenerateMetrics{PromptTokens: 1, GeneratedTokens: 1},
	}
	resolver := openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": model})
	handler := NewMux(resolver)

	req := httptest.NewRequest(http.MethodPost, anthropiccompat.DefaultMessagesPath, strings.NewReader(`{"model":"qwen","stream":true,"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}],"stop_sequences":["STOP"]}`))
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	body := rec.Body.String()
	if !strings.Contains(body, `"text":"Answer "`) {
		t.Fatalf("body = %s, want mid-delta cut to \"Answer \"", body)
	}
	if !strings.Contains(body, `"stop_reason":"stop_sequence"`) {
		t.Fatalf("body = %s, want stop_sequence reason", body)
	}
	if strings.Contains(body, "hidden") {
		t.Fatalf("body = %s, content past the stop cut leaked", body)
	}
}

// TestOpenAI_OllamaStream_Good_StreamsFlushTail covers the processor.Flush()
// != "" branch in serveOllamaStream for both wire shapes — the chat branch
// (ollamacompat.ChatResponse) and the generate branch
// (ollamacompat.GenerateResponse). "Visible <thi" streams "Visible " then the
// post-loop flush drains "<thi" as a final line on whichever shape the route
// selected. Driving both /api/chat and /api/generate in one table reaches the
// chat==true and chat==false sub-branches of the same flush block.
func TestOpenAI_OllamaStream_Good_StreamsFlushTail(t *testing.T) {
	cases := []struct {
		name string
		path string
		body string
		want string
	}{
		// The flushed "<thi" tail is HTML-escaped on the wire; matching the
		// escape-free ASCII suffix "3cthi" pins the tail on either shape.
		{
			name: "chat",
			path: ollamacompat.DefaultChatPath,
			body: `{"model":"qwen","stream":true,"messages":[{"role":"user","content":"hi"}]}`,
			want: `3cthi`,
		},
		{
			name: "generate",
			path: ollamacompat.DefaultGeneratePath,
			body: `{"model":"qwen","stream":true,"prompt":"hi"}`,
			want: `3cthi`,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			model := &openAIMockModel{
				tokens:  []inference.Token{{Text: "Visible <thi"}},
				metrics: inference.GenerateMetrics{PromptTokens: 1, GeneratedTokens: 1},
			}
			resolver := openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": model})
			handler := NewMux(resolver)

			req := httptest.NewRequest(http.MethodPost, tc.path, strings.NewReader(tc.body))
			rec := httptest.NewRecorder()
			handler.ServeHTTP(rec, req)

			if rec.Code != http.StatusOK {
				t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
			}
			body := rec.Body.String()
			if !strings.Contains(body, tc.want) {
				t.Fatalf("body = %s, want flushed tail %s", body, tc.want)
			}
		})
	}
}

// --- v0.9.0 audit triplet completion: New* constructor Bad/Ugly variants.
// The five constructors are total and lazy — they never return an error and do
// not touch the Metal backend until a request first resolves a model. The Bad
// variant feeds the degenerate/abusive input the constructor still tolerates;
// the Ugly variant drives the resulting handler through a boundary route that
// resolves without loading real weights (unknown path -> 404, wrong method ->
// 405). Behaviours asserted here were captured from the live handlers.

func TestOpenai_NewResolver_Bad(t *testing.T) {
	// Empty model path is not validated at construction — the resolver is built
	// lazily, so NewResolver still returns a usable metal-backed resolver and
	// only fails later when a request tries to load the (missing) weights.
	resolver := NewResolver("")
	if resolver == nil {
		t.Fatal("NewResolver(\"\") returned nil, want lazy resolver")
	}
	if resolver.BackendName != "metal" {
		t.Fatalf("BackendName = %q, want metal", resolver.BackendName)
	}
	if resolver.ModelPath != "" {
		t.Fatalf("ModelPath = %q, want empty (preserved verbatim)", resolver.ModelPath)
	}
}

func TestOpenai_NewResolver_Ugly(t *testing.T) {
	// Boundary: an unusually long path plus load options. The resolver records
	// the path verbatim (no normalisation/truncation) and threads the options
	// through to the lazy backend load.
	longPath := "/models/" + strings.Repeat("q", 4096)
	resolver := NewResolver(longPath, inference.WithContextLen(8192))
	if resolver == nil {
		t.Fatal("NewResolver(longPath) returned nil")
	}
	if resolver.ModelPath != longPath {
		t.Fatalf("ModelPath length = %d, want %d (preserved verbatim)", len(resolver.ModelPath), len(longPath))
	}
	if resolver.BackendName != "metal" {
		t.Fatalf("BackendName = %q, want metal", resolver.BackendName)
	}
}

func TestOpenai_NewHandler_Bad(t *testing.T) {
	// Empty model path: NewHandler wraps the lazy resolver, so the handler is
	// still constructed (non-nil) — the missing weights only surface on the
	// first request, not at construction.
	handler := NewHandler("")
	if handler == nil {
		t.Fatal("NewHandler(\"\") returned nil, want lazy handler")
	}
}

func TestOpenai_NewHandler_Ugly(t *testing.T) {
	// Boundary: load options supplied. NewHandler must still hand back a
	// non-nil http.Handler that satisfies the interface without loading.
	var handler http.Handler = NewHandler("/models/qwen3", inference.WithContextLen(1))
	if handler == nil {
		t.Fatal("NewHandler with options returned nil")
	}
}

func TestOpenai_NewModelMux_Bad(t *testing.T) {
	// Empty model path: the package-first mux is built lazily, so an unknown
	// route is answered by the ServeMux itself (404) without ever resolving a
	// model — proving construction does not depend on real weights.
	handler := NewModelMux("")
	if handler == nil {
		t.Fatal("NewModelMux(\"\") returned nil")
	}
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/no/such/route", nil))
	if rec.Code != http.StatusNotFound {
		t.Fatalf("unknown route status = %d, want 404", rec.Code)
	}
}

func TestOpenai_NewModelMux_Ugly(t *testing.T) {
	// Boundary: a mounted route hit with the wrong method. The method guard
	// fires before any model resolution, so NewModelMux yields 405 without
	// touching the Metal backend.
	handler := NewModelMux("/models/qwen3")
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, openaicompat.DefaultChatCompletionsPath, nil))
	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("GET on chat-completions status = %d, want 405", rec.Code)
	}
}

func TestOpenai_NewMux_Bad(t *testing.T) {
	// An unknown path on a fully-wired mux is a ServeMux miss (404) — the mux
	// only mounts the known compatibility routes, nothing catch-all.
	resolver := openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": &openAIMockModel{}})
	handler := NewMux(resolver)
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/v1/unknown", nil))
	if rec.Code != http.StatusNotFound {
		t.Fatalf("unknown route status = %d, want 404", rec.Code)
	}
}

func TestOpenai_NewMux_Ugly(t *testing.T) {
	// Boundary: a POST-only route reached with GET. NewMux returns the shared
	// "method not allowed" error envelope (405) before resolving the model.
	resolver := openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": &openAIMockModel{}})
	handler := NewMux(resolver)
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, openaicompat.DefaultChatCompletionsPath, nil))
	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("GET on chat-completions status = %d, want 405", rec.Code)
	}
	if !strings.Contains(rec.Body.String(), "method not allowed") {
		t.Fatalf("body = %s, want method-not-allowed envelope", rec.Body.String())
	}
}

func TestOpenai_NewMuxWithAdmin_Bad(t *testing.T) {
	// A wake callback that errors must propagate as a 500 with the action named
	// in the error envelope — NewMuxWithAdmin wires the host callback straight
	// onto the admin route.
	resolver := openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": &openAIMockModel{}})
	handler := NewMuxWithAdmin(resolver, AdminConfig{
		Wake: func(context.Context) error { return context.Canceled },
	})
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultAdminWakePath, nil))
	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("wake-callback-error status = %d, want 500", rec.Code)
	}
	if !strings.Contains(rec.Body.String(), `"param":"wake"`) {
		t.Fatalf("body = %s, want wake-scoped error", rec.Body.String())
	}
}

func TestOpenai_NewMuxWithAdmin_Ugly(t *testing.T) {
	// Boundary: a zero-value AdminConfig (no host callbacks). NewMuxWithAdmin
	// still mounts the admin routes, and the health endpoint fills in the
	// default "ok" / "go-mlx" payload itself.
	resolver := openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": &openAIMockModel{}})
	handler := NewMuxWithAdmin(resolver, AdminConfig{})
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, DefaultHealthPath, nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("zero-config health status = %d, want 200", rec.Code)
	}
	if !strings.Contains(rec.Body.String(), `"status":"ok"`) {
		t.Fatalf("body = %s, want default ok status", rec.Body.String())
	}
}
