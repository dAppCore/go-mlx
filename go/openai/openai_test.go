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

func TestOpenAI_NewResolver_Good_UsesMetalBackend(t *testing.T) {
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

func TestOpenAI_NewHandler_Good_ReturnsHTTPHandler(t *testing.T) {
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
	return inference.ModelInfo{Architecture: arch}
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

func TestOpenAI_NewMux_Good_MountsChatResponsesAndServices(t *testing.T) {
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

func TestOpenAI_NewMux_Good_MountsAnthropicAndOllama(t *testing.T) {
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

func TestOpenAI_NewMuxWithAdmin_Good_MountsAdminHandlers(t *testing.T) {
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

func TestOpenAI_NewModelMux_Good_UsesMetalResolver(t *testing.T) {
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
