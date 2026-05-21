// SPDX-Licence-Identifier: EUPL-1.2

// Package openai mounts OpenAI / Anthropic / Ollama compatibility handlers
// over a local inference backend (Metal by default).
//
//	handler := openai.NewHandler("/path/to/model", inference.WithContextLen(8192))
//	http.ListenAndServe(":8080", handler)
package openai

import (
	"context"
	"io"
	"net/http"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
	anthropiccompat "dappco.re/go/inference/anthropic"
	ollamacompat "dappco.re/go/inference/ollama"
	openaicompat "dappco.re/go/inference/openai"
	"dappco.re/go/inference/parser"
)

// NewResolver returns a resolver that lazily loads modelPath through the
// native Metal backend registered by go-mlx.
//
//	resolver := openai.NewResolver(modelPath)
func NewResolver(modelPath string, opts ...inference.LoadOption) *openaicompat.BackendResolver {
	return openaicompat.NewBackendResolver("metal", modelPath, opts...)
}

// NewHandler exposes modelPath through the shared OpenAI-compatible chat
// completions handler.
//
//	handler := openai.NewHandler(modelPath)
func NewHandler(modelPath string, opts ...inference.LoadOption) http.Handler {
	return openaicompat.NewHandler(NewResolver(modelPath, opts...))
}

// NewModelMux exposes a local MLX model through the package-first
// OpenAI-compatible route set. It lazily loads modelPath through the registered
// native Metal inference backend.
//
//	handler := openai.NewModelMux(modelPath)
func NewModelMux(modelPath string, opts ...inference.LoadOption) http.Handler {
	return NewMux(NewResolver(modelPath, opts...))
}

// NewMux mounts the shared local-inference endpoints over resolver. The
// handler is deliberately package-first: callers can host it from core/api,
// go-ai, a standalone server, or tests without making go-mlx depend on any of
// those layers.
//
//	handler := openai.NewMux(resolver)
func NewMux(resolver openaicompat.Resolver) http.Handler {
	return NewMuxWithAdmin(resolver, AdminConfig{})
}

// NewMuxWithAdmin mounts the same compatibility routes as NewMux plus
// package-first admin callbacks supplied by the host application.
//
//	handler := openai.NewMuxWithAdmin(resolver, openai.AdminConfig{Health: hostHealth})
func NewMuxWithAdmin(resolver openaicompat.Resolver, admin AdminConfig) http.Handler {
	mux := http.NewServeMux()
	mux.Handle(openaicompat.DefaultChatCompletionsPath, openaicompat.NewHandler(resolver))
	mux.Handle(openaicompat.DefaultResponsesPath, newOpenAIResponsesHandler(resolver))
	mux.Handle(openaicompat.DefaultEmbeddingsPath, openaicompat.NewEmbeddingsHandler(resolver))
	mux.Handle(openaicompat.DefaultRerankPath, openaicompat.NewRerankHandler(resolver))
	mux.Handle(openaicompat.DefaultCapabilitiesPath, openaicompat.NewCapabilityHandler(resolver))
	mux.Handle(openaicompat.DefaultCacheStatsPath, openaicompat.NewCacheStatsHandler(resolver))
	mux.Handle(openaicompat.DefaultCacheWarmPath, openaicompat.NewCacheWarmHandler(resolver))
	mux.Handle(openaicompat.DefaultCacheClearPath, openaicompat.NewCacheClearHandler(resolver))
	mux.Handle(openaicompat.DefaultCancelPath, openaicompat.NewCancelHandler(resolver))
	mux.Handle(anthropiccompat.DefaultMessagesPath, newAnthropicMessagesHandler(resolver))
	mux.Handle(ollamacompat.DefaultChatPath, newOllamaChatHandler(resolver))
	mux.Handle(ollamacompat.DefaultGeneratePath, newOllamaGenerateHandler(resolver))
	mux.Handle(ollamacompat.DefaultTagsPath, newOllamaTagsHandler(resolver))
	mux.Handle(ollamacompat.DefaultShowPath, newOllamaShowHandler(resolver))
	mountAdminHandlers(mux, resolver, admin)
	return mux
}

type openAIResponsesHandler struct {
	resolver openaicompat.Resolver
}

func newOpenAIResponsesHandler(resolver openaicompat.Resolver) http.Handler {
	return &openAIResponsesHandler{resolver: resolver}
}

func (h *openAIResponsesHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if h == nil || h.resolver == nil {
		writeOpenAIError(w, http.StatusServiceUnavailable, "responses handler is not configured", "model")
		return
	}
	if r == nil {
		writeOpenAIError(w, http.StatusBadRequest, "request is nil", "request")
		return
	}
	if r.Method != http.MethodPost {
		w.Header().Set("Allow", http.MethodPost)
		writeOpenAIError(w, http.StatusMethodNotAllowed, "method not allowed", "method")
		return
	}
	req, err := decodeOpenAIResponseRequest(r.Body)
	if err != nil {
		writeOpenAIError(w, http.StatusBadRequest, err.Error(), "body")
		return
	}
	if core.Trim(req.Model) == "" {
		writeOpenAIError(w, http.StatusBadRequest, "model is required", "model")
		return
	}
	opts, err := openaicompat.ResponseGenerateOptions(req)
	if err != nil {
		writeOpenAIError(w, http.StatusBadRequest, err.Error(), "request")
		return
	}
	stops, err := openaicompat.NormalizeStopSequences(req.Stop)
	if err != nil {
		writeOpenAIError(w, http.StatusBadRequest, err.Error(), "stop")
		return
	}
	model, err := h.resolver.ResolveModel(r.Context(), req.Model)
	if err != nil {
		writeOpenAIError(w, http.StatusNotFound, err.Error(), "model")
		return
	}
	messages := openaicompat.ResponseMessages(req)
	if req.Stream {
		serveOpenAIResponseStream(w, r.Context(), model, req, messages, stops, opts...)
		return
	}
	serveOpenAIResponse(w, r.Context(), model, req, messages, stops, opts...)
}

func decodeOpenAIResponseRequest(body io.Reader) (openaicompat.ResponseRequest, error) {
	var req openaicompat.ResponseRequest
	if err := decodeWireJSON(body, &req, "mlx.openai.responses"); err != nil {
		return openaicompat.ResponseRequest{}, err
	}
	return req, nil
}

func serveOpenAIResponse(w http.ResponseWriter, ctx context.Context, model inference.TextModel, req openaicompat.ResponseRequest, messages []inference.Message, stops []string, opts ...inference.GenerateOption) {
	id := openAIResponseID()
	tokens, err := collectOpenAIResponseTokens(ctx, model, id, req.Model, messages, opts...)
	if err != nil {
		writeOpenAIError(w, http.StatusInternalServerError, err.Error(), "model")
		return
	}
	if err := model.Err(); err != nil {
		writeOpenAIError(w, http.StatusInternalServerError, err.Error(), "model")
		return
	}
	visible, thought := parseOpenAIModelOutput(model, tokens, openAITokensText(tokens))
	response := openaicompat.NewTextResponse(id, req.Model, openaicompat.TruncateAtStopSequence(visible, stops), model.Metrics())
	if thought != "" {
		response.Thought = &thought
	}
	writeOpenAIJSON(w, http.StatusOK, response)
}

func serveOpenAIResponseStream(w http.ResponseWriter, ctx context.Context, model inference.TextModel, req openaicompat.ResponseRequest, messages []inference.Message, stops []string, opts ...inference.GenerateOption) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)
	flusher, _ := w.(http.Flusher)
	writeEvent := func(event openaicompat.ResponseStreamEvent) {
		_, _ = w.Write([]byte(core.Concat("data: ", core.JSONMarshalString(event), "\n\n")))
		if flusher != nil {
			flusher.Flush()
		}
	}

	id := openAIResponseID()
	writeEvent(openaicompat.ResponseStreamEvent{
		Type: "response.created",
		Response: &openaicompat.Response{
			ID:      id,
			Object:  "response",
			Created: time.Now().Unix(),
			Model:   req.Model,
		},
	})

	processor := parser.NewProcessor(parser.Config{Mode: parser.Capture}, parser.HintFromInference(model.Info()))
	tokens := []inference.Token{}
	raw := core.NewBuilder()
	visibleBuilder := core.NewBuilder()
	err := forEachOpenAIResponseToken(ctx, model, id, req.Model, messages, opts, func(token inference.Token) bool {
		tokens = append(tokens, token)
		raw.WriteString(token.Text)
		contentDelta := processor.Process(token.Text)
		if contentDelta == "" {
			return true
		}
		visibleBuilder.WriteString(contentDelta)
		event := openaicompat.ResponseStreamEvent{Type: "response.output_text.delta", Delta: contentDelta}
		writeEvent(event)
		return true
	})
	if contentTail := processor.Flush(); contentTail != "" {
		visibleBuilder.WriteString(contentTail)
		event := openaicompat.ResponseStreamEvent{Type: "response.output_text.delta", Delta: contentTail}
		writeEvent(event)
	}

	if err != nil {
		writeEvent(openaicompat.ResponseStreamEvent{Type: "response.error", Delta: err.Error()})
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
		if flusher != nil {
			flusher.Flush()
		}
		return
	}
	visible, thought := parseOpenAIModelOutput(model, tokens, raw.String())
	if visible == "" && visibleBuilder.String() != "" {
		visible = visibleBuilder.String()
	}
	response := openaicompat.NewTextResponse(id, req.Model, openaicompat.TruncateAtStopSequence(visible, stops), model.Metrics())
	if thought == "" {
		thought = processor.Reasoning()
	}
	if thought != "" {
		response.Thought = &thought
	}
	writeEvent(openaicompat.ResponseStreamEvent{Type: "response.completed", Response: &response})
	_, _ = w.Write([]byte("data: [DONE]\n\n"))
	if flusher != nil {
		flusher.Flush()
	}
}

func writeOpenAIJSON(w http.ResponseWriter, status int, payload any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_, _ = w.Write([]byte(core.JSONMarshalString(payload)))
}

func writeOpenAIError(w http.ResponseWriter, status int, message, param string) {
	writeOpenAIJSON(w, status, openaicompat.ErrorResponse{Error: openaicompat.ErrorObject{
		Message: message,
		Type:    "invalid_request_error",
		Param:   param,
		Code:    "invalid_request_error",
	}})
}

func openAIResponseID() string {
	return "resp_" + core.FormatInt(time.Now().UnixNano(), 10)
}

func collectOpenAIResponseTokens(ctx context.Context, model inference.TextModel, requestID, modelName string, messages []inference.Message, opts ...inference.GenerateOption) ([]inference.Token, error) {
	return collectCompatTokens(ctx, model, requestID, modelName, "", messages, opts...)
}

func collectCompatTokens(ctx context.Context, model inference.TextModel, requestID, modelName, prompt string, messages []inference.Message, opts ...inference.GenerateOption) ([]inference.Token, error) {
	tokens := []inference.Token{}
	err := forEachCompatToken(ctx, model, requestID, modelName, prompt, messages, opts, func(token inference.Token) bool {
		tokens = append(tokens, token)
		return true
	})
	return tokens, err
}

func forEachOpenAIResponseToken(ctx context.Context, model inference.TextModel, requestID, modelName string, messages []inference.Message, opts []inference.GenerateOption, yield func(inference.Token) bool) error {
	return forEachCompatToken(ctx, model, requestID, modelName, "", messages, opts, yield)
}

func forEachCompatToken(ctx context.Context, model inference.TextModel, requestID, modelName, prompt string, messages []inference.Message, opts []inference.GenerateOption, yield func(inference.Token) bool) error {
	if scheduler, ok := model.(inference.SchedulerModel); ok {
		handle, stream, err := scheduler.Schedule(ctx, inference.ScheduledRequest{
			ID:       requestID,
			Model:    modelName,
			Prompt:   prompt,
			Messages: append([]inference.Message(nil), messages...),
			Sampler:  inference.SamplerConfigFromGenerateConfig(inference.ApplyGenerateOpts(opts)),
		})
		if err != nil {
			return err
		}
		for scheduled := range stream {
			if !yield(scheduled.Token) {
				if cancellable, ok := model.(inference.CancellableModel); ok {
					_, _ = cancellable.CancelRequest(ctx, handle.ID)
				}
				return nil
			}
		}
		return nil
	}
	var stream func(func(inference.Token) bool)
	if len(messages) > 0 {
		stream = model.Chat(ctx, messages, opts...)
	} else {
		stream = model.Generate(ctx, prompt, opts...)
	}
	for token := range stream {
		if !yield(token) {
			return nil
		}
	}
	return nil
}

type anthropicMessagesHandler struct {
	resolver openaicompat.Resolver
}

func newAnthropicMessagesHandler(resolver openaicompat.Resolver) http.Handler {
	return &anthropicMessagesHandler{resolver: resolver}
}

func (h *anthropicMessagesHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if h == nil || h.resolver == nil {
		writeOpenAIError(w, http.StatusServiceUnavailable, "anthropic messages handler is not configured", "model")
		return
	}
	if r == nil {
		writeOpenAIError(w, http.StatusBadRequest, "request is nil", "request")
		return
	}
	if r.Method != http.MethodPost {
		w.Header().Set("Allow", http.MethodPost)
		writeOpenAIError(w, http.StatusMethodNotAllowed, "method not allowed", "method")
		return
	}
	var req anthropiccompat.MessageRequest
	if err := decodeWireJSON(r.Body, &req, "mlx.anthropic.messages"); err != nil {
		writeOpenAIError(w, http.StatusBadRequest, err.Error(), "body")
		return
	}
	if core.Trim(req.Model) == "" {
		writeOpenAIError(w, http.StatusBadRequest, "model is required", "model")
		return
	}
	stops, err := normalizeAnthropicStopSequences(req.StopSequences)
	if err != nil {
		writeOpenAIError(w, http.StatusBadRequest, err.Error(), "stop_sequences")
		return
	}
	model, err := h.resolver.ResolveModel(r.Context(), req.Model)
	if err != nil {
		writeOpenAIError(w, http.StatusNotFound, err.Error(), "model")
		return
	}
	messages := anthropiccompat.InferenceMessages(req)
	opts := anthropiccompat.GenerateOptions(req)
	if req.Stream {
		serveAnthropicMessageStream(w, r.Context(), model, req, messages, stops, opts...)
		return
	}
	tokens, err := collectCompatTokens(r.Context(), model, anthropicMessageID(), req.Model, "", messages, opts...)
	if err != nil {
		writeOpenAIError(w, http.StatusInternalServerError, err.Error(), "model")
		return
	}
	if err := model.Err(); err != nil {
		writeOpenAIError(w, http.StatusInternalServerError, err.Error(), "model")
		return
	}
	visible, _ := parseOpenAIModelOutput(model, tokens, openAITokensText(tokens))
	response := anthropiccompat.NewTextResponse(anthropicMessageID(), req.Model, openaicompat.TruncateAtStopSequence(visible, stops), model.Metrics())
	writeOpenAIJSON(w, http.StatusOK, response)
}

func serveAnthropicMessageStream(w http.ResponseWriter, ctx context.Context, model inference.TextModel, req anthropiccompat.MessageRequest, messages []inference.Message, stops []string, opts ...inference.GenerateOption) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)
	flusher, _ := w.(http.Flusher)
	messageID := anthropicMessageID()
	writeEvent := func(event, payload string) {
		_, _ = w.Write([]byte(core.Concat("event: ", event, "\n", "data: ", payload, "\n\n")))
		if flusher != nil {
			flusher.Flush()
		}
	}
	writeEvent("message_start", core.JSONMarshalString(anthropiccompat.MessageResponse{ID: messageID, Type: "message", Role: "assistant", Model: req.Model}))
	processor := parser.NewProcessor(parser.Config{Mode: parser.Capture}, parser.HintFromInference(model.Info()))
	emitted := ""
	_ = forEachCompatToken(ctx, model, messageID, req.Model, "", messages, opts, func(token inference.Token) bool {
		delta := processor.Process(token.Text)
		candidate := emitted + delta
		stopCut, stopHit := firstStopSequenceCut(candidate, stops)
		if stopHit {
			if stopCut <= len(emitted) {
				delta = ""
			} else {
				delta = candidate[len(emitted):stopCut]
			}
		}
		if delta != "" {
			writeEvent("content_block_delta", core.JSONMarshalString(map[string]any{"type": "content_block_delta", "delta": map[string]string{"type": "text_delta", "text": delta}}))
		}
		if stopHit {
			emitted = candidate[:stopCut]
			return false
		}
		emitted = candidate
		return true
	})
	if tail := processor.Flush(); tail != "" {
		writeEvent("content_block_delta", core.JSONMarshalString(map[string]any{"type": "content_block_delta", "delta": map[string]string{"type": "text_delta", "text": tail}}))
	}
	writeEvent("message_delta", core.JSONMarshalString(map[string]any{"type": "message_delta", "delta": map[string]string{"stop_reason": "end_turn"}}))
	writeEvent("message_stop", core.JSONMarshalString(map[string]string{"type": "message_stop"}))
}

type ollamaChatHandler struct{ resolver openaicompat.Resolver }
type ollamaGenerateHandler struct{ resolver openaicompat.Resolver }
type ollamaTagsHandler struct{ resolver openaicompat.Resolver }
type ollamaShowHandler struct{ resolver openaicompat.Resolver }

func newOllamaChatHandler(resolver openaicompat.Resolver) http.Handler {
	return &ollamaChatHandler{resolver: resolver}
}

func newOllamaGenerateHandler(resolver openaicompat.Resolver) http.Handler {
	return &ollamaGenerateHandler{resolver: resolver}
}

func newOllamaTagsHandler(resolver openaicompat.Resolver) http.Handler {
	return &ollamaTagsHandler{resolver: resolver}
}

func newOllamaShowHandler(resolver openaicompat.Resolver) http.Handler {
	return &ollamaShowHandler{resolver: resolver}
}

func (h *ollamaChatHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if !requireCompatMethod(w, r, http.MethodPost) {
		return
	}
	var req ollamacompat.ChatRequest
	if err := decodeWireJSON(r.Body, &req, "mlx.ollama.chat"); err != nil {
		writeOpenAIError(w, http.StatusBadRequest, err.Error(), "body")
		return
	}
	model, ok := resolveCompatModel(w, r.Context(), h.resolver, req.Model)
	if !ok {
		return
	}
	messages := ollamacompat.InferenceMessages(req.Messages)
	opts := ollamacompat.GenerateOptions(req.Options)
	if req.Stream {
		serveOllamaChatStream(w, r.Context(), model, req, messages, opts...)
		return
	}
	tokens, err := collectCompatTokens(r.Context(), model, ollamaRequestID(), req.Model, "", messages, opts...)
	if err != nil {
		writeOpenAIError(w, http.StatusInternalServerError, err.Error(), "model")
		return
	}
	if err := model.Err(); err != nil {
		writeOpenAIError(w, http.StatusInternalServerError, err.Error(), "model")
		return
	}
	visible, _ := parseOpenAIModelOutput(model, tokens, openAITokensText(tokens))
	writeOpenAIJSON(w, http.StatusOK, ollamacompat.NewChatResponse(req.Model, visible, model.Metrics()))
}

func (h *ollamaGenerateHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if !requireCompatMethod(w, r, http.MethodPost) {
		return
	}
	var req ollamacompat.GenerateRequest
	if err := decodeWireJSON(r.Body, &req, "mlx.ollama.generate"); err != nil {
		writeOpenAIError(w, http.StatusBadRequest, err.Error(), "body")
		return
	}
	model, ok := resolveCompatModel(w, r.Context(), h.resolver, req.Model)
	if !ok {
		return
	}
	opts := ollamacompat.GenerateOptions(req.Options)
	if req.Stream {
		serveOllamaGenerateStream(w, r.Context(), model, req, opts...)
		return
	}
	tokens, err := collectCompatTokens(r.Context(), model, ollamaRequestID(), req.Model, req.Prompt, nil, opts...)
	if err != nil {
		writeOpenAIError(w, http.StatusInternalServerError, err.Error(), "model")
		return
	}
	if err := model.Err(); err != nil {
		writeOpenAIError(w, http.StatusInternalServerError, err.Error(), "model")
		return
	}
	visible, _ := parseOpenAIModelOutput(model, tokens, openAITokensText(tokens))
	writeOpenAIJSON(w, http.StatusOK, ollamacompat.NewGenerateResponse(req.Model, visible, model.Metrics()))
}

func (h *ollamaTagsHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if !requireCompatMethod(w, r, http.MethodGet) {
		return
	}
	tags := []ollamacompat.ModelTag{}
	for _, name := range resolverModelNames(h.resolver) {
		tags = append(tags, ollamacompat.ModelTag{Name: name, Model: name})
	}
	writeOpenAIJSON(w, http.StatusOK, ollamacompat.TagsResponse{Models: tags})
}

func (h *ollamaShowHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if !requireCompatMethod(w, r, http.MethodPost) {
		return
	}
	var req ollamacompat.ShowRequest
	if err := decodeWireJSON(r.Body, &req, "mlx.ollama.show"); err != nil {
		writeOpenAIError(w, http.StatusBadRequest, err.Error(), "body")
		return
	}
	model, ok := resolveCompatModel(w, r.Context(), h.resolver, req.Model)
	if !ok {
		return
	}
	info := model.Info()
	details := map[string]string{
		"architecture": info.Architecture,
		"model_type":   model.ModelType(),
	}
	if info.QuantBits > 0 {
		details["quantization"] = core.Sprintf("q%d", info.QuantBits)
	}
	writeOpenAIJSON(w, http.StatusOK, ollamacompat.ShowResponse{Details: details})
}

func serveOllamaChatStream(w http.ResponseWriter, ctx context.Context, model inference.TextModel, req ollamacompat.ChatRequest, messages []inference.Message, opts ...inference.GenerateOption) {
	serveOllamaStream(w, ctx, model, req.Model, "", messages, true, opts...)
}

func serveOllamaGenerateStream(w http.ResponseWriter, ctx context.Context, model inference.TextModel, req ollamacompat.GenerateRequest, opts ...inference.GenerateOption) {
	serveOllamaStream(w, ctx, model, req.Model, req.Prompt, nil, false, opts...)
}

func serveOllamaStream(w http.ResponseWriter, ctx context.Context, model inference.TextModel, modelName, prompt string, messages []inference.Message, chat bool, opts ...inference.GenerateOption) {
	w.Header().Set("Content-Type", "application/x-ndjson")
	w.WriteHeader(http.StatusOK)
	flusher, _ := w.(http.Flusher)
	processor := parser.NewProcessor(parser.Config{Mode: parser.Capture}, parser.HintFromInference(model.Info()))
	writeLine := func(payload any) {
		_, _ = w.Write([]byte(core.Concat(core.JSONMarshalString(payload), "\n")))
		if flusher != nil {
			flusher.Flush()
		}
	}
	_ = forEachCompatToken(ctx, model, ollamaRequestID(), modelName, prompt, messages, opts, func(token inference.Token) bool {
		delta := processor.Process(token.Text)
		if delta == "" {
			return true
		}
		if chat {
			writeLine(ollamacompat.ChatResponse{Model: modelName, Message: ollamacompat.Message{Role: "assistant", Content: delta}})
		} else {
			writeLine(ollamacompat.GenerateResponse{Model: modelName, Response: delta})
		}
		return true
	})
	if tail := processor.Flush(); tail != "" {
		if chat {
			writeLine(ollamacompat.ChatResponse{Model: modelName, Message: ollamacompat.Message{Role: "assistant", Content: tail}})
		} else {
			writeLine(ollamacompat.GenerateResponse{Model: modelName, Response: tail})
		}
	}
	if chat {
		writeLine(ollamacompat.NewChatResponse(modelName, "", model.Metrics()))
	} else {
		writeLine(ollamacompat.NewGenerateResponse(modelName, "", model.Metrics()))
	}
}

func decodeWireJSON(body io.Reader, into any, scope string) error {
	if body == nil {
		return core.E(scope, "request body is nil", nil)
	}
	data, err := io.ReadAll(body)
	if err != nil {
		return core.E(scope, "read request body", err)
	}
	result := core.JSONUnmarshalString(string(data), into)
	if !result.OK {
		if err, ok := result.Value.(error); ok {
			return err
		}
		return core.E(scope, "invalid request body", nil)
	}
	return nil
}

func requireCompatMethod(w http.ResponseWriter, r *http.Request, method string) bool {
	if r == nil {
		writeOpenAIError(w, http.StatusBadRequest, "request is nil", "request")
		return false
	}
	if r.Method != method {
		w.Header().Set("Allow", method)
		writeOpenAIError(w, http.StatusMethodNotAllowed, "method not allowed", "method")
		return false
	}
	return true
}

func resolveCompatModel(w http.ResponseWriter, ctx context.Context, resolver openaicompat.Resolver, modelName string) (inference.TextModel, bool) {
	if resolver == nil {
		writeOpenAIError(w, http.StatusServiceUnavailable, "handler is not configured", "model")
		return nil, false
	}
	if core.Trim(modelName) == "" {
		writeOpenAIError(w, http.StatusBadRequest, "model is required", "model")
		return nil, false
	}
	model, err := resolver.ResolveModel(ctx, modelName)
	if err != nil {
		writeOpenAIError(w, http.StatusNotFound, err.Error(), "model")
		return nil, false
	}
	return model, true
}

type resolverModelNameLister interface {
	ModelNames() []string
}

func resolverModelNames(resolver openaicompat.Resolver) []string {
	if lister, ok := resolver.(resolverModelNameLister); ok {
		return lister.ModelNames()
	}
	if backend, ok := resolver.(*openaicompat.BackendResolver); ok && backend != nil && backend.ModelPath != "" {
		return []string{core.PathBase(backend.ModelPath)}
	}
	return nil
}

func firstStopSequenceCut(content string, stops []string) (int, bool) {
	if content == "" || len(stops) == 0 {
		return 0, false
	}
	best := -1
	for _, stop := range stops {
		if stop == "" {
			continue
		}
		idx := indexString(content, stop)
		if idx >= 0 && (best < 0 || idx < best) {
			best = idx
		}
	}
	if best < 0 {
		return 0, false
	}
	return best, true
}

func normalizeAnthropicStopSequences(stops []string) ([]string, error) {
	if len(stops) == 0 {
		return nil, nil
	}
	out := make([]string, 0, len(stops))
	for _, stop := range stops {
		if stop == "" {
			return nil, core.E("mlx.anthropic.messages", "stop_sequences must not contain empty strings", nil)
		}
		out = append(out, stop)
	}
	return out, nil
}

func anthropicMessageID() string {
	return "msg_" + core.FormatInt(time.Now().UnixNano(), 10)
}

func ollamaRequestID() string {
	return "ollama_" + core.FormatInt(time.Now().UnixNano(), 10)
}

func parseOpenAIModelOutput(model inference.TextModel, tokens []inference.Token, text string) (string, string) {
	var (
		result inference.ReasoningParseResult
		err    error
	)
	if p, ok := model.(inference.ReasoningParser); ok {
		result, err = p.ParseReasoning(tokens, text)
	} else if model != nil {
		result, err = parser.ForHint(parser.HintFromInference(model.Info())).ParseReasoning(tokens, text)
	} else {
		result, err = parser.ForHint(parser.Hint{}).ParseReasoning(tokens, text)
	}
	if err != nil {
		return text, ""
	}
	return result.VisibleText, reasoningText(result.Reasoning)
}

// indexString locates substr inside s, returning its index or -1.
func indexString(s, substr string) int {
	if substr == "" {
		return 0
	}
	if len(substr) > len(s) {
		return -1
	}
	for i := range len(s) - len(substr) + 1 {
		if s[i:i+len(substr)] == substr {
			return i
		}
	}
	return -1
}

func openAITokensText(tokens []inference.Token) string {
	builder := core.NewBuilder()
	builder.Grow(openAITokensTextLen(tokens))
	for _, token := range tokens {
		builder.WriteString(token.Text)
	}
	return builder.String()
}

func reasoningText(segments []inference.ReasoningSegment) string {
	if len(segments) == 0 {
		return ""
	}
	builder := core.NewBuilder()
	total := 0
	for _, segment := range segments {
		total += len(segment.Text)
	}
	builder.Grow(total)
	for _, segment := range segments {
		builder.WriteString(segment.Text)
	}
	return builder.String()
}

func openAITokensTextLen(tokens []inference.Token) int {
	total := 0
	for _, token := range tokens {
		total += len(token.Text)
	}
	return total
}
