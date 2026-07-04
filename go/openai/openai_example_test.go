// SPDX-Licence-Identifier: EUPL-1.2

package openai_test

import (
	"context"
	"fmt"
	"iter"
	"net/http"
	"net/http/httptest"
	"strings"

	core "dappco.re/go"
	"dappco.re/go/inference"
	openaicompat "dappco.re/go/inference/provider/openai"
	"dappco.re/go/mlx/openai"
)

// exampleModel is a tiny inference.TextModel that emits a fixed answer, so the
// runnable examples below produce deterministic output without loading a real
// model. It lives in the external _test package alongside the examples.
type exampleModel struct{}

func (exampleModel) Generate(context.Context, string, ...inference.GenerateOption) iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) { yield(inference.Token{Text: "Hello"}) }
}

func (exampleModel) Chat(context.Context, []inference.Message, ...inference.GenerateOption) iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) { yield(inference.Token{Text: "Hello"}) }
}

func (exampleModel) Classify(context.Context, []string, ...inference.GenerateOption) core.Result {
	return core.Ok([]inference.ClassifyResult(nil))
}

func (exampleModel) BatchGenerate(context.Context, []string, ...inference.GenerateOption) core.Result {
	return core.Ok([]inference.BatchResult(nil))
}

func (exampleModel) ModelType() string                  { return "example" }
func (exampleModel) Info() inference.ModelInfo          { return inference.ModelInfo{Architecture: "qwen3"} }
func (exampleModel) Metrics() inference.GenerateMetrics { return inference.GenerateMetrics{} }
func (exampleModel) Err() core.Result                   { return core.Ok(nil) }
func (exampleModel) Close() core.Result                 { return core.Ok(nil) }

// ExampleNewResolver shows the lazy Metal-backed resolver NewHandler / NewMux
// build on: it names the backend and remembers the model path, loading the
// weights only when a request first resolves the model.
func ExampleNewResolver() {
	resolver := openai.NewResolver("/models/qwen3")
	fmt.Println(resolver.BackendName)
	fmt.Println(resolver.ModelPath)
	// Output:
	// metal
	// /models/qwen3
}

// ExampleNewHandler mounts a single model behind the OpenAI chat-completions
// route. In production the path points at real weights; here the handler is
// constructed to show the one-liner wiring.
func ExampleNewHandler() {
	handler := openai.NewHandler("/models/qwen3", inference.WithContextLen(8192))
	fmt.Println(handler != nil)
	// Output: true
}

// ExampleNewModelMux mounts the full package-first route set (chat, responses,
// embeddings, Anthropic, Ollama, …) over a local model path.
func ExampleNewModelMux() {
	handler := openai.NewModelMux("/models/qwen3")
	fmt.Println(handler != nil)
	// Output: true
}

// ExampleNewMux serves the compatibility routes over a caller-supplied
// resolver — the seam tests and hosts use to inject a model without depending
// on the Metal backend. The example drives a chat completion and prints the
// stable response envelope fields.
func ExampleNewMux() {
	resolver := openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": exampleModel{}})
	handler := openai.NewMux(resolver)

	req := httptest.NewRequest(http.MethodPost, openaicompat.DefaultChatCompletionsPath,
		strings.NewReader(`{"model":"qwen","messages":[{"role":"user","content":"hi"}]}`))
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	body := rec.Body.String()
	fmt.Println(rec.Code)
	fmt.Println(strings.Contains(body, `"object":"chat.completion"`))
	fmt.Println(strings.Contains(body, `"content":"Hello"`))
	// Output:
	// 200
	// true
	// true
}

// ExampleNewMuxWithAdmin adds host-owned admin callbacks (health, wake, sleep)
// on top of the compatibility routes. The example hits the health endpoint and
// shows the default "ok" status the handler fills in.
func ExampleNewMuxWithAdmin() {
	resolver := openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": exampleModel{}})
	handler := openai.NewMuxWithAdmin(resolver, openai.AdminConfig{
		Health: func(context.Context) (openai.Health, error) {
			return openai.Health{Models: []string{"qwen3"}}, nil
		},
	})

	req := httptest.NewRequest(http.MethodGet, openai.DefaultHealthPath, nil)
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	body := rec.Body.String()
	fmt.Println(rec.Code)
	fmt.Println(strings.Contains(body, `"status":"ok"`))
	fmt.Println(strings.Contains(body, `"runtime":"go-mlx"`))
	// Output:
	// 200
	// true
	// true
}
