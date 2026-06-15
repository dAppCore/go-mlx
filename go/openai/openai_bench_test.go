// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the OpenAI / Anthropic / Ollama compatibility wire
// helpers. Per AX-11 — JSON decode of the inbound request and JSON
// encode of the streaming chunk / response body fire on every chat
// completion served by NewMux. The stop-sequence scanner runs per
// streamed delta in the Anthropic path, and the per-token text
// concatenation runs over the whole token vector at end-of-stream.
//
// Run:    go test -bench='BenchmarkOpenAI' -benchtime=100ms -benchmem -run='^$' ./go/openai

package openai

import (
	"context"
	"net/http/httptest"
	"strings"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	anthropiccompat "dappco.re/go/inference/anthropic"
	openaicompat "dappco.re/go/inference/openai"
)

// Sinks defeat compiler DCE.
var (
	openAIBenchSinkString   string
	openAIBenchSinkInt      int
	openAIBenchSinkBool     bool
	openAIBenchSinkResponse openaicompat.ResponseRequest
	openAIBenchSinkErr      error
	openAIBenchSinkStops    []string
)

// Representative request body — single-turn user message plus a
// system instruction. Mirrors the typical shape every wire handler
// must decode at request entry.
const openAIBenchSingleTurnBody = `{"model":"qwen3","input":[{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":"Please summarise the following short paragraph for me; keep it to one sentence."}],"temperature":0.7,"top_p":0.95,"max_output_tokens":256,"stream":true,"stop":["<|im_end|>"]}`

// Multi-turn request — exercises the slice-grow path inside the
// ResponseInputMessage decode loop. 5 turns is the realistic
// chat-history shape for an assistant call.
const openAIBenchMultiTurnBody = `{"model":"qwen3","input":[{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":"What is 2+2?"},{"role":"assistant","content":"4"},{"role":"user","content":"Are you sure?"},{"role":"assistant","content":"Yes."},{"role":"user","content":"Why?"}],"temperature":0.7,"max_output_tokens":256,"stream":true}`

// --- decodeOpenAIResponseRequest — front-of-handler JSON decode ---

func BenchmarkOpenAI_DecodeResponseRequest_SingleTurn(b *testing.B) {
	body := openAIBenchSingleTurnBody
	clen := int64(len(body))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		openAIBenchSinkResponse, openAIBenchSinkErr = decodeOpenAIResponseRequest(strings.NewReader(body), clen)
	}
}

func BenchmarkOpenAI_DecodeResponseRequest_MultiTurn(b *testing.B) {
	body := openAIBenchMultiTurnBody
	clen := int64(len(body))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		openAIBenchSinkResponse, openAIBenchSinkErr = decodeOpenAIResponseRequest(strings.NewReader(body), clen)
	}
}

// --- decodeWireJSON — shared path also hit by Anthropic + Ollama handlers ---

func BenchmarkOpenAI_DecodeWireJSON_ChatCompletionRequest(b *testing.B) {
	body := `{"model":"qwen3","messages":[{"role":"system","content":"be helpful"},{"role":"user","content":"hi"}],"temperature":0.7,"top_p":0.95,"max_tokens":256,"stream":true}`
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var req openaicompat.ChatCompletionRequest
		openAIBenchSinkErr = decodeWireJSON(strings.NewReader(body), &req, "bench.chat")
	}
}

// --- Streaming chunk marshal — fires per delta token in serveOpenAIResponseStream ---

func BenchmarkOpenAI_StreamEventMarshal_Delta(b *testing.B) {
	event := openaicompat.ResponseStreamEvent{
		Type:  "response.output_text.delta",
		Delta: "Answer",
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		openAIBenchSinkString = core.JSONMarshalString(event)
	}
}

// Full Response{} payload — what the response.completed terminal
// event carries. Larger surface than a delta because it embeds the
// output message body + usage block.
func BenchmarkOpenAI_StreamEventMarshal_Completed(b *testing.B) {
	visible := "The summary is concise and to the point."
	resp := openaicompat.NewTextResponse(
		"resp_bench",
		"qwen3",
		visible,
		inference.GenerateMetrics{PromptTokens: 200, GeneratedTokens: 32},
	)
	event := openaicompat.ResponseStreamEvent{Type: "response.completed", Response: &resp}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		openAIBenchSinkString = core.JSONMarshalString(event)
	}
}

// --- firstStopSequenceCut — Anthropic stream's per-delta scan ---
// Runs against the accumulated `emitted + delta` string on every
// streamed token; the loop scales as O(content × |stops|).

func BenchmarkOpenAI_FirstStopSequenceCut_Miss(b *testing.B) {
	content := strings.Repeat("answer fragment ", 32) // ~512 chars, no match
	stops := []string{"<|im_end|>", "<|eot_id|>"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		openAIBenchSinkInt, openAIBenchSinkBool = firstStopSequenceCut(content, stops)
	}
}

func BenchmarkOpenAI_FirstStopSequenceCut_LateHit(b *testing.B) {
	content := strings.Repeat("answer fragment ", 32) + "<|im_end|>"
	stops := []string{"<|im_end|>", "<|eot_id|>"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		openAIBenchSinkInt, openAIBenchSinkBool = firstStopSequenceCut(content, stops)
	}
}

func BenchmarkOpenAI_FirstStopSequenceCut_EarlyHit(b *testing.B) {
	content := "<|im_end|>" + strings.Repeat("answer fragment ", 32)
	stops := []string{"<|im_end|>", "<|eot_id|>"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		openAIBenchSinkInt, openAIBenchSinkBool = firstStopSequenceCut(content, stops)
	}
}

// --- indexString — primitive substring locator used by firstStopSequenceCut ---

func BenchmarkOpenAI_IndexString_Miss(b *testing.B) {
	content := strings.Repeat("answer fragment ", 32)
	stop := "<|im_end|>"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		openAIBenchSinkInt = indexString(content, stop)
	}
}

// --- openAITokensText — end-of-stream text join over the token vector ---

func BenchmarkOpenAI_TokensText_32Tokens(b *testing.B) {
	tokens := benchOpenAITokens(32)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		openAIBenchSinkString = openAITokensText(tokens)
	}
}

func BenchmarkOpenAI_TokensText_256Tokens(b *testing.B) {
	tokens := benchOpenAITokens(256)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		openAIBenchSinkString = openAITokensText(tokens)
	}
}

// --- reasoningText — captured-segment concat at stream completion ---

func BenchmarkOpenAI_ReasoningText_Captured(b *testing.B) {
	segments := []inference.ReasoningSegment{
		{Kind: "thinking", Text: "Let me work through this step by step. "},
		{Kind: "thinking", Text: "First I'll identify the key claim, "},
		{Kind: "thinking", Text: "then check it against the available evidence. "},
		{Kind: "thinking", Text: "Finally I'll summarise in one sentence."},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		openAIBenchSinkString = reasoningText(segments)
	}
}

// --- normalizeAnthropicStopSequences — per-request validation ---

func BenchmarkOpenAI_NormalizeAnthropicStops_Typical(b *testing.B) {
	stops := []string{"<|im_end|>", "<|eot_id|>", "</response>"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		openAIBenchSinkStops, openAIBenchSinkErr = normalizeAnthropicStopSequences(stops)
	}
}

// --- ID helpers — fire once per request ---

func BenchmarkOpenAI_ResponseID(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		openAIBenchSinkString = openAIResponseID()
	}
}

func BenchmarkOpenAI_AnthropicMessageID(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		openAIBenchSinkString = anthropicMessageID()
	}
}

func BenchmarkOpenAI_OllamaRequestID(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		openAIBenchSinkString = ollamaRequestID()
	}
}

// --- serveAnthropicMessageStream — the per-token streaming hot loop ---
// The NoStops variant is the common case (a chat request with no stop
// sequences); it must avoid the per-token emitted+delta accumulation that
// only the stop-sequence scan needs. The WithStops variant is the guard
// against regressing the stop-aware path.

func benchServeAnthropicStream(b *testing.B, stops []string) {
	model := &openAIMockModel{tokens: benchOpenAITokens(256)}
	req := anthropiccompat.MessageRequest{Model: "qwen3", Stream: true}
	messages := []inference.Message{{Role: "user", Content: "summarise the paragraph"}}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rec := httptest.NewRecorder()
		serveAnthropicMessageStream(rec, context.Background(), model, req, messages, stops)
		openAIBenchSinkInt = rec.Code
	}
}

func BenchmarkOpenAI_ServeAnthropicStream_NoStops(b *testing.B) {
	benchServeAnthropicStream(b, nil)
}

func BenchmarkOpenAI_ServeAnthropicStream_WithStops(b *testing.B) {
	benchServeAnthropicStream(b, []string{"<|im_end|>", "<|eot_id|>"})
}

// benchOpenAITokens builds a synthetic token vector with realistic
// text fragments — sub-word pieces around 4 characters each, sized
// to feed the openAITokensText concat path.
func benchOpenAITokens(count int) []inference.Token {
	fragments := []string{"The", " quick", " brown", " fox", " jumps", " over", " the", " lazy", " dog", "."}
	out := make([]inference.Token, 0, count)
	for i := range count {
		out = append(out, inference.Token{ID: int32(i), Text: fragments[i%len(fragments)]})
	}
	return out
}
