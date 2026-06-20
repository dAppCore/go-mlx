// SPDX-Licence-Identifier: EUPL-1.2

package openai

import (
	"context"
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

// streamScheduleErrModel is a ReasoningParser-free SchedulerModel whose Schedule
// rejects the request, so forEachCompatToken returns an error from inside a
// streaming handler. The HTTP headers are already on the wire by then, so the
// handlers cannot change the status — they log the error and still close the
// stream with a well-formed terminal frame. This drives the streamErr != nil
// branches of serveAnthropicMessageStream and serveOllamaStream (and, via the
// Responses route, serveOpenAIResponseStream's error path) that the
// non-streaming collect-error tests never reach.
type streamScheduleErrModel struct {
	openAIMockModel
}

func (m *streamScheduleErrModel) Schedule(context.Context, inference.ScheduledRequest) (inference.RequestHandle, <-chan inference.ScheduledToken, error) {
	return inference.RequestHandle{}, nil, core.NewError("stream schedule rejected")
}

// TestOpenAI_StreamHandlers_Bad_GenerationError covers the post-header
// streamErr != nil branch in each streaming wire handler: a scheduler that
// rejects the request makes forEachCompatToken return before any token, yet the
// stream must still finish 200 with its protocol's terminal frame so the client
// parser closes cleanly.
func TestOpenAI_StreamHandlers_Bad_GenerationError(t *testing.T) {
	model := &streamScheduleErrModel{}
	resolver := openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": model})
	handler := NewMux(resolver)

	cases := []struct {
		name     string
		path     string
		body     string
		terminal string
	}{
		{
			name:     "anthropic",
			path:     anthropiccompat.DefaultMessagesPath,
			body:     `{"model":"qwen","stream":true,"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}`,
			terminal: "event: message_stop",
		},
		{
			name:     "ollama chat",
			path:     ollamacompat.DefaultChatPath,
			body:     `{"model":"qwen","stream":true,"messages":[{"role":"user","content":"hi"}]}`,
			terminal: `"done":true`,
		},
		{
			name:     "ollama generate",
			path:     ollamacompat.DefaultGeneratePath,
			body:     `{"model":"qwen","stream":true,"prompt":"hi"}`,
			terminal: `"done":true`,
		},
		{
			// The Responses stream surfaces a generation error as a
			// response.error event followed by the [DONE] frame and returns
			// early — it never reaches response.completed.
			name:     "responses",
			path:     openaicompat.DefaultResponsesPath,
			body:     `{"model":"qwen","stream":true,"input":[{"role":"user","content":"hi"}]}`,
			terminal: "response.error",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rec := httptest.NewRecorder()
			handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, tc.path, strings.NewReader(tc.body)))
			// Headers were flushed before the generation error, so the status is
			// still the streaming 200 — the error is logged, not surfaced.
			if rec.Code != http.StatusOK {
				t.Fatalf("status = %d body=%s, want 200 (error logged, stream closes)", rec.Code, rec.Body.String())
			}
			if !strings.Contains(rec.Body.String(), tc.terminal) {
				t.Fatalf("body = %s, want terminal frame %q", rec.Body.String(), tc.terminal)
			}
		})
	}
}

// markerVisibleParserModel is a ReasoningParser whose ParseReasoning returns a
// VisibleText made entirely of a channel-control marker with a nil error. After
// cleanChannelMarkers strips the marker, parseOpenAIModelOutput hands back an
// empty visible string — even though the model genuinely streamed text. That is
// the exact precondition for serveOpenAIResponseStream's recovery branch:
// visible == "" yet the per-token visibleBuilder captured real deltas, so the
// completed response falls back to the streamed text instead of emitting empty.
type markerVisibleParserModel struct {
	openAIMockModel
}

func (m *markerVisibleParserModel) ParseReasoning([]inference.Token, string) (inference.ReasoningParseResult, error) {
	// "<|channel>" is one of the markers cleanChannelMarkers removes; cleaning
	// it yields "" so the post-parse visible is empty while the streamed
	// builder still holds the plain token text.
	return inference.ReasoningParseResult{VisibleText: "<|channel>"}, nil
}

// TestOpenAI_Responses_Good_StreamFallbackToBuilder drives the
// visible == "" && visibleBuilder != "" recovery in serveOpenAIResponseStream.
// The model streams plain text (so the processor accumulates a visible builder),
// but its parser classifies the final visible as a bare channel marker that
// cleans to "" — the handler must recover the streamed text into the completed
// response rather than send an empty reply.
func TestOpenAI_Responses_Good_StreamFallbackToBuilder(t *testing.T) {
	model := &markerVisibleParserModel{openAIMockModel: openAIMockModel{
		tokens:  []inference.Token{{Text: "An"}, {Text: "swer"}},
		metrics: inference.GenerateMetrics{PromptTokens: 1, GeneratedTokens: 2},
	}}
	resolver := openaicompat.NewStaticResolver(map[string]inference.TextModel{"qwen": model})
	handler := NewMux(resolver)

	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, openaicompat.DefaultResponsesPath,
		strings.NewReader(`{"model":"qwen","stream":true,"input":[{"role":"user","content":"hi"}]}`)))

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", rec.Code, rec.Body.String())
	}
	body := rec.Body.String()
	// The per-token deltas arrive as usual.
	if !strings.Contains(body, `"delta":"An"`) || !strings.Contains(body, `"delta":"swer"`) {
		t.Fatalf("body = %s, want streamed deltas", body)
	}
	// The completed response recovers the streamed text (Answer) from the
	// builder rather than the cleaned-to-empty parser output.
	if !strings.Contains(body, "response.completed") {
		t.Fatalf("body = %s, want response.completed", body)
	}
	if !strings.Contains(body, "Answer") {
		t.Fatalf("body = %s, want recovered builder text in completed response", body)
	}
}
