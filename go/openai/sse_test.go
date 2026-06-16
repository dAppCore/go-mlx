// SPDX-Licence-Identifier: EUPL-1.2

package openai

import (
	"bytes"
	"io"
	"testing"

	core "dappco.re/go"
	ollamacompat "dappco.re/go/inference/ollama"
	openaicompat "dappco.re/go/inference/openai"
)

// writeSSEData / writeSSEEvent must produce byte-identical frames to the
// legacy []byte(core.Concat(...)) form they replace — the streaming wire
// format is contract (Claude Code's + the OpenAI SDK's SSE parsers depend
// on the exact "data: <json>\n\n" / "event: <name>\ndata: <json>\n\n" shape).

func TestOpenai_writeSSEData_Good(t *testing.T) {
	var buf bytes.Buffer
	writeSSEData(&buf, `{"type":"x","delta":"hi"}`)
	want := "data: {\"type\":\"x\",\"delta\":\"hi\"}\n\n"
	if buf.String() != want {
		t.Fatalf("writeSSEData framing = %q, want %q", buf.String(), want)
	}
}

func TestOpenai_writeSSEData_MatchesLegacy(t *testing.T) {
	event := openaicompat.ResponseStreamEvent{Type: "response.output_text.delta", Delta: "Answer"}
	payload := core.JSONMarshalString(event)
	var buf bytes.Buffer
	writeSSEData(&buf, payload)
	legacy := core.Concat("data: ", payload, "\n\n")
	if buf.String() != legacy {
		t.Fatalf("writeSSEData = %q, want legacy concat %q", buf.String(), legacy)
	}
}

func TestOpenai_writeSSEData_Ugly(t *testing.T) {
	var buf bytes.Buffer
	writeSSEData(&buf, "")
	if buf.String() != "data: \n\n" {
		t.Fatalf("writeSSEData empty = %q, want %q", buf.String(), "data: \n\n")
	}
}

func TestOpenai_writeSSEEvent_Good(t *testing.T) {
	var buf bytes.Buffer
	writeSSEEvent(&buf, "message_stop", `{"type":"message_stop"}`)
	want := "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
	if buf.String() != want {
		t.Fatalf("writeSSEEvent framing = %q, want %q", buf.String(), want)
	}
}

func TestOpenai_writeSSEEvent_MatchesLegacy(t *testing.T) {
	name, payload := "content_block_delta", `{"delta":{"text":"x"}}`
	var buf bytes.Buffer
	writeSSEEvent(&buf, name, payload)
	legacy := core.Concat("event: ", name, "\n", "data: ", payload, "\n\n")
	if buf.String() != legacy {
		t.Fatalf("writeSSEEvent = %q, want legacy %q", buf.String(), legacy)
	}
}

// --- alloc proof: helper vs the legacy []byte(Concat(...)) it replaces.
// Both fire per delta token on the streaming hot path. The legacy form
// pays JSON-string + concat-string + []byte-conversion; the helper pays
// only the JSON string (prefix/suffix are package []byte, payload is a
// zero-copy core.AsBytes view).

func BenchmarkOpenAI_SSEData_Legacy(b *testing.B) {
	event := openaicompat.ResponseStreamEvent{Type: "response.output_text.delta", Delta: "Answer"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = io.Discard.Write([]byte(core.Concat("data: ", core.JSONMarshalString(event), "\n\n")))
	}
}

func BenchmarkOpenAI_SSEData_Helper(b *testing.B) {
	event := openaicompat.ResponseStreamEvent{Type: "response.output_text.delta", Delta: "Answer"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		writeSSEData(io.Discard, core.JSONMarshalString(event))
	}
}

func BenchmarkOpenAI_SSEEvent_Legacy(b *testing.B) {
	name, payload := "content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Answer"}}`
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = io.Discard.Write([]byte(core.Concat("event: ", name, "\n", "data: ", payload, "\n\n")))
	}
}

func BenchmarkOpenAI_SSEEvent_Helper(b *testing.B) {
	name, payload := "content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Answer"}}`
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		writeSSEEvent(io.Discard, name, payload)
	}
}

// --- NDJSON (Ollama streaming wire shape: <json>\n per delta token) ---

func TestOpenai_writeNDJSONLine_Good(t *testing.T) {
	var buf bytes.Buffer
	writeNDJSONLine(&buf, `{"model":"qwen3","response":"hi"}`)
	want := "{\"model\":\"qwen3\",\"response\":\"hi\"}\n"
	if buf.String() != want {
		t.Fatalf("writeNDJSONLine framing = %q, want %q", buf.String(), want)
	}
}

func TestOpenai_writeNDJSONLine_MatchesLegacy(t *testing.T) {
	payload := core.JSONMarshalString(ollamacompat.GenerateResponse{Model: "qwen3", Response: "Answer"})
	var buf bytes.Buffer
	writeNDJSONLine(&buf, payload)
	legacy := core.Concat(payload, "\n")
	if buf.String() != legacy {
		t.Fatalf("writeNDJSONLine = %q, want legacy concat %q", buf.String(), legacy)
	}
}

func BenchmarkOllama_NDJSON_Legacy(b *testing.B) {
	payload := ollamacompat.GenerateResponse{Model: "qwen3", Response: "Answer"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = io.Discard.Write([]byte(core.Concat(core.JSONMarshalString(payload), "\n")))
	}
}

func BenchmarkOllama_NDJSON_Helper(b *testing.B) {
	payload := ollamacompat.GenerateResponse{Model: "qwen3", Response: "Answer"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		writeNDJSONLine(io.Discard, core.JSONMarshalString(payload))
	}
}
