// SPDX-Licence-Identifier: EUPL-1.2

package openai

import (
	"bytes"
	"testing"

	core "dappco.re/go"
	ollamacompat "dappco.re/go/inference/provider/ollama"
)

// The Ollama streaming paths (/api/chat, /api/generate) build each per-token
// NDJSON frame by hand (writeOllamaChatFrame / writeOllamaGenerateFrame) instead
// of marshalling an ollamacompat.ChatResponse / GenerateResponse struct, to keep
// the per-token encode off the reflect path. The wire format is contract -- the
// Ollama client's NDJSON parser depends on the exact bytes -- so the hand-rolled
// builders must stay byte-identical to core.JSONMarshalString (encoding/json),
// which is what the path emitted before. Crucially that means the SAME
// HTML-safety escaping json.Marshal applies (< > & -> < > &): the sibling
// ollamacompat.Append* builders deliberately omit it (jsonenc), so they are NOT
// usable here -- only the in-package appendJSONStringHTML escaper matches.
//
// The oracle is the marshalled struct (with the exact field order, the always-
// present "done":false, and every omitempty count/duration suppressed at zero)
// terminated with "\n". These tests derive the expected bytes from that oracle
// rather than a hand-written template, so any drift in field order or escaping
// fails loudly.

func wantOllamaChatFrame(model, content string) string {
	resp := ollamacompat.ChatResponse{Model: model, Message: ollamacompat.Message{Role: "assistant", Content: content}}
	return core.Concat(core.JSONMarshalString(resp), "\n")
}

func wantOllamaGenerateFrame(model, response string) string {
	resp := ollamacompat.GenerateResponse{Model: model, Response: response}
	return core.Concat(core.JSONMarshalString(resp), "\n")
}

func TestOpenai_writeOllamaChatFrame_MatchesLegacy(t *testing.T) {
	for _, content := range escapeCorpus {
		if content == "" {
			continue // caller guards empty deltas; Content has no omitempty anyway
		}
		var buf bytes.Buffer
		writeOllamaChatFrame(&buf, nil, "qwen3", content)
		if got, want := buf.String(), wantOllamaChatFrame("qwen3", content); got != want {
			t.Errorf("writeOllamaChatFrame(content=%q) = %q, want legacy %q", content, got, want)
		}
	}
}

func TestOpenai_writeOllamaGenerateFrame_MatchesLegacy(t *testing.T) {
	for _, content := range escapeCorpus {
		if content == "" {
			continue
		}
		var buf bytes.Buffer
		writeOllamaGenerateFrame(&buf, nil, "qwen3", content)
		if got, want := buf.String(), wantOllamaGenerateFrame("qwen3", content); got != want {
			t.Errorf("writeOllamaGenerateFrame(content=%q) = %q, want legacy %q", content, got, want)
		}
	}
}

// The model name is also a variable, escaped string -- exercise it with the
// same hostile corpus, not just a plain "qwen3", since byte-identity must hold
// for any model name a caller routes.
func TestOpenai_writeOllamaFrames_VariableModel(t *testing.T) {
	for _, model := range escapeCorpus {
		var chatBuf, genBuf bytes.Buffer
		writeOllamaChatFrame(&chatBuf, nil, model, "hello")
		if got, want := chatBuf.String(), wantOllamaChatFrame(model, "hello"); got != want {
			t.Errorf("chat frame(model=%q) = %q, want %q", model, got, want)
		}
		writeOllamaGenerateFrame(&genBuf, nil, model, "hello")
		if got, want := genBuf.String(), wantOllamaGenerateFrame(model, "hello"); got != want {
			t.Errorf("generate frame(model=%q) = %q, want %q", model, got, want)
		}
	}
}

// TestOpenai_writeOllamaChatFrame_BufferReuse proves the returned buffer is
// reused across tokens without corrupting a frame.
func TestOpenai_writeOllamaChatFrame_BufferReuse(t *testing.T) {
	var buf []byte
	for _, s := range []string{"first", "a < b & c", "third \U0001F600", "x"} {
		var sink bytes.Buffer
		buf = writeOllamaChatFrame(&sink, buf, "qwen3", s)
		if got, want := sink.String(), wantOllamaChatFrame("qwen3", s); got != want {
			t.Fatalf("reused chat frame for %q = %q, want %q", s, got, want)
		}
	}
}

// FuzzOllamaChatFrame locks the hand-rolled chat NDJSON frame against the
// marshalled-struct-plus-newline oracle across both variable fields. The escaper
// fuzz proves appendJSONStringHTML == json.Marshal; this proves the surrounding
// literals (model/message/done punctuation, field order) are correct too.
func FuzzOllamaChatFrame(f *testing.F) {
	for _, s := range escapeCorpus {
		f.Add("qwen3", s)
		f.Add(s, "answer")
	}
	f.Fuzz(func(t *testing.T, model, content string) {
		if content == "" {
			return // empty delta is never emitted by the caller
		}
		var buf bytes.Buffer
		writeOllamaChatFrame(&buf, nil, model, content)
		if got, want := buf.String(), wantOllamaChatFrame(model, content); got != want {
			t.Fatalf("chat frame(model=%q content=%q) = %q, want %q", model, content, got, want)
		}
	})
}

// FuzzOllamaGenerateFrame locks the /api/generate frame the same way.
func FuzzOllamaGenerateFrame(f *testing.F) {
	for _, s := range escapeCorpus {
		f.Add("qwen3", s)
		f.Add(s, "answer")
	}
	f.Fuzz(func(t *testing.T, model, response string) {
		if response == "" {
			return
		}
		var buf bytes.Buffer
		writeOllamaGenerateFrame(&buf, nil, model, response)
		if got, want := buf.String(), wantOllamaGenerateFrame(model, response); got != want {
			t.Fatalf("generate frame(model=%q response=%q) = %q, want %q", model, response, got, want)
		}
	})
}
