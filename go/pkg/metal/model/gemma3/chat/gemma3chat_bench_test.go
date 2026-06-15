// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the Gemma chat formatter — Format renders the prompt once
// per chat-completion request on the serve path (and every Anthropic / Ollama
// compat handler routes through it), so a few microseconds and any avoidable
// heap allocation per render scale linearly with request rate (AX-11).
//
// These bench the formatter directly (gemma3chat.Format), not chat.Format —
// isolating the formatter from the neutral dispatcher's templateName work so
// the profile attributes allocations to this package's prompt-build path. The
// gemma3chat package imports chat (one-way), so calling Format here is no
// import cycle.
//
// Run: go test -bench='BenchmarkGemma3Chat' -benchtime=200ms -benchmem -run='^$' ./pkg/metal/model/gemma3/chat

package gemma3chat

import (
	"testing"

	"dappco.re/go/mlx/chat"
)

// Sink defeats compiler dead-code elimination of the rendered string.
var gemmaChatBenchSink string

// benchMessages builds a representative alternating user/assistant history.
// User and assistant turns are ~500 chars — roughly the inbound prompt size
// for a single chat turn. The alternation matters for the buffer-sizing
// path: assistant turns carry a byte more wrapping than user turns, so a
// no-system multi-turn render is where an under-sized Grow surfaces as a
// realloc.
func benchMessages(turnCount int) []chat.Message {
	user := "Could you please summarise the following short paragraph for me? " +
		"It talks about a small experimental setup measuring how a model " +
		"behaves when the prompt cache is warmed by a previous request and " +
		"a second request shares the same prefix; the observation is that " +
		"the second request completes in roughly half the time of the first, " +
		"which matches the expected savings from the cache hit path. Please " +
		"keep your summary to one sentence and avoid restating numbers."
	assistant := "Warming the prefix cache halves the second request latency " +
		"because the shared prefix tokens are reused from the cache rather " +
		"than recomputed; the rest of the time is spent on the new tail. " +
		"This matches the expected savings reported in the prompt cache " +
		"design notes and is consistent across the sample runs reported."
	out := make([]chat.Message, 0, turnCount)
	for i := range turnCount {
		if i%2 == 0 {
			out = append(out, chat.Message{Role: "user", Content: user})
		} else {
			out = append(out, chat.Message{Role: "assistant", Content: assistant})
		}
	}
	return out
}

// systemMessages prepends a system turn to an alternating history — the
// system-fold path (system content folded into the first user turn).
func systemMessages(turnCount int) []chat.Message {
	out := make([]chat.Message, 0, turnCount+1)
	out = append(out, chat.Message{Role: "system", Content: "You are a careful, concise assistant. Keep answers short."})
	out = append(out, benchMessages(turnCount)...)
	return out
}

// Single turn: one user message, fits the pre-sized buffer.
func BenchmarkGemma3Chat_Format_1Turn(b *testing.B) {
	messages := benchMessages(1)
	cfg := chat.Config{}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		gemmaChatBenchSink = Format(messages, cfg)
	}
}

// Five alternating turns, no system message — the case where an assistant
// turn's extra byte of wrapping per message can overflow an under-sized Grow.
func BenchmarkGemma3Chat_Format_5Turns(b *testing.B) {
	messages := benchMessages(5)
	cfg := chat.Config{}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		gemmaChatBenchSink = Format(messages, cfg)
	}
}

// Twenty alternating turns, no system — the under-budget shortfall scales
// with the assistant-turn count, so this is the clearest realloc signal.
func BenchmarkGemma3Chat_Format_20Turns(b *testing.B) {
	messages := benchMessages(20)
	cfg := chat.Config{}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		gemmaChatBenchSink = Format(messages, cfg)
	}
}

// System-fold path: the first user turn absorbs the system content. The
// system message's own per-message budget leaves slack, so this shape does
// not realloc — it isolates the steady-state floor from the sizing bug.
func BenchmarkGemma3Chat_Format_System5Turns(b *testing.B) {
	messages := systemMessages(5)
	cfg := chat.Config{}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		gemmaChatBenchSink = Format(messages, cfg)
	}
}
