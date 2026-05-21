// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for chat template rendering — Format, TemplateName,
// NormaliseRole. Per AX-11 — Format fires once per chat-completion
// (and Anthropic / Ollama compat handlers all route through it),
// so a few microseconds per render scales linearly with request
// rate. NormaliseRole + templateName fire per message and per call
// respectively, so even the cheap branches are inside the inner
// loop of every wire handler.
//
// Run:    go test -bench='BenchmarkChat' -benchtime=100ms -benchmem -run='^$' ./go/chat

package chat

import "testing"

// Sinks defeat compiler DCE.
var (
	chatBenchSinkString string
)

// benchMessages builds a representative chat history. Average user
// message length is ~500 chars (roughly the inbound prompt size for
// a single-turn assistant call); assistant replies are similarly
// shaped. The structure mirrors the multi-turn shape every wire
// handler routes through.
func benchMessages(turnCount int) []Message {
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
	out := make([]Message, 0, turnCount)
	for i := 0; i < turnCount; i++ {
		if i%2 == 0 {
			out = append(out, Message{Role: "user", Content: user})
		} else {
			out = append(out, Message{Role: "assistant", Content: assistant})
		}
	}
	return out
}

// --- Format: per-architecture rendering at the canonical 1/5/20 turn shapes ---

func BenchmarkChat_Format_Qwen_1Turn(b *testing.B) {
	messages := benchMessages(1)
	cfg := Config{Architecture: "qwen3"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		chatBenchSinkString = Format(messages, cfg)
	}
}

func BenchmarkChat_Format_Qwen_5Turns(b *testing.B) {
	messages := benchMessages(5)
	cfg := Config{Architecture: "qwen3"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		chatBenchSinkString = Format(messages, cfg)
	}
}

func BenchmarkChat_Format_Qwen_20Turns(b *testing.B) {
	messages := benchMessages(20)
	cfg := Config{Architecture: "qwen3"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		chatBenchSinkString = Format(messages, cfg)
	}
}

func BenchmarkChat_Format_Gemma_5Turns(b *testing.B) {
	messages := benchMessages(5)
	cfg := Config{Architecture: "gemma3"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		chatBenchSinkString = Format(messages, cfg)
	}
}

// Gemma 4 carries an extra Trim() per message — surfaces the cost
// against the plain Gemma branch which writes content as-is.
func BenchmarkChat_Format_Gemma4_5Turns(b *testing.B) {
	messages := benchMessages(5)
	cfg := Config{Architecture: "gemma4_text"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		chatBenchSinkString = Format(messages, cfg)
	}
}

func BenchmarkChat_Format_Llama_5Turns(b *testing.B) {
	messages := benchMessages(5)
	cfg := Config{Architecture: "llama3"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		chatBenchSinkString = Format(messages, cfg)
	}
}

func BenchmarkChat_Format_Plain_5Turns(b *testing.B) {
	messages := benchMessages(5)
	cfg := Config{Template: "plain"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		chatBenchSinkString = Format(messages, cfg)
	}
}

// --- TemplateName: pure dispatch on Architecture / Template strings ---
// Fires once per Format call — Trim + Lower + a switch table.

func BenchmarkChat_TemplateName_Architecture(b *testing.B) {
	cfg := Config{Architecture: "qwen3_moe"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		chatBenchSinkString = TemplateName(cfg)
	}
}

func BenchmarkChat_TemplateName_Template(b *testing.B) {
	cfg := Config{Template: "qwen"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		chatBenchSinkString = TemplateName(cfg)
	}
}

func BenchmarkChat_TemplateName_Empty(b *testing.B) {
	cfg := Config{}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		chatBenchSinkString = TemplateName(cfg)
	}
}

// --- NormaliseRole: fires per message in every Format call ---

func BenchmarkChat_NormaliseRole_Canonical(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		chatBenchSinkString = NormaliseRole("user")
	}
}

func BenchmarkChat_NormaliseRole_Alias(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		chatBenchSinkString = NormaliseRole("gpt")
	}
}

func BenchmarkChat_NormaliseRole_Unknown(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		chatBenchSinkString = NormaliseRole("custom-role")
	}
}
