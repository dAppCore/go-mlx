// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks + a byte-identity guard for the gemma4 chat formatter. Per
// AX-11 the formatter fires once per chat/serve request, so the per-render
// cost scales linearly with request rate. The hot internal is stripThinking,
// which runs on every assistant turn and — when the turn carries a
// <|channel>thought…<channel|> span — used to spin a second Builder plus a
// SplitN slice per span. These benches exercise that span path directly and
// at the request level; stripThinkingOld pins the pre-sweep behaviour so the
// optimised stripThinking is proven byte-identical across every edge case.
//
// Run: go test -bench='BenchmarkGemma4Chat' -benchtime=200ms -benchmem -run='^$' ./pkg/metal/model/gemma4/chat

package gemma4chat

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/chat"
)

// Sinks defeat compiler dead-code elimination.
var (
	gemma4BenchSinkString string
)

// stripThinkingOld is the pre-sweep implementation, retained only as the
// reference oracle for TestStripThinking_ByteIdentical. It must not be wired
// into the formatter — it exists so the optimised stripThinking is held
// byte-for-byte to the behaviour shipped before the AX-11 alloc sweep.
func stripThinkingOld(text string) string {
	if text == "" || !core.Contains(text, "<|channel>") {
		return core.Trim(text)
	}
	out := core.NewBuilder()
	for {
		parts := core.SplitN(text, "<|channel>", 2)
		out.WriteString(parts[0])
		if len(parts) != 2 {
			break
		}
		after := core.SplitN(parts[1], "<channel|>", 2)
		if len(after) != 2 {
			break
		}
		text = after[1]
	}
	return core.Trim(out.String())
}

// stripThinkingCases covers every branch a SplitN→Index rewrite could
// silently diverge on: empty, no-channel, single block, leading-text + block,
// unclosed open, multiple blocks, block-at-end, a stray close before any open,
// and whitespace-only.
var stripThinkingCases = []string{
	"",
	"no channel here",
	"<|channel>thought\nprivate<channel|>visible",
	"lead <|channel>x<channel|>vis",
	"<|channel>unclosed no end",
	"a<|channel>1<channel|>b<|channel>2<channel|>c",
	"<|channel>only<channel|>",
	"stray <channel|> before any open",
	"<|channel>t<channel|>",
	"  spaces  ",
	"head<|channel>mid<channel|>", // block at end, leading text present
	"<|channel>a<channel|><|channel>b<channel|>tail",
	"text <channel|> close then <|channel>open<channel|> done",
	"<|channel>",                                // bare open, no close
	"head<|channel>unclosed",                    // leading text + unclosed
	"<|channel>a<|channel>b<channel|>c",         // two opens, one close
	"<|channel>x<channel|>y<channel|>z",         // stray close inside visible suffix
	"<|channel>a<channel|>vis<|channel>unclosed", // block then unclosed → general loop
}

func TestStripThinking_ByteIdentical(t *testing.T) {
	for _, c := range stripThinkingCases {
		if got, want := stripThinking(c), stripThinkingOld(c); got != want {
			t.Errorf("stripThinking(%q) = %q, want %q (diverged from pre-sweep impl)", c, got, want)
		}
	}
}

// --- stripThinking: the span-stripping internal, called per assistant turn ---

// A realistic single-block thought turn: a private reasoning block followed by
// the visible answer — the dominant production shape. Hits the zero-alloc
// suffix fast path.
const benchThoughtTurn = "<|channel>thought\n" +
	"The user is asking for a one-sentence summary. I should keep it tight " +
	"and avoid restating the numbers they mentioned, focusing on the cache " +
	"hit explaining the halved latency on the second request.\n<channel|>" +
	"Warming the prefix cache halves the second request latency because the " +
	"shared prefix tokens are reused from the cache rather than recomputed."

func BenchmarkGemma4Chat_StripThinking_SingleBlock(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		gemma4BenchSinkString = stripThinking(benchThoughtTurn)
	}
}

// Two thought blocks interleaved with visible text — forces the stitching
// loop rather than the contiguous-suffix fast path.
const benchThoughtTurnMulti = "<|channel>plan\nFirst I outline the steps.<channel|>" +
	"Here is the first part of the answer. " +
	"<|channel>check\nNow I verify the second half holds.<channel|>" +
	"And here is the concluding part of the answer for the reader."

func BenchmarkGemma4Chat_StripThinking_MultiBlock(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		gemma4BenchSinkString = stripThinking(benchThoughtTurnMulti)
	}
}

// No-channel content takes the alloc-free Trim fast path; pins it at 0 allocs.
func BenchmarkGemma4Chat_StripThinking_NoChannel(b *testing.B) {
	const plain = "Warming the prefix cache halves the second request latency " +
		"because the shared prefix tokens are reused from the cache."
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		gemma4BenchSinkString = stripThinking(plain)
	}
}

// --- Format: request-level render with thought-channel assistant history ---

// benchThinkingMessages builds a multi-turn history whose assistant turns
// carry private thought channels — the shape stripThinking actually runs on
// in a served conversation. User turns mirror the ~500-char prompt from the
// neutral chat bench.
func benchThinkingMessages(turnCount int) []chat.Message {
	user := "Could you please summarise the following short paragraph for me? " +
		"It talks about a small experimental setup measuring how a model " +
		"behaves when the prompt cache is warmed by a previous request and " +
		"a second request shares the same prefix; the observation is that " +
		"the second request completes in roughly half the time of the first."
	out := make([]chat.Message, 0, turnCount)
	for i := range turnCount {
		if i%2 == 0 {
			out = append(out, chat.Message{Role: "user", Content: user})
		} else {
			out = append(out, chat.Message{Role: "assistant", Content: benchThoughtTurn})
		}
	}
	return out
}

func BenchmarkGemma4Chat_Format_Thinking_5Turns(b *testing.B) {
	messages := benchThinkingMessages(5)
	cfg := chat.Config{Architecture: "gemma4_text"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		gemma4BenchSinkString = Format(messages, cfg)
	}
}

func BenchmarkGemma4Chat_Format_Thinking_20Turns(b *testing.B) {
	messages := benchThinkingMessages(20)
	cfg := chat.Config{Architecture: "gemma4_text"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		gemma4BenchSinkString = Format(messages, cfg)
	}
}
