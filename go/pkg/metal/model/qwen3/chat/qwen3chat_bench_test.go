// SPDX-Licence-Identifier: EUPL-1.2

package qwen3chat

import (
	"testing"

	"dappco.re/go/mlx/chat"
)

// benchMessages is a representative multi-turn conversation with canonical role
// names (the documented hot path — wire handlers build messages with the
// canonical names, hitting NormaliseRole's alloc-free fast path) and realistic
// content lengths, so B/op reflects the formatter's own overhead rather than
// being swamped by trivially short turns.
var benchMessages = []chat.Message{
	{Role: "system", Content: "You are a careful, concise assistant. Answer in British English and prefer code-shaped formats (tables, lists, fenced blocks) where they aid clarity."},
	{Role: "user", Content: "I'm seeing higher-than-expected heap allocations in a Go request handler that formats chat prompts on every call. Where should I start looking?"},
	{Role: "assistant", Content: "Start with `go test -benchmem` on the formatter in isolation, then `go tool pprof -alloc_objects` to see allocs/op and where they originate. The usual culprits are per-message string operations, a second Builder, and SplitN in the build path."},
	{Role: "user", Content: "Got it — I'll add a representative benchmark and read the flat profile. Thanks."},
}

// BenchmarkFormat measures the ChatML formatter (Format) on the canonical hot
// path. Reads allocs/op and B/op; profile with
//
//	go test -bench=BenchmarkFormat -benchmem -memprofile=mem.out
//	go tool pprof -alloc_objects -list Format mem.out
func BenchmarkFormat(b *testing.B) {
	cfg := chat.Config{}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Format(benchMessages, cfg)
	}
}

// BenchmarkFormatLlama measures the Llama-header formatter (FormatLlama) on the
// canonical hot path.
func BenchmarkFormatLlama(b *testing.B) {
	cfg := chat.Config{}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = FormatLlama(benchMessages, cfg)
	}
}
