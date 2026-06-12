// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for session_agent.go — the Model-side fold helpers (folded
// prompt assembly, fold metadata, prefill text chunking). Per AX-11 —
// these fire per fold call. The session-side lifecycle adapters are
// benched in the session package, beside the code.
//
// Run:    go test -bench='BenchmarkSessionAgent' -benchmem -run='^$' ./go

package mlx

import (
	"testing"

	"dappco.re/go/mlx/agent"
)

// Sinks defeat compiler DCE.
var (
	sessionAgentBenchSinkString    string
	sessionAgentBenchSinkMap       map[string]string
	sessionAgentBenchSinkSleepOpts agent.SleepOptions
	sessionAgentBenchSinkChunks    []string
)

// --- agentMemoryFoldedPrompt ---

// Empty options — fast path; no Trim allocs.
func BenchmarkSessionAgent_FoldedPrompt_Empty(b *testing.B) {
	opts := AgentMemoryFoldOptions{}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sessionAgentBenchSinkString = agentMemoryFoldedPrompt(opts)
	}
}

// User-supplied FoldedPrompt — early-return path skipping the static
// header builder.
func BenchmarkSessionAgent_FoldedPrompt_UserPrompt(b *testing.B) {
	opts := AgentMemoryFoldOptions{FoldedPrompt: "user-supplied folded prompt body"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sessionAgentBenchSinkString = agentMemoryFoldedPrompt(opts)
	}
}

// Both summary + tail — the realistic fold case. Drives the Builder
// + the static header concat path.
func BenchmarkSessionAgent_FoldedPrompt_SummaryAndTail(b *testing.B) {
	opts := AgentMemoryFoldOptions{
		Summary:    "Summary of the previous 8k tokens of context, condensed to 200 chars roughly here.",
		RecentTail: "Recent tail keeping the last few exchanges verbatim for continuity.",
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sessionAgentBenchSinkString = agentMemoryFoldedPrompt(opts)
	}
}

// --- addAgentMemoryFoldMeta / addAgentMemoryMetadata ---

// Empty-value fast path. Dominant case for absent adapter/runtime fields.
func BenchmarkSessionAgent_AddFoldMeta_Empty(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sessionAgentBenchSinkMap = addAgentMemoryFoldMeta(nil, "key", "")
	}
}

// Real value into a nil map — single-key build.
func BenchmarkSessionAgent_AddFoldMeta_Build(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sessionAgentBenchSinkMap = addAgentMemoryFoldMeta(nil, "folded_state", "true")
	}
}

// --- agentMemoryTextChunks ---

// Empty input — fast path; iterator yields nothing.
func BenchmarkSessionAgent_TextChunks_Empty(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		seq := agentMemoryTextChunks("", 1024)
		for chunk := range seq {
			sessionAgentBenchSinkString = chunk
		}
	}
}

// Single yield — text shorter than chunkBytes.
func BenchmarkSessionAgent_TextChunks_Single(b *testing.B) {
	text := "Short folded prompt — under one chunk."
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		seq := agentMemoryTextChunks(text, 1024)
		for chunk := range seq {
			sessionAgentBenchSinkString = chunk
		}
	}
}

// Many chunks — drives the per-rune scan path.
func BenchmarkSessionAgent_TextChunks_Many(b *testing.B) {
	// 4kB of ASCII; chunkBytes 256 = 16 chunks.
	pad := make([]byte, 4096)
	for j := range pad {
		pad[j] = 'a' + byte(j%26)
	}
	text := string(pad)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		seq := agentMemoryTextChunks(text, 256)
		for chunk := range seq {
			sessionAgentBenchSinkString = chunk
		}
	}
}

// --- foldedAgentMemorySleepOptions ---

// Realistic options build — drives the meta map + labels-slice work.
func BenchmarkSessionAgent_FoldedSleepOpts(b *testing.B) {
	opts := agent.SleepOptions{
		Labels: []string{"env=prod", "agent=cladius"},
	}
	checkpoint := &agent.SleepReport{
		EntryURI:  "state://entry/parent",
		BundleURI: "state://bundle/parent",
		IndexURI:  "state://index/parent",
	}
	report := &AgentMemoryFoldReport{
		SummaryBytes:      300,
		RecentTailBytes:   800,
		FoldedPromptBytes: 1100,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sessionAgentBenchSinkSleepOpts = foldedAgentMemorySleepOptions(opts, checkpoint, report)
	}
}

// Options carry user-supplied Meta (3 entries). Exercises the
// cloneStringMap + pre-sized destination merge — the upstream call into
// addAgentMemoryFoldMeta then never grows the map.
func BenchmarkSessionAgent_FoldedSleepOpts_WithMeta(b *testing.B) {
	opts := agent.SleepOptions{
		Labels: []string{"env=prod"},
		Meta: map[string]string{
			"custom_a": "value-a",
			"custom_b": "value-b",
			"custom_c": "value-c",
		},
	}
	checkpoint := &agent.SleepReport{
		EntryURI:  "state://entry/parent",
		BundleURI: "state://bundle/parent",
		IndexURI:  "state://index/parent",
	}
	report := &AgentMemoryFoldReport{
		SummaryBytes:      300,
		RecentTailBytes:   800,
		FoldedPromptBytes: 1100,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sessionAgentBenchSinkSleepOpts = foldedAgentMemorySleepOptions(opts, checkpoint, report)
	}
}
