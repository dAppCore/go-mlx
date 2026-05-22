// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for session_agent.go — agent-memory wake/sleep lifecycle
// adapters and pure folded-prompt / metadata helpers. Per AX-11 — these
// helpers fire per turn (every Sleep, every fold call goes through the
// metadata + label adapter path), so their alloc shape sets the per-turn
// floor for the inference contract layer.
//
// Run:    go test -bench='BenchmarkSessionAgent' -benchmem -run='^$' ./go

package mlx

import (
	"testing"

	"dappco.re/go/inference"
	"dappco.re/go/mlx/agent"
)

// Sinks defeat compiler DCE.
var (
	sessionAgentBenchSinkString    string
	sessionAgentBenchSinkBool      bool
	sessionAgentBenchSinkMap       map[string]string
	sessionAgentBenchSinkLabels    []string
	sessionAgentBenchSinkSleepOpts agent.SleepOptions
	sessionAgentBenchSinkWakeOpts  agent.WakeOptions
	sessionAgentBenchSinkInfMeta   map[string]string
	sessionAgentBenchSinkChunks    []string
	sessionAgentBenchSinkInfWake   inference.AgentMemorySleepResult
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

// --- shouldPrefillFoldedAgentMemory ---

// No folded marker — the dominant case. Token count makes PrefixTokens
// positive so we actually exercise the meta + label scans.
func BenchmarkSessionAgent_ShouldPrefill_NoMarker(b *testing.B) {
	entry := agent.StateIndexEntry{
		TokenCount: 4096,
		Meta:       map[string]string{"adapter_hash": "abc"},
		Labels:     []string{"env=prod", "agent=cladius"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sessionAgentBenchSinkBool = shouldPrefillFoldedAgentMemory(entry)
	}
}

// Has folded_state=true marker — meta branch taken via canonical fast path.
func BenchmarkSessionAgent_ShouldPrefill_MetaTrue(b *testing.B) {
	entry := agent.StateIndexEntry{
		TokenCount: 4096,
		Meta:       map[string]string{"folded_state": "true"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sessionAgentBenchSinkBool = shouldPrefillFoldedAgentMemory(entry)
	}
}

// Has folded-state label only — exercises the labels-loop fast path.
func BenchmarkSessionAgent_ShouldPrefill_LabelHit(b *testing.B) {
	entry := agent.StateIndexEntry{
		TokenCount: 4096,
		Labels:     []string{"env=prod", "folded-state"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sessionAgentBenchSinkBool = shouldPrefillFoldedAgentMemory(entry)
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

// --- agentMemoryLabelsFromInference ---

// Nil labels — fast path returns nil.
func BenchmarkSessionAgent_LabelsFromInf_Empty(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sessionAgentBenchSinkLabels = agentMemoryLabelsFromInference(nil)
	}
}

// Three labels — common case.
func BenchmarkSessionAgent_LabelsFromInf_Three(b *testing.B) {
	in := map[string]string{
		"env":        "prod",
		"agent":      "cladius",
		"experiment": "",
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sessionAgentBenchSinkLabels = agentMemoryLabelsFromInference(in)
	}
}

// --- agentMemoryMetadataFromInference ---

// Empty req — all empty-fast-path branches.
func BenchmarkSessionAgent_MetadataFromInf_Empty(b *testing.B) {
	req := inference.AgentMemorySleepRequest{}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sessionAgentBenchSinkInfMeta = agentMemoryMetadataFromInference(req)
	}
}

// Realistic req with adapter + runtime — drives 7 addAgentMemoryMetadata.
func BenchmarkSessionAgent_MetadataFromInf_Full(b *testing.B) {
	req := inference.AgentMemorySleepRequest{
		Adapter: inference.AdapterIdentity{
			Hash:   "abc123",
			Path:   "/models/lora.safetensors",
			Format: "safetensors",
			Rank:   16,
			Alpha:  32.0,
		},
		Runtime: inference.RuntimeIdentity{
			Backend:   "metal",
			Device:    "Apple M3 Ultra",
			CacheMode: "page",
			Version:   "0.42",
		},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sessionAgentBenchSinkInfMeta = agentMemoryMetadataFromInference(req)
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

// --- agentMemorySleepOptionsFromInference ---

// Full req — drives both the metadata builder and the labels-from-inf
// path together; this is the per-turn cost.
func BenchmarkSessionAgent_SleepOptsFromInf(b *testing.B) {
	req := inference.AgentMemorySleepRequest{
		EntryURI: "state://entry",
		Adapter: inference.AdapterIdentity{
			Hash: "abc", Format: "safetensors", Rank: 16, Alpha: 32.0,
		},
		Runtime: inference.RuntimeIdentity{
			Backend: "metal", Device: "Apple M3 Ultra", Version: "0.42",
		},
		Labels: map[string]string{"agent": "cladius"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sessionAgentBenchSinkSleepOpts = agentMemorySleepOptionsFromInference(req)
	}
}

// --- toInferenceAgentMemorySleepResult ---

// Hot-path result formatter — Sleep returns this on every call.
func BenchmarkSessionAgent_ToInfSleepResult(b *testing.B) {
	report := &agent.SleepReport{
		EntryURI:      "state://entry",
		BundleURI:     "state://bundle",
		IndexURI:      "state://index",
		Title:         "session-42",
		SnapshotHash:  "abc",
		IndexHash:     "def",
		TokenCount:    4096,
		BlockSize:     128,
		BlocksWritten: 32,
		BlocksReused:  4,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = toInferenceAgentMemorySleepResult(report)
	}
}
