// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for JSONL ingestion + chat-shape normalization. Per AX-11 —
// LoadJSONL is invoked once per dataset open; cost scales with row count
// AND row shape (plain text vs alpaca-instruction vs openai-messages vs
// sharegpt-conversations). Training/eval pipelines routinely chew through
// 10k-100k row corpora at startup, so a 1us/row regression is 100ms wall
// time on a 100k corpus. MessagesToSample is the per-row chat normaliser
// the openai/sharegpt branches hit on every chat-format dataset row.
//
// Run:    go test -bench='BenchmarkJSONL|BenchmarkMessagesToSample' -benchmem -run='^$' ./go/dataset

package dataset

import (
	"strings"
	"testing"

	"dappco.re/go/inference"
	"dappco.re/go/mlx/chat"
)

// Sinks defeat compiler DCE.
var (
	jsonlBenchDataset  *JSONLDataset
	jsonlBenchErr      error
	jsonlBenchSample   Sample
	jsonlBenchOK       bool
	jsonlBenchSamples  []Sample
	jsonlBenchMessages []inference.Message
)

// Per-row templates representative of each branch in jsonRecord.toSample.
const (
	jsonlBenchRowText       = `{"text":"The quick brown fox jumps over the lazy dog."}`
	jsonlBenchRowPromptResp = `{"prompt":"Translate hello to French.","response":"Bonjour."}`
	jsonlBenchRowAlpaca     = `{"instruction":"Summarise the following","input":"long input passage here","output":"short answer"}`
	jsonlBenchRowOpenAI     = `{"messages":[` +
		`{"role":"system","content":"steady"},` +
		`{"role":"user","content":"ping"},` +
		`{"role":"assistant","content":"pong"}]}`
	jsonlBenchRowShareGPT = `{"conversations":[` +
		`{"from":"human","value":"hi"},` +
		`{"from":"gpt","value":"there"}]}`
	jsonlBenchRowReasoning = `{"problem":"2+2","thinking":"add the pair","solution":"4"}`
)

// repeatRow builds an N-row JSONL corpus by concatenating one shape
// repeatedly. The parser sees the same line shape on every step so the
// timer measures the steady-state per-row cost without inter-shape noise.
func repeatRow(row string, n int) string {
	if n <= 0 {
		return ""
	}
	var builder strings.Builder
	builder.Grow((len(row) + 1) * n)
	for i := 0; i < n; i++ {
		builder.WriteString(row)
		builder.WriteByte('\n')
	}
	return builder.String()
}

// mixedCorpus builds an N-row JSONL where each row cycles through the six
// shapes the parser supports. Closer to a real-world ingest mix.
func mixedCorpus(n int) string {
	shapes := []string{
		jsonlBenchRowText,
		jsonlBenchRowPromptResp,
		jsonlBenchRowAlpaca,
		jsonlBenchRowOpenAI,
		jsonlBenchRowShareGPT,
		jsonlBenchRowReasoning,
	}
	var builder strings.Builder
	for i := 0; i < n; i++ {
		builder.WriteString(shapes[i%len(shapes)])
		builder.WriteByte('\n')
	}
	return builder.String()
}

// --- LoadJSONL across shape and size ---

func BenchmarkJSONL_LoadJSONL_TextOnly_100Rows(b *testing.B) {
	corpus := repeatRow(jsonlBenchRowText, 100)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		jsonlBenchDataset, jsonlBenchErr = LoadJSONL(strings.NewReader(corpus), Config{})
	}
}

func BenchmarkJSONL_LoadJSONL_TextOnly_1000Rows(b *testing.B) {
	corpus := repeatRow(jsonlBenchRowText, 1000)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		jsonlBenchDataset, jsonlBenchErr = LoadJSONL(strings.NewReader(corpus), Config{})
	}
}

func BenchmarkJSONL_LoadJSONL_TextOnly_10000Rows(b *testing.B) {
	corpus := repeatRow(jsonlBenchRowText, 10000)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		jsonlBenchDataset, jsonlBenchErr = LoadJSONL(strings.NewReader(corpus), Config{})
	}
}

func BenchmarkJSONL_LoadJSONL_PromptResponse_1000Rows(b *testing.B) {
	corpus := repeatRow(jsonlBenchRowPromptResp, 1000)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		jsonlBenchDataset, jsonlBenchErr = LoadJSONL(strings.NewReader(corpus), Config{})
	}
}

func BenchmarkJSONL_LoadJSONL_Alpaca_1000Rows(b *testing.B) {
	corpus := repeatRow(jsonlBenchRowAlpaca, 1000)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		jsonlBenchDataset, jsonlBenchErr = LoadJSONL(strings.NewReader(corpus), Config{})
	}
}

// OpenAI messages exercise MessagesToSample + chat.Format on every row;
// the heaviest per-row branch.
func BenchmarkJSONL_LoadJSONL_OpenAIMessages_1000Rows(b *testing.B) {
	corpus := repeatRow(jsonlBenchRowOpenAI, 1000)
	cfg := Config{ChatTemplate: chat.Config{Architecture: "qwen3"}}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		jsonlBenchDataset, jsonlBenchErr = LoadJSONL(strings.NewReader(corpus), cfg)
	}
}

func BenchmarkJSONL_LoadJSONL_ShareGPT_1000Rows(b *testing.B) {
	corpus := repeatRow(jsonlBenchRowShareGPT, 1000)
	cfg := Config{ChatTemplate: chat.Config{Architecture: "qwen3"}}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		jsonlBenchDataset, jsonlBenchErr = LoadJSONL(strings.NewReader(corpus), cfg)
	}
}

func BenchmarkJSONL_LoadJSONL_Reasoning_1000Rows(b *testing.B) {
	corpus := repeatRow(jsonlBenchRowReasoning, 1000)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		jsonlBenchDataset, jsonlBenchErr = LoadJSONL(strings.NewReader(corpus), Config{})
	}
}

// Six-shape rotation — the real-world ingest mix.
func BenchmarkJSONL_LoadJSONL_Mixed_1000Rows(b *testing.B) {
	corpus := mixedCorpus(1000)
	cfg := Config{ChatTemplate: chat.Config{Architecture: "qwen3"}}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		jsonlBenchDataset, jsonlBenchErr = LoadJSONL(strings.NewReader(corpus), cfg)
	}
}

// --- NewJSONL — constructor path used by callers that already hold samples ---

func BenchmarkJSONL_NewJSONL_1000Rows(b *testing.B) {
	samples := benchSamples(1000)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		jsonlBenchDataset = NewJSONL(samples)
	}
}

// --- JSONLDataset.Next sweep — per-epoch iteration ---

func BenchmarkJSONL_NextSweep_1000Rows(b *testing.B) {
	ds := NewJSONL(benchSamples(1000))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := ds.Reset(); err != nil {
			b.Fatal(err)
		}
		for {
			sample, ok, err := ds.Next()
			jsonlBenchSample = sample
			jsonlBenchErr = err
			if !ok {
				break
			}
		}
	}
}

// Samples() is used by serialisation paths and replayable test fixtures.
func BenchmarkJSONL_Samples_1000Rows(b *testing.B) {
	ds := NewJSONL(benchSamples(1000))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		jsonlBenchSamples = ds.Samples()
	}
}

// --- MessagesToSample — per-row chat normaliser ---

func BenchmarkMessagesToSample_QwenTemplate_AssistantTail(b *testing.B) {
	messages := []inference.Message{
		{Role: "system", Content: "steady"},
		{Role: "user", Content: "ping"},
		{Role: "assistant", Content: "pong"},
	}
	cfg := chat.Config{Architecture: "qwen3"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		jsonlBenchSample, jsonlBenchOK, jsonlBenchErr = MessagesToSample(messages, cfg, "openai_messages")
	}
}

// User-tail variant exercises the "no assistant message" branch — used by
// chat datasets that ship prompt-only turns.
func BenchmarkMessagesToSample_QwenTemplate_UserTail(b *testing.B) {
	messages := []inference.Message{
		{Role: "system", Content: "steady"},
		{Role: "user", Content: "ping"},
	}
	cfg := chat.Config{Architecture: "qwen3"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		jsonlBenchSample, jsonlBenchOK, jsonlBenchErr = MessagesToSample(messages, cfg, "openai_messages")
	}
}

// Longer multi-turn conversation — closer to ShareGPT realistic shape.
func BenchmarkMessagesToSample_QwenTemplate_10Turn(b *testing.B) {
	messages := make([]inference.Message, 0, 10)
	messages = append(messages, inference.Message{Role: "system", Content: "steady"})
	for turn := 0; turn < 4; turn++ {
		messages = append(messages,
			inference.Message{Role: "user", Content: "user turn payload"},
			inference.Message{Role: "assistant", Content: "assistant turn payload"},
		)
	}
	messages = append(messages, inference.Message{Role: "user", Content: "trailing prompt"})
	cfg := chat.Config{Architecture: "qwen3"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		jsonlBenchSample, jsonlBenchOK, jsonlBenchErr = MessagesToSample(messages, cfg, "openai_messages")
	}
}
