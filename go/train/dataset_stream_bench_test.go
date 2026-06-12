// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for BuildDatasetBatches + normalizeDatasetBatchConfig.
// Per AX-11 — BuildDatasetBatches runs once per training run (and again
// per epoch when datasets are rebuilt), but its inner per-sample loop
// runs N×epochs times. The two interesting modes are non-packing (one
// example per row, padded inside SFT) and sequence-packing (the packer
// concatenates rows up to MaxSeqLen, flushing when the next row would
// overflow). Both go through buildSFTExample → tokenizer encode for each
// row, then the packer's per-flush slice clone.
//
// Tokenizer fixture (datasetStreamBenchTokenizer) is bench-only and is
// kept distinct from the existing fakeSFTTokenizer in sft_test.go to
// avoid coupling the bench file's lifetime to test-only state.
//
// Run:    go test -bench='BenchmarkDatasetStream' -benchmem -run='^$' ./go

package train

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/dataset"
	"dappco.re/go/mlx/spine"
)

// Sinks defeat compiler DCE.
var (
	dsStreamBenchBatches []SFTBatch
	dsStreamBenchErr     error
	dsStreamBenchConfig  dataset.BatchConfig
)

// datasetStreamBenchTokenizer is a fixed-vocab fake — sft.go's Tokenizer
// only needs Encode/EOS for BuildDatasetBatches to run. Encoded outputs
// are deterministic so the bench observes encode + pack overhead rather
// than tokenizer randomness.
type datasetStreamBenchTokenizer struct {
	promptIDs   []int32
	responseIDs []int32
	textIDs     []int32
	eos         int32
}

func (t datasetStreamBenchTokenizer) Encode(text string) []int32 {
	switch {
	case text == datasetStreamBenchPrompt:
		return append([]int32(nil), t.promptIDs...)
	case text == datasetStreamBenchResponse:
		return append([]int32(nil), t.responseIDs...)
	case text == datasetStreamBenchText:
		return append([]int32(nil), t.textIDs...)
	}
	out := make([]int32, 0, len(text))
	for _, r := range text {
		out = append(out, int32(r))
	}
	return out
}

func (t datasetStreamBenchTokenizer) Decode(tokens []int32) string {
	builder := core.NewBuilder()
	for _, token := range tokens {
		builder.WriteString(core.Sprintf("%d", token))
	}
	return builder.String()
}

func (t datasetStreamBenchTokenizer) TokenID(text string) (int32, bool) {
	tokens := t.Encode(text)
	if len(tokens) != 1 {
		return 0, false
	}
	return tokens[0], true
}

func (t datasetStreamBenchTokenizer) IDToken(id int32) string { return core.Sprintf("%d", id) }
func (t datasetStreamBenchTokenizer) DecodeOne(id int32) string {
	return t.Decode([]int32{id})
}
func (t datasetStreamBenchTokenizer) BOS() int32        { return 0 }
func (t datasetStreamBenchTokenizer) EOS() int32        { return t.eos }
func (t datasetStreamBenchTokenizer) HasBOSToken() bool { return false }

const (
	datasetStreamBenchPrompt   = "user:summarise the following passage"
	datasetStreamBenchResponse = "assistant:a concise summary in one sentence"
	datasetStreamBenchText     = "free-form paragraph used by the text branch"
)

// datasetStreamBenchTokens returns the prefilled token IDs used by the
// fake tokenizer. Numbers represent a 32-token prompt, 16-token response,
// and a 48-token text shape — close to the per-row scale of an alpaca
// or chat-style training row.
func datasetStreamBenchTokens() (prompt, response, text []int32) {
	prompt = make([]int32, 32)
	for i := range prompt {
		prompt[i] = int32(i + 100)
	}
	response = make([]int32, 16)
	for i := range response {
		response[i] = int32(i + 500)
	}
	text = make([]int32, 48)
	for i := range text {
		text[i] = int32(i + 900)
	}
	return prompt, response, text
}

// datasetStreamBenchSamples returns N prompt/response sample rows.
func datasetStreamBenchSamples(n int) []dataset.Sample {
	samples := make([]dataset.Sample, n)
	for i := range samples {
		samples[i] = dataset.Sample{Prompt: datasetStreamBenchPrompt, Response: datasetStreamBenchResponse}
	}
	return samples
}

// datasetStreamBenchTextSamples returns N free-form text rows.
func datasetStreamBenchTextSamples(n int) []dataset.Sample {
	samples := make([]dataset.Sample, n)
	for i := range samples {
		samples[i] = dataset.Sample{Text: datasetStreamBenchText}
	}
	return samples
}

// newDatasetStreamBenchTokenizer builds the Tokenizer wrapper around the
// fake tokenizer. *Tokenizer is the type BuildDatasetBatches expects.
func newDatasetStreamBenchTokenizer() *spine.Tokenizer {
	prompt, response, text := datasetStreamBenchTokens()
	return spine.NewTokenizer(datasetStreamBenchTokenizer{
		promptIDs:   prompt,
		responseIDs: response,
		textIDs:     text,
		eos:         9,
	})
}

// --- normalizeDatasetBatchConfig — defensive defaulting on every call ---

func BenchmarkDatasetStream_NormalizeBatchConfig_ZeroBatch(b *testing.B) {
	cfg := dataset.BatchConfig{}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dsStreamBenchConfig = normalizeDatasetBatchConfig(cfg)
	}
}

func BenchmarkDatasetStream_NormalizeBatchConfig_Populated(b *testing.B) {
	cfg := dataset.BatchConfig{BatchSize: 4, MaxSeqLen: 1024, SequencePacking: true}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dsStreamBenchConfig = normalizeDatasetBatchConfig(cfg)
	}
}

// --- BuildDatasetBatches — non-packing path ---

func BenchmarkDatasetStream_BuildDatasetBatches_NoPack_100Rows(b *testing.B) {
	tok := newDatasetStreamBenchTokenizer()
	samples := datasetStreamBenchSamples(100)
	cfg := dataset.BatchConfig{BatchSize: 4, MaxSeqLen: 128}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ds := dataset.NewSliceDataset(samples)
		dsStreamBenchBatches, dsStreamBenchErr = BuildDatasetBatches(tok, ds, cfg)
	}
}

func BenchmarkDatasetStream_BuildDatasetBatches_NoPack_1000Rows(b *testing.B) {
	tok := newDatasetStreamBenchTokenizer()
	samples := datasetStreamBenchSamples(1000)
	cfg := dataset.BatchConfig{BatchSize: 4, MaxSeqLen: 128}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ds := dataset.NewSliceDataset(samples)
		dsStreamBenchBatches, dsStreamBenchErr = BuildDatasetBatches(tok, ds, cfg)
	}
}

// --- BuildDatasetBatches — sequence-packing path (the datasetPacker hot path) ---

func BenchmarkDatasetStream_BuildDatasetBatches_Packed_100Rows(b *testing.B) {
	tok := newDatasetStreamBenchTokenizer()
	samples := datasetStreamBenchSamples(100)
	// MaxSeqLen large enough that packing flushes mid-pass — exercises
	// the add/flush ping-pong rather than dumping everything into one batch.
	cfg := dataset.BatchConfig{BatchSize: 1, MaxSeqLen: 256, SequencePacking: true}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ds := dataset.NewSliceDataset(samples)
		dsStreamBenchBatches, dsStreamBenchErr = BuildDatasetBatches(tok, ds, cfg)
	}
}

func BenchmarkDatasetStream_BuildDatasetBatches_Packed_1000Rows(b *testing.B) {
	tok := newDatasetStreamBenchTokenizer()
	samples := datasetStreamBenchSamples(1000)
	cfg := dataset.BatchConfig{BatchSize: 1, MaxSeqLen: 512, SequencePacking: true}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ds := dataset.NewSliceDataset(samples)
		dsStreamBenchBatches, dsStreamBenchErr = BuildDatasetBatches(tok, ds, cfg)
	}
}

// Aggressive packing — MaxSeqLen tight relative to row token count so the
// packer truncates often. Exercises the slice-clone branch in datasetPacker.add.
func BenchmarkDatasetStream_BuildDatasetBatches_Packed_TightSeq_1000Rows(b *testing.B) {
	tok := newDatasetStreamBenchTokenizer()
	samples := datasetStreamBenchSamples(1000)
	cfg := dataset.BatchConfig{BatchSize: 1, MaxSeqLen: 24, SequencePacking: true}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ds := dataset.NewSliceDataset(samples)
		dsStreamBenchBatches, dsStreamBenchErr = BuildDatasetBatches(tok, ds, cfg)
	}
}

// Text-only rows — exercise the "free-form text" branch of buildSFTExample.
func BenchmarkDatasetStream_BuildDatasetBatches_TextOnly_1000Rows(b *testing.B) {
	tok := newDatasetStreamBenchTokenizer()
	samples := datasetStreamBenchTextSamples(1000)
	cfg := dataset.BatchConfig{BatchSize: 4, MaxSeqLen: 128}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ds := dataset.NewSliceDataset(samples)
		dsStreamBenchBatches, dsStreamBenchErr = BuildDatasetBatches(tok, ds, cfg)
	}
}
