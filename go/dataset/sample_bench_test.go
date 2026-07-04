// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for dataset.Sample and the in-memory SliceDataset primitives.
// Per AX-11 — CloneSample is invoked on every read out of any replayable
// dataset (JSONLDataset.Next / SliceDataset returns a defensive copy on
// each Next call), so a few hundred nanoseconds of per-sample copy cost
// adds up across 10k-row corpora. CloneSamples is the bulk variant the
// JSONL loader uses at construction time.
//
// Run:    go test -bench='BenchmarkSample|BenchmarkSliceDataset|BenchmarkCloneSamples' -benchmem -run='^$' ./go/dataset

package dataset

import (
	"testing"

	core "dappco.re/go"
)

// Sinks defeat compiler DCE.
var (
	sampleBenchSample  Sample
	sampleBenchSamples []Sample
	sampleBenchOK      bool
	sampleBenchErr     error
)

// benchSample returns one representative supervised fine-tuning record.
// Meta map carries the format-label entry the JSONL loader stamps on every
// sample plus a couple of common training-side tags.
func benchSample() Sample {
	return Sample{
		Prompt:   "Translate 'hello world' to French.",
		Response: "Bonjour le monde.",
		Meta: map[string]string{
			"format":  "prompt_response",
			"source":  "alpaca-mt",
			"split":   "train",
			"quality": "high",
		},
	}
}

// benchTextSample exercises the text-only path (no prompt/response, no Meta).
// Common in raw-corpus rows that flow through CloneSample.
func benchTextSample() Sample {
	return Sample{Text: "The quick brown fox jumps over the lazy dog."}
}

// benchSamples returns N representative records. Pre-built once per
// bench to keep allocation off the timer.
func benchSamples(n int) []Sample {
	out := make([]Sample, n)
	template := benchSample()
	for i := range out {
		out[i] = Sample{
			Prompt:   template.Prompt,
			Response: template.Response,
			Meta:     core.MapClone(template.Meta),
		}
	}
	return out
}

// --- CloneSample (per-row hot path) ---

func BenchmarkSample_CloneSample_PromptResponse(b *testing.B) {
	sample := benchSample()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sampleBenchSample = CloneSample(sample)
	}
}

// Text-only rows have no Meta map — exercises the cloneStringMap nil-fast path.
func BenchmarkSample_CloneSample_TextNoMeta(b *testing.B) {
	sample := benchTextSample()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sampleBenchSample = CloneSample(sample)
	}
}

// --- CloneSamples (bulk path used by JSONL loader and NewJSONL) ---

func BenchmarkSample_CloneSamples_100Rows(b *testing.B) {
	samples := benchSamples(100)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sampleBenchSamples = CloneSamples(samples)
	}
}

func BenchmarkSample_CloneSamples_1000Rows(b *testing.B) {
	samples := benchSamples(1000)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sampleBenchSamples = CloneSamples(samples)
	}
}

func BenchmarkSample_CloneSamples_10000Rows(b *testing.B) {
	samples := benchSamples(10000)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sampleBenchSamples = CloneSamples(samples)
	}
}

// --- NewSliceDataset constructor (copies the slice header + samples) ---

func BenchmarkSliceDataset_NewSliceDataset_1000Rows(b *testing.B) {
	samples := benchSamples(1000)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ds := NewSliceDataset(samples)
		sampleBenchOK = ds != nil
	}
}

// --- SliceDataset.Next sweep — the per-epoch iteration cost ---

func BenchmarkSliceDataset_NextSweep_100Rows(b *testing.B) {
	samples := benchSamples(100)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ds := NewSliceDataset(samples)
		for {
			sample, ok, err := ds.Next()
			sampleBenchSample = sample
			sampleBenchErr = err
			if !ok {
				break
			}
		}
	}
}

func BenchmarkSliceDataset_NextSweep_1000Rows(b *testing.B) {
	samples := benchSamples(1000)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ds := NewSliceDataset(samples)
		for {
			sample, ok, err := ds.Next()
			sampleBenchSample = sample
			sampleBenchErr = err
			if !ok {
				break
			}
		}
	}
}

// Reset is a hot path in multi-epoch training; bench the rewind on its own.
func BenchmarkSliceDataset_Reset(b *testing.B) {
	samples := benchSamples(1000)
	ds := NewSliceDataset(samples)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sampleBenchErr = ds.Reset()
	}
}

// --- Func dataset adapter (single-call indirection) ---

func BenchmarkSampleFunc_Next(b *testing.B) {
	sample := benchSample()
	fn := Func(func() (Sample, bool, error) { return sample, true, nil })
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s, ok, err := fn.Next()
		sampleBenchSample = s
		sampleBenchOK = ok
		sampleBenchErr = err
	}
}
