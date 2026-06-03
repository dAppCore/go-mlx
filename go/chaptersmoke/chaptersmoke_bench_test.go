// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the chapter-smoke shell-level helpers. The Capture/Generate
// callbacks dominate any real run, so this file targets only what the package
// itself owns: per-chapter URI formation (slug + bundleURI), store-kind
// normalisation, and the countingStore record path (struck inside every
// Generate-time store Get/Resolve/ResolveBytes).
//
// Run: go test -bench='Benchmark' -benchmem -run='^$' ./go/chaptersmoke
package chaptersmoke

import (
	"context"
	"testing"

	state "dappco.re/go/inference/state"
)

// Sinks defeat compiler DCE.
var (
	benchString string
	benchKind   string
	benchOK     bool
	benchInt    int
)

func BenchmarkSlug_Empty(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		benchString = slug(i, "")
	}
}

func BenchmarkSlug_Clean(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		benchString = slug(i, "chapter-one")
	}
}

func BenchmarkSlug_MixedCase(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		benchString = slug(i, "Chapter 7: The Sealed Letter")
	}
}

func BenchmarkBundleURI(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		benchString = bundleURI(i, "chapter-one")
	}
}

func BenchmarkNormalizeStoreKind_Path(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		benchKind = normalizeStoreKind("", "/tmp/store/state-kv-chapters.mvlog")
	}
}

func BenchmarkNormalizeStoreKind_PathMP4(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		benchKind = normalizeStoreKind("", "/tmp/store/state-kv-chapters.mp4")
	}
}

func BenchmarkNormalizeStoreKind_Alias(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		benchKind = normalizeStoreKind("mvlog", "")
	}
}

func BenchmarkHasCaseInsensitiveSuffix_Hit(b *testing.B) {
	b.ReportAllocs()
	const path = "/tmp/store/state-kv-chapters.mp4"
	for i := 0; i < b.N; i++ {
		benchOK = hasCaseInsensitiveSuffix(path, ".mp4")
	}
}

func BenchmarkHasCaseInsensitiveSuffix_Miss(b *testing.B) {
	b.ReportAllocs()
	const path = "/tmp/store/state-kv-chapters.mvlog"
	for i := 0; i < b.N; i++ {
		benchOK = hasCaseInsensitiveSuffix(path, ".mp4")
	}
}

func BenchmarkAnswerPlausible_NoTerms(b *testing.B) {
	b.ReportAllocs()
	const answer = "Marcus identifies the chapter's pressure."
	for i := 0; i < b.N; i++ {
		benchOK = answerPlausible(answer, nil)
	}
}

func BenchmarkAnswerPlausible_TermsHit(b *testing.B) {
	b.ReportAllocs()
	const answer = "Marcus identifies the chapter's pressure."
	terms := []string{"Marcus"}
	for i := 0; i < b.N; i++ {
		benchOK = answerPlausible(answer, terms)
	}
}

func BenchmarkAnswerPlausible_TermsMulti(b *testing.B) {
	b.ReportAllocs()
	const answer = "Marcus and Julia plan the chapter together with the council."
	terms := []string{"Marcus", "Julia", "council"}
	for i := 0; i < b.N; i++ {
		benchOK = answerPlausible(answer, terms)
	}
}

func BenchmarkValidateStoreKind_Bad(b *testing.B) {
	b.ReportAllocs()
	var benchErr error
	for i := 0; i < b.N; i++ {
		benchErr = validateStoreKind("bogus")
	}
	_ = benchErr
}

func BenchmarkRun_Bad_MissingGenerate(b *testing.B) {
	b.ReportAllocs()
	cfg := Config{Chapters: []Input{{Text: "x", Question: "q"}}}
	runner := Runner{}
	ctx := context.Background()
	var benchErr error
	for i := 0; i < b.N; i++ {
		_, benchErr = Run(ctx, runner, cfg)
	}
	_ = benchErr
}

func BenchmarkQuestionPrompt(b *testing.B) {
	b.ReportAllocs()
	chapter := Input{Question: "who opens the sealed letter?"}
	for i := 0; i < b.N; i++ {
		benchString = questionPrompt(chapter)
	}
}

func BenchmarkCountingStore_Record_Small(b *testing.B) {
	store := newCountingStore(noopStore{})
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		store.record(i & 0x0F) // 16 unique chunks cycled
	}
	benchInt = store.UniqueReads()
}

func BenchmarkCountingStore_Record_Wide(b *testing.B) {
	store := newCountingStore(noopStore{})
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		store.record(i & 0xFFF) // 4096 unique chunks cycled
	}
	benchInt = store.UniqueReads()
}

func BenchmarkCountingStore_Record_AllUnique(b *testing.B) {
	store := newCountingStore(noopStore{})
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		store.record(i)
	}
	benchInt = store.UniqueReads()
}

func BenchmarkCountingStore_Hinted_FillsExpected(b *testing.B) {
	const expected = 64
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		store := newCountingStoreHint(noopStore{}, expected)
		for j := range expected {
			store.record(j)
		}
		benchInt = store.UniqueReads()
	}
}

func BenchmarkCountingStore_Unhinted_FillsExpected(b *testing.B) {
	const expected = 64
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		store := newCountingStore(noopStore{})
		for j := range expected {
			store.record(j)
		}
		benchInt = store.UniqueReads()
	}
}

// noopStore is a state.Store stub for record-only benchmarks; the underlying
// Get/Resolve paths are not exercised here — record() is what is being
// measured.
type noopStore struct{}

func (noopStore) Get(context.Context, int) (string, error)               { return "", nil }
func (noopStore) Resolve(context.Context, int) (state.Chunk, error)      { return state.Chunk{}, nil }
func (noopStore) ResolveBytes(context.Context, int) (state.Chunk, error) { return state.Chunk{}, nil }
