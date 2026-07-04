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
	benchChunk  state.Chunk
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

func BenchmarkNormalizeStoreKind_Alias(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		benchKind = normalizeStoreKind("mvlog", "")
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

// --- Resolve-path benches -------------------------------------------------
//
// The bench-coverage audit flagged countingStore.Resolve / ResolveBytes /
// Reads as 0%-bench-covered. These drive the three wrapper methods over both
// store shapes that state.Resolve / state.ResolveBytes branch on:
//
//   - resolverStore implements state.Resolver + state.BinaryResolver, so the
//     wrapper hits the fast delegate branch — measures the record() + forward
//     cost the orchestration pays on every Generate-time chunk read.
//   - getOnlyStore implements only the bare state.Store (Get), forcing the
//     fallback arms: Resolve builds a Chunk{Text} from Get, and ResolveBytes
//     materialises chunk.Data via []byte(chunk.Text) — the whole-chunk slurp
//     a text-only backend pays to satisfy the BinaryResolver byte contract.
//
// Both fixtures return a fixed, pre-built payload so the bench isolates the
// wrapper + state-package resolve cost rather than per-call text formatting.

// chapterChunkText is a chapter-sized payload (no per-call build) so the
// ResolveBytes []byte(Text) slurp shows a realistic B/op rather than a
// short-string outlier.
var chapterChunkText = func() string {
	const para = "Marcus unsealed the letter by candlelight; the council's verdict ran three lines and changed the chapter's pressure entirely. "
	buf := make([]byte, 0, len(para)*16)
	for range 16 {
		buf = append(buf, para...)
	}
	return string(buf)
}()

// resolverStore returns the fixed payload through the native Resolver /
// BinaryResolver branches — no per-ID formatting, so the bench measures the
// wrapper + delegate path, not string building.
type resolverStore struct{}

func (resolverStore) Get(context.Context, int) (string, error) { return chapterChunkText, nil }
func (resolverStore) Resolve(_ context.Context, id int) (state.Chunk, error) {
	return state.Chunk{Ref: state.ChunkRef{ChunkID: id}, Text: chapterChunkText}, nil
}
func (resolverStore) ResolveBytes(_ context.Context, id int) (state.Chunk, error) {
	return state.Chunk{Ref: state.ChunkRef{ChunkID: id}, Data: chapterChunkBytes}, nil
}

var chapterChunkBytes = []byte(chapterChunkText)

// getOnlyStore implements only state.Store, forcing the state.Resolve /
// state.ResolveBytes fallback branches (the latter slurps the whole text into
// a fresh []byte to satisfy the byte contract).
type getOnlyStore struct{}

func (getOnlyStore) Get(context.Context, int) (string, error) { return chapterChunkText, nil }

func BenchmarkCountingStore_Resolve_ResolverStore(b *testing.B) {
	cs := newCountingStore(resolverStore{})
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchChunk, _ = cs.Resolve(ctx, i&0xFF)
	}
}

func BenchmarkCountingStore_Resolve_GetOnlyStore(b *testing.B) {
	cs := newCountingStore(getOnlyStore{})
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchChunk, _ = cs.Resolve(ctx, i&0xFF)
	}
}

func BenchmarkCountingStore_ResolveBytes_BinaryStore(b *testing.B) {
	cs := newCountingStore(resolverStore{})
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchChunk, _ = cs.ResolveBytes(ctx, i&0xFF)
	}
}

// BenchmarkCountingStore_ResolveBytes_GetOnlyStore exercises the fallback
// slurp: state.ResolveBytes copies chunk.Text into a fresh []byte. The B/op
// here is the chapter payload size — intrinsic to the text→bytes contract,
// not a removable wrapper cost.
func BenchmarkCountingStore_ResolveBytes_GetOnlyStore(b *testing.B) {
	cs := newCountingStore(getOnlyStore{})
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchChunk, _ = cs.ResolveBytes(ctx, i&0xFF)
	}
}

// BenchmarkCountingStore_Reads covers the Reads accessor on the populated
// wrapper — a plain field read, expected 0 B/op (floor probe so the audit's
// 0%-covered flag clears).
func BenchmarkCountingStore_Reads(b *testing.B) {
	cs := newCountingStore(resolverStore{})
	ctx := context.Background()
	for i := range 256 {
		_, _ = cs.Resolve(ctx, i)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchInt = cs.Reads()
	}
}

// noopStore is a state.Store stub for record-only benchmarks; the underlying
// Get/Resolve paths are not exercised here — record() is what is being
// measured.
type noopStore struct{}

func (noopStore) Get(context.Context, int) (string, error)               { return "", nil }
func (noopStore) Resolve(context.Context, int) (state.Chunk, error)      { return state.Chunk{}, nil }
func (noopStore) ResolveBytes(context.Context, int) (state.Chunk, error) { return state.Chunk{}, nil }
