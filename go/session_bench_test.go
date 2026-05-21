// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for session.go — ModelSession lifecycle (NewSession,
// Prefill, Generate, CaptureKV, SaveKV, LoadKV, Fork, Reset, Close)
// plus the two pure-function helpers (sessionParserControlToken,
// sessionParserTokenText) that fire on every generated token.
//
// Per AX-11 — sessionParserControlToken fires per token returned from
// GenerateStream; the lifecycle paths fire per agent wake/sleep cycle.
// Bench drives all the wireable paths against a fake native session
// (the same test fixture shape as session_test.go) so we measure the
// Go-side glue without booting Metal.
//
// Run:    go test -bench='BenchmarkSession' -benchmem -run='^$' ./go

package mlx

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/mlx/bundle"
	"dappco.re/go/mlx/internal/metal"
	"dappco.re/go/mlx/kv"
)

// Sinks defeat compiler DCE.
var (
	sessionBenchSinkErr      error
	sessionBenchSinkBool     bool
	sessionBenchSinkString   string
	sessionBenchSinkSession  *ModelSession
	sessionBenchSinkSnapshot *kv.Snapshot
	sessionBenchSinkAnalysis *kv.Analysis
	sessionBenchSinkText     string
)

// benchSessionNativeKV builds a small-but-non-trivial fake KV snapshot
// that exercises the toRootKVSnapshot deep-copy path. Used by
// CaptureKV / SaveKV / LoadKV benches.
func benchSessionNativeKV(tokenCount int) *metal.KVSnapshot {
	tokens := make([]int32, tokenCount)
	gen := make([]int32, tokenCount/4+1)
	key := make([]float32, tokenCount)
	value := make([]float32, tokenCount)
	for i := range tokens {
		tokens[i] = int32(i + 1)
		key[i] = float32(i)
		value[i] = float32(i + 1000)
	}
	for i := range gen {
		gen[i] = int32(i)
	}
	return &metal.KVSnapshot{
		Version:       metal.KVSnapshotVersion,
		Architecture:  "qwen3",
		Tokens:        tokens,
		Generated:     gen,
		TokenOffset:   tokenCount,
		NumLayers:     2,
		NumHeads:      1,
		SeqLen:        tokenCount,
		HeadDim:       1,
		NumQueryHeads: 1,
		Layers: []metal.KVLayerSnapshot{
			{Layer: 0, CacheIndex: 0, Heads: []metal.KVHeadSnapshot{{Key: key, Value: value}}},
			{Layer: 1, CacheIndex: 1, Heads: []metal.KVHeadSnapshot{{Key: key, Value: value}}},
		},
	}
}

// --- sessionParserControlToken ---
// Pure substring scan; fires per emitted token during GenerateStream
// + SessionGenerate. Three shapes — short control token, miss path,
// long miss path.

func BenchmarkSession_SessionParserControlToken_ControlHit(b *testing.B) {
	text := "<start_of_turn>"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sessionBenchSinkBool = sessionParserControlToken(text)
	}
}

func BenchmarkSession_SessionParserControlToken_Miss(b *testing.B) {
	text := "ordinary token text"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sessionBenchSinkBool = sessionParserControlToken(text)
	}
}

func BenchmarkSession_SessionParserControlToken_Empty(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sessionBenchSinkBool = sessionParserControlToken("")
	}
}

// --- sessionParserTokenText ---
// tok=nil drops to the token.Text fast path; this is the common case
// because the root tokenizer is only set when the session was built
// from a Model that loaded a tokenizer. Measure both branches.

func BenchmarkSession_SessionParserTokenText_NilTokenizer(b *testing.B) {
	tok := metal.Token{ID: 42, Text: "hello"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sessionBenchSinkText = sessionParserTokenText(nil, tok)
	}
}

// --- NewSession via fakeNativeModel ---
// Measures the wrap cost: type assertion + Info() copy + struct init.

func BenchmarkSession_NewSession(b *testing.B) {
	native := &fakeNativeSession{}
	model := &Model{model: &fakeNativeModel{session: native}}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sess, err := model.NewSession()
		sessionBenchSinkErr = err
		sessionBenchSinkSession = sess
	}
}

// --- Prefill / AppendPrompt — pure Go glue, fake native is a no-op ---

func BenchmarkSession_Prefill(b *testing.B) {
	native := &fakeNativeSession{}
	session := &ModelSession{session: native}
	prompt := "The quick brown fox jumps over the lazy dog."
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sessionBenchSinkErr = session.Prefill(prompt)
	}
}

func BenchmarkSession_AppendPrompt(b *testing.B) {
	native := &fakeNativeSession{}
	session := &ModelSession{session: native}
	prompt := "Another sentence appended."
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sessionBenchSinkErr = session.AppendPrompt(prompt)
	}
}

// --- PrefillChunks / AppendPromptChunks ---
// The fake implements nativeSessionChunkPrefiller/Appender, so this
// measures the iter.Seq dispatch + slice collection inside the fake.

func BenchmarkSession_PrefillChunks(b *testing.B) {
	native := &fakeNativeSession{}
	session := &ModelSession{session: native}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sessionBenchSinkErr = session.PrefillChunks(context.Background(), seqStrings("prefix ", "middle ", "suffix"))
	}
}

func BenchmarkSession_AppendPromptChunks(b *testing.B) {
	native := &fakeNativeSession{}
	session := &ModelSession{session: native}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sessionBenchSinkErr = session.AppendPromptChunks(context.Background(), seqStrings("chunk-a", "chunk-b"))
	}
}

// --- PrefillTokens / AppendTokens ---

func BenchmarkSession_PrefillTokens(b *testing.B) {
	native := &fakeNativeSession{}
	session := &ModelSession{session: native}
	tokens := make([]int32, 512)
	for i := range tokens {
		tokens[i] = int32(i + 1)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sessionBenchSinkErr = session.PrefillTokens(context.Background(), tokens)
	}
}

func BenchmarkSession_AppendTokens(b *testing.B) {
	native := &fakeNativeSession{}
	session := &ModelSession{session: native}
	tokens := make([]int32, 512)
	for i := range tokens {
		tokens[i] = int32(i + 1)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sessionBenchSinkErr = session.AppendTokens(context.Background(), tokens)
	}
}

// --- CaptureKV ---
// Goes through toRootKVSnapshot deep-copy of the fake KV.

func BenchmarkSession_CaptureKV_512Tokens(b *testing.B) {
	native := &fakeNativeSession{kv: benchSessionNativeKV(512)}
	session := &ModelSession{session: native}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sess, err := session.CaptureKV()
		sessionBenchSinkSnapshot = sess
		sessionBenchSinkErr = err
	}
}

func BenchmarkSession_CaptureKV_2048Tokens(b *testing.B) {
	native := &fakeNativeSession{kv: benchSessionNativeKV(2048)}
	session := &ModelSession{session: native}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sess, err := session.CaptureKV()
		sessionBenchSinkSnapshot = sess
		sessionBenchSinkErr = err
	}
}

// --- AnalyzeKV ---
// Capture + Analyze rolled together — the inner-loop diagnostic path.

func BenchmarkSession_AnalyzeKV_512Tokens(b *testing.B) {
	native := &fakeNativeSession{kv: benchSessionNativeKV(512)}
	session := &ModelSession{session: native}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		analysis, err := session.AnalyzeKV()
		sessionBenchSinkAnalysis = analysis
		sessionBenchSinkErr = err
	}
}

// --- SaveKV / LoadKV roundtrip ---

func BenchmarkSession_SaveKV_512Tokens(b *testing.B) {
	native := &fakeNativeSession{kv: benchSessionNativeKV(512)}
	session := &ModelSession{session: native}
	dir := b.TempDir()
	path := core.JoinPath(dir, "snap.kvbin")
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sessionBenchSinkErr = session.SaveKV(path)
	}
}

func BenchmarkSession_LoadKV_512Tokens(b *testing.B) {
	native := &fakeNativeSession{kv: benchSessionNativeKV(512)}
	session := &ModelSession{session: native}
	dir := b.TempDir()
	path := core.JoinPath(dir, "snap.kvbin")
	if err := session.SaveKV(path); err != nil {
		b.Fatal(err)
	}
	restoreNative := &fakeNativeSession{}
	restoreSession := &ModelSession{session: restoreNative}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sessionBenchSinkErr = restoreSession.LoadKV(path)
	}
}

// --- RestoreKV (no IO — the inner restoration call) ---

func BenchmarkSession_RestoreKV_512Tokens(b *testing.B) {
	snapshot := toRootKVSnapshot(benchSessionNativeKV(512))
	native := &fakeNativeSession{}
	session := &ModelSession{session: native}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sessionBenchSinkErr = session.RestoreKV(snapshot)
	}
}

// --- Fork — exercises the agent-memory clone path ---

func BenchmarkSession_Fork(b *testing.B) {
	forked := &fakeNativeSession{}
	native := &fakeNativeSession{forked: forked}
	session := &ModelSession{
		session: native,
		info:    ModelInfo{Architecture: "qwen3"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sess, err := session.Fork()
		sessionBenchSinkSession = sess
		sessionBenchSinkErr = err
	}
}

// --- Reset / Err ---

func BenchmarkSession_Reset(b *testing.B) {
	native := &fakeNativeSession{}
	session := &ModelSession{session: native}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		session.Reset()
	}
}

func BenchmarkSession_Err(b *testing.B) {
	native := &fakeNativeSession{}
	session := &ModelSession{session: native}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sessionBenchSinkErr = session.Err()
	}
}

// --- Nil-guard fast paths ---
// Useful for callers that pass nil/closed sessions defensively; the
// short-circuit happens BEFORE any native dispatch.

func BenchmarkSession_NilGuard_Prefill(b *testing.B) {
	var session *ModelSession
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sessionBenchSinkErr = session.Prefill("ignored")
	}
}

func BenchmarkSession_NilGuard_Reset(b *testing.B) {
	var session *ModelSession
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		session.Reset()
	}
}

func BenchmarkSession_NilGuard_Close(b *testing.B) {
	var session *ModelSession
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sessionBenchSinkErr = session.Close()
	}
}

// --- RestoreBundle ---
// Sanity-check the compatibility-check + snapshot extraction path.

func BenchmarkSession_RestoreBundle(b *testing.B) {
	snapshot := toRootKVSnapshot(benchSessionNativeKV(256))
	bundleObj := &bundle.Bundle{
		Version: bundle.Version,
		Kind:    bundle.Kind,
		Model: bundle.Model{
			Architecture: "qwen3",
			NumLayers:    2,
		},
		KV: snapshot,
	}
	native := &fakeNativeSession{}
	session := &ModelSession{
		session: native,
		info:    ModelInfo{Architecture: "qwen3", NumLayers: 2},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sessionBenchSinkErr = session.RestoreBundle(bundleObj)
	}
}
