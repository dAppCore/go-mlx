// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for KV snapshot save/load + analysis primitives.
// Per AX-11 — Snapshot.Save fires per generation step (checkpointing);
// LoadWithOptions fires per session resume; Analyze runs on every
// resumed snapshot. The binary encoder (bytes / writeWithOptions)
// is the inner loop both Save and SaveStateBlocks hit.
//
// Run:    go test -bench='BenchmarkSnapshot|BenchmarkAnalyze|BenchmarkHash' -benchmem -run='^$' ./go/kv

package kv

import (
	"bytes"
	"context"
	"testing"

	core "dappco.re/go"
	state "dappco.re/go/inference/state"
)

// Sinks defeat compiler DCE.
var (
	benchSinkSnapshot *Snapshot
	benchSinkBytes    []byte
	benchSinkErr      error
	benchSinkString   string
	benchSinkAnalysis *Analysis
	benchSinkRef      state.ChunkRef
)

// benchSnapshot builds a representative snapshot — token count and
// layer/head shape sized to the qwen3-class range. Same fixture
// helper as the existing block-loading benches but exposed at file
// scope so the new save/load benches can share it.
func benchSnapshot(tokenCount int) *Snapshot {
	tokens := make([]int32, tokenCount)
	fullKey := make([]float32, tokenCount)
	fullValue := make([]float32, tokenCount)
	for i := range tokenCount {
		tokens[i] = int32(i + 1)
		fullKey[i] = float32(i)
		fullValue[i] = float32(i + 1000)
	}
	return &Snapshot{
		Version:       SnapshotVersion,
		Architecture:  "qwen3",
		Tokens:        tokens,
		TokenOffset:   tokenCount,
		NumLayers:     2,
		NumHeads:      1,
		SeqLen:        tokenCount,
		HeadDim:       1,
		NumQueryHeads: 1,
		Layers: []LayerSnapshot{
			{Layer: 0, CacheIndex: 0, Heads: []HeadSnapshot{{Key: fullKey, Value: fullValue}}},
			{Layer: 1, CacheIndex: 1, Heads: []HeadSnapshot{{Key: fullKey, Value: fullValue}}},
		},
	}
}

// --- Save / SaveWithOptions ---

func BenchmarkSnapshot_Save_512Tokens(b *testing.B) {
	dir := b.TempDir()
	snap := benchSnapshot(512)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkErr = snap.Save(core.JoinPath(dir, "snap.bin"))
	}
}

func BenchmarkSnapshot_Save_2048Tokens(b *testing.B) {
	dir := b.TempDir()
	snap := benchSnapshot(2048)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkErr = snap.Save(core.JoinPath(dir, "snap.bin"))
	}
}

// --- Encoder hot path: bytes() in-memory (no disk IO) ---

func BenchmarkSnapshot_Bytes_512Tokens(b *testing.B) {
	snap := benchSnapshot(512)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkBytes, benchSinkErr = snap.bytes()
	}
}

func BenchmarkSnapshot_Bytes_2048Tokens(b *testing.B) {
	snap := benchSnapshot(2048)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkBytes, benchSinkErr = snap.bytes()
	}
}

// --- writeWithOptions to a discarding writer (isolates the encoder
// from the alloc-the-return-slice cost in bytes()) ---

func BenchmarkSnapshot_WriteWithOptions_2048Tokens(b *testing.B) {
	snap := benchSnapshot(2048)
	var buf bytes.Buffer
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		buf.Reset()
		benchSinkErr = snap.writeWithOptions(&buf, SaveOptions{})
	}
}

// --- Load (full roundtrip) ---

func BenchmarkSnapshot_Load_512Tokens(b *testing.B) {
	dir := b.TempDir()
	path := core.JoinPath(dir, "snap.bin")
	if err := benchSnapshot(512).Save(path); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkSnapshot, benchSinkErr = Load(path)
	}
}

// --- Analyze ---

func BenchmarkAnalyze_512Tokens(b *testing.B) {
	snap := benchSnapshot(512)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkAnalysis = Analyze(snap)
	}
}

func BenchmarkAnalyze_2048Tokens(b *testing.B) {
	snap := benchSnapshot(2048)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkAnalysis = Analyze(snap)
	}
}

// --- HashSnapshot ---

func BenchmarkHashSnapshot_512Tokens(b *testing.B) {
	snap := benchSnapshot(512)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkString, benchSinkErr = HashSnapshot(snap)
	}
}

// --- SaveStateBlocks (the chunked-write path the existing
// block-load benches resolve from) ---

func BenchmarkSnapshot_SaveStateBlocks_3Blocks(b *testing.B) {
	store := state.NewInMemoryStore(nil)
	snap := benchSnapshot(1536) // 3 × 512-block
	opts := StateBlockOptions{BlockSize: 512, KVEncoding: EncodingNative}
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bundle, err := snap.SaveStateBlocks(ctx, store, opts)
		benchSinkErr = err
		if bundle != nil && len(bundle.Blocks) > 0 {
			benchSinkRef = bundle.Blocks[0].State
		}
	}
}
