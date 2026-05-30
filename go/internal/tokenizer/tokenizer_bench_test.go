// SPDX-Licence-Identifier: EUPL-1.2

package tokenizer

import (
	"testing"

	"dappco.re/go"

	coreio "dappco.re/go/io"
)

// benchTokenizer builds the production internal/tokenizer.Tokenizer from the
// minimal SentencePiece-style fixture. Mirrors writeTestTokenizer but for the
// testing.B path (no *testing.T helper available).
func benchTokenizer(b *testing.B) *Tokenizer {
	b.Helper()
	dir := b.TempDir()
	path := core.JoinPath(dir, "tokenizer.json")
	if err := coreio.Local.Write(path, minimalTokenizerJSON); err != nil {
		b.Fatalf("write bench tokenizer: %v", err)
	}
	tok, err := LoadTokenizer(path)
	if err != nil {
		b.Fatalf("load bench tokenizer: %v", err)
	}
	return tok
}

// BenchmarkDecodeOne_SentencePiece measures the per-emitted-token decode the
// generation loop hits once per token via tokenizer_common.go:97. Watch
// allocs/op: core.Replace allocates a fresh string per call even when no "▁"
// marker is present.
func BenchmarkDecodeOne_SentencePiece(b *testing.B) {
	tok := benchTokenizer(b)
	// id 5 == "he" (no SentencePiece marker — the common mid-word case)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = tok.DecodeOne(5)
	}
}

// BenchmarkDecodeOne_WordBoundary exercises the leading-space path (id 7 ==
// "▁h") — the marker IS present, so the Replace + prefix-strip both fire.
func BenchmarkDecodeOne_WordBoundary(b *testing.B) {
	tok := benchTokenizer(b)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = tok.DecodeOne(7)
	}
}

// BenchmarkDecodeToken_Streaming is the streaming sibling that keeps the
// leading space. Same Replace cost without the strip.
func BenchmarkDecodeToken_Streaming(b *testing.B) {
	tok := benchTokenizer(b)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = tok.DecodeToken(7)
	}
}

// BenchmarkEncode_Short measures the prompt-processing path — Encode runs the
// segment split + BPE merge + cache lookup. Cold cache on first call, warm
// thereafter (the cache is shared across iterations here, so this measures
// the warm-cache fast path).
func BenchmarkEncode_Short(b *testing.B) {
	tok := benchTokenizer(b)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = tok.Encode("hello")
	}
}

// BenchmarkBPEMerge_ColdSegment isolates the O(n²) merge scan on a fresh
// symbol slice — the per-pair string concat (symbols[i]+" "+symbols[i+1])
// allocates inside the inner loop on every rank lookup.
func BenchmarkBPEMerge_ColdSegment(b *testing.B) {
	tok := benchTokenizer(b)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		symbols := []string{"h", "e", "l", "l", "o"}
		_ = tok.bpeMerge(symbols)
	}
}
