// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"hash/fnv"
	"math"
	"testing"

	core "dappco.re/go"
)

// refTextHashEmbed is the pre-optimisation formula for the text-hash
// embedder, preserved verbatim as the characterisation oracle. The
// production memoryPretrainTextHashEmbedder must stay byte-identical to
// this: same FNV-1a(token ++ {lo,hi}) per (token, dimension), same
// L2-normalisation, same all-zero → out[0]=1 fallback. Only the
// allocation shape changes (one reused hasher + hoisted token bytes +
// stack salt instead of a fresh hasher + two []byte allocs per inner
// iteration).
func refTextHashEmbed(text string, dim int) []float32 {
	out := make([]float32, dim)
	for _, token := range core.Split(text, " ") {
		token = core.Trim(token)
		if token == "" {
			continue
		}
		for i := range out {
			h := fnv.New32a()
			_, _ = h.Write([]byte(token))
			_, _ = h.Write([]byte{byte(i), byte(i >> 8)})
			bucket := int(h.Sum32()%2001) - 1000
			out[i] += float32(bucket) / 1000
		}
	}
	var norm float64
	for _, value := range out {
		norm += float64(value * value)
	}
	if norm == 0 {
		out[0] = 1
		return out
	}
	scale := float32(1 / math.Sqrt(norm))
	for i := range out {
		out[i] *= scale
	}
	return out
}

func TestMemoryPretrainTextHashEmbedder_MatchesReference_Good(t *testing.T) {
	cases := []struct {
		text string
		dim  int
	}{
		{"hello world", 8},
		{"the quick brown fox jumps over", 16},
		{"single", 1},
		{"a a a b c", 32},
		{"   ", 4},                      // all-whitespace → every token trimmed away → norm==0 fallback
		{"", 4},                         // empty text → norm==0 fallback (out[0]=1)
		{"  spaced   out  tokens ", 12}, // irregular spacing exercises Split/Trim skips
	}
	ctx := context.Background()
	for _, tc := range cases {
		embed := memoryPretrainTextHashEmbedder(tc.dim)
		got, err := embed.Embed(ctx, tc.text)
		if err != nil {
			t.Fatalf("Embed(%q, %d) error = %v", tc.text, tc.dim, err)
		}
		want := refTextHashEmbed(tc.text, tc.dim)
		if len(got) != len(want) {
			t.Fatalf("Embed(%q, %d) len = %d, want %d", tc.text, tc.dim, len(got), len(want))
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("Embed(%q, %d) out[%d] = %v, want %v (full mismatch — optimisation drifted)",
					tc.text, tc.dim, i, got[i], want[i])
			}
		}
	}
}

// Baseline: the production embedder builds at ~4 allocs/op. It runs inside
// an Embedder-interface closure where the compiler can NOT stack-allocate a
// per-iteration fnv.New32a() (it escapes), so the naive inner-loop shape
// allocated ~3 × tokens × dim (measured 9218 allocs for a 12-token text at
// dim 256). The reused-hasher + zero-copy-token + stack-salt rewrite cut it
// to 4. NB: a STANDALONE copy of the naive formula benches at only ~2 allocs
// because the compiler inlines + stack-allocates it — do NOT use that as the
// baseline; the real path is this interface-dispatched closure. If this jumps
// back toward thousands, someone reverted the rewrite.
var memEmbedSink []float32

func BenchmarkMemoryPretrainTextHashEmbed_Build(b *testing.B) {
	text := "the quick brown fox jumps over the lazy dog again and again"
	embed := memoryPretrainTextHashEmbedder(256)
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		memEmbedSink, _ = embed.Embed(ctx, text)
	}
}
