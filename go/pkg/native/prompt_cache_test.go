// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"

	"dappco.re/go/mlx/pkg/model"
)

// TestGenerateCachedPrefixReuse proves native prompt caching: after a first turn warms the cache, a
// second prompt that shares a prefix is served by reusing that prefix's KV (re-prefilling only the
// suffix) and produces a TOKEN-IDENTICAL continuation to a cold Generate of the same full prompt — while
// CachedPrefixLen confirms the shared prefix was actually skipped.
func TestGenerateCachedPrefixReuse(t *testing.T) {
	requireNativeRuntime(t)
	const dModel, nHeads, nKV, headDim, dFF = 512, 8, 4, 64, 1024
	const vocab, nL, maxLen = 64, 3, 96
	layers := make([]DecodeLayerWeights, nL)
	types := make([]string, nL)
	for li := range layers {
		layers[li] = forwardLayer(dModel, nHeads, nKV, headDim, dFF, (li+1)*100)
		types[li] = "full_attention"
	}
	specs := model.DeriveLayers(types, 0)
	embed := toBF16Bytes(syntheticFloat32(vocab*dModel, 21))
	g := &BF16Model{Layers: layers, Embed: embed, FinalNorm: toBF16Bytes(syntheticFloat32(dModel, 22)), LMHead: embed, Tied: true}
	arch := model.Arch{
		Hidden: dModel, Heads: nHeads, KVHeads: nKV, HeadDim: headDim, FF: dFF, Vocab: vocab,
		GlobalHeadDim: headDim, GlobalKVHeads: nKV,
		Eps: 1e-5, AttnScale: 0.125, RopeBase: 10000, RopeScale: 1, RopeLocalBase: 10000,
		RotaryDim: headDim, RotaryDimLocal: headDim, Layer: specs,
	}
	mk := func() *ArchSession {
		s, err := NewArchSession(g, arch, maxLen)
		if err != nil {
			t.Fatalf("NewArchSession: %v", err)
		}
		return s
	}

	shared := []int32{1, 2, 3, 4, 5}
	full := []int32{1, 2, 3, 4, 5, 6, 7} // extends `shared`

	// cold reference: a fresh session decodes the full prompt.
	cold, err := mk().Generate(full, 8, -1)
	if err != nil {
		t.Fatalf("cold Generate: %v", err)
	}

	// warm session: a first turn on `shared`, then GenerateCached on `full` reuses the shared prefix.
	warm := mk()
	if _, err := warm.GenerateCached(shared, 6, -1); err != nil {
		t.Fatalf("warm turn 1: %v", err)
	}
	hit := warm.CachedPrefixLen(full)
	if hit != len(shared) {
		t.Fatalf("prompt-cache prefix hit = %d, want %d (the shared prefix)", hit, len(shared))
	}
	got, err := warm.GenerateCached(full, 8, -1)
	if err != nil {
		t.Fatalf("warm turn 2: %v", err)
	}
	if len(got) != len(cold) {
		t.Fatalf("length mismatch: warm=%d cold=%d", len(got), len(cold))
	}
	for i := range cold {
		if got[i] != cold[i] {
			t.Fatalf("token %d diverged: warm(cached)=%d cold=%d", i, got[i], cold[i])
		}
	}
	t.Logf("native prompt cache: reused %d-token prefix, continuation token-identical to a cold run over %d tokens", hit, len(got))
}

// TestCompactCacheContinuation proves cache compaction is correct: after decoding a sequence and
// compacting to the most recent `keep` tokens, the session continues TOKEN-IDENTICALLY to a fresh
// session prefilled with exactly those kept tokens (the eviction + re-prefill re-rotates RoPE correctly).
func TestCompactCacheContinuation(t *testing.T) {
	requireNativeRuntime(t)
	const dModel, nHeads, nKV, headDim, dFF = 512, 8, 4, 64, 1024
	const vocab, nL, maxLen, keep = 64, 3, 96, 4
	layers := make([]DecodeLayerWeights, nL)
	types := make([]string, nL)
	for li := range layers {
		layers[li] = forwardLayer(dModel, nHeads, nKV, headDim, dFF, (li+1)*100)
		types[li] = "full_attention"
	}
	specs := model.DeriveLayers(types, 0)
	embed := toBF16Bytes(syntheticFloat32(vocab*dModel, 21))
	g := &BF16Model{Layers: layers, Embed: embed, FinalNorm: toBF16Bytes(syntheticFloat32(dModel, 22)), LMHead: embed, Tied: true}
	arch := model.Arch{
		Hidden: dModel, Heads: nHeads, KVHeads: nKV, HeadDim: headDim, FF: dFF, Vocab: vocab,
		GlobalHeadDim: headDim, GlobalKVHeads: nKV,
		Eps: 1e-5, AttnScale: 0.125, RopeBase: 10000, RopeScale: 1, RopeLocalBase: 10000,
		RotaryDim: headDim, RotaryDimLocal: headDim, Layer: specs,
	}
	mk := func() *ArchSession {
		s, err := NewArchSession(g, arch, maxLen)
		if err != nil {
			t.Fatalf("NewArchSession: %v", err)
		}
		return s
	}

	// session A: decode a long-ish sequence, then compact to the most recent `keep` tokens.
	a := mk()
	if _, err := a.GenerateCached([]int32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, 6, -1); err != nil {
		t.Fatalf("A turn: %v", err)
	}
	resident := append([]int32(nil), a.cachedIDs...)
	kept := resident[len(resident)-keep:]
	if err := a.CompactCache(keep); err != nil {
		t.Fatalf("CompactCache: %v", err)
	}
	if a.Pos() != keep {
		t.Fatalf("post-compaction pos = %d, want %d", a.Pos(), keep)
	}
	cont := []int32{30, 31}
	genA, err := a.Generate(cont, 8, -1)
	if err != nil {
		t.Fatalf("A continue: %v", err)
	}

	// reference: a fresh session prefilled with exactly the kept tokens, same continuation.
	b := mk()
	full := append(append([]int32(nil), kept...), cont...)
	genB, err := b.Generate(full, 8, -1)
	if err != nil {
		t.Fatalf("B: %v", err)
	}
	if len(genA) != len(genB) {
		t.Fatalf("length mismatch: A=%d B=%d", len(genA), len(genB))
	}
	for i := range genA {
		if genA[i] != genB[i] {
			t.Fatalf("token %d diverged after compaction: A=%d B=%d", i, genA[i], genB[i])
		}
	}
	t.Logf("native cache compaction: kept %d recent tokens, continuation token-identical to a fresh session with that context", keep)
}
