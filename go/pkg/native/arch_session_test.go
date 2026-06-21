// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package native

import (
	"os"
	"testing"

	g4 "dappco.re/go/mlx/pkg/model/gemma4"
)

func idsEqual(a, b []int32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// TestArchSession gates the persistent serving session: a second Generate continues the
// running sequence from the carried-over cache, and its output is byte-identical to a fresh
// whole-sequence generate on the concatenated history — which proves the resident caches
// SURVIVED across the constructor + per-call autorelease pools and that the continuation is
// correct. Plus: Pos tracks the sequence length, a fresh session reproduces it, and a third
// turn runs (the buffer lifetime holds across many calls).
func TestArchSession(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 32
	const maxLen = 32
	arch, err := g4.Config{
		HiddenSize: dModel, NumHiddenLayers: 2, IntermediateSize: dFF,
		NumAttentionHeads: nHeads, NumKeyValueHeads: nKV, HeadDim: headDim,
		VocabSize: vocab, RMSNormEps: 1e-6,
	}.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	mk := func(n, salt int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32((i*salt+13)%97-48) * 0.02
		}
		return s
	}
	layers := make([]DecodeLayerWeights, len(arch.Layer))
	for li := range layers {
		layers[li] = forwardLayer(dModel, nHeads, nKV, headDim, dFF, (li+1)*100)
	}
	g := &BF16Model{Layers: layers, Embed: toBF16Bytes(mk(vocab*dModel, 11)), FinalNorm: toBF16Bytes(mk(dModel, 7))}
	g.LMHead, g.Tied = g.Embed, true

	sess, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession: %v", err)
	}
	promptA := []int32{1, 5, 3}
	gA, err := sess.Generate(promptA, 3, -1)
	if err != nil {
		t.Fatalf("Generate A: %v", err)
	}
	if sess.Pos() != len(promptA)+len(gA) {
		t.Fatalf("Pos after turn 1 = %d, want %d", sess.Pos(), len(promptA)+len(gA))
	}
	promptB := []int32{7, 2}
	gB, err := sess.Generate(promptB, 4, -1)
	if err != nil {
		t.Fatalf("Generate B: %v", err)
	}
	if sess.Pos() != len(promptA)+len(gA)+len(promptB)+len(gB) {
		t.Fatalf("Pos after turn 2 = %d, want %d", sess.Pos(), len(promptA)+len(gA)+len(promptB)+len(gB))
	}

	// the continuation must equal a fresh whole-sequence generate on the full history.
	concat := append(append(append([]int32{}, promptA...), gA...), promptB...)
	ref, err := GenerateGemma4BF16(g, arch, concat, 4, maxLen, -1)
	if err != nil {
		t.Fatalf("reference GenerateGemma4BF16: %v", err)
	}
	if !idsEqual(gB, ref) {
		t.Fatalf("session continuation %v != fresh whole-sequence %v (cache did not carry over correctly)", gB, ref)
	}

	// a fresh session reproduces both turns (deterministic).
	sess2, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession 2: %v", err)
	}
	gA2, _ := sess2.Generate(promptA, 3, -1)
	gB2, _ := sess2.Generate(promptB, 4, -1)
	if !idsEqual(gA2, gA) || !idsEqual(gB2, gB) {
		t.Fatalf("non-deterministic across sessions: A %v vs %v, B %v vs %v", gA2, gA, gB2, gB)
	}

	// a third turn runs (buffer lifetime holds across many calls).
	gC, err := sess.Generate([]int32{9}, 3, -1)
	if err != nil {
		t.Fatalf("Generate C: %v", err)
	}
	if len(gC) != 3 || sess.Pos() != 16 {
		t.Fatalf("turn 3: got %d tokens, Pos %d (want 3, 16)", len(gC), sess.Pos())
	}

	t.Logf("session: turn1 %v → turn2 %v continues the cache (≡ fresh whole-sequence on the 8-token history), turn3 %v; Pos %d; deterministic — persistent KV cache survives across calls", gA, gB, gC, sess.Pos())
}
