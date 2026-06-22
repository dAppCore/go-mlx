// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"

	"dappco.re/go/mlx/pkg/model"
)

// TestSessionStateRoundTrip proves native conversation continuity: a session is decoded, snapshotted
// with SerializeState, and the snapshot is RestoreState'd into a FRESH session — which then continues
// the conversation TOKEN-IDENTICALLY to the original. This is save/resume across a process restart with
// no cgo, the no-cgo equivalent of metal's EnableConversationContinuity.
func TestSessionStateRoundTrip(t *testing.T) {
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

	// session A: decode a first turn, then snapshot.
	a := mk()
	if _, err := a.Generate([]int32{1, 2, 3, 4, 5}, 6, -1); err != nil {
		t.Fatalf("A turn 1: %v", err)
	}
	blob, err := a.SerializeState()
	if err != nil {
		t.Fatalf("SerializeState: %v", err)
	}

	// session B: fresh, restore A's snapshot.
	b := mk()
	if err := b.RestoreState(blob); err != nil {
		t.Fatalf("RestoreState: %v", err)
	}
	if b.Pos() != a.Pos() {
		t.Fatalf("restored pos %d != saved pos %d", b.Pos(), a.Pos())
	}

	// both continue the conversation with the same next turn — must produce identical tokens.
	cont := []int32{20, 21, 22}
	genA, err := a.Generate(cont, 8, -1)
	if err != nil {
		t.Fatalf("A turn 2: %v", err)
	}
	genB, err := b.Generate(cont, 8, -1)
	if err != nil {
		t.Fatalf("B turn 2: %v", err)
	}
	if len(genA) != len(genB) {
		t.Fatalf("continuation length mismatch: A=%d B=%d", len(genA), len(genB))
	}
	for i := range genA {
		if genA[i] != genB[i] {
			t.Fatalf("token %d diverged after restore: A=%d B=%d", i, genA[i], genB[i])
		}
	}
	t.Logf("native continuity: serialize→restore→continue is token-identical over %d continuation tokens (snapshot %d bytes)", len(genA), len(blob))
}
