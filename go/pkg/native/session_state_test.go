// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"encoding/binary"
	"testing"

	"dappco.re/go/mlx/pkg/model"
	g4 "dappco.re/go/mlx/pkg/model/gemma4"
)

func sessionStateFixture(t testing.TB) (*BF16Model, model.Arch, int) {
	t.Helper()
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
	return g, arch, maxLen
}

func newSessionStateFixture(t testing.TB) *ArchSession {
	t.Helper()
	g, arch, maxLen := sessionStateFixture(t)
	s, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchSession: %v", err)
	}
	return s
}

func icbSessionStateFixture(t testing.TB) (*QuantModel, model.Arch, int) {
	t.Helper()
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 32
	const numLayers, pliDim, gs, bits = 2, 64, 64, 4
	const maxLen = 24
	cfg := g4.Config{
		HiddenSize: dModel, NumHiddenLayers: numLayers, IntermediateSize: dFF,
		NumAttentionHeads: nHeads, NumKeyValueHeads: nKV, HeadDim: headDim,
		VocabSize: vocab, RMSNormEps: 1e-6,
		HiddenSizePerLayerInput: pliDim, VocabSizePerLayerInput: vocab,
		Quantization: &model.QuantConfig{GroupSize: gs, Bits: bits},
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	ts := quantGemma4Tensors(t, arch, gs, bits)
	addPLETensors(t, ts, arch, gs, bits)
	lm, err := model.Assemble(ts, arch, model.StandardWeightNames())
	if err != nil {
		t.Fatalf("model.Assemble: %v", err)
	}
	g, err := loadedToQuant(lm, gs, bits)
	if err != nil {
		t.Fatalf("loadedToQuant: %v", err)
	}
	return g, arch, maxLen
}

func newICBSessionStateFixture(t testing.TB, g *QuantModel, arch model.Arch, maxLen int) *ArchSession {
	t.Helper()
	s, err := NewArchQuantSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewArchQuantSession: %v", err)
	}
	if s.state.icb == nil {
		t.Fatal("fixture must build an ICB replay session")
	}
	return s
}

func emptySessionStateBlob(pos, layers, cachedIDs int) []byte {
	blob := make([]byte, 12+4+4*cachedIDs)
	binary.LittleEndian.PutUint32(blob[0:], sessionStateMagic)
	binary.LittleEndian.PutUint32(blob[4:], uint32(pos))
	binary.LittleEndian.PutUint32(blob[8:], uint32(layers))
	binary.LittleEndian.PutUint32(blob[12:], uint32(cachedIDs))
	for i := 0; i < cachedIDs; i++ {
		binary.LittleEndian.PutUint32(blob[16+4*i:], uint32(i+1))
	}
	return blob
}

func TestSessionStateNoRuntimeValidation(t *testing.T) {
	icbSession := &ArchSession{state: archDecodeState{icb: &archICBReplay{}}}
	if _, err := icbSession.SerializeState(); err != nil {
		t.Fatalf("SerializeState(empty ICB) error = %v", err)
	}
	if err := icbSession.RestoreState(emptySessionStateBlob(0, 0, 0)); err != nil {
		t.Fatalf("RestoreState(empty ICB) error = %v", err)
	}

	if err := (&ArchSession{}).RestoreState(nil); err == nil {
		t.Fatal("RestoreState(nil) error = nil")
	}
	if err := (&ArchSession{}).RestoreState(emptySessionStateBlob(0, 1, 0)); err == nil {
		t.Fatal("RestoreState(layer mismatch) error = nil")
	}

	legacy := make([]byte, 12)
	binary.LittleEndian.PutUint32(legacy[0:], sessionStateMagic)
	binary.LittleEndian.PutUint32(legacy[4:], 7)
	if err := (&ArchSession{}).RestoreState(legacy); err != nil {
		t.Fatalf("RestoreState(legacy snapshot) error = %v", err)
	}

	if err := (&ArchSession{}).RestoreState(append(legacy, 0)); err == nil {
		t.Fatal("RestoreState(truncated metadata length) error = nil")
	}
	truncatedIDs := emptySessionStateBlob(0, 0, 1)[:16]
	if err := (&ArchSession{}).RestoreState(truncatedIDs); err == nil {
		t.Fatal("RestoreState(truncated metadata ids) error = nil")
	}
	trailing := append(emptySessionStateBlob(0, 0, 1), 0)
	if err := (&ArchSession{}).RestoreState(trailing); err == nil {
		t.Fatal("RestoreState(trailing metadata) error = nil")
	}
}

func TestSessionStateSerializeZeroLayerCachedIDs(t *testing.T) {
	saved := &ArchSession{pos: 3, cachedIDs: []int32{7, 8, 9}}
	blob, err := saved.SerializeState()
	if err != nil {
		t.Fatalf("SerializeState: %v", err)
	}

	restored := &ArchSession{}
	if err := restored.RestoreState(blob); err != nil {
		t.Fatalf("RestoreState: %v", err)
	}
	if restored.Pos() != saved.Pos() {
		t.Fatalf("restored pos = %d, want %d", restored.Pos(), saved.Pos())
	}
	next := []int32{7, 8, 9, 10}
	if got := restored.CachedPrefixLen(next); got != len(saved.cachedIDs) {
		t.Fatalf("restored cached prefix = %d, want %d", got, len(saved.cachedIDs))
	}
}

func TestSessionStateRestoresPromptCacheEntry(t *testing.T) {
	requireNativeRuntime(t)
	prompt := []int32{1, 2, 3, 4, 5}

	a := newSessionStateFixture(t)
	if err := a.WarmPromptCache(prompt); err != nil {
		t.Fatalf("WarmPromptCache: %v", err)
	}
	blob, err := a.SerializeState()
	if err != nil {
		t.Fatalf("SerializeState: %v", err)
	}

	b := newSessionStateFixture(t)
	if err := b.RestoreState(blob); err != nil {
		t.Fatalf("RestoreState: %v", err)
	}
	if hit := b.CachedPrefixLen(prompt); hit != len(prompt) {
		t.Fatalf("restored exact prompt-cache hit = %d, want %d", hit, len(prompt))
	}
	head := b.head
	headCalls := 0
	b.greedy = nil
	b.head = func(hidden []byte, skipSoftcap bool) ([]byte, error) {
		headCalls++
		return head(hidden, skipSoftcap)
	}
	got, err := b.GenerateCached(prompt, 3, -1)
	if err != nil {
		t.Fatalf("GenerateCached after RestoreState: %v", err)
	}
	if headCalls != len(got)-1 {
		t.Fatalf("restored exact prompt-cache head calls = %d, want %d", headCalls, len(got)-1)
	}

	cold := newSessionStateFixture(t)
	want, err := cold.Generate(prompt, 3, -1)
	if err != nil {
		t.Fatalf("cold Generate: %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("generated length = %d, want %d", len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("token %d after restored prompt-cache entry = %d, want %d", i, got[i], want[i])
		}
	}
}

// TestSessionStateRoundTrip proves native conversation continuity: a session is decoded, snapshotted
// with SerializeState, and the snapshot is RestoreState'd into a FRESH session — which then continues
// the conversation TOKEN-IDENTICALLY to the original. This is save/resume across a process restart with
// no cgo, the no-cgo equivalent of metal's EnableConversationContinuity.
func TestSessionStateRoundTrip(t *testing.T) {
	requireNativeRuntime(t)

	// session A: decode a first turn, then snapshot.
	a := newSessionStateFixture(t)
	if _, err := a.Generate([]int32{1, 2, 3, 4, 5}, 6, -1); err != nil {
		t.Fatalf("A turn 1: %v", err)
	}
	blob, err := a.SerializeState()
	if err != nil {
		t.Fatalf("SerializeState: %v", err)
	}

	// session B: fresh, restore A's snapshot.
	b := newSessionStateFixture(t)
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

func TestSessionStateRoundTripICBReplay(t *testing.T) {
	requireNativeRuntime(t)
	g, arch, maxLen := icbSessionStateFixture(t)

	a := newICBSessionStateFixture(t, g, arch, maxLen)
	if _, err := a.Generate([]int32{1, 5, 3, 2}, 4, -1); err != nil {
		t.Fatalf("A turn 1: %v", err)
	}
	blob, err := a.SerializeState()
	if err != nil {
		t.Fatalf("SerializeState ICB: %v", err)
	}

	b := newICBSessionStateFixture(t, g, arch, maxLen)
	if err := b.RestoreState(blob); err != nil {
		t.Fatalf("RestoreState ICB: %v", err)
	}
	if b.Pos() != a.Pos() {
		t.Fatalf("restored ICB pos %d != saved pos %d", b.Pos(), a.Pos())
	}

	cont := []int32{7, 8}
	genA, err := a.Generate(cont, 5, -1)
	if err != nil {
		t.Fatalf("A turn 2: %v", err)
	}
	genB, err := b.Generate(cont, 5, -1)
	if err != nil {
		t.Fatalf("B turn 2: %v", err)
	}
	if len(genA) != len(genB) {
		t.Fatalf("ICB continuation length mismatch: A=%d B=%d", len(genA), len(genB))
	}
	for i := range genA {
		if genA[i] != genB[i] {
			t.Fatalf("ICB token %d diverged after restore: A=%d B=%d", i, genA[i], genB[i])
		}
	}
}

// TestSessionStateRoundTripRestoresCachedPrefixMetadata proves state restore
// preserves the prompt-cache metadata that lets GenerateCached reuse resident KV
// rows. Token parity alone is insufficient here: a restored session can produce
// the same tokens by cold re-prefilling, but then the native engine has lost the
// resource-saving prefix hit that metal's prompt-cache restore path preserves.
func TestSessionStateRoundTripRestoresCachedPrefixMetadata(t *testing.T) {
	requireNativeRuntime(t)
	a := newSessionStateFixture(t)
	prompt := []int32{1, 2, 3, 4, 5}
	if _, err := a.GenerateCached(prompt, 6, -1); err != nil {
		t.Fatalf("GenerateCached warmup: %v", err)
	}
	nextPrompt := []int32{1, 2, 3, 4, 5, 6}
	wantHit := a.CachedPrefixLen(nextPrompt)
	if wantHit != len(prompt) {
		t.Fatalf("warm CachedPrefixLen = %d, want %d", wantHit, len(prompt))
	}
	blob, err := a.SerializeState()
	if err != nil {
		t.Fatalf("SerializeState: %v", err)
	}

	b := newSessionStateFixture(t)
	if err := b.RestoreState(blob); err != nil {
		t.Fatalf("RestoreState: %v", err)
	}
	if got := b.CachedPrefixLen(nextPrompt); got != wantHit {
		t.Fatalf("restored CachedPrefixLen = %d, want %d", got, wantHit)
	}
}

func TestSessionStateSerializeCachedPrefixAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)
	s := newSessionStateFixture(t)
	if _, err := s.GenerateCached([]int32{1, 2, 3, 4, 5}, 6, -1); err != nil {
		t.Fatalf("GenerateCached warmup: %v", err)
	}
	allocs := testing.AllocsPerRun(20, func() {
		if _, err := s.SerializeState(); err != nil {
			t.Fatalf("SerializeState: %v", err)
		}
	})
	if allocs > 82 {
		t.Fatalf("SerializeState allocations = %.0f, want <= 82", allocs)
	}
}
