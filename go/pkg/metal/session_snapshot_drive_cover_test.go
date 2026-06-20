// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"context"
	"testing"
)

// session_snapshot_drive_cover_test.go drives the session SNAPSHOT paths
// (CaptureKV with real layer state, Fork, restore) using a fake whose Forward
// populates the KV caches with synthetic [1,H,L,D] tensors — the step the plain
// logits-fake skipped. This reaches snapshotSessionCache's populated arms, the
// fork clone, and the capture layer-walk without a real model.

const snapHeads, snapHeadDim = 2, 4

// kvWritingFakeModel returns logits like logitsFakeModel but also writes the
// chunk's K/V into each cache so the session carries real snapshot state.
type kvWritingFakeModel struct {
	tok      *Tokenizer
	favoured int32
}

func (m *kvWritingFakeModel) seqLenOf(tokens *Array) int {
	if tokens == nil || !tokens.Valid() {
		return 1
	}
	switch n := tokens.NumDims(); {
	case n >= 2:
		return tokens.Dim(n - 1)
	default:
		return 1
	}
}

func (m *kvWritingFakeModel) writeCaches(tokens *Array, caches []Cache) {
	seqLen := m.seqLenOf(tokens)
	for _, c := range caches {
		if c == nil {
			continue
		}
		n := snapHeads * seqLen * snapHeadDim
		data := make([]float32, n)
		for i := range data {
			data[i] = float32(i) * 0.01
		}
		k := FromValues(data, 1, snapHeads, seqLen, snapHeadDim)
		v := FromValues(data, 1, snapHeads, seqLen, snapHeadDim)
		c.Update(k, v, seqLen)
		Free(k, v)
	}
}

func (m *kvWritingFakeModel) logits() *Array {
	vals := make([]float32, logitsFakeVocab)
	for i := range vals {
		vals[i] = -10
	}
	vals[m.favoured] = 100
	return FromValues(vals, 1, 1, logitsFakeVocab)
}

func (m *kvWritingFakeModel) Forward(tokens *Array, caches []Cache) *Array {
	m.writeCaches(tokens, caches)
	return m.logits()
}

func (m *kvWritingFakeModel) ForwardMasked(tokens *Array, _ *Array, caches []Cache) *Array {
	m.writeCaches(tokens, caches)
	return m.logits()
}
func (m *kvWritingFakeModel) NewCache() []Cache                   { return []Cache{NewKVCache()} }
func (m *kvWritingFakeModel) NumLayers() int                      { return 1 }
func (m *kvWritingFakeModel) Tokenizer() *Tokenizer               { return m.tok }
func (m *kvWritingFakeModel) ModelType() string                   { return "kv-writing-fake" }
func (m *kvWritingFakeModel) ApplyLoRA(_ LoRAConfig) *LoRAAdapter { return nil }

func newKVWritingSessionModel() *Model {
	return &Model{
		model:         &kvWritingFakeModel{tok: NewForDecode(nil), favoured: 3},
		modelType:     "kv-writing-fake",
		tokenizer:     NewForDecode(nil),
		parallelSlots: make(chan struct{}, 1),
	}
}

// CaptureKV over a session with populated caches walks the real layer state and
// produces a snapshot whose layer carries head tensors.
func TestSessionSnapshot_CaptureWithLayerState_Good(t *testing.T) {
	requireMetalRuntime(t)

	m := newKVWritingSessionModel()
	sess := m.NewSession().(*ModelSession)
	defer sess.Close()
	if err := sess.PrefillTokens(context.Background(), []int32{1, 2, 3, 4}); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}

	snap, err := sess.CaptureKV(context.Background())
	if err != nil {
		t.Fatalf("CaptureKV: %v", err)
	}
	if snap == nil || len(snap.Layers) == 0 {
		t.Fatalf("snapshot carried no layers: %+v", snap)
	}
	if len(snap.Tokens) != 4 {
		t.Errorf("snapshot tokens = %d, want 4", len(snap.Tokens))
	}
}

// Fork over populated KV caches clones the live state into an independent
// session that can keep decoding.
func TestSessionSnapshot_ForkPopulated_Good(t *testing.T) {
	requireMetalRuntime(t)

	m := newKVWritingSessionModel()
	sess := m.NewSession().(*ModelSession)
	defer sess.Close()
	if err := sess.PrefillTokens(context.Background(), []int32{1, 2, 3}); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}

	forked, err := sess.Fork(context.Background())
	if err != nil {
		t.Fatalf("Fork: %v", err)
	}
	defer forked.Close()

	var n int
	for range forked.Generate(context.Background(), GenerateConfig{MaxTokens: 2}) {
		n++
		if n >= 2 {
			break
		}
	}
	if err := forked.Err(); err != nil {
		t.Fatalf("forked session Err: %v", err)
	}
}

// Capture then restore the KV state into a fresh session — the round trip that
// exercises restoreKVCachesFromSnapshot's populated arms.
func TestSessionSnapshot_CaptureRestoreRoundTrip_Good(t *testing.T) {
	requireMetalRuntime(t)

	m := newKVWritingSessionModel()
	sess := m.NewSession().(*ModelSession)
	defer sess.Close()
	if err := sess.PrefillTokens(context.Background(), []int32{5, 6, 7}); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}
	snap, err := sess.CaptureKV(context.Background())
	if err != nil {
		t.Fatalf("CaptureKV: %v", err)
	}

	fresh := m.NewSession().(*ModelSession)
	defer fresh.Close()
	if err := fresh.RestoreKV(context.Background(), snap); err != nil {
		t.Fatalf("RestoreKV: %v", err)
	}
}
