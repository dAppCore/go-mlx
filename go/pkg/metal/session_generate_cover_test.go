// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package metal

import (
	"context"
	"testing"
)

// session_generate_cover_test.go drives the session prefill + decode loop with a
// synthetic model whose Forward returns a fixed [1,1,vocab] logits tensor — no
// weights, no real architecture. This reaches generateLocked / advanceTokenLocked
// / the prefill block / metrics assembly that the model-eval suites otherwise own.
// The fake deliberately does NOT implement LastTokenLogitsModel, so the forward
// dispatch takes the plain Forward arm predictably.

const logitsFakeVocab = 8

// logitsFakeModel returns a deterministic logits tensor favouring a fixed token
// so greedy decode is reproducible. It implements only the base InternalModel
// contract.
type logitsFakeModel struct {
	tok      *Tokenizer
	favoured int32
}

func (m *logitsFakeModel) logits() *Array {
	vals := make([]float32, logitsFakeVocab)
	for i := range vals {
		vals[i] = -10
	}
	vals[m.favoured] = 100
	return FromValues(vals, 1, 1, logitsFakeVocab)
}

func (m *logitsFakeModel) Forward(_ *Array, _ []Cache) *Array               { return m.logits() }
func (m *logitsFakeModel) ForwardMasked(_ *Array, _ *Array, _ []Cache) *Array { return m.logits() }
func (m *logitsFakeModel) NewCache() []Cache                                { return []Cache{NewKVCache()} }
func (m *logitsFakeModel) NumLayers() int                                   { return 1 }
func (m *logitsFakeModel) Tokenizer() *Tokenizer                            { return m.tok }
func (m *logitsFakeModel) ModelType() string                               { return "logits-fake" }
func (m *logitsFakeModel) ApplyLoRA(_ LoRAConfig) *LoRAAdapter              { return nil }

func newLogitsFakeSessionModel() *Model {
	return &Model{
		model:         &logitsFakeModel{tok: NewForDecode(nil), favoured: 3},
		modelType:     "logits-fake",
		tokenizer:     NewForDecode(nil),
		parallelSlots: make(chan struct{}, 1),
	}
}

// Prefill a few tokens, then greedily decode a bounded number — exercising the
// session prefill block, the decode loop, advanceTokenLocked, and metrics.
func TestSessionGenerate_GreedyDecodeLoop_Good(t *testing.T) {
	requireMetalRuntime(t)

	m := newLogitsFakeSessionModel()
	sess := m.NewSession().(*ModelSession)
	defer sess.Close()

	if err := sess.PrefillTokens(context.Background(), []int32{1, 2, 3}); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}

	var got []Token
	for tok := range sess.Generate(context.Background(), GenerateConfig{MaxTokens: 4}) {
		got = append(got, tok)
		if len(got) >= 4 {
			break
		}
	}
	if err := sess.Err(); err != nil {
		t.Fatalf("session Err after generate: %v", err)
	}
	if len(got) == 0 {
		t.Fatal("greedy decode yielded no tokens")
	}
	// The fake always favours token 3, so greedy decode emits it.
	for i, tok := range got {
		if tok.ID != 3 {
			t.Errorf("token %d = %d, want 3 (favoured)", i, tok.ID)
		}
	}
	// Metrics should have been assembled in the generateLocked defer.
	if m.LastMetrics().GeneratedTokens == 0 {
		t.Error("LastMetrics reports zero generated tokens")
	}
}

// Decode under a repeat penalty exercises the history-tracking branch of
// generateLocked (cfg.RepeatPenalty > 1 seeds and threads the token history).
func TestSessionGenerate_RepeatPenaltyHistory_Good(t *testing.T) {
	requireMetalRuntime(t)

	m := newLogitsFakeSessionModel()
	sess := m.NewSession().(*ModelSession)
	defer sess.Close()
	if err := sess.PrefillTokens(context.Background(), []int32{4, 5}); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}

	var n int
	for range sess.Generate(context.Background(), GenerateConfig{MaxTokens: 3, RepeatPenalty: 1.2, Temperature: 0.8, TopK: 50, TopP: 0.95}) {
		n++
		if n >= 3 {
			break
		}
	}
	if err := sess.Err(); err != nil {
		t.Fatalf("session Err under repeat penalty: %v", err)
	}
	if n == 0 {
		t.Error("decode under repeat penalty yielded nothing")
	}
}

// A second prefill+decode on a fresh session via the token path with a stop
// token set drives the min-tokens-before-stop suppression-sampler branch.
func TestSessionGenerate_StopTokenWindow_Good(t *testing.T) {
	requireMetalRuntime(t)

	m := newLogitsFakeSessionModel()
	sess := m.NewSession().(*ModelSession)
	defer sess.Close()
	if err := sess.PrefillTokens(context.Background(), []int32{1, 2, 3, 4}); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}

	var n int
	for range sess.Generate(context.Background(), GenerateConfig{MaxTokens: 3, StopTokens: []int32{7}, MinTokensBeforeStop: 2}) {
		n++
		if n >= 3 {
			break
		}
	}
	if err := sess.Err(); err != nil {
		t.Fatalf("session Err under stop-token window: %v", err)
	}
}
