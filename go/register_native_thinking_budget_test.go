// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package mlx

import (
	"context"
	"reflect"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/mlx/pkg/model"
	"dappco.re/go/mlx/pkg/native"
	pkgtokenizer "dappco.re/go/mlx/pkg/tokenizer"
)

const nativeThinkingBudgetTokenizerJSON = `{
  "model": {
    "type": "BPE",
    "vocab": {
      "▁": 1,
      "h": 2,
      "e": 3,
      "l": 4,
      "o": 5,
      "▁h": 6,
      "▁he": 7,
      "▁hel": 8,
      "▁hell": 9,
      "▁hello": 10
    },
    "merges": ["▁ h", "▁h e", "▁he l", "▁hel l", "▁hell o"]
  },
  "added_tokens": [
    {"id": 0, "content": "<bos>", "special": true},
    {"id": 11, "content": "<eos>", "special": true},
    {"id": 20, "content": "<|channel>", "special": true},
    {"id": 21, "content": "<channel|>", "special": true}
  ]
}`

type thinkingBudgetTextTokenModel struct {
	session *thinkingBudgetTextSession
}

func (m *thinkingBudgetTextTokenModel) Embed(id int32) ([]byte, error) { return []byte{byte(id)}, nil }

func (m *thinkingBudgetTextTokenModel) DecodeForward([][]byte) ([][]byte, error) {
	return [][]byte{{0}}, nil
}

func (m *thinkingBudgetTextTokenModel) Head([]byte) ([]byte, error) { return make([]byte, 64), nil }

func (m *thinkingBudgetTextTokenModel) Vocab() int { return 32 }

func (m *thinkingBudgetTextTokenModel) OpenSession() (model.DecodeStepper, error) {
	return m.session, nil
}

type thinkingBudgetTextSession struct {
	selected              []int32
	committed             []int32
	generateTransformCall int
	cachedTransformCall   int
	warmed                []int32
}

func (s *thinkingBudgetTextSession) Step([]byte) ([]byte, error) { return []byte{0}, nil }

func (s *thinkingBudgetTextSession) WarmPromptCache(ids []int32) error {
	s.warmed = append(s.warmed[:0], ids...)
	return nil
}

func (s *thinkingBudgetTextSession) GenerateCached([]int32, int, int) ([]int32, error) {
	return nil, core.NewError("GenerateCached must not bypass thinking-budget transform")
}

func (s *thinkingBudgetTextSession) ClearPromptCache() {}

func (s *thinkingBudgetTextSession) GenerateEachTransformed(ids []int32, maxNew, eos int, transform native.TokenTransform, yield func(int32) bool) ([]int32, error) {
	s.generateTransformCall++
	return s.generate(ids, maxNew, eos, transform, yield)
}

func (s *thinkingBudgetTextSession) GenerateCachedEachTransformed(ids []int32, maxNew, eos int, transform native.TokenTransform, yield func(int32) bool) ([]int32, error) {
	s.cachedTransformCall++
	return s.generate(ids, maxNew, eos, transform, yield)
}

func (s *thinkingBudgetTextSession) generate(_ []int32, maxNew, eos int, transform native.TokenTransform, yield func(int32) bool) ([]int32, error) {
	out := make([]int32, 0, maxNew)
	for _, id := range s.selected {
		if len(out) >= maxNew {
			break
		}
		committed := id
		if transform != nil {
			committed = transform(committed)
		}
		out = append(out, committed)
		s.committed = append(s.committed, committed)
		if yield != nil && !yield(committed) {
			break
		}
		if eos >= 0 && int(committed) == eos {
			break
		}
	}
	return out, nil
}

func TestNativeTextModelThinkingBudgetForcesCommittedClose(t *testing.T) {
	tok := loadNativeThinkingBudgetTokenizer(t)
	session := &thinkingBudgetTextSession{selected: []int32{20, 1, 2, 3, 4}}
	nativeModel := &nativeTextModel{
		tm:     &thinkingBudgetTextTokenModel{session: session},
		tok:    tok,
		maxLen: 32,
	}

	var got []int32
	for tok := range nativeModel.Generate(context.Background(), "hello", inference.WithMaxTokens(5), inference.WithThinkingBudget(2)) {
		got = append(got, tok.ID)
	}
	if err := resultError(nativeModel.Err()); err != nil {
		t.Fatalf("Generate Err: %v", err)
	}
	want := []int32{20, 1, 2, 21, 4}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("budgeted generated ids = %v, want %v", got, want)
	}
	if !reflect.DeepEqual(session.committed, want) {
		t.Fatalf("committed ids = %v, want %v", session.committed, want)
	}
	if session.generateTransformCall != 1 {
		t.Fatalf("GenerateEachTransformed calls = %d, want 1", session.generateTransformCall)
	}
	if !nativeModel.Metrics().ThinkingBudgetForced {
		t.Fatal("ThinkingBudgetForced = false, want true")
	}
}

func TestNativeTextModelThinkingBudgetUsesTransformedPromptCache(t *testing.T) {
	tok := loadNativeThinkingBudgetTokenizer(t)
	session := &thinkingBudgetTextSession{selected: []int32{20, 1, 2, 3}}
	nativeModel := &nativeTextModel{
		tm:     &thinkingBudgetTextTokenModel{session: session},
		tok:    tok,
		maxLen: 32,
	}
	if err := nativeModel.WarmPromptCache(context.Background(), "hello"); err != nil {
		t.Fatalf("WarmPromptCache: %v", err)
	}

	var got []int32
	for tok := range nativeModel.Generate(context.Background(), "hello", inference.WithMaxTokens(4), inference.WithThinkingBudget(1)) {
		got = append(got, tok.ID)
	}
	if err := resultError(nativeModel.Err()); err != nil {
		t.Fatalf("Generate Err: %v", err)
	}
	want := []int32{20, 1, 21, 3}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("budgeted cached ids = %v, want %v", got, want)
	}
	if session.cachedTransformCall != 1 {
		t.Fatalf("GenerateCachedEachTransformed calls = %d, want 1", session.cachedTransformCall)
	}
	if !nativeModel.Metrics().ThinkingBudgetForced {
		t.Fatal("ThinkingBudgetForced = false, want true")
	}
}

func loadNativeThinkingBudgetTokenizer(t *testing.T) *pkgtokenizer.Tokenizer {
	t.Helper()
	dir := t.TempDir()
	path := core.PathJoin(dir, "tokenizer.json")
	if result := core.WriteFile(path, []byte(nativeThinkingBudgetTokenizerJSON), 0o644); !result.OK {
		t.Fatalf("write tokenizer: %v", result.Value)
	}
	tok, err := pkgtokenizer.LoadTokenizer(path)
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	return tok
}
