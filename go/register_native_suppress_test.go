// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package mlx

import (
	"context"
	"reflect"
	"testing"

	"dappco.re/go/inference"
	"dappco.re/go/mlx/pkg/model"
	pkgtokenizer "dappco.re/go/mlx/pkg/tokenizer"
)

type suppressTextTokenModel struct {
	session *suppressTextSession
}

func (m *suppressTextTokenModel) Embed(id int32) ([]byte, error) { return []byte{byte(id)}, nil }

func (m *suppressTextTokenModel) DecodeForward([][]byte) ([][]byte, error) {
	return [][]byte{{0}}, nil
}

func (m *suppressTextTokenModel) Head([]byte) ([]byte, error) {
	logits := make([]byte, 64)
	logits[12*2], logits[12*2+1] = f32ToBF16BytesForNativeTextTest(1.0)
	logits[13*2], logits[13*2+1] = f32ToBF16BytesForNativeTextTest(0.75)
	return logits, nil
}

func (m *suppressTextTokenModel) Vocab() int { return 32 }

func (m *suppressTextTokenModel) OpenSession() (model.DecodeStepper, error) {
	return m.session, nil
}

type suppressTextSession struct {
	normalCalls     int
	suppressedCalls int
}

func (s *suppressTextSession) Step([]byte) ([]byte, error) { return []byte{0}, nil }

func (s *suppressTextSession) GenerateEach(ids []int32, maxNew, eos int, yield func(int32) bool) ([]int32, error) {
	s.normalCalls++
	return s.generate(maxNew, eos, nil, yield), nil
}

func (s *suppressTextSession) GenerateEachWithSuppression(ids []int32, maxNew, eos int, suppress []int32, yield func(int32) bool) ([]int32, error) {
	s.suppressedCalls++
	return s.generate(maxNew, eos, suppress, yield), nil
}

func (s *suppressTextSession) generate(maxNew, eos int, suppress []int32, yield func(int32) bool) []int32 {
	out := make([]int32, 0, maxNew)
	for _, id := range []int32{12, 13} {
		if tokenInSet(id, suppress) {
			continue
		}
		out = append(out, id)
		if yield != nil && !yield(id) {
			break
		}
		if eos >= 0 && int(id) == eos {
			break
		}
		if len(out) >= maxNew {
			break
		}
	}
	return out
}

func TestNativeTextModelGreedyHonoursSuppressTokens(t *testing.T) {
	tok, err := pkgtokenizer.LoadTokenizer(writeRootTokenizer(t))
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	session := &suppressTextSession{}
	nativeModel := &nativeTextModel{
		tm:     &suppressTextTokenModel{session: session},
		tok:    tok,
		maxLen: 32,
	}

	var got []int32
	for tok := range nativeModel.Generate(context.Background(), "hello", inference.WithMaxTokens(1), inference.WithSuppressTokens(12)) {
		got = append(got, tok.ID)
	}
	if err := resultError(nativeModel.Err()); err != nil {
		t.Fatalf("Generate Err: %v", err)
	}
	if want := []int32{13}; !reflect.DeepEqual(got, want) {
		t.Fatalf("suppressed generated ids = %v, want %v", got, want)
	}
	if session.suppressedCalls != 1 || session.normalCalls != 0 {
		t.Fatalf("GenerateEachWithSuppression/GenerateEach calls = %d/%d, want 1/0", session.suppressedCalls, session.normalCalls)
	}
}

func TestNativeTextModelGreedyMinTokensBeforeStopSuppressesFirstStop(t *testing.T) {
	tok, err := pkgtokenizer.LoadTokenizer(writeRootTokenizer(t))
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	session := &suppressTextSession{}
	nativeModel := &nativeTextModel{
		tm:     &suppressTextTokenModel{session: session},
		tok:    tok,
		maxLen: 32,
	}

	var got []int32
	for tok := range nativeModel.Generate(context.Background(), "hello", inference.WithMaxTokens(1), inference.WithStopTokens(12), inference.WithMinTokensBeforeStop(1)) {
		got = append(got, tok.ID)
	}
	if err := resultError(nativeModel.Err()); err != nil {
		t.Fatalf("Generate Err: %v", err)
	}
	if want := []int32{13}; !reflect.DeepEqual(got, want) {
		t.Fatalf("min-stop generated ids = %v, want %v", got, want)
	}
}
