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

type seededSampleTextTokenModel struct{}

func (m seededSampleTextTokenModel) Embed(id int32) ([]byte, error) { return []byte{byte(id)}, nil }

func (m seededSampleTextTokenModel) DecodeForward(inputs [][]byte) ([][]byte, error) {
	return inputs, nil
}

func (m seededSampleTextTokenModel) Head([]byte) ([]byte, error) {
	return make([]byte, m.Vocab()*2), nil
}

func (m seededSampleTextTokenModel) Vocab() int { return 4 }

func expectedSeededSampleIDs(seed uint64, maxNew int) []int32 {
	sampler := model.NewSampler(seed)
	logits := make([]byte, seededSampleTextTokenModel{}.Vocab()*2)
	out := make([]int32, 0, maxNew)
	for range maxNew {
		id, err := sampler.Sample(logits, seededSampleTextTokenModel{}.Vocab(), model.SampleParams{Temperature: 1})
		if err != nil {
			panic(err)
		}
		out = append(out, id)
	}
	return out
}

func TestNativeTextModelSampledHonoursSeed(t *testing.T) {
	tok, err := pkgtokenizer.LoadTokenizer(writeRootTokenizer(t))
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	nativeModel := &nativeTextModel{
		tm:     seededSampleTextTokenModel{},
		tok:    tok,
		maxLen: 32,
	}

	var got []int32
	for tok := range nativeModel.Generate(context.Background(), "hello", inference.WithMaxTokens(8), inference.WithTemperature(1), inference.WithSeed(123)) {
		got = append(got, tok.ID)
	}
	if err := nativeModel.Err(); err != nil {
		t.Fatalf("Generate Err: %v", err)
	}
	if want := expectedSeededSampleIDs(123, 8); !reflect.DeepEqual(got, want) {
		t.Fatalf("seeded sampled ids = %v, want %v", got, want)
	}
}
