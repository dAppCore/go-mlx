// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"testing"

	core "dappco.re/go"
)

type fakeSFTTokenizer struct {
	encoded map[string][]int32
	eos     int32
}

func (t fakeSFTTokenizer) Encode(text string) []int32 {
	if tokens, ok := t.encoded[text]; ok {
		return append([]int32(nil), tokens...)
	}
	out := make([]int32, 0, len(text))
	for _, r := range text {
		out = append(out, int32(r))
	}
	return out
}

func (t fakeSFTTokenizer) Decode(tokens []int32) string {
	builder := core.NewBuilder()
	for _, token := range tokens {
		builder.WriteString(core.Sprintf("%d", token))
	}
	return builder.String()
}

func (t fakeSFTTokenizer) TokenID(text string) (int32, bool) {
	tokens := t.Encode(text)
	if len(tokens) != 1 {
		return 0, false
	}
	return tokens[0], true
}

func (t fakeSFTTokenizer) IDToken(id int32) string { return core.Sprintf("%d", id) }
func (t fakeSFTTokenizer) BOS() int32              { return 0 }
func (t fakeSFTTokenizer) EOS() int32              { return t.eos }
func (t fakeSFTTokenizer) HasBOSToken() bool       { return false }

func TestSFTSliceDataset_Reset_Good(t *testing.T) {
	dataset := NewSFTSliceDataset([]SFTSample{
		{Prompt: "a", Response: "b"},
	})

	first, ok, err := dataset.Next()
	if err != nil {
		t.Fatalf("Next() error = %v", err)
	}
	if !ok || first.Prompt != "a" {
		t.Fatalf("first Next() = %+v ok=%v", first, ok)
	}
	if _, ok, err := dataset.Next(); err != nil || ok {
		t.Fatalf("exhausted Next() ok=%v err=%v, want ok=false err=nil", ok, err)
	}
	if err := dataset.Reset(); err != nil {
		t.Fatalf("Reset() error = %v", err)
	}
	again, ok, err := dataset.Next()
	if err != nil {
		t.Fatalf("Next() after Reset error = %v", err)
	}
	if !ok || again.Response != "b" {
		t.Fatalf("Next() after Reset = %+v ok=%v", again, ok)
	}
}

func TestBuildSFTBatches_MasksPromptAndAppendsEOS_Good(t *testing.T) {
	tokenizer := &Tokenizer{tok: fakeSFTTokenizer{
		encoded: map[string][]int32{
			"prompt":   {10, 11},
			"response": {20, 21},
		},
		eos: 2,
	}}
	dataset := NewSFTSliceDataset([]SFTSample{{Prompt: "prompt", Response: "response"}})

	batches, err := BuildSFTBatches(tokenizer, dataset, SFTConfig{BatchSize: 1})
	if err != nil {
		t.Fatalf("BuildSFTBatches() error = %v", err)
	}
	if len(batches) != 1 {
		t.Fatalf("batches len = %d, want 1", len(batches))
	}
	got := batches[0]
	wantInputs := []int{10, 11, 20, 21}
	wantTargets := []int{11, 20, 21, 2}
	wantMask := []float32{0, 1, 1, 1}
	if !equalIntSlices(got.Batch.Tokens[0], wantInputs) {
		t.Fatalf("inputs = %v, want %v", got.Batch.Tokens[0], wantInputs)
	}
	if !equalIntSlices(got.Targets[0], wantTargets) {
		t.Fatalf("targets = %v, want %v", got.Targets[0], wantTargets)
	}
	if !equalFloat32Slices(got.Batch.LossMask[0], wantMask) {
		t.Fatalf("loss mask = %v, want %v", got.Batch.LossMask[0], wantMask)
	}
}

func TestBuildSFTBatches_TextSampleTrainsWholeSequence_Good(t *testing.T) {
	tokenizer := &Tokenizer{tok: fakeSFTTokenizer{
		encoded: map[string][]int32{"full": {5, 6, 7}},
		eos:     9,
	}}
	dataset := NewSFTSliceDataset([]SFTSample{{Text: "full"}})

	batches, err := BuildSFTBatches(tokenizer, dataset, SFTConfig{BatchSize: 1, NoEOS: true})
	if err != nil {
		t.Fatalf("BuildSFTBatches() error = %v", err)
	}
	if len(batches) != 1 {
		t.Fatalf("batches len = %d, want 1", len(batches))
	}
	if !equalIntSlices(batches[0].Batch.Tokens[0], []int{5, 6}) {
		t.Fatalf("inputs = %v, want [5 6]", batches[0].Batch.Tokens[0])
	}
	if !equalIntSlices(batches[0].Targets[0], []int{6, 7}) {
		t.Fatalf("targets = %v, want [6 7]", batches[0].Targets[0])
	}
	if !equalFloat32Slices(batches[0].Batch.LossMask[0], []float32{1, 1}) {
		t.Fatalf("loss mask = %v, want [1 1]", batches[0].Batch.LossMask[0])
	}
}

func TestBuildSFTBatches_NilTokenizer_Bad(t *testing.T) {
	_, err := BuildSFTBatches(nil, NewSFTSliceDataset([]SFTSample{{Text: "x"}}), SFTConfig{})
	if err == nil {
		t.Fatal("expected nil tokenizer error")
	}
}

func equalIntSlices(a, b []int) bool {
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

func equalFloat32Slices(a, b []float32) bool {
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
