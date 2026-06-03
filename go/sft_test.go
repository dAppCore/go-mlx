// SPDX-Licence-Identifier: EUPL-1.2

package mlx

import (
	"context"
	core "dappco.re/go"
	"dappco.re/go/mlx/dataset"
	"dappco.re/go/mlx/pkg/metal"
	"dappco.re/go/mlx/probe"
	"errors"
	"testing"
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
func (t fakeSFTTokenizer) DecodeOne(id int32) string {
	return t.Decode([]int32{id})
}
func (t fakeSFTTokenizer) BOS() int32        { return 0 }
func (t fakeSFTTokenizer) EOS() int32        { return t.eos }
func (t fakeSFTTokenizer) HasBOSToken() bool { return false }

func TestSFTSliceDataset_Reset_Good(t *testing.T) {
	dataset := dataset.NewSliceDataset([]dataset.Sample{
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
	dataset := dataset.NewSliceDataset([]dataset.Sample{{Prompt: "prompt", Response: "response"}})

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
	dataset := dataset.NewSliceDataset([]dataset.Sample{{Text: "full"}})

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
	_, err := BuildSFTBatches(nil, dataset.NewSliceDataset([]dataset.Sample{{Text: "x"}}), SFTConfig{})
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

func TestModelTrainSFT_NilModel_Bad(t *testing.T) {
	coverageTokens := "Model TrainSFT"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	var model *Model
	_, err := model.TrainSFT(context.Background(), dataset.NewSliceDataset([]dataset.Sample{{Text: "x"}}), SFTConfig{})
	if err == nil {
		t.Fatal("expected nil model error")
	}
}

func TestModelTrainSFT_ValidationBranches_Bad(t *testing.T) {
	model := &Model{model: &fakeNativeModel{}}
	if _, err := model.TrainSFT(context.Background(), nil, SFTConfig{}); err == nil {
		t.Fatal("expected nil dataset error")
	}
	if _, err := model.TrainSFT(context.Background(), dataset.NewSliceDataset([]dataset.Sample{{Text: "x"}}), SFTConfig{}); err == nil {
		t.Fatal("expected nil tokenizer error")
	}

	model.tok = &Tokenizer{tok: &metal.Tokenizer{}}
	if _, err := model.TrainSFT(context.Background(), dataset.NewSliceDataset([]dataset.Sample{{Text: "x"}}), SFTConfig{}); err == nil {
		t.Fatal("expected nil LoRA adapter error")
	}
}

func TestSFTStreamingPacker_Good(t *testing.T) {
	var emitted []sftExample
	packer := newSFTStreamingPacker(4, func(example sftExample) error {
		emitted = append(emitted, example)
		return nil
	})

	if err := packer.add(sftExample{
		inputs:  []int{1, 2},
		targets: []int{2, 3},
		mask:    []float32{0, 1},
	}); err != nil {
		t.Fatalf("add first: %v", err)
	}
	if err := packer.add(sftExample{
		inputs:  []int{3, 4, 5},
		targets: []int{4, 5, 6},
		mask:    []float32{1, 1, 1},
	}); err != nil {
		t.Fatalf("add second: %v", err)
	}
	if err := packer.add(sftExample{
		inputs:  []int{6, 7, 8, 9, 10},
		targets: []int{7, 8, 9, 10, 11},
		mask:    []float32{1, 1, 1, 1, 1},
	}); err != nil {
		t.Fatalf("add long: %v", err)
	}
	if err := packer.finish(); err != nil {
		t.Fatalf("finish: %v", err)
	}

	if len(emitted) != 3 {
		t.Fatalf("emitted len = %d, want 3", len(emitted))
	}
	if !equalIntSlices(emitted[0].inputs, []int{1, 2}) {
		t.Fatalf("first packed inputs = %v, want [1 2]", emitted[0].inputs)
	}
	if !equalIntSlices(emitted[1].inputs, []int{3, 4, 5}) {
		t.Fatalf("second packed inputs = %v, want [3 4 5]", emitted[1].inputs)
	}
	if !equalIntSlices(emitted[2].inputs, []int{7, 8, 9, 10}) {
		t.Fatalf("trimmed packed inputs = %v, want last four tokens", emitted[2].inputs)
	}
	if len(packer.current.inputs) != 0 {
		t.Fatalf("packer current = %+v, want flushed", packer.current)
	}
}

func TestSFTStreamingPacker_BadAndHelpers(t *testing.T) {
	if err := (*sftStreamingPacker)(nil).finish(); err != nil {
		t.Fatalf("nil finish error = %v", err)
	}
	if err := (*sftStreamingPacker)(nil).add(sftExample{inputs: []int{1}}); err != nil {
		t.Fatalf("nil add error = %v", err)
	}
	packer := newSFTStreamingPacker(8, nil)
	if err := packer.add(sftExample{inputs: []int{1}}); err != nil {
		t.Fatalf("nil emit add error = %v", err)
	}
	if err := packer.flush(); err != nil {
		t.Fatalf("empty flush error = %v", err)
	}

	wantErr := errors.New("emit failed")
	packer = newSFTStreamingPacker(8, func(sftExample) error { return wantErr })
	if err := packer.add(sftExample{inputs: []int{1}, targets: []int{2}, mask: []float32{1}}); err != nil {
		t.Fatalf("add before failing flush error = %v", err)
	}
	if err := packer.finish(); !errors.Is(err, wantErr) {
		t.Fatalf("finish error = %v, want %v", err, wantErr)
	}

	if loss := sftAdapterStep(nil, nil, nil); loss != nil {
		t.Fatalf("sftAdapterStep(empty) = %+v, want nil", loss)
	}
	if sink := sftProbeSink(SFTConfig{ProbeSink: probe.NewRecorder()}); sink == nil {
		t.Fatal("sftProbeSink did not prefer direct SFT probe sink")
	}
	if sink := sftProbeSink(SFTConfig{LoRA: LoRAConfig{ProbeSink: probe.NewRecorder()}}); sink == nil {
		t.Fatal("sftProbeSink did not fall back to LoRA probe sink")
	}
}

func TestSFTDatasetEpoch_EmptyErrorAndCancelledBranches_Bad(t *testing.T) {
	var model *Model
	result := &SFTResult{}
	cfg := normalizeSFTConfig(SFTConfig{BatchSize: 2, GradientAccumulationSteps: 2})
	if err := model.runSFTDatasetEpoch(context.Background(), nil, dataset.NewSliceDataset(nil), nil, nil, cfg, result, 1); err != nil {
		t.Fatalf("empty epoch error = %v", err)
	}
	if result.Samples != 0 {
		t.Fatalf("empty epoch samples = %d, want 0", result.Samples)
	}

	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	if err := model.runSFTDatasetEpoch(cancelled, nil, dataset.NewSliceDataset([]dataset.Sample{{Text: "x"}}), nil, nil, cfg, result, 1); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled epoch error = %v, want context.Canceled", err)
	}
	if err := model.runSFTBatchGroup(cancelled, nil, nil, nil, cfg, result, 1); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled batch group error = %v, want context.Canceled", err)
	}

	native := &fakeNativeModel{loraAdapter: &metal.LoRAAdapter{}}
	adapter, err := (&Model{model: native}).sftAdapter(SFTConfig{LoRA: LoRAConfig{ProbeSink: probe.NewRecorder(), Lambda: 0.25}})
	if err != nil {
		t.Fatalf("sftAdapter() error = %v", err)
	}
	if adapter == nil || native.lastLoRAConfig.ProbeSink != nil || native.lastLoRAConfig.Lambda != 0.25 {
		t.Fatalf("adapter=%+v native config=%+v, want adapter with sanitised probe config", adapter, native.lastLoRAConfig)
	}
}
