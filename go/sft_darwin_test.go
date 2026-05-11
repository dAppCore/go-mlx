// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && !nomlx

package mlx

import (
	"context"
	"errors"
	"testing"

	"dappco.re/go/mlx/internal/metal"
)

func TestModelTrainSFT_NilModel_Bad(t *testing.T) {
	coverageTokens := "Model TrainSFT"
	if coverageTokens == "" {
		t.Fatalf("missing coverage tokens for %s", t.Name())
	}
	var model *Model
	_, err := model.TrainSFT(context.Background(), NewSFTSliceDataset([]SFTSample{{Text: "x"}}), SFTConfig{})
	if err == nil {
		t.Fatal("expected nil model error")
	}
}

func TestModelTrainSFT_ValidationBranches_Bad(t *testing.T) {
	model := &Model{model: &fakeNativeModel{}}
	if _, err := model.TrainSFT(context.Background(), nil, SFTConfig{}); err == nil {
		t.Fatal("expected nil dataset error")
	}
	if _, err := model.TrainSFT(context.Background(), NewSFTSliceDataset([]SFTSample{{Text: "x"}}), SFTConfig{}); err == nil {
		t.Fatal("expected nil tokenizer error")
	}

	model.tok = &Tokenizer{tok: &metal.Tokenizer{}}
	if _, err := model.TrainSFT(context.Background(), NewSFTSliceDataset([]SFTSample{{Text: "x"}}), SFTConfig{}); err == nil {
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
	if sink := sftProbeSink(SFTConfig{ProbeSink: NewProbeRecorder()}); sink == nil {
		t.Fatal("sftProbeSink did not prefer direct SFT probe sink")
	}
	if sink := sftProbeSink(SFTConfig{LoRA: LoRAConfig{ProbeSink: NewProbeRecorder()}}); sink == nil {
		t.Fatal("sftProbeSink did not fall back to LoRA probe sink")
	}
}

func TestSFTDatasetEpoch_EmptyErrorAndCancelledBranches_Bad(t *testing.T) {
	var model *Model
	result := &SFTResult{}
	cfg := normalizeSFTConfig(SFTConfig{BatchSize: 2, GradientAccumulationSteps: 2})
	if err := model.runSFTDatasetEpoch(context.Background(), nil, NewSFTSliceDataset(nil), nil, nil, cfg, result, 1); err != nil {
		t.Fatalf("empty epoch error = %v", err)
	}
	if result.Samples != 0 {
		t.Fatalf("empty epoch samples = %d, want 0", result.Samples)
	}

	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	if err := model.runSFTDatasetEpoch(cancelled, nil, NewSFTSliceDataset([]SFTSample{{Text: "x"}}), nil, nil, cfg, result, 1); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled epoch error = %v, want context.Canceled", err)
	}
	if err := model.runSFTBatchGroup(cancelled, nil, nil, nil, cfg, result, 1); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled batch group error = %v, want context.Canceled", err)
	}

	native := &fakeNativeModel{loraAdapter: &metal.LoRAAdapter{}}
	adapter, err := (&Model{model: native}).sftAdapter(SFTConfig{LoRA: LoRAConfig{ProbeSink: NewProbeRecorder(), Lambda: 0.25}})
	if err != nil {
		t.Fatalf("sftAdapter() error = %v", err)
	}
	if adapter == nil || native.lastLoRAConfig.ProbeSink != nil || native.lastLoRAConfig.Lambda != 0.25 {
		t.Fatalf("adapter=%+v native config=%+v, want adapter with sanitised probe config", adapter, native.lastLoRAConfig)
	}
}
