// SPDX-Licence-Identifier: EUPL-1.2

package train

import (
	"context"
	"errors"
	"testing"

	"dappco.re/go/mlx/dataset"
	"dappco.re/go/mlx/probe"
	"dappco.re/go/mlx/spine"
)

func TestSFTDatasetEpoch_EmptyErrorAndCancelledBranches_Bad(t *testing.T) {
	result := &SFTResult{}
	cfg := normalizeSFTConfig(SFTConfig{BatchSize: 2, GradientAccumulationSteps: 2})
	if err := RunSFTDatasetEpoch(context.Background(), nil, nil, dataset.NewSliceDataset(nil), nil, nil, cfg, result, 1); err != nil {
		t.Fatalf("empty epoch error = %v", err)
	}
	if result.Samples != 0 {
		t.Fatalf("empty epoch samples = %d, want 0", result.Samples)
	}

	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	if err := RunSFTDatasetEpoch(cancelled, nil, nil, dataset.NewSliceDataset([]dataset.Sample{{Text: "x"}}), nil, nil, cfg, result, 1); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled epoch error = %v, want context.Canceled", err)
	}
	if err := runSFTBatchGroup(cancelled, nil, nil, nil, nil, cfg, result, 1); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled batch group error = %v, want context.Canceled", err)
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
	if sink := sftProbeSink(SFTConfig{LoRA: spine.LoRAConfig{ProbeSink: probe.NewRecorder()}}); sink == nil {
		t.Fatal("sftProbeSink did not fall back to LoRA probe sink")
	}
}

func TestSFTEvalGenerateOptions_CarriesTemperature_Good(t *testing.T) {
	cfg := normalizeSFTConfig(SFTConfig{EvalMaxTokens: 64, EvalTemperature: 0.35})
	opts := sftEvalGenerateOptions(cfg)
	applied := spine.ApplyGenerateOptions(opts)
	if applied.MaxTokens != 64 || applied.Temperature != 0.35 {
		t.Fatalf("eval generate config = %+v, want max tokens and temperature", applied)
	}
}

// A dataset whose Next returns an error aborts the epoch and surfaces it,
// before any gradient machinery is reached.
func TestRunSFTDatasetEpoch_DatasetErrorPropagates_Bad(t *testing.T) {
	cfg := normalizeSFTConfig(SFTConfig{BatchSize: 2, GradientAccumulationSteps: 1})
	ds := dataset.Func(func() (dataset.Sample, bool, error) {
		return dataset.Sample{}, false, errors.New("read failed")
	})
	if err := RunSFTDatasetEpoch(context.Background(), nil, nil, ds, nil, nil, cfg, &SFTResult{}, 1); err == nil {
		t.Fatal("epoch error = nil, want dataset read error surfaced")
	}
}
