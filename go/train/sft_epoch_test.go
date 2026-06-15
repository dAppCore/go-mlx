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

// --- RunSFTDatasetEpoch — the example-build + accumulation loop ---

// TestSftEpoch_RunSFTDatasetEpoch_Good drives the epoch loop over a dataset of
// only-unusable rows (empty prompt/response with NoEOS collapses below the
// 2-token minimum). The loop walks every row, skips them all, and the terminal
// flushes no-op on the empty accumulator — the documented clean return, with no
// Metal reached because no batch ever forms. Samples stays 0.
func TestSftEpoch_RunSFTDatasetEpoch_Good(t *testing.T) {
	tok := newSFTBatchTestTokenizer()
	cfg := normalizeSFTConfig(SFTConfig{BatchSize: 2, GradientAccumulationSteps: 2, NoEOS: true})
	ds := dataset.NewSliceDataset([]dataset.Sample{
		{Prompt: "", Response: ""},
		{Prompt: "", Response: ""},
		{Prompt: "", Response: ""},
	})
	result := &SFTResult{}
	if err := RunSFTDatasetEpoch(context.Background(), nil, tok, ds, nil, nil, cfg, result, 1); err != nil {
		t.Fatalf("RunSFTDatasetEpoch() over unusable rows error = %v, want clean return", err)
	}
	if result.Samples != 0 {
		t.Fatalf("samples = %d, want 0 (every row unusable)", result.Samples)
	}
	if len(result.Losses) != 0 {
		t.Fatalf("losses = %d, want 0 (no batch reached the gradient step)", len(result.Losses))
	}
}

// TestSftEpoch_RunSFTDatasetEpoch_Bad asserts a dataset whose Next returns an
// error aborts the epoch and surfaces it, before any gradient machinery.
func TestSftEpoch_RunSFTDatasetEpoch_Bad(t *testing.T) {
	cfg := normalizeSFTConfig(SFTConfig{BatchSize: 2, GradientAccumulationSteps: 1})
	ds := dataset.Func(func() (dataset.Sample, bool, error) {
		return dataset.Sample{}, false, errors.New("read failed")
	})
	if err := RunSFTDatasetEpoch(context.Background(), nil, nil, ds, nil, nil, cfg, &SFTResult{}, 1); err == nil {
		t.Fatal("epoch error = nil, want dataset read error surfaced")
	}
}

// TestSftEpoch_RunSFTDatasetEpoch_Ugly covers the degenerate-but-legal shapes:
// an empty dataset returns nil with Samples 0, and a cancelled context aborts
// both the epoch loop and a directly-invoked batch group with context.Canceled.
func TestSftEpoch_RunSFTDatasetEpoch_Ugly(t *testing.T) {
	result := &SFTResult{}
	cfg := normalizeSFTConfig(SFTConfig{BatchSize: 2, GradientAccumulationSteps: 2})

	// Empty dataset: legal, clean return, nothing processed.
	if err := RunSFTDatasetEpoch(context.Background(), nil, nil, dataset.NewSliceDataset(nil), nil, nil, cfg, result, 1); err != nil {
		t.Fatalf("empty epoch error = %v", err)
	}
	if result.Samples != 0 {
		t.Fatalf("empty epoch samples = %d, want 0", result.Samples)
	}

	// Cancelled context: the loop's ctx.Err() guard aborts before processing.
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	if err := RunSFTDatasetEpoch(cancelled, nil, nil, dataset.NewSliceDataset([]dataset.Sample{{Text: "x"}}), nil, nil, cfg, result, 1); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled epoch error = %v, want context.Canceled", err)
	}
	if err := runSFTBatchGroup(cancelled, nil, nil, nil, nil, cfg, result, 1); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled batch group error = %v, want context.Canceled", err)
	}
}

// --- FinaliseScoreCascade — copy the cascade verdict onto the result ---

// TestSftEpoch_FinaliseScoreCascade_Good arms a cascade with a scored pass, then
// finalises: the best step by windowed composite and the full record set land on
// the result.
func TestSftEpoch_FinaliseScoreCascade_Good(t *testing.T) {
	result := &SFTResult{cascade: newSFTScoreCascade("", 0)}
	result.cascade.recordPass(7, []SFTEvalResult{{Step: 7, Prompt: "p", Text: "I notice the morning holds. I want to keep it."}})
	FinaliseScoreCascade(result)
	if result.BestScoreStep != 7 {
		t.Fatalf("best score step = %d, want 7", result.BestScoreStep)
	}
	if len(result.ScoreRecords) != 1 {
		t.Fatalf("score records = %d, want 1", len(result.ScoreRecords))
	}
}

// TestSftEpoch_FinaliseScoreCascade_Bad asserts FinaliseScoreCascade is nil-safe
// both ways: a nil result and a result with no cascade armed are both no-ops
// (the run never armed the instrument), not panics.
func TestSftEpoch_FinaliseScoreCascade_Bad(t *testing.T) {
	FinaliseScoreCascade(nil)
	result := &SFTResult{}
	FinaliseScoreCascade(result)
	if result.BestScoreStep != 0 || len(result.ScoreRecords) != 0 {
		t.Fatalf("un-armed finalise mutated result = %+v, want untouched", result)
	}
}

// TestSftEpoch_FinaliseScoreCascade_Ugly arms a cascade that never recorded a
// scored pass (every pass filtered on a step mismatch): finalising copies the
// empty record set and leaves the best step at zero — no verdict from nothing.
func TestSftEpoch_FinaliseScoreCascade_Ugly(t *testing.T) {
	result := &SFTResult{cascade: newSFTScoreCascade("", 0)}
	// Step mismatch → the record is filtered, nothing accumulates.
	result.cascade.recordPass(1, []SFTEvalResult{{Step: 99, Prompt: "p", Text: "skipped"}})
	FinaliseScoreCascade(result)
	if len(result.ScoreRecords) != 0 {
		t.Fatalf("score records = %d, want 0 (nothing was scored)", len(result.ScoreRecords))
	}
	if result.BestScoreStep != 0 {
		t.Fatalf("best score step = %d, want 0 (no verdict from an empty cascade)", result.BestScoreStep)
	}
}

// --- sftStreamingPacker / sftEvalGenerateOptions — private helpers ---
// (descriptive names so the unreferenced-symbols audit reads the real subject,
// not a triplet variant claim.)

// TestSftEpoch_StreamingPackerPacksAndTrims drives the streaming packer: rows
// concatenate up to maxSeqLen, an oversized row is trimmed to the tail, and the
// final flush empties the accumulator.
func TestSftEpoch_StreamingPackerPacksAndTrims(t *testing.T) {
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

// TestSftEpoch_StreamingPackerGuardsAndHelpers covers the nil/empty guards on
// the packer plus the sftAdapterStep empty-batch guard and the sftProbeSink
// preference order.
func TestSftEpoch_StreamingPackerGuardsAndHelpers(t *testing.T) {
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

// TestSftEpoch_EvalGenerateOptionsCarriesTemperature pins that the eval generate
// options forward the configured max tokens and temperature.
func TestSftEpoch_EvalGenerateOptionsCarriesTemperature(t *testing.T) {
	cfg := normalizeSFTConfig(SFTConfig{EvalMaxTokens: 64, EvalTemperature: 0.35})
	opts := sftEvalGenerateOptions(cfg)
	applied := spine.ApplyGenerateOptions(opts)
	if applied.MaxTokens != 64 || applied.Temperature != 0.35 {
		t.Fatalf("eval generate config = %+v, want max tokens and temperature", applied)
	}
}
